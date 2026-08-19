"""Map-level spectrum aggregation and plotting helpers for Raman notebooks."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import shutil
from collections.abc import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from raman_config import (
    NORMALIZATION_METHOD,
    NORMALIZATION_PEAK_CENTER_CM1,
    NORMALIZATION_PEAK_TOLERANCE_CM1,
    PEAK_RATIO_WAVENUMBER_RANGES,
    PLOT_WAVENUMBER_RANGES,
)
from raman_processing_utils import save_pixel_spectrum_comparison, select_max_intensity_pixel

_DATE_PATTERN = re.compile(r"(\d{8})")
_LASER_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:mw|w)", flags=re.IGNORECASE)

_EXPORT_PART_PREFIXES = {
    "plots": "01_plots",
    "spectra": "02_spectra",
    "tables": "03_tables",
    "code_snapshot": "04_code_snapshot",
    "avg_stack": "01_avg_stack",
    "norm_stack": "02_norm_stack",
    "norm_overlap": "03_norm_overlap",
    "peak_ratio": "04_peak_ratio",
    "max_signal": "05_max_signal",
    "cutpixel_map": "06_cutpixel_map",
    "despiked_baseline_anchor_stack": "07_despiked_baseline_anchor_stack",
    "groups": "01_groups",
    "hbn_subgroups": "02_hbn_subgroups",
}


def _normalize_substrate_token(token: str) -> str | None:
    """Map naming tokens to canonical group labels."""
    lowered = token.strip().lower()
    if lowered == "au":
        return "Au"
    if lowered == "ro":
        return "RO"
    if lowered == "hbn":
        return "hBN"
    return None


def _is_numeric_suffix(token: str) -> bool:
    """Return true for integer suffixes used in names like hBN_1."""
    return bool(re.fullmatch(r"\d+", token.strip()))


def _is_laser_power_token(token: str) -> bool:
    """Return true for laser power tokens like 10mW or 0.5W."""
    return bool(_LASER_PATTERN.fullmatch(token.strip()))


def _split_stem_tokens(file_name: str) -> list[str]:
    """Split file stem into underscore-separated naming tokens."""
    stem = Path(file_name).stem
    return [token for token in stem.split("_") if token]


def _strip_trailing_metadata_tokens(tokens: list[str]) -> list[str]:
    """Remove trailing date and laser-power tokens from a filename token list."""
    stripped_tokens = list(tokens)

    if stripped_tokens and re.fullmatch(r"\d{8}", stripped_tokens[-1]):
        stripped_tokens.pop()

    if stripped_tokens and _is_laser_power_token(stripped_tokens[-1]):
        stripped_tokens.pop()

    return stripped_tokens


def _compute_stack_step(
    spectra: list[np.ndarray],
    stack_scale: float,
    stack_extra_gap: float,
) -> float:
    """Estimate vertical spacing between stacked traces.

    Uses a robust span (95th - 5th percentile) per spectrum to avoid one
    outlier peak dominating the spacing for all curves.
    """
    if not spectra:
        return 1.0

    spans = []
    for spectrum in spectra:
        arr = np.asarray(spectrum, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        spans.append(float(np.percentile(finite, 95) - np.percentile(finite, 5)))

    if not spans:
        return 1.0

    base_step = float(np.median(spans))
    return max(base_step * float(stack_scale) + float(stack_extra_gap), 1e-12)


def _resolve_group_value(
    value: float | Mapping[str, float],
    group_name: str,
    fallback: float,
) -> float:
    """Resolve scalar or per-group value for one panel."""
    if isinstance(value, Mapping):
        if group_name in value:
            return float(value[group_name])
        parent_group = str(group_name).split("_", 1)[0]
        if parent_group in value:
            return float(value[parent_group])
        if "default" in value:
            return float(value["default"])
        if value:
            return float(next(iter(value.values())))
        return float(fallback)
    return float(value)


def extract_group(file_name: str) -> str:
    """Classify a file into Au, RO, hBN, or Other from naming tokens."""
    tokens = _split_stem_tokens(file_name)
    for token in tokens:
        normalized = _normalize_substrate_token(token)
        if normalized is not None:
            return normalized
    return "Other"


def extract_subgroup(file_name: str) -> str:
    """Return a comparison subgroup label, preserving hBN suffixes when present."""
    group = extract_group(file_name)
    if group != "hBN":
        return group

    tokens = _strip_trailing_metadata_tokens(_split_stem_tokens(file_name))
    for index, token in enumerate(tokens):
        normalized = _normalize_substrate_token(token)
        if normalized != "hBN":
            continue

        subgroup_suffix_tokens = tokens[index + 1 :]
        if not subgroup_suffix_tokens:
            return "hBN"

        return "hBN_" + "_".join(subgroup_suffix_tokens)

    return "hBN"


def extract_date(file_name: str) -> pd.Timestamp:
    """Extract YYYYMMDD date token from a file name."""
    match = _DATE_PATTERN.search(file_name)
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
    return pd.NaT


def _derive_sample_candidate(file_name: str, group_name: str | None = None) -> str:
    """Extract sample name from pattern: sample_substrate(_x)?_laser?(optional)_date."""
    tokens = _split_stem_tokens(file_name)
    if not tokens:
        return ""

    tokens = _strip_trailing_metadata_tokens(tokens)
    if not tokens:
        return ""

    # Find substrate token location so any hBN suffix stays out of the sample name.
    substrate_index: int | None = None
    for index, token in enumerate(tokens):
        normalized = _normalize_substrate_token(token)
        if normalized is None:
            continue

        substrate_index = index
        break

    if substrate_index is None:
        stem = "_".join(tokens)
        return stem.strip("_-")

    sample_tokens = tokens[:substrate_index]
    if sample_tokens:
        return "_".join(sample_tokens).strip("_-")

    # Fallback if the first token itself is the substrate marker.
    return "sample"


def infer_sample_name(
    avg_map_spectra: pd.DataFrame,
    fallback: str = "sample",
) -> str:
    """Infer a representative sample name from averaged map metadata."""
    if avg_map_spectra.empty or "file" not in avg_map_spectra.columns:
        return fallback

    candidates: list[str] = []
    for _, row in avg_map_spectra.iterrows():
        file_name = str(row.get("file", ""))
        group_name = str(row.get("group", "")) if "group" in avg_map_spectra.columns else None
        candidate = _derive_sample_candidate(file_name, group_name=group_name)
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return fallback

    return Counter(candidates).most_common(1)[0][0]


def _extract_sample_code(sample_name: str | None) -> str:
    """Extract compact sample code token (e.g., S6) from sample name text."""
    if sample_name is None:
        return ""

    match = re.search(r"\bS\d+\b", str(sample_name), flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(0).upper()


def minmax_norm(y: np.ndarray) -> np.ndarray:
    """Scale one spectrum to [0, 1]."""
    y_arr = np.asarray(y, dtype=float)
    finite = y_arr[np.isfinite(y_arr)]
    if finite.size == 0:
        return np.zeros_like(y_arr, dtype=float)

    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if np.isclose(y_max, y_min):
        return np.zeros_like(y_arr, dtype=float)
    return (y_arr - y_min) / (y_max - y_min)


def peak_window_norm(
    y: np.ndarray,
    x: np.ndarray,
    peak_center_cm1: float = 1590,
    peak_tolerance_cm1: float = 50,
) -> np.ndarray:
    """Scale one spectrum so the strongest point near `peak_center_cm1` is 1."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    mask &= np.abs(x_arr - float(peak_center_cm1)) <= float(peak_tolerance_cm1)

    if not np.any(mask):
        return np.zeros_like(y_arr, dtype=float)

    peak_height = float(np.max(y_arr[mask]))
    if np.isclose(peak_height, 0.0):
        peak_height = float(np.max(np.abs(y_arr[mask])))
        if np.isclose(peak_height, 0.0):
            return np.zeros_like(y_arr, dtype=float)

    return y_arr / peak_height


def normalize_spectrum(
    y: np.ndarray,
    x: np.ndarray,
    normalization_method: str,
    peak_center_cm1: float,
    peak_tolerance_cm1: float,
) -> np.ndarray:
    """Normalize one spectrum according to the configured method."""
    method = str(normalization_method).strip().lower()
    if method == "minmax":
        return minmax_norm(np.asarray(y, dtype=float))
    if method == "peak_1590":
        return peak_window_norm(
            y=np.asarray(y, dtype=float),
            x=np.asarray(x, dtype=float),
            peak_center_cm1=peak_center_cm1,
            peak_tolerance_cm1=peak_tolerance_cm1,
        )
    raise ValueError(
        "Unsupported normalization_method. Use 'minmax' or 'peak_1590'."
    )


def _resolve_pixel_keep_mask(parsed_item: dict, spectrum_key: str) -> np.ndarray:
    """Return a 2D boolean mask for pixels that are still eligible for averaging."""
    if "spectrum_keep_mask" in parsed_item:
        mask = np.asarray(parsed_item["spectrum_keep_mask"], dtype=bool)
        if mask.ndim != 2:
            raise ValueError(
                f"Expected a 2D spectrum_keep_mask, got shape {mask.shape}"
            )
        return mask

    cube = np.asarray(parsed_item[spectrum_key], dtype=float)
    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D spectra cube for '{spectrum_key}', got shape {cube.shape}"
        )

    with np.errstate(invalid="ignore"):
        valid_mask = np.isfinite(np.nanmean(cube, axis=2))
    return valid_mask


def _select_pixels_far_from_dropped_pixels(valid_mask: np.ndarray, target_count: int) -> np.ndarray:
    """Pick a fixed number of valid pixels, preferring pixels far from dropped regions."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected a 2D valid_mask, got shape {valid_mask.shape}")

    selected_mask = np.zeros_like(valid_mask, dtype=bool)
    valid_coords = np.argwhere(valid_mask)
    if valid_coords.size == 0 or target_count <= 0:
        return selected_mask

    if int(target_count) >= valid_coords.shape[0]:
        return valid_mask.copy()

    dropped_coords = np.argwhere(~valid_mask)
    if dropped_coords.size == 0:
        order = np.lexsort((valid_coords[:, 1], valid_coords[:, 0]))
    else:
        deltas = valid_coords[:, None, :] - dropped_coords[None, :, :]
        distances = np.sqrt(np.sum(deltas * deltas, axis=2))
        nearest_distance = np.min(distances, axis=1)
        order = np.lexsort((valid_coords[:, 1], valid_coords[:, 0], -nearest_distance))

    chosen_coords = valid_coords[order[: int(target_count)]]
    selected_mask[chosen_coords[:, 0], chosen_coords[:, 1]] = True
    return selected_mask


def _build_average_pixel_masks(
    parsed_collection: list[dict],
    spectrum_key: str,
    balance_pixel_count_groups: Iterable[str] | None,
) -> dict[str, np.ndarray]:
    """Build the per-file pixel masks used for balanced map averaging.

    Auto mode (`balance_pixel_count_groups is None`) balances within subgroup scope,
    so `hBN_1` and `hBN_2` are handled independently.
    """
    map_entries: list[dict] = []

    for item in parsed_collection:
        file_name = item["path"].name
        cube = item[spectrum_key]
        map_entries.append(
            {
                "file": file_name,
                "group": extract_group(file_name),
                "subgroup": extract_subgroup(file_name),
                "cube": cube,
                "valid_mask": _resolve_pixel_keep_mask(item, spectrum_key),
            }
        )

    if balance_pixel_count_groups is None:
        balance_scopes = {
            entry["subgroup"]
            for entry in map_entries
            if np.count_nonzero(entry["valid_mask"]) < int(entry["valid_mask"].size)
        }
    else:
        balance_scopes = {
            str(group).strip() for group in balance_pixel_count_groups or [] if str(group).strip()
        }
    balanced_targets: dict[str, int] = {}
    for scope_name in balance_scopes:
        scope_counts = [
            int(np.count_nonzero(entry["valid_mask"]))
            for entry in map_entries
            if (
                int(np.count_nonzero(entry["valid_mask"])) > 0
                and (entry["subgroup"] == scope_name or entry["group"] == scope_name)
            )
        ]
        if scope_counts:
            balanced_targets[scope_name] = min(scope_counts)

    average_pixel_masks: dict[str, np.ndarray] = {}
    for entry in map_entries:
        valid_mask = np.asarray(entry["valid_mask"], dtype=bool)
        target_count = balanced_targets.get(entry["subgroup"])
        if target_count is None:
            target_count = balanced_targets.get(entry["group"])
        average_pixel_masks[entry["file"]] = (
            _select_pixels_far_from_dropped_pixels(valid_mask, target_count)
            if target_count is not None
            else valid_mask.copy()
        )

    return average_pixel_masks


def annotate_average_pixel_masks(
    parsed_collection: list[dict],
    spectrum_key: str = "corrected_spectra_cube",
    balance_pixel_count_groups: Iterable[str] | None = None,
) -> list[dict]:
    """Attach the average-pixel mask to each parsed map for explorer overlays."""
    average_pixel_masks = _build_average_pixel_masks(
        parsed_collection=parsed_collection,
        spectrum_key=spectrum_key,
        balance_pixel_count_groups=balance_pixel_count_groups,
    )

    annotated_collection: list[dict] = []
    for item in parsed_collection:
        file_name = item["path"].name
        average_pixel_mask = average_pixel_masks.get(file_name)
        annotated_item = {
            **item,
            "average_pixel_mask": average_pixel_mask,
            "average_pixels_used": int(np.count_nonzero(average_pixel_mask)) if average_pixel_mask is not None else 0,
        }
        annotated_collection.append(annotated_item)

    return annotated_collection


def build_average_map_spectra(
    parsed_collection: list[dict],
    spectrum_key: str = "corrected_spectra_cube",
    keep_groups: Iterable[str] = ("Au", "RO", "hBN"),
    balance_pixel_count_groups: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Average each map (x/y pixels) into one spectrum and annotate metadata.

    When balance_pixel_count_groups is None, any subgroup with at least one dropped
    pixel automatically has its maps balanced to the same pixel count (for example,
    hBN_1 does not force hBN_2 to shrink). The selected pixels are the valid pixels
    farthest from dropped regions, so the retained sample stays away from gated-out
    areas as much as possible.
    """
    if not parsed_collection:
        return pd.DataFrame()

    average_pixel_masks = _build_average_pixel_masks(
        parsed_collection=parsed_collection,
        spectrum_key=spectrum_key,
        balance_pixel_count_groups=balance_pixel_count_groups,
    )

    records: list[dict] = []
    for item in parsed_collection:
        file_name = item["path"].name
        group = extract_group(file_name)
        pixels_used_mask = average_pixel_masks[file_name]
        cube = np.asarray(item[spectrum_key], dtype=float)
        selected_spectra = cube[pixels_used_mask, :]
        if selected_spectra.size == 0:
            mean_spectrum = np.full(np.asarray(item["wavenumber_cm1"], dtype=float).shape, np.nan)
        else:
            with np.errstate(invalid="ignore"):
                mean_spectrum = np.nanmean(selected_spectra, axis=0)

        records.append(
            {
                "file": file_name,
                "group": group,
                "subgroup": extract_subgroup(file_name),
                "date": extract_date(file_name),
                "wavenumber_cm1": item["wavenumber_cm1"],
                "pixels_available": int(np.count_nonzero(_resolve_pixel_keep_mask(item, spectrum_key))),
                "pixels_used": int(np.count_nonzero(pixels_used_mask)),
                "mean_spectrum": mean_spectrum,
                "average_pixel_mask": pixels_used_mask,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    keep_groups_set = set(keep_groups)
    if keep_groups_set:
        df = df[df["group"].isin(keep_groups_set)].copy()

    return df.sort_values(["group", "subgroup", "date", "file"]).reset_index(drop=True)


def plot_grouped_spectra(
    df: pd.DataFrame,
    value_col: str,
    title_suffix: str,
    y_label: str,
    title_prefix: str | None = None,
    group_col: str = "group",
    panel_order: Iterable[str] = ("Au", "RO", "hBN"),
    stack_scale: float | Mapping[str, float] = 1.0,
    stack_extra_gap: float | Mapping[str, float] = 0.0,
    wavenumber_min: float | None = None,
    wavenumber_max: float | None = None,
    output_path: Path | None = None,
) -> plt.Figure | None:
    """Plot stacked spectra and optionally export one figure per group."""
    if group_col not in df.columns:
        raise KeyError(f"DataFrame is missing required grouping column '{group_col}'")

    available_groups = set(df[group_col].dropna().unique())
    panels = [group_name for group_name in panel_order if group_name in available_groups]
    if not panels:
        return None

    if output_path is not None and len(panels) > 1 and output_path.exists():
        output_path.unlink()

    def _draw_group(axis: plt.Axes, group_name: str) -> None:
        subset = df[df[group_col] == group_name].sort_values(["date", "file"])
        spectra = [
            _slice_spectrum_to_wavenumber_range(
                wavenumber_cm1=row["wavenumber_cm1"],
                intensity=row[value_col],
                wavenumber_min=wavenumber_min,
                wavenumber_max=wavenumber_max,
            )[1]
            for _, row in subset.iterrows()
        ]
        group_stack_scale = _resolve_group_value(stack_scale, group_name, fallback=1.0)
        group_stack_extra_gap = _resolve_group_value(stack_extra_gap, group_name, fallback=0.0)
        offset_step = _compute_stack_step(
            spectra=spectra,
            stack_scale=group_stack_scale,
            stack_extra_gap=group_stack_extra_gap,
        )

        for stack_index, (_, row) in enumerate(subset.iterrows()):
            label_date = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "Unknown date"
            # Apply a deterministic offset so chronological ordering is visible.
            window_x, window_y = _slice_spectrum_to_wavenumber_range(
                wavenumber_cm1=row["wavenumber_cm1"],
                intensity=row[value_col],
                wavenumber_min=wavenumber_min,
                wavenumber_max=wavenumber_max,
            )
            stacked_y = window_y + stack_index * offset_step
            axis.plot(
                window_x,
                stacked_y,
                linewidth=1.6,
                label=f"{label_date} | {row['file']}",
            )

        base_title = f"{title_prefix}_{group_name}" if title_prefix else str(group_name)
        axis.set_title(f"{base_title} ({title_suffix})" if title_suffix else base_title)
        axis.set_xlabel("Wavenumber (cm^-1)")
        axis.set_ylabel(f"{y_label} (stacked)")
        if wavenumber_min is not None or wavenumber_max is not None:
            axis.set_xlim(left=wavenumber_min, right=wavenumber_max)
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(
            handles[::-1],
            labels[::-1],
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )

    last_fig: plt.Figure | None = None

    for export_index, group_name in enumerate(panels, start=1):
        fig, axis = plt.subplots(1, 1, figsize=(9, 5), sharex=True)
        _draw_group(axis, group_name)

        subset = df[df[group_col] == group_name].sort_values(["date", "file"])
        spectra = [
            _slice_spectrum_to_wavenumber_range(
                wavenumber_cm1=row["wavenumber_cm1"],
                intensity=row[value_col],
                wavenumber_min=wavenumber_min,
                wavenumber_max=wavenumber_max,
            )[1]
            for _, row in subset.iterrows()
        ]
        group_stack_scale = _resolve_group_value(stack_scale, group_name, fallback=1.0)
        group_stack_extra_gap = _resolve_group_value(stack_extra_gap, group_name, fallback=0.0)
        offset_step = _compute_stack_step(
            spectra=spectra,
            stack_scale=group_stack_scale,
            stack_extra_gap=group_stack_extra_gap,
        )

        fig.tight_layout(rect=(0, 0, 0.86, 1))
        if output_path is not None:
            if len(panels) == 1:
                target_path = output_path
            else:
                target_path = _build_per_group_output_path(output_path, group_name, export_index)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_path, dpi=200, bbox_inches="tight")
            _export_grouped_spectra_plot_csv(
                subset=subset,
                value_col=value_col,
                group_col=group_col,
                target_path=target_path,
                offset_step=offset_step,
            )
        plt.show()
        last_fig = fig

    return last_fig


def _prepare_export_subdir(output_dir: Path, *parts: str) -> Path:
    """Create and return a nested export directory."""
    numbered_parts = [_EXPORT_PART_PREFIXES.get(part, part) for part in parts]
    target_dir = output_dir.joinpath(*numbered_parts)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _build_group_plot_path(
    output_dir: Path,
    category: str,
    scope: str,
    stem_suffix: str | None = None,
) -> Path:
    """Return a placeholder PNG path whose directory determines export location."""
    file_name = "plot.png" if not stem_suffix else f"plot_{stem_suffix}.png"
    return _prepare_export_subdir(output_dir, "plots", category, scope) / file_name


def _build_per_group_output_path(output_path: Path, group_name: str, index: int) -> Path:
    """Build one export filename per group while preserving any range suffix in the stem."""
    safe_group = _sanitize_export_stem(str(group_name)) or "group"
    stem = _prefix_indexed_stem(safe_group, index=index)
    stem_remainder = output_path.stem.removeprefix("plot").lstrip("_")
    if stem_remainder:
        stem = f"{stem}_{stem_remainder}"
    return output_path.with_name(f"{stem}{output_path.suffix}")


def _coerce_optional_wavenumber_bound(value: object) -> float | None:
    """Convert a configured wavenumber bound to float while preserving None."""
    if value is None:
        return None
    return float(value)


def _format_wavenumber_bound_for_stem(value: float | None) -> str:
    """Return a filesystem-safe token for one wavenumber bound."""
    if value is None:
        return "auto"

    numeric_value = float(value)
    if float(numeric_value).is_integer():
        return str(int(numeric_value))

    return f"{numeric_value:g}".replace("-", "m").replace(".", "p")


def _build_wavenumber_range_stem(
    wavenumber_min: float | None,
    wavenumber_max: float | None,
    label: str | None = None,
) -> str:
    """Build an export suffix that identifies one wavenumber window."""
    if label is not None and str(label).strip():
        safe_label = _sanitize_export_stem(str(label).strip())
        if safe_label:
            return safe_label

    return (
        f"wn_{_format_wavenumber_bound_for_stem(wavenumber_min)}_"
        f"{_format_wavenumber_bound_for_stem(wavenumber_max)}cm-1"
    )


def resolve_plot_wavenumber_ranges(
    wavenumber_ranges: object = PLOT_WAVENUMBER_RANGES,
) -> list[dict[str, object]]:
    """Resolve one configured range or multiple configured ranges for Stage 6 exports."""
    if wavenumber_ranges is None:
        return [
            {
                "wavenumber_min": None,
                "wavenumber_max": None,
                "label": None,
                "export_stem_suffix": None,
            }
        ]

    if isinstance(wavenumber_ranges, Mapping):
        raw_range_specs = [wavenumber_ranges]
    elif isinstance(wavenumber_ranges, (list, tuple)) and len(wavenumber_ranges) == 2 and not any(
        isinstance(value, (list, tuple, Mapping)) for value in wavenumber_ranges
    ):
        raw_range_specs = [wavenumber_ranges]
    elif isinstance(wavenumber_ranges, Iterable) and not isinstance(wavenumber_ranges, (str, bytes)):
        raw_range_specs = list(wavenumber_ranges)
    else:
        raise ValueError(
            "PLOT_WAVENUMBER_RANGES must be a single (min, max) tuple or an iterable of range tuples"
        )

    if not raw_range_specs:
        raise ValueError("PLOT_WAVENUMBER_RANGES cannot be empty when provided")

    resolved_ranges: list[dict[str, object]] = []
    for range_spec in raw_range_specs:
        label = None
        raw_stack_scale_override = None
        raw_stack_extra_gap_override = None
        norm_stack_scale_override = None
        norm_stack_extra_gap_override = None
        if isinstance(range_spec, Mapping):
            range_min = range_spec.get("min", range_spec.get("wavenumber_min"))
            range_max = range_spec.get("max", range_spec.get("wavenumber_max"))
            label = range_spec.get("label")
            raw_stack_scale_override = range_spec.get("raw_stack_scale")
            raw_stack_extra_gap_override = range_spec.get("raw_stack_extra_gap")
            norm_stack_scale_override = range_spec.get("norm_stack_scale")
            norm_stack_extra_gap_override = range_spec.get("norm_stack_extra_gap")
        else:
            try:
                range_min, range_max = range_spec
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Each PLOT_WAVENUMBER_RANGES entry must be a 2-item tuple/list or a mapping"
                ) from exc

        resolved_min = _coerce_optional_wavenumber_bound(range_min)
        resolved_max = _coerce_optional_wavenumber_bound(range_max)

        if (
            resolved_min is not None
            and resolved_max is not None
            and resolved_min > resolved_max
        ):
            raise ValueError("Each configured wavenumber range must satisfy min <= max")

        resolved_ranges.append(
            {
                "wavenumber_min": resolved_min,
                "wavenumber_max": resolved_max,
                "label": None if label is None else str(label),
                "export_stem_suffix": None,
                "raw_stack_scale": raw_stack_scale_override,
                "raw_stack_extra_gap": raw_stack_extra_gap_override,
                "norm_stack_scale": norm_stack_scale_override,
                "norm_stack_extra_gap": norm_stack_extra_gap_override,
            }
        )

    if len(resolved_ranges) > 1:
        for resolved_range in resolved_ranges:
            resolved_range["export_stem_suffix"] = _build_wavenumber_range_stem(
                resolved_range["wavenumber_min"],
                resolved_range["wavenumber_max"],
                label=resolved_range["label"],
            )

    return resolved_ranges


def _resolve_range_plot_value(
    range_config: Mapping[str, object],
    key: str,
    default_value: float | Mapping[str, float],
) -> float | Mapping[str, float]:
    """Return a range-specific plot setting when provided, else the shared default."""
    override_value = range_config.get(key)
    if override_value is None:
        return default_value
    if isinstance(override_value, Mapping):
        return {str(name): float(value) for name, value in override_value.items()}
    return float(override_value)


def _slice_spectrum_to_wavenumber_range(
    wavenumber_cm1: np.ndarray,
    intensity: np.ndarray,
    wavenumber_min: float | None,
    wavenumber_max: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one spectrum restricted to the requested plotting window."""
    x = np.asarray(wavenumber_cm1, dtype=float)
    y = np.asarray(intensity, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if wavenumber_min is not None:
        mask &= x >= float(wavenumber_min)
    if wavenumber_max is not None:
        mask &= x <= float(wavenumber_max)

    return x[mask], y[mask]


def _export_grouped_spectra_plot_csv(
    subset: pd.DataFrame,
    value_col: str,
    group_col: str,
    target_path: Path,
    offset_step: float,
) -> None:
    """Export plotted stacked traces as a sidecar CSV next to the PNG."""
    rows: list[pd.DataFrame] = []
    for stack_index, (_, row) in enumerate(subset.iterrows()):
        wavenumber = np.asarray(row["wavenumber_cm1"], dtype=float)
        intensity = np.asarray(row[value_col], dtype=float)
        offset = float(stack_index) * float(offset_step)
        rows.append(
            pd.DataFrame(
                {
                    group_col: str(row[group_col]),
                    "subgroup": row.get("subgroup"),
                    "file": str(row["file"]),
                    "date": row["date"],
                    "stack_index": int(stack_index),
                    "stack_offset": offset,
                    "wavenumber_cm1": wavenumber,
                    "intensity": intensity,
                    "stacked_intensity": intensity + offset,
                }
            )
        )

    if not rows:
        return

    sidecar_path = target_path.with_suffix(".csv")
    pd.concat(rows, ignore_index=True).to_csv(sidecar_path, index=False)


def _export_overlap_plot_csv(
    subset: pd.DataFrame,
    group_col: str,
    target_path: Path,
) -> None:
    """Export normalized overlap traces as a sidecar CSV next to the PNG."""
    rows: list[pd.DataFrame] = []
    for _, row in subset.iterrows():
        rows.append(
            pd.DataFrame(
                {
                    group_col: str(row[group_col]),
                    "subgroup": row.get("subgroup"),
                    "file": str(row["file"]),
                    "date": row["date"],
                    "wavenumber_cm1": np.asarray(row["wavenumber_cm1"], dtype=float),
                    "normalized_intensity": np.asarray(row["mean_spectrum_norm"], dtype=float),
                }
            )
        )

    if not rows:
        return

    sidecar_path = target_path.with_suffix(".csv")
    pd.concat(rows, ignore_index=True).to_csv(sidecar_path, index=False)


def _export_peak_ratio_plot_csv(
    subset: pd.DataFrame,
    date_label_map: dict,
    target_path: Path,
) -> None:
    """Export peak-ratio trend points as a sidecar CSV next to the PNG."""
    export_df = subset.copy()
    export_df["date_label"] = export_df["date"].map(date_label_map)
    export_df.to_csv(target_path.with_suffix(".csv"), index=False)


def plot_normalized_overlap_by_group(
    avg_map_spectra: pd.DataFrame,
    groups: Iterable[str] = ("Au", "RO", "hBN"),
    wavenumber_min: float | None = None,
    wavenumber_max: float | None = None,
    sample_code: str | None = None,
    output_dir: Path | None = None,
    group_col: str = "group",
    output_name_suffix: str | None = None,
) -> list[Path]:
    """Plot non-stacked normalized overlaps per group and optionally export PNGs."""
    if avg_map_spectra.empty or "mean_spectrum_norm" not in avg_map_spectra.columns:
        return []

    if group_col not in avg_map_spectra.columns:
        raise KeyError(f"DataFrame is missing required grouping column '{group_col}'")

    exported_paths: list[Path] = []
    available_groups = set(avg_map_spectra[group_col].dropna().unique())

    for export_index, group_name in enumerate(groups, start=1):
        if group_name not in available_groups:
            continue

        subset = avg_map_spectra[(avg_map_spectra[group_col] == group_name)].sort_values(["date", "file"])
        if subset.empty:
            continue

        fig, axis = plt.subplots(figsize=(20, 5))

        for _, row in subset.iterrows():
            label_date = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "Unknown date"
            window_x, window_y = _slice_spectrum_to_wavenumber_range(
                wavenumber_cm1=row["wavenumber_cm1"],
                intensity=row["mean_spectrum_norm"],
                wavenumber_min=wavenumber_min,
                wavenumber_max=wavenumber_max,
            )
            axis.plot(
                window_x,
                window_y,
                linewidth=0.8,
                label=f"{label_date} | {row['file']}",
            )

        title_prefix = f"{sample_code}_{group_name}" if sample_code else str(group_name)
        axis.set_title(f"{title_prefix} (norm overlap)")
        axis.set_xlabel("Wavenumber (cm^-1)")
        axis.set_ylabel("Normalized intensity")
        if wavenumber_min is not None or wavenumber_max is not None:
            axis.set_xlim(left=wavenumber_min, right=wavenumber_max)
        axis.grid(alpha=0.25)

        # Match legend order with visual stacking top-to-bottom convention.
        handles, labels = axis.get_legend_handles_labels()
        axis.legend(
            handles[::-1],
            labels[::-1],
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )

        fig.tight_layout(rect=(0, 0, 0.82, 1))

        if output_dir is not None:
            safe_group = _sanitize_export_stem(str(group_name))
            out_name = _prefix_indexed_stem(safe_group, index=export_index)
            if output_name_suffix:
                out_name = f"{out_name}_{output_name_suffix}"
            out_path = output_dir / f"{out_name}.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            _export_overlap_plot_csv(
                subset=subset,
                group_col=group_col,
                target_path=out_path,
            )
            exported_paths.append(out_path)

        plt.show()

    return exported_paths


def plot_average_and_normalized_map_spectra(
    parsed_collection: list[dict],
    spectrum_key: str = "corrected_spectra_cube",
    groups: Iterable[str] = ("Au", "RO", "hBN"),
    balance_pixel_count_groups: Iterable[str] | None = None,
    normalization_method: str = NORMALIZATION_METHOD,
    normalization_peak_center_cm1: float = NORMALIZATION_PEAK_CENTER_CM1,
    normalization_peak_tolerance_cm1: float = NORMALIZATION_PEAK_TOLERANCE_CM1,
    raw_stack_scale: float | Mapping[str, float] = 1.0,
    raw_stack_extra_gap: float | Mapping[str, float] = 0.0,
    norm_stack_scale: float | Mapping[str, float] = 1.0,
    norm_stack_extra_gap: float | Mapping[str, float] = 0.0,
    wavenumber_ranges: object = PLOT_WAVENUMBER_RANGES,
    peak_ratio_wavenumber_ranges: object = PEAK_RATIO_WAVENUMBER_RANGES,
    output_dir: Path | None = None,
    sample_name: str | None = None,
) -> pd.DataFrame:
    """Build per-map average spectra, then plot raw and normalized versions.

    Peak-ratio windows are configured independently from plot/export windows.
    """
    avg_map_spectra = build_average_map_spectra(
        parsed_collection=parsed_collection,
        spectrum_key=spectrum_key,
        keep_groups=groups,
        balance_pixel_count_groups=balance_pixel_count_groups,
    )

    if avg_map_spectra.empty:
        return avg_map_spectra

    resolved_sample_name = sample_name or infer_sample_name(avg_map_spectra)
    sample_code = _extract_sample_code(resolved_sample_name)
    resolved_ranges = resolve_plot_wavenumber_ranges(wavenumber_ranges=wavenumber_ranges)
    resolved_peak_ratio_ranges = resolve_plot_wavenumber_ranges(
        wavenumber_ranges=peak_ratio_wavenumber_ranges
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    raw_title_suffix = "raw"

    avg_map_spectra["mean_spectrum_norm"] = avg_map_spectra.apply(
        lambda row: normalize_spectrum(
            y=np.asarray(row["mean_spectrum"], dtype=float),
            x=np.asarray(row["wavenumber_cm1"], dtype=float),
            normalization_method=normalization_method,
            peak_center_cm1=normalization_peak_center_cm1,
            peak_tolerance_cm1=normalization_peak_tolerance_cm1,
        ),
        axis=1,
    )

    normalized_title_suffix = "norm"
    normalized_y_label = "Normalized intensity"
    if str(normalization_method).strip().lower() == "minmax":
        normalized_title_suffix = "norm:minmax"
        normalized_y_label = "Normalized intensity (0-1)"
    elif str(normalization_method).strip().lower() == "peak_1590":
        normalized_title_suffix = (
            f"norm:peak@{normalization_peak_center_cm1:g}+/-{normalization_peak_tolerance_cm1:g}"
        )
        normalized_y_label = "Relative intensity (I / I_1590-window-max)"

    hbn_subgroup_spectra = avg_map_spectra[
        (avg_map_spectra["group"] == "hBN") & avg_map_spectra["subgroup"].notna()
    ].copy()
    hbn_subgroups = [str(group_name) for group_name in pd.unique(hbn_subgroup_spectra["subgroup"])]

    for range_config in resolved_ranges:
        range_min = range_config["wavenumber_min"]
        range_max = range_config["wavenumber_max"]
        range_suffix = range_config["export_stem_suffix"]
        range_raw_stack_scale = _resolve_range_plot_value(
            range_config,
            "raw_stack_scale",
            raw_stack_scale,
        )
        range_raw_stack_extra_gap = _resolve_range_plot_value(
            range_config,
            "raw_stack_extra_gap",
            raw_stack_extra_gap,
        )
        range_norm_stack_scale = _resolve_range_plot_value(
            range_config,
            "norm_stack_scale",
            norm_stack_scale,
        )
        range_norm_stack_extra_gap = _resolve_range_plot_value(
            range_config,
            "norm_stack_extra_gap",
            norm_stack_extra_gap,
        )

        current_raw_output_path = None
        current_norm_output_path = None
        if output_dir is not None:
            current_raw_output_path = _build_group_plot_path(
                output_dir,
                "avg_stack",
                "groups",
                stem_suffix=range_suffix,
            )
            current_norm_output_path = _build_group_plot_path(
                output_dir,
                "norm_stack",
                "groups",
                stem_suffix=range_suffix,
            )

        plot_grouped_spectra(
            avg_map_spectra,
            value_col="mean_spectrum",
            title_prefix=sample_code,
            title_suffix=raw_title_suffix,
            y_label="Intensity (a.u.)",
            group_col="group",
            panel_order=groups,
            stack_scale=range_raw_stack_scale,
            stack_extra_gap=range_raw_stack_extra_gap,
            wavenumber_min=range_min,
            wavenumber_max=range_max,
            output_path=current_raw_output_path,
        )

        plot_grouped_spectra(
            avg_map_spectra,
            value_col="mean_spectrum_norm",
            title_prefix=sample_code,
            title_suffix=normalized_title_suffix,
            y_label=normalized_y_label,
            group_col="group",
            panel_order=groups,
            stack_scale=range_norm_stack_scale,
            stack_extra_gap=range_norm_stack_extra_gap,
            wavenumber_min=range_min,
            wavenumber_max=range_max,
            output_path=current_norm_output_path,
        )

        plot_normalized_overlap_by_group(
            avg_map_spectra=avg_map_spectra,
            groups=groups,
            wavenumber_min=range_min,
            wavenumber_max=range_max,
            sample_code=sample_code,
            output_dir=_prepare_export_subdir(output_dir, "plots", "norm_overlap", "groups") if output_dir is not None else None,
            group_col="group",
            output_name_suffix=range_suffix,
        )

        if len(hbn_subgroups) > 1:
            subgroup_raw_output_path = None
            subgroup_norm_output_path = None
            if output_dir is not None:
                subgroup_raw_output_path = _build_group_plot_path(
                    output_dir,
                    "avg_stack",
                    "hbn_subgroups",
                    stem_suffix=range_suffix,
                )
                subgroup_norm_output_path = _build_group_plot_path(
                    output_dir,
                    "norm_stack",
                    "hbn_subgroups",
                    stem_suffix=range_suffix,
                )

            plot_grouped_spectra(
                hbn_subgroup_spectra,
                value_col="mean_spectrum",
                title_prefix=sample_code,
                title_suffix=raw_title_suffix,
                y_label="Intensity (a.u.)",
                group_col="subgroup",
                panel_order=hbn_subgroups,
                stack_scale=range_raw_stack_scale,
                stack_extra_gap=range_raw_stack_extra_gap,
                wavenumber_min=range_min,
                wavenumber_max=range_max,
                output_path=subgroup_raw_output_path,
            )

            plot_grouped_spectra(
                hbn_subgroup_spectra,
                value_col="mean_spectrum_norm",
                title_prefix=sample_code,
                title_suffix=normalized_title_suffix,
                y_label=normalized_y_label,
                group_col="subgroup",
                panel_order=hbn_subgroups,
                stack_scale=range_norm_stack_scale,
                stack_extra_gap=range_norm_stack_extra_gap,
                wavenumber_min=range_min,
                wavenumber_max=range_max,
                output_path=subgroup_norm_output_path,
            )

            plot_normalized_overlap_by_group(
                avg_map_spectra=hbn_subgroup_spectra,
                groups=hbn_subgroups,
                wavenumber_min=range_min,
                wavenumber_max=range_max,
                sample_code=sample_code,
                output_dir=_prepare_export_subdir(output_dir, "plots", "norm_overlap", "hbn_subgroups") if output_dir is not None else None,
                group_col="subgroup",
                output_name_suffix=range_suffix,
            )

    if len(hbn_subgroups) > 1:
        for peak_ratio_range_config in resolved_peak_ratio_ranges:
            ratio_min = peak_ratio_range_config["wavenumber_min"]
            ratio_max = peak_ratio_range_config["wavenumber_max"]
            ratio_suffix = peak_ratio_range_config["export_stem_suffix"]

            subgroup_peak_ratio_output_path = None
            if output_dir is not None:
                subgroup_peak_ratio_output_path = _build_group_plot_path(
                    output_dir,
                    "peak_ratio",
                    "hbn_subgroups",
                    stem_suffix=ratio_suffix,
                )

            subgroup_peak_ratio_df = build_peak_ratio_table(
                avg_map_spectra=hbn_subgroup_spectra,
                spectrum_col="mean_spectrum",
                distance=15,
                wavenumber_min=ratio_min,
                wavenumber_max=ratio_max,
            )
            plot_peak_ratio_by_date(
                peak_ratio_df=subgroup_peak_ratio_df,
                groups=hbn_subgroups,
                output_path=subgroup_peak_ratio_output_path,
                group_col="subgroup",
            )

    return avg_map_spectra


def calculate_top_two_peak_ratio(
    wavenumber_cm1: np.ndarray,
    spectrum: np.ndarray,
    distance: int = 10,
    wavenumber_min: float | None = None,
    wavenumber_max: float | None = None,
) -> dict:
    """Return top-two peaks with peak1 ordered at lower wavenumber than peak2."""
    y = np.asarray(spectrum, dtype=float)
    x = np.asarray(wavenumber_cm1, dtype=float)

    if (
        wavenumber_min is not None
        and wavenumber_max is not None
        and float(wavenumber_min) > float(wavenumber_max)
    ):
        raise ValueError("wavenumber_min must be <= wavenumber_max")

    # Restrict candidate peaks to the configured analysis window before
    # searching for the highest peaks.
    range_mask = np.ones_like(x, dtype=bool)
    if wavenumber_min is not None:
        range_mask &= x >= float(wavenumber_min)
    if wavenumber_max is not None:
        range_mask &= x <= float(wavenumber_max)

    window_mask = np.isfinite(x) & np.isfinite(y) & range_mask
    if not np.any(window_mask):
        return {
            "peak_count": 0,
            "peak1_wavenumber_cm1": np.nan,
            "peak1_intensity": np.nan,
            "peak2_wavenumber_cm1": np.nan,
            "peak2_intensity": np.nan,
            "peak_ratio": np.nan,
        }

    window_x = x[window_mask]
    window_y = y[window_mask]

    peaks, _ = find_peaks(window_y, distance=distance)
    if len(peaks) < 2:
        return {
            "peak_count": int(len(peaks)),
            "peak1_wavenumber_cm1": np.nan,
            "peak1_intensity": np.nan,
            "peak2_wavenumber_cm1": np.nan,
            "peak2_intensity": np.nan,
            "peak_ratio": np.nan,
        }

    peak_intensities = window_y[peaks]
    top_two_indices = np.argsort(peak_intensities)[-2:][::-1]
    top_two_peak_positions = peaks[top_two_indices]

    # Keep selecting the two strongest peaks, then order them by x-position
    # so peak1 is always the lower-wavenumber peak and ratio is peak1/peak2.
    ordered_by_wavenumber = top_two_peak_positions[np.argsort(window_x[top_two_peak_positions])]
    peak1_idx = int(ordered_by_wavenumber[0])
    peak2_idx = int(ordered_by_wavenumber[1])

    i1 = float(window_y[peak1_idx])
    i2 = float(window_y[peak2_idx])
    ratio = np.nan if np.isclose(i2, 0.0) else float(i1 / i2)

    return {
        "peak_count": int(len(peaks)),
        "peak1_wavenumber_cm1": float(window_x[peak1_idx]),
        "peak1_intensity": i1,
        "peak2_wavenumber_cm1": float(window_x[peak2_idx]),
        "peak2_intensity": i2,
        "peak_ratio": ratio,
    }


def build_peak_ratio_table(
    avg_map_spectra: pd.DataFrame,
    spectrum_col: str = "mean_spectrum",
    distance: int = 10,
    wavenumber_min: float | None = None,
    wavenumber_max: float | None = None,
    group_col: str = "group",
) -> pd.DataFrame:
    """Compute top-two peak ratios for each Stage 5 spectrum."""
    if avg_map_spectra.empty:
        return pd.DataFrame(
            columns=[
                "file",
                "group",
                "subgroup",
                "date",
                "peak_count",
                "peak1_wavenumber_cm1",
                "peak1_intensity",
                "peak2_wavenumber_cm1",
                "peak2_intensity",
                "peak_ratio",
            ]
        )

    records: list[dict] = []
    for _, row in avg_map_spectra.iterrows():
        peak_info = calculate_top_two_peak_ratio(
            wavenumber_cm1=row["wavenumber_cm1"],
            spectrum=row[spectrum_col],
            distance=distance,
            wavenumber_min=wavenumber_min,
            wavenumber_max=wavenumber_max,
        )
        records.append(
            {
                "file": row["file"],
                "group": row["group"],
                "subgroup": row.get("subgroup", row["group"]),
                "date": row["date"],
                **peak_info,
            }
        )

    peak_ratio_df = pd.DataFrame(records)
    if group_col not in peak_ratio_df.columns:
        raise KeyError(f"DataFrame is missing required grouping column '{group_col}'")
    return peak_ratio_df.sort_values([group_col, "subgroup", "date", "file"]).reset_index(drop=True)


def plot_peak_ratio_by_date(
    peak_ratio_df: pd.DataFrame,
    groups: Iterable[str] = ("Au", "RO", "hBN"),
    output_path: Path | None = None,
    group_col: str = "group",
) -> plt.Figure | None:
    """Plot per-spectrum peak height ratio trends over date.

    When multiple groups are requested and an output path is provided, each
    group's figure is exported as an individual file using
    ``<stem>_<group><suffix>``.
    """
    if peak_ratio_df.empty:
        return None

    if group_col not in peak_ratio_df.columns:
        raise KeyError(f"DataFrame is missing required grouping column '{group_col}'")

    available_groups = set(peak_ratio_df[group_col].dropna().unique())
    panels = [group_name for group_name in groups if group_name in available_groups]
    if not panels:
        return None

    if output_path is not None and len(panels) > 1 and output_path.exists():
        # Remove legacy combined export so output folder only contains per-group plots.
        output_path.unlink()

    unique_dates = sorted(peak_ratio_df["date"].dropna().unique())
    date_label_map = {
        date_value: f"date{index + 1}" for index, date_value in enumerate(unique_dates)
    }

    def _draw_group(axis: plt.Axes, group_name: str) -> bool:
        subset = peak_ratio_df[
            (peak_ratio_df[group_col] == group_name)
            & peak_ratio_df["date"].notna()
            & peak_ratio_df["peak_ratio"].notna()
        ].sort_values(["date", "file"])

        if subset.empty:
            return False

        x_labels = [date_label_map[date_value] for date_value in subset["date"]]

        axis.plot(
            x_labels,
            subset["peak_ratio"],
            marker="o",
            linewidth=1.8,
        )
        axis.set_title(f"{group_name} peak ratio by date")
        axis.set_xlabel("Date")
        axis.set_ylabel("Peak height ratio (I_max1 / I_max2)")
        axis.grid(alpha=0.25)

        return True

    last_fig: plt.Figure | None = None

    for export_index, group_name in enumerate(panels, start=1):
        fig, axis = plt.subplots(1, 1, figsize=(6, 4), sharey=False)

        subset = peak_ratio_df[
            (peak_ratio_df[group_col] == group_name)
            & peak_ratio_df["date"].notna()
            & peak_ratio_df["peak_ratio"].notna()
        ].sort_values(["date", "file"])

        if subset.empty:
            plt.close(fig)
            continue

        x_labels = [date_label_map[date_value] for date_value in subset["date"]]

        axis.plot(
            x_labels,
            subset["peak_ratio"],
            marker="o",
            linewidth=1.8,
        )
        axis.set_title(f"{group_name} peak ratio by date")
        axis.set_xlabel("Date")
        axis.set_ylabel("Peak height ratio (I_max1 / I_max2)")
        axis.grid(alpha=0.25)

        if date_label_map:
            date_legend_text = " | ".join(
                f"{label}={pd.Timestamp(date_value).strftime('%Y-%m-%d')}"
                for date_value, label in date_label_map.items()
            )
            fig.text(0.5, 0.01, date_legend_text, ha="center", va="bottom", fontsize=9)

        fig.tight_layout(rect=(0, 0.05, 1, 1))
        if output_path is not None:
            if len(panels) == 1:
                target_path = output_path
            else:
                target_path = _build_per_group_output_path(output_path, group_name, export_index)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_path, dpi=200, bbox_inches="tight")
            _export_peak_ratio_plot_csv(
                subset=subset,
                date_label_map=date_label_map,
                target_path=target_path,
            )
        plt.show()
        last_fig = fig

    return last_fig


def _sanitize_export_stem(file_name: str) -> str:
    """Build a filesystem-safe file stem from the source Raman file name."""
    stem = Path(file_name).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def _prefix_indexed_stem(stem: str, index: int) -> str:
    """Prefix a sanitized stem with a stable two-digit logical order."""
    safe_stem = _sanitize_export_stem(stem)
    return f"{int(index):02d}_{safe_stem}"


def _export_spectra_per_file(
    avg_map_spectra: pd.DataFrame,
    spectrum_col: str,
    output_dir: Path,
    file_suffix: str,
) -> None:
    """Export one spectrum per CSV and clear old CSVs before writing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for existing_csv in output_dir.glob("*.csv"):
        existing_csv.unlink()

    for index, (_, row) in enumerate(avg_map_spectra.iterrows(), start=1):
        one_df = pd.DataFrame(
            {
                "wavenumber_cm1": np.asarray(row["wavenumber_cm1"], dtype=float),
                "intensity": np.asarray(row[spectrum_col], dtype=float),
            }
        )

        prefixed_stem = _prefix_indexed_stem(str(row["file"]), index=index)
        one_df.to_csv(output_dir / f"{prefixed_stem}_{file_suffix}.csv", index=False)


def _backup_code_snapshot(output_dir: Path) -> Path:
    """Copy current .py and .ipynb files to a timestamped backup folder."""
    project_dir = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = output_dir / _EXPORT_PART_PREFIXES["code_snapshot"]
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_root / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)

    patterns = ("*.py", "*.ipynb")
    for pattern in patterns:
        for src in sorted(project_dir.glob(pattern)):
            if not src.is_file():
                continue
            shutil.copy2(src, backup_dir / src.name)

    return backup_dir


def _item_has_cut_pixels(parsed_item: dict, spectrum_key: str = "corrected_spectra_cube") -> bool:
    """Return true when at least one map pixel has been dropped/cut."""
    if "spectrum_keep_mask" in parsed_item:
        keep_mask = np.asarray(parsed_item["spectrum_keep_mask"], dtype=bool)
        if keep_mask.ndim == 2 and keep_mask.size > 0:
            return int(np.count_nonzero(keep_mask)) < int(keep_mask.size)

    cube = np.asarray(parsed_item.get(spectrum_key), dtype=float)
    if cube.ndim != 3:
        return False

    pixel_has_data = np.any(np.isfinite(cube), axis=2)
    if pixel_has_data.size == 0:
        return False
    return int(np.count_nonzero(pixel_has_data)) < int(pixel_has_data.size)


def _save_cut_pixel_map_slice(
    parsed_item: dict,
    output_path: Path,
    *,
    spectrum_key: str = "corrected_spectra_cube",
    color_scale_wavenumber_cm1: float = 562.0,
) -> float:
    """Save a map slice image at the nearest target wavenumber and return the used value."""
    cube = np.asarray(parsed_item[spectrum_key], dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D spectra cube for '{spectrum_key}', got shape {cube.shape}")

    wavenumber = np.asarray(parsed_item["wavenumber_cm1"], dtype=float)
    if wavenumber.ndim != 1 or wavenumber.size == 0:
        raise ValueError("wavenumber_cm1 must be a non-empty 1D array")

    wn_idx = int(np.argmin(np.abs(wavenumber - float(color_scale_wavenumber_cm1))))
    used_wavenumber = float(wavenumber[wn_idx])

    map_image = cube[:, :, wn_idx]
    map_image_display = map_image.T

    fig, map_ax = plt.subplots(figsize=(7.2, 6.0))
    im = map_ax.imshow(map_image_display, origin="upper", cmap="viridis", aspect="equal")

    average_pixel_mask = parsed_item.get("average_pixel_mask")
    if average_pixel_mask is not None:
        mask = np.asarray(average_pixel_mask, dtype=bool)
        if mask.ndim == 2 and mask.shape == map_image.shape and np.any(mask):
            selected_rows, selected_cols = np.nonzero(mask)
            map_ax.scatter(
                selected_rows,
                selected_cols,
                s=45,
                facecolors="none",
                edgecolors="white",
                linewidths=1.2,
                label="Average pixels",
            )
            legend_handles, legend_labels = map_ax.get_legend_handles_labels()
            map_ax.legend(
                legend_handles,
                legend_labels,
                fontsize=8,
                loc="upper right",
                frameon=True,
            )

    map_ax.set_title(
        "Baseline corrected"
        f" | {parsed_item['path'].name}"
        f" | Color scale wavenumber: {used_wavenumber:.2f} cm^-1"
    )
    map_ax.set_xlabel("X index (row)")
    map_ax.set_ylabel("Y index (column)")
    fig.colorbar(im, ax=map_ax, fraction=0.046, pad=0.04, label="Intensity")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return used_wavenumber


def _export_pixel_spectrum_csv(
    parsed_item: dict,
    output_path: Path,
    *,
    row_index: int,
    col_index: int,
    spectrum_key: str = "corrected_spectra_cube",
    previous_spectrum_key: str = "spectra_cube",
) -> None:
    """Export max-signal pixel traces used in the comparison plot as CSV."""
    wavenumber = np.asarray(parsed_item["wavenumber_cm1"], dtype=float)
    current = np.asarray(parsed_item[spectrum_key][row_index, col_index, :], dtype=float)

    export_df = pd.DataFrame(
        {
            "wavenumber_cm1": wavenumber,
            "intensity": current,
        }
    )

    if previous_spectrum_key in parsed_item:
        previous = np.asarray(parsed_item[previous_spectrum_key][row_index, col_index, :], dtype=float)
        export_df["previous_processed_intensity"] = previous

    if "baseline_cube" in parsed_item:
        baseline = np.asarray(parsed_item["baseline_cube"][row_index, col_index, :], dtype=float)
        export_df["baseline_intensity"] = baseline

    export_df.insert(0, "col_index", int(col_index))
    export_df.insert(0, "row_index", int(row_index))
    export_df.to_csv(output_path.with_suffix(".csv"), index=False)


def _export_cut_pixel_map_slice_csv(
    parsed_item: dict,
    output_path: Path,
    *,
    used_wavenumber_cm1: float,
    spectrum_key: str = "corrected_spectra_cube",
) -> None:
    """Export cut-pixel map slice data used by the heatmap figure as CSV."""
    cube = np.asarray(parsed_item[spectrum_key], dtype=float)
    wavenumber = np.asarray(parsed_item["wavenumber_cm1"], dtype=float)
    wn_idx = int(np.argmin(np.abs(wavenumber - float(used_wavenumber_cm1))))
    map_image = cube[:, :, wn_idx]

    row_indices, col_indices = np.indices(map_image.shape)
    export_df = pd.DataFrame(
        {
            "wavenumber_cm1": float(wavenumber[wn_idx]),
            "row_index": row_indices.ravel().astype(int),
            "col_index": col_indices.ravel().astype(int),
            "intensity": map_image.ravel(),
        }
    )
    export_df.to_csv(output_path.with_suffix(".csv"), index=False)


def _save_despiked_baseline_anchor_stack(
    parsed_item: dict,
    output_path: Path,
    *,
    despiked_key: str = "spectra_cube",
    baseline_key: str = "baseline_cube",
    anchor_mask_key: str = "noiseaware_anchor_mask_cube",
) -> dict[str, int | float | str]:
    """Save one map-level stack plot with despiked spectra, baseline, and anchors."""
    if despiked_key not in parsed_item:
        return {
            "status": "missing_despiked_cube",
            "pixels_plotted": 0,
            "anchors_plotted": 0,
            "offset_step": np.nan,
        }
    if baseline_key not in parsed_item:
        return {
            "status": "missing_baseline_cube",
            "pixels_plotted": 0,
            "anchors_plotted": 0,
            "offset_step": np.nan,
        }

    wavenumber = np.asarray(parsed_item["wavenumber_cm1"], dtype=float)
    despiked_cube = np.asarray(parsed_item[despiked_key], dtype=float)
    baseline_cube = np.asarray(parsed_item[baseline_key], dtype=float)

    if despiked_cube.ndim != 3 or baseline_cube.ndim != 3:
        return {
            "status": "invalid_cube_shape",
            "pixels_plotted": 0,
            "anchors_plotted": 0,
            "offset_step": np.nan,
        }

    if despiked_cube.shape != baseline_cube.shape or despiked_cube.shape[2] != wavenumber.size:
        return {
            "status": "shape_mismatch",
            "pixels_plotted": 0,
            "anchors_plotted": 0,
            "offset_step": np.nan,
        }

    keep_mask_raw = parsed_item.get("spectrum_keep_mask")
    if keep_mask_raw is not None:
        keep_mask = np.asarray(keep_mask_raw, dtype=bool)
    else:
        keep_mask = np.any(np.isfinite(despiked_cube), axis=2)

    if keep_mask.shape != despiked_cube.shape[:2]:
        keep_mask = np.any(np.isfinite(despiked_cube), axis=2)

    retained_indices = np.argwhere(keep_mask)
    if retained_indices.size == 0:
        return {
            "status": "no_retained_pixels",
            "pixels_plotted": 0,
            "anchors_plotted": 0,
            "offset_step": np.nan,
        }

    spectra_for_step = [
        np.asarray(despiked_cube[int(row_index), int(col_index), :], dtype=float)
        for row_index, col_index in retained_indices
    ]
    offset_step = _compute_stack_step(
        spectra=spectra_for_step,
        stack_scale=1.35,
        stack_extra_gap=0.1,
    )

    anchor_mask_cube = None
    if anchor_mask_key in parsed_item:
        candidate_anchor_mask_cube = np.asarray(parsed_item[anchor_mask_key], dtype=bool)
        if candidate_anchor_mask_cube.shape == despiked_cube.shape:
            anchor_mask_cube = candidate_anchor_mask_cube

    # Scale figure height with retained pixel count so dense maps remain readable.
    retained_count = int(retained_indices.shape[0])
    fig_height = max(8.0, min(70.0, 2.8 + retained_count * 0.3))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    total_anchors = 0
    y_min = np.inf
    y_max = -np.inf
    for stack_index, (row_index, col_index) in enumerate(retained_indices):
        row_i = int(row_index)
        col_i = int(col_index)
        offset = float(stack_index) * float(offset_step)

        despiked = np.asarray(despiked_cube[row_i, col_i, :], dtype=float)
        baseline = np.asarray(baseline_cube[row_i, col_i, :], dtype=float)
        finite_signal_mask = np.isfinite(wavenumber) & np.isfinite(despiked)
        finite_baseline_mask = np.isfinite(wavenumber) & np.isfinite(baseline)

        ax.plot(
            wavenumber[finite_signal_mask],
            despiked[finite_signal_mask] + offset,
            color="tab:blue",
            linewidth=0.7,
            alpha=0.35,
            label="Despiked spectra" if stack_index == 0 else None,
        )
        ax.plot(
            wavenumber[finite_baseline_mask],
            baseline[finite_baseline_mask] + offset,
            color="tab:red",
            linewidth=0.7,
            linestyle=":",
            alpha=0.35,
            label="Baseline" if stack_index == 0 else None,
        )

        if np.any(finite_signal_mask):
            y_values = despiked[finite_signal_mask] + offset
            y_min = min(y_min, float(np.nanmin(y_values)))
            y_max = max(y_max, float(np.nanmax(y_values)))
        if np.any(finite_baseline_mask):
            y_values = baseline[finite_baseline_mask] + offset
            y_min = min(y_min, float(np.nanmin(y_values)))
            y_max = max(y_max, float(np.nanmax(y_values)))

        if anchor_mask_cube is not None:
            anchor_mask = np.asarray(anchor_mask_cube[row_i, col_i, :], dtype=bool)
            finite_anchor_mask = anchor_mask & finite_signal_mask
            if np.any(finite_anchor_mask):
                total_anchors += int(np.count_nonzero(finite_anchor_mask))
                ax.scatter(
                    wavenumber[finite_anchor_mask],
                    despiked[finite_anchor_mask] + offset,
                    s=8,
                    facecolors="none",
                    edgecolors="tab:green",
                    linewidths=0.5,
                    alpha=0.6,
                    label="Noiseaware anchors" if stack_index == 0 else None,
                )

    file_name = parsed_item["path"].name
    ax.set_title(
        f"{file_name} | retained pixels={retained_count}"
    )
    ax.set_xlabel("Wavenumber (cm^-1)")
    ax.set_ylabel("Intensity + stack offset")
    if np.isfinite(y_min) and np.isfinite(y_max):
        y_span = max(y_max - y_min, 1e-12)
        pad = 0.02 * y_span
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.margins(x=0.01, y=0.0)
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "ok",
        "pixels_plotted": int(retained_indices.shape[0]),
        "anchors_plotted": int(total_anchors),
        "offset_step": float(offset_step),
    }


def _export_despiked_baseline_anchor_stack_csv(
    parsed_item: dict,
    output_path: Path,
    *,
    despiked_key: str = "spectra_cube",
    baseline_key: str = "baseline_cube",
    anchor_mask_key: str = "noiseaware_anchor_mask_cube",
) -> None:
    """Export stack-plot traces as CSV sidecar."""
    if despiked_key not in parsed_item or baseline_key not in parsed_item:
        return

    wavenumber = np.asarray(parsed_item["wavenumber_cm1"], dtype=float)
    despiked_cube = np.asarray(parsed_item[despiked_key], dtype=float)
    baseline_cube = np.asarray(parsed_item[baseline_key], dtype=float)
    if (
        despiked_cube.ndim != 3
        or baseline_cube.ndim != 3
        or despiked_cube.shape != baseline_cube.shape
        or despiked_cube.shape[2] != wavenumber.size
    ):
        return

    keep_mask_raw = parsed_item.get("spectrum_keep_mask")
    if keep_mask_raw is not None:
        keep_mask = np.asarray(keep_mask_raw, dtype=bool)
    else:
        keep_mask = np.any(np.isfinite(despiked_cube), axis=2)
    if keep_mask.shape != despiked_cube.shape[:2]:
        keep_mask = np.any(np.isfinite(despiked_cube), axis=2)

    retained_indices = np.argwhere(keep_mask)
    if retained_indices.size == 0:
        return

    spectra_for_step = [
        np.asarray(despiked_cube[int(row_index), int(col_index), :], dtype=float)
        for row_index, col_index in retained_indices
    ]
    offset_step = _compute_stack_step(
        spectra=spectra_for_step,
        stack_scale=1.0,
        stack_extra_gap=0.0,
    )

    anchor_mask_cube = None
    if anchor_mask_key in parsed_item:
        candidate_anchor_mask_cube = np.asarray(parsed_item[anchor_mask_key], dtype=bool)
        if candidate_anchor_mask_cube.shape == despiked_cube.shape:
            anchor_mask_cube = candidate_anchor_mask_cube

    rows: list[pd.DataFrame] = []
    for stack_index, (row_index, col_index) in enumerate(retained_indices):
        row_i = int(row_index)
        col_i = int(col_index)
        offset = float(stack_index) * float(offset_step)
        despiked = np.asarray(despiked_cube[row_i, col_i, :], dtype=float)
        baseline = np.asarray(baseline_cube[row_i, col_i, :], dtype=float)
        finite_mask = np.isfinite(wavenumber)
        anchor_mask = np.zeros_like(finite_mask, dtype=bool)
        if anchor_mask_cube is not None:
            anchor_mask = np.asarray(anchor_mask_cube[row_i, col_i, :], dtype=bool)

        rows.append(
            pd.DataFrame(
                {
                    "file": str(parsed_item["path"].name),
                    "row_index": row_i,
                    "col_index": col_i,
                    "stack_index": int(stack_index),
                    "stack_offset": offset,
                    "wavenumber_cm1": wavenumber[finite_mask],
                    "despiked_intensity": despiked[finite_mask],
                    "baseline_intensity": baseline[finite_mask],
                    "despiked_stacked_intensity": despiked[finite_mask] + offset,
                    "baseline_stacked_intensity": baseline[finite_mask] + offset,
                    "is_anchor": anchor_mask[finite_mask],
                }
            )
        )

    if not rows:
        return

    pd.concat(rows, ignore_index=True).to_csv(output_path.with_suffix(".csv"), index=False)


def export_stage6_outputs(
    data_dir: Path,
    avg_map_spectra: pd.DataFrame,
    peak_ratio_df: pd.DataFrame,
    output_folder_name: str | None = None,
    sample_name: str | None = None,
    corrected_parsed_files: list[dict] | None = None,
    cut_pixel_map_wavenumber_cm1: float = 562.0,
) -> Path:
    """Export map-analysis tables and snapshot current analysis code under DATA_DIR."""
    sample_stem = _sanitize_export_stem(sample_name or infer_sample_name(avg_map_spectra))
    if output_folder_name is None:
        output_folder_name = f"{sample_stem}_map_analysis_exports"

    output_dir = data_dir / output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra_root = _prepare_export_subdir(output_dir, "spectra")
    tables_root = _prepare_export_subdir(output_dir, "tables")

    # Clean up legacy aggregated exports from older pipeline versions.
    for legacy_name in ("average_spectra.csv", "normalized_spectra.csv"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    avg_dir = _prepare_export_subdir(output_dir, "spectra", "01_avg")
    norm_dir = _prepare_export_subdir(output_dir, "spectra", "02_norm")

    _export_spectra_per_file(
        avg_map_spectra=avg_map_spectra,
        spectrum_col="mean_spectrum",
        output_dir=avg_dir,
        file_suffix="average",
    )

    _export_spectra_per_file(
        avg_map_spectra=avg_map_spectra,
        spectrum_col="mean_spectrum_norm",
        output_dir=norm_dir,
        file_suffix="normalized",
    )

    if corrected_parsed_files:
        max_dir = _prepare_export_subdir(output_dir, "plots", "max_signal")
        cut_map_dir = _prepare_export_subdir(output_dir, "plots", "cutpixel_map")
        despiked_baseline_anchor_stack_dir = _prepare_export_subdir(
            output_dir,
            "plots",
            "despiked_baseline_anchor_stack",
        )
        requested_wn_stem = str(float(cut_pixel_map_wavenumber_cm1)).replace(".", "p")

        max_manifest: list[dict] = []
        cut_map_manifest: list[dict] = []
        despiked_baseline_anchor_stack_manifest: list[dict] = []
        for index, item in enumerate(corrected_parsed_files, start=1):
            file_name = item["path"].name
            safe_stem = _sanitize_export_stem(file_name)
            prefixed_stem = _prefix_indexed_stem(safe_stem, index=index)
            row_index, col_index = select_max_intensity_pixel(
                parsed_item=item,
                spectrum_key="corrected_spectra_cube",
            )
            output_path = max_dir / f"{prefixed_stem}.png"
            row_index, col_index = save_pixel_spectrum_comparison(
                parsed_item=item,
                output_path=output_path,
                spectrum_key="corrected_spectra_cube",
                stage_label="Baseline corrected",
                figure_title=(
                    "Highest maximum signal after baseline correction "
                    f"| {file_name} | Pixel ({row_index}, {col_index})"
                ),
                highlight_wavenumber=None,
                show_previous_overlay=True,
                show_baseline=True,
                show_noiseaware_anchors=True,
                previous_label="Previous processed spectrum",
            )
            max_manifest.append(
                {
                    "file": file_name,
                    "row_index": int(row_index),
                    "col_index": int(col_index),
                    "highest_max_signal_after_baseline": float(
                        np.nanmax(np.asarray(item["corrected_spectra_cube"], dtype=float))
                    ),
                    "image_path": str(output_path),
                }
            )
            _export_pixel_spectrum_csv(
                parsed_item=item,
                output_path=output_path,
                row_index=int(row_index),
                col_index=int(col_index),
                spectrum_key="corrected_spectra_cube",
                previous_spectrum_key="spectra_cube",
            )

            if _item_has_cut_pixels(item, spectrum_key="corrected_spectra_cube"):
                cut_map_output_path = cut_map_dir / f"{prefixed_stem}_{requested_wn_stem}cm-1.png"
                used_wavenumber = _save_cut_pixel_map_slice(
                    parsed_item=item,
                    output_path=cut_map_output_path,
                    spectrum_key="corrected_spectra_cube",
                    color_scale_wavenumber_cm1=cut_pixel_map_wavenumber_cm1,
                )
                _export_cut_pixel_map_slice_csv(
                    parsed_item=item,
                    output_path=cut_map_output_path,
                    used_wavenumber_cm1=used_wavenumber,
                    spectrum_key="corrected_spectra_cube",
                )
                keep_mask = np.asarray(item.get("spectrum_keep_mask"), dtype=bool)
                pixels_available = int(np.count_nonzero(keep_mask)) if keep_mask.ndim == 2 else np.nan
                pixels_total = int(keep_mask.size) if keep_mask.ndim == 2 else np.nan
                cut_map_manifest.append(
                    {
                        "file": file_name,
                        "color_scale_wavenumber_cm1": used_wavenumber,
                        "requested_wavenumber_cm1": float(cut_pixel_map_wavenumber_cm1),
                        "pixels_available": pixels_available,
                        "pixels_total": pixels_total,
                        "image_path": str(cut_map_output_path),
                    }
                )

            stack_output_path = despiked_baseline_anchor_stack_dir / f"{prefixed_stem}.png"
            stack_summary = _save_despiked_baseline_anchor_stack(
                parsed_item=item,
                output_path=stack_output_path,
                despiked_key="spectra_cube",
                baseline_key="baseline_cube",
                anchor_mask_key="noiseaware_anchor_mask_cube",
            )
            _export_despiked_baseline_anchor_stack_csv(
                parsed_item=item,
                output_path=stack_output_path,
                despiked_key="spectra_cube",
                baseline_key="baseline_cube",
                anchor_mask_key="noiseaware_anchor_mask_cube",
            )
            despiked_baseline_anchor_stack_manifest.append(
                {
                    "file": file_name,
                    "image_path": str(stack_output_path),
                    **stack_summary,
                }
            )

        pd.DataFrame(max_manifest).to_csv(
            max_dir / "manifest.csv",
            index=False,
        )
        if cut_map_manifest:
            cut_map_manifest_df = pd.DataFrame(cut_map_manifest)
        else:
            cut_map_manifest_df = pd.DataFrame(
                [
                    {
                        "status": "no_cut_pixel_maps",
                        "message": "No map was cut pixel.",
                    }
                ]
            )
        cut_map_manifest_df.to_csv(
            cut_map_dir / "manifest.csv",
            index=False,
        )
        pd.DataFrame(despiked_baseline_anchor_stack_manifest).to_csv(
            despiked_baseline_anchor_stack_dir / "manifest.csv",
            index=False,
        )

    peak_ratio_df.to_csv(tables_root / "peak_ratio.csv", index=False)

    # Snapshot current notebook + Python sources used to generate this export.
    _backup_code_snapshot(output_dir)

    return output_dir
