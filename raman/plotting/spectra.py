"""Spectrum-level plotting: stacked traces, normalized overlaps, peak-ratio trends."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

from ..config import (
    NORMALIZATION_METHOD,
    NORMALIZATION_PEAK_CENTER_CM1,
    NORMALIZATION_PEAK_TOLERANCE_CM1,
    PEAK_RATIO_WAVENUMBER_RANGES,
    PLOT_WAVENUMBER_RANGES,
)
from ..core.analysis import build_average_map_spectra, build_peak_ratio_table
from ..core.metadata import _extract_sample_code, infer_sample_name
from ..export.csv_export import (
    _export_grouped_spectra_plot_csv,
    _export_overlap_plot_csv,
    _export_peak_ratio_plot_csv,
)
from ..export.paths import (
    _build_wavenumber_range_stem,
    _coerce_optional_wavenumber_bound,
    _prefix_indexed_stem,
    _prepare_export_subdir,
    _sanitize_export_stem,
)
from .maps import _compute_stack_step


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


def _compute_grouped_spectra_data(
    subset: pd.DataFrame,
    value_col: str,
    stack_scale: float,
    stack_extra_gap: float,
    wavenumber_min: float | None,
    wavenumber_max: float | None,
) -> dict[str, Any]:
    """Compute sliced/stacked traces and offset step for one group panel (no plotting)."""
    windows = [
        _slice_spectrum_to_wavenumber_range(
            wavenumber_cm1=row["wavenumber_cm1"],
            intensity=row[value_col],
            wavenumber_min=wavenumber_min,
            wavenumber_max=wavenumber_max,
        )
        for _, row in subset.iterrows()
    ]
    offset_step = _compute_stack_step(
        spectra=[window_y for _, window_y in windows],
        stack_scale=stack_scale,
        stack_extra_gap=stack_extra_gap,
    )

    labels = []
    stacked_traces = []
    for stack_index, ((_, row), (window_x, window_y)) in enumerate(zip(subset.iterrows(), windows)):
        label_date = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "Unknown date"
        stacked_traces.append((window_x, window_y + stack_index * offset_step))
        labels.append(f"{label_date} | {row['file']}")

    return {
        "offset_step": offset_step,
        "stacked_traces": stacked_traces,
        "labels": labels,
    }


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
    import matplotlib.pyplot as plt

    if group_col not in df.columns:
        raise KeyError(f"DataFrame is missing required grouping column '{group_col}'")

    available_groups = set(df[group_col].dropna().unique())
    panels = [group_name for group_name in panel_order if group_name in available_groups]
    if not panels:
        return None

    if output_path is not None and len(panels) > 1 and output_path.exists():
        output_path.unlink()

    def _draw_group(axis: plt.Axes, group_name: str, group_data: dict[str, Any]) -> None:
        for (window_x, stacked_y), label in zip(group_data["stacked_traces"], group_data["labels"]):
            axis.plot(window_x, stacked_y, linewidth=1.6, label=label)

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
        subset = df[df[group_col] == group_name].sort_values(["date", "file"])
        group_stack_scale = _resolve_group_value(stack_scale, group_name, fallback=1.0)
        group_stack_extra_gap = _resolve_group_value(stack_extra_gap, group_name, fallback=0.0)
        group_data = _compute_grouped_spectra_data(
            subset=subset,
            value_col=value_col,
            stack_scale=group_stack_scale,
            stack_extra_gap=group_stack_extra_gap,
            wavenumber_min=wavenumber_min,
            wavenumber_max=wavenumber_max,
        )

        fig, axis = plt.subplots(1, 1, figsize=(9, 5), sharex=True)
        _draw_group(axis, group_name, group_data)

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
                offset_step=group_data["offset_step"],
            )
        plt.show()
        last_fig = fig

    return last_fig


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
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

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

