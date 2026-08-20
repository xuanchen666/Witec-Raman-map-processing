"""Map-level analysis: summaries, pixel-mask averaging, and peak-ratio tables."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .metadata import extract_date, extract_group, extract_subgroup

ParsedMap = Mapping[str, Any]
ParsedCollection = Sequence[ParsedMap]


def summarize_parsed_collection(parsed_collection: ParsedCollection) -> pd.DataFrame:
    """Build a compact DataFrame with map dimensions and scan metadata."""
    if not parsed_collection:
        return pd.DataFrame(
            columns=[
                "file",
                "points",
                "spectra",
                "cube_x",
                "cube_y",
                "size_x",
                "size_y",
                "scan_width",
                "scan_height",
            ]
        )

    return pd.DataFrame(
        {
            "file": [item["path"].name for item in parsed_collection],
            "points": [len(item["wavenumber_cm1"]) for item in parsed_collection],
            "spectra": [item["spectra_cube"].shape[0] * item["spectra_cube"].shape[1] for item in parsed_collection],
            "cube_x": [item["spectra_cube"].shape[0] for item in parsed_collection],
            "cube_y": [item["spectra_cube"].shape[1] for item in parsed_collection],
            "size_x": [int(item["header"].get("SizeX", 0)) for item in parsed_collection],
            "size_y": [int(item["header"].get("SizeY", 0)) for item in parsed_collection],
            "scan_width": [float(item["header"].get("ScanWidth", np.nan)) for item in parsed_collection],
            "scan_height": [float(item["header"].get("ScanHeight", np.nan)) for item in parsed_collection],
        }
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


def rank_despike_aggressiveness(
    original_collection: ParsedCollection,
    despiked_collection: ParsedCollection,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank spectra by despiking strength using the max absolute pointwise change."""
    if len(original_collection) != len(despiked_collection):
        raise ValueError("original_collection and despiked_collection must have the same length")

    records = []
    for map_index, (original_map, despiked_map) in enumerate(zip(original_collection, despiked_collection)):
        original_cube = original_map["spectra_cube"]
        despiked_cube = despiked_map["spectra_cube"]

        if original_cube.shape != despiked_cube.shape:
            raise ValueError(
                f"Shape mismatch at map index {map_index}: "
                f"{original_cube.shape} vs {despiked_cube.shape}"
            )

        abs_delta = np.abs(despiked_cube - original_cube)
        max_abs_change = abs_delta.max(axis=2)
        mean_abs_change = abs_delta.mean(axis=2)

        for row_index in range(max_abs_change.shape[0]):
            for col_index in range(max_abs_change.shape[1]):
                records.append(
                    {
                        "map_index": map_index,
                        "file": original_map["path"].name,
                        "row_index": row_index,
                        "col_index": col_index,
                        "max_abs_change": float(max_abs_change[row_index, col_index]),
                        "mean_abs_change": float(mean_abs_change[row_index, col_index]),
                    }
                )

    ranked = pd.DataFrame.from_records(records)
    if ranked.empty:
        return ranked

    ranked = ranked.sort_values(
        by=["max_abs_change", "mean_abs_change"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return ranked.head(top_n).copy()

