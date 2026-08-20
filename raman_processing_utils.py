"""Utility functions for Raman notebook processing stages.

These helpers keep the notebook concise while preserving the same behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any, cast

import numpy as np
import pandas as pd
from pybaselines import Baseline

from raman_noiseaware_baseline import auto_baseline_noiseaware

# Shared aliases used throughout this module to make function signatures easier to read.
ParsedMap = Mapping[str, Any]
ParsedMapMutable = dict[str, Any]
ParsedCollection = Sequence[ParsedMap]
ParsedCollectionMutable = list[ParsedMapMutable]
StageCollections = Mapping[str, Sequence[ParsedMap]]
StageSpectrumKeys = Mapping[str, str]


def _extract_map_group(file_name: str) -> str:
    """Classify map group from filename tokens for scope-based processing."""
    tokens = [token.strip().lower() for token in file_name.split("_") if token.strip()]
    for token in tokens:
        if token == "au":
            return "Au"
        if token == "ro":
            return "RO"
        if token == "hbn":
            return "hBN"
    return "Other"


def _extract_map_subgroup(file_name: str) -> str:
    """Extract subgroup label (e.g., hBN_1) while preserving major group labels."""
    tokens = [token for token in Path(file_name).stem.split("_") if token]
    if not tokens:
        return "Other"

    # Drop trailing date token (YYYYMMDD) and laser-power token (e.g., 10mW, 0.5W).
    if re.fullmatch(r"\d{8}", tokens[-1]):
        tokens.pop()
    if tokens and re.fullmatch(r"\d+(?:\.\d+)?(?:mw|w)", tokens[-1], flags=re.IGNORECASE):
        tokens.pop()

    normalized_tokens = [token.strip().lower() for token in tokens if token.strip()]
    for index, token in enumerate(normalized_tokens):
        if token == "au":
            return "Au"
        if token == "ro":
            return "RO"
        if token == "hbn":
            suffix_tokens = tokens[index + 1 :]
            if suffix_tokens:
                return "hBN_" + "_".join(suffix_tokens)
            return "hBN"

    return "Other"


def _normalize_scope_targets(apply_to_groups: Sequence[str] | None) -> set[str]:
    """Normalize configured scope selectors used by Stage 2 filters."""
    requested = (
        [str(group).strip() for group in apply_to_groups]
        if apply_to_groups is not None
        else ["all"]
    )
    normalized = {group.lower() for group in requested if group}
    if not normalized:
        normalized = {"all"}

    invalid_targets = [
        token
        for token in sorted(normalized)
        if token != "all"
        and token not in {"au", "ro", "hbn"}
        and not re.fullmatch(r"(?:au|ro|hbn)_[a-z0-9]+(?:_[a-z0-9]+)*", token)
    ]
    if invalid_targets:
        raise ValueError(
            "apply_to_groups can only contain: all, Au, RO, hBN, or subgroup labels "
            "like hBN_1; got invalid values "
            f"{invalid_targets}"
        )

    return normalized


def _scope_applies_to_map(
    normalized_scope_targets: set[str],
    map_group: str,
    map_subgroup: str,
) -> bool:
    """Return whether a map should be processed under a scope selector set."""
    if "all" in normalized_scope_targets:
        return True

    group_token = map_group.strip().lower()
    subgroup_token = map_subgroup.strip().lower()
    return group_token in normalized_scope_targets or subgroup_token in normalized_scope_targets


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


def filter_low_wavenumber_region(
    parsed_collection: ParsedCollection,
    min_wavenumber_cm1: float,
) -> ParsedCollectionMutable:
    """Trim each map to only include spectral points with wavenumber >= threshold."""
    filtered_collection = []

    for parsed in parsed_collection:
        wavenumber_mask = parsed["wavenumber_cm1"] >= min_wavenumber_cm1
        tidy_mask = parsed["tidy"]["wavenumber_cm1"] >= min_wavenumber_cm1

        filtered_collection.append(
            {
                **parsed,
                "wavenumber_cm1": parsed["wavenumber_cm1"][wavenumber_mask],
                "spectra_cube": parsed["spectra_cube"][..., wavenumber_mask],
                "tidy": parsed["tidy"].loc[tidy_mask].reset_index(drop=True),
            }
        )

    return filtered_collection


def filter_spectra_by_wavenumber_region_mean(
    parsed_collection: ParsedCollection,
    wavenumber_region_cm1: tuple[float, float],
    min_mean_intensity: float,
    apply_to_groups: Sequence[str] | None = None,
) -> tuple[ParsedCollectionMutable, pd.DataFrame]:
    """Drop spectra whose mean intensity in a target wavenumber window is too low.

    Spectra are dropped by setting their full trace to NaN while keeping map geometry,
    so downstream map-based tools (including interactive explorer) keep working.
    """
    region_start, region_end = wavenumber_region_cm1
    lower = min(float(region_start), float(region_end))
    upper = max(float(region_start), float(region_end))

    normalized_scope_targets = _normalize_scope_targets(apply_to_groups)

    filtered_collection: ParsedCollectionMutable = []
    report_records: list[dict[str, float | int | str]] = []

    for parsed in parsed_collection:
        map_group = _extract_map_group(parsed["path"].name)
        map_subgroup = _extract_map_subgroup(parsed["path"].name)
        gate_applied = _scope_applies_to_map(
            normalized_scope_targets=normalized_scope_targets,
            map_group=map_group,
            map_subgroup=map_subgroup,
        )

        wavenumber = np.asarray(parsed["wavenumber_cm1"], dtype=float)
        cube = np.asarray(parsed["spectra_cube"], dtype=float)

        if gate_applied:
            wn_mask = (wavenumber >= lower) & (wavenumber <= upper)
            if not np.any(wn_mask):
                raise ValueError(
                    f"No wavenumber points found in [{lower}, {upper}] cm^-1 for {parsed['path'].name}"
                )

            with np.errstate(invalid="ignore"):
                region_mean = np.nanmean(cube[..., wn_mask], axis=2)

            keep_mask = np.isfinite(region_mean) & (region_mean >= float(min_mean_intensity))

            dropped_cube = cube.copy()
            dropped_cube[~keep_mask, :] = np.nan

            tidy = parsed["tidy"]
            keep_coords = pd.DataFrame(np.argwhere(keep_mask), columns=["x_index", "y_index"])
            if keep_coords.empty:
                tidy_filtered = tidy.iloc[0:0].copy()
            else:
                tidy_filtered = (
                    tidy.merge(keep_coords, on=["x_index", "y_index"], how="inner")
                    .sort_values(["spectrum_index", "wavenumber_cm1"])
                    .reset_index(drop=True)
                )
        else:
            keep_mask = np.ones(cube.shape[:2], dtype=bool)
            dropped_cube = cube.copy()
            tidy_filtered = parsed["tidy"].copy()

        filtered_collection.append(
            {
                **parsed,
                "spectra_cube": dropped_cube,
                "tidy": tidy_filtered,
                "spectrum_keep_mask": keep_mask,
                "spectrum_gate_config": {
                    "enabled": True,
                    "apply_to_groups": sorted(normalized_scope_targets),
                    "gate_applied": gate_applied,
                    "map_group": map_group,
                    "map_subgroup": map_subgroup,
                    "wavenumber_region_cm1": (lower, upper),
                    "min_mean_intensity": float(min_mean_intensity),
                },
            }
        )

        total_spectra = int(keep_mask.size)
        kept_spectra = int(np.count_nonzero(keep_mask))
        dropped_spectra = total_spectra - kept_spectra
        report_records.append(
            {
                "file": parsed["path"].name,
                "group": map_group,
                "subgroup": map_subgroup,
                "gate_applied": bool(gate_applied),
                "window_start_cm1": lower,
                "window_end_cm1": upper,
                "threshold": float(min_mean_intensity),
                "spectra_total": total_spectra,
                "spectra_kept": kept_spectra,
                "spectra_dropped": dropped_spectra,
                "drop_fraction": (dropped_spectra / total_spectra) if total_spectra else np.nan,
            }
        )

    report_df = pd.DataFrame.from_records(report_records)
    return filtered_collection, report_df


def _resolve_existing_keep_mask(parsed: ParsedMap, spectrum_key: str = "spectra_cube") -> np.ndarray:
    """Return the current 2D keep mask, or infer it from finite spectra."""
    if "spectrum_keep_mask" in parsed:
        keep_mask = np.asarray(parsed["spectrum_keep_mask"], dtype=bool)
        if keep_mask.ndim != 2:
            raise ValueError(
                f"Expected a 2D spectrum_keep_mask, got shape {keep_mask.shape}"
            )
        return keep_mask

    cube = np.asarray(parsed[spectrum_key], dtype=float)
    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3D spectra cube for '{spectrum_key}', got shape {cube.shape}"
        )

    with np.errstate(invalid="ignore"):
        return np.isfinite(np.nanmean(cube, axis=2))


def filter_spectra_by_border_pixels(
    parsed_collection: ParsedCollection,
    border_width: int = 1,
    apply_to_groups: Sequence[str] | None = None,
    spectrum_key: str = "spectra_cube",
) -> tuple[ParsedCollectionMutable, pd.DataFrame]:
    """Drop an outer ring of pixels for selected groups while preserving map geometry."""
    border_width = int(border_width)
    if border_width < 0:
        raise ValueError("border_width must be greater than or equal to 0")

    normalized_scope_targets = _normalize_scope_targets(apply_to_groups)

    filtered_collection: ParsedCollectionMutable = []
    report_records: list[dict[str, Any]] = []

    for parsed in parsed_collection:
        map_group = _extract_map_group(parsed["path"].name)
        map_subgroup = _extract_map_subgroup(parsed["path"].name)
        border_applied = _scope_applies_to_map(
            normalized_scope_targets=normalized_scope_targets,
            map_group=map_group,
            map_subgroup=map_subgroup,
        )

        cube = np.asarray(parsed[spectrum_key], dtype=float)
        if cube.ndim != 3:
            raise ValueError(
                f"Expected a 3D spectra cube for '{spectrum_key}', got shape {cube.shape}"
            )

        keep_mask = _resolve_existing_keep_mask(parsed, spectrum_key=spectrum_key).copy()
        if border_applied and border_width > 0:
            rows, cols = cube.shape[:2]
            border_mask = np.ones((rows, cols), dtype=bool)
            border_mask[:border_width, :] = False
            border_mask[-border_width:, :] = False
            border_mask[:, :border_width] = False
            border_mask[:, -border_width:] = False
            keep_mask &= border_mask

        filtered_cube = cube.copy()
        filtered_cube[~keep_mask, :] = np.nan

        tidy = parsed["tidy"]
        keep_coords = pd.DataFrame(np.argwhere(keep_mask), columns=["x_index", "y_index"])
        if keep_coords.empty:
            tidy_filtered = tidy.iloc[0:0].copy()
        else:
            tidy_filtered = (
                tidy.merge(keep_coords, on=["x_index", "y_index"], how="inner")
                .sort_values(["spectrum_index", "wavenumber_cm1"])
                .reset_index(drop=True)
            )

        filtered_collection.append(
            {
                **parsed,
                spectrum_key: filtered_cube,
                "tidy": tidy_filtered,
                "spectrum_keep_mask": keep_mask,
                "border_filter_config": {
                    "enabled": True,
                    "apply_to_groups": sorted(normalized_scope_targets),
                    "border_applied": bool(border_applied),
                    "map_group": map_group,
                    "map_subgroup": map_subgroup,
                    "border_width": int(border_width),
                },
            }
        )

        total_spectra = int(keep_mask.size)
        kept_spectra = int(np.count_nonzero(keep_mask))
        dropped_spectra = total_spectra - kept_spectra
        report_records.append(
            {
                "file": parsed["path"].name,
                "group": map_group,
                "subgroup": map_subgroup,
                "border_applied": bool(border_applied),
                "border_width": int(border_width),
                "spectra_total": total_spectra,
                "spectra_kept": kept_spectra,
                "spectra_dropped": dropped_spectra,
                "drop_fraction": (dropped_spectra / total_spectra) if total_spectra else np.nan,
            }
        )

    report_df = pd.DataFrame.from_records(report_records)
    return filtered_collection, report_df


def filter_spectra_by_max_intensity(
    parsed_collection: ParsedCollection,
    max_intensity: float,
    apply_to_groups: Sequence[str] | None = None,
    spectrum_key: str = "spectra_cube",
) -> tuple[ParsedCollectionMutable, pd.DataFrame]:
    """Drop spectra whose maximum intensity exceeds the configured threshold."""
    normalized_scope_targets = _normalize_scope_targets(apply_to_groups)

    filtered_collection: ParsedCollectionMutable = []
    report_records: list[dict[str, Any]] = []

    for parsed in parsed_collection:
        map_group = _extract_map_group(parsed["path"].name)
        map_subgroup = _extract_map_subgroup(parsed["path"].name)
        gate_applied = _scope_applies_to_map(
            normalized_scope_targets=normalized_scope_targets,
            map_group=map_group,
            map_subgroup=map_subgroup,
        )

        cube = np.asarray(parsed[spectrum_key], dtype=float)
        keep_mask = _resolve_existing_keep_mask(parsed, spectrum_key=spectrum_key).copy()

        if gate_applied:
            finite_cube = np.where(np.isfinite(cube), cube, -np.inf)
            over_threshold_mask = np.max(finite_cube, axis=2) > float(max_intensity)

            over_threshold_mask &= keep_mask
            keep_mask &= ~over_threshold_mask

        filtered_cube = cube.copy()
        filtered_cube[~keep_mask, :] = np.nan

        tidy = parsed["tidy"]
        keep_coords = pd.DataFrame(np.argwhere(keep_mask), columns=["x_index", "y_index"])
        if keep_coords.empty:
            tidy_filtered = tidy.iloc[0:0].copy()
        else:
            tidy_filtered = (
                tidy.merge(keep_coords, on=["x_index", "y_index"], how="inner")
                .sort_values(["spectrum_index", "wavenumber_cm1"])
                .reset_index(drop=True)
            )

        filtered_collection.append(
            {
                **parsed,
                spectrum_key: filtered_cube,
                "tidy": tidy_filtered,
                "spectrum_keep_mask": keep_mask,
                "max_intensity_gate_config": {
                    "max_intensity": float(max_intensity),
                    "apply_to_groups": sorted(normalized_scope_targets),
                    "gate_applied": bool(gate_applied),
                    "map_group": map_group,
                    "map_subgroup": map_subgroup,
                },
            }
        )

        total_spectra = int(keep_mask.size)
        kept_spectra = int(np.count_nonzero(keep_mask))
        dropped_spectra = total_spectra - kept_spectra
        report_records.append(
            {
                "file": parsed["path"].name,
                "group": map_group,
                "subgroup": map_subgroup,
                "gate_applied": bool(gate_applied),
                "threshold": float(max_intensity),
                "spectra_total": total_spectra,
                "spectra_kept": kept_spectra,
                "spectra_dropped": dropped_spectra,
                "drop_fraction": (dropped_spectra / total_spectra) if total_spectra else np.nan,
            }
        )

    report_df = pd.DataFrame.from_records(report_records)
    return filtered_collection, report_df


def despike_parsed_collection(
    parsed_collection: ParsedCollection,
    neigh: int = 4,
    threshold: int = 3,
) -> ParsedCollectionMutable:
    """Apply rampy.despiking to every spectrum of every parsed map."""
    import rampy as rp

    despiked_collection = []

    for parsed in parsed_collection:
        wavenumber = parsed["wavenumber_cm1"]
        spectra_cube = parsed["spectra_cube"]
        despiked_cube = np.empty_like(spectra_cube, dtype=float)

        # Iterate over every pixel spectrum in the current map cube.
        for row_index in range(spectra_cube.shape[0]):
            for col_index in range(spectra_cube.shape[1]):
                spectrum = spectra_cube[row_index, col_index, :]
                if not np.isfinite(spectrum).all():
                    despiked_cube[row_index, col_index, :] = spectrum
                    continue
                despiked_cube[row_index, col_index, :] = rp.despiking(
                    x=wavenumber,
                    y=spectrum,
                    neigh=neigh,
                    threshold=int(threshold),
                )

        despiked_collection.append(
            {
                **parsed,
                "spectra_cube": despiked_cube,
            }
        )

    return despiked_collection


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


def _build_poly_weights(
    wavenumber: np.ndarray,
    mask_regions: Sequence[tuple[float, float]] | None,
) -> np.ndarray | None:
    """Build per-point weights for poly baseline using user-provided wavenumber regions."""
    if not mask_regions:
        return None

    weights = np.zeros_like(wavenumber, dtype=float)
    for region_start, region_end in mask_regions:
        lower = min(float(region_start), float(region_end))
        upper = max(float(region_start), float(region_end))
        weights[(wavenumber >= lower) & (wavenumber <= upper)] = 1.0

    if not np.any(weights > 0):
        raise ValueError("poly_mask_regions did not select any wavenumber points")

    return weights


def _build_noiseaware_anchor_mask(
    wavenumber: np.ndarray,
    anchor_x_values: Sequence[float] | None,
) -> np.ndarray:
    """Map stored noiseaware anchor x positions back to the spectrum grid."""
    mask = np.zeros_like(wavenumber, dtype=bool)
    if wavenumber.ndim != 1 or not anchor_x_values:
        return mask

    for anchor_x in np.asarray(anchor_x_values, dtype=float):
        if not np.isfinite(anchor_x):
            continue
        insert_index = int(np.searchsorted(wavenumber, anchor_x))
        candidate_indices = [
            candidate_index
            for candidate_index in (insert_index, insert_index - 1)
            if 0 <= candidate_index < wavenumber.size
        ]
        if not candidate_indices:
            continue

        nearest_index = min(
            candidate_indices,
            key=lambda candidate_index: abs(float(wavenumber[candidate_index]) - float(anchor_x)),
        )
        mask[nearest_index] = True

    return mask


def _build_noiseaware_anchor_value_grid(shape: tuple[int, int]) -> np.ndarray:
    """Create a per-pixel object grid for raw noiseaware anchor values."""
    grid = np.empty(shape, dtype=object)
    grid.fill(None)
    return grid


def _normalize_noiseaware_anchor_values(values: Sequence[float] | None) -> tuple[float, ...]:
    """Store the exact Stage 5 pre-median anchor values as finite float tuples."""
    if values is None:
        return ()

    normalized: list[float] = []
    for value in values:
        numeric_value = float(value)
        if np.isfinite(numeric_value):
            normalized.append(numeric_value)
    return tuple(normalized)


def _get_noiseaware_anchor_pairs(
    parsed_item: ParsedMap,
    row_index: int,
    col_index: int,
) -> list[tuple[float, float]]:
    """Return persisted Stage 5 pre-median anchor x/y pairs for one pixel."""
    anchor_x_grid = parsed_item.get("noiseaware_anchor_x_values_grid")
    anchor_y_grid = parsed_item.get("noiseaware_anchor_y_values_grid")
    if anchor_x_grid is None or anchor_y_grid is None:
        return []

    anchor_x_grid = np.asarray(anchor_x_grid, dtype=object)
    anchor_y_grid = np.asarray(anchor_y_grid, dtype=object)
    if anchor_x_grid.ndim != 2 or anchor_y_grid.ndim != 2:
        return []
    if row_index >= anchor_x_grid.shape[0] or col_index >= anchor_x_grid.shape[1]:
        return []
    if row_index >= anchor_y_grid.shape[0] or col_index >= anchor_y_grid.shape[1]:
        return []

    anchor_x_values = _normalize_noiseaware_anchor_values(
        cast(Sequence[float] | None, anchor_x_grid[row_index, col_index]),
    )
    anchor_y_values = _normalize_noiseaware_anchor_values(
        cast(Sequence[float] | None, anchor_y_grid[row_index, col_index]),
    )
    return list(zip(anchor_x_values, anchor_y_values))


def apply_baseline_correction(
    parsed_collection: ParsedCollection,
    fixed_half_window: int | None,
    window_kwargs: dict,
    baseline_method: str = "mor",
    airpls_kwargs: dict | None = None,
    poly_kwargs: dict | None = None,
    poly_mask_regions: Sequence[tuple[float, float]] | None = None,
    rolling_ball_kwargs: dict | None = None,
    noiseaware_kwargs: dict | None = None,
    noiseaware_peak_regions: Sequence[tuple[float, float]] | None = None,
) -> ParsedCollectionMutable:
    """Fit and subtract a baseline for every spectrum of every map."""
    corrected_collection = []
    normalized_method = baseline_method.lower()

    valid_methods = {"mor", "airpls", "poly", "rolling_ball", "noiseaware"}
    if normalized_method not in valid_methods:
        raise ValueError(f"baseline_method must be one of {sorted(valid_methods)}")

    airpls_kwargs = {} if airpls_kwargs is None else airpls_kwargs
    poly_kwargs = {} if poly_kwargs is None else poly_kwargs
    rolling_ball_kwargs = {} if rolling_ball_kwargs is None else rolling_ball_kwargs
    noiseaware_kwargs = {} if noiseaware_kwargs is None else noiseaware_kwargs
    normalized_peak_regions = (
        [(float(lower), float(upper)) for lower, upper in noiseaware_peak_regions]
        if noiseaware_peak_regions
        else []
    )

    for parsed in parsed_collection:
        wavenumber = parsed["wavenumber_cm1"]
        spectra_cube = parsed["spectra_cube"]
        baseline_fitter = Baseline(x_data=wavenumber)
        poly_weights = _build_poly_weights(
            wavenumber=wavenumber,
            mask_regions=poly_mask_regions,
        )

        baseline_cube = np.empty_like(spectra_cube, dtype=float)
        corrected_cube = np.empty_like(spectra_cube, dtype=float)
        stat_cube = np.full(spectra_cube.shape[:2], np.nan, dtype=float)
        noiseaware_anchor_mask_cube = (
            np.zeros_like(spectra_cube, dtype=bool)
            if normalized_method == "noiseaware"
            else None
        )
        noiseaware_anchor_x_values_grid = (
            _build_noiseaware_anchor_value_grid(spectra_cube.shape[:2])
            if normalized_method == "noiseaware"
            else None
        )
        noiseaware_anchor_y_values_grid = (
            _build_noiseaware_anchor_value_grid(spectra_cube.shape[:2])
            if normalized_method == "noiseaware"
            else None
        )
        if normalized_method == "mor":
            stat_label = "half_window"
        elif normalized_method == "airpls":
            stat_label = "tol_history_len"
        elif normalized_method == "poly":
            stat_label = "mask_points"
        elif normalized_method == "rolling_ball":
            stat_label = "rolling_ball_half_window"
        else:
            stat_label = "anchors_used"

        # Fit baseline per pixel so local differences are preserved.
        for row_index in range(spectra_cube.shape[0]):
            for col_index in range(spectra_cube.shape[1]):
                spectrum = spectra_cube[row_index, col_index, :]
                if not np.isfinite(spectrum).all():
                    baseline_cube[row_index, col_index, :] = np.nan
                    corrected_cube[row_index, col_index, :] = np.nan
                    continue

                if normalized_method == "mor":
                    if fixed_half_window is None:
                        baseline, params = baseline_fitter.mor(
                            spectrum,
                            window_kwargs=window_kwargs,
                        )
                        stat_value = int(params.get("half_window", 0))
                    else:
                        baseline, params = baseline_fitter.mor(
                            spectrum,
                            half_window=fixed_half_window,
                        )
                        stat_value = int(params.get("half_window", fixed_half_window))
                else:
                    if normalized_method == "airpls":
                        baseline, params = baseline_fitter.airpls(
                            spectrum,
                            **airpls_kwargs,
                        )
                        stat_value = int(len(params.get("tol_history", [])))
                    elif normalized_method == "poly":
                        baseline, _ = baseline_fitter.poly(
                            spectrum,
                            weights=poly_weights,
                            **poly_kwargs,
                        )
                        stat_value = int(np.count_nonzero(poly_weights)) if poly_weights is not None else int(len(wavenumber))
                    elif normalized_method == "rolling_ball":
                        baseline, params = baseline_fitter.rolling_ball(
                            spectrum,
                            **rolling_ball_kwargs,
                        )
                        stat_value = int(params.get("half_window", 0))
                    elif normalized_method == "noiseaware":
                        _, baseline, info = auto_baseline_noiseaware(
                            wavenumber,
                            spectrum,
                            peak_regions=noiseaware_peak_regions,
                            **noiseaware_kwargs,
                        )
                        stat_value = int(info["anchors_used"])
                        anchor_x_values = _normalize_noiseaware_anchor_values(
                            cast(Sequence[float] | None, info.get("anchor_x_pre_median")),
                        )
                        anchor_y_values = _normalize_noiseaware_anchor_values(
                            cast(Sequence[float] | None, info.get("anchor_y_pre_median")),
                        )
                        if noiseaware_anchor_mask_cube is not None:
                            noiseaware_anchor_mask_cube[row_index, col_index, :] = _build_noiseaware_anchor_mask(
                                np.asarray(wavenumber, dtype=float),
                                anchor_x_values,
                            )
                        if noiseaware_anchor_x_values_grid is not None:
                            noiseaware_anchor_x_values_grid[row_index, col_index] = anchor_x_values
                        if noiseaware_anchor_y_values_grid is not None:
                            noiseaware_anchor_y_values_grid[row_index, col_index] = anchor_y_values

                baseline_cube[row_index, col_index, :] = baseline
                corrected_cube[row_index, col_index, :] = spectrum - baseline
                stat_cube[row_index, col_index] = stat_value

        corrected_item = {
            **parsed,
            "baseline_cube": baseline_cube,
            "corrected_spectra_cube": corrected_cube,
            "baseline_method": normalized_method,
            "baseline_stat_cube": stat_cube,
            "baseline_stat_label": stat_label,
            "baseline_fit_config": {
                "noiseaware_kwargs": dict(noiseaware_kwargs),
                "noiseaware_peak_regions": normalized_peak_regions,
            },
        }
        if normalized_method == "mor":
            corrected_item["mor_half_window_cube"] = stat_cube
        elif normalized_method == "airpls":
            corrected_item["airpls_iteration_cube"] = stat_cube
        elif normalized_method == "poly":
            corrected_item["poly_mask_points_cube"] = stat_cube
        elif normalized_method == "rolling_ball":
            corrected_item["rolling_ball_half_window_cube"] = stat_cube
        elif normalized_method == "noiseaware":
            corrected_item["noiseaware_anchors_used_cube"] = stat_cube
            if noiseaware_anchor_mask_cube is not None:
                corrected_item["noiseaware_anchor_mask_cube"] = noiseaware_anchor_mask_cube
            if noiseaware_anchor_x_values_grid is not None:
                corrected_item["noiseaware_anchor_x_values_grid"] = noiseaware_anchor_x_values_grid
            if noiseaware_anchor_y_values_grid is not None:
                corrected_item["noiseaware_anchor_y_values_grid"] = noiseaware_anchor_y_values_grid

        corrected_collection.append(corrected_item)

    return corrected_collection


def apply_mor_baseline(
    parsed_collection: ParsedCollection,
    fixed_half_window: int | None,
    window_kwargs: dict,
) -> ParsedCollectionMutable:
    """Backward-compatible wrapper for MOR-only baseline correction."""
    return apply_baseline_correction(
        parsed_collection=parsed_collection,
        baseline_method="mor",
        fixed_half_window=fixed_half_window,
        window_kwargs=window_kwargs,
        airpls_kwargs={},
    )


def select_max_intensity_pixel(
    parsed_item: ParsedMap,
    spectrum_key: str = "corrected_spectra_cube",
) -> tuple[int, int]:
    """Return the pixel whose selected spectrum has the highest maximum signal value."""
    cube = np.asarray(parsed_item[spectrum_key], dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D spectra cube for '{spectrum_key}', got shape {cube.shape}")

    finite_cube = np.where(np.isfinite(cube), cube, -np.inf)
    peak_map = finite_cube.max(axis=2)
    if not np.isfinite(peak_map).any():
        raise ValueError(f"No finite values found in '{spectrum_key}'")

    flat_index = int(np.argmax(peak_map))
    row_index, col_index = np.unravel_index(flat_index, peak_map.shape)
    return int(row_index), int(col_index)


def _compute_pixel_spectrum_comparison_data(
    parsed_item: ParsedMap,
    row_index: int,
    col_index: int,
    *,
    spectrum_key: str = "corrected_spectra_cube",
    stage_label: str = "Baseline corrected",
    show_previous_overlay: bool = True,
    show_baseline: bool = True,
    show_noiseaware_anchors: bool = False,
    previous_label: str = "Previous processed spectrum",
    previous_parsed_item: ParsedMap | None = None,
    previous_spectrum_key: str = "spectra_cube",
    baseline_label: str | None = None,
) -> dict[str, object]:
    """Resolve traces/labels for the pixel spectrum comparison plot (no plotting)."""
    wavenumber = np.asarray(parsed_item["wavenumber_cm1"], dtype=float)
    selected_spectrum = np.asarray(parsed_item[spectrum_key][row_index, col_index, :], dtype=float)

    previous_trace: dict[str, object] | None = None
    if show_previous_overlay and previous_parsed_item is not None and previous_spectrum_key in previous_parsed_item:
        previous_spectrum = np.asarray(previous_parsed_item[previous_spectrum_key][row_index, col_index, :], dtype=float)
        previous_wavenumber = np.asarray(previous_parsed_item.get("wavenumber_cm1", wavenumber), dtype=float)
        if previous_wavenumber.shape[0] != previous_spectrum.shape[0]:
            # Fallback to the current stage axis only when dimensions match.
            if wavenumber.shape[0] == previous_spectrum.shape[0]:
                previous_wavenumber = wavenumber
            else:
                min_len = min(previous_wavenumber.shape[0], previous_spectrum.shape[0])
                previous_wavenumber = previous_wavenumber[:min_len]
                previous_spectrum = previous_spectrum[:min_len]
        previous_trace = {
            "wavenumber": previous_wavenumber,
            "intensity": previous_spectrum,
            "label": previous_label,
        }
    elif show_previous_overlay and spectrum_key != "spectra_cube" and "spectra_cube" in parsed_item:
        previous_spectrum = np.asarray(parsed_item["spectra_cube"][row_index, col_index, :], dtype=float)
        previous_trace = {
            "wavenumber": wavenumber,
            "intensity": previous_spectrum,
            "label": previous_label,
        }

    baseline_trace: dict[str, object] | None = None
    if show_baseline and "baseline_cube" in parsed_item:
        baseline = np.asarray(parsed_item["baseline_cube"][row_index, col_index, :], dtype=float)
        resolved_baseline_label = baseline_label or f"{str(parsed_item.get('baseline_method', 'baseline')).upper()} baseline"
        baseline_trace = {
            "wavenumber": wavenumber,
            "intensity": baseline,
            "label": resolved_baseline_label,
        }

    anchor_x = np.asarray([], dtype=float)
    anchor_y = np.asarray([], dtype=float)
    if show_noiseaware_anchors:
        baseline_method = str(parsed_item.get("baseline_method", "")).lower()
        if baseline_method == "noiseaware":
            # Anchors belong to the pre-baseline signal (previous processed spectrum),
            # not the already baseline-corrected curve.
            if spectrum_key != "spectra_cube" and "spectra_cube" in parsed_item:
                anchor_source_spectrum = np.asarray(parsed_item["spectra_cube"][row_index, col_index, :], dtype=float)
            else:
                anchor_source_spectrum = selected_spectrum

            anchor_pairs = _get_noiseaware_anchor_pairs(parsed_item, row_index, col_index)
            if anchor_pairs:
                # Use the persisted exact pre-median x/y values rather than snapping to the grid.
                anchor_x = np.asarray([pair[0] for pair in anchor_pairs], dtype=float)
                anchor_y = np.asarray([pair[1] for pair in anchor_pairs], dtype=float)
            elif "noiseaware_anchor_mask_cube" in parsed_item:
                anchor_mask = np.asarray(parsed_item["noiseaware_anchor_mask_cube"][row_index, col_index, :], dtype=bool)
                if anchor_mask.shape[0] != wavenumber.shape[0] or anchor_mask.shape[0] != anchor_source_spectrum.shape[0]:
                    anchor_mask = np.zeros_like(wavenumber, dtype=bool)
                finite_anchor_mask = anchor_mask & np.isfinite(wavenumber) & np.isfinite(anchor_source_spectrum)
                anchor_x = wavenumber[finite_anchor_mask]
                anchor_y = anchor_source_spectrum[finite_anchor_mask]

            if anchor_x.size and anchor_x.size > 250:
                sample_step = max(1, anchor_x.size // 250)
                anchor_x = anchor_x[::sample_step]
                anchor_y = anchor_y[::sample_step]

    return {
        "wavenumber": wavenumber,
        "selected_spectrum": selected_spectrum,
        "stage_label": stage_label,
        "previous_trace": previous_trace,
        "baseline_trace": baseline_trace,
        "anchor_x": anchor_x,
        "anchor_y": anchor_y,
    }


def plot_pixel_spectrum_comparison(
    ax,
    parsed_item: ParsedMap,
    row_index: int,
    col_index: int,
    *,
    spectrum_key: str = "corrected_spectra_cube",
    stage_label: str = "Baseline corrected",
    figure_title: str | None = None,
    highlight_wavenumber: float | None = None,
    show_previous_overlay: bool = True,
    show_baseline: bool = True,
    show_noiseaware_anchors: bool = False,
    previous_label: str = "Previous processed spectrum",
    previous_parsed_item: ParsedMap | None = None,
    previous_spectrum_key: str = "spectra_cube",
    baseline_label: str | None = None,
) -> None:
    """Plot corrected, previous-stage, and baseline spectra on one axis."""
    data = _compute_pixel_spectrum_comparison_data(
        parsed_item,
        row_index,
        col_index,
        spectrum_key=spectrum_key,
        stage_label=stage_label,
        show_previous_overlay=show_previous_overlay,
        show_baseline=show_baseline,
        show_noiseaware_anchors=show_noiseaware_anchors,
        previous_label=previous_label,
        previous_parsed_item=previous_parsed_item,
        previous_spectrum_key=previous_spectrum_key,
        baseline_label=baseline_label,
    )

    ax.plot(
        data["wavenumber"],
        data["selected_spectrum"],
        color="tab:blue",
        linewidth=1.5,
        label=data["stage_label"],
    )

    if data["previous_trace"] is not None:
        ax.plot(
            data["previous_trace"]["wavenumber"],
            data["previous_trace"]["intensity"],
            color="0.45",
            linewidth=1.0,
            linestyle="--",
            label=data["previous_trace"]["label"],
        )

    if data["baseline_trace"] is not None:
        ax.plot(
            data["baseline_trace"]["wavenumber"],
            data["baseline_trace"]["intensity"],
            color="tab:red",
            linewidth=1.2,
            linestyle=":",
            label=data["baseline_trace"]["label"],
        )

    if data["anchor_x"].size and data["anchor_y"].size:
        ax.scatter(
            data["anchor_x"],
            data["anchor_y"],
            s=20,
            facecolors="none",
            edgecolors="tab:green",
            linewidths=1.0,
            alpha=0.9,
            label="Background anchors (pre-median)",
        )

    if highlight_wavenumber is not None:
        ax.axvline(float(highlight_wavenumber), color="tab:orange", linestyle="--", linewidth=1.0)

    ax.set_title(figure_title or f"Pixel ({row_index}, {col_index})")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (CCD cts)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)


def save_pixel_spectrum_comparison(
    parsed_item: ParsedMap,
    output_path: Path,
    *,
    spectrum_key: str = "corrected_spectra_cube",
    stage_label: str = "Baseline corrected",
    figure_title: str | None = None,
    highlight_wavenumber: float | None = None,
    show_previous_overlay: bool = True,
    show_baseline: bool = True,
    show_noiseaware_anchors: bool = False,
    previous_label: str = "Previous processed spectrum",
    previous_parsed_item: ParsedMap | None = None,
    previous_spectrum_key: str = "spectra_cube",
    baseline_label: str | None = None,
) -> tuple[int, int]:
    """Save a single pixel spectrum comparison figure and return the selected pixel."""
    import matplotlib.pyplot as plt

    row_index, col_index = select_max_intensity_pixel(parsed_item, spectrum_key=spectrum_key)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    plot_pixel_spectrum_comparison(
        ax=ax,
        parsed_item=parsed_item,
        row_index=row_index,
        col_index=col_index,
        spectrum_key=spectrum_key,
        stage_label=stage_label,
        figure_title=figure_title,
        highlight_wavenumber=highlight_wavenumber,
        show_previous_overlay=show_previous_overlay,
        show_baseline=show_baseline,
        show_noiseaware_anchors=show_noiseaware_anchors,
        previous_label=previous_label,
        previous_parsed_item=previous_parsed_item,
        previous_spectrum_key=previous_spectrum_key,
        baseline_label=baseline_label,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return row_index, col_index
