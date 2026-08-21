"""Pixel- and wavenumber-axis filtering for parsed Raman map collections."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metadata import extract_group, extract_subgroup

ParsedMap = Mapping[str, Any]
ParsedMapMutable = dict[str, Any]
ParsedCollection = Sequence[ParsedMap]
ParsedCollectionMutable = list[ParsedMapMutable]


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
        map_group = extract_group(parsed["path"].name)
        map_subgroup = extract_subgroup(parsed["path"].name)
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
        map_group = extract_group(parsed["path"].name)
        map_subgroup = extract_subgroup(parsed["path"].name)
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
        map_group = extract_group(parsed["path"].name)
        map_subgroup = extract_subgroup(parsed["path"].name)
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


def _map_matches_pixel_exclusion_key(map_path_name: str, key: str) -> bool:
    """Return whether a config key identifies the given map filename."""
    key = key.strip()
    map_stem = map_path_name.rsplit(".", 1)[0]
    return key == map_path_name or key == map_stem or key in map_path_name


def filter_specific_pixels_by_map(
    parsed_collection: ParsedCollection,
    pixel_exclusions_by_map: Mapping[str, Sequence[tuple[int, int]]],
    spectrum_key: str = "spectra_cube",
) -> tuple[ParsedCollectionMutable, pd.DataFrame]:
    """Drop specific (x_index, y_index) pixels for maps matched by filename.

    ``pixel_exclusions_by_map`` keys can be an exact filename, filename stem,
    or any substring found in the filename (e.g. a distinguishing fragment of
    the map's name), so different maps can each get their own pixel list.
    """
    filtered_collection: ParsedCollectionMutable = []
    report_records: list[dict[str, Any]] = []

    for parsed in parsed_collection:
        map_name = parsed["path"].name
        map_group = extract_group(map_name)
        map_subgroup = extract_subgroup(map_name)

        excluded_pixels: list[tuple[int, int]] = []
        for key, pixels in pixel_exclusions_by_map.items():
            if _map_matches_pixel_exclusion_key(map_name, str(key)):
                excluded_pixels.extend((int(x), int(y)) for x, y in pixels)

        cube = np.asarray(parsed[spectrum_key], dtype=float)
        keep_mask = _resolve_existing_keep_mask(parsed, spectrum_key=spectrum_key).copy()

        if excluded_pixels:
            rows, cols = cube.shape[:2]
            for x_index, y_index in excluded_pixels:
                if not (0 <= x_index < rows and 0 <= y_index < cols):
                    raise ValueError(
                        f"Pixel ({x_index}, {y_index}) out of bounds for {map_name} "
                        f"with shape ({rows}, {cols})"
                    )
                keep_mask[x_index, y_index] = False

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
                "specific_pixel_filter_config": {
                    "map_group": map_group,
                    "map_subgroup": map_subgroup,
                    "excluded_pixels": sorted(set(excluded_pixels)),
                },
            }
        )

        total_spectra = int(keep_mask.size)
        kept_spectra = int(np.count_nonzero(keep_mask))
        dropped_spectra = total_spectra - kept_spectra
        report_records.append(
            {
                "file": map_name,
                "group": map_group,
                "subgroup": map_subgroup,
                "pixels_excluded": len(set(excluded_pixels)),
                "spectra_total": total_spectra,
                "spectra_kept": kept_spectra,
                "spectra_dropped": dropped_spectra,
                "drop_fraction": (dropped_spectra / total_spectra) if total_spectra else np.nan,
            }
        )

    report_df = pd.DataFrame.from_records(report_records)
    return filtered_collection, report_df

