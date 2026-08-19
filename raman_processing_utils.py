"""Utility functions for Raman notebook processing stages.

These helpers keep the notebook concise while preserving the same behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rampy as rp
from pybaselines import Baseline

from raman_noiseaware_baseline import auto_baseline_noiseaware

# Shared aliases used throughout this module to make function signatures easier to read.
ParsedMap = Mapping[str, Any]
ParsedMapMutable = dict[str, Any]
ParsedCollection = Sequence[ParsedMap]
ParsedCollectionMutable = list[ParsedMapMutable]
StageCollections = Mapping[str, Sequence[ParsedMap]]
StageSpectrumKeys = Mapping[str, str]


def save_explorer_snapshot(
    output_dir: Path,
    stage_collections: StageCollections,
    stage_spectrum_keys: StageSpectrumKeys,
    map_mode: str = "max",
) -> Path:
    """Persist all explorer stage data to a timestamped gzip-compressed pickle file."""
    import gzip
    import pickle
    from datetime import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"explorer_snapshot_{timestamp}.pkl.gz"

    snapshot = {
        "stage_collections": {stage: list(items) for stage, items in stage_collections.items()},
        "stage_spectrum_keys": dict(stage_spectrum_keys),
        "map_mode": map_mode,
    }
    with gzip.open(output_path, "wb") as handle:
        pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path


def find_latest_explorer_snapshot(snapshot_dir: Path) -> Path:
    """Return the most recently saved `explorer_snapshot_*.pkl.gz` file in a directory."""
    snapshot_dir = Path(snapshot_dir)
    candidates = sorted(snapshot_dir.glob("explorer_snapshot_*.pkl.gz"))
    if not candidates:
        raise FileNotFoundError(f"No explorer snapshot files found in: {snapshot_dir}")
    return candidates[-1]


def load_explorer_snapshot(input_path: Path) -> tuple[StageCollections, StageSpectrumKeys, str]:
    """Load explorer stage data previously written by `save_explorer_snapshot`."""
    import gzip
    import pickle

    input_path = Path(input_path)
    with gzip.open(input_path, "rb") as handle:
        snapshot = pickle.load(handle)

    return snapshot["stage_collections"], snapshot["stage_spectrum_keys"], snapshot["map_mode"]


def _plot_average_pixel_overlay(map_ax, item: ParsedMap) -> int:
    """Overlay average-selected pixels on a map axis, if the mask is available."""
    average_pixel_mask = item.get("average_pixel_mask")
    if average_pixel_mask is None:
        return 0

    mask = np.asarray(average_pixel_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return 0

    selected_coords = np.argwhere(mask)
    map_ax.scatter(
        selected_coords[:, 0],
        selected_coords[:, 1],
        s=42,
        facecolors="none",
        edgecolors="white",
        linewidths=0.7,
        label="Average pixels",
    )
    return int(selected_coords.shape[0])


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


def _format_noiseaware_anchor_html(anchor_pairs: Sequence[tuple[float, float]]) -> str:
    """Format persisted Stage 5 pre-median anchors for explorer metadata display."""
    if not anchor_pairs:
        return "<b>Stage 5 anchors:</b> n/a"

    preview_limit = 8
    formatted_pairs = [
        f"({anchor_x:.2f} cm<sup>-1</sup>, {anchor_y:.2f})"
        for anchor_x, anchor_y in anchor_pairs[:preview_limit]
    ]
    suffix = "" if len(anchor_pairs) <= preview_limit else f" ... (+{len(anchor_pairs) - preview_limit} more)"
    return (
        f"<b>Stage 5 anchors:</b> {len(anchor_pairs)} &nbsp; "
        f"<b>Pre-median x/y:</b> {'; '.join(formatted_pairs)}{suffix}"
    )


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


def _resolve_stage_spectrum_keys(
    stage_collections: StageCollections,
    stage_spectrum_keys: StageSpectrumKeys | None = None,
) -> dict[str, str]:
    """Resolve the spectrum cube key for each processing stage."""
    default_stage_spectrum_keys = {
        stage_name: "spectra_cube" for stage_name in stage_collections
    }
    for stage_name, collection in stage_collections.items():
        if collection and "corrected_spectra_cube" in collection[0]:
            default_stage_spectrum_keys[stage_name] = "corrected_spectra_cube"

    if stage_spectrum_keys is None:
        return default_stage_spectrum_keys

    return {
        **default_stage_spectrum_keys,
        **stage_spectrum_keys,
    }


def _build_map_image(cube: np.ndarray, map_mode: str, wn_idx: int) -> np.ndarray:
    """Build the 2D image shown in the map panel."""
    if map_mode == "slice":
        return cube[:, :, wn_idx]
    if map_mode == "mean":
        with np.errstate(invalid="ignore"):
            return np.nanmean(cube, axis=2)
    if map_mode == "max":
        with np.errstate(invalid="ignore"):
            return np.nanmax(cube, axis=2)
    raise ValueError("map_mode must be one of ['max', 'mean', 'slice']")


def _find_stage_item(
    stage_collections: StageCollections,
    stage_name: str,
    file_name: str,
) -> ParsedMap | None:
    """Return the map item for a given stage and file name, if available."""
    for item in stage_collections.get(stage_name, []):
        if item["path"].name == file_name:
            return item
    return None


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
    wavenumber = np.asarray(parsed_item["wavenumber_cm1"], dtype=float)
    selected_spectrum = np.asarray(parsed_item[spectrum_key][row_index, col_index, :], dtype=float)

    ax.plot(
        wavenumber,
        selected_spectrum,
        color="tab:blue",
        linewidth=1.5,
        label=stage_label,
    )

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
        ax.plot(
            previous_wavenumber,
            previous_spectrum,
            color="0.45",
            linewidth=1.0,
            linestyle="--",
            label=previous_label,
        )
    elif show_previous_overlay and spectrum_key != "spectra_cube" and "spectra_cube" in parsed_item:
        previous_spectrum = np.asarray(parsed_item["spectra_cube"][row_index, col_index, :], dtype=float)
        ax.plot(
            wavenumber,
            previous_spectrum,
            color="0.45",
            linewidth=1.0,
            linestyle="--",
            label=previous_label,
        )

    if show_baseline and "baseline_cube" in parsed_item:
        baseline = np.asarray(parsed_item["baseline_cube"][row_index, col_index, :], dtype=float)
        resolved_baseline_label = baseline_label or f"{str(parsed_item.get('baseline_method', 'baseline')).upper()} baseline"
        ax.plot(
            wavenumber,
            baseline,
            color="tab:red",
            linewidth=1.2,
            linestyle=":",
            label=resolved_baseline_label,
        )

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
            else:
                anchor_x = np.asarray([], dtype=float)
                anchor_y = np.asarray([], dtype=float)

            if anchor_x.size and anchor_y.size:
                if anchor_x.size > 250:
                    sample_step = max(1, anchor_x.size // 250)
                    anchor_x = anchor_x[::sample_step]
                    anchor_y = anchor_y[::sample_step]
                ax.scatter(
                    anchor_x,
                    anchor_y,
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


def launch_raman_map_explorer(
    stage_collections: StageCollections,
    stage_spectrum_keys: StageSpectrumKeys | None = None,
    map_mode: str = "max",
):
    """Launch an interactive map viewer where clicking a pixel selects its spectrum."""
    from io import BytesIO

    import ipywidgets as widgets
    import matplotlib
    from IPython.display import Image as IPythonImage
    from IPython.display import display

    try:
        import ipympl.backend_nbagg  # noqa: F401

        supports_map_click = "ipympl" in matplotlib.get_backend().lower()
    except (ImportError, ValueError):
        supports_map_click = False

    if not stage_collections:
        raise ValueError("stage_collections is empty")

    valid_map_modes = {"slice", "max", "mean"}
    if map_mode not in valid_map_modes:
        raise ValueError(f"map_mode must be one of {sorted(valid_map_modes)}")

    stage_spectrum_keys = _resolve_stage_spectrum_keys(
        stage_collections=stage_collections,
        stage_spectrum_keys=stage_spectrum_keys,
    )

    stage_names = list(stage_collections.keys())
    stage_dropdown = widgets.Dropdown(
        options=stage_names,
        value=stage_names[0],
        description="Stage:",
        layout=widgets.Layout(width="300px"),
    )
    file_dropdown = widgets.Dropdown(
        options=[],
        description="Map file:",
        layout=widgets.Layout(width="520px"),
    )
    map_mode_dropdown = widgets.Dropdown(
        options=[
            ("Slice", "slice"),
            ("Maximum", "max"),
            ("Mean", "mean"),
        ],
        value=map_mode,
        description="Map view:",
        layout=widgets.Layout(width="220px"),
    )
    map_index_slider = widgets.SelectionSlider(
        options=[("0.00", 0.0)],
        value=0.0,
        description="Wavenumber:",
        continuous_update=False,
        layout=widgets.Layout(width="1400px"),
    )
    prev_point_button = widgets.Button(description="< pt", layout=widgets.Layout(width="55px"))
    next_point_button = widgets.Button(description="pt >", layout=widgets.Layout(width="55px"))
    view_range_slider = widgets.FloatRangeSlider(
        value=[0.0, 1.0],
        min=0.0,
        max=1.0,
        step=1.0,
        description="View range:",
        continuous_update=False,
        layout=widgets.Layout(width="1400px"),
    )
    reset_view_range_button = widgets.Button(description="Reset range", layout=widgets.Layout(width="110px"))
    row_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="X:",
        continuous_update=False,
        layout=widgets.Layout(width="350px"),
    )
    col_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="Y:",
        continuous_update=False,
        layout=widgets.Layout(width="350px"),
    )
    info_html = widgets.HTML(
        value=(
            "Click directly on the map to select a pixel, or use the X/Y sliders for exact selection. "
            "Use the &lt; pt / pt &gt; buttons (or arrow keys after clicking the slider) "
            "to step through every wavenumber point one at a time. "
            "Use the View range slider to zoom the spectrum plot to a wavenumber window; "
            "click Reset range to return to the full spectrum."
        )
        if supports_map_click
        else (
            "Use the X/Y sliders for exact pixel selection. "
            "Direct map clicking is available when the ipympl backend is active. "
            "Use the &lt; pt / pt &gt; buttons (or arrow keys after clicking the slider) "
            "to step through every wavenumber point one at a time. "
            "Use the View range slider to zoom the spectrum plot to a wavenumber window; "
            "click Reset range to return to the full spectrum."
        )
    )
    output = widgets.Output()
    active_canvas_holder: dict[str, Any] = {"figure": None}

    def _step_point(delta: int) -> None:
        options = map_index_slider.options
        if not options:
            return
        new_index = int(np.clip(map_index_slider.index + delta, 0, len(options) - 1))
        map_index_slider.index = new_index

    prev_point_button.on_click(lambda _btn: _step_point(-1))
    next_point_button.on_click(lambda _btn: _step_point(1))

    def _reset_view_range(_btn=None) -> None:
        view_range_slider.value = [view_range_slider.min, view_range_slider.max]

    reset_view_range_button.on_click(_reset_view_range)

    def _current_item() -> tuple[str, ParsedMap, str]:
        stage_name = cast(str, stage_dropdown.value)
        selected_file = cast(str, file_dropdown.value)
        collection = stage_collections[stage_name]

        for item in collection:
            if item["path"].name == selected_file:
                return stage_name, item, stage_spectrum_keys[stage_name]

        if collection:
            return stage_name, collection[0], stage_spectrum_keys[stage_name]

        raise ValueError(f"No maps available for stage '{stage_name}'")

    def _update_file_options(*_):
        stage_name = cast(str, stage_dropdown.value)
        collection = stage_collections[stage_name]
        names = [item["path"].name for item in collection]
        file_dropdown.options = names
        file_dropdown.value = names[0] if names else None

    def _update_slider_range(*_):
        if file_dropdown.value is None:
            return
        _, item, _ = _current_item()
        wavenumber = np.asarray(item["wavenumber_cm1"], dtype=float)
        if wavenumber.size == 0:
            return

        previous_value = float(map_index_slider.value)
        options = [(f"{float(wn):.2f}", float(wn)) for wn in wavenumber]
        map_index_slider.options = options
        closest_idx = int(np.argmin(np.abs(wavenumber - previous_value)))
        map_index_slider.index = closest_idx

        wn_min, wn_max = float(np.nanmin(wavenumber)), float(np.nanmax(wavenumber))
        wn_step = float(np.min(np.diff(np.sort(wavenumber)))) if wavenumber.size > 1 else 1.0
        previous_range = list(view_range_slider.value)
        was_full_range = view_range_slider.max <= view_range_slider.min or (
            np.isclose(previous_range[0], view_range_slider.min) and np.isclose(previous_range[1], view_range_slider.max)
        )
        # Widen bounds before narrowing so ipywidgets never rejects an out-of-range value.
        view_range_slider.min = min(view_range_slider.min, wn_min)
        view_range_slider.max = max(view_range_slider.max, wn_max)
        view_range_slider.step = wn_step
        if was_full_range:
            view_range_slider.value = [wn_min, wn_max]
        else:
            view_range_slider.value = [
                float(np.clip(previous_range[0], wn_min, wn_max)),
                float(np.clip(previous_range[1], wn_min, wn_max)),
            ]
        view_range_slider.min = wn_min
        view_range_slider.max = wn_max

        active_stage = cast(str, stage_dropdown.value)
        n_rows, n_cols, _ = item[stage_spectrum_keys[active_stage]].shape
        row_slider.max = max(0, n_rows - 1)
        col_slider.max = max(0, n_cols - 1)
        if row_slider.value > row_slider.max:
            row_slider.value = row_slider.max
        if col_slider.value > col_slider.max:
            col_slider.value = col_slider.max

    def _on_file_change(*_):
        _update_slider_range()
        _render()

    def _on_stage_change(*_):
        # Suppress file_dropdown's own observer so the value it sets below doesn't re-trigger a duplicate render.
        file_dropdown.unobserve(_on_file_change, names="value")
        try:
            _update_file_options()
        finally:
            file_dropdown.observe(_on_file_change, names="value")
        _update_slider_range()
        _render()

    def _render(*_):
        with output:
            is_initial_interactive_render = supports_map_click and active_canvas_holder["figure"] is None
            if not supports_map_click or is_initial_interactive_render:
                output.clear_output(wait=True)

            if file_dropdown.value is None:
                print("No files available for selected stage")
                return

            stage_name, item, spectrum_key = _current_item()
            if spectrum_key not in item:
                raise KeyError(f"Spectrum key '{spectrum_key}' not found in stage '{stage_name}'")

            cube = item[spectrum_key]
            wavenumber = item["wavenumber_cm1"]

            n_rows, n_cols, _ = cube.shape
            row_value = int(np.clip(row_slider.value, 0, n_rows - 1))
            col_value = int(np.clip(col_slider.value, 0, n_cols - 1))
            slider_index = map_index_slider.index
            wn_idx = int(np.clip(0 if slider_index is None else slider_index, 0, len(wavenumber) - 1))

            active_map_mode = cast(str, map_mode_dropdown.value)
            map_image = _build_map_image(cube, active_map_mode, wn_idx)
            map_image_display = map_image.T

            previous_stage_item: ParsedMap | None = None
            previous_stage_label = "Previous processed spectrum"
            if stage_name in stage_names:
                stage_position = stage_names.index(stage_name)
                if stage_position > 0:
                    previous_stage_name = stage_names[stage_position - 1]
                    previous_stage_item = _find_stage_item(stage_collections, previous_stage_name, item["path"].name)
                    if previous_stage_item is not None:
                        previous_stage_label = f"Previous stage ({previous_stage_name})"
                        if stage_name == "Despiked" and previous_stage_name == "Filtered":
                            previous_stage_label = "Before despike (Filtered)"

                show_previous_overlay = stage_name not in {"Raw parsed", "Filtered"}
                if previous_stage_item is None:
                    show_previous_overlay = False

            if supports_map_click and active_canvas_holder["figure"] is None:
                # ipympl auto-displays pyplot figures unless interactive mode is paused here.
                with plt.ioff():
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
                map_ax, spectrum_ax = axes
                active_canvas_holder["figure"] = fig
            elif supports_map_click:
                fig = active_canvas_holder["figure"]
                fig.clear()
                map_ax, spectrum_ax = fig.subplots(1, 2)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
                map_ax, spectrum_ax = axes

            im = map_ax.imshow(map_image_display, origin="upper", cmap="viridis", aspect="equal")
            map_ax.scatter(row_value, col_value, s=80, c="red", edgecolors="white", linewidths=1.2)
            average_pixels_used = _plot_average_pixel_overlay(map_ax, item)
            map_ax.set_title(f"{stage_name} | {item['path'].name}")
            map_ax.set_xlabel("X index")
            map_ax.set_ylabel("Y index")
            fig.colorbar(im, ax=map_ax, fraction=0.046, pad=0.04, label="Intensity")
            if average_pixels_used:
                legend_handles, legend_labels = map_ax.get_legend_handles_labels()
                map_ax.legend(
                    legend_handles,
                    legend_labels,
                    fontsize=8,
                    loc="upper right",
                    frameon=True,
                )

            plot_pixel_spectrum_comparison(
                ax=spectrum_ax,
                parsed_item=item,
                row_index=row_value,
                col_index=col_value,
                spectrum_key=spectrum_key,
                stage_label=stage_name,
                figure_title=f"Pixel ({row_value}, {col_value})",
                highlight_wavenumber=wavenumber[wn_idx],
                show_previous_overlay=show_previous_overlay,
                show_baseline=True,
                show_noiseaware_anchors=True,
                previous_label=previous_stage_label,
                previous_parsed_item=previous_stage_item,
                previous_spectrum_key="spectra_cube",
            )

            view_min, view_max = view_range_slider.value
            if view_max > view_min:
                spectrum_ax.set_xlim(view_min, view_max)
                visible_y_values = []
                for line in spectrum_ax.get_lines():
                    x_data, y_data = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
                    in_range = (x_data >= view_min) & (x_data <= view_max) & np.isfinite(y_data)
                    if in_range.any():
                        visible_y_values.append(y_data[in_range])
                if visible_y_values:
                    stacked_y = np.concatenate(visible_y_values)
                    y_low, y_high = float(np.min(stacked_y)), float(np.max(stacked_y))
                    y_margin = (y_high - y_low) * 0.08 if y_high > y_low else max(abs(y_high), 1.0) * 0.08
                    spectrum_ax.set_ylim(y_low - y_margin, y_high + y_margin)

            map_mode_label = getattr(map_mode_dropdown, "label", None) or str(map_mode_dropdown.value).capitalize()
            slice_label = (
                f"{wavenumber[wn_idx]:.2f} cm<sup>-1</sup>"
                if map_mode_dropdown.value == "slice"
                else f"{map_mode_label} over spectrum"
            )
            info_html.value = (
                f"<b>Stage:</b> {stage_name} &nbsp; "
                f"<b>File:</b> {item['path'].name} &nbsp; "
                f"<b>Pixel (x,y):</b> ({row_value}, {col_value}) &nbsp; "
                f"<b>Average pixels:</b> {average_pixels_used if average_pixels_used else 'n/a'} &nbsp; "
                f"<b>Map view:</b> {slice_label}"
            )
            if stage_name in {"Stage 5 Baseline corrected", "Stage 6 Map-average plotting"}:
                info_html.value += "<br>" + _format_noiseaware_anchor_html(
                    _get_noiseaware_anchor_pairs(item, row_value, col_value),
                )

            def _onclick(event):
                if event.inaxes is not map_ax or event.xdata is None or event.ydata is None:
                    return
                row_slider.value = int(np.clip(round(event.xdata), 0, n_rows - 1))
                col_slider.value = int(np.clip(round(event.ydata), 0, n_cols - 1))

            fig.canvas.mpl_connect("button_press_event", _onclick)
            fig.tight_layout()
            if supports_map_click:
                fig.canvas.header_visible = False
                fig.canvas.footer_visible = False
                fig.canvas.toolbar_visible = False
                if is_initial_interactive_render:
                    display(fig.canvas)
                else:
                    fig.canvas.draw_idle()
            else:
                image_buffer = BytesIO()
                fig.savefig(image_buffer, format="png", dpi=120, bbox_inches="tight")
                display(IPythonImage(data=image_buffer.getvalue(), format="png"))
                plt.close(fig)

    stage_dropdown.observe(_on_stage_change, names="value")
    file_dropdown.observe(_on_file_change, names="value")
    map_mode_dropdown.observe(_render, names="value")
    map_index_slider.observe(_render, names="value")
    view_range_slider.observe(_render, names="value")
    row_slider.observe(_render, names="value")
    col_slider.observe(_render, names="value")

    _update_file_options()
    _update_slider_range()

    controls = widgets.VBox(
        [
            widgets.HBox([stage_dropdown, file_dropdown]),
            widgets.HBox([map_mode_dropdown, row_slider, col_slider]),
            widgets.HBox([prev_point_button, map_index_slider, next_point_button]),
            widgets.HBox([view_range_slider, reset_view_range_button]),
            info_html,
        ]
    )
    viewer = widgets.VBox([controls, output])
    display(viewer)
    _render()
    return viewer
