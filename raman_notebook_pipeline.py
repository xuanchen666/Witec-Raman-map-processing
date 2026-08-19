"""Thin orchestration helpers to keep Raman notebooks clean and reproducible."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

from raman_config import (
    BORDER_FILTER_APPLY_TO_GROUPS,
    BORDER_FILTER_BORDER_WIDTH,
    BORDER_FILTER_ENABLED,
    MAX_INTENSITY_APPLY_TO_GROUPS,
    MAX_PIXEL_INTENSITY,
    SPECTRUM_GATE_APPLY_TO_GROUPS,
    SPECTRUM_GATE_ENABLED,
    SPECTRUM_GATE_MIN_MEAN_INTENSITY,
    SPECTRUM_GATE_WAVENUMBER_REGION_CM1,
)
from raman_parser import ParsedRamanExport, parse_raman_export
from raman_processing_utils import (
    apply_baseline_correction,
    despike_parsed_collection,
    filter_spectra_by_border_pixels,
    filter_spectra_by_max_intensity,
    filter_spectra_by_wavenumber_region_mean,
    filter_low_wavenumber_region,
    summarize_parsed_collection,
)


class Stage1ParseArtifacts(TypedDict):
    """Outputs produced by Stage 1 (parse)."""

    txt_files: list[Path]
    parsed_files: list[ParsedRamanExport]
    parsed_summary: pd.DataFrame
    all_tidy: pd.DataFrame


class _Stage2SpectrumGateArtifacts(TypedDict):
    """Outputs produced by the Stage 2 spectrum-gate substep."""

    spectra_gated_parsed_files: list[dict]
    spectra_gated_all_tidy: pd.DataFrame
    spectra_gated_summary: pd.DataFrame
    spectra_gate_report: pd.DataFrame


class _Stage2BorderFilterArtifacts(TypedDict):
    """Outputs produced by the Stage 2 border-filter substep."""

    border_filtered_parsed_files: list[dict]
    border_filtered_all_tidy: pd.DataFrame
    border_filtered_summary: pd.DataFrame
    border_filter_report: pd.DataFrame


class Stage2PixelFilterArtifacts(TypedDict):
    """Outputs produced by Stage 2 (all pixel-level filters)."""

    pixel_filtered_parsed_files: list[dict]
    pixel_filtered_all_tidy: pd.DataFrame
    pixel_filtered_summary: pd.DataFrame
    border_filter_report: pd.DataFrame
    spectrum_gate_report: pd.DataFrame
    max_intensity_gate_report: pd.DataFrame


class Stage3LowWavenumberArtifacts(TypedDict):
    """Outputs produced by Stage 3 (wavenumber-axis filtering only)."""

    low_wavenumber_parsed_files: list[dict]
    low_wavenumber_all_tidy: pd.DataFrame
    low_wavenumber_summary: pd.DataFrame


class Stage4DespikeArtifacts(TypedDict):
    """Outputs produced by Stage 4 (despike)."""

    pre_despike_parsed_files: list[dict]
    despiked_parsed_files: list[dict]
    despiked_summary: pd.DataFrame
    max_intensity_gate_report: pd.DataFrame


class Stage5BaselineArtifacts(TypedDict):
    """Outputs produced by Stage 5 (baseline correction)."""

    corrected_parsed_files: list[dict]
    corrected_summary: pd.DataFrame


def list_txt_files(data_dir: Path) -> list[Path]:
    """Return deterministically ordered Raman export files."""
    return sorted(data_dir.glob("*.txt"))


def run_stage1_parse(data_dir: Path) -> Stage1ParseArtifacts:
    """Parse all exports and build top-level summary tables."""
    txt_files = list_txt_files(data_dir)
    parsed_files = [parse_raman_export(path) for path in txt_files]
    parsed_summary = summarize_parsed_collection(parsed_files)

    all_tidy = (
        pd.concat([item["tidy"] for item in parsed_files], ignore_index=True)
        if parsed_files
        else pd.DataFrame()
    )

    return {
        "txt_files": txt_files,
        "parsed_files": parsed_files,
        "parsed_summary": parsed_summary,
        "all_tidy": all_tidy,
    }


def _run_stage2_spectrum_gate(
    parsed_files: list[dict],
    enabled: bool,
    wavenumber_region_cm1: tuple[float, float],
    min_mean_intensity: float,
    apply_to_groups: tuple[str, ...] = ("all",),
) -> _Stage2SpectrumGateArtifacts:
    """Optionally drop low-signal spectra within Stage 2."""
    if enabled:
        spectra_gated_parsed_files, spectra_gate_report = filter_spectra_by_wavenumber_region_mean(
            parsed_collection=parsed_files,
            wavenumber_region_cm1=wavenumber_region_cm1,
            min_mean_intensity=min_mean_intensity,
            apply_to_groups=apply_to_groups,
        )
    else:
        spectra_gated_parsed_files = list(parsed_files)
        spectra_gate_report = pd.DataFrame(
            {
                "file": [item["path"].name for item in parsed_files],
                "group": "Not applied",
                "gate_applied": False,
                "window_start_cm1": float(min(wavenumber_region_cm1)),
                "window_end_cm1": float(max(wavenumber_region_cm1)),
                "threshold": float(min_mean_intensity),
                "spectra_total": [int(item["spectra_cube"].shape[0] * item["spectra_cube"].shape[1]) for item in parsed_files],
                "spectra_kept": [int(item["spectra_cube"].shape[0] * item["spectra_cube"].shape[1]) for item in parsed_files],
                "spectra_dropped": 0,
                "drop_fraction": 0.0,
            }
        )

    spectra_gated_all_tidy = (
        pd.concat([item["tidy"] for item in spectra_gated_parsed_files], ignore_index=True)
        if spectra_gated_parsed_files
        else pd.DataFrame()
    )
    spectra_gated_summary = summarize_parsed_collection(spectra_gated_parsed_files)
    if not spectra_gated_summary.empty and "spectra_kept" in spectra_gate_report.columns:
        spectra_gated_summary = spectra_gated_summary.merge(
            spectra_gate_report[["file", "spectra_kept", "spectra_dropped", "drop_fraction"]],
            on="file",
            how="left",
        )

    return {
        "spectra_gated_parsed_files": spectra_gated_parsed_files,
        "spectra_gated_all_tidy": spectra_gated_all_tidy,
        "spectra_gated_summary": spectra_gated_summary,
        "spectra_gate_report": spectra_gate_report,
    }


def _run_stage2_border_filter(
    parsed_files: list[dict],
    border_width: int | None = None,
    apply_to_groups: tuple[str, ...] | None = None,
    enabled: bool | None = None,
) -> _Stage2BorderFilterArtifacts:
    """Optionally remove an outer ring of pixels from selected map groups."""
    if enabled is None:
        enabled = bool(BORDER_FILTER_ENABLED)
    if border_width is None:
        border_width = int(BORDER_FILTER_BORDER_WIDTH)
    if apply_to_groups is None:
        apply_to_groups = tuple(BORDER_FILTER_APPLY_TO_GROUPS)

    if not enabled:
        border_filtered_parsed_files = list(parsed_files)
        border_filter_report = pd.DataFrame(
            {
                "file": [item["path"].name for item in parsed_files],
                "group": ["Not applied" for _ in parsed_files],
                "border_applied": False,
                "border_width": int(border_width),
                "spectra_total": [int(item["spectra_cube"].shape[0] * item["spectra_cube"].shape[1]) for item in parsed_files],
                "spectra_kept": [int(item["spectra_cube"].shape[0] * item["spectra_cube"].shape[1]) for item in parsed_files],
                "spectra_dropped": 0,
                "drop_fraction": 0.0,
            }
        )
    else:
        border_filtered_parsed_files, border_filter_report = filter_spectra_by_border_pixels(
            parsed_collection=parsed_files,
            border_width=border_width,
            apply_to_groups=apply_to_groups,
        )

    border_filtered_all_tidy = (
        pd.concat([item["tidy"] for item in border_filtered_parsed_files], ignore_index=True)
        if border_filtered_parsed_files
        else pd.DataFrame()
    )
    border_filtered_summary = summarize_parsed_collection(border_filtered_parsed_files)
    if not border_filtered_summary.empty and "spectra_kept" in border_filter_report.columns:
        border_filtered_summary = border_filtered_summary.merge(
            border_filter_report[["file", "spectra_kept", "spectra_dropped", "drop_fraction"]],
            on="file",
            how="left",
        )

    return {
        "border_filtered_parsed_files": border_filtered_parsed_files,
        "border_filtered_all_tidy": border_filtered_all_tidy,
        "border_filtered_summary": border_filtered_summary,
        "border_filter_report": border_filter_report,
    }


def _summarize_pixel_keep_status(parsed_files: list[dict], threshold: float | None) -> pd.DataFrame:
    """Build a compact per-map keep/drop report from current keep masks."""
    spectra_total: list[int] = []
    spectra_kept: list[int] = []

    for item in parsed_files:
        total = int(item["spectra_cube"].shape[0] * item["spectra_cube"].shape[1])
        spectra_total.append(total)

        if "spectrum_keep_mask" in item:
            keep_mask = np.asarray(item["spectrum_keep_mask"], dtype=bool)
            spectra_kept.append(int(np.count_nonzero(keep_mask)))
            continue

        with np.errstate(invalid="ignore"):
            inferred_keep_mask = np.isfinite(np.nanmean(item["spectra_cube"], axis=2))
        spectra_kept.append(int(np.count_nonzero(inferred_keep_mask)))

    spectra_dropped = [
        int(total - kept) for total, kept in zip(spectra_total, spectra_kept)
    ]
    drop_fraction = [
        (dropped / total) if total else np.nan
        for total, dropped in zip(spectra_total, spectra_dropped)
    ]

    return pd.DataFrame(
        {
            "file": [item["path"].name for item in parsed_files],
            "threshold": np.nan if threshold is None else float(threshold),
            "spectra_total": spectra_total,
            "spectra_kept": spectra_kept,
            "spectra_dropped": spectra_dropped,
            "drop_fraction": drop_fraction,
        }
    )


def run_stage2_pixel_filter(
    parsed_files: list[dict],
    border_enabled: bool | None = None,
    border_width: int | None = None,
    border_apply_to_groups: tuple[str, ...] | None = None,
    spectrum_gate_enabled: bool | None = None,
    spectrum_gate_wavenumber_region_cm1: tuple[float, float] | None = None,
    spectrum_gate_min_mean_intensity: float | None = None,
    spectrum_gate_apply_to_groups: tuple[str, ...] | None = None,
    max_pixel_intensity: float | None = None,
    max_intensity_apply_to_groups: tuple[str, ...] | None = None,
) -> Stage2PixelFilterArtifacts:
    """Run all optional pixel-cut filters in sequence for Stage 2."""
    if border_enabled is None:
        border_enabled = bool(BORDER_FILTER_ENABLED)
    if border_width is None:
        border_width = int(BORDER_FILTER_BORDER_WIDTH)
    if border_apply_to_groups is None:
        border_apply_to_groups = tuple(BORDER_FILTER_APPLY_TO_GROUPS)

    if spectrum_gate_enabled is None:
        spectrum_gate_enabled = bool(SPECTRUM_GATE_ENABLED)
    if spectrum_gate_wavenumber_region_cm1 is None:
        spectrum_gate_wavenumber_region_cm1 = (
            float(SPECTRUM_GATE_WAVENUMBER_REGION_CM1[0]),
            float(SPECTRUM_GATE_WAVENUMBER_REGION_CM1[1]),
        )
    if spectrum_gate_min_mean_intensity is None:
        spectrum_gate_min_mean_intensity = float(SPECTRUM_GATE_MIN_MEAN_INTENSITY)
    if spectrum_gate_apply_to_groups is None:
        spectrum_gate_apply_to_groups = tuple(SPECTRUM_GATE_APPLY_TO_GROUPS)
    if max_pixel_intensity is None:
        max_pixel_intensity = (
            float(MAX_PIXEL_INTENSITY)
            if MAX_PIXEL_INTENSITY is not None
            else None
        )
    if max_intensity_apply_to_groups is None:
        max_intensity_apply_to_groups = tuple(MAX_INTENSITY_APPLY_TO_GROUPS)

    stage2_border = _run_stage2_border_filter(
        parsed_files=parsed_files,
        border_width=border_width,
        apply_to_groups=border_apply_to_groups,
        enabled=border_enabled,
    )
    border_filtered = stage2_border["border_filtered_parsed_files"]

    stage2_spectrum_gate = _run_stage2_spectrum_gate(
        parsed_files=border_filtered,
        enabled=spectrum_gate_enabled,
        wavenumber_region_cm1=spectrum_gate_wavenumber_region_cm1,
        min_mean_intensity=float(spectrum_gate_min_mean_intensity),
        apply_to_groups=spectrum_gate_apply_to_groups,
    )
    spectrum_gated = stage2_spectrum_gate["spectra_gated_parsed_files"]

    if max_pixel_intensity is not None:
        pixel_filtered, max_intensity_gate_report = filter_spectra_by_max_intensity(
            parsed_collection=spectrum_gated,
            max_intensity=float(max_pixel_intensity),
            apply_to_groups=max_intensity_apply_to_groups,
        )
    else:
        pixel_filtered = list(spectrum_gated)
        max_intensity_gate_report = _summarize_pixel_keep_status(
            parsed_files=pixel_filtered,
            threshold=None,
        )

    pixel_filtered_all_tidy = (
        pd.concat([item["tidy"] for item in pixel_filtered], ignore_index=True)
        if pixel_filtered
        else pd.DataFrame()
    )
    pixel_filtered_summary = summarize_parsed_collection(pixel_filtered)
    if not pixel_filtered_summary.empty and "spectra_kept" in max_intensity_gate_report.columns:
        pixel_filtered_summary = pixel_filtered_summary.merge(
            max_intensity_gate_report[["file", "spectra_kept", "spectra_dropped", "drop_fraction"]],
            on="file",
            how="left",
        )

    return {
        "pixel_filtered_parsed_files": pixel_filtered,
        "pixel_filtered_all_tidy": pixel_filtered_all_tidy,
        "pixel_filtered_summary": pixel_filtered_summary,
        "border_filter_report": stage2_border["border_filter_report"],
        "spectrum_gate_report": stage2_spectrum_gate["spectra_gate_report"],
        "max_intensity_gate_report": max_intensity_gate_report,
    }


def run_stage3_low_wavenumber_filter(
    parsed_files: list[dict],
    min_wavenumber_cm1: float,
) -> Stage3LowWavenumberArtifacts:
    """Run Stage 3 low-wavenumber trimming without changing pixel count."""
    filtered_parsed_files = filter_low_wavenumber_region(
        parsed_files,
        min_wavenumber_cm1,
    )
    filtered_all_tidy = (
        pd.concat([item["tidy"] for item in filtered_parsed_files], ignore_index=True)
        if filtered_parsed_files
        else pd.DataFrame()
    )
    filtered_summary = summarize_parsed_collection(filtered_parsed_files)
    if not filtered_summary.empty:
        filtered_summary["min_wavenumber"] = [
            float(item["wavenumber_cm1"].min()) for item in filtered_parsed_files
        ]

    return {
        "low_wavenumber_parsed_files": filtered_parsed_files,
        "low_wavenumber_all_tidy": filtered_all_tidy,
        "low_wavenumber_summary": filtered_summary,
    }


def run_stage4_despike(
    parsed_files: list[dict],
    neigh: int,
    threshold: int,
) -> Stage4DespikeArtifacts:
    """Run Stage 4 despike without applying an extra pixel cutoff."""
    despiked_files = despike_parsed_collection(
        parsed_files,
        neigh=neigh,
        threshold=threshold,
    )
    max_intensity_gate_report = _summarize_pixel_keep_status(
        parsed_files=despiked_files,
        threshold=None,
    )
    despiked_summary = summarize_parsed_collection(despiked_files)
    if not despiked_summary.empty:
        despiked_summary = despiked_summary.merge(
            max_intensity_gate_report[["file", "spectra_kept", "spectra_dropped", "drop_fraction"]],
            on="file",
            how="left",
        )

    return {
        "pre_despike_parsed_files": parsed_files,
        "despiked_parsed_files": despiked_files,
        "despiked_summary": despiked_summary,
        "max_intensity_gate_report": max_intensity_gate_report,
    }


def run_stage5_baseline(
    parsed_files: list[dict],
    mor_half_window: int | None,
    mor_window_kwargs: dict,
    baseline_method: str = "mor",
    airpls_kwargs: dict | None = None,
    poly_kwargs: dict | None = None,
    poly_mask_regions: list[tuple[float, float]] | None = None,
    rolling_ball_kwargs: dict | None = None,
    noiseaware_kwargs: dict | None = None,
    noiseaware_peak_regions: list[tuple[float, float]] | None = None,
) -> Stage5BaselineArtifacts:
    """Run Stage 5 baseline correction and build a compact QC summary."""
    corrected_parsed_files = apply_baseline_correction(
        parsed_collection=parsed_files,
        fixed_half_window=mor_half_window,
        window_kwargs=mor_window_kwargs,
        baseline_method=baseline_method,
        airpls_kwargs=airpls_kwargs,
        poly_kwargs=poly_kwargs,
        poly_mask_regions=poly_mask_regions,
        rolling_ball_kwargs=rolling_ball_kwargs,
        noiseaware_kwargs=noiseaware_kwargs,
        noiseaware_peak_regions=noiseaware_peak_regions,
    )

    stat_label = (
        corrected_parsed_files[0]["baseline_stat_label"]
        if corrected_parsed_files
        else "baseline_stat"
    )

    corrected_summary = pd.DataFrame(
        {
            "file": [item["path"].name for item in corrected_parsed_files],
            f"{stat_label}_min": [float(np.nanmin(item["baseline_stat_cube"])) for item in corrected_parsed_files],
            f"{stat_label}_max": [float(np.nanmax(item["baseline_stat_cube"])) for item in corrected_parsed_files],
            "points": [len(item["wavenumber_cm1"]) for item in corrected_parsed_files],
            "min_corrected": [float(np.nanmin(item["corrected_spectra_cube"])) for item in corrected_parsed_files],
            "max_corrected": [float(np.nanmax(item["corrected_spectra_cube"])) for item in corrected_parsed_files],
        }
    )

    return {
        "corrected_parsed_files": corrected_parsed_files,
        "corrected_summary": corrected_summary,
    }


def build_explorer_stage_mappings(
    parsed_files: list[ParsedRamanExport],
    stage2_pixel_filtered_parsed_files: list[dict],
    stage3_low_wavenumber_parsed_files: list[dict],
    stage4_despiked_parsed_files: list[dict],
    stage5_corrected_parsed_files: list[dict],
    stage6_map_average_parsed_files: list[dict] | None = None,
) -> tuple[dict[str, list], dict[str, str]]:
    """Build collection/cube-key mappings aligned with the notebook stage order."""
    stage_collections: dict[str, list] = {
        "Stage 1 Parsed": parsed_files,
        "Stage 2 Pixel filtered": stage2_pixel_filtered_parsed_files,
        "Stage 3 Low-wavenumber filtered": stage3_low_wavenumber_parsed_files,
        "Stage 4 Despiked": stage4_despiked_parsed_files,
        "Stage 5 Baseline corrected": stage5_corrected_parsed_files,
    }
    stage_spectrum_keys: dict[str, str] = {
        "Stage 1 Parsed": "spectra_cube",
        "Stage 2 Pixel filtered": "spectra_cube",
        "Stage 3 Low-wavenumber filtered": "spectra_cube",
        "Stage 4 Despiked": "spectra_cube",
        "Stage 5 Baseline corrected": "corrected_spectra_cube",
    }

    if stage6_map_average_parsed_files is not None:
        stage_collections["Stage 6 Map-average plotting"] = stage6_map_average_parsed_files
        stage_spectrum_keys["Stage 6 Map-average plotting"] = "corrected_spectra_cube"

    return stage_collections, stage_spectrum_keys
