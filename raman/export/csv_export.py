"""CSV sidecar export helpers for plotted/exported Raman map data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..plotting.maps import _compute_despiked_baseline_anchor_stack_data
from .paths import _prefix_indexed_stem


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


def _export_despiked_baseline_anchor_stack_csv(
    parsed_item: dict,
    output_path: Path,
    *,
    despiked_key: str = "spectra_cube",
    baseline_key: str = "baseline_cube",
    anchor_mask_key: str = "noiseaware_anchor_mask_cube",
) -> None:
    """Export stack-plot traces as CSV sidecar."""
    stack_data = _compute_despiked_baseline_anchor_stack_data(
        parsed_item,
        despiked_key=despiked_key,
        baseline_key=baseline_key,
        anchor_mask_key=anchor_mask_key,
        stack_scale=1.0,
        stack_extra_gap=0.0,
    )
    if stack_data["status"] != "ok":
        return

    rows: list[pd.DataFrame] = []
    for trace in stack_data["pixel_traces"]:
        row_i = trace["row_index"]
        col_i = trace["col_index"]
        stack_index = trace["stack_index"]
        offset = trace["offset"]
        wavenumber = trace["wavenumber"]
        despiked = trace["despiked"]
        baseline = trace["baseline"]
        finite_mask = np.isfinite(wavenumber)
        anchor_mask = trace["anchor_mask"]

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

