"""Stage 6 export orchestration and explorer snapshot persistence."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core.analysis import _item_has_cut_pixels, select_max_intensity_pixel
from ..core.metadata import infer_sample_name
from ..plotting.explorer import save_pixel_spectrum_comparison
from ..plotting.maps import _save_cut_pixel_map_slice, _save_despiked_baseline_anchor_stack
from .csv_export import (
    _export_cut_pixel_map_slice_csv,
    _export_despiked_baseline_anchor_stack_csv,
    _export_pixel_spectrum_csv,
    _export_spectra_per_file,
)
from .paths import _EXPORT_PART_PREFIXES, _prefix_indexed_stem, _prepare_export_subdir, _sanitize_export_stem

ParsedMap = Mapping[str, Any]
StageCollections = Mapping[str, Sequence[ParsedMap]]
StageSpectrumKeys = Mapping[str, str]


def _backup_code_snapshot(output_dir: Path) -> Path:
    """Copy current .py and .ipynb files to a timestamped backup folder."""
    # Two levels above raman/export/snapshot.py is the project root.
    project_dir = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = output_dir / _EXPORT_PART_PREFIXES["code_snapshot"]
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_dir = backup_root / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)

    for pattern in ("*.py", "*.ipynb"):
        for src in sorted(project_dir.glob(pattern)):
            if src.is_file():
                shutil.copy2(src, backup_dir / src.name)

    # Snapshot the raman/ package tree recursively, preserving its layout.
    package_dir = project_dir / "raman"
    for src in sorted(package_dir.rglob("*.py")):
        if not src.is_file():
            continue
        dest = backup_dir / src.relative_to(project_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    return backup_dir


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

