# Raman Processing

A processing pipeline for WITec Raman area-scan `.txt` exports: parses raw maps into
tidy pixel-level spectra, applies optional pixel/spectral filtering, despikes cosmic-ray
artifacts, fits and subtracts a baseline, averages each map into a representative
spectrum, and produces plots/CSV exports plus an interactive map explorer.

## Project layout

```
raman/
├── config.py              # Central, editable configuration (paths, per-stage parameters)
├── core/                  # Pure computation, no plotting dependency
│   ├── metadata.py        #   file parsing + group/subgroup/date/sample-name extraction
│   ├── filters.py         #   border / spectrum-gate / max-intensity / low-wavenumber filters, despike
│   ├── baseline.py        #   MOR / airPLS / poly / rolling-ball / noise-aware baseline correction
│   └── analysis.py        #   map averaging, peak-ratio tables, summaries
├── plotting/               # matplotlib-dependent visualization
│   ├── spectra.py          #   stacked/overlap spectra, peak-ratio trend plots
│   ├── maps.py              #   2D map-slice and despiked/baseline anchor-stack plots
│   └── explorer.py          #   interactive `launch_raman_map_explorer` widget
├── export/                 # File output helpers
│   ├── paths.py             #   filesystem-safe path/stem helpers
│   ├── csv_export.py        #   CSV sidecar writers
│   └── snapshot.py          #   Stage 6 export orchestration + explorer snapshot save/load
└── pipeline.py              # Thin `run_stageN_*` orchestration used by the notebook

Raman_processing.ipynb        # Main pipeline notebook (Stages 1-6 + interactive explorer)
Raman_explorer_reopen.ipynb   # Reopen the explorer from a saved snapshot, without rerunning the pipeline
tests/                        # pytest suite for the pure-compute helpers (headless, no matplotlib import)
```

## Pipeline stages

1. **Parse** – read every `.txt` export in `raman.config.DATA_DIR` into a tidy table + 3D spectra cube.
2. **Pixel filter** – border-pixel trim, mean-intensity spectrum gate, max-intensity cutoff (each independently scoped to a group/subgroup like `hBN` or `hBN_1`), and a final specific-pixel exclusion step for dropping individual `(x_index, y_index)` pixels on named maps.
3. **Low-wavenumber filter** – trim the spectral axis below a configured cutoff.
4. **Despike** – suppress cosmic-ray spikes via `rampy.despiking`.
5. **Baseline correction** – `mor`, `airpls`, `poly`, `rolling_ball`, or `noiseaware`.
6. **Map averaging & export** – average retained pixels per map, split by group, normalize, compute peak ratios, and export plots/CSVs/code snapshots.

An interactive explorer (`raman.plotting.explorer.launch_raman_map_explorer`) lets you
step through every stage/file/pixel; its state can be saved with
`raman.export.snapshot.save_explorer_snapshot` and reopened later from
`Raman_explorer_reopen.ipynb` without rerunning Stages 1-6.

### Stage 6 exports

Stage 6 writes `<sample>_map_analysis_exports` below `DATA_DIR`. Its plot exports are
consolidated into PDFs: `01_plots/stage6_spectra_overview.pdf` contains the average,
normalized, overlap, and peak-ratio pages, while
`01_plots/05_cutpixel_map/stage6_cutpixel_map.pdf` contains one cut-pixel map per
page. CSV data remains available alongside these PDFs.

The `01_plots` subdirectories are numbered continuously:
`01_avg_stack`, `02_norm_stack`, `03_norm_overlap`, `04_peak_ratio`,
`05_cutpixel_map`, and `06_despiked_baseline_anchor_stack`. The latter contains the
despiked spectrum, baseline, and noise-aware anchor data as CSV files; it no longer
creates a separate stack PNG. The obsolete maximum-signal export is no longer
generated.

After Stage 6, the notebook creates a self-contained HTML pipeline report at
`05_report/<sample>_pipeline_report.html`, including the Stage 1-6 summary tables and
relative links to the PDFs and CSV export locations.

### Excluding specific pixels

To drop individual noisy/damaged pixels from specific maps (rather than filtering by
threshold or scope), set `Stage2PixelFilter.SPECIFIC_PIXEL_EXCLUSIONS` in
`raman/config.py`:

```python
SPECIFIC_PIXEL_EXCLUSIONS: dict[str, list[tuple[int, int]]] = {
    "S6_hBN_1_10mW_20260505.txt": [(1, 3), (2, 3)],
}
```

Each key can be an exact filename, filename stem, or any distinguishing substring of
the filename; the matching map has the listed `(x_index, y_index)` pixels dropped
(set to NaN) as the last Stage 2 pixel-filter substep, after the border filter,
spectrum gate, and max-intensity cutoff. The Stage 2 notebook cell reports the result
in `specific_pixel_exclusion_report`.

## Getting started

1. Create/activate the project virtual environment (`.venv`) and install dependencies:
   `numpy`, `pandas`, `scipy`, `matplotlib`, `pybaselines`, `rampy`, `ipywidgets`, `ipympl`, `pytest`.
2. Edit `raman/config.py` to point `DATA_DIR` at your Raman export folder and adjust
   per-stage parameters (filters, baseline method, plotting ranges, explorer settings).
3. Open `Raman_processing.ipynb` and run all cells top to bottom.

## Tests

```
.venv/Scripts/python.exe -m pytest -q
```

The test suite covers the pure-compute helpers under `raman/core` and `raman/plotting`
and asserts that importing `raman.core.*` never pulls in `matplotlib`, so it can run
headless in CI.
