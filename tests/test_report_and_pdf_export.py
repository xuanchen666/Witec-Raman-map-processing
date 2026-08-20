"""Tests for PDF-sink plotting behavior and the pipeline HTML report.

Unlike the other `test_plotting_*` files, these tests exercise real figure/PdfPages
output. matplotlib is imported lazily inside each test (never at module level) so
pytest's collection-time import of this file does not break
`test_plotting_explorer.py::test_importing_modules_does_not_pull_in_matplotlib`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import raman.export.report as rrep
import raman.export.paths as rpaths
import raman.plotting.maps as rpm
import raman.plotting.spectra as rps


def _make_avg_map_spectra() -> pd.DataFrame:
    wavenumber = np.linspace(200.0, 1800.0, 50)
    rows = []
    for i in range(2):
        rows.append(
            {
                "group": "hBN",
                "subgroup": None,
                "file": f"map{i}.txt",
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                "wavenumber_cm1": wavenumber,
                "mean_spectrum": np.random.default_rng(i).normal(size=50),
            }
        )
    return pd.DataFrame(rows)


def test_plot_grouped_spectra_appends_to_pdf_instead_of_png(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    subset = _make_avg_map_spectra()
    pdf_path = tmp_path / "overview.pdf"
    output_path = tmp_path / "plots" / "avg_stack" / "groups" / "plot.png"

    with PdfPages(pdf_path) as pdf:
        rps.plot_grouped_spectra(
            subset,
            value_col="mean_spectrum",
            title_suffix="raw",
            y_label="Intensity",
            group_col="group",
            panel_order=("hBN",),
            output_path=output_path,
            pdf=pdf,
        )

    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    assert not output_path.exists()
    assert output_path.with_suffix(".csv").exists()


def test_save_cut_pixel_map_slice_appends_to_pdf_instead_of_png(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    wavenumber = np.array([100.0, 200.0, 300.0])
    cube = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    parsed_item = {
        "path": Path("map.txt"),
        "wavenumber_cm1": wavenumber,
        "corrected_spectra_cube": cube,
    }
    pdf_path = tmp_path / "cutpixel_map.pdf"
    output_path = tmp_path / "01_map_200p0cm-1.png"

    with PdfPages(pdf_path) as pdf:
        used_wavenumber = rpm._save_cut_pixel_map_slice(
            parsed_item=parsed_item,
            output_path=output_path,
            color_scale_wavenumber_cm1=210.0,
            pdf=pdf,
        )

    assert used_wavenumber == 200.0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    assert not output_path.exists()


def test_build_pipeline_report_writes_html_with_expected_sections(tmp_path: Path):
    stage_tables = {
        "Stage 1: Parsed Files Summary": pd.DataFrame({"file": ["a.txt"], "pixels": [4]}),
        "Stage 5: Baseline-Corrected Summary": pd.DataFrame(),
    }
    plot_links = {"Stage 6 spectra overview": tmp_path / "01_plots" / "stage6_spectra_overview.pdf"}
    csv_links = {"Peak ratio table": tmp_path / "03_tables" / "peak_ratio.csv"}

    report_path = rrep.build_pipeline_report(
        output_dir=tmp_path,
        sample_name="S6_sample",
        stage_tables=stage_tables,
        plot_links=plot_links,
        csv_links=csv_links,
    )

    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "Stage 1: Parsed Files Summary" in html
    assert "Stage 6 spectra overview" in html
    assert "Peak ratio table" in html
    assert "No data." in html  # empty Stage 5 DataFrame


def test_plot_export_directories_are_numbered_one_to_six(tmp_path: Path):
    plot_parts = (
        "avg_stack",
        "norm_stack",
        "norm_overlap",
        "peak_ratio",
        "cutpixel_map",
        "despiked_baseline_anchor_stack",
    )

    directory_names = [
        rpaths._prepare_export_subdir(tmp_path, "plots", part).name
        for part in plot_parts
    ]

    assert directory_names == [
        "01_avg_stack",
        "02_norm_stack",
        "03_norm_overlap",
        "04_peak_ratio",
        "05_cutpixel_map",
        "06_despiked_baseline_anchor_stack",
    ]
