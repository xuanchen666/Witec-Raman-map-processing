"""Headless tests for the pure-compute helpers in raman_map_analysis.py.

These tests must not require matplotlib: only numpy/pandas plus the target
module are imported so batch/CI environments without a display can run them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import raman_map_analysis as rma


def _make_spectrum_row(file_name: str, date: str, values: np.ndarray) -> dict:
    wavenumber = np.linspace(200.0, 1800.0, values.size)
    return {
        "group": "hBN",
        "subgroup": None,
        "file": file_name,
        "date": pd.Timestamp(date),
        "wavenumber_cm1": wavenumber,
        "mean_spectrum": values,
    }


def test_compute_grouped_spectra_data_stacks_with_increasing_offset():
    rng = np.random.default_rng(0)
    base_values = rng.normal(size=50)
    rows = [
        _make_spectrum_row(f"a{i}.txt", "2026-01-01", base_values.copy())
        for i in range(3)
    ]
    subset = pd.DataFrame(rows).sort_values(["date", "file"])

    data = rma._compute_grouped_spectra_data(
        subset=subset,
        value_col="mean_spectrum",
        stack_scale=1.0,
        stack_extra_gap=0.0,
        wavenumber_min=None,
        wavenumber_max=None,
    )

    assert data["offset_step"] > 0
    assert len(data["stacked_traces"]) == 3
    assert len(data["labels"]) == 3
    # Later stack indices are shifted up by increasing multiples of offset_step.
    _, first_y = data["stacked_traces"][0]
    _, second_y = data["stacked_traces"][1]
    assert np.all(second_y - first_y == data["offset_step"])


def test_compute_grouped_spectra_data_respects_wavenumber_window():
    row = _make_spectrum_row("a.txt", "2026-01-01", np.linspace(0.0, 1.0, 100))
    subset = pd.DataFrame([row])

    data = rma._compute_grouped_spectra_data(
        subset=subset,
        value_col="mean_spectrum",
        stack_scale=1.0,
        stack_extra_gap=0.0,
        wavenumber_min=500.0,
        wavenumber_max=600.0,
    )

    window_x, _ = data["stacked_traces"][0]
    assert window_x.min() >= 500.0
    assert window_x.max() <= 600.0


def test_compute_cut_pixel_map_slice_data_picks_nearest_wavenumber():
    wavenumber = np.array([100.0, 200.0, 300.0, 400.0])
    cube = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4)
    parsed_item = {
        "path": Path("map.txt"),
        "wavenumber_cm1": wavenumber,
        "corrected_spectra_cube": cube,
    }

    data = rma._compute_cut_pixel_map_slice_data(
        parsed_item,
        spectrum_key="corrected_spectra_cube",
        color_scale_wavenumber_cm1=210.0,
    )

    assert data["wn_idx"] == 1
    assert data["used_wavenumber"] == 200.0
    np.testing.assert_array_equal(data["map_image"], cube[:, :, 1])
    assert data["selected_rows"] is None


def test_compute_cut_pixel_map_slice_data_reports_average_pixel_mask():
    wavenumber = np.array([100.0, 200.0])
    cube = np.zeros((2, 2, 2), dtype=float)
    mask = np.array([[True, False], [False, False]])
    parsed_item = {
        "path": Path("map.txt"),
        "wavenumber_cm1": wavenumber,
        "corrected_spectra_cube": cube,
        "average_pixel_mask": mask,
    }

    data = rma._compute_cut_pixel_map_slice_data(parsed_item, color_scale_wavenumber_cm1=100.0)

    assert data["selected_rows"] is not None
    np.testing.assert_array_equal(data["selected_rows"], [0])
    np.testing.assert_array_equal(data["selected_cols"], [0])


def test_compute_despiked_baseline_anchor_stack_data_missing_keys():
    result = rma._compute_despiked_baseline_anchor_stack_data({"path": Path("x.txt")})
    assert result["status"] == "missing_despiked_cube"


def test_compute_despiked_baseline_anchor_stack_data_ok():
    wavenumber = np.linspace(200.0, 800.0, 20)
    despiked_cube = np.ones((2, 1, 20), dtype=float)
    baseline_cube = np.zeros((2, 1, 20), dtype=float)
    anchor_mask_cube = np.zeros((2, 1, 20), dtype=bool)
    anchor_mask_cube[0, 0, 5] = True

    parsed_item = {
        "path": Path("map.txt"),
        "wavenumber_cm1": wavenumber,
        "spectra_cube": despiked_cube,
        "baseline_cube": baseline_cube,
        "noiseaware_anchor_mask_cube": anchor_mask_cube,
    }

    data = rma._compute_despiked_baseline_anchor_stack_data(parsed_item)

    assert data["status"] == "ok"
    assert data["retained_indices"].shape[0] == 2
    assert data["offset_step"] > 0
    assert data["total_anchors"] == 1
    assert len(data["pixel_traces"]) == 2

    # Different stack_scale/stack_extra_gap (as used by the CSV exporter) must
    # change the offset_step but keep the same retained pixel set and anchors.
    csv_data = rma._compute_despiked_baseline_anchor_stack_data(
        parsed_item, stack_scale=1.0, stack_extra_gap=0.0
    )
    assert csv_data["status"] == "ok"
    assert csv_data["total_anchors"] == data["total_anchors"]
    assert csv_data["offset_step"] != data["offset_step"]
