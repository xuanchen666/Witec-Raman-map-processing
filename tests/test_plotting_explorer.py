"""Headless tests for the pure-compute helpers in raman/plotting/explorer.py.

These tests must not require matplotlib: only numpy plus the target module
are imported so batch/CI environments without a display can run them.
"""

import sys

import numpy as np

import raman.plotting.explorer as rpe


def test_importing_modules_does_not_pull_in_matplotlib():
    assert "matplotlib" not in sys.modules


def test_move_selected_pixel_uses_visual_arrow_key_directions():
    assert rpe._move_selected_pixel(2, 3, "left", max_row_index=4, max_col_index=5) == (1, 3)
    assert rpe._move_selected_pixel(2, 3, "right", max_row_index=4, max_col_index=5) == (3, 3)
    assert rpe._move_selected_pixel(2, 3, "up", max_row_index=4, max_col_index=5) == (2, 2)
    assert rpe._move_selected_pixel(2, 3, "down", max_row_index=4, max_col_index=5) == (2, 4)


def test_move_selected_pixel_clamps_to_map_bounds_and_ignores_other_keys():
    assert rpe._move_selected_pixel(0, 0, "left", max_row_index=4, max_col_index=5) == (0, 0)
    assert rpe._move_selected_pixel(0, 0, "up", max_row_index=4, max_col_index=5) == (0, 0)
    assert rpe._move_selected_pixel(4, 5, "right", max_row_index=4, max_col_index=5) == (4, 5)
    assert rpe._move_selected_pixel(4, 5, "down", max_row_index=4, max_col_index=5) == (4, 5)
    assert rpe._move_selected_pixel(2, 3, "a", max_row_index=4, max_col_index=5) == (2, 3)


def test_compute_pixel_spectrum_comparison_data_basic():
    wavenumber = np.linspace(200.0, 800.0, 10)
    corrected = np.arange(10, dtype=float)
    parsed_item = {
        "wavenumber_cm1": wavenumber,
        "corrected_spectra_cube": corrected.reshape(1, 1, 10),
    }

    data = rpe._compute_pixel_spectrum_comparison_data(
        parsed_item,
        row_index=0,
        col_index=0,
        show_previous_overlay=False,
        show_baseline=False,
        show_noiseaware_anchors=False,
    )

    np.testing.assert_array_equal(data["wavenumber"], wavenumber)
    np.testing.assert_array_equal(data["selected_spectrum"], corrected)
    assert data["previous_trace"] is None
    assert data["baseline_trace"] is None
    assert data["anchor_x"].size == 0


def test_compute_pixel_spectrum_comparison_data_previous_overlay_fallback():
    wavenumber = np.linspace(200.0, 800.0, 10)
    corrected = np.arange(10, dtype=float)
    previous = corrected * 2
    parsed_item = {
        "wavenumber_cm1": wavenumber,
        "corrected_spectra_cube": corrected.reshape(1, 1, 10),
        "spectra_cube": previous.reshape(1, 1, 10),
    }

    data = rpe._compute_pixel_spectrum_comparison_data(
        parsed_item,
        row_index=0,
        col_index=0,
        show_previous_overlay=True,
        show_baseline=False,
    )

    assert data["previous_trace"] is not None
    np.testing.assert_array_equal(data["previous_trace"]["intensity"], previous)


def test_compute_pixel_spectrum_comparison_data_baseline_trace():
    wavenumber = np.linspace(200.0, 800.0, 5)
    corrected = np.zeros(5)
    baseline = np.ones(5)
    parsed_item = {
        "wavenumber_cm1": wavenumber,
        "corrected_spectra_cube": corrected.reshape(1, 1, 5),
        "baseline_cube": baseline.reshape(1, 1, 5),
        "baseline_method": "mor",
    }

    data = rpe._compute_pixel_spectrum_comparison_data(
        parsed_item,
        row_index=0,
        col_index=0,
        show_previous_overlay=False,
        show_baseline=True,
    )

    assert data["baseline_trace"]["label"] == "MOR baseline"
    np.testing.assert_array_equal(data["baseline_trace"]["intensity"], baseline)


def test_compute_pixel_spectrum_comparison_data_anchor_mask_fallback():
    wavenumber = np.linspace(200.0, 800.0, 6)
    spectra_cube = np.arange(6, dtype=float).reshape(1, 1, 6)
    anchor_mask = np.zeros(6, dtype=bool)
    anchor_mask[[1, 3]] = True
    parsed_item = {
        "wavenumber_cm1": wavenumber,
        "corrected_spectra_cube": spectra_cube,
        "spectra_cube": spectra_cube,
        "baseline_method": "noiseaware",
        "noiseaware_anchor_mask_cube": anchor_mask.reshape(1, 1, 6),
    }

    data = rpe._compute_pixel_spectrum_comparison_data(
        parsed_item,
        row_index=0,
        col_index=0,
        show_previous_overlay=False,
        show_baseline=False,
        show_noiseaware_anchors=True,
    )

    assert data["anchor_x"].size == 2
    np.testing.assert_array_equal(data["anchor_x"], wavenumber[[1, 3]])


def test_build_map_image_modes():
    cube = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    slice_image = rpe._build_map_image(cube, "slice", wn_idx=1)
    np.testing.assert_array_equal(slice_image, cube[:, :, 1])

    mean_image = rpe._build_map_image(cube, "mean", wn_idx=0)
    np.testing.assert_array_almost_equal(mean_image, np.nanmean(cube, axis=2))
