"""Headless tests for the pure-compute helpers in raman/plotting/spectra.py.

These tests must not require matplotlib: only numpy/pandas plus the target
module are imported so batch/CI environments without a display can run them.
"""

import numpy as np
import pandas as pd

import raman.plotting.spectra as rps


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

    data = rps._compute_grouped_spectra_data(
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

    data = rps._compute_grouped_spectra_data(
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
