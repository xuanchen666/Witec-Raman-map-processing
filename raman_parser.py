"""
Raman Area Scan Parser Module

This module provides functions to parse Raman .txt export files 
into a tidy table with one row per single spectrum, and 3D data cubes.
"""

from __future__ import annotations

import csv
import re
from io import StringIO
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd

# Regular expression to extract X and Y coordinates from column labels like "Spectrum (1/2)"
POSITION_RE = re.compile(r"\((\d+)/(\d+)\)$")


class ParsedRamanExport(TypedDict):
    """Structured output returned by parse_raman_export."""

    path: Path
    header: dict[str, str]
    column_labels: list[str]
    unit_labels: list[str]
    wavenumber_cm1: np.ndarray
    wide: pd.DataFrame
    spectra_2d: np.ndarray
    spectra_cube: np.ndarray
    tidy: pd.DataFrame


def _read_lines(path: Path) -> list[str]:
    """Read the file line by line with utf-8 encoding."""
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _split_tab_row(line: str) -> list[str]:
    """Split a tab-separated line into a list of strings."""
    return next(csv.reader([line], delimiter="\t", quotechar="'", skipinitialspace=False))


def _parse_header(lines: list[str]) -> dict[str, str]:
    """Parse the key-value settings from the header section before the [Data] tag."""
    header = {}
    for line in lines:
        if line.startswith("[Data]"):
            break
        if " = " in line:
            key, value = line.split(" = ", 1)
            header[key.strip()] = value.strip().strip("'")
    return header


def _parse_positions(labels: list[str]) -> pd.DataFrame:
    """Extract spatial indices (x and y) from the spectrum column labels."""
    records = []
    for idx, label in enumerate(labels):
        match = POSITION_RE.search(label)
        if match is None:
            records.append({"spectrum_index": idx, "label": label, "x_index": np.nan, "y_index": np.nan})
            continue
        records.append(
            {
                "spectrum_index": idx,
                "label": label,
                "x_index": int(match.group(1)),
                "y_index": int(match.group(2)),
            }
        )
    return pd.DataFrame.from_records(records)


def _validate_position_table(positions: pd.DataFrame, size_x: int, size_y: int, path: Path) -> None:
    """Validate that all spectrum labels map to one unique grid coordinate."""
    expected_positions = size_x * size_y
    if len(positions) != expected_positions:
        raise ValueError(
            f"Position count mismatch in {path.name}: got {len(positions)} positions, expected {expected_positions}"
        )
    if positions[["x_index", "y_index"]].isna().any().any():
        raise ValueError(f"Unparsed spectrum labels found in {path.name}")
    if positions[["x_index", "y_index"]].duplicated().any():
        raise ValueError(f"Duplicate spectrum positions found in {path.name}")


def parse_raman_export(path: Path) -> ParsedRamanExport:
    """
    Parse a single Raman .txt export file.
    
    Returns a dictionary containing the extracted metadata, 
    the raw data, a 3D spectra cube, and a tidy dataframe.
    """
    lines = _read_lines(path)
    try:
        data_idx = lines.index("[Data]")
    except ValueError as exc:
        raise ValueError(f"No [Data] section found in {path.name}") from exc

    header_info = _parse_header(lines[:data_idx])
    column_labels = _split_tab_row(lines[data_idx + 1])
    unit_labels = _split_tab_row(lines[data_idx + 2])
    size_x = int(header_info.get("SizeX", 0))
    size_y = int(header_info.get("SizeY", 0))

    numeric_block = "\n".join(line for line in lines[data_idx + 3 :] if line.strip())
    raw = pd.read_csv(StringIO(numeric_block), sep=r"\s+", header=None, engine="python")

    if raw.shape[1] != len(column_labels):
        raise ValueError(
            f"Column mismatch in {path.name}: data has {raw.shape[1]} columns, header has {len(column_labels)}"
        )

    wide = raw.copy()
    wide.columns = column_labels
    x_axis_name = wide.columns[0]
    wide = wide.rename(columns={x_axis_name: "wavenumber_cm1"})

    wavenumber_values = wide["wavenumber_cm1"].to_numpy(dtype=float)
    spectrum_columns = [col for col in wide.columns if col != "wavenumber_cm1"]
    spectra_2d = wide[spectrum_columns].to_numpy(dtype=float).T
    positions = _parse_positions(spectrum_columns)

    # This catches malformed exports before we build derived arrays.
    _validate_position_table(positions=positions, size_x=size_x, size_y=size_y, path=path)

    spectra_cube = np.empty((size_x, size_y, len(wavenumber_values)), dtype=float)
    for spectrum_index, row in enumerate(positions.itertuples(index=False)):
        spectra_cube[row.x_index, row.y_index, :] = wide[spectrum_columns[spectrum_index]].to_numpy(dtype=float)

    tidy = wide.melt(id_vars="wavenumber_cm1", var_name="spectrum_label", value_name="intensity")
    tidy["source_file"] = path.name
    tidy = tidy.merge(positions, left_on="spectrum_label", right_on="label", how="left")
    tidy = tidy.drop(columns=["label"])

    return {
        "path": path,
        "header": header_info,
        "column_labels": column_labels,
        "unit_labels": unit_labels,
        "wavenumber_cm1": wavenumber_values,
        "wide": wide,
        "spectra_2d": spectra_2d,
        "spectra_cube": spectra_cube,
        "tidy": tidy,
    }


def single_spectrum_frame(parsed: ParsedRamanExport, x_index: int, y_index: int) -> pd.DataFrame:
    """Return one spectrum as a two-column DataFrame for downstream baseline correction."""
    return pd.DataFrame(
        {
            "wavenumber_cm1": parsed["wavenumber_cm1"],
            "intensity": parsed["spectra_cube"][x_index, y_index, :],
        }
    )
