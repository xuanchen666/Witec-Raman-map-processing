"""File parsing and filename-derived metadata (group/subgroup/date/sample name).

Merges the former raman_parser.py (file reading) with the metadata-extraction
helpers previously duplicated across raman_processing_utils.py and
raman_map_analysis.py, since both operate on raw file identity with zero
plotting dependency.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd


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


_DATE_PATTERN = re.compile(r"(\d{8})")


_LASER_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:mw|w)", flags=re.IGNORECASE)


def _normalize_substrate_token(token: str) -> str | None:
    """Map naming tokens to canonical group labels."""
    lowered = token.strip().lower()
    if lowered == "au":
        return "Au"
    if lowered == "ro":
        return "RO"
    if lowered == "hbn":
        return "hBN"
    return None


def _is_numeric_suffix(token: str) -> bool:
    """Return true for integer suffixes used in names like hBN_1."""
    return bool(re.fullmatch(r"\d+", token.strip()))


def _is_laser_power_token(token: str) -> bool:
    """Return true for laser power tokens like 10mW or 0.5W."""
    return bool(_LASER_PATTERN.fullmatch(token.strip()))


def _split_stem_tokens(file_name: str) -> list[str]:
    """Split file stem into underscore-separated naming tokens."""
    stem = Path(file_name).stem
    return [token for token in stem.split("_") if token]


def _strip_trailing_metadata_tokens(tokens: list[str]) -> list[str]:
    """Remove trailing date and laser-power tokens from a filename token list."""
    stripped_tokens = list(tokens)

    if stripped_tokens and re.fullmatch(r"\d{8}", stripped_tokens[-1]):
        stripped_tokens.pop()

    if stripped_tokens and _is_laser_power_token(stripped_tokens[-1]):
        stripped_tokens.pop()

    return stripped_tokens


def extract_group(file_name: str) -> str:
    """Classify a file into Au, RO, hBN, or Other from naming tokens."""
    tokens = _split_stem_tokens(file_name)
    for token in tokens:
        normalized = _normalize_substrate_token(token)
        if normalized is not None:
            return normalized
    return "Other"


def extract_subgroup(file_name: str) -> str:
    """Return a comparison subgroup label, preserving hBN suffixes when present."""
    group = extract_group(file_name)
    if group != "hBN":
        return group

    tokens = _strip_trailing_metadata_tokens(_split_stem_tokens(file_name))
    for index, token in enumerate(tokens):
        normalized = _normalize_substrate_token(token)
        if normalized != "hBN":
            continue

        subgroup_suffix_tokens = tokens[index + 1 :]
        if not subgroup_suffix_tokens:
            return "hBN"

        return "hBN_" + "_".join(subgroup_suffix_tokens)

    return "hBN"


def extract_date(file_name: str) -> pd.Timestamp:
    """Extract YYYYMMDD date token from a file name."""
    match = _DATE_PATTERN.search(file_name)
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
    return pd.NaT


def _derive_sample_candidate(file_name: str, group_name: str | None = None) -> str:
    """Extract sample name from pattern: sample_substrate(_x)?_laser?(optional)_date."""
    tokens = _split_stem_tokens(file_name)
    if not tokens:
        return ""

    tokens = _strip_trailing_metadata_tokens(tokens)
    if not tokens:
        return ""

    # Find substrate token location so any hBN suffix stays out of the sample name.
    substrate_index: int | None = None
    for index, token in enumerate(tokens):
        normalized = _normalize_substrate_token(token)
        if normalized is None:
            continue

        substrate_index = index
        break

    if substrate_index is None:
        stem = "_".join(tokens)
        return stem.strip("_-")

    sample_tokens = tokens[:substrate_index]
    if sample_tokens:
        return "_".join(sample_tokens).strip("_-")

    # Fallback if the first token itself is the substrate marker.
    return "sample"


def infer_sample_name(
    avg_map_spectra: pd.DataFrame,
    fallback: str = "sample",
) -> str:
    """Infer a representative sample name from averaged map metadata."""
    if avg_map_spectra.empty or "file" not in avg_map_spectra.columns:
        return fallback

    candidates: list[str] = []
    for _, row in avg_map_spectra.iterrows():
        file_name = str(row.get("file", ""))
        group_name = str(row.get("group", "")) if "group" in avg_map_spectra.columns else None
        candidate = _derive_sample_candidate(file_name, group_name=group_name)
        if candidate:
            candidates.append(candidate)

    if not candidates:
        return fallback

    return Counter(candidates).most_common(1)[0][0]


def _extract_sample_code(sample_name: str | None) -> str:
    """Extract compact sample code token (e.g., S6) from sample name text."""
    if sample_name is None:
        return ""

    match = re.search(r"\bS\d+\b", str(sample_name), flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(0).upper()

