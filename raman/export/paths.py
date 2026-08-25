"""Filesystem-safe path/stem helpers shared by CSV export and plotting modules."""

from __future__ import annotations

import re
from pathlib import Path


_EXPORT_PART_PREFIXES = {
    "plots": "01_plots",
    "spectra": "02_spectra",
    "tables": "03_tables",
    "code_snapshot": "04_code_snapshot",
    "report": "05_report",
    "avg_stack": "01_avg_stack",
    "norm_stack": "02_norm_stack",
    "norm_overlap": "03_norm_overlap",
    "peak_ratio": "04_peak_ratio",
    "cutpixel_map": "05_cutpixel_map",
    "despiked_baseline_anchor_stack": "06_despiked_baseline_anchor_stack",
    "groups": "01_groups",
}


def _prepare_export_subdir(output_dir: Path, *parts: str) -> Path:
    """Create and return a nested export directory."""
    numbered_parts = [_EXPORT_PART_PREFIXES.get(part, part) for part in parts]
    target_dir = output_dir.joinpath(*numbered_parts)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _coerce_optional_wavenumber_bound(value: object) -> float | None:
    """Convert a configured wavenumber bound to float while preserving None."""
    if value is None:
        return None
    return float(value)


def _format_wavenumber_bound_for_stem(value: float | None) -> str:
    """Return a filesystem-safe token for one wavenumber bound."""
    if value is None:
        return "auto"

    numeric_value = float(value)
    if float(numeric_value).is_integer():
        return str(int(numeric_value))

    return f"{numeric_value:g}".replace("-", "m").replace(".", "p")


def _sanitize_export_stem(file_name: str) -> str:
    """Build a filesystem-safe file stem from the source Raman file name."""
    stem = Path(file_name).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def _build_wavenumber_range_stem(
    wavenumber_min: float | None,
    wavenumber_max: float | None,
    label: str | None = None,
) -> str:
    """Build an export suffix that identifies one wavenumber window."""
    if label is not None and str(label).strip():
        safe_label = _sanitize_export_stem(str(label).strip())
        if safe_label:
            return safe_label

    return (
        f"wn_{_format_wavenumber_bound_for_stem(wavenumber_min)}_"
        f"{_format_wavenumber_bound_for_stem(wavenumber_max)}cm-1"
    )


def _prefix_indexed_stem(stem: str, index: int) -> str:
    """Prefix a sanitized stem with a stable two-digit logical order."""
    safe_stem = _sanitize_export_stem(stem)
    return f"{int(index):02d}_{safe_stem}"

