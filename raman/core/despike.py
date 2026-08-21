"""Cosmic-ray despiking for parsed Raman map collections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .filters import ParsedCollection, ParsedCollectionMutable


def _build_despike_exclude_mask(
    wavenumber: np.ndarray,
    exclude_regions_cm1: Sequence[tuple[float, float]] | None,
) -> np.ndarray:
    """Build a boolean mask of wavenumber points to leave untouched by despiking."""
    exclude_mask = np.zeros_like(wavenumber, dtype=bool)
    if not exclude_regions_cm1:
        return exclude_mask

    for region_start, region_end in exclude_regions_cm1:
        lower = min(float(region_start), float(region_end))
        upper = max(float(region_start), float(region_end))
        exclude_mask |= (wavenumber >= lower) & (wavenumber <= upper)

    return exclude_mask


def despike_parsed_collection(
    parsed_collection: ParsedCollection,
    neigh: int = 4,
    threshold: int = 3,
    exclude_regions_cm1: Sequence[tuple[float, float]] | None = None,
) -> ParsedCollectionMutable:
    """Apply rampy.despiking to every spectrum of every parsed map.

    ``exclude_regions_cm1`` restores the original (pre-despike) values inside
    the given wavenumber windows, so real sharp features there are not
    mistaken for spikes.
    """
    import rampy as rp

    despiked_collection: ParsedCollectionMutable = []

    for parsed in parsed_collection:
        wavenumber = np.asarray(parsed["wavenumber_cm1"], dtype=float)
        spectra_cube = parsed["spectra_cube"]
        despiked_cube = np.empty_like(spectra_cube, dtype=float)

        exclude_mask = _build_despike_exclude_mask(wavenumber, exclude_regions_cm1)

        # Iterate over every pixel spectrum in the current map cube.
        for row_index in range(spectra_cube.shape[0]):
            for col_index in range(spectra_cube.shape[1]):
                spectrum = spectra_cube[row_index, col_index, :]
                if not np.isfinite(spectrum).all():
                    despiked_cube[row_index, col_index, :] = spectrum
                    continue

                despiked_spectrum = rp.despiking(
                    x=wavenumber,
                    y=spectrum,
                    neigh=neigh,
                    threshold=int(threshold),
                )
                if np.any(exclude_mask):
                    despiked_spectrum = np.asarray(despiked_spectrum, dtype=float).copy()
                    despiked_spectrum[exclude_mask] = spectrum[exclude_mask]

                despiked_cube[row_index, col_index, :] = despiked_spectrum

        despiked_collection.append(
            {
                **parsed,
                "spectra_cube": despiked_cube,
                "despike_config": {
                    "neigh": int(neigh),
                    "threshold": int(threshold),
                    "exclude_regions_cm1": (
                        [tuple(map(float, region)) for region in exclude_regions_cm1]
                        if exclude_regions_cm1
                        else []
                    ),
                },
            }
        )

    return despiked_collection
