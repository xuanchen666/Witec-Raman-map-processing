"""2D map-slice and despiked/baseline anchor-stack plotting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages


def _compute_stack_step(
    spectra: list[np.ndarray],
    stack_scale: float,
    stack_extra_gap: float,
) -> float:
    """Estimate vertical spacing between stacked traces.

    Uses a robust span (95th - 5th percentile) per spectrum to avoid one
    outlier peak dominating the spacing for all curves.
    """
    if not spectra:
        return 1.0

    spans = []
    for spectrum in spectra:
        arr = np.asarray(spectrum, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        spans.append(float(np.percentile(finite, 95) - np.percentile(finite, 5)))

    if not spans:
        return 1.0

    base_step = float(np.median(spans))
    return max(base_step * float(stack_scale) + float(stack_extra_gap), 1e-12)


def _compute_cut_pixel_map_slice_data(
    parsed_item: dict,
    *,
    spectrum_key: str = "corrected_spectra_cube",
    color_scale_wavenumber_cm1: float = 562.0,
) -> dict[str, Any]:
    """Compute the map slice image and average-pixel overlay at the nearest target wavenumber."""
    cube = np.asarray(parsed_item[spectrum_key], dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D spectra cube for '{spectrum_key}', got shape {cube.shape}")

    wavenumber = np.asarray(parsed_item["wavenumber_cm1"], dtype=float)
    if wavenumber.ndim != 1 or wavenumber.size == 0:
        raise ValueError("wavenumber_cm1 must be a non-empty 1D array")

    wn_idx = int(np.argmin(np.abs(wavenumber - float(color_scale_wavenumber_cm1))))
    used_wavenumber = float(wavenumber[wn_idx])

    map_image = cube[:, :, wn_idx]
    map_image_display = map_image.T

    selected_rows: np.ndarray | None = None
    selected_cols: np.ndarray | None = None
    average_pixel_mask = parsed_item.get("average_pixel_mask")
    if average_pixel_mask is not None:
        mask = np.asarray(average_pixel_mask, dtype=bool)
        if mask.ndim == 2 and mask.shape == map_image.shape and np.any(mask):
            selected_rows, selected_cols = np.nonzero(mask)

    return {
        "wn_idx": wn_idx,
        "used_wavenumber": used_wavenumber,
        "map_image": map_image,
        "map_image_display": map_image_display,
        "selected_rows": selected_rows,
        "selected_cols": selected_cols,
    }


def _compute_despiked_baseline_anchor_stack_data(
    parsed_item: dict,
    *,
    despiked_key: str = "spectra_cube",
    baseline_key: str = "baseline_cube",
    anchor_mask_key: str = "noiseaware_anchor_mask_cube",
    stack_scale: float = 1.35,
    stack_extra_gap: float = 0.1,
) -> dict[str, Any]:
    """Compute retained-pixel traces, offsets, and anchors for the despiked/baseline stack plot."""
    if despiked_key not in parsed_item:
        return {"status": "missing_despiked_cube"}
    if baseline_key not in parsed_item:
        return {"status": "missing_baseline_cube"}

    wavenumber = np.asarray(parsed_item["wavenumber_cm1"], dtype=float)
    despiked_cube = np.asarray(parsed_item[despiked_key], dtype=float)
    baseline_cube = np.asarray(parsed_item[baseline_key], dtype=float)

    if despiked_cube.ndim != 3 or baseline_cube.ndim != 3:
        return {"status": "invalid_cube_shape"}

    if despiked_cube.shape != baseline_cube.shape or despiked_cube.shape[2] != wavenumber.size:
        return {"status": "shape_mismatch"}

    keep_mask_raw = parsed_item.get("spectrum_keep_mask")
    if keep_mask_raw is not None:
        keep_mask = np.asarray(keep_mask_raw, dtype=bool)
    else:
        keep_mask = np.any(np.isfinite(despiked_cube), axis=2)

    if keep_mask.shape != despiked_cube.shape[:2]:
        keep_mask = np.any(np.isfinite(despiked_cube), axis=2)

    retained_indices = np.argwhere(keep_mask)
    if retained_indices.size == 0:
        return {"status": "no_retained_pixels"}

    spectra_for_step = [
        np.asarray(despiked_cube[int(row_index), int(col_index), :], dtype=float)
        for row_index, col_index in retained_indices
    ]
    offset_step = _compute_stack_step(
        spectra=spectra_for_step,
        stack_scale=stack_scale,
        stack_extra_gap=stack_extra_gap,
    )

    anchor_mask_cube = None
    if anchor_mask_key in parsed_item:
        candidate_anchor_mask_cube = np.asarray(parsed_item[anchor_mask_key], dtype=bool)
        if candidate_anchor_mask_cube.shape == despiked_cube.shape:
            anchor_mask_cube = candidate_anchor_mask_cube

    total_anchors = 0
    y_min = np.inf
    y_max = -np.inf
    pixel_traces: list[dict[str, Any]] = []
    for stack_index, (row_index, col_index) in enumerate(retained_indices):
        row_i = int(row_index)
        col_i = int(col_index)
        offset = float(stack_index) * float(offset_step)

        despiked = np.asarray(despiked_cube[row_i, col_i, :], dtype=float)
        baseline = np.asarray(baseline_cube[row_i, col_i, :], dtype=float)
        finite_signal_mask = np.isfinite(wavenumber) & np.isfinite(despiked)
        finite_baseline_mask = np.isfinite(wavenumber) & np.isfinite(baseline)

        if np.any(finite_signal_mask):
            y_values = despiked[finite_signal_mask] + offset
            y_min = min(y_min, float(np.nanmin(y_values)))
            y_max = max(y_max, float(np.nanmax(y_values)))
        if np.any(finite_baseline_mask):
            y_values = baseline[finite_baseline_mask] + offset
            y_min = min(y_min, float(np.nanmin(y_values)))
            y_max = max(y_max, float(np.nanmax(y_values)))

        finite_anchor_mask = np.zeros_like(finite_signal_mask, dtype=bool)
        anchor_x = np.asarray([], dtype=float)
        anchor_y = np.asarray([], dtype=float)
        if anchor_mask_cube is not None:
            anchor_mask = np.asarray(anchor_mask_cube[row_i, col_i, :], dtype=bool)
            finite_anchor_mask = anchor_mask & finite_signal_mask
            if np.any(finite_anchor_mask):
                total_anchors += int(np.count_nonzero(finite_anchor_mask))
                anchor_x = wavenumber[finite_anchor_mask]
                anchor_y = despiked[finite_anchor_mask] + offset

        pixel_traces.append(
            {
                "row_index": row_i,
                "col_index": col_i,
                "stack_index": stack_index,
                "offset": offset,
                "wavenumber": wavenumber,
                "despiked": despiked,
                "baseline": baseline,
                "finite_signal_mask": finite_signal_mask,
                "finite_baseline_mask": finite_baseline_mask,
                "anchor_mask": finite_anchor_mask,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
            }
        )

    return {
        "status": "ok",
        "retained_indices": retained_indices,
        "offset_step": offset_step,
        "pixel_traces": pixel_traces,
        "total_anchors": total_anchors,
        "y_min": y_min,
        "y_max": y_max,
    }


def _save_cut_pixel_map_slice(
    parsed_item: dict,
    output_path: Path,
    *,
    spectrum_key: str = "corrected_spectra_cube",
    color_scale_wavenumber_cm1: float = 562.0,
    pdf: "PdfPages | None" = None,
) -> float:
    """Save a map slice image at the nearest target wavenumber and return the used value."""
    import matplotlib.pyplot as plt

    slice_data = _compute_cut_pixel_map_slice_data(
        parsed_item,
        spectrum_key=spectrum_key,
        color_scale_wavenumber_cm1=color_scale_wavenumber_cm1,
    )
    used_wavenumber = slice_data["used_wavenumber"]

    fig, map_ax = plt.subplots(figsize=(7.2, 6.0))
    im = map_ax.imshow(slice_data["map_image_display"], origin="upper", cmap="viridis", aspect="equal")

    if slice_data["selected_rows"] is not None:
        map_ax.scatter(
            slice_data["selected_rows"],
            slice_data["selected_cols"],
            s=45,
            facecolors="none",
            edgecolors="white",
            linewidths=1.2,
            label="Average pixels",
        )
        legend_handles, legend_labels = map_ax.get_legend_handles_labels()
        map_ax.legend(
            legend_handles,
            legend_labels,
            fontsize=8,
            loc="upper right",
            frameon=True,
        )

    map_ax.set_title(
        "Baseline corrected"
        f" | {parsed_item['path'].name}"
        f" | Color scale wavenumber: {used_wavenumber:.2f} cm^-1"
    )
    map_ax.set_xlabel("X index (row)")
    map_ax.set_ylabel("Y index (column)")
    fig.colorbar(im, ax=map_ax, fraction=0.046, pad=0.04, label="Intensity")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf is not None:
        pdf.savefig(fig, dpi=200, bbox_inches="tight")
    else:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return used_wavenumber

