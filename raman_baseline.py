"""Baseline correction for Raman spectra: MOR/airPLS/poly/rolling_ball dispatch plus noise-aware fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from pybaselines import Baseline
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.ndimage import binary_dilation, label, percentile_filter, uniform_filter1d
from scipy.signal import find_peaks

# Shared aliases used throughout this module to make function signatures easier to read.
ParsedMap = Mapping[str, Any]
ParsedMapMutable = dict[str, Any]
ParsedCollection = Sequence[ParsedMap]
ParsedCollectionMutable = list[ParsedMapMutable]


def _normalize_anchor_indices(
    n_points: int,
    forced_anchor_indices: int | list[int] | tuple[int, ...] | None,
) -> list[int]:
    """Normalize forced anchor index input to sorted unique in-range indices."""
    if forced_anchor_indices is None:
        return []

    if isinstance(forced_anchor_indices, (int, np.integer)):
        raw_indices = [int(forced_anchor_indices)]
    else:
        raw_indices = []
        for value in forced_anchor_indices:
            if isinstance(value, (int, np.integer)):
                raw_indices.append(int(value))

    unique_sorted = sorted(set(raw_indices))
    return [index for index in unique_sorted if 0 <= index < int(n_points)]


def _normalize_anchor_wavenumbers_to_indices(
    x: np.ndarray,
    forced_anchor_wavenumbers: float | list[float] | tuple[float, ...] | None,
) -> list[int]:
    """Map requested anchor wavenumbers to nearest valid x-axis indices."""
    if forced_anchor_wavenumbers is None:
        return []

    if isinstance(forced_anchor_wavenumbers, (int, float, np.integer, np.floating)):
        raw_wavenumbers = [float(forced_anchor_wavenumbers)]
    else:
        raw_wavenumbers = []
        for value in forced_anchor_wavenumbers:
            if isinstance(value, (int, float, np.integer, np.floating)):
                raw_wavenumbers.append(float(value))

    if not raw_wavenumbers:
        return []

    x = np.asarray(x, dtype=float)
    finite_x_mask = np.isfinite(x)
    if not np.any(finite_x_mask):
        return []

    finite_indices = np.flatnonzero(finite_x_mask)
    finite_x = x[finite_x_mask]

    resolved_indices: list[int] = []
    for target_wavenumber in raw_wavenumbers:
        if not np.isfinite(target_wavenumber):
            continue
        nearest_in_finite = int(np.argmin(np.abs(finite_x - target_wavenumber)))
        resolved_indices.append(int(finite_indices[nearest_in_finite]))

    return sorted(set(resolved_indices))


def _exclude_forced_anchors_in_peak_regions(
    forced_anchor_indices: list[int],
    explicit_peak_mask: np.ndarray,
) -> list[int]:
    """Drop forced anchors that fall inside user-defined explicit peak regions."""
    if not forced_anchor_indices:
        return []
    if explicit_peak_mask.size == 0:
        return forced_anchor_indices
    return [index for index in forced_anchor_indices if not bool(explicit_peak_mask[index])]


def _mad_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)) + 1e-12)


def _rolling_mad(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, float)
    window = max(7, int(window) | 1)
    half_window = window // 2
    output = np.empty_like(values)
    for index in range(len(values)):
        lower = max(0, index - half_window)
        upper = min(len(values), index + half_window + 1)
        output[index] = _mad_sigma(values[lower:upper])
    return output


def _low_quantile_hint(y: np.ndarray, q_low: float, smooth_win: int) -> np.ndarray:
    quantile = float(np.clip(q_low, 0, 50))
    baseline = percentile_filter(
        y,
        size=max(5, smooth_win),
        percentile=quantile,
        mode="nearest",
    )
    return uniform_filter1d(
        baseline,
        size=max(3, smooth_win // 2),
        mode="nearest",
    )


def _region_mask(
    x: np.ndarray,
    regions: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None,
) -> np.ndarray:
    """Return a boolean mask covering the explicitly provided x-axis regions."""
    mask = np.zeros_like(x, dtype=bool)
    if not regions:
        return mask

    for region_start, region_end in regions:
        lower = min(float(region_start), float(region_end))
        upper = max(float(region_start), float(region_end))
        mask |= (x >= lower) & (x <= upper)
    return mask


def _adaptive_noise_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    q_low: float = 10,
    smooth_win: int = 11,
    mad_win: int = 31,
    z_keep: float = 2.5,
    min_region: int = 15,
    expand_runs: int = 5,
    peak_regions: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an adaptive noise-only mask using local robust thresholds."""
    y = np.asarray(y, float)
    excluded_regions = _region_mask(x, peak_regions)
    hint = _low_quantile_hint(y, q_low=q_low, smooth_win=smooth_win)
    residual = y - hint
    sigma = _rolling_mad(residual, max(9, mad_win | 1))
    z_score = np.abs(residual) / sigma

    global_median = np.median(z_score)
    local_scale = uniform_filter1d(
        z_score,
        size=max(5, mad_win // 3),
        mode="nearest",
    )
    adaptive_threshold = z_keep * np.clip(
        local_scale / (global_median + 1e-12),
        0.7,
        1.5,
    )
    preliminary = z_score <= adaptive_threshold
    preliminary[excluded_regions] = False
    preliminary = binary_dilation(preliminary, iterations=max(1, expand_runs))
    preliminary[excluded_regions] = False

    label_result = cast(tuple[np.ndarray, int], label(preliminary))
    labels = label_result[0]
    region_count = int(label_result[1])
    mask = np.zeros_like(preliminary, dtype=bool)
    for region in range(1, region_count + 1):
        indices = np.where(labels == region)[0]
        if len(indices) >= int(min_region):
            mask[indices] = True
    return mask, hint


def _suppress_true_peaks(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    hint: np.ndarray | None = None,
    smooth_win: int = 9,
    prom_k: float = 3.0,
    min_width_pts: int = 5,
    extra_dilate: int = 5,
    peak_regions: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
) -> np.ndarray:
    if hint is None:
        hint = uniform_filter1d(
            y,
            size=max(3, smooth_win | 1),
            mode="nearest",
        )

    residual = y - hint
    smoothed_residual = uniform_filter1d(
        residual,
        size=max(3, smooth_win | 1),
        mode="nearest",
    )
    sigma = _mad_sigma(smoothed_residual[mask]) if np.any(mask) else _mad_sigma(smoothed_residual)
    prominence_threshold = max(prom_k * sigma, 1e-9)

    peaks, _ = find_peaks(smoothed_residual, prominence=prominence_threshold)
    half_threshold = 0.5 * prominence_threshold
    for peak in peaks:
        lower = peak
        while lower > 0 and smoothed_residual[lower] > half_threshold:
            lower -= 1
        upper = peak
        while upper < len(y) - 1 and smoothed_residual[upper] > half_threshold:
            upper += 1
        lower = max(0, lower - extra_dilate)
        upper = min(len(y) - 1, upper + extra_dilate)
        if upper - lower + 1 >= max(int(min_width_pts), 3):
            mask[lower : upper + 1] = False

    explicit_regions = _region_mask(x, peak_regions)
    if np.any(explicit_regions):
        mask[explicit_regions] = False
    return mask


def _anchors_from_mask(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    label_result = cast(tuple[np.ndarray, int], label(mask))
    labels = label_result[0]
    region_count = int(label_result[1])
    anchor_x: list[float] = []
    anchor_y: list[float] = []
    spans: list[tuple[int, int]] = []
    for region in range(1, region_count + 1):
        indices = np.where(labels == region)[0]
        if indices.size:
            anchor_x.append(float(np.median(x[indices])))
            anchor_y.append(float(np.median(y[indices])))
            spans.append((int(indices[0]), int(indices[-1])))
    return np.array(anchor_x), np.array(anchor_y), spans


def _fit_spline(
    anchor_x: np.ndarray,
    anchor_y: np.ndarray,
    x_eval: np.ndarray,
    s_scale: float = 0.5,
    interp_mode: str = "spline",
) -> np.ndarray:
    anchor_x = np.asarray(anchor_x, float)
    anchor_y = np.asarray(anchor_y, float)
    if anchor_x.size < 2:
        fill_value = float(np.median(anchor_y) if anchor_y.size else 0.0)
        return np.full_like(x_eval, fill_value)

    order = np.argsort(anchor_x)
    anchor_x, anchor_y = anchor_x[order], anchor_y[order]
    for index in range(1, len(anchor_x)):
        if anchor_x[index] <= anchor_x[index - 1]:
            anchor_x[index] = anchor_x[index - 1] + 1e-6

    mode = str(interp_mode).strip().lower()
    if mode not in {"spline", "pchip"}:
        mode = "spline"

    smoothness = max(
        1e-12,
        (len(anchor_x) * np.var(anchor_y) + 1e-12) * s_scale,
    )
    try:
        if mode == "pchip":
            return np.asarray(PchipInterpolator(anchor_x, anchor_y)(x_eval), dtype=float)

        spline = UnivariateSpline(anchor_x, anchor_y, s=smoothness, k=3)
        return np.asarray(spline(x_eval), dtype=float)
    except Exception:
        return np.asarray(np.interp(x_eval, anchor_x, anchor_y), dtype=float)


def _robust_connecting_baseline(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    remove_k_sigma: float = 1.0,
    min_consecutive_remove: int = 2,
    max_iters: int = 3,
    fit_interp: str = "spline",
    forced_anchor_indices: int | list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    anchor_x, anchor_y, spans = _anchors_from_mask(x, y, mask)
    forced_indices = _normalize_anchor_indices(len(x), forced_anchor_indices)
    forced_flags = np.zeros_like(anchor_x, dtype=bool)

    if forced_indices:
        for forced_index in forced_indices:
            x_value = float(x[forced_index])
            y_value = float(y[forced_index])

            if anchor_x.size == 0:
                anchor_x = np.array([x_value], dtype=float)
                anchor_y = np.array([y_value], dtype=float)
                forced_flags = np.array([True], dtype=bool)
                spans.append((forced_index, forced_index))
                continue

            insert_at = int(np.searchsorted(anchor_x, x_value))
            if insert_at < anchor_x.size and abs(float(anchor_x[insert_at]) - x_value) <= 1e-12:
                anchor_y[insert_at] = y_value
                forced_flags[insert_at] = True
                continue

            anchor_x = np.insert(anchor_x, insert_at, x_value)
            anchor_y = np.insert(anchor_y, insert_at, y_value)
            forced_flags = np.insert(forced_flags, insert_at, True)
            spans.insert(insert_at, (forced_index, forced_index))

    if anchor_x.size == 0:
        fallback = _low_quantile_hint(y, q_low=10, smooth_win=31)
        return fallback, np.array([], bool), np.array([]), np.array([]), []

    edge_tolerance = (x[-1] - x[0]) * 0.02
    edge_points = max(5, len(y) // 40)
    if anchor_x[0] > x[0] + edge_tolerance:
        anchor_x = np.insert(anchor_x, 0, x[0])
        anchor_y = np.insert(anchor_y, 0, np.median(y[:edge_points]))
        forced_flags = np.insert(forced_flags, 0, False)
    if anchor_x[-1] < x[-1] - edge_tolerance:
        anchor_x = np.append(anchor_x, x[-1])
        anchor_y = np.append(anchor_y, np.median(y[-edge_points:]))
        forced_flags = np.append(forced_flags, False)

    use = np.ones_like(anchor_x, dtype=bool)
    removed_spans: list[tuple[int, int]] = []
    for _ in range(max_iters):
        baseline = _fit_spline(anchor_x[use], anchor_y[use], x, s_scale=0.5, interp_mode=fit_interp)
        residual = anchor_y - np.interp(anchor_x, x, baseline)
        over = (residual > remove_k_sigma * _mad_sigma(residual)) & (~forced_flags)
        over_indices = np.flatnonzero(over)
        if over_indices.size == 0:
            break

        groups: list[tuple[int, int]] = []
        group_start = previous = int(over_indices[0])
        for current in over_indices[1:]:
            current = int(current)
            if current == previous + 1:
                previous = current
            else:
                groups.append((group_start, previous))
                group_start = previous = current
        groups.append((group_start, previous))

        removed_in_iteration = False
        for lower, upper in groups:
            if upper - lower + 1 >= min_consecutive_remove:
                use[lower : upper + 1] = False
                if lower < len(spans):
                    removed_spans.extend(spans[lower : min(upper + 1, len(spans))])
                removed_in_iteration = True
        if not removed_in_iteration:
            break

    used_x = anchor_x[use]
    used_y = anchor_y[use]
    baseline = _fit_spline(used_x, used_y, x, s_scale=0.5, interp_mode=fit_interp)
    return baseline, use, used_x, used_y, removed_spans


def _regions_from_intersections(
    y: np.ndarray,
    baseline: np.ndarray,
    *,
    sigma_k: float = 0.6,
    min_width_pts: int = 20,
) -> list[tuple[int, int]]:
    residual = y - baseline
    over = residual > sigma_k * _mad_sigma(residual)
    indices = np.flatnonzero(over)
    if indices.size == 0:
        return []

    regions: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for current in indices[1:]:
        current = int(current)
        if current == previous + 1:
            previous = current
        else:
            if previous - start + 1 >= min_width_pts:
                regions.append((start, previous))
            start = previous = current
    if previous - start + 1 >= min_width_pts:
        regions.append((start, previous))
    return regions


def _smooth_baseline(
    baseline: np.ndarray,
    strength: float = 0.7,
    passes: int = 2,
) -> np.ndarray:
    baseline = np.asarray(baseline, float)
    window = int(max(5, min(len(baseline) // 50, (len(baseline) * strength) // 8))) | 1
    smoothed = baseline.copy()
    for _ in range(max(1, passes)):
        smoothed = uniform_filter1d(smoothed, size=window, mode="nearest")
    return smoothed


def _bridge_explicit_regions(
    x: np.ndarray,
    baseline: np.ndarray,
    peak_regions: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None,
    *,
    bridge_interp: str = "pchip",
) -> np.ndarray:
    """Replace baseline in explicit peak regions using the selected interpolation mode."""
    if not peak_regions:
        return baseline

    mode = str(bridge_interp).strip().lower()
    if mode not in {"linear", "pchip", "spline"}:
        mode = "pchip"

    bridged = np.asarray(baseline, float).copy()
    n_points = len(x)
    for region_start, region_end in peak_regions:
        lower = min(float(region_start), float(region_end))
        upper = max(float(region_start), float(region_end))
        start_index = int(np.searchsorted(x, lower, side="left"))
        end_index = int(np.searchsorted(x, upper, side="right") - 1)
        if start_index >= n_points or end_index < 0:
            continue
        start_index = max(0, start_index)
        end_index = min(n_points - 1, end_index)
        left_index = max(0, start_index - 1)
        right_index = min(n_points - 1, end_index + 1)

        if right_index <= left_index:
            bridged[start_index : end_index + 1] = float(bridged[left_index])
            continue

        left_prev_index = max(0, left_index - 1)
        right_next_index = min(n_points - 1, right_index + 1)

        control_indices = [left_prev_index, left_index, right_index, right_next_index]
        unique_control_indices: list[int] = []
        for index in control_indices:
            if not unique_control_indices or index != unique_control_indices[-1]:
                unique_control_indices.append(index)

        control_x = x[unique_control_indices]
        control_y = bridged[unique_control_indices]

        region_indices = np.arange(start_index, end_index + 1)
        region_positions = x[region_indices]

        if mode == "linear":
            bridge = np.interp(
                region_positions,
                [x[left_index], x[right_index]],
                [float(bridged[left_index]), float(bridged[right_index])],
            )
        elif control_x.size >= 3 and np.all(np.diff(control_x) > 0):
            if mode == "spline":
                bridge = _fit_spline(control_x, control_y, region_positions, s_scale=0.5)
            else:
                bridge = PchipInterpolator(control_x, control_y)(region_positions)
        else:
            bridge = np.interp(
                region_positions,
                [x[left_index], x[right_index]],
                [float(bridged[left_index]), float(bridged[right_index])],
            )
        bridged[start_index : end_index + 1] = bridge

    return bridged


def auto_baseline_noiseaware(
    x: np.ndarray,
    y: np.ndarray,
    *,
    q_low: float = 10,
    smooth_win: int = 11,
    mad_win: int = 31,
    z_keep: float = 2.2,
    min_region: int = 12,
    prom_k: float = 2.5,
    peak_dilate: int = 6,
    fit_mode: str = "auto",
    bias_strength: float = 0.5,
    pin_ends: bool = True,
    outdir: str | Path | None = None,
    name: str = "spectrum",
    debug: bool = False,
    peak_regions: list[tuple[float, float]] | tuple[tuple[float, float], ...] | None = None,
    remove_k_sigma: float = 1.0,
    fit_interp: str = "spline",
    bridge_interp: str = "pchip",
    forced_anchor_indices: int | list[int] | tuple[int, ...] | None = None,
    forced_anchor_wavenumbers: float | list[float] | tuple[float, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return corrected signal, baseline, and diagnostics for one spectrum."""
    del fit_mode, pin_ends, debug
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    explicit_peak_regions = peak_regions
    normalized_forced_anchor_indices = _normalize_anchor_indices(len(x), forced_anchor_indices)
    normalized_forced_anchor_indices_from_wavenumbers = _normalize_anchor_wavenumbers_to_indices(
        x,
        forced_anchor_wavenumbers,
    )
    normalized_forced_anchor_indices = sorted(
        set(normalized_forced_anchor_indices)
        | set(normalized_forced_anchor_indices_from_wavenumbers)
    )

    normalized_fit_interp = str(fit_interp).strip().lower()
    if normalized_fit_interp not in {"spline", "pchip"}:
        normalized_fit_interp = "spline"

    mask, hint = _adaptive_noise_mask(
        x,
        y,
        q_low=q_low,
        smooth_win=smooth_win,
        mad_win=mad_win,
        z_keep=z_keep,
        min_region=min_region,
        peak_regions=explicit_peak_regions,
    )
    mask = _suppress_true_peaks(
        x,
        y,
        mask,
        hint=hint,
        smooth_win=max(5, smooth_win // 2),
        prom_k=prom_k,
        extra_dilate=peak_dilate,
        peak_regions=explicit_peak_regions,
    )

    explicit_peak_mask = _region_mask(x, explicit_peak_regions)
    if np.any(explicit_peak_mask):
        mask[explicit_peak_mask] = False

    normalized_forced_anchor_indices = _exclude_forced_anchors_in_peak_regions(
        normalized_forced_anchor_indices,
        explicit_peak_mask,
    )

    baseline_initial, _, anchors_used_x, anchors_used_y, _ = _robust_connecting_baseline(
        x,
        y,
        mask,
        remove_k_sigma=remove_k_sigma,
        min_consecutive_remove=2,
        max_iters=3,
        fit_interp=normalized_fit_interp,
        forced_anchor_indices=normalized_forced_anchor_indices,
    )
    detected_peak_regions = _regions_from_intersections(
        y,
        baseline_initial,
        sigma_k=0.6,
        min_width_pts=max(10, len(x) // 100),
    )

    refined_mask = mask.copy()
    for lower, upper in detected_peak_regions:
        refined_mask[max(0, lower) : min(len(y) - 1, upper) + 1] = False

    residual = y - baseline_initial
    sigma_local = (
        _mad_sigma(residual[refined_mask])
        if np.any(refined_mask)
        else _mad_sigma(residual)
    )
    adaptive_bias = bias_strength * sigma_local
    baseline = np.minimum(baseline_initial, y - adaptive_bias)
    baseline = _bridge_explicit_regions(
        x,
        baseline,
        explicit_peak_regions,
        bridge_interp=bridge_interp,
    )
    baseline = _smooth_baseline(baseline, strength=0.7, passes=2)
    corrected = y - baseline

    if outdir is not None:
        import matplotlib.pyplot as plt

        output_directory = Path(outdir)
        output_directory.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(10, 3.4), dpi=150)
        axis.plot(x, y, "k-", linewidth=1.0, label="spectrum")
        axis.plot(x, baseline, "g--", linewidth=1.4, label=f"baseline (bias={adaptive_bias:.2f} sigma)")
        mask_indices = np.flatnonzero(refined_mask)
        if mask_indices.size:
            step = max(1, mask_indices.size // 300)
            selected = mask_indices[::step]
            axis.scatter(x[selected], y[selected], s=12, c="red", alpha=0.6, label="noise mask")
        axis.plot(x, baseline_initial, linewidth=1.0, alpha=0.6, color="#54c6d8", label="connecting baseline")
        for lower, upper in detected_peak_regions:
            axis.axvspan(x[max(0, lower)], x[min(len(x) - 1, upper)], color="orange", alpha=0.18, linewidth=0)
        for lower, upper in (explicit_peak_regions or []):
            region_left = min(lower, upper)
            region_right = max(lower, upper)
            axis.axvspan(region_left, region_right, color="purple", alpha=0.12, linewidth=0)
        axis.set_title(f"Noise mask & baseline: {name}")
        axis.set_xlabel("Raman shift (cm$^{-1}$)")
        axis.set_ylabel("Intensity (a.u.)")
        axis.legend(loc="upper right", frameon=True)
        figure.savefig(output_directory / f"{name}_noise_mask_debug.png", bbox_inches="tight")
        plt.close(figure)

    info = {
        "method": "noiseaware+robust-connect+adaptive",
        "adaptive_bias_sigma": adaptive_bias,
        "anchors_used": int(anchors_used_x.size),
        "anchor_x_pre_median": anchors_used_x.tolist(),
        "anchor_y_pre_median": anchors_used_y.tolist(),
        "explicit_peak_regions": [(float(lower), float(upper)) for lower, upper in (explicit_peak_regions or [])],
        "detected_peak_regions": [(int(lower), int(upper)) for lower, upper in detected_peak_regions],
        "explicit_peak_regions_bridged": bool(explicit_peak_regions),
        "fit_interp": normalized_fit_interp,
        "explicit_peak_bridge_interp": str(bridge_interp).strip().lower(),
        "forced_anchor_indices": normalized_forced_anchor_indices,
        "forced_anchor_wavenumbers": []
        if forced_anchor_wavenumbers is None
        else [float(value) for value in np.atleast_1d(forced_anchor_wavenumbers) if np.isfinite(value)],
    }
    return corrected, baseline, info


def _build_poly_weights(
    wavenumber: np.ndarray,
    mask_regions: Sequence[tuple[float, float]] | None,
) -> np.ndarray | None:
    """Build per-point weights for poly baseline using user-provided wavenumber regions."""
    if not mask_regions:
        return None

    weights = np.zeros_like(wavenumber, dtype=float)
    for region_start, region_end in mask_regions:
        lower = min(float(region_start), float(region_end))
        upper = max(float(region_start), float(region_end))
        weights[(wavenumber >= lower) & (wavenumber <= upper)] = 1.0

    if not np.any(weights > 0):
        raise ValueError("poly_mask_regions did not select any wavenumber points")

    return weights


def _build_noiseaware_anchor_mask(
    wavenumber: np.ndarray,
    anchor_x_values: Sequence[float] | None,
) -> np.ndarray:
    """Map stored noiseaware anchor x positions back to the spectrum grid."""
    mask = np.zeros_like(wavenumber, dtype=bool)
    if wavenumber.ndim != 1 or not anchor_x_values:
        return mask

    for anchor_x in np.asarray(anchor_x_values, dtype=float):
        if not np.isfinite(anchor_x):
            continue
        insert_index = int(np.searchsorted(wavenumber, anchor_x))
        candidate_indices = [
            candidate_index
            for candidate_index in (insert_index, insert_index - 1)
            if 0 <= candidate_index < wavenumber.size
        ]
        if not candidate_indices:
            continue

        nearest_index = min(
            candidate_indices,
            key=lambda candidate_index: abs(float(wavenumber[candidate_index]) - float(anchor_x)),
        )
        mask[nearest_index] = True

    return mask


def _build_noiseaware_anchor_value_grid(shape: tuple[int, int]) -> np.ndarray:
    """Create a per-pixel object grid for raw noiseaware anchor values."""
    grid = np.empty(shape, dtype=object)
    grid.fill(None)
    return grid


def _normalize_noiseaware_anchor_values(values: Sequence[float] | None) -> tuple[float, ...]:
    """Store the exact Stage 5 pre-median anchor values as finite float tuples."""
    if values is None:
        return ()

    normalized: list[float] = []
    for value in values:
        numeric_value = float(value)
        if np.isfinite(numeric_value):
            normalized.append(numeric_value)
    return tuple(normalized)


def apply_baseline_correction(
    parsed_collection: ParsedCollection,
    fixed_half_window: int | None,
    window_kwargs: dict,
    baseline_method: str = "mor",
    airpls_kwargs: dict | None = None,
    poly_kwargs: dict | None = None,
    poly_mask_regions: Sequence[tuple[float, float]] | None = None,
    rolling_ball_kwargs: dict | None = None,
    noiseaware_kwargs: dict | None = None,
    noiseaware_peak_regions: Sequence[tuple[float, float]] | None = None,
) -> ParsedCollectionMutable:
    """Fit and subtract a baseline for every spectrum of every map."""
    corrected_collection = []
    normalized_method = baseline_method.lower()

    valid_methods = {"mor", "airpls", "poly", "rolling_ball", "noiseaware"}
    if normalized_method not in valid_methods:
        raise ValueError(f"baseline_method must be one of {sorted(valid_methods)}")

    airpls_kwargs = {} if airpls_kwargs is None else airpls_kwargs
    poly_kwargs = {} if poly_kwargs is None else poly_kwargs
    rolling_ball_kwargs = {} if rolling_ball_kwargs is None else rolling_ball_kwargs
    noiseaware_kwargs = {} if noiseaware_kwargs is None else noiseaware_kwargs
    normalized_peak_regions = (
        [(float(lower), float(upper)) for lower, upper in noiseaware_peak_regions]
        if noiseaware_peak_regions
        else []
    )

    for parsed in parsed_collection:
        wavenumber = parsed["wavenumber_cm1"]
        spectra_cube = parsed["spectra_cube"]
        baseline_fitter = Baseline(x_data=wavenumber)
        poly_weights = _build_poly_weights(
            wavenumber=wavenumber,
            mask_regions=poly_mask_regions,
        )

        baseline_cube = np.empty_like(spectra_cube, dtype=float)
        corrected_cube = np.empty_like(spectra_cube, dtype=float)
        stat_cube = np.full(spectra_cube.shape[:2], np.nan, dtype=float)
        noiseaware_anchor_mask_cube = (
            np.zeros_like(spectra_cube, dtype=bool)
            if normalized_method == "noiseaware"
            else None
        )
        noiseaware_anchor_x_values_grid = (
            _build_noiseaware_anchor_value_grid(spectra_cube.shape[:2])
            if normalized_method == "noiseaware"
            else None
        )
        noiseaware_anchor_y_values_grid = (
            _build_noiseaware_anchor_value_grid(spectra_cube.shape[:2])
            if normalized_method == "noiseaware"
            else None
        )
        if normalized_method == "mor":
            stat_label = "half_window"
        elif normalized_method == "airpls":
            stat_label = "tol_history_len"
        elif normalized_method == "poly":
            stat_label = "mask_points"
        elif normalized_method == "rolling_ball":
            stat_label = "rolling_ball_half_window"
        else:
            stat_label = "anchors_used"

        # Fit baseline per pixel so local differences are preserved.
        for row_index in range(spectra_cube.shape[0]):
            for col_index in range(spectra_cube.shape[1]):
                spectrum = spectra_cube[row_index, col_index, :]
                if not np.isfinite(spectrum).all():
                    baseline_cube[row_index, col_index, :] = np.nan
                    corrected_cube[row_index, col_index, :] = np.nan
                    continue

                if normalized_method == "mor":
                    if fixed_half_window is None:
                        baseline, params = baseline_fitter.mor(
                            spectrum,
                            window_kwargs=window_kwargs,
                        )
                        stat_value = int(params.get("half_window", 0))
                    else:
                        baseline, params = baseline_fitter.mor(
                            spectrum,
                            half_window=fixed_half_window,
                        )
                        stat_value = int(params.get("half_window", fixed_half_window))
                else:
                    if normalized_method == "airpls":
                        baseline, params = baseline_fitter.airpls(
                            spectrum,
                            **airpls_kwargs,
                        )
                        stat_value = int(len(params.get("tol_history", [])))
                    elif normalized_method == "poly":
                        baseline, _ = baseline_fitter.poly(
                            spectrum,
                            weights=poly_weights,
                            **poly_kwargs,
                        )
                        stat_value = int(np.count_nonzero(poly_weights)) if poly_weights is not None else int(len(wavenumber))
                    elif normalized_method == "rolling_ball":
                        baseline, params = baseline_fitter.rolling_ball(
                            spectrum,
                            **rolling_ball_kwargs,
                        )
                        stat_value = int(params.get("half_window", 0))
                    elif normalized_method == "noiseaware":
                        _, baseline, info = auto_baseline_noiseaware(
                            wavenumber,
                            spectrum,
                            peak_regions=noiseaware_peak_regions,
                            **noiseaware_kwargs,
                        )
                        stat_value = int(info["anchors_used"])
                        anchor_x_values = _normalize_noiseaware_anchor_values(
                            cast(Sequence[float] | None, info.get("anchor_x_pre_median")),
                        )
                        anchor_y_values = _normalize_noiseaware_anchor_values(
                            cast(Sequence[float] | None, info.get("anchor_y_pre_median")),
                        )
                        if noiseaware_anchor_mask_cube is not None:
                            noiseaware_anchor_mask_cube[row_index, col_index, :] = _build_noiseaware_anchor_mask(
                                np.asarray(wavenumber, dtype=float),
                                anchor_x_values,
                            )
                        if noiseaware_anchor_x_values_grid is not None:
                            noiseaware_anchor_x_values_grid[row_index, col_index] = anchor_x_values
                        if noiseaware_anchor_y_values_grid is not None:
                            noiseaware_anchor_y_values_grid[row_index, col_index] = anchor_y_values

                baseline_cube[row_index, col_index, :] = baseline
                corrected_cube[row_index, col_index, :] = spectrum - baseline
                stat_cube[row_index, col_index] = stat_value

        corrected_item = {
            **parsed,
            "baseline_cube": baseline_cube,
            "corrected_spectra_cube": corrected_cube,
            "baseline_method": normalized_method,
            "baseline_stat_cube": stat_cube,
            "baseline_stat_label": stat_label,
            "baseline_fit_config": {
                "noiseaware_kwargs": dict(noiseaware_kwargs),
                "noiseaware_peak_regions": normalized_peak_regions,
            },
        }
        if normalized_method == "mor":
            corrected_item["mor_half_window_cube"] = stat_cube
        elif normalized_method == "airpls":
            corrected_item["airpls_iteration_cube"] = stat_cube
        elif normalized_method == "poly":
            corrected_item["poly_mask_points_cube"] = stat_cube
        elif normalized_method == "rolling_ball":
            corrected_item["rolling_ball_half_window_cube"] = stat_cube
        elif normalized_method == "noiseaware":
            corrected_item["noiseaware_anchors_used_cube"] = stat_cube
            if noiseaware_anchor_mask_cube is not None:
                corrected_item["noiseaware_anchor_mask_cube"] = noiseaware_anchor_mask_cube
            if noiseaware_anchor_x_values_grid is not None:
                corrected_item["noiseaware_anchor_x_values_grid"] = noiseaware_anchor_x_values_grid
            if noiseaware_anchor_y_values_grid is not None:
                corrected_item["noiseaware_anchor_y_values_grid"] = noiseaware_anchor_y_values_grid

        corrected_collection.append(corrected_item)

    return corrected_collection


def apply_mor_baseline(
    parsed_collection: ParsedCollection,
    fixed_half_window: int | None,
    window_kwargs: dict,
) -> ParsedCollectionMutable:
    """Backward-compatible wrapper for MOR-only baseline correction."""
    return apply_baseline_correction(
        parsed_collection=parsed_collection,
        baseline_method="mor",
        fixed_half_window=fixed_half_window,
        window_kwargs=window_kwargs,
        airpls_kwargs={},
    )

