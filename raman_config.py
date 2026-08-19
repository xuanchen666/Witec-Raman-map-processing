"""Central configuration for Raman_processing notebook pipeline.

How to use this file:
1) Edit grouped blocks under each stage namespace class.
2) Keep using existing flat constant names in code and notebooks.
   Flat names are exported at the bottom for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path


# -----------------------------------------------------------------------------
# 0) Paths and data source
# -----------------------------------------------------------------------------
class Paths:
    """Input data locations and path-level settings."""

    DATA_DIR = Path(
        r"C:\Users\xuli\OneDrive - empa.ch\INT Lab 205 - 17AGNR_Xuanchen_Rafaela_Riya\02_Processed Raman data_ambient stability\S6_17AGNR_High cov_RO"
    )


# -----------------------------------------------------------------------------
# 1) Stage 1: parsing
# -----------------------------------------------------------------------------
# Parsing uses parser defaults and does not expose extra runtime knobs here.


# -----------------------------------------------------------------------------
# 2) Stage 2: pixel-level filtering
# Order: border filter -> spectrum gate -> optional max-intensity cutoff
# -----------------------------------------------------------------------------
class Stage2PixelFilter:
    """All pixel-cut controls grouped in the exact Stage 2 execution order."""

    # Scope selector rule used by Stage 2 functions:
    # - "all"   => apply to every map
    # - "hBN"   => apply to whole hBN group (all hBN subgroups)
    # - "hBN_1" => apply only to that subgroup
    # You can also target Au/RO similarly, and mix values in one tuple.

    # 2.1) Border-pixel filtering
    BORDER_ENABLED = True  # Enable border-pixel filtering before spectrum gating.
    BORDER_WIDTH = 1
    BORDER_APPLY_TO_GROUPS = ("hBN_1",)

    # 2.2) Spectrum gate by mean intensity in a target window
    SPECTRUM_GATE_ENABLED = False
    SPECTRUM_GATE_WAVENUMBER_REGION_CM1 = (1360, 1370)
    SPECTRUM_GATE_MIN_MEAN_INTENSITY = 10000
    SPECTRUM_GATE_APPLY_TO_GROUPS = ("hBN",)

    # 2.3) Optional max-intensity cutoff
    MAX_PIXEL_INTENSITY = None  # If None, this cutoff is skipped.
    MAX_INTENSITY_APPLY_TO_GROUPS = ("hBN",)


# -----------------------------------------------------------------------------
# 3) Stage 3: low-wavenumber filtering (spectral-axis trim only)
# -----------------------------------------------------------------------------
class Stage3LowWavenumberFilter:
    """Drop data points below the configured wavenumber threshold."""

    MIN_WAVENUMBER_CM1 = 75


# -----------------------------------------------------------------------------
# 4) Stage 4: despiking (rampy.despiking)
# -----------------------------------------------------------------------------
class Stage4Despike:
    """Control despiking strength and aggressive-pixel shortlist size."""

    DESPIKE_NEIGH = 6  # Number of neighboring points used for local comparison.
    DESPIKE_THRESHOLD = 10  # Spike detection sensitivity threshold.
    TOP_N_AGGRESSIVE_DESPIKE = 0  # Number of most-changed spectra shown in QC.


# -----------------------------------------------------------------------------
# 5) Stage 5: baseline correction
# Methods: "mor", "airpls", "poly", "rolling_ball", "noiseaware"
# -----------------------------------------------------------------------------
class Stage5Baseline:
    """Baseline method selection and per-method parameters."""

    METHOD = "noiseaware"  # Select baseline algorithm.

    MOR_HALF_WINDOW = None  # Half-window for MOR; None enables auto search.
    MOR_WINDOW_KWARGS = {
        "min_half_window": 1,  # Minimum half-window tested in MOR auto search.
        "max_half_window": 200,  # Maximum half-window tested in MOR auto search.
        "increment": 1,  # Step size when sweeping MOR half-window.
        "max_hits": 20,  # Max candidate windows accepted before stopping search.
        "window_tol": 1e-5,  # Convergence tolerance for MOR window selection.
    }
    AIRPLS_KWARGS = {
        "lam": 50_000,  # Smoothness penalty; larger values give flatter baselines.
        "diff_order": 2,  # Difference order for baseline roughness control.
        "max_iter": 100,  # Maximum optimization iterations.
        "tol": 1e-4,  # Iteration stop tolerance.
    }
    POLY_KWARGS = {
        "poly_order": 10,  # Polynomial degree used for baseline fitting.
    }
    # Optional fitting windows for "poly" in cm^-1.
    # Example: [(75, 500), (1800, 2000)]
    POLY_MASK_REGIONS = None  # Fit baseline only inside these wavenumber windows.
    ROLLING_BALL_KWARGS = {
        "half_window": None,  # Ball radius proxy; None lets method choose automatically.
        "smooth_half_window": None,  # Optional pre-smoothing window before rolling-ball.
    }
    NOISEAWARE_KWARGS = {
        "q_low": 15,  # Low-quantile level for initial background candidate.
        "smooth_win": 50,  # Smoothing window for baseline trend extraction.
        "mad_win": 31,  # Window for local MAD noise estimation.
        "z_keep": 6,  # Keep points within this robust z-score from background.
        "min_region": 5,  # Minimum contiguous background region length.
        "prom_k": 7,  # Peak prominence multiplier relative to local noise.
        "peak_dilate": 3,  # Expand detected peak masks by this many points.
        "remove_k_sigma": 5,  # Threshold for removing connecting anchors based on robust residuals.
        "fit_interp": "pchip",  # Interpolation for the anchor-based connecting baseline outside explicit peak regions: "pchip" or "spline".
        "bridge_interp": "linear",  # Interpolation used only inside explicit user-defined peak regions: "linear", "pchip", or "spline".
        "forced_anchor_wavenumbers": [2000],  # cm^-1 targets; algorithm snaps each value to the nearest available point on the current spectrum grid.
        "bias_strength": 0.5,  # Pull baseline slightly toward lower envelope.
    }
    # Regions always treated as peaks by noiseaware methods, in cm^-1.
    PEAK_REGIONS = [(1350, 1600), (2450, 3300)]  # Force these intervals to stay peak-masked.


# -----------------------------------------------------------------------------
# 6) Stage 6: map-average plotting
# -----------------------------------------------------------------------------
class Stage6MapAveragePlot:
    """Plot source selection, axis bounds, stacking, and normalization."""

    # "spectra_cube" = pre-baseline, "corrected_spectra_cube" = baseline-corrected
    SOURCE_SPECTRUM_KEY = "corrected_spectra_cube"
    MAP_GROUPS = ("Au", "RO", "hBN")

    # Stage 6 export windows in cm^-1.
    # Use one tuple for one output set, for example: (1200, 1700)
    # Use multiple tuples for multiple output sets, for example:
    # ((75, 250), (1200, 1700))
    # If one range needs its own stack spacing, use a dict entry instead:
    # {
    #     "min": 75,
    #     "max": 250,
    #     "raw_stack_scale": {"RO": 0.2, "hBN": 0.1},
    #     "norm_stack_scale": {"RO": 0.5, "hBN": 0.5},
    # }
    # Set either bound to None to keep matplotlib automatic behavior on that side.
    
    PLOT_WAVENUMBER_RANGES = (
    {
        "min": 75,
        "max": 250,
        "raw_stack_scale": {"RO": 2, "hBN": 2},
        "raw_stack_extra_gap": {"RO": 0, "hBN": 0},
        "norm_stack_scale": {"RO": 1.2, "hBN": 1},
        "norm_stack_extra_gap": {"RO": 0, "hBN": 0},
    },
    (1200, 1700),
)

    # Independent range(s) for peak-ratio calculations.
    # This is intentionally separate from PLOT_WAVENUMBER_RANGES so peak ratio
    # does not have to be calculated for every exported plot range.
    # Use one tuple, multiple tuples, or None for full-spectrum peak search.
    PEAK_RATIO_WAVENUMBER_RANGES = (1200, 1700)

    # Offset model for grouped spectra plots:
    # offset = span * stack_scale + stack_extra_gap
    # Each group can be assigned its own spacing. RO is now given an explicit entry
    # so it no longer depends on the first mapping value. The values below keep RO
    # visually consistent with Au while making the configuration explicit.
    RAW_STACK_SCALE = {"Au": 3, "RO": 1, "hBN": 5}
    RAW_STACK_EXTRA_GAP = {"Au": 0, "RO": 0, "hBN": 10}
    NORM_STACK_SCALE = {"Au": 2, "RO": 1, "hBN": 5}
    NORM_STACK_EXTRA_GAP = {"Au": 0, "RO": 0, "hBN": 0}

    # "minmax" => [0, 1], "peak_1590" => divide by local max near 1590 cm^-1
    NORMALIZATION_METHOD = "peak_1590"
    NORMALIZATION_PEAK_CENTER_CM1 = 1590
    NORMALIZATION_PEAK_TOLERANCE_CM1 = 50

    # Wavenumber used for cut-pixel map slice export color scale.
    CUT_PIXEL_MAP_WAVENUMBER_CM1 = 1590


# -----------------------------------------------------------------------------
# 7) Interactive explorer
# -----------------------------------------------------------------------------
class Explorer:
    """Interactive map viewer behavior."""

    EXPLORER_MAP_MODE = "max"

    # Directory for timestamped snapshots used to reopen the explorer without rerunning the full pipeline.
    EXPLORER_SNAPSHOT_DIR = Paths.DATA_DIR / "explorer_snapshots"


# -----------------------------------------------------------------------------
# Legacy class aliases
# Keep class-style access compatible with older code.
# -----------------------------------------------------------------------------
class Stage1SpectrumGate:
    """Compatibility view for historical Stage1SpectrumGate names."""

    ENABLED = Stage2PixelFilter.SPECTRUM_GATE_ENABLED
    WAVENUMBER_REGION_CM1 = Stage2PixelFilter.SPECTRUM_GATE_WAVENUMBER_REGION_CM1
    MIN_MEAN_INTENSITY = Stage2PixelFilter.SPECTRUM_GATE_MIN_MEAN_INTENSITY
    APPLY_TO_GROUPS = Stage2PixelFilter.SPECTRUM_GATE_APPLY_TO_GROUPS


class Stage2BorderFilter:
    """Compatibility view for historical Stage2BorderFilter names."""

    ENABLED = Stage2PixelFilter.BORDER_ENABLED
    BORDER_WIDTH = Stage2PixelFilter.BORDER_WIDTH
    APPLY_TO_GROUPS = Stage2PixelFilter.BORDER_APPLY_TO_GROUPS


class Stage2LowWavenumberFilter:
    """Compatibility alias for historical class naming."""

    MIN_WAVENUMBER_CM1 = Stage3LowWavenumberFilter.MIN_WAVENUMBER_CM1


class Stage3Despike:
    """Compatibility view for historical Stage3Despike names."""

    DESPIKE_NEIGH = Stage4Despike.DESPIKE_NEIGH
    DESPIKE_THRESHOLD = Stage4Despike.DESPIKE_THRESHOLD
    TOP_N_AGGRESSIVE_DESPIKE = Stage4Despike.TOP_N_AGGRESSIVE_DESPIKE
    MAX_PIXEL_INTENSITY = Stage2PixelFilter.MAX_PIXEL_INTENSITY
    MAX_INTENSITY_APPLY_TO_GROUPS = Stage2PixelFilter.MAX_INTENSITY_APPLY_TO_GROUPS


class Stage4Baseline:
    """Compatibility alias for historical class naming."""

    METHOD = Stage5Baseline.METHOD
    MOR_HALF_WINDOW = Stage5Baseline.MOR_HALF_WINDOW
    MOR_WINDOW_KWARGS = Stage5Baseline.MOR_WINDOW_KWARGS
    AIRPLS_KWARGS = Stage5Baseline.AIRPLS_KWARGS
    POLY_KWARGS = Stage5Baseline.POLY_KWARGS
    POLY_MASK_REGIONS = Stage5Baseline.POLY_MASK_REGIONS
    ROLLING_BALL_KWARGS = Stage5Baseline.ROLLING_BALL_KWARGS
    NOISEAWARE_KWARGS = Stage5Baseline.NOISEAWARE_KWARGS
    PEAK_REGIONS = Stage5Baseline.PEAK_REGIONS


class Stage5Plot:
    """Compatibility alias for historical class naming."""

    SOURCE_SPECTRUM_KEY = Stage6MapAveragePlot.SOURCE_SPECTRUM_KEY
    MAP_GROUPS = Stage6MapAveragePlot.MAP_GROUPS
    PLOT_WAVENUMBER_RANGES = Stage6MapAveragePlot.PLOT_WAVENUMBER_RANGES
    PEAK_RATIO_WAVENUMBER_RANGES = Stage6MapAveragePlot.PEAK_RATIO_WAVENUMBER_RANGES
    RAW_STACK_SCALE = Stage6MapAveragePlot.RAW_STACK_SCALE
    RAW_STACK_EXTRA_GAP = Stage6MapAveragePlot.RAW_STACK_EXTRA_GAP
    NORM_STACK_SCALE = Stage6MapAveragePlot.NORM_STACK_SCALE
    NORM_STACK_EXTRA_GAP = Stage6MapAveragePlot.NORM_STACK_EXTRA_GAP
    NORMALIZATION_METHOD = Stage6MapAveragePlot.NORMALIZATION_METHOD
    NORMALIZATION_PEAK_CENTER_CM1 = Stage6MapAveragePlot.NORMALIZATION_PEAK_CENTER_CM1
    NORMALIZATION_PEAK_TOLERANCE_CM1 = Stage6MapAveragePlot.NORMALIZATION_PEAK_TOLERANCE_CM1
    CUT_PIXEL_MAP_WAVENUMBER_CM1 = Stage6MapAveragePlot.CUT_PIXEL_MAP_WAVENUMBER_CM1


# -----------------------------------------------------------------------------
# Backward-compatible flat exports
# Keep these names stable for existing scripts and notebooks.
# -----------------------------------------------------------------------------
DATA_DIR = Paths.DATA_DIR

SPECTRUM_GATE_ENABLED = Stage2PixelFilter.SPECTRUM_GATE_ENABLED
SPECTRUM_GATE_WAVENUMBER_REGION_CM1 = Stage2PixelFilter.SPECTRUM_GATE_WAVENUMBER_REGION_CM1
SPECTRUM_GATE_MIN_MEAN_INTENSITY = Stage2PixelFilter.SPECTRUM_GATE_MIN_MEAN_INTENSITY
SPECTRUM_GATE_APPLY_TO_GROUPS = Stage2PixelFilter.SPECTRUM_GATE_APPLY_TO_GROUPS

MIN_WAVENUMBER_CM1 = Stage3LowWavenumberFilter.MIN_WAVENUMBER_CM1

BORDER_FILTER_ENABLED = Stage2PixelFilter.BORDER_ENABLED
BORDER_FILTER_BORDER_WIDTH = Stage2PixelFilter.BORDER_WIDTH
BORDER_FILTER_APPLY_TO_GROUPS = Stage2PixelFilter.BORDER_APPLY_TO_GROUPS

DESPIKE_NEIGH = Stage4Despike.DESPIKE_NEIGH
DESPIKE_THRESHOLD = Stage4Despike.DESPIKE_THRESHOLD
TOP_N_AGGRESSIVE_DESPIKE = Stage4Despike.TOP_N_AGGRESSIVE_DESPIKE
MAX_PIXEL_INTENSITY = Stage2PixelFilter.MAX_PIXEL_INTENSITY
MAX_INTENSITY_APPLY_TO_GROUPS = Stage2PixelFilter.MAX_INTENSITY_APPLY_TO_GROUPS

BASELINE_METHOD = Stage5Baseline.METHOD
MOR_HALF_WINDOW = Stage5Baseline.MOR_HALF_WINDOW
MOR_WINDOW_KWARGS = Stage5Baseline.MOR_WINDOW_KWARGS
AIRPLS_KWARGS = Stage5Baseline.AIRPLS_KWARGS
POLY_KWARGS = Stage5Baseline.POLY_KWARGS
POLY_MASK_REGIONS = Stage5Baseline.POLY_MASK_REGIONS
ROLLING_BALL_KWARGS = Stage5Baseline.ROLLING_BALL_KWARGS
NOISEAWARE_KWARGS = Stage5Baseline.NOISEAWARE_KWARGS
NOISEAWARE_PEAK_REGIONS = Stage5Baseline.PEAK_REGIONS

SOURCE_SPECTRUM_KEY = Stage6MapAveragePlot.SOURCE_SPECTRUM_KEY
MAP_GROUPS = Stage6MapAveragePlot.MAP_GROUPS
PLOT_WAVENUMBER_RANGES = Stage6MapAveragePlot.PLOT_WAVENUMBER_RANGES
PEAK_RATIO_WAVENUMBER_RANGES = Stage6MapAveragePlot.PEAK_RATIO_WAVENUMBER_RANGES
RAW_STACK_SCALE = Stage6MapAveragePlot.RAW_STACK_SCALE
RAW_STACK_EXTRA_GAP = Stage6MapAveragePlot.RAW_STACK_EXTRA_GAP
NORM_STACK_SCALE = Stage6MapAveragePlot.NORM_STACK_SCALE
NORM_STACK_EXTRA_GAP = Stage6MapAveragePlot.NORM_STACK_EXTRA_GAP
NORMALIZATION_METHOD = Stage6MapAveragePlot.NORMALIZATION_METHOD
NORMALIZATION_PEAK_CENTER_CM1 = Stage6MapAveragePlot.NORMALIZATION_PEAK_CENTER_CM1
NORMALIZATION_PEAK_TOLERANCE_CM1 = Stage6MapAveragePlot.NORMALIZATION_PEAK_TOLERANCE_CM1
CUT_PIXEL_MAP_WAVENUMBER_CM1 = Stage6MapAveragePlot.CUT_PIXEL_MAP_WAVENUMBER_CM1

EXPLORER_MAP_MODE = Explorer.EXPLORER_MAP_MODE
EXPLORER_SNAPSHOT_DIR = Explorer.EXPLORER_SNAPSHOT_DIR