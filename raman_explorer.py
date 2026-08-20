"""Interactive Raman map explorer widget and its snapshot persistence.

Split out of raman_processing_utils.py so matplotlib/ipywidgets code stays isolated
from the pure computation helpers used by the rest of the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from raman_processing_utils import (
    ParsedMap,
    StageCollections,
    StageSpectrumKeys,
    _get_noiseaware_anchor_pairs,
    plot_pixel_spectrum_comparison,
)


def save_explorer_snapshot(
    output_dir: Path,
    stage_collections: StageCollections,
    stage_spectrum_keys: StageSpectrumKeys,
    map_mode: str = "max",
) -> Path:
    """Persist all explorer stage data to a timestamped gzip-compressed pickle file."""
    import gzip
    import pickle
    from datetime import datetime

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"explorer_snapshot_{timestamp}.pkl.gz"

    snapshot = {
        "stage_collections": {stage: list(items) for stage, items in stage_collections.items()},
        "stage_spectrum_keys": dict(stage_spectrum_keys),
        "map_mode": map_mode,
    }
    with gzip.open(output_path, "wb") as handle:
        pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output_path


def find_latest_explorer_snapshot(snapshot_dir: Path) -> Path:
    """Return the most recently saved `explorer_snapshot_*.pkl.gz` file in a directory."""
    snapshot_dir = Path(snapshot_dir)
    candidates = sorted(snapshot_dir.glob("explorer_snapshot_*.pkl.gz"))
    if not candidates:
        raise FileNotFoundError(f"No explorer snapshot files found in: {snapshot_dir}")
    return candidates[-1]


def load_explorer_snapshot(input_path: Path) -> tuple[StageCollections, StageSpectrumKeys, str]:
    """Load explorer stage data previously written by `save_explorer_snapshot`."""
    import gzip
    import pickle

    input_path = Path(input_path)
    with gzip.open(input_path, "rb") as handle:
        snapshot = pickle.load(handle)

    return snapshot["stage_collections"], snapshot["stage_spectrum_keys"], snapshot["map_mode"]


def _plot_average_pixel_overlay(map_ax, item: ParsedMap) -> int:
    """Overlay average-selected pixels on a map axis, if the mask is available."""
    average_pixel_mask = item.get("average_pixel_mask")
    if average_pixel_mask is None:
        return 0

    mask = np.asarray(average_pixel_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return 0

    selected_coords = np.argwhere(mask)
    map_ax.scatter(
        selected_coords[:, 0],
        selected_coords[:, 1],
        s=42,
        facecolors="none",
        edgecolors="white",
        linewidths=0.7,
        label="Average pixels",
    )
    return int(selected_coords.shape[0])


def _resolve_stage_spectrum_keys(
    stage_collections: StageCollections,
    stage_spectrum_keys: StageSpectrumKeys | None = None,
) -> dict[str, str]:
    """Resolve the spectrum cube key for each processing stage."""
    default_stage_spectrum_keys = {
        stage_name: "spectra_cube" for stage_name in stage_collections
    }
    for stage_name, collection in stage_collections.items():
        if collection and "corrected_spectra_cube" in collection[0]:
            default_stage_spectrum_keys[stage_name] = "corrected_spectra_cube"

    if stage_spectrum_keys is None:
        return default_stage_spectrum_keys

    return {
        **default_stage_spectrum_keys,
        **stage_spectrum_keys,
    }


def _build_map_image(cube: np.ndarray, map_mode: str, wn_idx: int) -> np.ndarray:
    """Build the 2D image shown in the map panel."""
    if map_mode == "slice":
        return cube[:, :, wn_idx]
    if map_mode == "mean":
        with np.errstate(invalid="ignore"):
            return np.nanmean(cube, axis=2)
    if map_mode == "max":
        with np.errstate(invalid="ignore"):
            return np.nanmax(cube, axis=2)
    raise ValueError("map_mode must be one of ['max', 'mean', 'slice']")


def _find_stage_item(
    stage_collections: StageCollections,
    stage_name: str,
    file_name: str,
) -> ParsedMap | None:
    """Return the map item for a given stage and file name, if available."""
    for item in stage_collections.get(stage_name, []):
        if item["path"].name == file_name:
            return item
    return None


def _format_noiseaware_anchor_html(anchor_pairs) -> str:
    """Format persisted Stage 5 pre-median anchors for explorer metadata display."""
    if not anchor_pairs:
        return "<b>Stage 5 anchors:</b> n/a"

    preview_limit = 8
    formatted_pairs = [
        f"({anchor_x:.2f} cm<sup>-1</sup>, {anchor_y:.2f})"
        for anchor_x, anchor_y in anchor_pairs[:preview_limit]
    ]
    suffix = "" if len(anchor_pairs) <= preview_limit else f" ... (+{len(anchor_pairs) - preview_limit} more)"
    return (
        f"<b>Stage 5 anchors:</b> {len(anchor_pairs)} &nbsp; "
        f"<b>Pre-median x/y:</b> {'; '.join(formatted_pairs)}{suffix}"
    )


def launch_raman_map_explorer(
    stage_collections: StageCollections,
    stage_spectrum_keys: StageSpectrumKeys | None = None,
    map_mode: str = "max",
):
    """Launch an interactive map viewer where clicking a pixel selects its spectrum."""
    from io import BytesIO

    import ipywidgets as widgets
    import matplotlib
    import matplotlib.pyplot as plt
    from IPython.display import Image as IPythonImage
    from IPython.display import display

    try:
        import ipympl.backend_nbagg  # noqa: F401

        supports_map_click = "ipympl" in matplotlib.get_backend().lower()
    except (ImportError, ValueError):
        supports_map_click = False

    if not stage_collections:
        raise ValueError("stage_collections is empty")

    valid_map_modes = {"slice", "max", "mean"}
    if map_mode not in valid_map_modes:
        raise ValueError(f"map_mode must be one of {sorted(valid_map_modes)}")

    stage_spectrum_keys = _resolve_stage_spectrum_keys(
        stage_collections=stage_collections,
        stage_spectrum_keys=stage_spectrum_keys,
    )

    stage_names = list(stage_collections.keys())
    stage_dropdown = widgets.Dropdown(
        options=stage_names,
        value=stage_names[0],
        description="Stage:",
        layout=widgets.Layout(width="300px"),
    )
    file_dropdown = widgets.Dropdown(
        options=[],
        description="Map file:",
        layout=widgets.Layout(width="520px"),
    )
    map_mode_dropdown = widgets.Dropdown(
        options=[
            ("Slice", "slice"),
            ("Maximum", "max"),
            ("Mean", "mean"),
        ],
        value=map_mode,
        description="Map view:",
        layout=widgets.Layout(width="220px"),
    )
    map_index_slider = widgets.SelectionSlider(
        options=[("0.00", 0.0)],
        value=0.0,
        description="Wavenumber:",
        continuous_update=False,
        layout=widgets.Layout(width="1400px"),
    )
    prev_point_button = widgets.Button(description="< pt", layout=widgets.Layout(width="55px"))
    next_point_button = widgets.Button(description="pt >", layout=widgets.Layout(width="55px"))
    view_range_slider = widgets.FloatRangeSlider(
        value=[0.0, 1.0],
        min=0.0,
        max=1.0,
        step=1.0,
        description="View range:",
        continuous_update=False,
        layout=widgets.Layout(width="1400px"),
    )
    reset_view_range_button = widgets.Button(description="Reset range", layout=widgets.Layout(width="110px"))
    row_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="X:",
        continuous_update=False,
        layout=widgets.Layout(width="350px"),
    )
    col_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=0,
        step=1,
        description="Y:",
        continuous_update=False,
        layout=widgets.Layout(width="350px"),
    )
    info_html = widgets.HTML(
        value=(
            "Click directly on the map to select a pixel, or use the X/Y sliders for exact selection. "
            "Use the &lt; pt / pt &gt; buttons (or arrow keys after clicking the slider) "
            "to step through every wavenumber point one at a time. "
            "Use the View range slider to zoom the spectrum plot to a wavenumber window; "
            "click Reset range to return to the full spectrum."
        )
        if supports_map_click
        else (
            "Use the X/Y sliders for exact pixel selection. "
            "Direct map clicking is available when the ipympl backend is active. "
            "Use the &lt; pt / pt &gt; buttons (or arrow keys after clicking the slider) "
            "to step through every wavenumber point one at a time. "
            "Use the View range slider to zoom the spectrum plot to a wavenumber window; "
            "click Reset range to return to the full spectrum."
        )
    )
    output = widgets.Output()
    active_canvas_holder: dict[str, Any] = {"figure": None}

    def _step_point(delta: int) -> None:
        options = map_index_slider.options
        if not options:
            return
        new_index = int(np.clip(map_index_slider.index + delta, 0, len(options) - 1))
        map_index_slider.index = new_index

    prev_point_button.on_click(lambda _btn: _step_point(-1))
    next_point_button.on_click(lambda _btn: _step_point(1))

    def _reset_view_range(_btn=None) -> None:
        view_range_slider.value = [view_range_slider.min, view_range_slider.max]

    reset_view_range_button.on_click(_reset_view_range)

    def _current_item() -> tuple[str, ParsedMap, str]:
        stage_name = cast(str, stage_dropdown.value)
        selected_file = cast(str, file_dropdown.value)
        collection = stage_collections[stage_name]

        for item in collection:
            if item["path"].name == selected_file:
                return stage_name, item, stage_spectrum_keys[stage_name]

        if collection:
            return stage_name, collection[0], stage_spectrum_keys[stage_name]

        raise ValueError(f"No maps available for stage '{stage_name}'")

    def _update_file_options(*_):
        stage_name = cast(str, stage_dropdown.value)
        collection = stage_collections[stage_name]
        names = [item["path"].name for item in collection]
        file_dropdown.options = names
        file_dropdown.value = names[0] if names else None

    def _update_slider_range(*_):
        if file_dropdown.value is None:
            return
        _, item, _ = _current_item()
        wavenumber = np.asarray(item["wavenumber_cm1"], dtype=float)
        if wavenumber.size == 0:
            return

        previous_value = float(map_index_slider.value)
        options = [(f"{float(wn):.2f}", float(wn)) for wn in wavenumber]
        map_index_slider.options = options
        closest_idx = int(np.argmin(np.abs(wavenumber - previous_value)))
        map_index_slider.index = closest_idx

        wn_min, wn_max = float(np.nanmin(wavenumber)), float(np.nanmax(wavenumber))
        wn_step = float(np.min(np.diff(np.sort(wavenumber)))) if wavenumber.size > 1 else 1.0
        previous_range = list(view_range_slider.value)
        was_full_range = view_range_slider.max <= view_range_slider.min or (
            np.isclose(previous_range[0], view_range_slider.min) and np.isclose(previous_range[1], view_range_slider.max)
        )
        # Widen bounds before narrowing so ipywidgets never rejects an out-of-range value.
        view_range_slider.min = min(view_range_slider.min, wn_min)
        view_range_slider.max = max(view_range_slider.max, wn_max)
        view_range_slider.step = wn_step
        if was_full_range:
            view_range_slider.value = [wn_min, wn_max]
        else:
            view_range_slider.value = [
                float(np.clip(previous_range[0], wn_min, wn_max)),
                float(np.clip(previous_range[1], wn_min, wn_max)),
            ]
        view_range_slider.min = wn_min
        view_range_slider.max = wn_max

        active_stage = cast(str, stage_dropdown.value)
        n_rows, n_cols, _ = item[stage_spectrum_keys[active_stage]].shape
        row_slider.max = max(0, n_rows - 1)
        col_slider.max = max(0, n_cols - 1)
        if row_slider.value > row_slider.max:
            row_slider.value = row_slider.max
        if col_slider.value > col_slider.max:
            col_slider.value = col_slider.max

    def _on_file_change(*_):
        _update_slider_range()
        _render()

    def _on_stage_change(*_):
        # Suppress file_dropdown's own observer so the value it sets below doesn't re-trigger a duplicate render.
        file_dropdown.unobserve(_on_file_change, names="value")
        try:
            _update_file_options()
        finally:
            file_dropdown.observe(_on_file_change, names="value")
        _update_slider_range()
        _render()

    def _render(*_):
        with output:
            is_initial_interactive_render = supports_map_click and active_canvas_holder["figure"] is None
            if not supports_map_click or is_initial_interactive_render:
                output.clear_output(wait=True)

            if file_dropdown.value is None:
                print("No files available for selected stage")
                return

            stage_name, item, spectrum_key = _current_item()
            if spectrum_key not in item:
                raise KeyError(f"Spectrum key '{spectrum_key}' not found in stage '{stage_name}'")

            cube = item[spectrum_key]
            wavenumber = item["wavenumber_cm1"]

            n_rows, n_cols, _ = cube.shape
            row_value = int(np.clip(row_slider.value, 0, n_rows - 1))
            col_value = int(np.clip(col_slider.value, 0, n_cols - 1))
            slider_index = map_index_slider.index
            wn_idx = int(np.clip(0 if slider_index is None else slider_index, 0, len(wavenumber) - 1))

            active_map_mode = cast(str, map_mode_dropdown.value)
            map_image = _build_map_image(cube, active_map_mode, wn_idx)
            map_image_display = map_image.T

            previous_stage_item: ParsedMap | None = None
            previous_stage_label = "Previous processed spectrum"
            if stage_name in stage_names:
                stage_position = stage_names.index(stage_name)
                if stage_position > 0:
                    previous_stage_name = stage_names[stage_position - 1]
                    previous_stage_item = _find_stage_item(stage_collections, previous_stage_name, item["path"].name)
                    if previous_stage_item is not None:
                        previous_stage_label = f"Previous stage ({previous_stage_name})"
                        if stage_name == "Despiked" and previous_stage_name == "Filtered":
                            previous_stage_label = "Before despike (Filtered)"

                show_previous_overlay = stage_name not in {"Raw parsed", "Filtered"}
                if previous_stage_item is None:
                    show_previous_overlay = False

            if supports_map_click and active_canvas_holder["figure"] is None:
                # ipympl auto-displays pyplot figures unless interactive mode is paused here.
                with plt.ioff():
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
                map_ax, spectrum_ax = axes
                active_canvas_holder["figure"] = fig
            elif supports_map_click:
                fig = active_canvas_holder["figure"]
                fig.clear()
                map_ax, spectrum_ax = fig.subplots(1, 2)
            else:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
                map_ax, spectrum_ax = axes

            im = map_ax.imshow(map_image_display, origin="upper", cmap="viridis", aspect="equal")
            map_ax.scatter(row_value, col_value, s=80, c="red", edgecolors="white", linewidths=1.2)
            average_pixels_used = _plot_average_pixel_overlay(map_ax, item)
            map_ax.set_title(f"{stage_name} | {item['path'].name}")
            map_ax.set_xlabel("X index")
            map_ax.set_ylabel("Y index")
            fig.colorbar(im, ax=map_ax, fraction=0.046, pad=0.04, label="Intensity")
            if average_pixels_used:
                legend_handles, legend_labels = map_ax.get_legend_handles_labels()
                map_ax.legend(
                    legend_handles,
                    legend_labels,
                    fontsize=8,
                    loc="upper right",
                    frameon=True,
                )

            plot_pixel_spectrum_comparison(
                ax=spectrum_ax,
                parsed_item=item,
                row_index=row_value,
                col_index=col_value,
                spectrum_key=spectrum_key,
                stage_label=stage_name,
                figure_title=f"Pixel ({row_value}, {col_value})",
                highlight_wavenumber=wavenumber[wn_idx],
                show_previous_overlay=show_previous_overlay,
                show_baseline=True,
                show_noiseaware_anchors=True,
                previous_label=previous_stage_label,
                previous_parsed_item=previous_stage_item,
                previous_spectrum_key="spectra_cube",
            )

            view_min, view_max = view_range_slider.value
            if view_max > view_min:
                spectrum_ax.set_xlim(view_min, view_max)
                visible_y_values = []
                for line in spectrum_ax.get_lines():
                    x_data, y_data = np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)
                    in_range = (x_data >= view_min) & (x_data <= view_max) & np.isfinite(y_data)
                    if in_range.any():
                        visible_y_values.append(y_data[in_range])
                if visible_y_values:
                    stacked_y = np.concatenate(visible_y_values)
                    y_low, y_high = float(np.min(stacked_y)), float(np.max(stacked_y))
                    y_margin = (y_high - y_low) * 0.08 if y_high > y_low else max(abs(y_high), 1.0) * 0.08
                    spectrum_ax.set_ylim(y_low - y_margin, y_high + y_margin)

            map_mode_label = getattr(map_mode_dropdown, "label", None) or str(map_mode_dropdown.value).capitalize()
            slice_label = (
                f"{wavenumber[wn_idx]:.2f} cm<sup>-1</sup>"
                if map_mode_dropdown.value == "slice"
                else f"{map_mode_label} over spectrum"
            )
            info_html.value = (
                f"<b>Stage:</b> {stage_name} &nbsp; "
                f"<b>File:</b> {item['path'].name} &nbsp; "
                f"<b>Pixel (x,y):</b> ({row_value}, {col_value}) &nbsp; "
                f"<b>Average pixels:</b> {average_pixels_used if average_pixels_used else 'n/a'} &nbsp; "
                f"<b>Map view:</b> {slice_label}"
            )
            if stage_name in {"Stage 5 Baseline corrected", "Stage 6 Map-average plotting"}:
                info_html.value += "<br>" + _format_noiseaware_anchor_html(
                    _get_noiseaware_anchor_pairs(item, row_value, col_value),
                )

            def _onclick(event):
                if event.inaxes is not map_ax or event.xdata is None or event.ydata is None:
                    return
                row_slider.value = int(np.clip(round(event.xdata), 0, n_rows - 1))
                col_slider.value = int(np.clip(round(event.ydata), 0, n_cols - 1))

            fig.canvas.mpl_connect("button_press_event", _onclick)
            fig.tight_layout()
            if supports_map_click:
                fig.canvas.header_visible = False
                fig.canvas.footer_visible = False
                fig.canvas.toolbar_visible = False
                if is_initial_interactive_render:
                    display(fig.canvas)
                else:
                    fig.canvas.draw_idle()
            else:
                image_buffer = BytesIO()
                fig.savefig(image_buffer, format="png", dpi=120, bbox_inches="tight")
                display(IPythonImage(data=image_buffer.getvalue(), format="png"))
                plt.close(fig)

    stage_dropdown.observe(_on_stage_change, names="value")
    file_dropdown.observe(_on_file_change, names="value")
    map_mode_dropdown.observe(_render, names="value")
    map_index_slider.observe(_render, names="value")
    view_range_slider.observe(_render, names="value")
    row_slider.observe(_render, names="value")
    col_slider.observe(_render, names="value")

    _update_file_options()
    _update_slider_range()

    controls = widgets.VBox(
        [
            widgets.HBox([stage_dropdown, file_dropdown]),
            widgets.HBox([map_mode_dropdown, row_slider, col_slider]),
            widgets.HBox([prev_point_button, map_index_slider, next_point_button]),
            widgets.HBox([view_range_slider, reset_view_range_button]),
            info_html,
        ]
    )
    viewer = widgets.VBox([controls, output])
    display(viewer)
    _render()
    return viewer
