"""
SWV Batch Analysis  Streamlit UI
Run with:  python -m streamlit run app.py
"""

import bisect
from datetime import datetime, timedelta
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import is_color_like, to_hex
import numpy as np
import pandas as pd
import pywt
import streamlit as st
from scipy.stats import skew

from core import (
    build_titration_langmuir_summary_table,
    build_titration_measurement_accuracy_table,
    build_titration_step_table,
    compute_drift_fields,
    filter_extreme_titration_outliers,
    infer_titration_response_directions,
    infer_titration_response_baselines,
    plot_cv_overlaid_cycles,
    plot_cv_trace,
    plot_drift_vs_scan,
    plot_failed_traces,
    plot_grouped_overlaid_traces,
    plot_metric_vs_scan,
    plot_overlaid_traces,
    plot_single_trace,
    plot_titration_langmuir,
    plot_titration_concentration_accuracy,
    plot_titration_concentration_vs_measurement,
    plot_titration_plateaus,
    plot_titration_snr,
    run_cv_batch,
    run_batch,
)
from core.analysis import analyze_swv_arrays
from core.processing import (
    detect_dominant_peak,
    rotate_offset_using_bracketing_minima,
    rotate_offset_using_prominent_bracketing_minima,
)
from bo_session_viewer import render_bo_session_app
from core.mat_conversion import convert_mat_folders_to_swv_csv
from core.io import collect_measurement_csvs_from_folders, parse_measurement_time_from_filename


def _pick_folder_windows() -> str:
    """
    Using Tk/Tcl dialogs inside the Streamlit process can trigger thread-related
    crashes/errors (e.g., Tcl_AsyncDelete). Run the Tk dialog in a short-lived
    subprocess instead.
    """
    code = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root=tk.Tk()\n"
        "root.withdraw()\n"
        "root.wm_attributes('-topmost', True)\n"
        "p=filedialog.askdirectory(title='Select electrochemistry data folder')\n"
        "root.destroy()\n"
        "print(p or '')\n"
    )
    return subprocess.check_output([sys.executable, "-c", code], text=True).strip()


def _append_unique_folder(folders: List[str], picked: str) -> List[str]:
    picked_clean = picked.strip()
    if not picked_clean:
        return list(folders)
    if picked_clean in folders:
        return list(folders)
    return [*folders, picked_clean]


def _analysis_selection_key(analysis_mode: str, folders: List[str]) -> Tuple[str, Tuple[str, ...]]:
    return analysis_mode, tuple(folders)


def _clear_loaded_analysis_state() -> None:
    st.session_state.results = None
    st.session_state.last_results = None
    st.session_state.results_mode = None
    st.session_state.last_results_mode = None
    st.session_state.results_folder_key = None
    st.session_state.swv_annotation_signature = None
    st.session_state.swv_annotated_results = None
    st.session_state.analysis_cache_key = None
    st.session_state.analysis_cache_results = None


ANALYSIS_CACHE_SCHEMA_VERSION = 2


def _analysis_input_signature(folders: Tuple[str, ...]) -> tuple:
    """Fingerprint measurement and method files that affect a batch result."""
    candidate_paths = set()
    for folder_text in folders:
        folder = Path(folder_text)
        if folder.is_file():
            candidate_paths.add(folder)
            search_folder = folder.parent
        elif folder.is_dir():
            search_folder = folder
            try:
                candidate_paths.update(
                    path for path in search_folder.iterdir()
                    if path.is_file() and path.suffix.lower() == ".csv"
                )
            except OSError:
                continue
        else:
            continue

        method_folders = []
        for method_parent in (search_folder, search_folder.parent):
            try:
                method_folders.extend(
                    path for path in method_parent.iterdir()
                    if (
                        path.is_dir()
                        and path.name.lower().replace(" ", "_") == "methods_used"
                    )
                )
            except OSError:
                continue
        for method_folder in method_folders:
            try:
                candidate_paths.update(
                    path for path in method_folder.iterdir()
                    if path.is_file() and path.suffix.lower() == ".ms"
                )
            except OSError:
                continue

    records = []
    for path in sorted(candidate_paths, key=lambda item: str(item).lower()):
        try:
            stat = path.stat()
        except OSError:
            continue
        records.append((str(path), int(stat.st_mtime_ns), int(stat.st_size)))
    return (ANALYSIS_CACHE_SCHEMA_VERSION, tuple(records))


def _measurement_voltage_bounds(
    folders: Tuple[str, ...],
    mode: str,
) -> Optional[Tuple[float, float]]:
    """Return the finite voltage extent across the selected native data files."""
    lower = math.inf
    upper = -math.inf
    for measurement in collect_measurement_csvs_from_folders(list(folders), mode=mode):
        try:
            voltage = pd.read_csv(
                measurement.path,
                usecols=["Potential (V)"],
            )["Potential (V)"].to_numpy(dtype=float)
        except Exception:
            continue
        finite = voltage[np.isfinite(voltage)]
        if finite.size:
            lower = min(lower, float(np.min(finite)))
            upper = max(upper, float(np.max(finite)))
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
        return None
    return lower, upper


def safe_download_stem(label: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_")
    return stem or "plot"


def _parse_numeric_list(text: str) -> List[float]:
    values = []
    for token in re.split(r"[\n,]+", str(text or "")):
        try:
            value = float(token.strip())
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _parse_text_list(text: str) -> List[str]:
    return [
        token.strip()
        for token in re.split(r"[\n,]+", str(text or ""))
        if token.strip()
    ]


_SWV_METRICS_STYLE_SUFFIXES = frozenset({
    "width_px",
    "height_px",
    "text_size_points",
    "line_width_scale",
    "line_color_override",
    "margin_px",
    "perimeter_width",
    "perimeter_color",
    "marker_size",
    "marker_opacity",
    "show_legend",
    "show_grid",
})
_SWV_OVERLAY_WIDTH_SCALE = 0.67
_SWV_OVERLAY_TEXT_SCALE = 1.0 / _SWV_OVERLAY_WIDTH_SCALE


def _resolve_swv_plot_setting(
    settings_prefix: str,
    suffix: str,
    default: Any,
    *,
    inherit_metrics_style: bool = False,
) -> Any:
    """Resolve shared plot styling, optionally preferring Metrics-wide values."""
    if inherit_metrics_style and suffix in _SWV_METRICS_STYLE_SUFFIXES:
        metrics_key = f"swv_metrics_all_{suffix}"
        if metrics_key in st.session_state:
            return st.session_state[metrics_key]
    scoped_key = f"{settings_prefix}_{suffix}"
    if scoped_key in st.session_state:
        return st.session_state[scoped_key]
    return st.session_state.get(f"swv_plot_{suffix}", default)


def _sync_shared_swv_style_to_metrics(suffix: str) -> None:
    """Keep the sidebar style control and Metrics-wide default in agreement."""
    source_key = f"swv_plot_{suffix}"
    if source_key in st.session_state:
        st.session_state[f"swv_metrics_all_{suffix}"] = st.session_state[source_key]


def _apply_swv_plot_formatting(
    fig: plt.Figure,
    dpi: int,
    plot_kind: Optional[str] = None,
    settings_prefix: str = "swv_plot",
) -> None:
    def plot_setting(suffix: str, default: Any) -> Any:
        return _resolve_swv_plot_setting(
            settings_prefix,
            suffix,
            default,
            inherit_metrics_style=plot_kind == "swv_trace",
        )

    width_px = int(plot_setting("width_px", 1200))
    if plot_kind == "swv_trace":
        width_px = max(1, int(round(width_px * _SWV_OVERLAY_WIDTH_SCALE)))
    height_px = int(plot_setting("height_px", 600))
    text_size = float(plot_setting("text_size_points", 10.0))
    if plot_kind == "swv_trace":
        text_size *= _SWV_OVERLAY_TEXT_SCALE
    line_scale = float(plot_setting("line_width_scale", 1.0))
    perimeter_width = float(plot_setting("perimeter_width", 0.8))
    perimeter_color = str(
        plot_setting("perimeter_color", "#222222") or "#222222"
    ).strip()
    if not is_color_like(perimeter_color):
        perimeter_color = "#222222"
    marker_size = float(plot_setting("marker_size", 6.0))
    marker_opacity = float(plot_setting("marker_opacity", 0.85))
    show_legend = bool(plot_setting("show_legend", True))
    show_grid = bool(plot_setting("show_grid", False))
    line_color = str(plot_setting("line_color_override", "") or "").strip()
    if line_color and not is_color_like(line_color):
        line_color = ""

    fig.set_size_inches(width_px / float(dpi), height_px / float(dpi), forward=True)
    title_override = str(plot_setting("title_override", "") or "").strip()
    x_label_override = str(plot_setting("x_label_override", "") or "").strip()
    y_label_override = str(plot_setting("y_label_override", "") or "").strip()
    colorbar_label_override = str(
        st.session_state.get("swv_plot_colorbar_label_override", "") or ""
    ).strip()

    main_axes = [
        axis for axis in fig.axes
        if not bool(getattr(axis, "_swv_colorbar_axis", False))
        and axis.get_label() != "<colorbar>"
    ]
    colorbar_axes = [axis for axis in fig.axes if axis not in main_axes]
    for axis in main_axes:
        if title_override:
            axis.set_title(title_override)
        if x_label_override:
            axis.set_xlabel(x_label_override)
        if y_label_override:
            axis.set_ylabel(y_label_override)
        axis.title.set_fontsize(text_size * 1.2)
        axis.xaxis.label.set_fontsize(text_size)
        axis.yaxis.label.set_fontsize(text_size)
        axis.tick_params(labelsize=text_size * 0.9)
        if show_grid:
            axis.grid(True, alpha=0.2)
        else:
            axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(perimeter_width > 0)
            spine.set_linewidth(perimeter_width)
            spine.set_edgecolor(perimeter_color)
        if bool(getattr(axis, "_swv_force_black_y_axis", False)):
            axis.yaxis.label.set_color("black")
            axis.tick_params(axis="y", colors="black")
            axis.spines["left"].set_edgecolor("black")
            axis.spines["right"].set_edgecolor("black")
        legend = axis.get_legend()
        if legend is not None:
            legend.set_visible(show_legend)
            for legend_text in legend.get_texts():
                legend_text.set_fontsize(text_size * 0.8)
            legend.get_title().set_fontsize(text_size * 0.9)
        for line in axis.lines:
            line.set_linewidth(max(0.1, line.get_linewidth() * line_scale))
            if line_color and not bool(getattr(line, "_swv_preserve_color", False)):
                line.set_color(line_color)
        for collection in axis.collections:
            if hasattr(collection, "get_sizes") and hasattr(collection, "set_sizes"):
                sizes = collection.get_sizes()
                if len(sizes):
                    collection.set_sizes(np.full_like(sizes, marker_size ** 2, dtype=float))
            if not bool(getattr(collection, "_swv_preserve_alpha", False)):
                collection.set_alpha(marker_opacity)

        if (
            plot_kind == "swv_trace"
            and st.session_state.get("swv_plot_manual_x_limits", False)
        ):
            x_min = float(st.session_state.get("swv_plot_x_min", axis.get_xlim()[0]))
            x_max = float(st.session_state.get("swv_plot_x_max", axis.get_xlim()[1]))
            if x_min < x_max:
                axis.set_xlim(x_min, x_max)
        if (
            plot_kind == "swv_trace"
            and st.session_state.get("swv_plot_manual_y_limits", False)
        ):
            y_min = float(st.session_state.get("swv_plot_y_min", axis.get_ylim()[0]))
            y_max = float(st.session_state.get("swv_plot_y_max", axis.get_ylim()[1]))
            if y_min < y_max:
                axis.set_ylim(y_min, y_max)

        for axis_key, matplotlib_axis in (("x", axis.xaxis), ("y", axis.yaxis)):
            labels = _parse_text_list(
                plot_setting(f"{axis_key}_tick_labels", "")
            )
            if not labels:
                continue
            positions = _parse_numeric_list(
                plot_setting(f"{axis_key}_tick_positions", "")
            )
            if not positions:
                positions = list(matplotlib_axis.get_ticklocs())
            count = min(len(positions), len(labels))
            if count:
                matplotlib_axis.set_ticks(positions[:count], labels=labels[:count])

    colorbar_tick_labels = _parse_text_list(
        st.session_state.get("swv_plot_colorbar_tick_labels", "")
    )
    colorbar_tick_positions = _parse_numeric_list(
        st.session_state.get("swv_plot_colorbar_tick_positions", "")
    )
    for axis in colorbar_axes:
        axis.tick_params(labelsize=text_size * 0.9)
        axis.title.set_fontsize(text_size)
        if colorbar_label_override:
            axis.set_ylabel(colorbar_label_override)
        axis.yaxis.label.set_fontsize(text_size)
        if colorbar_tick_labels:
            positions = colorbar_tick_positions or list(axis.yaxis.get_ticklocs())
            count = min(len(positions), len(colorbar_tick_labels))
            if count:
                axis.set_yticks(positions[:count], labels=colorbar_tick_labels[:count])


def _position_swv_colorbars(fig: plt.Figure) -> None:
    if bool(getattr(fig, "_swv_manual_layout", False)):
        return
    main_axes = [
        axis for axis in fig.axes
        if axis.get_label() != "<colorbar>"
        and not bool(getattr(axis, "_swv_colorbar_axis", False))
    ]
    colorbar_axes = [axis for axis in fig.axes if axis not in main_axes]
    if not main_axes or not colorbar_axes:
        return
    main_position = main_axes[0].get_position()
    side = str(st.session_state.get("swv_colorbar_side", "right")).lower()
    height_fraction = min(
        1.0,
        max(
            0.2,
            float(st.session_state.get("swv_colorbar_height_percent", 85)) / 100.0,
        ),
    )
    bar_height = main_position.height * height_fraction
    bar_bottom = main_position.y0 + (main_position.height - bar_height) / 2.0
    for index, colorbar_axis in enumerate(colorbar_axes):
        current_position = colorbar_axis.get_position()
        bar_width = min(max(current_position.width, 0.015), 0.025)
        if side == "left":
            bar_left = max(
                0.01,
                main_position.x0 - 0.035 - bar_width - index * (bar_width + 0.03),
            )
        else:
            bar_left = min(
                0.98 - bar_width,
                main_position.x1 + 0.025 + index * (bar_width + 0.03),
            )
        colorbar_axis.set_position([
            bar_left,
            bar_bottom,
            bar_width,
            bar_height,
        ])


def _apply_all_metrics_plot_settings() -> None:
    """Copy the Metrics-wide form values into the shared SWV plot settings."""
    setting_map = {
        "swv_plot_width_px": "swv_metrics_all_width_px",
        "swv_plot_height_px": "swv_metrics_all_height_px",
        "swv_plot_text_size_points": "swv_metrics_all_text_size_points",
        "swv_plot_line_width_scale": "swv_metrics_all_line_width_scale",
        "swv_plot_line_color_override": "swv_metrics_all_line_color_override",
        "swv_plot_margin_px": "swv_metrics_all_margin_px",
        "swv_plot_perimeter_width": "swv_metrics_all_perimeter_width",
        "swv_plot_perimeter_color": "swv_metrics_all_perimeter_color",
        "swv_plot_show_legend": "swv_metrics_all_show_legend",
        "swv_plot_show_grid": "swv_metrics_all_show_grid",
        "swv_plot_marker_size": "swv_metrics_all_marker_size",
        "swv_plot_marker_opacity": "swv_metrics_all_marker_opacity",
        "swv_plot_title_override": "swv_metrics_all_title_override",
        "swv_plot_x_label_override": "swv_metrics_all_x_label_override",
        "swv_plot_y_label_override": "swv_metrics_all_y_label_override",
    }
    for target_key, source_key in setting_map.items():
        if source_key in st.session_state:
            st.session_state[target_key] = st.session_state[source_key]

    # Saved plot-specific widget state otherwise supersedes the shared values.
    plot_override_suffixes = (
        "_override_plot_text",
        "_custom_title",
        "_custom_xlabel",
        "_custom_ylabel",
        "_show_legend",
        "_custom_legend_title",
        "_custom_legend_labels",
        "_override_series_colors",
        "_custom_series_colors",
    )
    protected_keys = {*setting_map, *setting_map.values()}
    rendered_plot_keys = tuple(
        str(key)
        for key in st.session_state.get("_swv_rendered_plot_keys", [])
    )
    for state_key in list(st.session_state):
        if (
            isinstance(state_key, str)
            and state_key not in protected_keys
            and any(
                state_key.startswith(f"{plot_key}_")
                for plot_key in rendered_plot_keys
            )
            and state_key.endswith(plot_override_suffixes)
        ):
            del st.session_state[state_key]
    st.session_state["_swv_metrics_all_settings_applied"] = True


def _render_all_metrics_plot_settings() -> None:
    """Render one form that controls the shared formatting of all Metrics plots."""
    with st.expander("All plot settings", expanded=False):
        st.caption(
            "These settings apply to every Metrics plot. Applying them clears "
            "conflicting changes saved in individual Plot settings popups."
        )
        with st.form("swv_metrics_all_plot_settings_form"):
            canvas_columns = st.columns(2)
            canvas_columns[0].number_input(
                "Plot width (px)", min_value=600, max_value=2200,
                value=int(st.session_state.get("swv_plot_width_px", 1200)),
                step=20, key="swv_metrics_all_width_px",
            )
            canvas_columns[1].number_input(
                "Plot height (px)", min_value=300, max_value=1200,
                value=int(st.session_state.get("swv_plot_height_px", 600)),
                step=20, key="swv_metrics_all_height_px",
            )

            style_columns = st.columns(2)
            style_columns[0].slider(
                "Plot text size", min_value=6.0, max_value=36.0,
                value=float(st.session_state.get("swv_plot_text_size_points", 10.0)),
                step=0.5, format="%.1f pt",
                key="swv_metrics_all_text_size_points",
            )
            style_columns[1].slider(
                "Line thickness", min_value=0.25, max_value=5.0,
                value=float(st.session_state.get("swv_plot_line_width_scale", 1.0)),
                step=0.25, format="%.2fx",
                key="swv_metrics_all_line_width_scale",
            )
            st.text_input(
                "Line color override",
                value=str(st.session_state.get("swv_plot_line_color_override", "")),
                key="swv_metrics_all_line_color_override",
                help="Leave blank to retain each plot's assigned series colors.",
            )

            frame_columns = st.columns(3)
            frame_columns[0].slider(
                "Outer margin", min_value=0, max_value=200,
                value=int(st.session_state.get("swv_plot_margin_px", 40)),
                step=5, format="%d px", key="swv_metrics_all_margin_px",
            )
            frame_columns[1].number_input(
                "Perimeter width", min_value=0.0, max_value=5.0,
                value=float(st.session_state.get("swv_plot_perimeter_width", 0.8)),
                step=0.2, key="swv_metrics_all_perimeter_width",
            )
            frame_columns[2].text_input(
                "Perimeter color",
                value=str(st.session_state.get("swv_plot_perimeter_color", "#222222")),
                key="swv_metrics_all_perimeter_color",
            )

            marker_columns = st.columns(2)
            marker_columns[0].slider(
                "Marker size", min_value=2.0, max_value=20.0,
                value=float(st.session_state.get("swv_plot_marker_size", 6.0)),
                step=1.0, key="swv_metrics_all_marker_size",
            )
            marker_columns[1].slider(
                "Marker opacity", min_value=0.05, max_value=1.0,
                value=float(st.session_state.get("swv_plot_marker_opacity", 0.85)),
                step=0.05, key="swv_metrics_all_marker_opacity",
            )
            visibility_columns = st.columns(2)
            visibility_columns[0].checkbox(
                "Show legends",
                value=bool(st.session_state.get("swv_plot_show_legend", True)),
                key="swv_metrics_all_show_legend",
            )
            visibility_columns[1].checkbox(
                "Show background grid",
                value=bool(st.session_state.get("swv_plot_show_grid", False)),
                key="swv_metrics_all_show_grid",
            )

            st.markdown("**Text overrides (optional)**")
            st.text_input(
                "Title override",
                value=str(st.session_state.get("swv_plot_title_override", "")),
                key="swv_metrics_all_title_override",
                help="Leave blank to retain each plot's own title.",
            )
            label_columns = st.columns(2)
            label_columns[0].text_input(
                "X-axis label override",
                value=str(st.session_state.get("swv_plot_x_label_override", "")),
                key="swv_metrics_all_x_label_override",
            )
            label_columns[1].text_input(
                "Y-axis label override",
                value=str(st.session_state.get("swv_plot_y_label_override", "")),
                key="swv_metrics_all_y_label_override",
            )
            st.form_submit_button(
                "Apply to all plots", type="primary", use_container_width=True,
                on_click=_apply_all_metrics_plot_settings,
            )
        if st.session_state.pop("_swv_metrics_all_settings_applied", False):
            st.success("Applied these settings to all Metrics plots.")


def _apply_all_langmuir_plot_settings() -> None:
    """Apply the Langmuir-only form without changing other Metrics plots."""
    suffixes = (
        "width_px", "height_px", "text_size_points", "line_width_scale",
        "line_color_override", "margin_px", "perimeter_width",
        "perimeter_color", "show_legend", "show_grid", "marker_size",
        "marker_opacity", "title_override", "x_label_override",
        "y_label_override",
    )
    for suffix in suffixes:
        source_key = f"swv_langmuir_all_{suffix}"
        if source_key in st.session_state:
            st.session_state[f"swv_langmuir_plot_{suffix}"] = (
                st.session_state[source_key]
            )

    plot_override_suffixes = (
        "_override_plot_text", "_custom_title", "_custom_xlabel",
        "_custom_ylabel", "_show_legend", "_custom_legend_title",
        "_custom_legend_labels", "_override_series_colors",
        "_custom_series_colors",
    )
    for plot_key in st.session_state.get("_swv_rendered_plot_keys", []):
        if not str(plot_key).startswith("titration_langmuir_"):
            continue
        for state_key in list(st.session_state):
            if (
                isinstance(state_key, str)
                and state_key.startswith(f"{plot_key}_")
                and state_key.endswith(plot_override_suffixes)
            ):
                del st.session_state[state_key]
    st.session_state["_swv_langmuir_all_settings_applied"] = True


def _render_all_langmuir_plot_settings() -> None:
    """Render shared controls scoped only to Langmuir fit plots."""
    with st.expander("Langmuir plot settings", expanded=True):
        st.caption(
            "These values affect every Langmuir fit plot and leave the other "
            "Metrics plots unchanged."
        )
        with st.form("swv_langmuir_all_plot_settings_form"):
            size_columns = st.columns(2)
            size_columns[0].number_input(
                "Plot width (px)", min_value=600, max_value=2200,
                value=int(st.session_state.get("swv_langmuir_plot_width_px", 1200)),
                step=20, key="swv_langmuir_all_width_px",
            )
            size_columns[1].number_input(
                "Plot height (px)", min_value=300, max_value=1200,
                value=int(st.session_state.get("swv_langmuir_plot_height_px", 600)),
                step=20, key="swv_langmuir_all_height_px",
            )
            style_columns = st.columns(2)
            style_columns[0].slider(
                "Text size", min_value=6.0, max_value=36.0,
                value=float(st.session_state.get("swv_langmuir_plot_text_size_points", 10.0)),
                step=0.5, format="%.1f pt", key="swv_langmuir_all_text_size_points",
            )
            style_columns[1].slider(
                "Line thickness", min_value=0.25, max_value=5.0,
                value=float(st.session_state.get("swv_langmuir_plot_line_width_scale", 1.0)),
                step=0.25, format="%.2fx", key="swv_langmuir_all_line_width_scale",
            )
            st.text_input(
                "Line color override",
                value=str(st.session_state.get("swv_langmuir_plot_line_color_override", "")),
                key="swv_langmuir_all_line_color_override",
                help="Leave blank to retain the Optimized and Manual method blues.",
            )
            frame_columns = st.columns(3)
            frame_columns[0].slider(
                "Outer margin", min_value=0, max_value=200,
                value=int(st.session_state.get("swv_langmuir_plot_margin_px", 40)),
                step=5, format="%d px", key="swv_langmuir_all_margin_px",
            )
            frame_columns[1].number_input(
                "Perimeter width", min_value=0.0, max_value=5.0,
                value=float(st.session_state.get("swv_langmuir_plot_perimeter_width", 0.8)),
                step=0.2, key="swv_langmuir_all_perimeter_width",
            )
            frame_columns[2].text_input(
                "Perimeter color",
                value=str(st.session_state.get("swv_langmuir_plot_perimeter_color", "#222222")),
                key="swv_langmuir_all_perimeter_color",
            )
            marker_columns = st.columns(2)
            marker_columns[0].slider(
                "Marker size", min_value=2.0, max_value=20.0,
                value=float(st.session_state.get("swv_langmuir_plot_marker_size", 6.0)),
                step=1.0, key="swv_langmuir_all_marker_size",
            )
            marker_columns[1].slider(
                "Marker opacity", min_value=0.05, max_value=1.0,
                value=float(st.session_state.get("swv_langmuir_plot_marker_opacity", 0.85)),
                step=0.05, key="swv_langmuir_all_marker_opacity",
            )
            visibility_columns = st.columns(2)
            visibility_columns[0].checkbox(
                "Show legends",
                value=bool(st.session_state.get("swv_langmuir_plot_show_legend", True)),
                key="swv_langmuir_all_show_legend",
            )
            visibility_columns[1].checkbox(
                "Show grid",
                value=bool(st.session_state.get("swv_langmuir_plot_show_grid", False)),
                key="swv_langmuir_all_show_grid",
            )
            st.markdown("**Text overrides (optional)**")
            st.text_input(
                "Title override",
                value=str(st.session_state.get("swv_langmuir_plot_title_override", "")),
                key="swv_langmuir_all_title_override",
            )
            label_columns = st.columns(2)
            label_columns[0].text_input(
                "X-axis label override",
                value=str(st.session_state.get("swv_langmuir_plot_x_label_override", "")),
                key="swv_langmuir_all_x_label_override",
            )
            label_columns[1].text_input(
                "Y-axis label override",
                value=str(st.session_state.get("swv_langmuir_plot_y_label_override", "")),
                key="swv_langmuir_all_y_label_override",
            )
            st.form_submit_button(
                "Apply to all Langmuir plots", type="primary",
                use_container_width=True,
                on_click=_apply_all_langmuir_plot_settings,
            )
        if st.session_state.pop("_swv_langmuir_all_settings_applied", False):
            st.success("Applied these settings to all Langmuir fit plots.")


def render_downloadable_pyplot(
    container,
    fig: plt.Figure,
    *,
    key: str,
    file_stem: str,
    dpi: int = 150,
    plot_kind: Optional[str] = None,
) -> None:
    settings_prefix = (
        "swv_langmuir_plot" if plot_kind == "langmuir_fit" else "swv_plot"
    )

    def active_plot_setting(suffix: str, default: Any) -> Any:
        return _resolve_swv_plot_setting(
            settings_prefix,
            suffix,
            default,
            inherit_metrics_style=plot_kind == "swv_trace",
        )

    if globals().get("analysis_mode") == "SWV":
        rendered_plot_keys = set(
            st.session_state.get("_swv_rendered_plot_keys", [])
        )
        rendered_plot_keys.add(key)
        st.session_state["_swv_rendered_plot_keys"] = sorted(rendered_plot_keys)
        _apply_swv_plot_formatting(
            fig,
            dpi,
            plot_kind=plot_kind,
            settings_prefix=settings_prefix,
        )
    plot_slot = container.empty()
    download_col, settings_col = container.columns([1, 1])

    primary_axis = fig.axes[0] if fig.axes else None
    default_title = primary_axis.get_title() if primary_axis is not None else ""
    default_xlabel = primary_axis.get_xlabel() if primary_axis is not None else ""
    default_ylabel = primary_axis.get_ylabel() if primary_axis is not None else ""
    legend = primary_axis.get_legend() if primary_axis is not None else None
    default_legend_title = (
        legend.get_title().get_text()
        if legend is not None
        else ""
    )
    default_legend_labels = (
        [text.get_text() for text in legend.get_texts()]
        if legend is not None
        else []
    )
    series_artists = []
    if primary_axis is not None:
        legend_artists, legend_artist_labels = primary_axis.get_legend_handles_labels()
        series_artists = [
            artist
            for artist, label in zip(legend_artists, legend_artist_labels)
            if str(label) and not str(label).startswith("_")
        ]
    default_series_colors = []
    for artist in series_artists:
        artist_color = None
        for getter_name in ("get_color", "get_facecolor", "get_edgecolor"):
            getter = getattr(artist, getter_name, None)
            if getter is None:
                continue
            try:
                artist_color = getter()
                if isinstance(artist_color, np.ndarray) and artist_color.ndim > 1:
                    artist_color = artist_color[0] if len(artist_color) else None
                if artist_color is not None:
                    break
            except (TypeError, ValueError):
                continue
        try:
            default_series_colors.append(to_hex(artist_color, keep_alpha=True))
        except (TypeError, ValueError):
            default_series_colors.append(str(artist_color or ""))
    acceptance_regions = (
        [
            collection for collection in primary_axis.collections
            if bool(getattr(collection, "_swv_preserve_alpha", False))
        ]
        if primary_axis is not None else []
    )
    output_dpi = int(dpi)
    display_width_px = int(active_plot_setting("width_px", 1200))
    if plot_kind == "swv_trace":
        display_width_px = max(
            1,
            int(round(display_width_px * _SWV_OVERLAY_WIDTH_SCALE)),
        )
    reconstruction_width_px = display_width_px
    reconstruction_height_px = int(st.session_state.get("swv_plot_height_px", 600))
    reconstruction_text_size = float(
        st.session_state.get("swv_plot_text_size_points", 10.0)
    )
    reconstruction_title_size = reconstruction_text_size * 1.2
    reconstruction_x_label_size = reconstruction_text_size
    reconstruction_y_label_size = reconstruction_text_size
    reconstruction_tick_size = reconstruction_text_size * 0.9
    reconstruction_legend_size = reconstruction_text_size * 0.8
    reconstruction_line_scale = 1.0
    reconstruction_line_color = ""
    reconstruction_margin_px = int(st.session_state.get("swv_plot_margin_px", 40))
    reconstruction_perimeter_width = float(
        st.session_state.get("swv_plot_perimeter_width", 0.8)
    )
    reconstruction_perimeter_color = str(
        st.session_state.get("swv_plot_perimeter_color", "#222222")
    )
    reconstruction_marker_size = float(
        st.session_state.get("swv_plot_marker_size", 6.0)
    )
    reconstruction_marker_opacity = float(
        st.session_state.get("swv_plot_marker_opacity", 0.85)
    )
    reconstruction_show_grid = bool(
        st.session_state.get("swv_plot_show_grid", False)
    )
    reconstruction_uses_doubling_levels = bool(
        primary_axis is not None
        and getattr(
            primary_axis,
            "_swv_concentration_doubling_scale",
            False,
        )
    )
    reconstruction_concentration_scale = (
        "Doubling levels"
        if reconstruction_uses_doubling_levels
        else (
            "Logarithmic"
            if primary_axis is not None
            and primary_axis.get_yscale() in {"log", "symlog"}
            else "Linear"
        )
    )
    manual_reconstruction_x_limits = False
    reconstruction_x_min = None
    reconstruction_x_max = None
    reconstruction_x_tick_positions_text = ""
    reconstruction_x_tick_labels_text = ""
    reconstruction_y_tick_positions_text = ""
    reconstruction_y_tick_labels_text = ""

    with settings_col.popover("Plot settings", use_container_width=True):
        override_plot_text = st.checkbox(
            "Override plot text",
            key=f"{key}_override_plot_text",
        )
        custom_title = st.text_input(
            "Title",
            value=default_title,
            key=f"{key}_custom_title",
            disabled=not override_plot_text,
        )
        custom_xlabel = st.text_input(
            "X-axis label",
            value=default_xlabel,
            key=f"{key}_custom_xlabel",
            disabled=not override_plot_text,
        )
        custom_ylabel = st.text_input(
            "Y-axis label",
            value=default_ylabel,
            key=f"{key}_custom_ylabel",
            disabled=not override_plot_text,
        )
        show_legend = st.checkbox(
            "Show legend",
            value=legend is not None and legend.get_visible(),
            key=f"{key}_show_legend",
            disabled=legend is None,
        )
        custom_legend_title = st.text_input(
            "Legend title",
            value=default_legend_title,
            key=f"{key}_custom_legend_title",
            disabled=not override_plot_text or legend is None,
        )
        custom_legend_labels_text = st.text_area(
            "Legend entries (one per line)",
            value="\n".join(default_legend_labels),
            height=min(180, max(80, 28 * len(default_legend_labels))),
            key=f"{key}_custom_legend_labels",
            disabled=not override_plot_text or legend is None,
            help="Entries correspond to the existing legend items in their current order.",
        )
        override_series_colors = st.checkbox(
            "Override series colors",
            key=f"{key}_override_series_colors",
            disabled=not series_artists,
            help="Applies colors to plotted series in their current legend order.",
        )
        custom_series_colors_text = st.text_area(
            "Series colors (one per line)",
            value="\n".join(default_series_colors),
            height=min(180, max(80, 28 * len(default_series_colors))),
            key=f"{key}_custom_series_colors",
            disabled=not override_series_colors or not series_artists,
            help=(
                "Use Matplotlib/CSS colors such as tab:blue, crimson, #4b0082, "
                "or rgba-compatible hex values."
            ),
        )
        requested_series_colors = [
            color.strip() for color in custom_series_colors_text.splitlines()
        ]
        invalid_series_colors = [
            color for color in requested_series_colors
            if color and not is_color_like(color)
        ]
        if override_series_colors and invalid_series_colors:
            st.caption(
                "Invalid color(s) ignored: " + ", ".join(invalid_series_colors)
            )
        acceptance_region_alpha = None
        if acceptance_regions:
            current_alpha = acceptance_regions[0].get_alpha()
            acceptance_region_alpha = st.slider(
                "20% acceptance-region alpha",
                min_value=0.0,
                max_value=1.0,
                value=(
                    float(current_alpha)
                    if current_alpha is not None else 0.10
                ),
                step=0.05,
                key=f"{key}_acceptance_region_alpha",
                help=(
                    "Controls the opacity of the shaded ±20% prediction-error region."
                ),
            )
        manual_reconstruction_y_limits = False
        reconstruction_y_min = None
        reconstruction_y_max = None
        if plot_kind == "concentration_reconstruction" and primary_axis is not None:
            st.divider()
            st.markdown("**Canvas and export**")
            reconstruction_width_px = int(st.number_input(
                "Plot width (px)",
                min_value=400,
                max_value=3000,
                value=int(reconstruction_width_px),
                step=20,
                key=f"{key}_plot_width_px",
            ))
            reconstruction_height_px = int(st.number_input(
                "Plot height (px)",
                min_value=240,
                max_value=2000,
                value=int(reconstruction_height_px),
                step=20,
                key=f"{key}_plot_height_px",
            ))
            output_dpi = int(st.slider(
                "Download resolution",
                min_value=72,
                max_value=600,
                value=int(dpi),
                step=1,
                format="%d DPI",
                key=f"{key}_plot_dpi",
            ))

            st.markdown("**Text**")
            reconstruction_text_size = float(st.slider(
                "Plot text size",
                min_value=6.0,
                max_value=72.0,
                value=float(reconstruction_text_size),
                step=0.5,
                format="%.1f pt",
                key=f"{key}_text_size",
                help="Sets the base size used by plot text, as in the BO viewer.",
            ))
            use_individual_text_sizes = st.checkbox(
                "Set individual text sizes",
                key=f"{key}_individual_text_sizes",
                help=(
                    "When off, the base text-size control scales the title, axis "
                    "labels, ticks, and legend together."
                ),
            )
            reconstruction_title_size = float(st.number_input(
                "Title size",
                min_value=1.0,
                max_value=100.0,
                value=float(reconstruction_text_size * 1.2),
                step=0.5,
                key=f"{key}_title_size",
                disabled=not use_individual_text_sizes,
            ))
            reconstruction_tick_size = float(st.number_input(
                "Tick size",
                min_value=1.0,
                max_value=100.0,
                value=float(reconstruction_text_size * 0.9),
                step=0.5,
                key=f"{key}_tick_size",
                disabled=not use_individual_text_sizes,
            ))
            reconstruction_x_label_size = float(st.number_input(
                "X label size",
                min_value=1.0,
                max_value=100.0,
                value=float(reconstruction_text_size),
                step=0.5,
                key=f"{key}_x_label_size",
                disabled=not use_individual_text_sizes,
            ))
            reconstruction_y_label_size = float(st.number_input(
                "Y label size",
                min_value=1.0,
                max_value=100.0,
                value=float(reconstruction_text_size),
                step=0.5,
                key=f"{key}_y_label_size",
                disabled=not use_individual_text_sizes,
            ))
            reconstruction_legend_size = float(st.number_input(
                "Legend text size",
                min_value=1.0,
                max_value=100.0,
                value=float(reconstruction_text_size * 0.8),
                step=0.5,
                key=f"{key}_legend_size",
                disabled=not use_individual_text_sizes,
            ))
            if not use_individual_text_sizes:
                reconstruction_title_size = reconstruction_text_size * 1.2
                reconstruction_x_label_size = reconstruction_text_size
                reconstruction_y_label_size = reconstruction_text_size
                reconstruction_tick_size = reconstruction_text_size * 0.9
                reconstruction_legend_size = reconstruction_text_size * 0.8

            st.markdown("**Lines, markers, and frame**")
            reconstruction_line_scale = float(st.slider(
                "Line thickness",
                min_value=0.25,
                max_value=5.0,
                value=1.0,
                step=0.25,
                format="%.2fx",
                key=f"{key}_line_width_scale",
                help="Scales the existing line widths for this plot.",
            ))
            reconstruction_line_color = st.text_input(
                "Line color override",
                key=f"{key}_line_color_override",
                help=(
                    "Optional Matplotlib/CSS color applied to every line. Leave "
                    "blank to retain the plot or per-series colors."
                ),
            ).strip()
            if reconstruction_line_color and not is_color_like(reconstruction_line_color):
                st.caption("Unrecognized line color; existing colors will be retained.")
            reconstruction_marker_size = float(st.slider(
                "Marker size",
                min_value=2.0,
                max_value=30.0,
                value=float(reconstruction_marker_size),
                step=1.0,
                key=f"{key}_marker_size",
            ))
            reconstruction_marker_opacity = float(st.slider(
                "Marker opacity",
                min_value=0.05,
                max_value=1.0,
                value=float(reconstruction_marker_opacity),
                step=0.05,
                key=f"{key}_marker_opacity",
            ))
            reconstruction_margin_px = int(st.slider(
                "Outer plot margin",
                min_value=0,
                max_value=260,
                value=int(reconstruction_margin_px),
                step=5,
                format="%d px",
                key=f"{key}_plot_margin_px",
            ))
            reconstruction_perimeter_width = float(st.number_input(
                "Perimeter width",
                min_value=0.0,
                max_value=10.0,
                value=float(reconstruction_perimeter_width),
                step=0.2,
                key=f"{key}_perimeter_width",
            ))
            reconstruction_perimeter_color = st.text_input(
                "Perimeter color",
                value=reconstruction_perimeter_color,
                key=f"{key}_perimeter_color",
            ).strip()
            if (
                reconstruction_perimeter_color
                and not is_color_like(reconstruction_perimeter_color)
            ):
                st.caption("Unrecognized perimeter color; #222222 will be used.")
            reconstruction_show_grid = st.checkbox(
                "Show background grid",
                value=bool(reconstruction_show_grid),
                key=f"{key}_show_grid",
            )

            with st.expander("Axis tick label overrides", expanded=False):
                st.caption(
                    "Enter comma-separated positions and displayed labels. Leave "
                    "positions blank to relabel the existing ticks in order."
                )
                reconstruction_x_tick_positions_text = st.text_input(
                    "X tick positions",
                    key=f"{key}_x_tick_positions",
                    placeholder="0, 10, 20",
                )
                reconstruction_x_tick_labels_text = st.text_input(
                    "X tick labels",
                    key=f"{key}_x_tick_labels",
                    placeholder="Start, 10, 20",
                )
                reconstruction_y_tick_positions_text = st.text_input(
                    "Y tick positions",
                    key=f"{key}_y_tick_positions",
                    placeholder="-100, 0, 100",
                )
                reconstruction_y_tick_labels_text = st.text_input(
                    "Y tick labels",
                    key=f"{key}_y_tick_labels",
                    placeholder="-100, 0, 100",
                )

            st.markdown("**Displayed axis limits**")
            if reconstruction_uses_doubling_levels:
                st.caption(
                    "The concentration axis uses doubling levels: buffer is 0, "
                    "the lowest selected dose is 1, and every doubling adds 1."
                )
            else:
                reconstruction_concentration_scale = st.radio(
                    "Concentration-axis scale",
                    ["Linear", "Logarithmic"],
                    index=(
                        0 if reconstruction_concentration_scale == "Linear" else 1
                    ),
                    key=f"{key}_concentration_axis_scale",
                    help=(
                        "Logarithmic mode automatically fits a positive Y range to "
                        "the LOD-floored data and displayed uncertainty bounds."
                    ),
                )
            manual_reconstruction_x_limits = st.checkbox(
                "Manual x-axis limits",
                key=f"{key}_manual_x_limits",
                help="Overrides the displayed measurement-number range for this plot only.",
            )
            default_x_min, default_x_max = primary_axis.get_xlim()
            reconstruction_x_min = st.number_input(
                "X minimum",
                value=float(default_x_min),
                key=f"{key}_x_min",
                disabled=not manual_reconstruction_x_limits,
            )
            reconstruction_x_max = st.number_input(
                "X maximum",
                value=float(default_x_max),
                key=f"{key}_x_max",
                disabled=not manual_reconstruction_x_limits,
            )
            if (
                manual_reconstruction_x_limits
                and reconstruction_x_min >= reconstruction_x_max
            ):
                st.caption("X minimum must be smaller than X maximum.")
            manual_reconstruction_y_limits = st.checkbox(
                "Manual y-axis limits",
                key=f"{key}_manual_y_limits",
                help=(
                    "Overrides the displayed doubling-level range for this plot only."
                    if reconstruction_uses_doubling_levels
                    else "Overrides the reconstructed concentration range for this plot only."
                ),
            )
            default_y_min, default_y_max = primary_axis.get_ylim()
            reconstruction_y_min = st.number_input(
                (
                    "Y minimum (doubling level)"
                    if reconstruction_uses_doubling_levels else "Y minimum"
                ),
                value=float(default_y_min),
                key=f"{key}_y_min",
                disabled=not manual_reconstruction_y_limits,
            )
            reconstruction_y_max = st.number_input(
                (
                    "Y maximum (doubling level)"
                    if reconstruction_uses_doubling_levels else "Y maximum"
                ),
                value=float(default_y_max),
                key=f"{key}_y_max",
                disabled=not manual_reconstruction_y_limits,
            )
            if (
                manual_reconstruction_y_limits
                and (
                    reconstruction_y_min >= reconstruction_y_max
                    or (
                        reconstruction_concentration_scale == "Logarithmic"
                        and reconstruction_y_min <= 0
                    )
                )
            ):
                st.caption(
                    "Y minimum must be positive for logarithmic scaling and "
                    "smaller than Y maximum."
                    if reconstruction_concentration_scale == "Logarithmic"
                    else "Y minimum must be smaller than Y maximum."
                )

    if primary_axis is not None:
        if override_plot_text:
            primary_axis.set_title(custom_title)
            primary_axis.set_xlabel(custom_xlabel)
            primary_axis.set_ylabel(custom_ylabel)
        if legend is not None:
            legend.set_visible(show_legend)
            if override_plot_text:
                legend.get_title().set_text(custom_legend_title)
                custom_legend_labels = custom_legend_labels_text.splitlines()
                for index, legend_text in enumerate(legend.get_texts()):
                    if index < len(custom_legend_labels):
                        legend_text.set_text(custom_legend_labels[index])
        if override_series_colors:
            legend_handles = []
            if legend is not None:
                legend_handles = list(
                    getattr(
                        legend,
                        "legend_handles",
                        getattr(legend, "legendHandles", []),
                    )
                )
            for index, artist in enumerate(series_artists):
                if index >= len(requested_series_colors):
                    break
                color = requested_series_colors[index]
                if color and is_color_like(color):
                    if hasattr(artist, "set_color"):
                        artist.set_color(color)
                    if hasattr(artist, "set_facecolor"):
                        artist.set_facecolor(color)
                    if hasattr(artist, "set_edgecolor"):
                        artist.set_edgecolor(color)
                    if index < len(legend_handles):
                        legend_handle = legend_handles[index]
                        if hasattr(legend_handle, "set_color"):
                            legend_handle.set_color(color)
                        if hasattr(legend_handle, "set_facecolor"):
                            legend_handle.set_facecolor(color)
                        if hasattr(legend_handle, "set_edgecolor"):
                            legend_handle.set_edgecolor(color)
        if acceptance_region_alpha is not None:
            for acceptance_region in acceptance_regions:
                acceptance_region.set_alpha(acceptance_region_alpha)
        if plot_kind == "concentration_reconstruction":
            fig.set_size_inches(
                reconstruction_width_px / output_dpi,
                reconstruction_height_px / output_dpi,
                forward=True,
            )
            display_width_px = reconstruction_width_px
            primary_axis.title.set_fontsize(reconstruction_title_size)
            primary_axis.xaxis.label.set_fontsize(reconstruction_x_label_size)
            primary_axis.yaxis.label.set_fontsize(reconstruction_y_label_size)
            primary_axis.tick_params(labelsize=reconstruction_tick_size)
            if reconstruction_concentration_scale == "Doubling levels":
                # The plotting function has already installed concentration
                # tick positions/labels on a linear transformed-data axis.
                # Calling set_yscale("linear") again resets that fixed locator
                # and exposes the internal doubling-level coordinates.
                pass
            elif reconstruction_concentration_scale == "Linear":
                primary_axis.set_yscale("linear")
            else:
                primary_axis.set_yscale("log")
                if not manual_reconstruction_y_limits:
                    automatic_log_limits = getattr(
                        primary_axis,
                        "_swv_reconstruction_log_ylim",
                        None,
                    )
                    if automatic_log_limits is not None:
                        primary_axis.set_ylim(*automatic_log_limits)
            for errorbar_artist in (
                *primary_axis.lines,
                *primary_axis.collections,
            ):
                if bool(
                    getattr(
                        errorbar_artist,
                        "_swv_concentration_errorbar",
                        False,
                    )
                ):
                    errorbar_artist.set_transform(primary_axis.transData)
            for annotation in primary_axis.texts:
                if not bool(
                    getattr(annotation, "_swv_preserve_fontsize", False)
                ):
                    annotation.set_fontsize(reconstruction_text_size)
            if reconstruction_show_grid:
                primary_axis.grid(True, alpha=0.2)
            else:
                primary_axis.grid(False)
            perimeter_color = (
                reconstruction_perimeter_color
                if is_color_like(reconstruction_perimeter_color)
                else "#222222"
            )
            for spine in primary_axis.spines.values():
                spine.set_visible(reconstruction_perimeter_width > 0)
                spine.set_linewidth(reconstruction_perimeter_width)
                spine.set_edgecolor(perimeter_color)
            valid_line_color = (
                reconstruction_line_color
                if is_color_like(reconstruction_line_color)
                else None
            )
            for line in primary_axis.lines:
                line.set_linewidth(max(
                    0.1,
                    line.get_linewidth() * reconstruction_line_scale,
                ))
                if valid_line_color:
                    line.set_color(valid_line_color)
                if line.get_marker() not in (None, "", "None", "none", " "):
                    line.set_markersize(reconstruction_marker_size)
            for collection in primary_axis.collections:
                if bool(getattr(collection, "_swv_preserve_alpha", False)):
                    continue
                if hasattr(collection, "get_sizes") and hasattr(collection, "set_sizes"):
                    sizes = collection.get_sizes()
                    if len(sizes):
                        collection.set_sizes(np.full(
                            len(sizes),
                            reconstruction_marker_size ** 2,
                            dtype=float,
                        ))
                        collection.set_alpha(reconstruction_marker_opacity)
            current_legend = primary_axis.get_legend()
            if current_legend is not None:
                for legend_text in current_legend.get_texts():
                    legend_text.set_fontsize(reconstruction_legend_size)
                current_legend.get_title().set_fontsize(reconstruction_legend_size)
                current_legend_handles = list(
                    getattr(
                        current_legend,
                        "legend_handles",
                        getattr(current_legend, "legendHandles", []),
                    )
                )
                for legend_handle in current_legend_handles:
                    if hasattr(legend_handle, "set_markersize"):
                        legend_handle.set_markersize(reconstruction_marker_size)
                    if hasattr(legend_handle, "get_sizes") and hasattr(
                        legend_handle, "set_sizes"
                    ):
                        sizes = legend_handle.get_sizes()
                        if len(sizes):
                            legend_handle.set_sizes(np.full(
                                len(sizes),
                                reconstruction_marker_size ** 2,
                                dtype=float,
                            ))

            for axis_key, positions_text, labels_text in (
                (
                    "x",
                    reconstruction_x_tick_positions_text,
                    reconstruction_x_tick_labels_text,
                ),
                (
                    "y",
                    reconstruction_y_tick_positions_text,
                    reconstruction_y_tick_labels_text,
                ),
            ):
                labels = _parse_text_list(labels_text)
                if not labels:
                    continue
                matplotlib_axis = (
                    primary_axis.xaxis if axis_key == "x" else primary_axis.yaxis
                )
                positions = _parse_numeric_list(positions_text)
                if not positions:
                    positions = list(matplotlib_axis.get_ticklocs())
                count = min(len(positions), len(labels))
                if count:
                    matplotlib_axis.set_ticks(
                        positions[:count],
                        labels=labels[:count],
                    )

        if (
            manual_reconstruction_x_limits
            and reconstruction_x_min is not None
            and reconstruction_x_max is not None
            and reconstruction_x_min < reconstruction_x_max
        ):
            primary_axis.set_xlim(
                float(reconstruction_x_min),
                float(reconstruction_x_max),
            )
        if (
            manual_reconstruction_y_limits
            and reconstruction_y_min is not None
            and reconstruction_y_max is not None
            and reconstruction_y_min < reconstruction_y_max
            and (
                reconstruction_concentration_scale != "Logarithmic"
                or reconstruction_y_min > 0
            )
        ):
            primary_axis.set_ylim(
                float(reconstruction_y_min),
                float(reconstruction_y_max),
            )
        if not bool(getattr(fig, "_swv_manual_layout", False)):
            try:
                margin_px = (
                    reconstruction_margin_px
                    if plot_kind == "concentration_reconstruction"
                    else int(active_plot_setting("margin_px", 40))
                )
                fig.tight_layout(pad=max(0.5, margin_px / 30.0))
            except (RuntimeError, ValueError):
                pass
        if globals().get("analysis_mode") == "SWV":
            _position_swv_colorbars(fig)

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=output_dpi,
        bbox_inches=(
            None
            if plot_kind in {
                "concentration_accuracy",
                "concentration_measurement",
                "concentration_reconstruction",
                "swv_trace",
            }
            else "tight"
        ),
    )
    if globals().get("analysis_mode") == "SWV":
        plot_slot.image(
            buffer.getvalue(),
            width=display_width_px,
        )
    else:
        plot_slot.pyplot(fig)
    download_col.download_button(
        "Download plot",
        data=buffer.getvalue(),
        file_name=f"{safe_download_stem(file_stem)}.png",
        mime="image/png",
        key=f"{key}_download",
        use_container_width=True,
    )
    plt.close(fig)

# 
# Page config
# 
st.set_page_config(
    page_title="Electrochemistry Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Leave room below Streamlit's fixed top header so status/progress UI is not clipped. */
    .block-container { padding-top: 3.5rem; }
    div[data-testid="stSidebarContent"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# 
# Analysis runner. Result reuse is managed explicitly in session state so
# progress widgets are never called from inside Streamlit's cache replay.
# 
def run_batch_dispatch(
    analysis_mode,
    folders,          # tuple so it's hashable
    crop_range,
    smooth_window,
    smooth_polyorder,
    minima_search_window_V,
    use_prominent_minima,
    use_double_correction,
    min_peak_height_uA,
    min_start_voltage,
    scan_windows,
    scan_range,
    time_range,
    compute_skew,
    compute_wavelet_energy,
    compute_wavelet_denoised_trace,
    use_wavelet_for_correction,
    _parallel_workers,
    edge_trim_fraction,
    min_peak_prominence_uA,
    input_signature,
    _progress_callback=None,
):
    # The signature is included in the caller's explicit session cache key.
    del input_signature
    if analysis_mode == "CV":
        return run_cv_batch(
            folders=list(folders),
            crop_range=crop_range,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            edge_trim_fraction=edge_trim_fraction,
            min_peak_prominence_uA=min_peak_prominence_uA,
            scan_windows=scan_windows,
            scan_range=scan_range,
        )

    return run_batch(
        folders=list(folders),
        crop_range=crop_range,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        minima_search_window_V=minima_search_window_V,
        use_prominent_minima=use_prominent_minima,
        use_double_correction=use_double_correction,
        min_peak_height_uA=min_peak_height_uA,
        min_start_voltage=min_start_voltage,
        scan_windows=scan_windows,
        scan_range=scan_range,
        time_range=time_range,
        compute_skew=compute_skew,
        compute_wavelet_energy=compute_wavelet_energy,
        compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
        use_wavelet_for_correction=use_wavelet_for_correction,
        parallel_workers=_parallel_workers,
        progress_callback=_progress_callback,
    )


def collect_titration_rows(
    all_results,
    metric_cfg,
    channels,
    vlines,
    scan_range,
    edge_trim_fraction,
    concentration_unit="",
    vlines_by_channel=None,
    baseline_mode="none",
    included_step_labels=None,
    remove_extreme_outliers=False,
):
    rows = []
    for label, (metric_key, ylabel) in metric_cfg.items():
        metric_rows = build_titration_step_table(
            all_results,
            metric=metric_key,
            vlines=vlines,
            channels=channels,
            vlines_by_channel=vlines_by_channel,
            scan_range=scan_range,
            edge_trim_fraction=edge_trim_fraction,
            concentration_unit=concentration_unit,
            baseline_mode=baseline_mode,
            included_step_labels=included_step_labels,
            remove_extreme_outliers=remove_extreme_outliers,
        )
        for row in metric_rows:
            rows.append({
                "metric_label": label,
                "metric_key": metric_key,
                "metric_ylabel": ylabel,
                **row,
            })
    return rows


def collect_langmuir_summary_rows(
    all_results,
    metric_cfg,
    channels,
    vlines,
    scan_range,
    edge_trim_fraction,
    step_concentrations=None,
    concentration_unit="",
    vlines_by_channel=None,
    baseline_mode="none",
    included_step_labels=None,
    remove_extreme_outliers=False,
):
    rows = []
    for label, (metric_key, ylabel) in metric_cfg.items():
        if not supports_langmuir(metric_key):
            continue
        metric_rows = build_titration_langmuir_summary_table(
            all_results,
            metric=metric_key,
            vlines=vlines,
            channels=channels,
            vlines_by_channel=vlines_by_channel,
            scan_range=scan_range,
            edge_trim_fraction=edge_trim_fraction,
            step_concentrations=step_concentrations,
            concentration_unit=concentration_unit,
            baseline_mode=baseline_mode,
            included_step_labels=included_step_labels,
            remove_extreme_outliers=remove_extreme_outliers,
        )
        for row in metric_rows:
            rows.append({
                "metric_label": label,
                "metric_key": metric_key,
                "metric_ylabel": ylabel,
                **row,
            })
    return rows


def collect_titration_measurement_accuracy_rows(
    all_results,
    metric_cfg,
    channels,
    vlines,
    scan_range,
    edge_trim_fraction,
    concentration_unit="",
    vlines_by_channel=None,
    baseline_mode="none",
    included_step_labels=None,
    remove_extreme_outliers=False,
    include_buffer_measurements=False,
):
    rows = []
    for label, (metric_key, ylabel) in metric_cfg.items():
        if not supports_langmuir(metric_key):
            continue
        metric_rows = build_titration_measurement_accuracy_table(
            all_results,
            metric=metric_key,
            vlines=vlines,
            channels=channels,
            vlines_by_channel=vlines_by_channel,
            scan_range=scan_range,
            edge_trim_fraction=edge_trim_fraction,
            concentration_unit=concentration_unit,
            baseline_mode=baseline_mode,
            included_step_labels=included_step_labels,
            remove_extreme_outliers=remove_extreme_outliers,
            include_buffer_measurements=include_buffer_measurements,
        )
        for row in metric_rows:
            rows.append({
                "metric_label": label,
                "metric_key": metric_key,
                "metric_ylabel": ylabel,
                **row,
            })
    return rows


def _serialize_vlines(vlines: List[Tuple[float, str]]) -> str:
    return json.dumps(
        [
            {"scan": float(scan_value), "label": str(label)}
            for scan_value, label in vlines
        ],
        separators=(",", ":"),
    )


def _format_channels(channels: Optional[List[int]]) -> str:
    if not channels:
        return ""
    return ",".join(str(int(ch)) for ch in channels)


def _serialize_channels(channels: Optional[List[int]]) -> str:
    return json.dumps(
        [int(ch) for ch in channels] if channels else [],
        separators=(",", ":"),
    )


def build_export_metadata(
    analysis_mode: str,
    crop_range: Tuple[float, float],
    smooth_window: int,
    smooth_polyorder: int,
    active_vlines: List[Tuple[float, str]],
    selected_channels: Optional[List[int]] = None,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
    time_range: Optional[Tuple[datetime, datetime]] = None,
    minima_search_window_V: Optional[float] = None,
    min_peak_height_uA: Optional[float] = None,
    min_start_voltage_V: Optional[float] = None,
    edge_trim_fraction: Optional[float] = None,
    min_peak_prominence_uA: Optional[float] = None,
    titration_edge_trim_fraction: Optional[float] = None,
    peak_height_source_key: Optional[str] = None,
    peak_height_source_label: Optional[str] = None,
    compute_wavelet_denoised_trace: Optional[bool] = None,
    use_wavelet_for_correction: Optional[bool] = None,
    titration_concentration_unit: Optional[str] = None,
    titration_baseline_mode: Optional[str] = None,
    titration_included_step_labels: Optional[List[str]] = None,
    remove_extreme_titration_outliers: Optional[bool] = None,
    show_titration_uloq: Optional[bool] = None,
    show_titration_lod: Optional[bool] = None,
) -> dict:
    metadata = {
        "analysis_crop_min_V": float(crop_range[0]),
        "analysis_crop_max_V": float(crop_range[1]),
        "analysis_smooth_window": int(smooth_window),
        "analysis_smooth_polyorder": int(smooth_polyorder),
        "analysis_scan_range_start": (
            float(scan_range[0]) if scan_range is not None else None
        ),
        "analysis_scan_range_end": (
            float(scan_range[1]) if scan_range is not None else None
        ),
        "analysis_scan_windows": format_scan_windows(scan_windows) if scan_windows else "",
        "analysis_filename_time_start": (
            time_range[0].isoformat(sep=" ") if time_range is not None else None
        ),
        "analysis_filename_time_end": (
            time_range[1].isoformat(sep=" ") if time_range is not None else None
        ),
        "analysis_vline_count": int(len(active_vlines)),
        "analysis_vlines_json": _serialize_vlines(active_vlines),
        "analysis_selected_channel_count": (
            int(len(selected_channels)) if selected_channels is not None else None
        ),
        "analysis_selected_channels": _format_channels(selected_channels),
        "analysis_selected_channels_json": _serialize_channels(selected_channels),
    }

    if analysis_mode == "SWV":
        metadata.update(
            analysis_minima_search_window_V=(
                float(minima_search_window_V)
                if minima_search_window_V is not None else None
            ),
            analysis_min_peak_height_uA=(
                float(min_peak_height_uA)
                if min_peak_height_uA is not None else None
            ),
            analysis_min_start_voltage_V=(
                float(min_start_voltage_V)
                if min_start_voltage_V is not None else None
            ),
            analysis_titration_edge_trim_fraction=(
                float(titration_edge_trim_fraction)
                if titration_edge_trim_fraction is not None else None
            ),
            analysis_titration_concentration_unit=titration_concentration_unit or "",
            analysis_titration_baseline_mode=titration_baseline_mode or "none",
            analysis_titration_included_step_labels_json=json.dumps(
                titration_included_step_labels
                if titration_included_step_labels is not None
                else [],
                separators=(",", ":"),
            ),
            analysis_remove_extreme_titration_outliers=bool(
                remove_extreme_titration_outliers
            ),
            analysis_show_titration_uloq=bool(show_titration_uloq),
            analysis_show_titration_lod=bool(show_titration_lod),
            analysis_peak_height_source_key=peak_height_source_key or "",
            analysis_peak_height_source_label=peak_height_source_label or "",
            analysis_compute_wavelet_denoised_trace=bool(compute_wavelet_denoised_trace),
            analysis_use_wavelet_for_correction=bool(use_wavelet_for_correction),
        )
    else:
        metadata.update(
            analysis_edge_trim_fraction=(
                float(edge_trim_fraction)
                if edge_trim_fraction is not None else None
            ),
            analysis_min_peak_prominence_uA=(
                float(min_peak_prominence_uA)
                if min_peak_prominence_uA is not None else None
            ),
        )

    return metadata


def export_file_name(analysis_mode: str, stem: str) -> str:
    prefix = "cv" if analysis_mode == "CV" else "swv"
    return f"{prefix}_{stem}.csv"


def build_results_export_rows(analysis_mode: str, results: List[dict]) -> Tuple[List[dict], List[str]]:
    if analysis_mode == "CV":
        export_keys = [
            "channel", "ec_label", "measurement_index", "scan_number", "original_scan_number",
            "cycle_count_in_file", "method_nscans", "timestamp", "file_name", "status",
            "oxidation_peak_voltage", "oxidation_peak_current", "oxidation_peak_prominence",
            "reduction_peak_voltage", "reduction_peak_current", "reduction_peak_prominence",
            "peak_separation_V", "peak_current_ratio", "loop_area_abs",
            "oxidation_peak_voltage_drift", "reduction_peak_voltage_drift",
            "peak_separation_drift", "loop_area_abs_drift", "error",
        ]
    else:
        export_keys = [
            "channel", "swv_method_group", "frequency_hz",
            "swv_sweep_start_V", "swv_sweep_end_V", "swv_step_size_V", "swv_amplitude_V",
            "scan_number", "filtered_source_scan_number", "original_scan_number",
            "timestamp", "measurement_time", "file_name", "status",
            "peak_voltage", "peak_current_selected", "peak_current_background_drift_corrected",
            "peak_current_background_recentered", "peak_current", "peak_current_smoothed_corrected",
            "peak_current_raw", "bracket_width_V",
            "skew", "peak_offset_norm", "wavelet_energy",
            "background_current_rms", "background_current_median", "background_drift_rms_reference",
            "background_current_median_reference", "background_current_offset_uA", "background_drift_percent",
            "peak_voltage_drift", "bracket_width_drift", "skew_drift", "peak_offset_norm_drift", "error",
        ]

    csv_rows = []
    for r in results:
        row = {}
        for k in export_keys:
            if analysis_mode == "SWV" and k == "frequency_hz":
                row[k] = r.get("swv_frequency_hz")
            else:
                row[k] = r.get(k)
        csv_rows.append(row)
    return csv_rows, export_keys


def _safe_folder_name(text: str, fallback: str = "analysis") -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    safe = "_".join(part for part in safe.split("_") if part)
    return safe[:80] or fallback


def build_experiment_export_payload(
    analysis_mode: str,
    results: List[dict],
    export_metadata: dict,
    metric_cfg: Dict[str, Tuple[str, str]],
    channels: List[int],
    active_vlines: List[Tuple[float, str]],
    scan_range: Optional[Tuple[int, int]],
    enable_titration_analysis: bool,
    titration_ready: bool,
    titration_edge_trim_fraction: float,
    fit_titration_langmuir: bool,
    titration_concentration_unit: str,
    titration_results: Optional[List[dict]] = None,
    titration_channels: Optional[List[Any]] = None,
    titration_vlines: Optional[List[Tuple[float, str]]] = None,
    titration_vlines_by_channel: Optional[Dict[Any, List[Tuple[float, str]]]] = None,
    titration_scan_range: Optional[Tuple[int, int]] = None,
    titration_baseline_mode: str = "none",
    titration_included_step_labels: Optional[List[str]] = None,
    remove_extreme_titration_outliers: bool = False,
) -> Dict[str, pd.DataFrame]:
    result_rows, _export_keys = build_results_export_rows(analysis_mode, results)
    payload = {
        "signal_processing_inputs": pd.DataFrame([export_metadata]),
        "results": pd.DataFrame(result_rows),
    }

    if enable_titration_analysis and titration_ready:
        analysis_results = titration_results if titration_results is not None else results
        analysis_channels = titration_channels if titration_channels is not None else channels
        analysis_vlines = titration_vlines if titration_vlines is not None else active_vlines
        analysis_scan_range = titration_scan_range if titration_results is not None else scan_range
        titration_rows = collect_titration_rows(
            analysis_results,
            metric_cfg=metric_cfg,
            channels=analysis_channels,
            vlines=analysis_vlines,
            vlines_by_channel=titration_vlines_by_channel,
            scan_range=analysis_scan_range,
            edge_trim_fraction=titration_edge_trim_fraction,
            concentration_unit=titration_concentration_unit,
            baseline_mode=titration_baseline_mode,
            included_step_labels=titration_included_step_labels,
            remove_extreme_outliers=remove_extreme_titration_outliers,
        )
        if titration_rows:
            payload["titration_steps"] = pd.DataFrame(titration_rows)

        if fit_titration_langmuir:
            langmuir_rows = collect_langmuir_summary_rows(
                analysis_results,
                metric_cfg=metric_cfg,
                channels=analysis_channels,
                vlines=analysis_vlines,
                vlines_by_channel=titration_vlines_by_channel,
                scan_range=analysis_scan_range,
                edge_trim_fraction=titration_edge_trim_fraction,
                concentration_unit=titration_concentration_unit,
                baseline_mode=titration_baseline_mode,
                included_step_labels=titration_included_step_labels,
                remove_extreme_outliers=remove_extreme_titration_outliers,
            )
            if langmuir_rows:
                payload["langmuir_fit_summary"] = pd.DataFrame(langmuir_rows)
            accuracy_rows = collect_titration_measurement_accuracy_rows(
                analysis_results,
                metric_cfg=metric_cfg,
                channels=analysis_channels,
                vlines=analysis_vlines,
                vlines_by_channel=titration_vlines_by_channel,
                scan_range=analysis_scan_range,
                edge_trim_fraction=titration_edge_trim_fraction,
                concentration_unit=titration_concentration_unit,
                baseline_mode=titration_baseline_mode,
                included_step_labels=titration_included_step_labels,
                remove_extreme_outliers=remove_extreme_titration_outliers,
            )
            if accuracy_rows:
                payload["titration_measurement_accuracy"] = pd.DataFrame(
                    accuracy_rows
                )

    return payload


def write_experiment_output_bundle(
    export_root: Path,
    experiment_name: str,
    experiment_notes: str,
    analysis_mode: str,
    source_folders: List[str],
    export_payload: Dict[str, pd.DataFrame],
    export_metadata: dict,
) -> Path:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_name = f"{timestamp}_{_safe_folder_name(experiment_name)}"
    bundle_dir = export_root / "outputs" / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=False)

    files = {}
    for key, df in export_payload.items():
        file_name = export_file_name(analysis_mode, key)
        df.to_csv(bundle_dir / file_name, index=False)
        files[key] = file_name

    manifest = {
        "schema_version": "1.0",
        "created_at": created_at,
        "app": "swv_app",
        "analysis_mode": analysis_mode,
        "experiment_name": experiment_name,
        "experiment_notes": experiment_notes,
        "source_folders": source_folders,
        "files": files,
        "metadata": export_metadata,
    }
    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return bundle_dir


def _fig_to_image(fig: plt.Figure, dpi: int = 130) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def _grid_dimensions(n_items: int, max_cols: int = 3) -> Tuple[int, int]:
    if n_items <= 0:
        return 0, 0
    cols = min(max_cols, max(1, math.ceil(math.sqrt(n_items))))
    rows = int(math.ceil(n_items / cols))
    return rows, cols


def build_plot_grid_page(
    title: str,
    plot_factories: List[Tuple[str, Callable[[], Optional[plt.Figure]]]],
    max_cols: int = 3,
    image_dpi: int = 130,
) -> Optional[plt.Figure]:
    images = []
    for label, make_plot in plot_factories:
        fig = make_plot()
        if fig is None:
            continue
        images.append((label, _fig_to_image(fig, dpi=image_dpi)))

    if not images:
        return None

    rows, cols = _grid_dimensions(len(images), max_cols=max_cols)
    page_width = max(11.0, cols * 5.0)
    page_height = max(8.5, rows * 4.2 + 0.6)
    page, axes = plt.subplots(rows, cols, figsize=(page_width, page_height))
    axes_arr = np.asarray(axes).reshape(-1)

    for ax, (_label, image) in zip(axes_arr, images):
        ax.imshow(image)
        ax.axis("off")

    for ax in axes_arr[len(images):]:
        ax.axis("off")

    page.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    page.tight_layout(rect=(0, 0, 1, 0.975))
    return page


def build_export_pdf(
    analysis_mode: str,
    results: List[dict],
    ok_results_by_channel: Dict[Any, List[dict]],
    channels: List[Any],
    metric_cfg: Dict[str, Tuple[str, str]],
    drift_cfg: Dict[str, Tuple[str, str, str]],
    active_vlines: List[Tuple[float, str]],
    scan_range: Optional[Tuple[int, int]],
    xlabel: str,
    metrics_layout: str = "Combined",
    drift_layout: str = "Combined",
    highlight_metric_channel: Optional[Any] = None,
    highlight_drift_channel: Optional[Any] = None,
    remove_extreme_titration_outliers: bool = False,
    channel_colors: Optional[Dict[Any, Any]] = None,
) -> bytes:
    pdf_buf = io.BytesIO()

    if analysis_mode == "CV":
        raw_overlay_key = "raw_current"
        fitted_overlay_key = "smoothed_current"
        fitted_overlay_label = "Smoothed overlays"
    else:
        raw_overlay_key = "raw_current"
        fitted_overlay_key = "smoothed_corrected_current"
        fitted_overlay_label = "Smoothed corrected overlays"

    with PdfPages(pdf_buf) as pdf:
        raw_factories = []
        fitted_factories = []
        for ch in channels:
            ch_res = ok_results_by_channel.get(ch, [])
            if analysis_mode == "CV":
                raw_factories.append((
                    f"Ch{ch}",
                    lambda ch=ch, ch_res=ch_res: plot_cv_overlaid_cycles(
                        ch_res,
                        y_key=raw_overlay_key,
                        title=f"Ch{ch} raw",
                        show_peak_markers=True,
                        show_peak_reference_vlines=True,
                    ),
                ))
                fitted_factories.append((
                    f"Ch{ch}",
                    lambda ch=ch, ch_res=ch_res: plot_cv_overlaid_cycles(
                        ch_res,
                        y_key=fitted_overlay_key,
                        title=f"Ch{ch} smoothed",
                        show_peak_markers=True,
                        show_peak_reference_vlines=True,
                    ),
                ))
            else:
                raw_factories.append((
                    f"Ch{ch}",
                    lambda ch=ch, ch_res=ch_res: plot_overlaid_traces(
                        ch_res,
                        y_key=raw_overlay_key,
                        title=f"Ch{ch} raw",
                    ),
                ))
                fitted_factories.append((
                    f"Ch{ch}",
                    lambda ch=ch, ch_res=ch_res: plot_overlaid_traces(
                        ch_res,
                        y_key=fitted_overlay_key,
                        title=f"Ch{ch} smoothed corrected",
                        show_peak_markers=True,
                        show_zero_baseline=True,
                    ),
                ))

        for page_title, factories in (
            ("Raw overlays by channel", raw_factories),
            (fitted_overlay_label + " by channel", fitted_factories),
        ):
            page = build_plot_grid_page(page_title, factories, max_cols=3)
            if page:
                pdf.savefig(page, bbox_inches="tight")
                plt.close(page)

        metric_items = list(metric_cfg.items())

        def _metric_results(metric: str) -> List[dict]:
            if not remove_extreme_titration_outliers:
                return results
            return filter_extreme_titration_outliers(
                results,
                metric=metric,
                vlines=active_vlines,
                channels=channels,
            )

        if metrics_layout == "Individual channels":
            for label, (metric, ylabel) in metric_items:
                factories = [
                    (
                        f"Ch{ch}",
                        lambda metric=metric, ylabel=ylabel, label=label, ch=ch: plot_metric_vs_scan(
                            _metric_results(metric),
                            metric=metric,
                            channels=[ch],
                            title=f"Ch{ch} | {label}",
                            ylabel=ylabel,
                            vlines=active_vlines,
                            scan_range=scan_range,
                            figsize=(5, 3),
                            xlabel=xlabel,
                            channel_colors=channel_colors,
                        ),
                    )
                    for ch in channels
                ]
                page = build_plot_grid_page(f"Metrics | {label}", factories, max_cols=3)
                if page:
                    pdf.savefig(page, bbox_inches="tight")
                    plt.close(page)
        else:
            factories = [
                (
                    label,
                    lambda metric=metric, ylabel=ylabel, label=label: plot_metric_vs_scan(
                        _metric_results(metric),
                        metric=metric,
                        channels=channels,
                        title=label,
                        ylabel=ylabel,
                        vlines=active_vlines,
                        scan_range=scan_range,
                        highlight_channel=highlight_metric_channel,
                        xlabel=xlabel,
                        channel_colors=channel_colors,
                    ),
                )
                for label, (metric, ylabel) in metric_items
            ]
            page = build_plot_grid_page("Metrics", factories, max_cols=2)
            if page:
                pdf.savefig(page, bbox_inches="tight")
                plt.close(page)

        drift_items = list(drift_cfg.items())
        if drift_layout == "Individual channels":
            for label, (drift_key, ylabel, _caption) in drift_items:
                factories = [
                    (
                        f"Ch{ch}",
                        lambda drift_key=drift_key, ylabel=ylabel, label=label, ch=ch: plot_drift_vs_scan(
                            results,
                            drift_metric=drift_key,
                            channels=[ch],
                            title=f"Ch{ch} | {label}",
                            ylabel=ylabel,
                            vlines=active_vlines,
                            scan_range=scan_range,
                            figsize=(5, 3),
                            xlabel=xlabel,
                            channel_colors=channel_colors,
                        ),
                    )
                    for ch in channels
                ]
                page = build_plot_grid_page(f"Drift | {label}", factories, max_cols=3)
                if page:
                    pdf.savefig(page, bbox_inches="tight")
                    plt.close(page)
        else:
            factories = [
                (
                    label,
                    lambda drift_key=drift_key, ylabel=ylabel, label=label: plot_drift_vs_scan(
                        results,
                        drift_metric=drift_key,
                        channels=channels,
                        title=label,
                        ylabel=ylabel,
                        vlines=active_vlines,
                        scan_range=scan_range,
                        highlight_channel=highlight_drift_channel,
                        xlabel=xlabel,
                        channel_colors=channel_colors,
                    ),
                )
                for label, (drift_key, ylabel, _caption) in drift_items
            ]
            page = build_plot_grid_page("Drift", factories, max_cols=2)
            if page:
                pdf.savefig(page, bbox_inches="tight")
                plt.close(page)

    pdf_buf.seek(0)
    return pdf_buf.getvalue()


LANGMUIR_METRIC_KEYS = frozenset({"peak_current_selected", "wavelet_energy"})
DEFAULT_SWV_VLINES_TEXT = ""
DEFAULT_SWV_CROP_RANGE = (-0.5, -0.1)
DEFAULT_SWV_MIN_START_VOLTAGE = -0.6
SWV_VOLTAGE_DEFAULTS_VERSION = 2
DEFAULT_SWV_GROUP_COLORMAPS = (
    "plasma",
    "viridis",
    "inferno",
    "cividis",
    "magma",
    "turbo",
)
VLINE_ANNOTATION_HELP = (
    "One marker per line: scan,label. The scan is the x-axis position. "
    "For titration Kd, start the label with the concentration for the interval after that marker. "
    "Use buffer for zero ligand. Example: 40,10 uM."
)
VLINE_ANNOTATION_PLACEHOLDER = (
    "0, buffer\n"
    "20, buffer\n"
    "40, 10 uM\n"
    "60, 20 uM\n"
    "80, 40 uM\n"
    "100, 80 uM\n"
    "120, 160 uM\n"
    "140, 320 uM\n"
    "160, 640 uM\n"
    "180, 1.28 mM\n"
    "200, 2.56 mM\n"
    "220, end"
)


def supports_langmuir(metric_key: str) -> bool:
    return metric_key in LANGMUIR_METRIC_KEYS


def parse_colormap_names(text: str) -> Tuple[List[str], List[str]]:
    available = {name.lower(): name for name in plt.colormaps()}
    parsed = []
    invalid = []
    for token in (part.strip() for part in str(text or "").split(",")):
        if not token:
            continue
        canonical = available.get(token.lower())
        if canonical is None:
            invalid.append(token)
        else:
            parsed.append(canonical)
    return parsed, invalid


def build_drift_options(analysis_mode: str, compute_skew: bool = True) -> Dict[str, Tuple[str, str, str]]:
    if analysis_mode == "CV":
        return {
            "Reduction peak drift (V)": (
                "reduction_peak_voltage_drift",
                "Reduction Peak Drift (V)",
                "Shift in the reduction peak position relative to the first valid cycle.",
            ),
            "Oxidation peak drift (V)": (
                "oxidation_peak_voltage_drift",
                "Oxidation Peak Drift (V)",
                "Shift in the oxidation peak position relative to the first valid cycle.",
            ),
            "Peak separation drift (V)": (
                "peak_separation_drift",
                "Peak Separation Drift (V)",
                "Change in oxidation minus reduction peak spacing over time.",
            ),
            "Loop area drift": (
                "loop_area_abs_drift",
                "Loop Area Drift (uA*V)",
                "Change in the enclosed CV loop area relative to the first valid cycle.",
            ),
        }

    drift_options = {
        "Peak voltage drift (V)": (
            "peak_voltage_drift",
            "Peak voltage (V)",
            "Shift in peak position  indicates a change in the redox potential.",
        ),
        "Bracket width drift (V)": (
            "bracket_width_drift",
            "Bracket width (V)",
            "Change in the distance between the left and right correction anchors.",
        ),
        "Background drift (%)": (
            "background_drift_percent",
            "Background Drift (%)",
            "Percent change in outside-crop background RMS relative to the channel reference scans.",
        ),
        "Skew drift": (
            "skew_drift",
            "Skew",
            "Change in corrected-trace asymmetry  sensitive to baseline shape changes.",
        ),
        "Peak offset (normalized) drift": (
            "peak_offset_norm_drift",
            "Peak offset (normalized)",
            "Shift relative to the scan's own bracket center; pure whole-peak shifts can stay small.",
        ),
    }
    if not compute_skew:
        drift_options.pop("Skew drift", None)
    return drift_options


def annotate_swv_peak_height_metrics(
    results: List[dict],
    selected_peak_height_source: str,
    minima_search_window_V: float,
    use_prominent_minima: bool,
    compute_skew: bool,
    compute_wavelet_energy: bool,
    apply_background_recentering: bool = False,
    smooth_window: int = 9,
    smooth_polyorder: int = 2,
    use_double_correction: bool = False,
    compute_wavelet_denoised_trace: bool = False,
    use_wavelet_for_correction: bool = False,
) -> List[dict]:
    background_drift_reference_scans = 3

    def _compute_trace_based_metrics(row: dict, trace_key: str) -> dict:
        voltage = row.get("voltage")
        trace = row.get(trace_key)
        raw_current = row.get("raw_current")
        if voltage is None or trace is None:
            return {}

        try:
            v = np.asarray(voltage, dtype=float)
            y = np.asarray(trace, dtype=float)
            raw = np.asarray(raw_current, dtype=float) if raw_current is not None else None
            if len(v) < 5 or len(y) != len(v):
                return {}

            peak_idx_seed = int(detect_dominant_peak(y))
            corr = (
                rotate_offset_using_prominent_bracketing_minima(v, y, peak_idx_seed, minima_search_window_V)
                if use_prominent_minima
                else rotate_offset_using_bracketing_minima(v, y, peak_idx_seed, minima_search_window_V)
            )
            left_idx = int(corr["left_idx"])
            right_idx = int(corr["right_idx"])
            segment = y[left_idx:right_idx + 1]
            peak_idx = left_idx + int(detect_dominant_peak(segment, boundary_margin=0))

            v_left = float(v[left_idx])
            v_right = float(v[right_idx])
            denom = (v_right - v_left) / 2.0
            peak_offset_norm = np.nan
            if abs(denom) > 1e-12:
                peak_offset_norm = float((float(v[peak_idx]) - ((v_left + v_right) / 2.0)) / denom)

            wavelet_energy = np.nan
            if compute_wavelet_energy:
                coeffs = pywt.wavedec(y, "haar", level=3)
                wavelet_energy = float(sum(np.sum(c**2) for c in coeffs))

            return {
                "peak_idx_corr": peak_idx,
                "left_min_idx": left_idx,
                "right_min_idx": right_idx,
                "peak_voltage": float(v[peak_idx]),
                "peak_current": float(y[peak_idx]),
                "peak_current_raw": float(raw[peak_idx]) if raw is not None and 0 <= peak_idx < len(raw) else np.nan,
                "bracket_width_V": float(v_right - v_left),
                "peak_offset_norm": peak_offset_norm,
                "skew": float(skew(y)) if compute_skew else np.nan,
                "wavelet_energy": wavelet_energy,
            }
        except Exception:
            return {}

    for row in results:
        if row.get("status") != "OK":
            row["peak_current_corrected"] = row.get("peak_current", np.nan)
            row["peak_current_smoothed_corrected"] = np.nan
            row["peak_current_selected"] = row.get(selected_peak_height_source, row.get("peak_current", np.nan))
            row["background_drift_rms_reference"] = np.nan
            row["background_drift_fraction"] = np.nan
            row["background_drift_percent"] = np.nan
            row["peak_current_background_drift_corrected"] = np.nan
            row["background_current_median_reference"] = np.nan
            row["background_current_offset_uA"] = np.nan
            row["peak_current_background_recentered"] = np.nan
            continue

        corrected_metrics = _compute_trace_based_metrics(row, "corrected_current")
        smoothed_metrics = _compute_trace_based_metrics(row, "smoothed_corrected_current")

        row["peak_current_corrected"] = corrected_metrics.get("peak_current", np.nan)
        row["peak_current_smoothed_corrected"] = smoothed_metrics.get("peak_current", np.nan)

        selected_metrics = (
            smoothed_metrics
            if selected_peak_height_source == "peak_current_smoothed_corrected"
            else corrected_metrics
        )
        for key in (
            "peak_idx_corr",
            "left_min_idx",
            "right_min_idx",
            "peak_voltage",
            "peak_current",
            "peak_current_raw",
            "bracket_width_V",
            "peak_offset_norm",
            "skew",
            "wavelet_energy",
        ):
            if key in selected_metrics:
                row[key] = selected_metrics[key]

        row["peak_current_selected"] = selected_metrics.get("peak_current", row.get("peak_current", np.nan))

    ref_candidates_by_channel: Dict[int, List[float]] = {}
    sorted_results = sorted(results, key=lambda r: (r.get("channel", -1), r.get("scan_number", np.inf)))
    for row in sorted_results:
        if row.get("status") != "OK":
            continue
        ch = row.get("channel")
        bg = row.get("background_current_rms")
        if ch is None or bg is None or not np.isfinite(bg):
            continue
        ref_candidates_by_channel.setdefault(int(ch), [])
        if len(ref_candidates_by_channel[int(ch)]) < background_drift_reference_scans:
            ref_candidates_by_channel[int(ch)].append(float(bg))

    ref_by_channel = {
        ch: float(np.median(vals))
        for ch, vals in ref_candidates_by_channel.items()
        if vals and np.all(np.isfinite(vals))
    }
    median_ref_candidates_by_channel: Dict[int, List[float]] = {}
    for row in sorted_results:
        if row.get("status") != "OK":
            continue
        ch = row.get("channel")
        bg_median = row.get("background_current_median")
        if ch is None or bg_median is None or not np.isfinite(bg_median):
            continue
        median_ref_candidates_by_channel.setdefault(int(ch), [])
        if len(median_ref_candidates_by_channel[int(ch)]) < background_drift_reference_scans:
            median_ref_candidates_by_channel[int(ch)].append(float(bg_median))

    median_ref_by_channel = {
        ch: float(np.median(vals))
        for ch, vals in median_ref_candidates_by_channel.items()
        if vals and np.all(np.isfinite(vals))
    }

    for row in results:
        ch = row.get("channel")
        ref = ref_by_channel.get(int(ch)) if ch is not None else None
        median_ref = median_ref_by_channel.get(int(ch)) if ch is not None else None
        bg = row.get("background_current_rms")
        bg_median = row.get("background_current_median")
        peak_selected = row.get("peak_current_selected")
        row["background_drift_rms_reference"] = ref if ref is not None else np.nan
        row["background_current_median_reference"] = median_ref if median_ref is not None else np.nan
        row["background_current_offset_uA"] = np.nan
        row["peak_current_background_recentered"] = np.nan
        if (
            row.get("status") == "OK"
            and ref is not None
            and np.isfinite(ref)
            and abs(ref) > 1e-12
            and bg is not None
            and np.isfinite(bg)
        ):
            bg_norm = float(bg) / float(ref)
            drift_fraction = bg_norm - 1.0
            row["background_drift_fraction"] = drift_fraction
            row["background_drift_percent"] = 100.0 * drift_fraction
            if peak_selected is not None and np.isfinite(peak_selected) and abs(bg_norm) > 1e-12:
                row["peak_current_background_drift_corrected"] = float(peak_selected) / bg_norm
            else:
                row["peak_current_background_drift_corrected"] = np.nan
        else:
            row["background_drift_fraction"] = np.nan
            row["background_drift_percent"] = np.nan
            row["peak_current_background_drift_corrected"] = np.nan

        if (
            apply_background_recentering
            and row.get("status") == "OK"
            and median_ref is not None
            and np.isfinite(median_ref)
            and bg_median is not None
            and np.isfinite(bg_median)
        ):
            row["background_current_offset_uA"] = float(bg_median) - float(median_ref)
            voltage = row.get("voltage")
            raw_current = row.get("raw_current")
            if voltage is not None and raw_current is not None:
                try:
                    v = np.asarray(voltage, dtype=float)
                    raw = np.asarray(raw_current, dtype=float)
                    if len(v) >= 5 and len(raw) == len(v):
                        recentered = raw - float(row["background_current_offset_uA"])
                        recentered_metrics = analyze_swv_arrays(
                            v_raw=v,
                            i_raw=recentered,
                            crop_range=(float(np.min(v)) - 1e-9, float(np.max(v)) + 1e-9),
                            smooth_window=smooth_window,
                            smooth_polyorder=smooth_polyorder,
                            minima_search_window_V=minima_search_window_V,
                            use_prominent_minima=use_prominent_minima,
                            use_double_correction=use_double_correction,
                            min_peak_height_uA=None,
                            compute_skew=compute_skew,
                            compute_wavelet_energy=compute_wavelet_energy,
                            compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
                            use_wavelet_for_correction=use_wavelet_for_correction,
                            file_path=row.get("file_path"),
                        )
                        recentered_row = {
                            "voltage": recentered_metrics.get("voltage"),
                            "raw_current": recentered_metrics.get("raw_current"),
                            "corrected_current": recentered_metrics.get("corrected_current"),
                            "smoothed_corrected_current": recentered_metrics.get("smoothed_corrected_current"),
                        }
                        recentered_selected = (
                            _compute_trace_based_metrics(recentered_row, "smoothed_corrected_current")
                            if selected_peak_height_source == "peak_current_smoothed_corrected"
                            else _compute_trace_based_metrics(recentered_row, "corrected_current")
                        )
                        row["peak_current_background_recentered"] = recentered_selected.get("peak_current", np.nan)
                except Exception:
                    row["peak_current_background_recentered"] = np.nan

    compute_drift_fields(results)

    return results


def format_scan_window(scan_window: Tuple[int, int]) -> str:
    return f"{scan_window[0]}:{scan_window[1]}"


def format_scan_windows(scan_windows: List[Tuple[int, int]]) -> str:
    return ", ".join(format_scan_window(scan_window) for scan_window in scan_windows)


def parse_scan_windows(
    text: str,
    base_scan_range: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Tuple[int, int]], List[str]]:
    windows: List[Tuple[int, int]] = []
    errors: List[str] = []
    seen = set()

    normalized = text.replace("&", "\n").replace(",", "\n")
    for token in [part.strip() for part in normalized.splitlines() if part.strip()]:
        if ":" not in token:
            errors.append(f"Ignored '{token}': use start:end format.")
            continue

        start_text, end_text = [part.strip() for part in token.split(":", 1)]
        try:
            start = int(float(start_text))
            end = int(float(end_text))
        except ValueError:
            errors.append(f"Ignored '{token}': start and end must be numbers.")
            continue

        if end <= start:
            errors.append(f"Ignored '{token}': end must be greater than start.")
            continue

        if base_scan_range is not None:
            start = max(start, int(base_scan_range[0]))
            end = min(end, int(base_scan_range[1]))
            if end <= start:
                errors.append(
                    f"Ignored '{token}': it falls outside the active scan range "
                    f"{format_scan_window(base_scan_range)}."
                )
                continue

        window = (start, end)
        if window in seen:
            continue

        seen.add(window)
        windows.append(window)

    return windows, errors


def parse_vlines(text: str) -> Tuple[List[Tuple[float, str]], List[str]]:
    vlines: List[Tuple[float, str]] = []
    errors: List[str] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        token = raw_line.strip()
        if not token:
            continue

        parts = token.split(",", 1)
        if len(parts) != 2:
            errors.append(
                f"Ignored line {line_number}: use scan,label format, "
                "for example 45,10 nM, target added."
            )
            continue

        scan_text, label = parts[0].strip(), parts[1].strip()
        if not label:
            errors.append(f"Ignored line {line_number}: label cannot be blank.")
            continue

        try:
            scan_value = float(scan_text)
        except ValueError:
            errors.append(f"Ignored line {line_number}: scan index must be numeric.")
            continue

        vlines.append((scan_value, label))

    return vlines, errors


AUTOTITRATION_VLINE_DETECTION_VERSION = 2

_AUTOTITRATION_MEASUREMENT_RE = re.compile(
    r"Queue start ->\s*"
    r"(?P<label>[^|]+?)\s*\|"
    r".*?MUX ch\s*(?P<channel>\d+)"
    r".*?rep\s*(?P<replicate>\d+)\s*/\s*(?P<replicate_count>\d+)",
    re.IGNORECASE,
)
_AUTOTITRATION_CONCENTRATION_RE = re.compile(
    r"^\s*(?P<concentration>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?P<unit>pM|nM|[uµμ]M|mM|M)\b",
    re.IGNORECASE,
)
_AUTOTITRATION_TAG_RE = re.compile(
    r"\[Tag\].*?_(?P<scan>\d+)_ch(?P<channel>\d+)\s*$",
    re.IGNORECASE,
)


def _parse_autotitration_measurement_label(
    label: str,
) -> Optional[Tuple[str, str]]:
    """Return the concentration identity encoded by a queued SWV label."""
    if re.search(r"\bbuffer\b", label, flags=re.IGNORECASE):
        return "buffer", ""

    concentration_match = _AUTOTITRATION_CONCENTRATION_RE.search(label)
    if concentration_match is None:
        return None
    unit = (
        concentration_match.group("unit")
        .replace("µ", "u")
        .replace("μ", "u")
    )
    return concentration_match.group("concentration"), unit


def _autotitration_session_logs(folders: List[str]) -> List[Path]:
    logs: List[Path] = []
    seen: set[Path] = set()
    for raw_folder in folders:
        selected = Path(raw_folder).expanduser()
        start = selected.parent if selected.is_file() else selected
        for candidate_root in [start, *list(start.parents)[:4]]:
            candidate = candidate_root / "session_log.txt"
            if candidate.is_file():
                resolved = candidate.resolve()
                if resolved not in seen:
                    logs.append(resolved)
                    seen.add(resolved)
        if start.is_dir():
            try:
                descendants = start.rglob("session_log.txt")
                for candidate in descendants:
                    if not candidate.is_file():
                        continue
                    resolved = candidate.resolve()
                    if resolved not in seen:
                        logs.append(resolved)
                        seen.add(resolved)
            except OSError:
                continue
    return logs


def detect_autotitration_vlines(
    folders: List[str],
    results: List[dict],
) -> Tuple[List[Tuple[float, str]], List[Path]]:
    """Map logged autotitration concentrations onto the active result scan axis."""
    logs = _autotitration_session_logs(folders)
    logged_measurements: Dict[Tuple[int, int], Tuple[str, str]] = {}
    used_logs: List[Path] = []
    for log_path in logs:
        pending_measurement: Optional[Tuple[str, str, int]] = None
        matched_count = 0
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            # A tag belongs only to the latest queue item. In particular, do
            # not let an unrecognized queue item inherit a preceding target
            # concentration.
            if "queue start ->" in line.lower():
                pending_measurement = None
            measurement_match = _AUTOTITRATION_MEASUREMENT_RE.search(line)
            if measurement_match:
                measurement_identity = _parse_autotitration_measurement_label(
                    measurement_match.group("label")
                )
                if measurement_identity is None:
                    continue
                concentration, unit = measurement_identity
                pending_measurement = (
                    concentration,
                    unit,
                    int(measurement_match.group("channel")),
                )
                continue
            tag_match = _AUTOTITRATION_TAG_RE.search(line)
            if tag_match and pending_measurement is not None:
                concentration, unit, expected_channel = pending_measurement
                tag_channel = int(tag_match.group("channel"))
                if tag_channel == expected_channel:
                    logged_measurements[
                        (int(tag_match.group("scan")), tag_channel)
                    ] = (concentration, unit)
                    matched_count += 1
                pending_measurement = None
        if matched_count:
            used_logs.append(log_path)

    matched_rows = []
    for row in results:
        try:
            key = (int(row.get("scan_id_from_name")), int(row.get("channel")))
            plotted_scan = float(row.get("scan_number"))
        except (TypeError, ValueError):
            continue
        concentration = logged_measurements.get(key)
        if concentration is None or not math.isfinite(plotted_scan):
            continue
        matched_rows.append({
            "global_scan": key[0],
            "channel": key[1],
            "plotted_scan": plotted_scan,
            "concentration": concentration[0],
            "unit": concentration[1],
        })
    if not matched_rows:
        return [], used_logs

    matched_rows.sort(key=lambda row: (row["global_scan"], row["channel"]))
    segments: List[dict] = []
    for row in matched_rows:
        identity = (row["concentration"], row["unit"])
        if not segments or segments[-1]["identity"] != identity:
            segments.append({"identity": identity, "rows": []})
        segments[-1]["rows"].append(row)

    detected_vlines: List[Tuple[float, str]] = []
    for segment in segments:
        first_by_channel: Dict[int, float] = {}
        for row in segment["rows"]:
            channel = int(row["channel"])
            first_by_channel[channel] = min(
                first_by_channel.get(channel, float("inf")),
                float(row["plotted_scan"]),
            )
        if not first_by_channel:
            continue
        boundary = float(np.median(list(first_by_channel.values())))
        if np.isclose(boundary, round(boundary)):
            boundary = float(round(boundary))
        concentration, unit = segment["identity"]
        if detected_vlines and boundary <= detected_vlines[-1][0]:
            continue
        detected_vlines.append((boundary, f"{concentration} {unit}".strip()))

    if detected_vlines and segments:
        last_by_channel: Dict[int, float] = {}
        for row in segments[-1]["rows"]:
            channel = int(row["channel"])
            last_by_channel[channel] = max(
                last_by_channel.get(channel, float("-inf")),
                float(row["plotted_scan"]),
            )
        if last_by_channel:
            closing_boundary = float(
                np.median([value + 1.0 for value in last_by_channel.values()])
            )
            if np.isclose(closing_boundary, round(closing_boundary)):
                closing_boundary = float(round(closing_boundary))
            if closing_boundary > detected_vlines[-1][0]:
                detected_vlines.append((closing_boundary, "end"))
    return detected_vlines, used_logs


def _vlines_input_text(vlines: List[Tuple[float, str]]) -> str:
    return "\n".join(
        f"{int(scan) if float(scan).is_integer() else scan:g},{label}"
        for scan, label in vlines
    )


def titration_step_selection_options(
    vlines: List[Tuple[float, str]],
) -> List[str]:
    bases = [
        "buffer"
        if str(label).strip().lower().startswith("buffer")
        else str(label).strip()
        for _position, label in vlines[:-1]
    ]
    totals = {base: bases.count(base) for base in set(bases)}
    occurrences: Dict[str, int] = {}
    options = []
    for base in bases:
        occurrences[base] = occurrences.get(base, 0) + 1
        if base == "buffer" or totals[base] > 1:
            options.append(f"{base}_{occurrences[base]}")
        else:
            options.append(base)
    return options


def scan_in_windows(scan_number: float, scan_windows: List[Tuple[int, int]]) -> bool:
    return any(start <= scan_number < end for start, end in scan_windows)


def vline_in_windows(vline_position: float, scan_windows: List[Tuple[int, int]]) -> bool:
    return any(start <= vline_position <= end for start, end in scan_windows)


def remap_scan_number(
    scan_number: float,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
) -> float:
    if scan_windows:
        offset = 0
        for start, end in scan_windows:
            if start <= scan_number < end:
                return float(offset + (scan_number - start))
            offset += end - start
        raise ValueError(f"Scan {scan_number} is outside selected analysis windows.")
    if scan_range is not None:
        return float(scan_number - scan_range[0])
    return float(scan_number)


def remap_vline_position(
    vline_position: float,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
) -> float:
    if scan_windows:
        offset = 0
        for start, end in scan_windows:
            if start <= vline_position <= end:
                return float(offset + (vline_position - start))
            offset += end - start
        raise ValueError(f"Vline {vline_position} is outside selected analysis windows.")
    if scan_range is not None:
        return float(vline_position - scan_range[0])
    return float(vline_position)


def remap_vlines_to_active_scan_range(
    vlines,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
):
    if not vlines:
        return []

    if scan_windows:
        remapped = []
        for x, label in vlines:
            if vline_in_windows(float(x), scan_windows):
                remapped.append(
                    (remap_vline_position(float(x), scan_windows=scan_windows), label)
                )
        return remapped

    if scan_range is not None:
        return [
            (remap_vline_position(float(x), scan_range=scan_range), label)
            for x, label in vlines
            if scan_range[0] <= x <= scan_range[1]
        ]

    return list(vlines)


def _method_group_sort_key(label: str) -> Tuple[int, float, str]:
    if label.endswith(" Hz"):
        number_text = label[:-3].strip()
        try:
            return (0, float(number_text), label)
        except ValueError:
            pass
    return (1, float("inf"), label)


def remap_vlines_to_filtered_scan_axis(
    vlines: List[Tuple[float, str]],
    kept_scan_numbers: List[float],
) -> List[Tuple[float, str]]:
    if not vlines or not kept_scan_numbers:
        return []

    sorted_scans = sorted(float(x) for x in kept_scan_numbers)
    exact_map = {scan_number: idx + 1 for idx, scan_number in enumerate(sorted_scans)}
    remapped: List[Tuple[float, str]] = []

    for x, label in vlines:
        scan_x = float(x)
        insert_at = bisect.bisect_left(sorted_scans, scan_x)

        if insert_at < len(sorted_scans) and sorted_scans[insert_at] == scan_x:
            remapped.append((float(exact_map[scan_x]), label))
            continue

        if 0 < insert_at < len(sorted_scans):
            remapped.append((float(insert_at) + 0.5, label))

    return remapped


def filter_vlines_to_results_axis(
    vlines: List[Tuple[float, str]],
    results: List[dict],
) -> List[Tuple[float, str]]:
    if not vlines or not results:
        return []

    scan_numbers = [
        float(row["scan_number"])
        for row in results
        if row.get("scan_number") is not None
    ]
    if not scan_numbers:
        return []

    min_scan = min(scan_numbers)
    max_scan = max(scan_numbers)
    in_range = [
        (float(x), str(label))
        for x, label in vlines
        if min_scan <= float(x) <= max_scan
    ]
    left_candidates = [
        (float(x), str(label))
        for x, label in vlines
        if float(x) < min_scan
    ]
    # Autotitration deliberately closes its final interval one scan beyond the
    # last observed measurement. Preserve that synthetic `end` boundary so the
    # last concentration has a right edge and can produce a plateau.
    right_end_candidates = [
        (float(x), str(label))
        for x, label in vlines
        if float(x) > max_scan and str(label).strip().lower() == "end"
    ]
    if right_end_candidates:
        in_range.append(min(right_end_candidates, key=lambda item: item[0]))
    if left_candidates:
        nearest_left = max(left_candidates, key=lambda item: item[0])
        return [nearest_left] + in_range
    return in_range


def _channel_display_sort_key(channel: Any) -> tuple:
    text = str(channel)
    match = re.fullmatch(
        r"(\d+)(?:\s+group\s+(\d+))?(?:\s*\|.*)?",
        text,
        re.IGNORECASE,
    )
    if match:
        channel_number = int(match.group(1))
        group_number = int(match.group(2) or 0)
        return (0, channel_number, group_number, text)
    try:
        return (0, int(channel), 0, text)
    except (TypeError, ValueError):
        return (1, text, 0, text)


def _high_contrast_response_shades(count: int) -> np.ndarray:
    """Alternate light and dark shades for neighboring channel/method options."""
    if count <= 0:
        return np.asarray([], dtype=float)
    if count == 1:
        return np.asarray([0.72], dtype=float)
    light_count = (count + 1) // 2
    dark_count = count // 2
    light = np.linspace(0.42, 0.55, light_count)
    dark = np.linspace(0.80, 0.95, dark_count)
    return np.asarray([
        light[index // 2] if index % 2 == 0 else dark[index // 2]
        for index in range(count)
    ])


def _swv_method_blue(channel: Any) -> Optional[Any]:
    """Use one stable blue shade for each displayed SWV method number."""
    match = re.match(
        r"(?:Ch)?\d+\s+group\s+(\d+)(?:\s*\||\s*$)",
        str(channel).strip(),
        re.IGNORECASE,
    )
    if not match:
        return None
    method_shades = {1: 0.90, 2: 0.40}
    shade = method_shades.get(int(match.group(1)))
    return plt.get_cmap("Blues")(shade) if shade is not None else None


def _channel_option_value(option: str) -> Any:
    text = option.removeprefix("Ch")
    try:
        return int(text)
    except ValueError:
        return text


def _compact_titration_channel_label(channel: Any) -> str:
    text = str(channel).split("|", 1)[0].strip()
    match = re.fullmatch(r"(?:Ch)?(\d+)\s+group\s+(\d+)", text, re.IGNORECASE)
    if match:
        return f"Channel {match.group(1)} · method {match.group(2)}"
    channel_match = re.fullmatch(r"(?:Ch(?:annel)?)?\s*(\d+)", text, re.IGNORECASE)
    return f"Channel {channel_match.group(1)}" if channel_match else text


def _titration_diagnostic_row_groups(
    rows: List[dict],
    layout: str,
) -> List[Tuple[str, str, List[dict]]]:
    """Split diagnostic rows for all-group, original-channel, or method-group plots."""
    if not rows:
        return []
    if layout == "All groups":
        return [("all", "All groups", rows)]

    grouped: Dict[Any, List[dict]] = {}
    for row in rows:
        if layout == "Per SWV group":
            group_key = row.get("channel")
        else:
            group_key = row.get("original_channel")
            if group_key is None:
                channel_text = str(row.get("channel", ""))
                match = re.match(r"(?:Ch)?(\d+)", channel_text, re.IGNORECASE)
                group_key = int(match.group(1)) if match else row.get("channel")
        grouped.setdefault(group_key, []).append(row)

    output = []
    for index, group_key in enumerate(
        sorted(grouped, key=_channel_display_sort_key),
        start=1,
    ):
        label = _compact_titration_channel_label(group_key)
        output.append((f"group_{index}", label, grouped[group_key]))
    return output


def _swv_modulo_channel_label(channel: Any, group_index: int) -> str:
    return f"{channel} group {group_index}"


SWV_SETTING_FIELDS = (
    "swv_frequency_hz",
    "swv_sweep_start_V",
    "swv_sweep_end_V",
    "swv_step_size_V",
    "swv_amplitude_V",
)


def _swv_setting_value(row: dict, key: str) -> Optional[float]:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def swv_settings_signature(row: dict) -> Tuple[Optional[float], ...]:
    return tuple(_swv_setting_value(row, key) for key in SWV_SETTING_FIELDS)


def format_swv_settings_label(row: dict) -> str:
    frequency, sweep_start, sweep_end, step_size, amplitude = swv_settings_signature(row)
    parts = []
    if frequency is not None:
        parts.append(f"{frequency:g} Hz")
    if sweep_start is not None and sweep_end is not None:
        parts.append(f"sweep {sweep_start:g}→{sweep_end:g} V")
    elif sweep_start is not None:
        parts.append(f"start {sweep_start:g} V")
    elif sweep_end is not None:
        parts.append(f"end {sweep_end:g} V")
    if step_size is not None:
        parts.append(f"step {step_size:g} V")
    if amplitude is not None:
        parts.append(f"amplitude {amplitude:g} V")
    return "; ".join(parts) if parts else "SWV settings unavailable"


def format_swv_overlay_title(channel: Any, rows: List[dict]) -> str:
    """Return a concise method/channel title for an SWV overlay plot."""
    original_channel = next(
        (
            row.get("original_channel")
            for row in rows
            if row.get("original_channel") is not None
        ),
        channel,
    )
    channel_match = re.search(r"\d+", str(original_channel))
    channel_label = (
        f"Channel {channel_match.group(0)}"
        if channel_match
        else f"Channel {original_channel}"
    )

    method_names = []
    for row in rows:
        try:
            group_index = int(row.get("display_group_index"))
        except (TypeError, ValueError):
            continue
        method_name = {
            1: "Optimized Method",
            2: "Manual Method",
        }.get(group_index)
        if method_name and method_name not in method_names:
            method_names.append(method_name)

    if not method_names:
        display_match = re.search(
            r"\bgroup\s+(\d+)\b",
            str(channel),
            re.IGNORECASE,
        )
        if display_match:
            method_name = {
                1: "Optimized Method",
                2: "Manual Method",
            }.get(int(display_match.group(1)))
            if method_name:
                method_names.append(method_name)

    method_label = " and ".join(method_names)
    return f"{method_label} | {channel_label}" if method_label else channel_label


def apply_swv_settings_split_for_display(results: List[dict]) -> List[dict]:
    """Split each channel by its complete SWV method settings."""
    def _scan_sort_key(row: dict) -> float:
        scan_number = row.get("scan_number")
        return float(scan_number) if scan_number is not None else float("inf")

    rows_by_channel: Dict[Any, List[dict]] = {}
    for row in results:
        channel = row.get("channel")
        if channel is not None:
            rows_by_channel.setdefault(channel, []).append(row)

    replacements: Dict[int, dict] = {}
    for channel, channel_rows in rows_by_channel.items():
        group_indexes: Dict[Tuple[Optional[float], ...], int] = {}
        group_counts: Dict[Tuple[Optional[float], ...], int] = {}
        for row in sorted(channel_rows, key=_scan_sort_key):
            signature = swv_settings_signature(row)
            if signature not in group_indexes:
                group_indexes[signature] = len(group_indexes) + 1
                group_counts[signature] = 0
            group_counts[signature] += 1
            settings_label = format_swv_settings_label(row)
            updated = dict(row)
            updated["original_channel"] = channel
            updated["swv_settings_group"] = group_indexes[signature]
            updated["swv_settings_signature"] = signature
            updated["swv_settings_label"] = settings_label
            updated["display_group_index"] = group_indexes[signature]
            updated["display_group_trace_index"] = group_counts[signature]
            updated["display_group_source_scan_number"] = row.get("scan_number")
            updated["scan_number"] = group_counts[signature]
            updated["channel"] = (
                f"{channel} group {group_indexes[signature]} | {settings_label}"
            )
            replacements[id(row)] = updated

    return [replacements.get(id(row), dict(row)) for row in results]


def apply_swv_modulo_split_for_display(
    results: List[dict],
    modulo_count: int,
) -> List[dict]:
    if modulo_count <= 1:
        return list(results)

    def _scan_sort_key(row: dict) -> float:
        scan_number = row.get("scan_number")
        return float(scan_number) if scan_number is not None else float("inf")

    rows_by_channel: Dict[Any, List[dict]] = {}
    for row in results:
        channel = row.get("channel")
        if channel is None:
            continue
        rows_by_channel.setdefault(channel, []).append(row)

    replacements: Dict[int, dict] = {}
    for channel, channel_rows in rows_by_channel.items():
        ordered_rows = sorted(channel_rows, key=_scan_sort_key)
        settings_by_group: Dict[int, List[str]] = {}
        for index, row in enumerate(ordered_rows):
            group_index = (index % modulo_count) + 1
            settings_label = format_swv_settings_label(row)
            group_settings = settings_by_group.setdefault(group_index, [])
            if settings_label not in group_settings:
                group_settings.append(settings_label)

        for index, row in enumerate(ordered_rows):
            group_index = (index % modulo_count) + 1
            updated = dict(row)
            group_trace_index = (index // modulo_count) + 1
            group_settings = settings_by_group[group_index]
            group_settings_label = (
                group_settings[0]
                if len(group_settings) == 1
                else f"mixed settings ({len(group_settings)} methods)"
            )
            updated["original_channel"] = channel
            updated["modulo_group"] = group_index
            updated["modulo_group_trace_index"] = group_trace_index
            updated["modulo_source_scan_number"] = row.get("scan_number")
            updated["swv_settings_label"] = format_swv_settings_label(row)
            updated["modulo_group_settings_label"] = group_settings_label
            updated["display_group_index"] = group_index
            updated["display_group_trace_index"] = group_trace_index
            updated["display_group_source_scan_number"] = row.get("scan_number")
            updated["scan_number"] = group_trace_index
            updated["channel"] = (
                f"{_swv_modulo_channel_label(channel, group_index)} | "
                f"{group_settings_label}"
            )
            replacements[id(row)] = updated

    return [replacements.get(id(row), dict(row)) for row in results]


def remap_vlines_to_swv_display_group(
    vlines: List[Tuple[float, str]],
    group_rows: List[dict],
) -> List[Tuple[float, str]]:
    """Map source-axis annotation intervals onto one group-local iteration axis."""
    if not vlines or not group_rows:
        return []
    ordered_vlines = sorted(
        [(float(x), str(label)) for x, label in vlines],
        key=lambda item: item[0],
    )
    source_and_local = sorted(
        [
            (
                float(row["display_group_source_scan_number"]),
                float(row["scan_number"]),
            )
            for row in group_rows
            if row.get("display_group_source_scan_number") is not None
            and row.get("scan_number") is not None
        ],
        key=lambda item: item[0],
    )
    if not source_and_local:
        return []

    remapped: List[Tuple[float, str]] = []
    final_local_value: Optional[float] = None
    for index, (interval_start, label) in enumerate(ordered_vlines[:-1]):
        interval_end = ordered_vlines[index + 1][0]
        local_values = [
            local
            for source, local in source_and_local
            if interval_start <= source < interval_end
        ]
        if not local_values:
            continue
        local_start = min(local_values)
        if not remapped or not np.isclose(remapped[-1][0], local_start):
            remapped.append((local_start, label))
        final_local_value = max(local_values)

    if final_local_value is not None and remapped:
        closing_value = final_local_value + 1.0
        if closing_value > remapped[-1][0]:
            remapped.append((closing_value, ordered_vlines[-1][1]))
    return remapped


def merge_group_vlines(
    grouped_vlines: List[List[Tuple[float, str]]],
) -> List[Tuple[float, str]]:
    merged: List[Tuple[float, str]] = []
    for vlines in grouped_vlines:
        for x, label in vlines:
            if any(
                np.isclose(existing_x, x) and existing_label == label
                for existing_x, existing_label in merged
            ):
                continue
            merged.append((float(x), str(label)))
    return sorted(merged, key=lambda item: (item[0], item[1]))


def group_swv_display_channels(
    results: List[dict],
    channels: List[Any],
) -> Dict[Any, List[Any]]:
    """Map each original SWV channel to its ordered display groups."""
    selected_channels = set(channels)
    grouped: Dict[Any, List[Any]] = {}
    group_order: Dict[Any, int] = {}
    for row in results:
        display_channel = row.get("channel")
        original_channel = row.get("original_channel")
        if (
            display_channel not in selected_channels
            or original_channel is None
            or row.get("display_group_index") is None
        ):
            continue
        display_channels = grouped.setdefault(original_channel, [])
        if display_channel not in display_channels:
            display_channels.append(display_channel)
            group_order[display_channel] = int(row["display_group_index"])

    for display_channels in grouped.values():
        display_channels.sort(key=lambda channel: group_order[channel])
    return dict(sorted(grouped.items(), key=lambda item: _channel_display_sort_key(item[0])))


def build_channel_indexes(
    results: List[dict],
    scan_range: Optional[Tuple[int, int]] = None,
) -> dict:
    def _scan_sort_key(row: dict) -> float:
        scan_number = row.get("scan_number")
        return float(scan_number) if scan_number is not None else float("inf")

    all_by_channel = {}
    ok_by_channel = {}
    failed_by_channel = {}

    for row in results:
        channel = row.get("channel")
        if channel is None:
            continue
        all_by_channel.setdefault(channel, []).append(row)
        if row.get("status") == "OK":
            ok_by_channel.setdefault(channel, []).append(row)
        elif row.get("status") == "FAILED":
            failed_by_channel.setdefault(channel, []).append(row)

    for mapping in (all_by_channel, ok_by_channel, failed_by_channel):
        for channel, rows in mapping.items():
            mapping[channel] = sorted(rows, key=_scan_sort_key)

    if scan_range is None:
        ok_in_range_by_channel = ok_by_channel
        failed_in_range_by_channel = failed_by_channel
    else:
        start, end = scan_range
        ok_in_range_by_channel = {
            channel: [
                row for row in rows
                if row.get("scan_number") is not None and start <= row["scan_number"] <= end
            ]
            for channel, rows in ok_by_channel.items()
        }
        failed_in_range_by_channel = {
            channel: [
                row for row in rows
                if row.get("scan_number") is not None and start <= row["scan_number"] <= end
            ]
            for channel, rows in failed_by_channel.items()
        }

    return {
        "all_by_channel": all_by_channel,
        "ok_by_channel": ok_by_channel,
        "failed_by_channel": failed_by_channel,
        "ok_in_range_by_channel": ok_in_range_by_channel,
        "failed_in_range_by_channel": failed_in_range_by_channel,
    }


def reindex_swv_results_for_display(
    results: List[dict],
    vlines: List[Tuple[float, str]],
) -> Tuple[List[dict], List[Tuple[float, str]], Optional[Tuple[int, int]]]:
    if not results:
        return [], [], None

    rows_by_channel = {}
    for row in results:
        channel = row.get("channel")
        scan_number = row.get("scan_number")
        if channel is None or scan_number is None:
            continue
        rows_by_channel.setdefault(channel, []).append(row)

    if not rows_by_channel:
        return list(results), [], None

    reindexed_rows = {}
    for channel, channel_rows in rows_by_channel.items():
        ordered_rows = sorted(channel_rows, key=lambda r: float(r["scan_number"]))
        for idx, row in enumerate(ordered_rows, start=1):
            updated = dict(row)
            updated["filtered_source_scan_number"] = row.get("scan_number")
            updated["scan_number"] = idx
            reindexed_rows[id(row)] = updated

    reindexed_results: List[dict] = [
        reindexed_rows.get(id(row), dict(row))
        for row in results
    ]

    compute_drift_fields(reindexed_results)
    reference_channel = max(
        rows_by_channel,
        key=lambda ch: (len(rows_by_channel[ch]), -int(ch)),
    )
    reference_scan_numbers = sorted(
        float(row["filtered_source_scan_number"])
        for row in reindexed_rows.values()
        if row.get("channel") == reference_channel and row.get("filtered_source_scan_number") is not None
    )
    remapped_vlines = remap_vlines_to_filtered_scan_axis(vlines, reference_scan_numbers)
    return reindexed_results, remapped_vlines, (1, len(reference_scan_numbers))


# 
# Session state
# 
for k, v in dict(
    results=None,
    last_results=None,
    results_mode=None,
    last_results_mode=None,
    results_folder_key=None,
    analysis_cache_key=None,
    analysis_cache_results=None,
    folders=[],
    run_count=0,
    swv_post_method_filter_enabled=False,
    swv_post_selected_method_groups=[],
    swv_post_vlines_input=DEFAULT_SWV_VLINES_TEXT,
    swv_enable_titration_analysis=False,
    swv_titration_edge_trim_fraction=0.15,
    swv_remove_extreme_titration_outliers=False,
    swv_show_titration_uloq=False,
    swv_show_titration_lod=False,
    swv_show_titration_fit_details=False,
    swv_fit_titration_langmuir=True,
    swv_titration_concentration_unit="uM",
    mat_conversion_report=None,
).items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.get("_swv_clean_langmuir_plot_defaults_v1", False):
    st.session_state["swv_show_titration_uloq"] = False
    st.session_state["swv_show_titration_lod"] = False
    st.session_state["swv_show_titration_fit_details"] = False
    st.session_state["_swv_clean_langmuir_plot_defaults_v1"] = True


# 
# Sidebar
# 
with st.sidebar:
    analysis_mode = st.radio(
        "Analysis mode",
        ["SWV", "CV", "BO Session"],
        horizontal=True,
        help="Analyze SWV/CV data or inspect a saved Bayesian-optimization session.",
    )

if analysis_mode == "BO Session":
    render_bo_session_app()
    st.stop()

with st.sidebar:
    st.title("⚡ SWV Analysis" if analysis_mode == "SWV" else "⚡ CV Analysis")
    st.divider()

    #  Folders 
    st.subheader(" Data Folders")

    c1, c2 = st.columns(2)

    if c1.button("  Browse (Windows)", use_container_width=True, disabled=not sys.platform.startswith("win")):
        try:
            picked = _pick_folder_windows()
            if picked:
                st.session_state.folders = _append_unique_folder(st.session_state.folders, picked)
        except subprocess.CalledProcessError as e:
            st.error(f"Windows folder picker failed: {e}")
        except Exception as e:
            st.error(f"Windows folder picker failed: {e}")

    if c2.button("  Browse (macOS)", use_container_width=True, disabled=sys.platform != "darwin"):
        try:
            # Use Finder's native picker via AppleScript (Tk dialogs can crash Streamlit on macOS).
            script = 'POSIX path of (choose folder with prompt "Select electrochemistry data folder")'
            picked = subprocess.check_output(["osascript", "-e", script], text=True).strip()
            if picked:
                st.session_state.folders = _append_unique_folder(st.session_state.folders, picked)
        except FileNotFoundError:
            st.error("macOS folder picker failed: `osascript` not found.")
        except subprocess.CalledProcessError:
            # User cancel returns a non-zero exit code.
            st.info("Folder selection canceled.")
        except Exception as e:
            st.error(f"macOS folder picker failed: {e}")

    if sys.platform == "darwin":
        st.caption("macOS picker only works when Streamlit runs locally (not over SSH/remote server).")

    raw_folders = st.text_area(
        "Folders (one per line  or browse above)",
        value="\n".join(st.session_state.folders),
        height=90,
        help="You can analyze multiple folders together. Each browse click adds one folder, and you can also paste multiple paths here one per line.",
    )
    edited = [f.strip() for f in raw_folders.splitlines() if f.strip()]
    st.session_state.folders = edited
    folders = edited

    current_selection_key = _analysis_selection_key(analysis_mode, folders)
    loaded_selection_key = st.session_state.get("results_folder_key")
    if st.session_state.get("results") is not None and loaded_selection_key != current_selection_key:
        _clear_loaded_analysis_state()

    if folders:
        if st.button("  Clear all folders", use_container_width=True):
            st.session_state.folders = []
            _clear_loaded_analysis_state()
            st.rerun()

    folder_errors = [f for f in folders if not os.path.isdir(f)]
    if folder_errors:
        for fe in folder_errors:
            st.error(f"Not found: `{fe}`")

    if analysis_mode == "SWV" and folders and not folder_errors:
        st.caption("MAT-file discovery runs only when conversion is requested.")
        if st.button("Find and convert MAT files for SWV", use_container_width=True):
            with st.spinner("Finding and converting MATLAB files to native SWV CSVs..."):
                report = convert_mat_folders_to_swv_csv(folders)
            st.session_state.mat_conversion_report = report
            updated_folders = list(st.session_state.folders)
            for output_folder in report["output_folders"]:
                updated_folders = _append_unique_folder(updated_folders, output_folder)
            st.session_state.folders = updated_folders
            st.rerun()

    conversion_report = st.session_state.get("mat_conversion_report")
    if analysis_mode == "SWV" and conversion_report:
        converted_count = len(conversion_report.get("converted", []))
        failed_count = len(conversion_report.get("failed", []))
        if converted_count:
            st.success(f"Converted {converted_count} MAT file(s) to SWV CSV.")
        if failed_count:
            st.warning(f"{failed_count} MAT file(s) could not be converted.")
            with st.expander("MAT conversion errors"):
                st.dataframe(
                    pd.DataFrame(conversion_report["failed"]),
                    use_container_width=True,
                    hide_index=True,
                )
        if not converted_count and not failed_count:
            st.info("No MATLAB files were found in the selected folders.")

    st.divider()

    #  Crop & voltage 
    st.subheader(" Voltage / Crop")
    col1, col2 = st.columns(2)
    if analysis_mode == "SWV":
        crop_state_key = ("SWV", tuple(folders), DEFAULT_SWV_CROP_RANGE)
        if st.session_state.get("swv_crop_folder_key") != crop_state_key:
            st.session_state.swv_crop_min = DEFAULT_SWV_CROP_RANGE[0]
            st.session_state.swv_crop_max = DEFAULT_SWV_CROP_RANGE[1]
            st.session_state.swv_crop_folder_key = crop_state_key
        if (
            st.session_state.get("_swv_voltage_defaults_version")
            != SWV_VOLTAGE_DEFAULTS_VERSION
        ):
            st.session_state.swv_min_start_voltage = DEFAULT_SWV_MIN_START_VOLTAGE
            st.session_state["_swv_voltage_defaults_version"] = (
                SWV_VOLTAGE_DEFAULTS_VERSION
            )
        crop_min = col1.number_input(
            "Crop min (V)",
            value=float(st.session_state.get("swv_crop_min", DEFAULT_SWV_CROP_RANGE[0])),
            step=0.01,
            format="%.3f",
            key="swv_crop_min",
        )
        crop_max = col2.number_input(
            "Crop max (V)",
            value=float(st.session_state.get("swv_crop_max", DEFAULT_SWV_CROP_RANGE[1])),
            step=0.01,
            format="%.3f",
            key="swv_crop_max",
        )
        min_start_voltage = st.number_input(
            "Min start voltage (V)",
            value=float(st.session_state.get(
                "swv_min_start_voltage",
                DEFAULT_SWV_MIN_START_VOLTAGE,
            )),
            step=0.01,
            format="%.3f",
            help="Skip files whose first voltage point is below this value.",
            key="swv_min_start_voltage",
        )
    else:
        crop_state_key = ("CV", tuple(folders))
        if st.session_state.get("cv_crop_folder_key") != crop_state_key:
            bounds = _measurement_voltage_bounds(tuple(folders), mode="cv")
            if bounds is not None:
                st.session_state.cv_crop_min = bounds[0]
                st.session_state.cv_crop_max = bounds[1]
            st.session_state.cv_crop_folder_key = crop_state_key
        crop_min = col1.number_input(
            "Crop min (V)",
            value=-0.20,
            step=0.01,
            format="%.3f",
            key="cv_crop_min",
        )
        crop_max = col2.number_input(
            "Crop max (V)",
            value=0.90,
            step=0.01,
            format="%.3f",
            key="cv_crop_max",
        )
        min_start_voltage = None
        st.caption("CV cropping is applied to both the forward and reverse sweep before peak detection.")

    st.divider()

    #  Smoothing 
    st.subheader(" Smoothing")
    if analysis_mode == "SWV":
        smooth_window = st.slider(
            "Savitzky-Golay window",
            min_value=3,
            max_value=100,
            value=15,
            step=2,
            key="swv_smooth_window",
        )
        smooth_polyorder = st.slider("Polynomial order", min_value=1, max_value=5, value=2, key="swv_smooth_polyorder")
    else:
        smooth_window = st.slider("Savitzky-Golay window", min_value=3, max_value=31, value=11, step=2, key="cv_smooth_window")
        smooth_polyorder = st.slider("Polynomial order", min_value=1, max_value=5, value=2, key="cv_smooth_polyorder")

    st.divider()

    minima_search_window = 0.30
    use_prominent_minima = False
    use_double_correction = False
    use_wavelet_for_correction = False
    apply_background_recentering = False
    min_peak_height = None
    edge_trim_fraction = 0.05
    min_peak_prominence = None
    if analysis_mode == "SWV":
        #  Peak / baseline 
        st.subheader(" Peak / Baseline")
        minima_search_window = st.number_input(
            "Minima search window (V)", value=0.30, step=0.01, format="%.3f",
            help="Voltage window either side of peak when searching for bracketing minima.",
        )
        use_double_correction = st.checkbox(
            "Double baseline correction",
            value=True,
            help=(
                "Optional refinement: after the first baseline rotation, run one more "
                "bracketing-minima correction on the once-corrected trace so the anchors "
                "can better match the shifted minima."
            ),
        )
        if use_double_correction:
            st.caption(
                "Adds a second correction pass to refine anchors after the first rotation. "
                "Single-trace inspectors will show an extra second-pass panel."
            )
        use_peak_cutoff = st.checkbox("Enforce min peak height", value=True)
        if use_peak_cutoff:
            min_peak_height = st.number_input("Min peak height (uA)", value=0.001, step=0.001, format="%.3f")
    else:
        st.subheader(" CV Peak Detection")
        edge_trim_fraction = st.slider(
            "Ignore sweep edges",
            min_value=0.0,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Skips this fraction of points at the start and end of each sweep when looking for oxidation/reduction peaks.",
        )
        enforce_cv_prominence = st.checkbox(
            "Enforce min peak prominence",
            value=False,
            help="Uses the detrended sweep to reject weak or ambiguous CV peaks.",
        )
        if enforce_cv_prominence:
            min_peak_prominence = st.number_input(
                "Min peak prominence (uA)",
                value=0.010,
                step=0.005,
                format="%.3f",
            )
        st.caption(
            "CV uses light processing only: sweep-wise smoothing, linear detrending, edge trimming, and one oxidation plus one reduction peak."
        )

    st.divider()

    st.subheader("Performance")
    swv_parallel_workers = 1
    if analysis_mode == "SWV":
        available_worker_count = max(1, int(os.cpu_count() or 1))
        swv_parallel_workers = int(st.number_input(
            "Parallel analysis workers",
            min_value=1,
            max_value=available_worker_count,
            value=min(4, available_worker_count),
            step=1,
            help=(
                "Analyzes independent SWV files concurrently. Four workers is a "
                "memory-conscious default; reduce this if the system becomes unresponsive."
            ),
        ))
        compute_skew = st.checkbox("Compute skew metric", value=True)
        compute_wavelet_energy = st.checkbox("Compute wavelet energy", value=True)
        with st.expander("Experimental", expanded=False):
            use_prominent_minima = st.checkbox(
                "Use prominent local minima for bracketing",
                value=False,
                help="Experimental comparison mode: uses peaks of the inverted smoothed signal and takes the most prominent local minimum on each side of the detected peak.",
            )
            compute_wavelet_denoised_trace = st.checkbox(
                "Compute wavelet-denoised trace",
                value=False,
                help="Adds an optional denoised trace for visual comparison in overlays and trace inspectors. It does not change the current peak metrics.",
            )
            use_wavelet_for_correction = st.checkbox(
                "Use wavelet-denoised trace for baseline correction",
                value=False,
                disabled=not compute_wavelet_denoised_trace,
                help="Optional experiment: use the wavelet-denoised trace instead of the Savitzky-Golay smoothed trace for the first-pass baseline correction and anchor search.",
            )
            apply_background_recentering = st.checkbox(
                "Apply additive background recentering",
                value=False,
                help=(
                    "Experimental on-demand mode: estimate the outside-crop background offset from the raw trace, "
                    "recenter the cropped raw SWV trace to the channel reference background, then recompute the peak metrics."
                ),
            )
            if apply_background_recentering:
                st.caption(
                    "This leaves the default analysis untouched until enabled. When on, the app does one extra in-memory "
                    "recompute pass per valid SWV scan to estimate a background-recentered peak."
                )
    else:
        compute_skew = False
        compute_wavelet_energy = False
        compute_wavelet_denoised_trace = False
        use_wavelet_for_correction = False
        apply_background_recentering = False
        st.caption("CV mode skips the heavier SWV-only skew and wavelet metrics.")
    use_cache = st.checkbox("Use cached results", value=True, help="Disable to force a full re-run with progress.")

    st.divider()

    #  Channels 
    st.subheader(" Channels")
    channels_input = st.text_input(
        "Channels to plot (comma-separated, blank = all)",
        value="1,2,3,4,5,6,7,8,9,10",
    )
    channels_to_plot: Optional[List[int]] = None
    if channels_input.strip():
        try:
            channels_to_plot = [int(c.strip()) for c in channels_input.split(",") if c.strip()]
        except ValueError:
            st.error("Invalid channel list  use integers separated by commas.")
    swv_grouping_mode = "None"
    use_swv_settings_grouping = False
    use_swv_modulo_split = False
    swv_modulo_split_count = 2
    swv_group_overlay_colormaps = list(DEFAULT_SWV_GROUP_COLORMAPS)
    swv_plot_show_legend = True
    swv_plot_show_grid = False
    swv_colorbar_height_fraction = 0.85
    swv_colorbar_side = "right"
    if analysis_mode == "SWV":
        group_c1, group_c2 = st.columns([2, 1])
        swv_grouping_mode = group_c1.selectbox(
            "Group plotted traces by",
            ["None", "SWV settings", "Modulo sequence"],
            key="swv_grouping_mode",
            help=(
                "SWV settings uses the method-file frequency, sweep bounds, step size, "
                "and amplitude. Modulo sequence is available as a metadata-free fallback."
            ),
        )
        use_swv_settings_grouping = swv_grouping_mode == "SWV settings"
        use_swv_modulo_split = swv_grouping_mode == "Modulo sequence"
        swv_modulo_split_count = int(group_c2.number_input(
            "Modulo groups",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            disabled=not use_swv_modulo_split,
            key="swv_modulo_split_count",
            help=(
                "For m groups, traces are assigned per channel as group 1, "
                "group 2, ... group m, then repeated."
            ),
        ))
        if use_swv_settings_grouping:
            st.caption(
                "Traces with identical frequency, sweep start/end, step size, and "
                "amplitude are grouped together per channel."
            )
        elif use_swv_modulo_split:
            st.caption(
                f"Modulo fallback is active: each selected channel is plotted as "
                f"{swv_modulo_split_count} chronological groups. Scan vlines are "
                "interpreted on the group-local scan axis."
            )
        group_colormap_text = st.text_input(
            "Group overlay colormaps",
            value=",".join(DEFAULT_SWV_GROUP_COLORMAPS),
            key="swv_group_overlay_colormaps",
            disabled=not (use_swv_settings_grouping or use_swv_modulo_split),
            help=(
                "Comma-separated Matplotlib colormap names. Group 1 uses the first, "
                "group 2 the second, and so on; the list cycles if needed."
            ),
        )
        parsed_group_colormaps, invalid_group_colormaps = parse_colormap_names(
            group_colormap_text
        )
        if invalid_group_colormaps:
            st.warning(
                "Unknown colormap(s): "
                + ", ".join(invalid_group_colormaps)
                + ". Valid entries will still be used."
            )
        if parsed_group_colormaps:
            swv_group_overlay_colormaps = parsed_group_colormaps
        else:
            swv_group_overlay_colormaps = list(DEFAULT_SWV_GROUP_COLORMAPS)

        with st.expander("SWV Plot Formatting & Exploration", expanded=False):
            st.slider(
                "Plot width",
                min_value=600,
                max_value=2200,
                value=1200,
                step=20,
                format="%d px",
                key="swv_plot_width_px",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("width_px",),
                help="Exact width used for browser plots and downloaded PNGs.",
            )
            st.slider(
                "Plot height",
                min_value=300,
                max_value=1200,
                value=600,
                step=20,
                format="%d px",
                key="swv_plot_height_px",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("height_px",),
                help="Exact height used for browser plots and downloaded PNGs.",
            )
            st.slider(
                "Plot text size",
                min_value=6.0,
                max_value=36.0,
                value=10.0,
                step=0.5,
                format="%.1f pt",
                key="swv_plot_text_size_points",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("text_size_points",),
            )
            st.slider(
                "Line thickness",
                min_value=0.25,
                max_value=5.0,
                value=1.0,
                step=0.25,
                format="%.2fx",
                key="swv_plot_line_width_scale",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("line_width_scale",),
            )
            line_color_override = st.text_input(
                "Line color override",
                key="swv_plot_line_color_override",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("line_color_override",),
                help=(
                    "Optional Matplotlib/CSS color. Leave blank to preserve the "
                    "selected colormaps."
                ),
            )
            if line_color_override.strip() and not is_color_like(line_color_override):
                st.caption("Unrecognized line color; the selected colormaps are being used.")
            st.slider(
                "Outer plot margin",
                min_value=0,
                max_value=200,
                value=40,
                step=5,
                format="%d px",
                key="swv_plot_margin_px",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("margin_px",),
            )
            st.slider(
                "Plot perimeter thickness",
                min_value=0.0,
                max_value=5.0,
                value=0.8,
                step=0.2,
                key="swv_plot_perimeter_width",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("perimeter_width",),
            )
            swv_perimeter_color = st.text_input(
                "Plot perimeter color",
                value="#222222",
                key="swv_plot_perimeter_color",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("perimeter_color",),
                help="Use a Matplotlib/CSS color name or hex value.",
            )
            if (
                swv_perimeter_color.strip()
                and not is_color_like(swv_perimeter_color)
            ):
                st.caption(
                    "Unrecognized perimeter color; #222222 is being used."
                )
            swv_plot_show_legend = st.checkbox(
                "Show plot legends",
                value=True,
                key="swv_plot_show_legend",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("show_legend",),
            )
            swv_plot_show_grid = st.checkbox(
                "Show background grid",
                value=False,
                key="swv_plot_show_grid",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("show_grid",),
            )
            marker_columns = st.columns(2)
            marker_columns[0].slider(
                "Marker size",
                min_value=2.0,
                max_value=20.0,
                value=6.0,
                step=1.0,
                key="swv_plot_marker_size",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("marker_size",),
            )
            marker_columns[1].slider(
                "Marker opacity",
                min_value=0.05,
                max_value=1.0,
                value=0.85,
                step=0.05,
                key="swv_plot_marker_opacity",
                on_change=_sync_shared_swv_style_to_metrics,
                args=("marker_opacity",),
            )

            st.markdown("**Displayed axis limits**")
            st.checkbox(
                "Set voltage limits manually",
                key="swv_plot_manual_x_limits",
            )
            x_limit_columns = st.columns(2)
            x_limit_columns[0].number_input(
                "Voltage minimum",
                value=float(crop_min),
                step=0.01,
                format="%.4f",
                key="swv_plot_x_min",
                disabled=not st.session_state.get("swv_plot_manual_x_limits", False),
            )
            x_limit_columns[1].number_input(
                "Voltage maximum",
                value=float(crop_max),
                step=0.01,
                format="%.4f",
                key="swv_plot_x_max",
                disabled=not st.session_state.get("swv_plot_manual_x_limits", False),
            )
            st.checkbox(
                "Set current limits manually",
                key="swv_plot_manual_y_limits",
            )
            y_limit_columns = st.columns(2)
            y_limit_columns[0].number_input(
                "Current minimum",
                value=-0.1,
                step=0.1,
                format="%.4f",
                key="swv_plot_y_min",
                disabled=not st.session_state.get("swv_plot_manual_y_limits", False),
            )
            y_limit_columns[1].number_input(
                "Current maximum",
                value=1.5,
                step=0.1,
                format="%.4f",
                key="swv_plot_y_max",
                disabled=not st.session_state.get("swv_plot_manual_y_limits", False),
            )

            st.markdown("**Global text overrides**")
            st.text_input(
                "Title override",
                key="swv_plot_title_override",
                help="Leave blank to use each plot's generated title.",
            )
            st.text_input(
                "X-axis label override",
                key="swv_plot_x_label_override",
            )
            st.text_input(
                "Y-axis label override",
                key="swv_plot_y_label_override",
            )
            st.text_input(
                "Colorbar label override",
                key="swv_plot_colorbar_label_override",
            )

            st.markdown("**Tick label overrides**")
            st.caption(
                "Enter comma-separated original positions and displayed labels. "
                "Leave positions blank to relabel existing ticks in order."
            )
            for axis_key, axis_label in (
                ("x", "X axis"),
                ("y", "Y axis"),
                ("colorbar", "Colorbar"),
            ):
                st.caption(axis_label)
                tick_columns = st.columns(2)
                tick_columns[0].text_input(
                    "Positions",
                    key=f"swv_plot_{axis_key}_tick_positions",
                    label_visibility="collapsed",
                    placeholder="0, 1, 2",
                )
                tick_columns[1].text_input(
                    "Labels",
                    key=f"swv_plot_{axis_key}_tick_labels",
                    label_visibility="collapsed",
                    placeholder="A, B, C",
                )

            colorbar_columns = st.columns(2)
            colorbar_height_percent = colorbar_columns[0].slider(
                "Colorbar height",
                min_value=20,
                max_value=100,
                value=85,
                step=5,
                format="%d%%",
                key="swv_colorbar_height_percent",
            )
            swv_colorbar_height_fraction = colorbar_height_percent / 100.0
            swv_colorbar_side = colorbar_columns[1].radio(
                "Colorbar side",
                ["right", "left"],
                horizontal=True,
                key="swv_colorbar_side",
            )

    st.divider()

    #  Analysis subsection
    scan_range: Optional[Tuple[int, int]] = None
    scan_windows: List[Tuple[int, int]] = []
    time_range: Optional[Tuple[datetime, datetime]] = None
    time_selection_invalid = False
    use_scan_range = False
    if analysis_mode == "SWV":
        st.subheader("Analysis Subsection")
        use_scan_range = st.checkbox(
            "Analyze subsection(s) of data",
            value=False,
            help=(
                "Limit analysis using zero-based chronological file positions or timestamps "
                "extracted from SWV filenames. Only matching files are opened and analyzed."
            ),
        )
        if use_scan_range:
            subsection_basis = st.radio(
                "Select subsection by",
                ["File position", "Time from filename"],
                horizontal=True,
                key="swv_subsection_basis",
            )
            if subsection_basis == "File position":
                scan_windows_input = st.text_area(
                    "Scan window(s)",
                    value="0:260",
                    height=80,
                    help=(
                        "Use zero-based chronological file positions in start:end format with end excluded. "
                        "Use commas to concatenate multiple chunks, or separate with & or new lines. "
                        "Example: 0:20, 20:40, 60:80, 80:100"
                    ),
                )
                scan_windows, scan_window_errors = parse_scan_windows(scan_windows_input)
                for err in scan_window_errors:
                    st.warning(err)
                if scan_windows:
                    st.caption(f"Active analysis windows: {format_scan_windows(scan_windows)}")
                    st.caption(
                        "Windows use chronological source-file order within each channel, then "
                        "concatenate into one continuous analysis axis without opening unselected CSVs."
                    )
                    if len(scan_windows) == 1:
                        scan_range = scan_windows[0]
                elif scan_windows_input.strip():
                    st.error("No valid scan windows were parsed.")
            else:
                filename_times = []
                if folders and not folder_errors:
                    filename_time_catalog_signature = tuple(
                        (folder, os.stat(folder).st_mtime_ns)
                        for folder in folders
                    )
                    if (
                        st.session_state.get("_swv_filename_time_catalog_signature")
                        != filename_time_catalog_signature
                    ):
                        st.session_state["_swv_filename_time_catalog"] = sorted(
                            measurement.measurement_time
                            for measurement in collect_measurement_csvs_from_folders(
                                folders,
                                mode="swv",
                            )
                            if measurement.measurement_time is not None
                        )
                        st.session_state["_swv_filename_time_catalog_signature"] = (
                            filename_time_catalog_signature
                        )
                    filename_times = st.session_state.get(
                        "_swv_filename_time_catalog",
                        [],
                    )
                if not filename_times:
                    time_selection_invalid = True
                    st.error(
                        "No YYYYMMDD_HHMM or YYYYMMDD_HHMMSS timestamps were found in the SWV filenames."
                    )
                else:
                    filename_hours = sorted({
                        value.replace(minute=0, second=0, microsecond=0)
                        for value in filename_times
                    })
                    filename_time_signature = (
                        2,
                        tuple(folders),
                        filename_hours[0],
                        filename_hours[-1],
                    )
                    if (
                        st.session_state.get("_swv_filename_time_bounds_signature")
                        != filename_time_signature
                    ):
                        st.session_state["swv_filename_time_start"] = filename_hours[0]
                        st.session_state["swv_filename_time_end"] = filename_hours[-1]
                        st.session_state["_swv_filename_time_bounds_signature"] = (
                            filename_time_signature
                        )
                    time_c1, time_c2 = st.columns(2)
                    time_start_hour = time_c1.selectbox(
                        "Start filename hour",
                        filename_hours,
                        format_func=lambda value: value.strftime("%Y-%m-%d %H:00"),
                        key="swv_filename_time_start",
                        help="Includes the entire selected starting hour.",
                    )
                    time_end_hour = time_c2.selectbox(
                        "End filename hour",
                        filename_hours,
                        format_func=lambda value: value.strftime("%Y-%m-%d %H:00"),
                        key="swv_filename_time_end",
                        help="Includes the entire selected ending hour.",
                    )
                    if time_start_hour <= time_end_hour:
                        time_end_inclusive = (
                            time_end_hour + timedelta(hours=1) - timedelta(microseconds=1)
                        )
                        time_range = (time_start_hour, time_end_inclusive)
                        selected_filename_count = sum(
                            time_start_hour <= value <= time_end_inclusive
                            for value in filename_times
                        )
                        st.caption(
                            f"{selected_filename_count:,} source file(s) fall within the selected full-hour range."
                        )
                    else:
                        time_selection_invalid = True
                        st.error(
                            "Start filename hour must be no later than the end hour."
                        )
    else:
        st.subheader(" Cycle View")
        st.caption("CV metrics are tracked across detected cycle number within each EC block, so scan windows and vlines are not used here.")

    #  Failed traces 
    max_failed = 40
    if analysis_mode == "SWV":
        st.subheader(" Failed Traces")
        max_failed = st.number_input("Max failed traces to plot", value=40, min_value=1)

    st.divider()
    scan_selection_invalid = use_scan_range and not scan_windows and time_range is None
    scan_selection_invalid = scan_selection_invalid or time_selection_invalid

    run_clicked = st.button(
        "  Run Analysis",
        type="primary",
        disabled=not folders or bool(folder_errors) or scan_selection_invalid,
        use_container_width=True,
    )


# 
# Run analysis
# 
if run_clicked and folders and not folder_errors:
    st.session_state.folders = folders
    try:
        if use_cache:
            input_signature = _analysis_input_signature(tuple(folders))
            requested_cache_key = (
                ANALYSIS_CACHE_SCHEMA_VERSION,
                analysis_mode,
                tuple(folders),
                (float(crop_min), float(crop_max)),
                int(smooth_window),
                int(smooth_polyorder),
                float(minima_search_window),
                bool(use_prominent_minima),
                bool(use_double_correction),
                min_peak_height,
                float(min_start_voltage),
                tuple(scan_windows),
                None if scan_windows else scan_range,
                time_range,
                bool(compute_skew),
                bool(compute_wavelet_energy),
                bool(compute_wavelet_denoised_trace),
                bool(use_wavelet_for_correction),
                float(edge_trim_fraction),
                min_peak_prominence,
                input_signature,
            )
            if (
                st.session_state.get("analysis_cache_key") == requested_cache_key
                and st.session_state.get("analysis_cache_results") is not None
            ):
                results = st.session_state.analysis_cache_results
                st.caption("Reused cached analysis results.")
            else:
                progress_bar = None
                progress_text = None

                if analysis_mode == "SWV":
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    cached_progress_state = {"pct": -1}

                    def _cached_swv_progress(done, total, name):
                        pct = int((done / max(total, 1)) * 100)
                        if done != 1 and done != total and pct <= cached_progress_state["pct"]:
                            return
                        cached_progress_state["pct"] = pct
                        progress_bar.progress(pct)
                        phase = "Analyzing selected traces"
                        display_name = name.removeprefix("Analyzing ")
                        progress_text.caption(
                            f"{phase} {done}/{total}: {display_name}"
                        )

                with st.spinner("Running analysis (first run may take a moment, cached runs are instant)"):
                    results = run_batch_dispatch(
                        analysis_mode=analysis_mode,
                        folders=tuple(folders),
                        crop_range=(crop_min, crop_max),
                        smooth_window=smooth_window,
                        smooth_polyorder=smooth_polyorder,
                        minima_search_window_V=minima_search_window,
                        use_prominent_minima=use_prominent_minima,
                        use_double_correction=use_double_correction,
                        min_peak_height_uA=min_peak_height,
                        min_start_voltage=min_start_voltage,
                        scan_windows=tuple(scan_windows),
                        scan_range=None if scan_windows else scan_range,
                        time_range=time_range,
                        compute_skew=compute_skew,
                        compute_wavelet_energy=compute_wavelet_energy,
                        compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
                        use_wavelet_for_correction=use_wavelet_for_correction,
                        _parallel_workers=swv_parallel_workers,
                        edge_trim_fraction=edge_trim_fraction,
                        min_peak_prominence_uA=min_peak_prominence,
                        input_signature=input_signature,
                        _progress_callback=(
                            _cached_swv_progress if analysis_mode == "SWV" else None
                        ),
                    )
                st.session_state.analysis_cache_key = requested_cache_key
                st.session_state.analysis_cache_results = results
                if progress_bar is not None and progress_text is not None:
                    progress_bar.progress(100)
                    progress_text.caption("Analysis complete.")
        else:
            progress_bar = st.progress(0)
            progress_text = st.empty()
            progress_state = {"pct": -1}

            def _progress(done, total, name):
                pct = int((done / max(total, 1)) * 100)
                if done != 1 and done != total and pct <= progress_state["pct"]:
                    return
                progress_state["pct"] = pct
                progress_bar.progress(pct)
                if name.startswith("Checking "):
                    phase = "Checking source files"
                    display_name = name.removeprefix("Checking ")
                else:
                    phase = "Analyzing selected traces"
                    display_name = name.removeprefix("Analyzing ")
                progress_text.caption(
                    f"{phase} {done}/{total}: {display_name}"
                )

            if analysis_mode == "CV":
                results = run_cv_batch(
                    folders=list(folders),
                    crop_range=(crop_min, crop_max),
                    smooth_window=smooth_window,
                    smooth_polyorder=smooth_polyorder,
                    edge_trim_fraction=edge_trim_fraction,
                    min_peak_prominence_uA=min_peak_prominence,
                    scan_windows=tuple(scan_windows),
                    scan_range=None if scan_windows else scan_range,
                    progress_callback=_progress,
                )
            else:
                results = run_batch(
                    folders=list(folders),
                    crop_range=(crop_min, crop_max),
                    smooth_window=smooth_window,
                    smooth_polyorder=smooth_polyorder,
                    minima_search_window_V=minima_search_window,
                    use_prominent_minima=use_prominent_minima,
                    use_double_correction=use_double_correction,
                    min_peak_height_uA=min_peak_height,
                    min_start_voltage=min_start_voltage,
                    scan_windows=tuple(scan_windows),
                    scan_range=None if scan_windows else scan_range,
                    time_range=time_range,
                    compute_skew=compute_skew,
                    compute_wavelet_energy=compute_wavelet_energy,
                    compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
                    use_wavelet_for_correction=use_wavelet_for_correction,
                    parallel_workers=swv_parallel_workers,
                    progress_callback=_progress,
                )
            progress_bar.progress(100)
            progress_text.caption("Analysis complete.")

        st.session_state.results = results
        if results:
            st.session_state.last_results = results
            st.session_state.last_results_mode = analysis_mode
        st.session_state.results_mode = analysis_mode
        st.session_state.results_folder_key = _analysis_selection_key(analysis_mode, folders)
        st.session_state.swv_annotation_signature = None
        st.session_state.swv_annotated_results = None
        st.session_state.run_count += 1
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()


# 
# Guard  nothing run yet
# 
results = st.session_state.get("results")
results_mode = st.session_state.get("results_mode")
if results is None:
    st.info(" Configure parameters in the sidebar, then click **Run Analysis**.")
    st.stop()
elif results_mode != analysis_mode:
    st.info(f"Current results are for {results_mode or 'another mode'}. Run {analysis_mode} analysis to populate this view.")
    st.stop()
if len(results) == 0:
    st.warning("No results returned for the current selection. Check folder paths and file naming pattern.")
    st.stop()

# Streamlit can retain analysis rows across a code reload. Backfill the parsed
# filename time so rows created by an earlier app version also support time plots.
display_subsection_bounds: Optional[Tuple[int, int]] = None
display_subsection_time_bounds: Optional[Tuple[datetime, datetime]] = None
if analysis_mode == "SWV":
    for row in results:
        if row.get("measurement_time") is None:
            row["measurement_time"] = parse_measurement_time_from_filename(
                str(row.get("file_name") or row.get("file_path") or "")
            )

    analyzed_scan_numbers = sorted({
        int(row["scan_number"])
        for row in results
        if row.get("scan_number") is not None
    })
    analyzed_filename_hours = sorted({
        row["measurement_time"].replace(minute=0, second=0, microsecond=0)
        for row in results
        if row.get("measurement_time") is not None
    })
    if analyzed_scan_numbers:
        analyzed_scan_min = analyzed_scan_numbers[0]
        analyzed_scan_max = analyzed_scan_numbers[-1]
        display_subsection_signature = (
            current_selection_key,
            st.session_state.get("run_count", 0),
            analyzed_scan_min,
            analyzed_scan_max,
            analyzed_filename_hours[0] if analyzed_filename_hours else None,
            analyzed_filename_hours[-1] if analyzed_filename_hours else None,
        )
        if (
            st.session_state.get("_swv_display_subsection_signature")
            != display_subsection_signature
        ):
            st.session_state["swv_display_subsection_enabled"] = False
            st.session_state["swv_display_subsection_basis"] = "Scan number"
            st.session_state["swv_display_scan_start"] = analyzed_scan_min
            st.session_state["swv_display_scan_end"] = analyzed_scan_max
            if analyzed_filename_hours:
                st.session_state["swv_display_time_start"] = analyzed_filename_hours[0]
                st.session_state["swv_display_time_end"] = analyzed_filename_hours[-1]
            st.session_state["_swv_display_subsection_signature"] = (
                display_subsection_signature
            )

        with st.expander("Display subsection (no reanalysis)", expanded=False):
            display_subsection_enabled = st.checkbox(
                "Show only a smaller subsection of the analyzed results",
                key="swv_display_subsection_enabled",
            )
            display_basis_options = ["Scan number"]
            if analyzed_filename_hours:
                display_basis_options.append("Time from filename")
            if (
                st.session_state.get("swv_display_subsection_basis")
                not in display_basis_options
            ):
                st.session_state["swv_display_subsection_basis"] = "Scan number"
            display_subsection_basis = st.radio(
                "Select and sort display subsection by",
                display_basis_options,
                horizontal=True,
                key="swv_display_subsection_basis",
                disabled=not display_subsection_enabled,
            )
            display_c1, display_c2 = st.columns(2)
            if display_subsection_basis == "Time from filename":
                display_time_start = display_c1.selectbox(
                    "Display start hour",
                    analyzed_filename_hours,
                    format_func=lambda value: value.strftime("%Y-%m-%d %H:00"),
                    key="swv_display_time_start",
                    disabled=not display_subsection_enabled,
                )
                display_time_end = display_c2.selectbox(
                    "Display end hour",
                    analyzed_filename_hours,
                    format_func=lambda value: value.strftime("%Y-%m-%d %H:00"),
                    key="swv_display_time_end",
                    disabled=not display_subsection_enabled,
                )
            else:
                display_scan_start = int(display_c1.number_input(
                    "Analyzed scan start",
                    min_value=analyzed_scan_min,
                    max_value=analyzed_scan_max,
                    step=1,
                    key="swv_display_scan_start",
                    disabled=not display_subsection_enabled,
                ))
                display_scan_end = int(display_c2.number_input(
                    "Analyzed scan end",
                    min_value=analyzed_scan_min,
                    max_value=analyzed_scan_max,
                    step=1,
                    key="swv_display_scan_end",
                    disabled=not display_subsection_enabled,
                ))

        if display_subsection_enabled:
            full_result_count = len(results)
            if display_subsection_basis == "Time from filename":
                if display_time_start > display_time_end:
                    st.warning("Display start hour must be no later than the end hour.")
                else:
                    display_time_end_inclusive = (
                        display_time_end + timedelta(hours=1) - timedelta(microseconds=1)
                    )
                    display_subsection_time_bounds = (
                        display_time_start,
                        display_time_end_inclusive,
                    )
                    displayed_result_count = sum(
                        1 for row in results
                        if row.get("measurement_time") is not None
                        and display_time_start
                        <= row["measurement_time"]
                        <= display_time_end_inclusive
                    )
                    st.caption(
                        f"Displaying {displayed_result_count:,} of {full_result_count:,} analyzed rows "
                        f"from {display_time_start:%Y-%m-%d %H:00} through "
                        f"{display_time_end:%Y-%m-%d %H:59}. Analysis was not rerun."
                    )
            elif display_scan_start > display_scan_end:
                st.warning("Display scan start must be less than or equal to the end.")
            else:
                displayed_result_count = sum(
                    1 for row in results
                    if row.get("scan_number") is not None
                    and display_scan_start <= int(row["scan_number"]) <= display_scan_end
                )
                display_subsection_bounds = (
                    display_scan_start,
                    display_scan_end,
                )
                st.caption(
                    f"Displaying {displayed_result_count:,} of {full_result_count:,} analyzed rows "
                    f"for scans {display_scan_start}–{display_scan_end}. Analysis was not rerun."
                )

if analysis_mode == "CV":
    ec_labels = [label for label in ["EC3", "EC4"] if any(r.get("ec_label") == label for r in results)]
    other_labels = sorted({r.get("ec_label") for r in results if r.get("ec_label") not in set(ec_labels) and r.get("ec_label")})
    cv_label_options = ec_labels + other_labels
    selected_cv_label = st.radio(
        "CV block",
        cv_label_options,
        horizontal=True,
        index=0 if cv_label_options else None,
        help="Metrics are shown across cycle number inside the selected EC block.",
    ) if cv_label_options else None
    if selected_cv_label:
        results = [r for r in results if r.get("ec_label") == selected_cv_label]
    st.caption(
        "CV files are segmented into repeated cycles using the method metadata and turning points in the voltage trace."
    )

selected_peak_height_source = "peak_current"
selected_peak_height_source_label = "Corrected"
selected_peak_height_metric_label = "Peak current (corrected)"
selected_peak_height_ylabel = "Change in Peak Height (uA)"
if analysis_mode == "SWV":
    peak_height_source_options = {
        "Corrected": (
            "peak_current",
            "Peak current (corrected)",
            "Change in Peak Height (uA)",
        ),
        "Corrected + smoothed": (
            "peak_current_smoothed_corrected",
            "Peak current (corrected + smoothed)",
            "Change in Peak Height (uA)",
        ),
    }
    peak_source_label = st.radio(
        "SWV peak height source",
        options=list(peak_height_source_options.keys()),
        index=1,
        horizontal=True,
        key="swv_peak_height_source_label",
        help=(
            "Use one consistent SWV trace basis for the derived SWV metrics. "
            "The selected trace is used for peak finding, anchor placement, and downstream metric values."
        ),
    )
    (
        selected_peak_height_source,
        selected_peak_height_metric_label,
        selected_peak_height_ylabel,
    ) = peak_height_source_options[peak_source_label]
    selected_peak_height_source_label = peak_source_label
    annotation_signature = (
        st.session_state.get("run_count", 0),
        selected_peak_height_source,
        float(minima_search_window),
        bool(use_prominent_minima),
        bool(compute_skew),
        bool(compute_wavelet_energy),
        bool(apply_background_recentering),
        int(smooth_window),
        int(smooth_polyorder),
        bool(use_double_correction),
        bool(compute_wavelet_denoised_trace),
        bool(use_wavelet_for_correction),
    )
    if st.session_state.get("swv_annotation_signature") != annotation_signature:
        results = annotate_swv_peak_height_metrics(
            results,
            selected_peak_height_source,
            minima_search_window_V=minima_search_window,
            use_prominent_minima=use_prominent_minima,
            compute_skew=compute_skew,
            compute_wavelet_energy=compute_wavelet_energy,
            apply_background_recentering=apply_background_recentering,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            use_double_correction=use_double_correction,
            compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
            use_wavelet_for_correction=use_wavelet_for_correction,
        )
        st.session_state.swv_annotation_signature = annotation_signature
        st.session_state.swv_annotated_results = results
    else:
        results = st.session_state.get("swv_annotated_results", results)
    st.caption(
        "The selected SWV trace now drives the derived SWV metrics as a set: peak height, "
        "peak voltage, bracket width, peak offset, skew, wavelet energy, and the related drift fields."
    )
    if display_subsection_bounds is not None:
        display_scan_start, display_scan_end = display_subsection_bounds
        results = [
            row for row in results
            if row.get("scan_number") is not None
            and display_scan_start <= int(row["scan_number"]) <= display_scan_end
        ]
    elif display_subsection_time_bounds is not None:
        display_time_start, display_time_end = display_subsection_time_bounds
        results = sorted(
            (
                row for row in results
                if row.get("measurement_time") is not None
                and display_time_start <= row["measurement_time"] <= display_time_end
            ),
            key=lambda row: (
                row["measurement_time"],
                _channel_display_sort_key(row.get("channel")),
                row.get("scan_number", math.inf),
            ),
        )

if analysis_mode == "CV":
    metric_cfg = {
        "Oxidation peak current": ("oxidation_peak_current", "Oxidation Peak Current (uA)"),
        "Oxidation peak voltage": ("oxidation_peak_voltage", "Oxidation Peak Voltage (V)"),
        "Reduction peak current": ("reduction_peak_current", "Reduction Peak Current (uA)"),
        "Reduction peak voltage": ("reduction_peak_voltage", "Reduction Peak Voltage (V)"),
        "Peak separation": ("peak_separation_V", "Peak Separation (V)"),
        "Peak current ratio": ("peak_current_ratio", "Oxidation / |Reduction|"),
        "Loop area": ("loop_area_abs", "Loop Area (uA*V)"),
    }
else:
    metric_cfg = {
        selected_peak_height_metric_label: ("peak_current_selected", selected_peak_height_ylabel),
        "Peak current (raw)":       ("peak_current_raw", "Raw Current at Peak (uA)"),
        "Bracket width (V)":        ("bracket_width_V",  "Distance between left/right correction anchors (V)"),
        "Skew":                     ("skew",             "Skew (corrected trace)"),
        "Peak offset (normalized)": ("peak_offset_norm", "Peak offset from bracket center (normalized, bracket-relative)"),
        "Wavelet energy":           ("wavelet_energy",   "Wavelet Energy (a.u.)"),
        "Background metrics | RMS": ("background_current_rms", "Outside-Crop RMS (uA)"),
    }
    if apply_background_recentering:
        metric_cfg["Background recentered peak"] = (
            "peak_current_background_recentered",
            "Background-Recentered Peak (uA)",
        )
    if not compute_skew:
        metric_cfg.pop("Skew", None)
    if not compute_wavelet_energy:
        metric_cfg.pop("Wavelet energy", None)

has_wavelet_denoised_trace = (
    analysis_mode == "SWV"
    and any(r.get("wavelet_denoised_current") is not None for r in results)
)

plot_scan_range = None if scan_windows else scan_range
active_vlines: List[Tuple[float, str]] = []
vlines: List[Tuple[float, str]] = []
enable_titration_analysis = False
titration_edge_trim_fraction = 0.15
fit_titration_langmuir = False
titration_concentration_unit = "uM"
titration_baseline_mode = "none"
titration_included_step_labels: Optional[List[str]] = None
remove_extreme_titration_outliers = False
show_titration_uloq = False
show_titration_lod = False
show_titration_fit_details = False
swv_method_filter_enabled = False
swv_method_filter_applied = False
selected_swv_method_groups: List[str] = []
if analysis_mode == "SWV":
    available_method_groups = sorted(
        {r.get("swv_method_group", "Unknown method") for r in results},
        key=_method_group_sort_key,
    )
    if available_method_groups:
        selected_groups_state = [
            group
            for group in st.session_state.get("swv_post_selected_method_groups", [])
            if group in available_method_groups
        ]
        if not selected_groups_state:
            selected_groups_state = list(available_method_groups)
        st.session_state["swv_post_selected_method_groups"] = selected_groups_state

    if available_method_groups:
        mf_c1, mf_c2 = st.columns([1, 3])
        swv_method_filter_enabled = mf_c1.checkbox(
            "Filter by SWV method",
            key="swv_post_method_filter_enabled",
            help="Uses the SWV method file to split the dataset into method groups without rerunning analysis.",
        )
        if swv_method_filter_enabled:
            selected_swv_method_groups = mf_c2.multiselect(
                "SWV method groups",
                options=available_method_groups,
                key="swv_post_selected_method_groups",
                help="Selected groups are concatenated into a fresh sequential scan axis for display.",
            )
        else:
            selected_swv_method_groups = list(available_method_groups)
    else:
        swv_method_filter_enabled = False
        selected_swv_method_groups = []
        st.caption("No SWV method metadata was found for this result set.")

    selected_group_set = set(selected_swv_method_groups)
    swv_method_filter_applied = bool(available_method_groups) and swv_method_filter_enabled and (
        selected_group_set != set(available_method_groups)
    )
    if swv_method_filter_applied:
        results = [
            r for r in results
            if r.get("swv_method_group", "Unknown method") in selected_group_set
        ]
        results, _, plot_scan_range = reindex_swv_results_for_display(results, [])
        st.caption(
            "SWV method filtering is display-only. The filtered subset is concatenated and renumbered "
            "without rerunning the expensive analysis."
        )

    detected_autotitration_vlines, autotitration_logs = (
        detect_autotitration_vlines(folders, results)
    )
    detected_autotitration_text = _vlines_input_text(
        detected_autotitration_vlines
    )
    autotitration_log_signature = tuple(
        (
            str(path),
            path.stat().st_mtime_ns,
            path.stat().st_size,
        )
        for path in autotitration_logs
        if path.is_file()
    )
    autotitration_detection_key = (
        AUTOTITRATION_VLINE_DETECTION_VERSION,
        current_selection_key,
        autotitration_log_signature,
        tuple(selected_swv_method_groups),
        len(results),
    )
    if (
        detected_autotitration_text
        and st.session_state.get("_swv_autotitration_detection_key")
        != autotitration_detection_key
    ):
        current_vline_text = str(
            st.session_state.get(
                "swv_post_vlines_input",
                DEFAULT_SWV_VLINES_TEXT,
            )
            or ""
        )
        previous_auto_text = str(
            st.session_state.get("_swv_autotitration_generated_vlines", "")
            or ""
        )
        if not current_vline_text.strip() or current_vline_text == previous_auto_text:
            st.session_state["swv_post_vlines_input"] = (
                detected_autotitration_text
            )
        st.session_state["_swv_autotitration_generated_vlines"] = (
            detected_autotitration_text
        )
        st.session_state["_swv_autotitration_detection_key"] = (
            autotitration_detection_key
        )

    with st.expander("Scan Annotations", expanded=False):
        if detected_autotitration_vlines:
            detected_concentrations = [
                label
                for _scan, label in detected_autotitration_vlines
                if label != "end"
            ]
            st.success(
                "Autotitration detected. Loaded concentration boundaries for "
                + ", ".join(detected_concentrations)
                + "."
            )
            st.caption(
                "The detected indices are loaded into Vline annotations below "
                "and can be edited before applying the display controls."
            )
        with st.form("swv_post_analysis_controls"):
            st.caption(
                "Vlines are written as scan,label. They use the current plotted scan axis, "
                "including subsection-relative numbering."
            )
            if use_swv_settings_grouping or use_swv_modulo_split:
                st.caption(
                    "Enter annotations on the source scan axis. Metrics plots "
                    "automatically remap them to each SWV-settings or modulo "
                    "group's local iteration axis."
                )
            st.caption(
                "For titration, consecutive vlines define steps. The concentration at the left vline "
                "is used for that step; the last vline simply closes the final interval."
            )
            st.text_area(
                "Vline annotations",
                height=180,
                key="swv_post_vlines_input",
                help=VLINE_ANNOTATION_HELP,
                placeholder=VLINE_ANNOTATION_PLACEHOLDER,
            )

            enable_titration_analysis = st.checkbox(
                "Treat vline intervals as titration steps",
                key="swv_enable_titration_analysis",
                help="Each interval between consecutive vertical lines becomes one titration step.",
            )
            if enable_titration_analysis:
                if not st.session_state.get("_swv_titration_trim_initialized_for_toggle", False):
                    st.session_state["swv_titration_edge_trim_fraction"] = 0.15
                    st.session_state["_swv_titration_trim_initialized_for_toggle"] = True
                titration_edge_trim_fraction = st.slider(
                    "Plateau edge trim fraction",
                    min_value=0.0,
                    max_value=0.4,
                    step=0.05,
                    key="swv_titration_edge_trim_fraction",
                    help="Uses only the middle portion of each step when estimating the plateau median.",
                )
                remove_extreme_titration_outliers = st.checkbox(
                    "Remove extreme titration outliers",
                    key="swv_remove_extreme_titration_outliers",
                    help=(
                        "Within each channel/method and titration interval, excludes only "
                        "extreme values (robust modified z-score above 5) from metric plots, "
                        "plateaus, Langmuir fits, SNR, and predicted concentrations."
                    ),
                )
                fit_titration_langmuir = st.checkbox(
                    "Fit Langmuir-style curve to step plateaus",
                    key="swv_fit_titration_langmuir",
                    help="Only reports Kd when titration vline labels include concentrations.",
                )
                show_titration_fit_details = st.checkbox(
                    "Show Kd and fit details on Langmuir plots",
                    key="swv_show_titration_fit_details",
                    help=(
                        "Adds Kd and saturation annotations plus the detailed fit note. "
                        "By default Langmuir plots show only fits, points, and error bars."
                    ),
                )
                show_titration_uloq = st.checkbox(
                    "Show ULOQ on plots",
                    key="swv_show_titration_uloq",
                    help=(
                        "Shows projected ULOQ lines, labels, and curve extensions. "
                        "ULOQ values remain available in tables and exports when hidden."
                    ),
                )
                show_titration_lod = st.checkbox(
                    "Show LOD on plots",
                    key="swv_show_titration_lod",
                    help=(
                        "Shows fitted LOD lines, labels, and the SNR = 3 cutoff region. "
                        "LOD values remain available in tables and exports when hidden."
                    ),
                )
                titration_concentration_unit = st.selectbox(
                    "Kd/report concentration unit",
                    options=["pM", "nM", "uM", "mM", "M"],
                    key="swv_titration_concentration_unit",
                    help=(
                        "Vline labels with units are converted into this unit for fitting/reporting. "
                        "Unitless numeric labels also use this unit."
                    ),
                )
                baseline_mode_label = st.selectbox(
                    "Titration baseline mode",
                    options=["None", "Immediately preceding buffer"],
                    key="swv_titration_baseline_mode",
                    help=(
                        "For alternating buffer/target runs, removes drift using the most "
                        "recent buffer and references every target to the buffer immediately "
                        "before the earliest selected target. The Langmuir baseline B is "
                        "fixed to that anchor buffer plateau."
                    ),
                )
                titration_baseline_mode = (
                    "preceding_buffer"
                    if baseline_mode_label == "Immediately preceding buffer"
                    else "none"
                )
                current_annotation_vlines, _selection_errors = parse_vlines(
                    st.session_state.get(
                        "swv_post_vlines_input",
                        DEFAULT_SWV_VLINES_TEXT,
                    )
                )
                concentration_options = titration_step_selection_options(
                    current_annotation_vlines
                )
                option_signature = tuple(concentration_options)
                previous_signature = tuple(
                    st.session_state.get(
                        "_swv_titration_concentration_option_signature",
                        (),
                    )
                )
                if option_signature != previous_signature:
                    previous_selection = st.session_state.get(
                        "swv_titration_included_step_labels"
                    )
                    if previous_selection is None or not previous_signature:
                        next_selection = concentration_options
                    else:
                        next_selection = [
                            label
                            for label in previous_selection
                            if label in concentration_options
                        ]
                        next_selection.extend(
                            label
                            for label in concentration_options
                            if label not in previous_signature
                        )
                    st.session_state["swv_titration_included_step_labels"] = (
                        next_selection
                    )
                    st.session_state[
                        "_swv_titration_concentration_option_signature"
                    ] = option_signature
                titration_included_step_labels = st.multiselect(
                    "Concentrations included in titration statistics",
                    options=concentration_options,
                    default=concentration_options,
                    key="swv_titration_included_step_labels",
                    help=(
                        "Only selected intervals are shown in titration statistics and "
                        "used for Langmuir, Kd, and LOD calculations. Buffers are numbered "
                        "chronologically. Deselected buffers may still correct drift for "
                        "their following target, but cannot define B or contribute to LOD."
                    ),
                )
                st.caption(
                    "For every Langmuir fit, B is fixed to the buffer immediately "
                    "before the earliest selected target concentration."
                )
                st.caption(
                    "Kd labels must start with concentration. Examples: "
                    "`40,10 uM`, `180,1.28 mM`, or `0,buffer`."
                )
            else:
                st.session_state["_swv_titration_trim_initialized_for_toggle"] = False
            st.form_submit_button("Apply Display Controls", use_container_width=True)

        vlines, vline_errors = parse_vlines(st.session_state.get("swv_post_vlines_input", DEFAULT_SWV_VLINES_TEXT))
        for err in vline_errors:
            st.warning(err)

        active_vlines = filter_vlines_to_results_axis(vlines, results)
        kept_vlines = len(active_vlines)
        if use_scan_range and scan_windows:
            st.caption(
                f"{kept_vlines} vline(s) are inside the current subsection axis built from {format_scan_windows(scan_windows)}."
            )
        elif swv_method_filter_applied:
            st.caption(
                f"{kept_vlines} vline(s) are inside the current filtered display axis."
            )
        else:
            st.caption(f"{kept_vlines} vline(s) are inside the current analyzed scan axis.")

titration_ready = enable_titration_analysis and len(active_vlines) >= 2
x_axis_label = (
    "Cycle number"
    if analysis_mode == "CV"
    else ("Filtered scan number" if swv_method_filter_applied else "Scan number")
)
ok_results     = [r for r in results if r.get("status") == "OK"]
failed_results = [r for r in results if r.get("status") == "FAILED"]
channel_indexes = build_channel_indexes(results, scan_range=plot_scan_range)
results_by_channel = channel_indexes["all_by_channel"]
failed_results_by_channel = channel_indexes["failed_in_range_by_channel"]
ok_plot_results_by_channel = channel_indexes["ok_in_range_by_channel"]
all_channels   = sorted(results_by_channel, key=_channel_display_sort_key)
channels_display = channels_to_plot if channels_to_plot else all_channels

plot_results = results
plot_active_vlines = active_vlines
plot_x_axis_label = (
    "SWV Measurement Number" if analysis_mode == "SWV" else x_axis_label
)
plot_display_scan_range = plot_scan_range
plot_channel_indexes = channel_indexes
plot_results_by_channel = results_by_channel
plot_failed_results_by_channel = failed_results_by_channel
plot_ok_results_by_channel = ok_plot_results_by_channel
plot_channels_display = channels_display
plot_vlines_by_channel: Dict[Any, List[Tuple[float, str]]] = {}
use_swv_display_grouping = (
    analysis_mode == "SWV"
    and (use_swv_settings_grouping or use_swv_modulo_split)
)
if use_swv_display_grouping:
    if use_swv_settings_grouping:
        plot_results = apply_swv_settings_split_for_display(results)
    else:
        plot_results = apply_swv_modulo_split_for_display(
            results,
            swv_modulo_split_count,
        )
    compute_drift_fields(plot_results)
    plot_active_vlines = active_vlines
    plot_display_scan_range = None
    plot_channel_indexes = build_channel_indexes(
        plot_results,
        scan_range=plot_display_scan_range,
    )
    plot_results_by_channel = plot_channel_indexes["all_by_channel"]
    plot_failed_results_by_channel = plot_channel_indexes["failed_in_range_by_channel"]
    plot_ok_results_by_channel = plot_channel_indexes["ok_in_range_by_channel"]
    selected_channel_set = (
        {str(channel) for channel in channels_display}
        if channels_display
        else None
    )
    plot_channels_display = [
        channel
        for channel in sorted(plot_results_by_channel, key=_channel_display_sort_key)
        if (
            selected_channel_set is None
            or str(plot_results_by_channel[channel][0].get("original_channel")) in selected_channel_set
        )
    ]
    plot_vlines_by_channel = {
        display_channel: remap_vlines_to_swv_display_group(
            active_vlines,
            plot_results_by_channel.get(display_channel, []),
        )
        for display_channel in plot_channels_display
    }
    plot_active_vlines = merge_group_vlines(
        list(plot_vlines_by_channel.values())
    )

# Titration analysis must use the same rows, local iteration axes, and annotation
# remapping as the SWV display. Otherwise interleaved methods are mistaken for
# chronological replicates of one plateau.
titration_results = plot_results
titration_channels = plot_channels_display
titration_active_vlines = plot_active_vlines
titration_vlines_by_channel = plot_vlines_by_channel or None
titration_scan_range = plot_display_scan_range
auto_split_titration_by_settings = False
if analysis_mode == "SWV" and enable_titration_analysis and not use_swv_display_grouping:
    selected_original_channels = set(channels_display)
    signatures_by_channel: Dict[Any, set] = {}
    for row in results:
        channel = row.get("channel")
        if channel in selected_original_channels:
            signatures_by_channel.setdefault(channel, set()).add(
                swv_settings_signature(row)
            )
    auto_split_titration_by_settings = any(
        len(signatures) > 1 for signatures in signatures_by_channel.values()
    )
    if auto_split_titration_by_settings:
        titration_results = apply_swv_settings_split_for_display(results)
        titration_channels = [
            channel
            for channel in sorted(
                {row.get("channel") for row in titration_results},
                key=_channel_display_sort_key,
            )
            if any(
                row.get("channel") == channel
                and row.get("original_channel") in selected_original_channels
                for row in titration_results
            )
        ]
        titration_vlines_by_channel = {
            channel: remap_vlines_to_swv_display_group(
                active_vlines,
                [row for row in titration_results if row.get("channel") == channel],
            )
            for channel in titration_channels
        }
        titration_active_vlines = merge_group_vlines(
            list(titration_vlines_by_channel.values())
        )
        titration_scan_range = None

fitted_langmuir_metric_labels: List[str] = []
langmuir_response_directions_by_metric: Dict[str, Dict[Any, str]] = {}
response_baselines_by_metric: Dict[str, Dict[Any, float]] = {}
if titration_ready and fit_titration_langmuir:
    for metric_label, (metric_key, _ylabel) in metric_cfg.items():
        if not supports_langmuir(metric_key):
            continue
        fit_rows = build_titration_langmuir_summary_table(
            titration_results,
            metric=metric_key,
            vlines=titration_active_vlines,
            channels=titration_channels,
            vlines_by_channel=titration_vlines_by_channel,
            scan_range=titration_scan_range,
            edge_trim_fraction=titration_edge_trim_fraction,
            concentration_unit=titration_concentration_unit,
            baseline_mode=titration_baseline_mode,
            included_step_labels=titration_included_step_labels,
            remove_extreme_outliers=remove_extreme_titration_outliers,
        )
        if any(row.get("langmuir_fit_used") for row in fit_rows):
            fitted_langmuir_metric_labels.append(metric_label)
        direction_candidates: Dict[Any, set] = {}
        for row in fit_rows:
            if not row.get("langmuir_fit_used"):
                continue
            direction = str(row.get("langmuir_response_direction", "")).strip()
            if direction not in {"signal-on", "signal-off"}:
                continue
            for channel_key in {
                row.get("channel"),
                row.get("original_channel"),
            }:
                if channel_key is not None:
                    direction_candidates.setdefault(channel_key, set()).add(direction)
        langmuir_response_directions_by_metric[metric_key] = {
            channel: next(iter(directions))
            for channel, directions in direction_candidates.items()
            if len(directions) == 1
        }

# A mixed-direction metric plot should not fall back to the generic palette
# merely because one channel's nonlinear fit failed. Supplement fitted
# directions with robust target-minus-preceding-buffer plateau changes.
if analysis_mode == "SWV" and (
    len(titration_active_vlines) >= 2 or titration_vlines_by_channel
):
    inferred_directions = infer_titration_response_directions(
        titration_results,
        metric="peak_current_selected",
        vlines=titration_active_vlines,
        channels=titration_channels,
        vlines_by_channel=titration_vlines_by_channel,
        scan_range=titration_scan_range,
        edge_trim_fraction=titration_edge_trim_fraction,
        concentration_unit=titration_concentration_unit,
        included_step_labels=titration_included_step_labels,
        remove_extreme_outliers=remove_extreme_titration_outliers,
    )
    peak_directions = langmuir_response_directions_by_metric.setdefault(
        "peak_current_selected",
        {},
    )
    inferred_original_direction_candidates: Dict[Any, set] = {}
    for channel, direction in inferred_directions.items():
        peak_directions.setdefault(channel, direction)
        original_channels = {
            row.get("original_channel", channel)
            for row in titration_results
            if row.get("channel") == channel
        }
        for original_channel in original_channels:
            inferred_original_direction_candidates.setdefault(
                original_channel,
                set(),
            ).add(direction)
    for original_channel, directions in (
        inferred_original_direction_candidates.items()
    ):
        if len(directions) == 1:
            peak_directions.setdefault(
                original_channel,
                next(iter(directions)),
            )
    langmuir_response_directions_by_metric["peak_current_raw"] = dict(
        peak_directions
    )
    for baseline_metric in (
        "peak_current_selected",
        "peak_current_raw",
        "wavelet_energy",
    ):
        channel_baselines = infer_titration_response_baselines(
            titration_results,
            metric=baseline_metric,
            vlines=titration_active_vlines,
            channels=titration_channels,
            vlines_by_channel=titration_vlines_by_channel,
            scan_range=titration_scan_range,
            edge_trim_fraction=titration_edge_trim_fraction,
            concentration_unit=titration_concentration_unit,
            included_step_labels=titration_included_step_labels,
            remove_extreme_outliers=remove_extreme_titration_outliers,
        )
        mapped_baselines = dict(channel_baselines)
        original_baseline_candidates: Dict[Any, List[float]] = {}
        for channel, baseline in channel_baselines.items():
            for row in titration_results:
                if row.get("channel") == channel:
                    original_baseline_candidates.setdefault(
                        row.get("original_channel", channel),
                        [],
                    ).append(float(baseline))
                    break
        for original_channel, baselines in original_baseline_candidates.items():
            mapped_baselines.setdefault(
                original_channel,
                float(np.median(baselines)),
            )
        response_baselines_by_metric[baseline_metric] = mapped_baselines

# Assign response colors once from the complete channel-option universe. Every
# plot receives this same mapping, so selecting a subset or changing diagnostic
# grouping cannot silently reassign a channel's shade.
response_palette_channels = list(dict.fromkeys([
    *plot_channels_display,
    *titration_channels,
]))
canonical_direction_sources = [
    langmuir_response_directions_by_metric.get("peak_current_selected", {}),
    *langmuir_response_directions_by_metric.values(),
]
consistent_response_directions: Dict[Any, str] = {}
for palette_channel in response_palette_channels:
    primary_direction = str(
        canonical_direction_sources[0].get(palette_channel, "")
    ).strip().lower()
    if primary_direction in {"signal-on", "signal-off"}:
        consistent_response_directions[palette_channel] = primary_direction
        continue
    direct_candidates = {
        str(direction_map.get(palette_channel, "")).strip().lower()
        for direction_map in canonical_direction_sources
        if str(direction_map.get(palette_channel, "")).strip().lower()
        in {"signal-on", "signal-off"}
    }
    if len(direct_candidates) == 1:
        consistent_response_directions[palette_channel] = next(
            iter(direct_candidates)
        )
        continue
    channel_rows = [
        row for row in titration_results
        if row.get("channel") == palette_channel
    ]
    original_channels = {
        row.get("original_channel", palette_channel)
        for row in channel_rows
    }
    primary_inherited_candidates = {
        str(canonical_direction_sources[0].get(original_channel, "")).strip().lower()
        for original_channel in original_channels
        if str(canonical_direction_sources[0].get(original_channel, "")).strip().lower()
        in {"signal-on", "signal-off"}
    }
    if len(primary_inherited_candidates) == 1:
        consistent_response_directions[palette_channel] = next(
            iter(primary_inherited_candidates)
        )
        continue
    inherited_candidates = {
        str(direction_map.get(original_channel, "")).strip().lower()
        for direction_map in canonical_direction_sources
        for original_channel in original_channels
        if str(direction_map.get(original_channel, "")).strip().lower()
        in {"signal-on", "signal-off"}
    }
    if len(inherited_candidates) == 1:
        consistent_response_directions[palette_channel] = next(
            iter(inherited_candidates)
        )

consistent_channel_colors: Dict[Any, Any] = {}
for response_direction, colormap_name in (
    ("signal-on", "Oranges"),
    ("signal-off", "Blues"),
):
    direction_channels = [
        channel for channel in response_palette_channels
        if consistent_response_directions.get(channel) == response_direction
    ]
    shade_values = _high_contrast_response_shades(len(direction_channels))
    consistent_channel_colors.update({
        channel: plt.get_cmap(colormap_name)(shade_values[index])
        for index, channel in enumerate(direction_channels)
    })

# Method identity takes precedence over response direction in SWV metric plots:
# Method 1 is the same blue in every physical channel, as is Method 2.
consistent_channel_colors.update({
    channel: method_color
    for channel in response_palette_channels
    if (method_color := _swv_method_blue(channel)) is not None
})

# A physical channel can be the selector option while its titration fits are
# automatically split into SWV methods. Reuse the selector option's exact shade
# whenever a child did not receive a direct palette assignment.
for palette_channel in response_palette_channels:
    if palette_channel in consistent_channel_colors:
        continue
    channel_rows = [
        row for row in titration_results
        if row.get("channel") == palette_channel
    ]
    parent_colors = {
        consistent_channel_colors[original_channel]
        for original_channel in {
            row.get("original_channel", palette_channel)
            for row in channel_rows
        }
        if original_channel in consistent_channel_colors
    }
    if len(parent_colors) == 1:
        consistent_channel_colors[palette_channel] = next(iter(parent_colors))

display_metric_cfg = (
    {
        label: metric_cfg[label]
        for label in fitted_langmuir_metric_labels
    }
    if enable_titration_analysis and fit_titration_langmuir
    else metric_cfg
)

ch_options = ["All channels"] + [f"Ch{ch}" for ch in plot_channels_display]

#  Summary banner 
c1, c2, c3, c4 = st.columns(4)
if analysis_mode == "CV":
    total_files = len({r.get("file_path") for r in results if r.get("file_path")})
    c1.metric("Cycles", len(results))
    c2.metric("Files", total_files)
    c3.metric("Failed cycles", len(failed_results))
    c4.metric("Channels found", len(all_channels))
else:
    c1.metric("Total files", len(results))
    c2.metric(" Successful", len(ok_results))
    c3.metric(" Failed", len(failed_results))
    c4.metric("Channels found", len(all_channels))

st.divider()
if not results:
    st.info("No measurements match the current SWV method filter.")
    st.stop()

if use_swv_display_grouping:
    setting_summary = []
    for display_channel in plot_channels_display:
        group_rows = plot_results_by_channel.get(display_channel, [])
        if not group_rows:
            continue
        first_row = group_rows[0]
        settings_label = (
            first_row.get("swv_settings_label")
            if use_swv_settings_grouping
            else first_row.get("modulo_group_settings_label")
        )
        settings_are_consistent = not str(settings_label).startswith("mixed settings")
        setting_summary.append({
            "Channel": first_row.get("original_channel"),
            "Group": first_row.get("display_group_index"),
            "SWV settings": settings_label,
            "Frequency (Hz)": first_row.get("swv_frequency_hz") if settings_are_consistent else None,
            "Sweep start (V)": first_row.get("swv_sweep_start_V") if settings_are_consistent else None,
            "Sweep end (V)": first_row.get("swv_sweep_end_V") if settings_are_consistent else None,
            "Step (V)": first_row.get("swv_step_size_V") if settings_are_consistent else None,
            "Amplitude (V)": first_row.get("swv_amplitude_V") if settings_are_consistent else None,
            "Measurements": len(group_rows),
        })
    with st.expander(
        (
            f"Detected SWV Setting Groups ({len(setting_summary)})"
            if use_swv_settings_grouping
            else f"Modulo Groups and Detected Settings ({len(setting_summary)})"
        ),
        expanded=True,
    ):
        if setting_summary:
            st.dataframe(pd.DataFrame(setting_summary), use_container_width=True, hide_index=True)
            if any(
                row.get("swv_settings_label") == "SWV settings unavailable"
                for row in plot_results
            ):
                missing_method_count = sum(
                    1 for row in plot_results
                    if (
                        row.get("swv_settings_label") == "SWV settings unavailable"
                        and not row.get("method_exists")
                    )
                )
                unparsed_method_count = sum(
                    1 for row in plot_results
                    if (
                        row.get("swv_settings_label") == "SWV settings unavailable"
                        and row.get("method_exists")
                    )
                )
                st.warning(
                    "Some measurements do not contain parseable SWV settings "
                    f"({missing_method_count} missing method files; "
                    f"{unparsed_method_count} method files found but not parsed). "
                    "Run Analysis again after method files are added or changed."
                )
            if any(
                str(row.get("modulo_group_settings_label", "")).startswith("mixed settings")
                for row in plot_results
            ):
                st.warning(
                    "At least one modulo group contains multiple SWV methods. Use "
                    "'SWV settings' grouping for an unambiguous comparison."
                )
        else:
            st.info("No SWV setting metadata is available for the selected channels.")

if analysis_mode == "SWV":
    iteration_values = [
        int(row["scan_number"])
        for row in plot_results
        if row.get("scan_number") is not None
    ]
    if iteration_values:
        iteration_min = min(iteration_values)
        iteration_max = max(iteration_values)
        with st.expander("Plot Iteration Range", expanded=False):
            limit_plot_iteration_range = st.checkbox(
                "Limit plotted iterations",
                key="swv_limit_plot_iteration_range",
                help=(
                    "Filters displayed and exported SWV plots without rerunning analysis. "
                    "When traces are grouped, the values use the group-local iteration axis."
                ),
            )
            range_signature = (
                iteration_min,
                iteration_max,
                swv_grouping_mode,
                int(swv_modulo_split_count),
                tuple(str(channel) for channel in plot_channels_display),
            )
            if st.session_state.get("_swv_plot_iteration_range_signature") != range_signature:
                st.session_state["_swv_plot_iteration_range_signature"] = range_signature
                st.session_state["swv_plot_iteration_start"] = iteration_min
                st.session_state["swv_plot_iteration_end"] = iteration_max

            range_c1, range_c2 = st.columns(2)
            iteration_start = int(range_c1.number_input(
                "First iteration",
                min_value=iteration_min,
                max_value=iteration_max,
                step=1,
                key="swv_plot_iteration_start",
                disabled=not limit_plot_iteration_range,
            ))
            iteration_end = int(range_c2.number_input(
                "Last iteration",
                min_value=iteration_min,
                max_value=iteration_max,
                step=1,
                key="swv_plot_iteration_end",
                disabled=not limit_plot_iteration_range,
            ))

            if limit_plot_iteration_range:
                selected_iteration_range = (
                    min(iteration_start, iteration_end),
                    max(iteration_start, iteration_end),
                )
                plot_display_scan_range = selected_iteration_range
                plot_active_vlines = [
                    (x, label)
                    for x, label in plot_active_vlines
                    if selected_iteration_range[0] <= x <= selected_iteration_range[1]
                ]
                plot_vlines_by_channel = {
                    channel: [
                        (x, label)
                        for x, label in channel_vlines
                        if selected_iteration_range[0]
                        <= x
                        <= selected_iteration_range[1]
                    ]
                    for channel, channel_vlines
                    in plot_vlines_by_channel.items()
                }
                plot_channel_indexes = build_channel_indexes(
                    plot_results,
                    scan_range=plot_display_scan_range,
                )
                plot_results_by_channel = plot_channel_indexes["all_by_channel"]
                plot_failed_results_by_channel = plot_channel_indexes["failed_in_range_by_channel"]
                plot_ok_results_by_channel = plot_channel_indexes["ok_in_range_by_channel"]
                st.caption(
                    f"Plotting iterations {selected_iteration_range[0]}–"
                    f"{selected_iteration_range[1]} (inclusive)."
                )

# 
# Tabs
# 
view_options = ["Overlays", "Metrics", "Drift", "Data Table", "Export"]
view_options.insert(3, "Failures")
view = st.radio("View", view_options, horizontal=True)



# 
# TAB: Overlays
# 
if view == "Overlays":
    st.subheader("Overlaid traces per channel")

    if analysis_mode == "CV":
        ov_c1, ov_c2, ov_c3, ov_c4, ov_c5 = st.columns([2, 2, 1, 1, 1])
        trace_type = ov_c1.radio(
            "Trace type",
            ["Smoothed", "Raw", "Detrended"],
            horizontal=True,
            key="cv_overlay_type",
        )
        cmap_name = ov_c2.selectbox(
            "Colour map",
            ["plasma", "viridis", "inferno", "magma", "cividis", "turbo"],
            key="cv_overlay_cmap",
        )
        show_peak_markers = ov_c3.checkbox(
            "Show peak points",
            value=True,
            help="Marks the oxidation and reduction peaks detected for each cycle.",
        )
        show_baseline = ov_c4.checkbox(
            "Show baseline",
            value=False,
            help="Shows the per-sweep linear background used for detrending on smoothed traces.",
        )
        show_peak_reference_vlines = ov_c5.checkbox(
            "Peak vlines",
            value=True,
            help="Adds vertical lines for initial, average, and final oxidation/reduction peak voltages in the displayed cycles.",
        )
        overlay_line_alpha = st.slider(
            "Overlay line opacity",
            min_value=0.05,
            max_value=1.0,
            value=0.90,
            step=0.05,
            key="cv_overlay_line_alpha",
            help="Lower values make overlaid trace lines more transparent.",
        )
        key_map = {
            "Raw": "raw_current",
            "Smoothed": "smoothed_current",
            "Detrended": "detrended_current",
        }
        y_key = key_map[trace_type]

        for ch in plot_channels_display:
            ch_res = plot_ok_results_by_channel.get(ch, [])
            if not ch_res:
                continue
            with st.expander(f"Channel {ch}  ({len(ch_res)} cycles)", expanded=len(plot_channels_display) <= 4):
                fig = plot_cv_overlaid_cycles(
                    ch_res,
                    y_key=y_key,
                    title=f"{trace_type}  Ch{ch}",
                    ylabel="Current (uA)",
                    colormap_name=cmap_name,
                    alpha=overlay_line_alpha,
                    show_peak_markers=show_peak_markers,
                    show_zero_baseline=(y_key == "detrended_current"),
                    show_baseline=show_baseline,
                    show_peak_reference_vlines=show_peak_reference_vlines,
                )
                if fig:
                    render_downloadable_pyplot(
                        st,
                        fig,
                        key=f"cv_overlay_{ch}_{y_key}",
                        file_stem=f"cv_overlay_ch{ch}_{trace_type}",
                    )
                else:
                    st.warning("No plottable traces for this channel.")
    else:
        overlay_groups_by_channel = False
        if use_swv_display_grouping:
            overlay_layout = st.radio(
                "Group layout",
                ["Separate group plots", "Overlay groups by channel"],
                horizontal=True,
                key="swv_overlay_group_layout",
                help=(
                    "Overlay groups by channel creates one plot per original channel "
                    "and assigns each group its sidebar colormap."
                ),
            )
            overlay_groups_by_channel = overlay_layout == "Overlay groups by channel"

        ov_c1, ov_c2, ov_c3, ov_c4, ov_c5 = st.columns([2, 2, 1, 1, 1])
        trace_type_options = [
            "Corrected",
            "Smoothed Corrected",
            "Normalized Smoothed Corrected",
            "Raw",
            "Offset Raw",
            "Smoothed",
        ]
        if has_wavelet_denoised_trace:
            trace_type_options.append("Wavelet Denoised")
        trace_type   = ov_c1.radio("Trace type", trace_type_options,
                                    horizontal=True, key="overlay_type")
        cmap_name    = ov_c2.selectbox("Colour map",
                                       ["plasma", "viridis", "inferno", "magma", "cividis", "turbo"],
                                       key="overlay_cmap",
                                       disabled=overlay_groups_by_channel,
                                       help=(
                                           "Used for separate plots. Grouped overlays use "
                                           "the colormap list in the left sidebar."
                                       ))
        show_anchors = ov_c3.checkbox("Show correction anchors", value=True,
                                      help="Dots mark the two bracketing-minima points used for baseline correction.")
        show_peak_markers = ov_c4.checkbox("Show peak points", value=False,
                                           help="Marks the detected peak on each displayed trace.")
        show_baseline = ov_c5.checkbox("Show 0 baseline", value=True,
                                       help="Draws a dashed horizontal zero-current reference line.")
        overlay_line_alpha = st.slider(
            "Overlay line opacity",
            min_value=0.05,
            max_value=1.0,
            value=0.85,
            step=0.05,
            key="swv_overlay_line_alpha",
            help="Lower values make overlaid trace lines more transparent.",
        )
        overlay_show_legend = bool(_resolve_swv_plot_setting(
            "swv_plot",
            "show_legend",
            True,
            inherit_metrics_style=True,
        ))
        overlay_show_grid = bool(_resolve_swv_plot_setting(
            "swv_plot",
            "show_grid",
            False,
            inherit_metrics_style=True,
        ))
        overlay_margin_px = float(_resolve_swv_plot_setting(
            "swv_plot",
            "margin_px",
            40,
            inherit_metrics_style=True,
        ))
        overlay_width_px = float(_resolve_swv_plot_setting(
            "swv_plot",
            "width_px",
            1200,
            inherit_metrics_style=True,
        )) * _SWV_OVERLAY_WIDTH_SCALE

        key_map = {
            "Corrected": "corrected_current",
            "Smoothed Corrected": "smoothed_corrected_current",
            "Normalized Smoothed Corrected": "smoothed_corrected_current",
            "Raw": "raw_current",
            "Offset Raw": "raw_current",
            "Smoothed": "smoothed_current",
            "Wavelet Denoised": "wavelet_denoised_current",
        }
        y_key = key_map[trace_type]
        normalize_to_peak = trace_type == "Normalized Smoothed Corrected"
        offset_to_baseline = trace_type == "Offset Raw"
        overlay_ylabel = (
            "Normalized current (peak = 1)"
            if normalize_to_peak
            else ("Offset raw current (uA)" if offset_to_baseline else "Current (uA)")
        )

        if overlay_groups_by_channel:
            grouped_channels = group_swv_display_channels(
                plot_results,
                plot_channels_display,
            )
            for original_ch, display_groups in grouped_channels.items():
                grouped_trace_sets = []
                total_trace_count = 0
                for group_position, display_group in enumerate(display_groups):
                    group_rows = plot_ok_results_by_channel.get(display_group, [])
                    if not group_rows:
                        continue
                    first_row = group_rows[0]
                    if use_swv_settings_grouping:
                        group_label = str(first_row.get("swv_settings_label") or display_group)
                    else:
                        group_label = (
                            f"Group {first_row.get('display_group_index')} | "
                            f"{first_row.get('modulo_group_settings_label') or 'settings unavailable'}"
                        )
                    group_colormap = swv_group_overlay_colormaps[
                        group_position % len(swv_group_overlay_colormaps)
                    ]
                    grouped_trace_sets.append(
                        (group_label, group_rows, group_colormap)
                    )
                    total_trace_count += len(group_rows)
                if not grouped_trace_sets:
                    continue
                with st.expander(
                    (
                        f"Channel {original_ch} ({len(grouped_trace_sets)} groups, "
                        f"{total_trace_count} traces)"
                    ),
                    expanded=len(grouped_channels) <= 4,
                ):
                    fig = plot_grouped_overlaid_traces(
                        grouped_trace_sets,
                        y_key=y_key,
                        title=format_swv_overlay_title(
                            original_ch,
                            [
                                row
                                for _, group_rows, _ in grouped_trace_sets
                                for row in group_rows
                            ],
                        ),
                        ylabel=overlay_ylabel,
                        alpha=overlay_line_alpha,
                        show_anchors=show_anchors,
                        show_peak_markers=(
                            show_peak_markers and y_key != "wavelet_denoised_current"
                        ),
                        show_zero_baseline=(
                            show_baseline
                            and (
                                y_key in ("corrected_current", "smoothed_corrected_current")
                                or offset_to_baseline
                            )
                        ),
                        normalize_to_peak=normalize_to_peak,
                        offset_to_baseline=offset_to_baseline,
                        colorbar_height_fraction=swv_colorbar_height_fraction,
                        colorbar_side=swv_colorbar_side,
                        show_legend=overlay_show_legend,
                        show_grid=overlay_show_grid,
                        outer_margin_fraction=(
                            overlay_margin_px / max(overlay_width_px, 1.0)
                        ),
                    )
                    if fig:
                        render_downloadable_pyplot(
                            st,
                            fig,
                            key=(
                                f"swv_grouped_overlay_{original_ch}_{y_key}_"
                                f"{normalize_to_peak}_{offset_to_baseline}"
                            ),
                            file_stem=(
                                f"swv_grouped_overlay_ch{original_ch}_{trace_type}"
                            ),
                            plot_kind="swv_trace",
                        )
                    else:
                        st.warning("No plottable traces for this channel.")
        else:
            for ch in plot_channels_display:
                ch_res = plot_ok_results_by_channel.get(ch, [])
                if not ch_res:
                    continue
                with st.expander(f"Channel {ch}  ({len(ch_res)} traces)", expanded=len(plot_channels_display) <= 4):
                    fig = plot_overlaid_traces(
                        ch_res, y_key=y_key,
                        title=format_swv_overlay_title(ch, ch_res),
                        ylabel=overlay_ylabel,
                        colormap_name=cmap_name,
                        alpha=overlay_line_alpha,
                        show_anchors=show_anchors,
                        show_peak_markers=(show_peak_markers and y_key != "wavelet_denoised_current"),
                        show_zero_baseline=(
                            show_baseline
                            and (
                                y_key in ("corrected_current", "smoothed_corrected_current")
                                or offset_to_baseline
                            )
                        ),
                        normalize_to_peak=normalize_to_peak,
                        offset_to_baseline=offset_to_baseline,
                    )
                    if fig:
                        render_downloadable_pyplot(
                            st,
                            fig,
                            key=(
                                f"swv_overlay_{ch}_{y_key}_{normalize_to_peak}_"
                                f"{offset_to_baseline}"
                            ),
                            file_stem=f"swv_overlay_ch{ch}_{trace_type}",
                            plot_kind="swv_trace",
                        )
                    else:
                        st.warning("No plottable traces for this channel.")


# 
# TAB: Metrics
# 
if view == "Metrics":
    st.subheader("Metrics")
    if analysis_mode == "SWV":
        if enable_titration_analysis and fit_titration_langmuir:
            metrics_settings_tab, langmuir_settings_tab = st.tabs([
                "All Metrics plot settings",
                "All Langmuir plot settings",
            ])
            with metrics_settings_tab:
                _render_all_metrics_plot_settings()
            with langmuir_settings_tab:
                _render_all_langmuir_plot_settings()
        else:
            _render_all_metrics_plot_settings()

    individual_method_view = "Individual channels per method"
    metric_view_options = ["Combined", "Individual channels"]
    if analysis_mode == "SWV":
        metric_view_options.append(individual_method_view)
    grouped_overlay_view = None
    if use_swv_settings_grouping:
        grouped_overlay_view = "Overlay SWV settings by channel"
    elif use_swv_modulo_split:
        grouped_overlay_view = "Overlay modulo groups by channel"
    if grouped_overlay_view is not None:
        metric_view_options.append(grouped_overlay_view)
    if st.session_state.get("metric_view_mode") not in metric_view_options:
        st.session_state["metric_view_mode"] = "Combined"
    view_mode = st.radio(
        "View mode",
        metric_view_options,
        horizontal=True,
        key="metric_view_mode",
    )
    combined_plot_channels = list(plot_channels_display)
    if view_mode == "Combined":
        combined_channel_options_signature = tuple(
            str(channel) for channel in plot_channels_display
        )
        previous_options_signature = st.session_state.get(
            "_metric_combined_channel_options_signature"
        )
        stored_combined_channels = st.session_state.get(
            "metric_combined_channels"
        )
        if previous_options_signature != combined_channel_options_signature:
            valid_stored_channels = [
                channel for channel in (stored_combined_channels or [])
                if channel in plot_channels_display
            ]
            if stored_combined_channels and not valid_stored_channels:
                valid_stored_channels = list(plot_channels_display)
            st.session_state["metric_combined_channels"] = (
                valid_stored_channels
                if stored_combined_channels is not None
                else list(plot_channels_display)
            )
            st.session_state[
                "_metric_combined_channel_options_signature"
            ] = combined_channel_options_signature
        combined_plot_channels = st.multiselect(
            "Channels to combine",
            options=plot_channels_display,
            format_func=_compact_titration_channel_label,
            key="metric_combined_channels",
            help="Only these channels are overlaid in Combined metric and titration plots.",
        )
        if not combined_plot_channels:
            st.warning("Select at least one channel to create combined plots.")

    combined_original_channels = set()
    for channel in combined_plot_channels:
        channel_rows = plot_results_by_channel.get(channel, [])
        if channel_rows:
            combined_original_channels.update(
                row.get("original_channel", channel) for row in channel_rows
            )
        else:
            combined_original_channels.add(channel)
    combined_titration_channels = [
        channel
        for channel in titration_channels
        if channel in combined_plot_channels
        or any(
            row.get("channel") == channel
            and row.get("original_channel", channel) in combined_original_channels
            for row in titration_results
        )
    ]
    combined_plot_active_vlines = (
        merge_group_vlines([
            plot_vlines_by_channel.get(channel, [])
            for channel in combined_plot_channels
        ])
        if plot_vlines_by_channel else plot_active_vlines
    )
    combined_titration_active_vlines = (
        merge_group_vlines([
            (titration_vlines_by_channel or {}).get(channel, [])
            for channel in combined_titration_channels
        ])
        if titration_vlines_by_channel else titration_active_vlines
    )

    normalized_peak_label = "Peak current (min–max normalized per channel)"
    percent_change_peak_label = "Peak current (% change from first value per channel)"
    metric_options = list(display_metric_cfg.keys())
    if (
        analysis_mode == "SWV"
        and view_mode in ("Combined", "Individual channels", individual_method_view)
        and not (enable_titration_analysis and fit_titration_langmuir)
    ):
        metric_options.extend([normalized_peak_label, percent_change_peak_label])
    stored_metric_selection = st.session_state.get("metrics_to_display")
    if stored_metric_selection is not None:
        st.session_state["metrics_to_display"] = [
            label for label in stored_metric_selection if label in metric_options
        ]

    metric_columns = st.columns([3, 1, 1]) if analysis_mode == "SWV" else st.columns([3, 1])
    m_c1, m_c2 = metric_columns[:2]
    default_metric_selection = (
        [selected_peak_height_metric_label]
        if (
            analysis_mode == "SWV"
            and selected_peak_height_metric_label in metric_options
        )
        else ([] if analysis_mode == "SWV" else metric_options)
    )
    selected_metrics = m_c1.multiselect(
        "Metrics to display",
        options=metric_options,
        default=default_metric_selection,
        key="metrics_to_display",
    )
    selected_fitted_langmuir_metric_labels = [
        label
        for label in fitted_langmuir_metric_labels
        if label in selected_metrics
    ]
    selected_diagnostic_metric_cfg = {
        label: display_metric_cfg[label]
        for label in selected_fitted_langmuir_metric_labels
    }
    if enable_titration_analysis and fit_titration_langmuir:
        if fitted_langmuir_metric_labels:
            st.caption(
                "Only metrics with at least one successful Langmuir fit under the "
                "current concentration and buffer selections are displayed."
            )
        else:
            st.warning(
                "No metric produced a successful Langmuir fit with the current "
                "concentration and buffer selections."
            )
    if percent_change_peak_label in selected_metrics:
        st.caption(
            "Percent change is calculated from each channel's first finite displayed value. "
            "A channel whose first value is zero is omitted from that plot."
        )
    if analysis_mode == "SWV":
        st.caption(
            "Background RMS is computed as the RMS of the raw signal outside the selected crop window, "
            "so the peak-analysis region is excluded by construction. Background drift uses the median "
            "background RMS of the first 3 valid scans in each channel as the reference."
        )
        if apply_background_recentering:
            st.caption(
                "Additive background recentering is active. The app estimates a signed outside-crop background "
                "offset from the raw trace, recenters each scan to its channel reference background, and then "
                "reruns the usual SWV correction workflow for the recentered peak only."
            )
    highlight_channel_options = (
        combined_plot_channels if view_mode == "Combined" else plot_channels_display
    )
    ch_options   = ["All channels"] + [f"Ch{ch}" for ch in highlight_channel_options]
    if st.session_state.get("metric_ch_sel") not in ch_options:
        st.session_state["metric_ch_sel"] = "All channels"
    ch_selection = m_c2.selectbox("Highlight channel", ch_options, key="metric_ch_sel",
                                   help="Selecting one channel dims the others.")
    highlight_ch = None
    if ch_selection != "All channels":
        highlight_ch = _channel_option_value(ch_selection)
    highlight_titration_channels = None
    if highlight_ch is not None:
        if auto_split_titration_by_settings:
            highlight_titration_channels = sorted(
                {
                    row.get("channel")
                    for row in titration_results
                    if row.get("original_channel") == highlight_ch
                },
                key=_channel_display_sort_key,
            )
        else:
            highlight_titration_channels = [highlight_ch]

    metric_x_key = "scan_number"
    metric_x_label = plot_x_axis_label
    if analysis_mode == "SWV":
        axis_options = ["Scan number", "Time from filename"]
        if st.session_state.get("swv_metric_x_axis") not in axis_options:
            st.session_state["swv_metric_x_axis"] = "Scan number"
        metric_axis = metric_columns[2].selectbox(
            "X axis",
            axis_options,
            key="swv_metric_x_axis",
            help="Time is read from the YYYYMMDD_HHMM portion of each native SWV filename.",
        )
        if metric_axis == "Time from filename":
            metric_x_key = "measurement_time"
            metric_x_label = "Measurement time"
            missing_time_count = sum(r.get("measurement_time") is None for r in plot_results)
            if missing_time_count == len(plot_results):
                st.warning(
                    "No YYYYMMDD_HHMM timestamps were found in the selected SWV filenames."
                )
            elif missing_time_count:
                st.caption(
                    f"{missing_time_count} file(s) without a native filename timestamp are omitted from time-axis plots."
                )

    grouped_channels_by_original = (
        group_swv_display_channels(plot_results, plot_channels_display)
        if view_mode == grouped_overlay_view
        else {}
    )
    method_view_results: List[dict] = []
    method_view_channels: List[Any] = []
    method_view_vlines_by_channel: Dict[Any, List[Tuple[float, str]]] = {}
    if view_mode == individual_method_view:
        if use_swv_settings_grouping:
            method_view_results = plot_results
            method_view_channels = list(plot_channels_display)
            method_view_vlines_by_channel = dict(plot_vlines_by_channel)
        else:
            method_view_results = apply_swv_settings_split_for_display(results)
            compute_drift_fields(method_view_results)
            selected_original_channels = set(channels_display)
            method_view_channels = [
                channel
                for channel in sorted(
                    {row.get("channel") for row in method_view_results},
                    key=_channel_display_sort_key,
                )
                if any(
                    row.get("channel") == channel
                    and row.get("original_channel") in selected_original_channels
                    for row in method_view_results
                )
            ]
            method_view_vlines_by_channel = {
                channel: remap_vlines_to_swv_display_group(
                    active_vlines,
                    [
                        row for row in method_view_results
                        if row.get("channel") == channel
                    ],
                )
                for channel in method_view_channels
            }
        st.caption(
            "Each plot contains one physical-channel/SWV-method combination; "
            "method iterations are kept independent."
        )
    if grouped_overlay_view is not None and view_mode == grouped_overlay_view:
        if use_swv_settings_grouping:
            st.caption(
                "Each plot represents one original channel. Curves are grouped by the "
                "full SWV settings shown in the legend and overlaid on their shared "
                "setting-local iteration axis."
            )
        else:
            st.caption(
                "Each plot represents one original channel. Its modulo groups are overlaid "
                "against their shared group-local iteration axis."
            )

    # Langmuir plots always group displayed SWV methods by physical channel so
    # the Optimized and Manual fits can be compared directly on one axis.
    langmuir_channels_by_original: Dict[Any, List[Any]] = {}
    langmuir_source_channels = (
        combined_titration_channels
        if view_mode == "Combined"
        else titration_channels
    )
    for display_channel in langmuir_source_channels:
        channel_rows = [
            row for row in titration_results
            if row.get("channel") == display_channel
        ]
        original_channel = (
            channel_rows[0].get("original_channel")
            if channel_rows else None
        )
        if original_channel is None:
            channel_match = re.match(
                r"(?:Ch(?:annel)?)?\s*(\d+)",
                str(display_channel),
                re.IGNORECASE,
            )
            original_channel = (
                int(channel_match.group(1))
                if channel_match else display_channel
            )
        channel_group = langmuir_channels_by_original.setdefault(
            original_channel,
            [],
        )
        if display_channel not in channel_group:
            channel_group.append(display_channel)

    if enable_titration_analysis:
        if not titration_ready:
            st.warning("Titration analysis needs at least two vertical lines inside the active scan range.")
        else:
            kept_pct = int(round((1.0 - (2.0 * titration_edge_trim_fraction)) * 100))
            kept_pct = max(kept_pct, 0)
            st.caption(
                f"Titration mode is on. Each vline interval becomes one step, and plateau values are "
                f"estimated from the median of the middle {kept_pct}% of scans in that step."
            )
            if remove_extreme_titration_outliers:
                st.caption(
                    "Extreme-outlier removal is active (modified z-score > 5 within "
                    "each channel/method and titration interval)."
                )
            if auto_split_titration_by_settings:
                st.caption(
                    "Multiple SWV methods were detected. Titration plateaus and fits are "
                    "automatically separated by the complete SWV settings."
                )
            if titration_baseline_mode == "preceding_buffer":
                st.caption(
                    "Targets are corrected as target − preceding buffer + first buffer. "
                    "Here, 'first buffer' means the buffer immediately before the earliest "
                    "selected target. The Langmuir baseline B is fixed to that plateau; unpaired "
                    "targets and buffer intervals are omitted."
                )
            if titration_included_step_labels is not None:
                st.caption(
                    f"Using {len(titration_included_step_labels)} of "
                    f"{len(concentration_options)} concentration label(s) for "
                    "titration statistics and fitting."
                )
            if fit_titration_langmuir:
                st.caption(
                    "Langmuir fits and concentration diagnostics are shown only for "
                    "the selected Metrics to display; "
                    "Kd requires concentrations in the vline labels. LOD uses 3σ of "
                    "the buffer plateau divided by the fitted initial slope."
                )

    for label in selected_metrics:
        normalize_per_channel = label == normalized_peak_label
        percent_change_per_channel = label == percent_change_peak_label
        normalization_mode = (
            "minmax" if normalize_per_channel
            else "percent" if percent_change_per_channel
            else "none"
        )
        if normalize_per_channel:
            metric = "peak_current_selected"
            ylabel = "Normalized Peak Current (0–1)"
        elif percent_change_per_channel:
            metric = "peak_current_selected"
            ylabel = "Peak Current Change from First Value (%)"
        else:
            metric, ylabel = metric_cfg[label]
        metric_response_directions = None
        if not normalize_per_channel and not percent_change_per_channel:
            metric_response_directions = (
                langmuir_response_directions_by_metric.get(metric)
                if metric in {"peak_current_selected", "peak_current_raw"}
                else langmuir_response_directions_by_metric.get(
                    "peak_current_selected"
                )
            )
        buffer_offset_metric_labels = {
            "peak_current_selected": "Change in Peak Height (uA)",
            "peak_current_raw": "Change in Peak Height (uA)",
            "wavelet_energy": "Wavelet Energy Change from Buffer (a.u.)",
        }
        metric_direction_colors_only = metric not in {
            "peak_current_selected",
            "peak_current_raw",
        }
        offset_combined_metric_to_buffer = (
            view_mode == "Combined"
            and len(combined_plot_channels) > 1
            and metric in buffer_offset_metric_labels
            and not normalize_per_channel
            and not percent_change_per_channel
        )
        combined_plot_ylabel = (
            buffer_offset_metric_labels[metric]
            if offset_combined_metric_to_buffer else ylabel
        )
        combined_normalization_mode = (
            "buffer_offset_combined"
            if offset_combined_metric_to_buffer else normalization_mode
        )
        metric_plot_results = (
            filter_extreme_titration_outliers(
                plot_results,
                metric=metric,
                vlines=plot_active_vlines,
                channels=plot_channels_display,
                vlines_by_channel=plot_vlines_by_channel or None,
            )
            if remove_extreme_titration_outliers and titration_ready
            else plot_results
        )
        method_metric_plot_results = method_view_results
        if (
            view_mode == individual_method_view
            and remove_extreme_titration_outliers
            and titration_ready
        ):
            method_metric_plot_results = filter_extreme_titration_outliers(
                method_view_results,
                metric=metric,
                vlines=merge_group_vlines(
                    list(method_view_vlines_by_channel.values())
                ),
                channels=method_view_channels,
                vlines_by_channel=method_view_vlines_by_channel,
            )
        st.markdown(f"**{label}**")

        if view_mode == "Combined":
            fig = (
                plot_metric_vs_scan(
                    metric_plot_results,
                    metric=metric,
                    channels=combined_plot_channels,
                    title=label,
                    ylabel=combined_plot_ylabel,
                    vlines=combined_plot_active_vlines,
                    scan_range=plot_display_scan_range,
                    highlight_channel=highlight_ch,
                    xlabel=metric_x_label,
                    x_key=metric_x_key,
                    normalize_per_channel=normalize_per_channel,
                    percent_change_per_channel=percent_change_per_channel,
                    response_directions=metric_response_directions,
                    response_baselines=response_baselines_by_metric.get(metric),
                    offset_to_response_baseline=offset_combined_metric_to_buffer,
                    response_direction_colors_only=(
                        metric_direction_colors_only
                        and not offset_combined_metric_to_buffer
                    ),
                    channel_colors=consistent_channel_colors,
                )
                if combined_plot_channels else None
            )
            if fig:
                render_downloadable_pyplot(
                    st,
                    fig,
                    key=(
                        f"metric_combined_{metric}_{highlight_ch or 'all'}_"
                        f"normalization_{combined_normalization_mode}"
                    ),
                    file_stem=f"metric_{label}_combined",
                )
        elif view_mode == individual_method_view:
            method_column_count = max(1, min(len(method_view_channels), 3))
            cols = st.columns(method_column_count)
            for i, ch in enumerate(method_view_channels):
                fig = plot_metric_vs_scan(
                    method_metric_plot_results,
                    metric=metric,
                    channels=[ch],
                    title=_compact_titration_channel_label(ch),
                    ylabel=ylabel,
                    vlines=method_view_vlines_by_channel.get(ch, []),
                    scan_range=None,
                    figsize=(5, 3),
                    xlabel=(
                        metric_x_label
                        if metric_x_key == "measurement_time"
                        else "SWV Measurement Number"
                    ),
                    x_key=metric_x_key,
                    normalize_per_channel=normalize_per_channel,
                    percent_change_per_channel=percent_change_per_channel,
                    response_directions=metric_response_directions,
                    response_direction_colors_only=metric_direction_colors_only,
                    channel_colors=consistent_channel_colors,
                )
                if fig:
                    with cols[i % method_column_count]:
                        render_downloadable_pyplot(
                            st,
                            fig,
                            key=(
                                f"metric_channel_method_{i}_{metric}_"
                                f"normalization_{normalization_mode}"
                            ),
                            file_stem=(
                                f"metric_{label}_{_compact_titration_channel_label(ch)}"
                            ),
                        )
        elif view_mode == "Individual channels":
            cols = st.columns(min(len(plot_channels_display), 3))
            for i, ch in enumerate(plot_channels_display):
                fig = plot_metric_vs_scan(
                    metric_plot_results, metric=metric, channels=[ch],
                    title=f"Channel {ch}", ylabel=ylabel,
                    vlines=(
                        plot_vlines_by_channel.get(ch, plot_active_vlines)
                        if use_swv_display_grouping
                        else plot_active_vlines
                    ),
                    scan_range=plot_display_scan_range, figsize=(5, 3),
                    xlabel=metric_x_label, x_key=metric_x_key,
                    normalize_per_channel=normalize_per_channel,
                    percent_change_per_channel=percent_change_per_channel,
                    response_directions=metric_response_directions,
                    response_direction_colors_only=metric_direction_colors_only,
                    channel_colors=consistent_channel_colors,
                )
                if fig:
                    with cols[i % min(len(plot_channels_display), 3)]:
                        render_downloadable_pyplot(
                            st,
                            fig,
                            key=(
                                f"metric_ch{ch}_{metric}_"
                                f"normalization_{normalization_mode}"
                            ),
                            file_stem=f"metric_{label}_ch{ch}",
                        )
        else:
            original_channels = list(grouped_channels_by_original)
            cols = st.columns(min(len(original_channels), 3))
            for i, (original_ch, display_groups) in enumerate(grouped_channels_by_original.items()):
                original_channel_results = [
                    row for row in metric_plot_results
                    if row.get("channel") in display_groups
                ]
                offset_group_metric_to_buffer = (
                    len(display_groups) > 1
                    and metric in buffer_offset_metric_labels
                    and not normalize_per_channel
                    and not percent_change_per_channel
                )
                fig = plot_metric_vs_scan(
                    original_channel_results,
                    metric=metric,
                    channels=display_groups,
                    title=f"Channel {original_ch}",
                    ylabel=(
                        buffer_offset_metric_labels[metric]
                        if offset_group_metric_to_buffer else ylabel
                    ),
                    vlines=merge_group_vlines([
                        plot_vlines_by_channel.get(display_group, [])
                        for display_group in display_groups
                    ]),
                    scan_range=plot_display_scan_range,
                    figsize=(5, 3),
                    xlabel=metric_x_label,
                    x_key=metric_x_key,
                    response_directions=metric_response_directions,
                    response_baselines=response_baselines_by_metric.get(metric),
                    offset_to_response_baseline=offset_group_metric_to_buffer,
                    response_direction_colors_only=(
                        metric_direction_colors_only
                        and not offset_group_metric_to_buffer
                    ),
                    channel_colors=consistent_channel_colors,
                )
                if fig:
                    with cols[i % min(len(original_channels), 3)]:
                        render_downloadable_pyplot(
                            st,
                            fig,
                            key=f"metric_group_overlay_ch{original_ch}_{metric}_{swv_grouping_mode}",
                            file_stem=f"metric_{label}_ch{original_ch}_group_overlay",
                        )

        if titration_ready and not normalize_per_channel and not percent_change_per_channel:
            st.caption("Titration plateaus")
            if view_mode == "Combined":
                offset_combined_plateaus_to_buffer = (
                    len(combined_titration_channels) > 1
                    and metric in buffer_offset_metric_labels
                )
                fig = (
                    plot_titration_plateaus(
                        titration_results,
                        metric=metric,
                        channels=combined_titration_channels,
                        title=f"{label} | plateau fit",
                        ylabel=(
                            buffer_offset_metric_labels[metric]
                            if offset_combined_plateaus_to_buffer else ylabel
                        ),
                        vlines=combined_titration_active_vlines,
                        vlines_by_channel=titration_vlines_by_channel,
                        scan_windows=None,
                        scan_range=titration_scan_range,
                        edge_trim_fraction=titration_edge_trim_fraction,
                        baseline_mode=titration_baseline_mode,
                        included_step_labels=titration_included_step_labels,
                        remove_extreme_outliers=remove_extreme_titration_outliers,
                        response_directions=metric_response_directions,
                        response_baselines=response_baselines_by_metric.get(metric),
                        offset_to_response_baseline=(
                            offset_combined_plateaus_to_buffer
                        ),
                        highlight_channel=(
                            highlight_ch
                            if not use_swv_display_grouping and not auto_split_titration_by_settings
                            else None
                        ),
                        channel_colors=consistent_channel_colors,
                    )
                    if combined_titration_channels else None
                )
                if fig:
                    render_downloadable_pyplot(
                        st,
                        fig,
                        key=f"titration_plateau_combined_{metric}_{highlight_ch or 'all'}",
                        file_stem=f"titration_plateau_{label}_combined",
                    )
            else:
                cols = st.columns(min(len(titration_channels), 3))
                for i, ch in enumerate(titration_channels):
                    fig = plot_titration_plateaus(
                        titration_results,
                        metric=metric,
                        channels=[ch],
                        title=f"Channel {ch} | plateau fit",
                        ylabel=ylabel,
                        vlines=titration_active_vlines,
                        vlines_by_channel=titration_vlines_by_channel,
                        scan_windows=None,
                        scan_range=titration_scan_range,
                        edge_trim_fraction=titration_edge_trim_fraction,
                        baseline_mode=titration_baseline_mode,
                        included_step_labels=titration_included_step_labels,
                        remove_extreme_outliers=remove_extreme_titration_outliers,
                        figsize=(5, 3),
                        response_directions=metric_response_directions,
                        channel_colors=consistent_channel_colors,
                    )
                    if fig:
                        with cols[i % min(len(titration_channels), 3)]:
                            render_downloadable_pyplot(
                                st,
                                fig,
                                key=f"titration_plateau_ch{ch}_{metric}",
                                file_stem=f"titration_plateau_{label}_ch{ch}",
                            )

            if fit_titration_langmuir and supports_langmuir(metric):
                st.caption(
                    "Langmuir-style fits of plateau midpoints; both methods are "
                    "overlaid within each physical channel."
                )
                langmuir_column_count = max(
                    1,
                    min(len(langmuir_channels_by_original), 2),
                )
                cols = st.columns(langmuir_column_count)
                for i, (original_ch, method_channels) in enumerate(
                    langmuir_channels_by_original.items()
                ):
                    fig = plot_titration_langmuir(
                            titration_results,
                            metric=metric,
                            channels=method_channels,
                            title=f"Channel {original_ch} | Langmuir Fit",
                            ylabel=(
                                "Peak Height (uA)"
                                if metric == "peak_current_selected"
                                else ylabel
                            ),
                            vlines=titration_active_vlines,
                            vlines_by_channel=titration_vlines_by_channel,
                            scan_windows=None,
                            scan_range=titration_scan_range,
                            edge_trim_fraction=titration_edge_trim_fraction,
                            figsize=(7, 4.25),
                            fit_langmuir=True,
                            concentration_unit=titration_concentration_unit,
                            baseline_mode=titration_baseline_mode,
                            included_step_labels=titration_included_step_labels,
                            remove_extreme_outliers=remove_extreme_titration_outliers,
                            show_uloq=show_titration_uloq,
                            show_lod=show_titration_lod,
                            show_fit_details=show_titration_fit_details,
                            show_legend=True,
                            response_directions=metric_response_directions,
                            channel_colors=consistent_channel_colors,
                        )
                    if fig:
                        with cols[i % langmuir_column_count]:
                            render_downloadable_pyplot(
                                st,
                                fig,
                                key=(
                                    f"titration_langmuir_channel_"
                                    f"{original_ch}_{metric}"
                                ),
                                file_stem=(
                                    f"titration_langmuir_{label}_"
                                    f"channel_{original_ch}"
                                ),
                                plot_kind="langmuir_fit",
                            )

        st.divider()

    if (
        titration_ready
        and fit_titration_langmuir
        and selected_fitted_langmuir_metric_labels
    ):
        st.markdown("### Titration SNR and concentration accuracy")
        st.caption(
            "SNR uses the median standard deviation of selected buffer plateaus. "
            "Concentrations are back-calculated for each individual SWV by inverting "
            "its fitted Langmuir curve."
        )
        titration_diagnostic_layout = st.radio(
            "SNR and concentration-accuracy plot grouping",
            options=["Per channel", "Per SWV group", "All groups"],
            horizontal=True,
            key="swv_titration_diagnostic_plot_grouping",
            help=(
                "Per channel overlays the fitted SWV-setting groups belonging to one "
                "physical channel. Per SWV group creates one plot for each fitted "
                "channel/method combination."
            ),
        )

        metrics_snr_rows = []
        snr_plot_rows_by_metric: Dict[str, List[dict]] = {}
        snr_fit_rows_by_metric: Dict[str, List[dict]] = {}
        for metric_label in selected_fitted_langmuir_metric_labels:
            metric_key, _ylabel = metric_cfg[metric_label]
            metric_step_rows = build_titration_step_table(
                titration_results,
                metric=metric_key,
                vlines=titration_active_vlines,
                vlines_by_channel=titration_vlines_by_channel,
                channels=titration_channels,
                scan_range=titration_scan_range,
                edge_trim_fraction=titration_edge_trim_fraction,
                concentration_unit=titration_concentration_unit,
                baseline_mode=titration_baseline_mode,
                included_step_labels=titration_included_step_labels,
                remove_extreme_outliers=remove_extreme_titration_outliers,
            )
            snr_plot_rows_by_metric[metric_label] = metric_step_rows
            metric_fit_rows = build_titration_langmuir_summary_table(
                titration_results,
                metric=metric_key,
                vlines=titration_active_vlines,
                vlines_by_channel=titration_vlines_by_channel,
                channels=titration_channels,
                scan_range=titration_scan_range,
                edge_trim_fraction=titration_edge_trim_fraction,
                concentration_unit=titration_concentration_unit,
                baseline_mode=titration_baseline_mode,
                included_step_labels=titration_included_step_labels,
                remove_extreme_outliers=remove_extreme_titration_outliers,
            )
            snr_fit_rows_by_metric[metric_label] = metric_fit_rows
            fit_by_channel = {row["channel"]: row for row in metric_fit_rows}
            for row in metric_step_rows:
                fit_row = fit_by_channel.get(row["channel"], {})
                metrics_snr_rows.append({
                    "Metric": metric_label,
                    "Channel / method": row["channel"],
                    "Selection": row.get("step_selection_key"),
                    "Concentration": row.get("step_concentration"),
                    "Unit": row.get("step_concentration_unit"),
                    "Plateau": row.get("plateau_value"),
                    "Fixed B": row.get("fixed_langmuir_baseline"),
                    "Selected-buffer noise SD": row.get("snr_noise_std"),
                    "Plateau SNR": row.get("titration_snr"),
                    "LOD": fit_row.get("limit_of_detection"),
                    "Fitted SNR=3 concentration": fit_row.get(
                        "snr_3_cutoff_concentration"
                    ),
                    "ULOQ": fit_row.get("upper_limit_of_quantification"),
                    "ULOQ projected": fit_row.get(
                        "upper_limit_of_quantification_is_extrapolated"
                    ),
                })

        if metrics_snr_rows:
            snr_df = pd.DataFrame(metrics_snr_rows)
            st.markdown("#### SNR by titration concentration")
            st.dataframe(snr_df, use_container_width=True, height=260)
            st.download_button(
                "Download titration SNR CSV",
                data=snr_df.to_csv(index=False).encode(),
                file_name=export_file_name(analysis_mode, "titration_snr"),
                mime="text/csv",
                use_container_width=True,
                key="metrics_titration_snr_download",
            )
            for metric_label, metric_rows in snr_plot_rows_by_metric.items():
                diagnostic_groups = _titration_diagnostic_row_groups(
                    metric_rows,
                    titration_diagnostic_layout,
                )
                plot_columns = (
                    st.columns(min(2, len(diagnostic_groups)))
                    if len(diagnostic_groups) > 1 else [st]
                )
                for group_index, (group_key, group_label, group_rows) in enumerate(
                    diagnostic_groups
                ):
                    group_channels = {row.get("channel") for row in group_rows}
                    group_fit_rows = [
                        row for row in snr_fit_rows_by_metric.get(metric_label, [])
                        if row.get("channel") in group_channels
                    ]
                    snr_figure = plot_titration_snr(
                        group_rows,
                        title=f"{metric_label} | {group_label} | SNR by concentration",
                        concentration_unit=titration_concentration_unit,
                        fit_summary_rows=group_fit_rows,
                        show_uloq=show_titration_uloq,
                        show_lod=show_titration_lod,
                        response_directions=(
                            consistent_response_directions
                        ),
                        channel_colors=consistent_channel_colors,
                    )
                    if snr_figure is not None:
                        render_downloadable_pyplot(
                            plot_columns[group_index % len(plot_columns)],
                            snr_figure,
                            key=(
                                f"titration_snr_plot_{metric_cfg[metric_label][0]}_"
                                f"{titration_diagnostic_layout}_{group_key}"
                            ),
                            file_stem=f"titration_snr_{metric_label}_{group_label}",
                        )
        else:
            st.info("No concentration-level SNR rows are available.")

        metrics_accuracy_rows = collect_titration_measurement_accuracy_rows(
            titration_results,
            metric_cfg=selected_diagnostic_metric_cfg,
            channels=titration_channels,
            vlines=titration_active_vlines,
            vlines_by_channel=titration_vlines_by_channel,
            scan_range=titration_scan_range,
            edge_trim_fraction=titration_edge_trim_fraction,
            concentration_unit=titration_concentration_unit,
            baseline_mode=titration_baseline_mode,
            included_step_labels=titration_included_step_labels,
            remove_extreme_outliers=remove_extreme_titration_outliers,
        )
        metrics_measurement_concentration_rows = (
            collect_titration_measurement_accuracy_rows(
                titration_results,
                metric_cfg=selected_diagnostic_metric_cfg,
                channels=titration_channels,
                vlines=titration_active_vlines,
                vlines_by_channel=titration_vlines_by_channel,
                scan_range=titration_scan_range,
                edge_trim_fraction=titration_edge_trim_fraction,
                concentration_unit=titration_concentration_unit,
                baseline_mode=titration_baseline_mode,
                included_step_labels=titration_included_step_labels,
                remove_extreme_outliers=remove_extreme_titration_outliers,
                include_buffer_measurements=True,
            )
        )
        st.markdown("#### Concentration accuracy for each measured SWV")
        if metrics_accuracy_rows:
            accuracy_display_df = pd.DataFrame([
                {
                    "Metric": row["metric_label"],
                    "Channel / method": row["channel"],
                    "Scan": row["scan_number"],
                    "Source scan": row["source_scan_number"],
                    "Selection": row["step_selection_key"],
                    "Known concentration": row["known_concentration"],
                    "Predicted concentration": row["predicted_concentration"],
                    "Raw mapped concentration": row.get(
                        "unbounded_predicted_concentration"
                    ),
                    "Reported at LOD": row.get(
                        "concentration_censored_at_lod", False
                    ),
                    "Predicted concentration SD": row.get(
                        "predicted_concentration_std"
                    ),
                    "Predicted concentration 1σ lower": row.get(
                        "predicted_concentration_lower_1sigma"
                    ),
                    "Predicted concentration 1σ upper": row.get(
                        "predicted_concentration_upper_1sigma"
                    ),
                    "Mapping uncertainty method": row.get(
                        "predicted_concentration_uncertainty_method"
                    ),
                    "Unit": row["concentration_unit"],
                    "Absolute error": row["absolute_concentration_error"],
                    "Signed error (%)": row["signed_percent_error"],
                    "Absolute error (%)": row["absolute_percent_error"],
                    "Log10 error": row["log10_concentration_error"],
                    "Measurement SNR": row["measurement_snr"],
                    "LOD": row.get("limit_of_detection"),
                    "ULOQ": row.get("upper_limit_of_quantification"),
                    "ULOQ projected": row.get(
                        "upper_limit_of_quantification_is_extrapolated"
                    ),
                    "File": row["file_name"],
                }
                for row in metrics_accuracy_rows
            ])
            st.dataframe(accuracy_display_df, use_container_width=True, height=340)
            valid_errors = accuracy_display_df["Absolute error (%)"].dropna()
            if not valid_errors.empty:
                valid_absolute_errors = accuracy_display_df["Absolute error"].dropna()
                rmse = (
                    float(np.sqrt(np.mean(np.square(valid_absolute_errors))))
                    if not valid_absolute_errors.empty else None
                )
                summary_cols = st.columns(4)
                summary_cols[0].metric(
                    "SWVs predicted",
                    f"{len(valid_errors)}",
                )
                summary_cols[1].metric(
                    "Median absolute error",
                    f"{valid_errors.median():.2f}%",
                )
                summary_cols[2].metric(
                    "Within ±20%",
                    f"{100.0 * (valid_errors <= 20.0).mean():.1f}%",
                )
                summary_cols[3].metric(
                    "RMSE",
                    (
                        f"{rmse:.4g} {titration_concentration_unit}"
                        if rmse is not None else "—"
                    ),
                    help=(
                        "Root mean square concentration prediction error. Lower is better; "
                        "zero is perfect agreement."
                    ),
                )
            st.caption(
                "These are back-calculated calibration residuals, not held-out "
                "validation errors. Blank predictions fall outside the physical "
                "Langmuir inversion domain."
            )
            st.download_button(
                "Download per-SWV concentration accuracy CSV",
                data=accuracy_display_df.to_csv(index=False).encode(),
                file_name=export_file_name(
                    analysis_mode,
                    "titration_measurement_accuracy",
                ),
                mime="text/csv",
                use_container_width=True,
                key="metrics_titration_accuracy_download",
            )
            for metric_label in selected_fitted_langmuir_metric_labels:
                metric_accuracy_rows = [
                    row for row in metrics_measurement_concentration_rows
                    if row["metric_label"] == metric_label
                ]
                diagnostic_groups = _titration_diagnostic_row_groups(
                    metric_accuracy_rows,
                    titration_diagnostic_layout,
                )
                plot_columns = (
                    st.columns(min(2, len(diagnostic_groups)))
                    if len(diagnostic_groups) > 1 else [st]
                )
                for group_index, (group_key, group_label, group_rows) in enumerate(
                    diagnostic_groups
                ):
                    group_channels = {
                        row.get("channel") for row in group_rows
                    }
                    group_measurement_vlines = (
                        merge_group_vlines([
                            titration_vlines_by_channel.get(channel, [])
                            for channel in group_channels
                        ])
                        if titration_vlines_by_channel
                        else titration_active_vlines
                    )
                    accuracy_figure = plot_titration_concentration_accuracy(
                        group_rows,
                        title="Predicted vs. Known",
                        concentration_unit=titration_concentration_unit,
                        show_uloq=show_titration_uloq,
                        show_lod=show_titration_lod,
                        channel_colors=consistent_channel_colors,
                        response_directions=consistent_response_directions,
                    )
                    if accuracy_figure is not None:
                        render_downloadable_pyplot(
                            plot_columns[group_index % len(plot_columns)],
                            accuracy_figure,
                            key=(
                                "titration_accuracy_plot_"
                                f"{metric_cfg[metric_label][0]}_"
                                f"{titration_diagnostic_layout}_{group_key}"
                            ),
                            file_stem=(
                                f"titration_accuracy_{metric_label}_{group_label}"
                            ),
                            plot_kind="concentration_accuracy",
                        )
                    measurement_figure = plot_titration_concentration_vs_measurement(
                        group_rows,
                        title=(
                            f"{group_label} | Concentration by Measurement"
                        ),
                        concentration_unit=titration_concentration_unit,
                        channel_colors=consistent_channel_colors,
                        response_directions=consistent_response_directions,
                        vlines=group_measurement_vlines,
                    )
                    if measurement_figure is not None:
                        render_downloadable_pyplot(
                            plot_columns[group_index % len(plot_columns)],
                            measurement_figure,
                            key=(
                                "titration_concentration_measurement_plot_"
                                f"{metric_cfg[metric_label][0]}_"
                                f"{titration_diagnostic_layout}_{group_key}"
                            ),
                            file_stem=(
                                f"titration_concentration_measurement_"
                                f"{metric_label}_{group_label}"
                            ),
                            plot_kind="concentration_measurement",
                        )
        else:
            st.info(
                "No individual SWV concentration predictions are available from "
                "the current fits."
            )


# 
# TAB: Drift
# 
if view == "Drift":
    st.subheader(
        "Drift metrics (relative to each channel's first cycle in the selected EC block)"
        if analysis_mode == "CV"
        else "Drift metrics (relative to each channel's first scan)"
    )
    if analysis_mode == "CV":
        st.markdown(
            "CV drift is computed **per channel** relative to the first valid cycle, so you can track "
            "how oxidation, reduction, and peak separation move over time."
        )
    else:
        st.markdown(
            "These metrics are computed **per channel** and the first valid scan for each channel "
            "is used as the reference (zero line). This lets you compare channels even if they "
            "started at different absolute values."
        )
        st.caption(
            "Peak voltage drift is the absolute peak-position shift. Peak offset (normalized) is "
            "relative to that scan's own left/right correction anchors, so whole-peak translations "
            "can stay small if the anchors move with the peak."
        )

    dr_c1, dr_c2 = st.columns([3, 1])
    drift_options = build_drift_options(
        analysis_mode,
        compute_skew=compute_skew if analysis_mode == "SWV" else True,
    )

    selected_drift = dr_c1.multiselect(
        "Drift metrics to display",
        options=list(drift_options.keys()),
        default=list(drift_options.keys()),
    )
    if st.session_state.get("drift_ch_sel") not in ch_options:
        st.session_state["drift_ch_sel"] = "All channels"
    dr_ch_sel = dr_c2.selectbox("Highlight channel", ch_options, key="drift_ch_sel")
    drift_highlight = None
    if dr_ch_sel != "All channels":
        drift_highlight = _channel_option_value(dr_ch_sel)

    drift_view_mode = st.radio("View mode", ["Combined", "Individual channels"],
                               horizontal=True, key="drift_view_mode")

    for label in selected_drift:
        drift_key, ylabel, caption = drift_options[label]
        st.markdown(f"**{label}**")
        st.caption(f"_{caption}_")

        if drift_view_mode == "Combined":
            fig = plot_drift_vs_scan(
                plot_results, drift_metric=drift_key, channels=plot_channels_display,
                title=label, ylabel=ylabel, vlines=plot_active_vlines,
                scan_range=plot_display_scan_range, highlight_channel=drift_highlight, xlabel=plot_x_axis_label,
                channel_colors=consistent_channel_colors,
            )
            if fig:
                render_downloadable_pyplot(
                    st,
                    fig,
                    key=f"drift_combined_{drift_key}_{drift_highlight or 'all'}",
                    file_stem=f"drift_{label}_combined",
                )
            else:
                st.warning(f"No data available for {label}.")
        else:
            cols = st.columns(min(len(plot_channels_display), 3))
            for i, ch in enumerate(plot_channels_display):
                fig = plot_drift_vs_scan(
                    plot_results, drift_metric=drift_key, channels=[ch],
                    title=f"Ch{ch}", ylabel=ylabel, vlines=plot_active_vlines,
                    scan_range=plot_display_scan_range, figsize=(5, 3), xlabel=plot_x_axis_label,
                    channel_colors=consistent_channel_colors,
                )
                if fig:
                    with cols[i % min(len(plot_channels_display), 3)]:
                        render_downloadable_pyplot(
                            st,
                            fig,
                            key=f"drift_ch{ch}_{drift_key}",
                            file_stem=f"drift_{label}_ch{ch}",
                        )

        st.divider()


# 
# TAB: Failures
# 
if view == "Failures":
    st.subheader(
        f"Failed cycles  ({len(failed_results)} total)"
        if analysis_mode == "CV"
        else f"Failed traces  ({len(failed_results)} total)"
    )

    if not failed_results:
        st.success("No failures ")
    else:
        if analysis_mode == "CV":
            fail_df = pd.DataFrame([
                {
                    "EC": r.get("ec_label", ""),
                    "Channel": r["channel"],
                    "Cycle #": r["scan_number"],
                    "File": r.get("file_name", ""),
                    "Error": r.get("error", ""),
                }
                for r in failed_results
            ])
            st.dataframe(fail_df, use_container_width=True, height=220)
            st.divider()

            for ch in plot_channels_display:
                ch_failed = plot_failed_results_by_channel.get(ch, [])
                if not ch_failed:
                    continue
                with st.expander(f"Ch{ch}  {len(ch_failed)} failed cycles", expanded=False):
                    for yk, title_suffix in (
                        ("raw_current", "Raw"),
                        ("smoothed_current", "Smoothed"),
                        ("detrended_current", "Detrended"),
                    ):
                        fig = plot_cv_overlaid_cycles(
                            ch_failed,
                            y_key=yk,
                            title=f"Ch{ch} failed cycles | {title_suffix}",
                            ylabel="Current (uA)",
                            show_peak_markers=False,
                            show_zero_baseline=(yk == "detrended_current"),
                            show_baseline=False,
                            show_peak_reference_vlines=False,
                        )
                        if fig:
                            render_downloadable_pyplot(
                                st,
                                fig,
                                key=f"cv_failed_overlay_ch{ch}_{yk}",
                                file_stem=f"cv_failed_ch{ch}_{title_suffix}",
                            )

            st.divider()
            st.markdown("#### Failed-cycle inspector")
            fail_options_map = {
                f"{r.get('ec_label', 'CV')}  Ch{r['channel']}  Cycle {r['scan_number']}  {r.get('file_name','')}": r
                for r in failed_results
            }
            chosen_label = st.selectbox("Pick a failed cycle", list(fail_options_map.keys()), key="cv_failed_cycle_sel")
            if chosen_label:
                chosen = fail_options_map[chosen_label]
                st.caption(f"Error: {chosen.get('error', '')}")
                if chosen.get("partial_error"):
                    st.caption(f"Partial trace note: {chosen.get('partial_error')}")
                fig = plot_cv_trace(chosen)
                if fig:
                    render_downloadable_pyplot(
                        st,
                        fig,
                        key=(
                            f"cv_failed_trace_ch{chosen.get('channel')}_"
                            f"{chosen.get('scan_number')}"
                        ),
                        file_stem=(
                            f"cv_failed_trace_ch{chosen.get('channel')}_"
                            f"cycle_{chosen.get('scan_number')}"
                        ),
                    )
                else:
                    st.warning("No trace data available for this failed cycle.")
        else:
            fail_df = pd.DataFrame([
                {"Channel": r["channel"], "Scan #": r["scan_number"],
                 "File": r.get("file_name", ""), "Error": r.get("error", "")}
                for r in failed_results
            ])
            st.dataframe(fail_df, use_container_width=True, height=200)
            st.divider()

            for ch in plot_channels_display:
                ch_failed = plot_failed_results_by_channel.get(ch, [])
                if not ch_failed:
                    continue
                to_plot = ch_failed[:int(max_failed)]
                with st.expander(f"Ch{ch}  {len(ch_failed)} failures", expanded=False):
                    for yk, yl in (
                        ("raw_current",       "Raw Current (uA)"),
                        ("smoothed_current",  "Smoothed Current (uA)"),
                        ("wavelet_denoised_current", "Wavelet-Denoised Current (uA)"),
                        ("corrected_current", "Corrected Current (uA)"),
                        ("smoothed_corrected_current", "Smoothed Corrected Current (uA)"),
                    ):
                        if yk == "wavelet_denoised_current" and not has_wavelet_denoised_trace:
                            continue
                        fig = plot_failed_traces(
                            to_plot, y_key=yk, ylabel=yl,
                            title=f"Ch{ch}  {yl}",
                            show_peak_markers=(yk not in ("raw_current", "wavelet_denoised_current")),
                            show_zero_baseline=(yk in ("corrected_current", "smoothed_corrected_current")),
                            show_local_baselines=(yk == "smoothed_current"),
                            show_minima_candidates=(yk == "smoothed_current"),
                        )
                        if fig:
                            render_downloadable_pyplot(
                                st,
                                fig,
                                key=f"swv_failed_overlay_ch{ch}_{yk}",
                                file_stem=f"swv_failed_ch{ch}_{yk}",
                            )

            st.divider()
            st.markdown("####  Single-trace inspector")
            fail_options_map = {
                f"Ch{r['channel']}  Scan {r['scan_number']}  {r.get('file_name','')}": r
                for r in failed_results
            }
            chosen_label = st.selectbox("Pick a failed trace", list(fail_options_map.keys()))
            if chosen_label:
                chosen = fail_options_map[chosen_label]
                if chosen.get("error"):
                    st.caption(f"Error: {chosen['error']}")
                if chosen.get("voltage") is not None:
                    fig = plot_single_trace(chosen)
                    render_downloadable_pyplot(
                        st,
                        fig,
                        key=(
                            f"swv_failed_trace_ch{chosen.get('channel')}_"
                            f"{chosen.get('scan_number')}"
                        ),
                        file_stem=(
                            f"swv_failed_trace_ch{chosen.get('channel')}_"
                            f"scan_{chosen.get('scan_number')}"
                        ),
                    )
                else:
                    st.warning("No trace data available for this file.")


# 
# TAB: Data Table
# 
if view == "Data Table":
    st.subheader("Results table")

    if analysis_mode == "CV":
        scalar_keys = [
            "channel", "ec_label", "measurement_index", "scan_number", "original_scan_number",
            "cycle_count_in_file", "method_nscans", "file_name", "status",
            "oxidation_peak_voltage", "oxidation_peak_current", "oxidation_peak_prominence",
            "reduction_peak_voltage", "reduction_peak_current", "reduction_peak_prominence",
            "peak_separation_V", "peak_current_ratio", "loop_area_abs",
            "oxidation_peak_voltage_drift", "reduction_peak_voltage_drift",
            "peak_separation_drift", "loop_area_abs_drift", "error",
        ]
    else:
        scalar_keys = [
            "channel", "swv_method_group", "swv_frequency_hz",
            "scan_number", "filtered_source_scan_number", "original_scan_number",
            "measurement_time", "file_name", "status",
            "peak_voltage", "peak_current_selected", "peak_current_background_drift_corrected", "peak_current_background_recentered", "peak_current", "peak_current_smoothed_corrected",
            "peak_current_raw", "bracket_width_V",
            "skew", "peak_offset_norm", "wavelet_energy",
            "background_current_rms", "background_current_median", "background_drift_rms_reference",
            "background_current_median_reference", "background_current_offset_uA", "background_drift_percent",
            "peak_voltage_drift", "bracket_width_drift", "skew_drift", "peak_offset_norm_drift", "error",
        ]
    df = pd.DataFrame([{k: r.get(k) for k in scalar_keys} for r in results])

    tf1, tf2 = st.columns(2)
    status_filter = tf1.multiselect("Status",  ["OK", "FAILED"], default=["OK", "FAILED"])
    ch_filter     = tf2.multiselect("Channel", sorted(df["channel"].dropna().unique().tolist()),
                                    default=sorted(df["channel"].dropna().unique().tolist()))
    mask = df["status"].isin(status_filter) & df["channel"].isin(ch_filter)
    filtered_df = df[mask].reset_index(drop=True)
    filtered_results = [
        r for r in results
        if r.get("status") in status_filter and r.get("channel") in ch_filter
    ]
    filtered_titration_results = [
        r for r in titration_results
        if r.get("status") in status_filter
        and (
            r.get("original_channel", r.get("channel")) in ch_filter
        )
    ]
    filtered_titration_channels = sorted(
        {r.get("channel") for r in filtered_titration_results},
        key=_channel_display_sort_key,
    )

    st.dataframe(filtered_df, use_container_width=True, height=400)
    st.caption(f"{mask.sum()} rows shown")

    if enable_titration_analysis:
        st.divider()
        st.markdown("#### Titration step table")
        if not titration_ready:
            st.info("Add at least two vertical lines inside the active scan range to build titration steps.")
        else:
            titration_table_metric_cfg = (
                display_metric_cfg
                if fit_titration_langmuir
                else metric_cfg
            )
            default_titration_metrics = (
                [selected_peak_height_metric_label]
                if selected_peak_height_metric_label in titration_table_metric_cfg
                else list(titration_table_metric_cfg.keys())[:1]
            )
            stored_titration_metrics = st.session_state.get(
                "table_titration_metrics"
            )
            if stored_titration_metrics is not None:
                st.session_state["table_titration_metrics"] = [
                    label
                    for label in stored_titration_metrics
                    if label in titration_table_metric_cfg
                ]
            titration_metric_labels = st.multiselect(
                "Titration metrics to tabulate",
                options=list(titration_table_metric_cfg.keys()),
                default=default_titration_metrics,
                key="table_titration_metrics",
            )
            titration_rows = []
            for label in titration_metric_labels:
                metric_key, ylabel = titration_table_metric_cfg[label]
                for row in build_titration_step_table(
                    filtered_titration_results,
                    metric=metric_key,
                    vlines=titration_active_vlines,
                    vlines_by_channel=titration_vlines_by_channel,
                    channels=filtered_titration_channels,
                    scan_range=titration_scan_range,
                    edge_trim_fraction=titration_edge_trim_fraction,
                    concentration_unit=titration_concentration_unit,
                    baseline_mode=titration_baseline_mode,
                    included_step_labels=titration_included_step_labels,
                    remove_extreme_outliers=remove_extreme_titration_outliers,
                ):
                    titration_rows.append({
                        "Metric": label,
                        "Channel": row["channel"],
                        "Step #": row["step_index"],
                        "Selection key": row.get("step_selection_key"),
                        "Step label": row["step_display_label"],
                        "Concentration": row["step_concentration"],
                        "Unit": row["step_concentration_unit"],
                        "Note": row["step_note"],
                        "Left marker": row["left_vline_label"],
                        "Right marker": row["right_vline_label"],
                        "Step start": row["step_start_scan"],
                        "Step end": row["step_end_scan"],
                        "Midpoint": row["midpoint_scan"],
                        "Plateau value": row["plateau_value"],
                        "Raw plateau value": row["raw_plateau_value"],
                        "Baseline step": row["baseline_step_index"],
                        "Baseline value": row["baseline_value"],
                        "Anchor buffer step": row.get("anchor_buffer_step_index"),
                        "Fixed Langmuir B": row.get("fixed_langmuir_baseline"),
                        "Plateau SNR": row.get("titration_snr"),
                        "SNR noise SD": row.get("snr_noise_std"),
                        "Plateau MAD": row["plateau_mad"],
                        "Step scans": row["step_scan_count"],
                        "Plateau scans": row["plateau_scan_count"],
                    })

            if titration_rows:
                titration_df = pd.DataFrame(titration_rows)
                st.dataframe(titration_df, use_container_width=True, height=260)
                st.caption(f"{len(titration_df)} titration step rows shown")
            else:
                st.info("No titration steps with valid plateau data match the current filters.")

        if fit_titration_langmuir:
            st.markdown("#### Langmuir fit summary")
            if not titration_ready:
                st.info("Add at least two vertical lines inside the active scan range to build Langmuir fits.")
            else:
                langmuir_rows = collect_langmuir_summary_rows(
                    filtered_titration_results,
                    metric_cfg=metric_cfg,
                    channels=filtered_titration_channels,
                    vlines=titration_active_vlines,
                    vlines_by_channel=titration_vlines_by_channel,
                    scan_range=titration_scan_range,
                    edge_trim_fraction=titration_edge_trim_fraction,
                    concentration_unit=titration_concentration_unit,
                    baseline_mode=titration_baseline_mode,
                    included_step_labels=titration_included_step_labels,
                    remove_extreme_outliers=remove_extreme_titration_outliers,
                )
                if langmuir_rows:
                    langmuir_df = pd.DataFrame([
                        {
                            "Metric": row["metric_label"],
                            "Channel": row["channel"],
                            "Fit status": row["langmuir_fit_status"],
                            "Response direction": row.get(
                                "langmuir_response_direction"
                            ),
                            f"Kd ({row.get('langmuir_kd_unit') or titration_concentration_unit})": row["langmuir_kd"],
                            f"LOD ({row.get('limit_of_detection_unit') or titration_concentration_unit})": row["limit_of_detection"],
                            "LOD method": row["limit_of_detection_method"],
                            f"ULOQ ({row.get('upper_limit_of_quantification_unit') or titration_concentration_unit})": row.get("upper_limit_of_quantification"),
                            "ULOQ method": row.get("upper_limit_of_quantification_method"),
                            "ULOQ noise SD": row.get("upper_limit_of_quantification_noise_sigma"),
                            "ULOQ noise source": row.get("upper_limit_of_quantification_noise_source"),
                            "ULOQ projected beyond data": row.get("upper_limit_of_quantification_is_extrapolated"),
                            "Baseline": row["langmuir_baseline"],
                            "Baseline fixed": row.get("langmuir_baseline_fixed", False),
                            "Anchor buffer step": row.get("anchor_buffer_step_index"),
                            "Amplitude": row["langmuir_amplitude"],
                            "Saturation step": row["saturation_step_index"],
                            "Saturation concentration": row["saturation_concentration"],
                            "Unit": row["fit_axis_unit"],
                            "Saturation plateau": row["saturation_plateau_value"],
                            "Step count": row["step_count"],
                            "Pre-sat steps": row["pre_saturation_step_count"],
                            "Post-sat steps": row["post_saturation_step_count"],
                            "Post-sat poly degree": row["post_saturation_polynomial_degree"],
                            "Left marker @ sat": row["saturation_left_vline_label"],
                            "Right marker @ sat": row["saturation_right_vline_label"],
                        }
                        for row in langmuir_rows
                    ])
                    st.dataframe(langmuir_df, use_container_width=True, height=220)
                    st.caption(
                        "Kd is reported only when the fitted titration steps have numeric concentrations "
                        "from their left vline labels. LOD is reported only when buffer "
                        "plateaus provide a nonzero noise estimate."
                    )
                else:
                    st.info("No Langmuir fit summaries are available for the current filters.")

                accuracy_rows = collect_titration_measurement_accuracy_rows(
                    filtered_titration_results,
                    metric_cfg=metric_cfg,
                    channels=filtered_titration_channels,
                    vlines=titration_active_vlines,
                    vlines_by_channel=titration_vlines_by_channel,
                    scan_range=titration_scan_range,
                    edge_trim_fraction=titration_edge_trim_fraction,
                    concentration_unit=titration_concentration_unit,
                    baseline_mode=titration_baseline_mode,
                    included_step_labels=titration_included_step_labels,
                    remove_extreme_outliers=remove_extreme_titration_outliers,
                )
                st.markdown("#### Per-SWV concentration accuracy")
                if accuracy_rows:
                    accuracy_df = pd.DataFrame(accuracy_rows)
                    st.dataframe(accuracy_df, use_container_width=True, height=300)
                    valid_accuracy = accuracy_df["absolute_percent_error"].dropna()
                    if not valid_accuracy.empty:
                        valid_absolute_errors = accuracy_df[
                            "absolute_concentration_error"
                        ].dropna()
                        rmse = (
                            float(np.sqrt(np.mean(np.square(valid_absolute_errors))))
                            if not valid_absolute_errors.empty else None
                        )
                        st.caption(
                            f"{len(valid_accuracy)} SWVs inverted successfully; median absolute "
                            f"back-calculated concentration error = {valid_accuracy.median():.2f}%; "
                            f"RMSE = {rmse:.4g} {titration_concentration_unit}. "
                            "These are calibration-fit residuals, not held-out validation errors."
                        )
                else:
                    st.info(
                        "No per-SWV concentration predictions are available from the "
                        "current selected concentrations and Langmuir fits."
                    )

    if analysis_mode == "SWV":
        st.divider()
        st.markdown("#### Single-trace inspector")

        if not filtered_results:
            st.info("No measurements match the current filters.")
        else:
            measurement_options = {
                f"Ch{r['channel']}  Scan {r['scan_number']}  {r.get('status', '')}  {r.get('file_name', '')}": r
                for r in filtered_results
            }
            chosen_label = st.selectbox("Pick a measurement", list(measurement_options.keys()))
            chosen = measurement_options[chosen_label]

            meta_cols = st.columns(4)
            meta_cols[0].caption(f"Channel: {chosen.get('channel', '')}")
            meta_cols[1].caption(f"Scan: {chosen.get('scan_number', '')}")
            meta_cols[2].caption(f"Status: {chosen.get('status', '')}")
            meta_cols[3].caption(f"File: {chosen.get('file_name', '')}")
            if chosen.get("swv_method_group"):
                st.caption(f"Method group: {chosen.get('swv_method_group')}")
            if chosen.get("filtered_source_scan_number") is not None:
                st.caption(f"Source scan on prior axis: {chosen.get('filtered_source_scan_number')}")

            if chosen.get("error"):
                st.caption(f"Error: {chosen.get('error')}")

            if chosen.get("voltage") is not None:
                fig = plot_single_trace(chosen)
                render_downloadable_pyplot(
                    st,
                    fig,
                    key=(
                        f"table_swv_trace_ch{chosen.get('channel')}_"
                        f"{chosen.get('scan_number')}_{chosen.get('status')}"
                    ),
                    file_stem=(
                        f"swv_trace_ch{chosen.get('channel')}_"
                        f"scan_{chosen.get('scan_number')}"
                    ),
                )
            else:
                st.warning("No trace data available for this measurement.")
    elif filtered_results:
        with st.expander("Cycle diagnostics (optional)", expanded=False):
            measurement_options = {
                f"{r.get('ec_label', 'CV')}  Ch{r['channel']}  Cycle {r['scan_number']}  {r.get('status', '')}  {r.get('file_name', '')}": r
                for r in filtered_results
            }
            chosen_label = st.selectbox("Pick a CV cycle", list(measurement_options.keys()), key="cv_cycle_diag")
            chosen = measurement_options[chosen_label]
            if chosen.get("error"):
                st.caption(f"Error: {chosen.get('error')}")
            fig = plot_cv_trace(chosen)
            if fig:
                render_downloadable_pyplot(
                    st,
                    fig,
                    key=(
                        f"table_cv_trace_ch{chosen.get('channel')}_"
                        f"{chosen.get('scan_number')}_{chosen.get('status')}"
                    ),
                    file_stem=(
                        f"cv_trace_ch{chosen.get('channel')}_"
                        f"cycle_{chosen.get('scan_number')}"
                    ),
                )


# 
# TAB: Export
# 
if view == "Export":
    st.subheader("Export results")

    export_metadata = build_export_metadata(
        analysis_mode=analysis_mode,
        crop_range=(crop_min, crop_max),
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        active_vlines=active_vlines,
        selected_channels=channels_display,
        scan_windows=scan_windows,
        scan_range=plot_scan_range,
        time_range=time_range,
        minima_search_window_V=minima_search_window if analysis_mode == "SWV" else None,
        min_peak_height_uA=min_peak_height if analysis_mode == "SWV" else None,
        min_start_voltage_V=min_start_voltage if analysis_mode == "SWV" else None,
        edge_trim_fraction=edge_trim_fraction if analysis_mode == "CV" else None,
        min_peak_prominence_uA=min_peak_prominence if analysis_mode == "CV" else None,
        titration_edge_trim_fraction=titration_edge_trim_fraction if analysis_mode == "SWV" else None,
        peak_height_source_key=selected_peak_height_source if analysis_mode == "SWV" else None,
        peak_height_source_label=selected_peak_height_source_label if analysis_mode == "SWV" else None,
        compute_wavelet_denoised_trace=compute_wavelet_denoised_trace if analysis_mode == "SWV" else None,
        use_wavelet_for_correction=use_wavelet_for_correction if analysis_mode == "SWV" else None,
        titration_concentration_unit=titration_concentration_unit if analysis_mode == "SWV" else None,
        titration_baseline_mode=titration_baseline_mode if analysis_mode == "SWV" else None,
        titration_included_step_labels=(
            titration_included_step_labels if analysis_mode == "SWV" else None
        ),
        remove_extreme_titration_outliers=(
            remove_extreme_titration_outliers if analysis_mode == "SWV" else None
        ),
        show_titration_uloq=(
            show_titration_uloq if analysis_mode == "SWV" else None
        ),
        show_titration_lod=(
            show_titration_lod if analysis_mode == "SWV" else None
        ),
    )
    export_payload = build_experiment_export_payload(
        analysis_mode=analysis_mode,
        results=results,
        export_metadata=export_metadata,
        metric_cfg=metric_cfg,
        channels=channels_display,
        active_vlines=active_vlines,
        scan_range=plot_scan_range,
        enable_titration_analysis=enable_titration_analysis,
        titration_ready=titration_ready,
        titration_edge_trim_fraction=titration_edge_trim_fraction,
        fit_titration_langmuir=fit_titration_langmuir,
        titration_concentration_unit=titration_concentration_unit,
        titration_results=titration_results,
        titration_channels=titration_channels,
        titration_vlines=titration_active_vlines,
        titration_vlines_by_channel=titration_vlines_by_channel,
        titration_scan_range=titration_scan_range,
        titration_baseline_mode=titration_baseline_mode,
        titration_included_step_labels=titration_included_step_labels,
        remove_extreme_titration_outliers=remove_extreme_titration_outliers,
    )

    st.markdown("#### Save experiment output folder")
    default_experiment_name = Path(folders[0]).name if folders else f"{analysis_mode.lower()}_analysis"
    experiment_name = st.text_input(
        "Experiment name",
        value=default_experiment_name,
        help="Used in the output folder name and manifest.",
        key="export_experiment_name",
    )
    experiment_notes = st.text_area(
        "Experiment notes",
        height=80,
        help="Optional notes saved in manifest.json for the future comparison app.",
        key="export_experiment_notes",
    )
    export_root = None
    if len(folders) == 1:
        export_root = Path(folders[0])
    elif len(folders) > 1:
        selected_export_root = st.selectbox(
            "Save outputs inside",
            options=folders,
            help=(
                "Multiple data folders are selected. Choose which folder should receive the outputs/ "
                "subfolder for this combined analysis. All source folders are still recorded in manifest.json."
            ),
            key="export_output_root",
        )
        export_root = Path(selected_export_root)
    if export_root is not None:
        st.caption(f"Output location: `{export_root / 'outputs'}`")
    else:
        st.info("Select at least one data folder before saving an experiment output bundle.")

    if st.button(
        "Save experiment output to outputs/",
        use_container_width=True,
        disabled=export_root is None,
    ):
        try:
            bundle_dir = write_experiment_output_bundle(
                export_root=export_root,
                experiment_name=experiment_name,
                experiment_notes=experiment_notes,
                analysis_mode=analysis_mode,
                source_folders=folders,
                export_payload=export_payload,
                export_metadata=export_metadata,
            )
            st.success(f"Saved experiment output bundle: `{bundle_dir}`")
        except FileExistsError:
            st.error("An output folder with this timestamp/name already exists. Try again in a moment.")
        except Exception as e:
            st.error(f"Could not save experiment output bundle: {e}")

    st.divider()

    st.markdown("####  Signal Processing Inputs CSV")
    signal_processing_csv = export_payload["signal_processing_inputs"].to_csv(index=False).encode()
    st.download_button(
        "  Download signal_processing_inputs.csv",
        data=signal_processing_csv,
        file_name=export_file_name(analysis_mode, "signal_processing_inputs"),
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("####  Results CSV")
    csv_bytes = export_payload["results"].to_csv(index=False).encode()
    st.download_button("  Download results.csv", data=csv_bytes,
                       file_name=export_file_name(analysis_mode, "results"), mime="text/csv",
                       use_container_width=True)

    if enable_titration_analysis:
        st.markdown("####  Titration step CSV")
        if not titration_ready:
            st.info("Add at least two vertical lines inside the active scan range to export titration steps.")
        elif "titration_steps" in export_payload:
            titration_csv = export_payload["titration_steps"].to_csv(index=False).encode()
            st.download_button(
                "  Download titration_steps.csv",
                data=titration_csv,
                file_name=export_file_name(analysis_mode, "titration_steps"),
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No titration step rows are available for export with the current settings.")

    if enable_titration_analysis and fit_titration_langmuir:
        st.markdown("####  Langmuir fit summary CSV")
        if not titration_ready:
            st.info("Add at least two vertical lines inside the active scan range to export Langmuir fit summaries.")
        elif "langmuir_fit_summary" in export_payload:
            langmuir_csv = export_payload["langmuir_fit_summary"].to_csv(index=False).encode()
            st.download_button(
                "  Download langmuir_fit_summary.csv",
                data=langmuir_csv,
                file_name=export_file_name(analysis_mode, "langmuir_fit_summary"),
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No Langmuir fit summary rows are available for export with the current settings.")

        st.markdown("#### Per-SWV concentration accuracy CSV")
        if "titration_measurement_accuracy" in export_payload:
            accuracy_csv = export_payload["titration_measurement_accuracy"].to_csv(
                index=False
            ).encode()
            st.download_button(
                "Download titration_measurement_accuracy.csv",
                data=accuracy_csv,
                file_name=export_file_name(
                    analysis_mode,
                    "titration_measurement_accuracy",
                ),
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No per-SWV concentration accuracy rows are available for export.")

    st.divider()

    drift_export_cfg = build_drift_options(
        analysis_mode,
        compute_skew=compute_skew if analysis_mode == "SWV" else True,
    )

    st.markdown("####  All plots PDF")
    pdf_c1, pdf_c2 = st.columns(2)
    pdf_metric_layout = pdf_c1.radio(
        "Metrics pages",
        ["Combined", "Individual channels"],
        horizontal=True,
        key="export_pdf_metric_layout",
        help="Combined puts all selected channels in each metric plot. Individual channels creates one page per metric type with separate channel plots.",
    )
    pdf_drift_layout = pdf_c2.radio(
        "Drift pages",
        ["Combined", "Individual channels"],
        horizontal=True,
        key="export_pdf_drift_layout",
        help="Combined puts all selected channels in each drift plot. Individual channels creates one page per drift type with separate channel plots.",
    )

    if st.button("  Build all plots PDF", use_container_width=True):
        pdf_bytes = build_export_pdf(
            analysis_mode=analysis_mode,
            results=plot_results,
            ok_results_by_channel=plot_ok_results_by_channel,
            channels=plot_channels_display,
            metric_cfg=display_metric_cfg,
            drift_cfg=drift_export_cfg,
            active_vlines=plot_active_vlines,
            scan_range=plot_display_scan_range,
            xlabel=plot_x_axis_label,
            metrics_layout=pdf_metric_layout,
            drift_layout=pdf_drift_layout,
            remove_extreme_titration_outliers=remove_extreme_titration_outliers,
            channel_colors=consistent_channel_colors,
        )
        st.download_button(
            "  Download all_plots.pdf",
            data=pdf_bytes,
            file_name="cv_all_plots.pdf" if analysis_mode == "CV" else "swv_all_plots.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    st.divider()

    st.markdown("####  Figures ZIP")
    fig_format = st.selectbox("Format", ["png", "pdf", "svg"], index=0)
    fig_dpi    = st.slider("DPI (PNG only)", 72, 300, 150)

    if st.button("  Build figures ZIP", use_container_width=True):
        zip_buf = io.BytesIO()

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:

            def _save(fig, path):
                buf = io.BytesIO()
                fig.savefig(buf, format=fig_format, dpi=fig_dpi, bbox_inches="tight")
                zf.writestr(path, buf.getvalue())
                plt.close(fig)

            for title, (metric, ylabel) in display_metric_cfg.items():
                offset_export_metric_to_buffer = (
                    len(combined_plot_channels) > 1
                    and metric in {"peak_current_selected", "peak_current_raw"}
                )
                export_response_directions = (
                    langmuir_response_directions_by_metric.get(metric)
                    if metric in {"peak_current_selected", "peak_current_raw"}
                    else langmuir_response_directions_by_metric.get(
                        "peak_current_selected"
                    )
                )
                export_metric_results = (
                    filter_extreme_titration_outliers(
                        plot_results,
                        metric=metric,
                        vlines=combined_plot_active_vlines,
                        channels=combined_plot_channels,
                        vlines_by_channel=plot_vlines_by_channel or None,
                    )
                    if remove_extreme_titration_outliers and titration_ready
                    else plot_results
                )
                fig = (
                    plot_metric_vs_scan(
                        export_metric_results,
                        metric=metric,
                        channels=combined_plot_channels,
                        title=title,
                        ylabel=(
                            "Change in Peak Height (uA)"
                            if offset_export_metric_to_buffer else ylabel
                        ),
                        vlines=combined_plot_active_vlines,
                        scan_range=plot_display_scan_range,
                        xlabel=plot_x_axis_label,
                        response_directions=export_response_directions,
                        response_baselines=response_baselines_by_metric.get(metric),
                        offset_to_response_baseline=offset_export_metric_to_buffer,
                        response_direction_colors_only=(
                            metric not in {
                                "peak_current_selected",
                                "peak_current_raw",
                            }
                        ),
                        channel_colors=consistent_channel_colors,
                    )
                    if combined_plot_channels else None
                )
                if fig:
                    _save(fig, f"metrics/{metric}.{fig_format}")

            if titration_ready:
                for title, (metric, ylabel) in display_metric_cfg.items():
                    offset_export_plateaus_to_buffer = (
                        len(combined_titration_channels) > 1
                        and metric in {
                            "peak_current_selected",
                            "peak_current_raw",
                            "wavelet_energy",
                        }
                    )
                    export_plateau_directions = (
                        langmuir_response_directions_by_metric.get(metric)
                        if metric in {"peak_current_selected", "peak_current_raw"}
                        else langmuir_response_directions_by_metric.get(
                            "peak_current_selected"
                        )
                    )
                    export_plateau_ylabel = {
                        "peak_current_selected": "Change in Peak Height (uA)",
                        "peak_current_raw": "Change in Peak Height (uA)",
                        "wavelet_energy": "Wavelet Energy Change from Buffer (a.u.)",
                    }.get(metric, ylabel)
                    fig = (
                        plot_titration_plateaus(
                            titration_results,
                            metric=metric,
                            channels=combined_titration_channels,
                            title=f"{title} | plateau fit",
                            ylabel=(
                                export_plateau_ylabel
                                if offset_export_plateaus_to_buffer else ylabel
                            ),
                            vlines=combined_titration_active_vlines,
                            vlines_by_channel=titration_vlines_by_channel,
                            scan_windows=None,
                            scan_range=titration_scan_range,
                            edge_trim_fraction=titration_edge_trim_fraction,
                            baseline_mode=titration_baseline_mode,
                            included_step_labels=titration_included_step_labels,
                            remove_extreme_outliers=remove_extreme_titration_outliers,
                            response_directions=export_plateau_directions,
                            response_baselines=response_baselines_by_metric.get(metric),
                            offset_to_response_baseline=(
                                offset_export_plateaus_to_buffer
                            ),
                            channel_colors=consistent_channel_colors,
                        )
                        if combined_titration_channels else None
                    )
                    if fig:
                        _save(fig, f"titration/plateaus/{metric}.{fig_format}")

                    if fit_titration_langmuir and supports_langmuir(metric):
                        fig = (
                            plot_titration_langmuir(
                                titration_results,
                                metric=metric,
                                channels=combined_titration_channels,
                                title=f"{title} | Langmuir Fit",
                                ylabel=(
                                    "Peak Height (uA)"
                                    if metric == "peak_current_selected"
                                    else ylabel
                                ),
                                vlines=combined_titration_active_vlines,
                                vlines_by_channel=titration_vlines_by_channel,
                                scan_windows=None,
                                scan_range=titration_scan_range,
                                edge_trim_fraction=titration_edge_trim_fraction,
                                fit_langmuir=True,
                                concentration_unit=titration_concentration_unit,
                                baseline_mode=titration_baseline_mode,
                                included_step_labels=titration_included_step_labels,
                                remove_extreme_outliers=remove_extreme_titration_outliers,
                                show_uloq=show_titration_uloq,
                                show_lod=show_titration_lod,
                                show_fit_details=show_titration_fit_details,
                                response_directions=(
                                    consistent_response_directions
                                ),
                                channel_colors=consistent_channel_colors,
                            )
                            if combined_titration_channels else None
                        )
                        if fig:
                            _save(fig, f"titration/langmuir/{metric}.{fig_format}")

            for title, (dk, ylabel, _caption) in drift_export_cfg.items():
                fig = plot_drift_vs_scan(plot_results, drift_metric=dk, channels=plot_channels_display,
                                         title=title, ylabel=ylabel,
                                         vlines=plot_active_vlines, scan_range=plot_display_scan_range, xlabel=plot_x_axis_label,
                                         channel_colors=consistent_channel_colors)
                if fig:
                    _save(fig, f"drift/{dk}.{fig_format}")

            for ch in plot_channels_display:
                ch_res = plot_ok_results_by_channel.get(ch, [])
                if analysis_mode == "CV":
                    for yk, lbl in (
                        ("smoothed_current", "smoothed"),
                        ("raw_current", "raw"),
                        ("detrended_current", "detrended"),
                    ):
                        fig = plot_cv_overlaid_cycles(
                            ch_res,
                            y_key=yk,
                            title=f"Ch{ch}  {lbl}",
                            show_peak_markers=True,
                            show_zero_baseline=(yk == "detrended_current"),
                            show_peak_reference_vlines=True,
                        )
                        if fig:
                            _save(fig, f"overlays/ch{ch}_{lbl}.{fig_format}")
                else:
                    export_overlay_keys = [
                        ("corrected_current", "corrected", False, "Current (uA)"),
                        ("smoothed_corrected_current", "smoothed_corrected", False, "Current (uA)"),
                        (
                            "smoothed_corrected_current",
                            "normalized_smoothed_corrected",
                            True,
                            "Normalized current (peak = 1)",
                        ),
                        ("raw_current", "raw", False, "Current (uA)"),
                    ]
                    if has_wavelet_denoised_trace:
                        export_overlay_keys.append(("wavelet_denoised_current", "wavelet_denoised", False, "Current (uA)"))
                    for yk, lbl, normalize_to_peak, ylabel in export_overlay_keys:
                        fig = plot_overlaid_traces(ch_res, y_key=yk,
                                                   title=f"Ch{ch}  {lbl}",
                                                   ylabel=ylabel,
                                                   show_anchors=(yk == "corrected_current"),
                                                   show_zero_baseline=(yk in ("corrected_current", "smoothed_corrected_current")),
                                                   normalize_to_peak=normalize_to_peak)
                        if fig:
                            _save(fig, f"overlays/ch{ch}_{lbl}.{fig_format}")

        zip_buf.seek(0)
        st.download_button("  Download figures.zip", data=zip_buf,
                           file_name="cv_figures.zip" if analysis_mode == "CV" else "swv_figures.zip", mime="application/zip",
                           use_container_width=True)
