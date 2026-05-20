"""
SWV Batch Analysis  Streamlit UI
Run with:  python -m streamlit run app.py
"""

import io
import math
import os
import subprocess
import sys
import zipfile
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from core import (
    build_titration_step_table,
    plot_drift_vs_scan,
    plot_failed_traces,
    plot_metric_vs_scan,
    plot_overlaid_traces,
    plot_single_trace,
    plot_titration_langmuir,
    plot_titration_plateaus,
    run_batch,
)


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
        "p=filedialog.askdirectory(title='Select SWV data folder')\n"
        "root.destroy()\n"
        "print(p or '')\n"
    )
    return subprocess.check_output([sys.executable, "-c", code], text=True).strip()

# 
# Page config
# 
st.set_page_config(
    page_title="SWV Analysis",
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
# Cached analysis  only re-runs when params change
# 
@st.cache_data(show_spinner=False)
def cached_run_batch(
    folders,          # tuple so it's hashable
    crop_range,
    smooth_window,
    smooth_polyorder,
    minima_search_window_V,
    use_prominent_minima,
    use_double_correction,
    min_peak_height_uA,
    min_start_voltage,
    scan_range,
    compute_skew,
    compute_wavelet_energy,
):
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
        scan_range=scan_range,
        compute_skew=compute_skew,
        compute_wavelet_energy=compute_wavelet_energy,
    )


def collect_titration_rows(
    all_results,
    metric_cfg,
    channels,
    vlines,
    scan_range,
    edge_trim_fraction,
):
    rows = []
    for label, (metric_key, ylabel) in metric_cfg.items():
        metric_rows = build_titration_step_table(
            all_results,
            metric=metric_key,
            vlines=vlines,
            channels=channels,
            scan_range=scan_range,
            edge_trim_fraction=edge_trim_fraction,
        )
        for row in metric_rows:
            rows.append({
                "metric_label": label,
                "metric_key": metric_key,
                "metric_ylabel": ylabel,
                **row,
            })
    return rows


def _chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size], (i // size) + 1


def _make_grid(n_items: int, max_cols: int = 3):
    cols = min(max_cols, max(n_items, 1))
    rows = int(math.ceil(max(n_items, 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows), squeeze=False)
    return fig, axes.flatten()


def _filtered_results_for_channel(results, ch, scan_range=None):
    ch_res = [r for r in results if r.get("status") == "OK" and r.get("channel") == ch]
    if scan_range:
        ch_res = [r for r in ch_res if scan_range[0] <= r["scan_number"] <= scan_range[1]]
    return sorted(ch_res, key=lambda r: r["scan_number"])


def _add_vlines_to_axis(ax, vlines, scan_range=None, y_frac: float = 0.85):
    if not vlines:
        return
    for x, label in vlines:
        if scan_range and not (scan_range[0] <= x <= scan_range[1]):
            continue
        ax.axvline(x=x, color="gray", linestyle="--", alpha=0.45, lw=0.8)
        ax.text(
            x, y_frac, label,
            rotation=90, va="center", ha="center",
            transform=ax.get_xaxis_transform(),
            fontsize=6, color="gray",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=1.0),
        )


def _plot_overlay_grid_page(
    results,
    channels,
    y_key,
    title,
    ylabel="Current (uA)",
    colormap_name="plasma",
    scan_range=None,
):
    fig, axes = _make_grid(len(channels), max_cols=3)
    for ax, ch in zip(axes, channels):
        ch_res = _filtered_results_for_channel(results, ch, scan_range=scan_range)
        usable = [r for r in ch_res if r.get(y_key) is not None and r.get("voltage") is not None]
        cmap = plt.get_cmap(colormap_name, max(len(usable), 2))
        for i, r in enumerate(usable):
            denom = max(len(usable) - 1, 1)
            ax.plot(r["voltage"], r[y_key], color=cmap(i / denom), lw=0.65, alpha=0.85)
        if y_key in ("corrected_current", "smoothed_corrected_current"):
            ax.axhline(0, color="gray", lw=0.7, linestyle="--", alpha=0.6)
        ax.set_title(f"Ch{ch} ({len(usable)} traces)", fontsize=10)
        ax.set_xlabel("Voltage (V)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(False)
    for ax in axes[len(channels):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def _plot_combined_metric_page(
    results,
    metric_items,
    channels,
    title,
    vlines=None,
    scan_range=None,
    drift=False,
):
    fig, axes = _make_grid(len(metric_items), max_cols=2)
    all_ch = sorted({r["channel"] for r in results})
    channels = [ch for ch in channels if ch in all_ch]
    cmap = plt.get_cmap("tab10")
    colors = {ch: cmap(i % 10) for i, ch in enumerate(all_ch)}

    for ax, (label, metric, ylabel) in zip(axes, metric_items):
        if drift:
            ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.55)
        plotted = False
        for ch in channels:
            ch_res = _filtered_results_for_channel(results, ch, scan_range=scan_range)
            x = [r["scan_number"] for r in ch_res]
            y = [r.get(metric, np.nan) for r in ch_res]
            if not x or all(np.isnan(v) for v in y):
                continue
            ax.plot(x, y, marker="o", ms=2.4, lw=1.1, color=colors[ch], alpha=0.9, label=f"Ch{ch}")
            plotted = True
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Scan number", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        _add_vlines_to_axis(ax, vlines, scan_range=scan_range)
        if scan_range:
            ax.set_xlim(scan_range)
        if plotted:
            ax.legend(title="Channel", loc="best", fontsize=6, title_fontsize=7)
    for ax in axes[len(metric_items):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _plot_individual_metric_pages(
    pdf,
    results,
    metric_items,
    channels,
    title_prefix,
    vlines=None,
    scan_range=None,
    drift=False,
    channels_per_page=12,
):
    for label, metric, ylabel in metric_items:
        pages = list(_chunked(channels, channels_per_page))
        for page_channels, page_num in pages:
            fig, axes = _make_grid(len(page_channels), max_cols=3)
            for ax, ch in zip(axes, page_channels):
                ch_res = _filtered_results_for_channel(results, ch, scan_range=scan_range)
                x = [r["scan_number"] for r in ch_res]
                y = [r.get(metric, np.nan) for r in ch_res]
                if drift:
                    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.55)
                if x and not all(np.isnan(v) for v in y):
                    ax.plot(x, y, marker="o", ms=2.5, lw=1.2, color="tab:blue", alpha=0.9)
                ax.set_title(f"Ch{ch}", fontsize=10)
                ax.set_xlabel("Scan number", fontsize=8)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.tick_params(labelsize=7)
                _add_vlines_to_axis(ax, vlines, scan_range=scan_range)
                if scan_range:
                    ax.set_xlim(scan_range)
            for ax in axes[len(page_channels):]:
                ax.set_visible(False)
            suffix = f" page {page_num}" if len(pages) > 1 else ""
            fig.suptitle(f"{title_prefix}: {label}{suffix}", fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)


# 
# Session state
# 
for k, v in dict(results=None, last_results=None, folders=[], run_count=0).items():
    if k not in st.session_state:
        st.session_state[k] = v


# 
# Sidebar
# 
with st.sidebar:
    st.title("⚡ SWV Analysis")
    st.divider()

    #  Folders 
    st.subheader(" Data Folders")

    c1, c2 = st.columns(2)

    if c1.button("  Browse (Windows)", use_container_width=True, disabled=not sys.platform.startswith("win")):
        try:
            picked = _pick_folder_windows()
            if picked and picked not in st.session_state.folders:
                st.session_state.folders.append(picked)
        except subprocess.CalledProcessError as e:
            st.error(f"Windows folder picker failed: {e}")
        except Exception as e:
            st.error(f"Windows folder picker failed: {e}")

    if c2.button("  Browse (macOS)", use_container_width=True, disabled=sys.platform != "darwin"):
        try:
            # Use Finder's native picker via AppleScript (Tk dialogs can crash Streamlit on macOS).
            script = 'POSIX path of (choose folder with prompt "Select SWV data folder")'
            picked = subprocess.check_output(["osascript", "-e", script], text=True).strip()
            if picked and picked not in st.session_state.folders:
                st.session_state.folders.append(picked)
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
        help="You can also paste paths directly here.",
    )
    edited = [f.strip() for f in raw_folders.splitlines() if f.strip()]
    st.session_state.folders = edited
    folders = edited

    if folders:
        if st.button("  Clear all folders", use_container_width=True):
            st.session_state.folders = []
            st.rerun()

    folder_errors = [f for f in folders if not os.path.isdir(f)]
    if folder_errors:
        for fe in folder_errors:
            st.error(f"Not found: `{fe}`")

    st.divider()

    #  Crop & voltage 
    st.subheader(" Voltage / Crop")
    col1, col2 = st.columns(2)
    crop_min = col1.number_input("Crop min (V)", value=-0.61, step=0.01, format="%.3f")
    crop_max = col2.number_input("Crop max (V)", value=-0.30, step=0.01, format="%.3f")
    min_start_voltage = st.number_input(
        "Min start voltage (V)", value=-0.70, step=0.01, format="%.3f",
        help="Skip files whose first voltage point is below this value.",
    )

    st.divider()

    #  Smoothing 
    st.subheader(" Smoothing")
    smooth_window    = st.slider("Savitzky-Golay window", min_value=3, max_value=31, value=15, step=2)
    smooth_polyorder = st.slider("Polynomial order", min_value=1, max_value=5, value=2)

    st.divider()

    #  Peak / baseline 
    st.subheader(" Peak / Baseline")
    minima_search_window = st.number_input(
        "Minima search window (V)", value=0.30, step=0.01, format="%.3f",
        help="Voltage window either side of peak when searching for bracketing minima.",
    )
    use_prominent_minima = st.checkbox(
        "Use prominent local minima for bracketing",
        value=False,
        help="Experimental comparison mode: uses peaks of the inverted smoothed signal and takes the most prominent local minimum on each side of the detected peak.",
    )
    use_double_correction = st.checkbox(
        "Double baseline correction",
        value=False,
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
    min_peak_height = None
    if use_peak_cutoff:
        min_peak_height = st.number_input("Min peak height (uA)", value=0.001, step=0.001, format="%.3f")

    st.divider()

    st.subheader("Performance")
    compute_skew = st.checkbox("Compute skew metric", value=True)
    compute_wavelet_energy = st.checkbox("Compute wavelet energy", value=True)
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

    st.divider()

    #  Scan range 
    st.subheader(" Scan Range")
    use_scan_range = st.checkbox("Limit scan range", value=False)
    scan_range: Optional[Tuple[int, int]] = None
    if use_scan_range:
        sr_c1, sr_c2 = st.columns(2)
        scan_range = (int(sr_c1.number_input("From", value=0, min_value=0)),
                      int(sr_c2.number_input("To",   value=260, min_value=0)))

    st.divider()

    #  Vlines 
    st.subheader(" Vertical Lines")
    vlines_input = st.text_area(
        "scan,label  one per line",
        value="\n".join([
            "10,LSV 7",  "20,LSV 3",  "30,LSV 9",  "40,LSV 2",  "50,LSV 10",
            "60,LSV 5",  "70,LSV 1",  "80,LSV 4",  "90,LSV 8",  "100,LSV 6",
            "120,Buffer added", "140,DS added", "160,Buffer added",
            "170,LSV 7", "180,LSV 3", "190,LSV 9", "200,LSV 2", "210,LSV 10",
            "220,LSV 5", "230,LSV 1", "240,LSV 4", "250,LSV 8", "260,LSV 6",
        ]),
        height=180,
    )
    vlines: List[Tuple[float, str]] = []
    for line in vlines_input.splitlines():
        parts = line.strip().split(",", 1)
        if len(parts) == 2:
            try:
                vlines.append((float(parts[0].strip()), parts[1].strip()))
            except ValueError:
                pass

    enable_titration_analysis = st.checkbox(
        "Treat vline intervals as titration steps",
        value=False,
        help="Each interval between consecutive vertical lines becomes one titration step.",
    )
    titration_edge_trim_fraction = 0.15
    fit_titration_langmuir = False
    if enable_titration_analysis:
        titration_edge_trim_fraction = st.slider(
            "Plateau edge trim fraction",
            min_value=0.0,
            max_value=0.4,
            value=0.15,
            step=0.05,
            help="Uses only the middle portion of each step when estimating the plateau median.",
        )
        fit_titration_langmuir = st.checkbox(
            "Fit Langmuir-style curve to step plateaus",
            value=True,
            help="Uses titration step index as a proxy x-axis and fits a simple Langmuir isotherm.",
        )

    st.divider()

    #  Failed traces 
    st.subheader(" Failed Traces")
    max_failed = st.number_input("Max failed traces to plot", value=40, min_value=1)

    st.divider()

    run_clicked = st.button(
        "  Run Analysis",
        type="primary",
        disabled=not folders or bool(folder_errors),
        use_container_width=True,
    )


# 
# Run analysis
# 
if run_clicked and folders and not folder_errors:
    st.session_state.folders = folders
    try:
        if use_cache:
            with st.spinner("Running analysis (first run may take a moment, cached runs are instant)"):
                results = cached_run_batch(
                    folders=tuple(folders),
                    crop_range=(crop_min, crop_max),
                    smooth_window=smooth_window,
                    smooth_polyorder=smooth_polyorder,
                    minima_search_window_V=minima_search_window,
                    use_prominent_minima=use_prominent_minima,
                    use_double_correction=use_double_correction,
                    min_peak_height_uA=min_peak_height,
                    min_start_voltage=min_start_voltage,
                    scan_range=scan_range,
                    compute_skew=compute_skew,
                    compute_wavelet_energy=compute_wavelet_energy,
                )
        else:
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def _progress(done, total, name):
                pct = int((done / max(total, 1)) * 100)
                progress_bar.progress(pct)
                progress_text.caption(f"Analyzing {done}/{total}: {name}")

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
                scan_range=scan_range,
                compute_skew=compute_skew,
                compute_wavelet_energy=compute_wavelet_energy,
                progress_callback=_progress,
            )
            progress_bar.progress(100)
            progress_text.caption("Analysis complete.")

        st.session_state.results = results
        if results:
            st.session_state.last_results = results
        st.session_state.run_count += 1
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()


# 
# Guard  nothing run yet
# 
results = st.session_state.get("results")
if results is None:
    if st.session_state.get("last_results") is not None:
        st.warning("Showing last successful results (current run returned nothing).")
        results = st.session_state.last_results
    else:
        st.info(" Configure parameters in the sidebar, then click **Run Analysis**.")
        st.stop()
if len(results) == 0:
    if st.session_state.get("last_results") is not None:
        st.warning("No results returned. Showing last successful results.")
        results = st.session_state.last_results
    else:
        st.warning("No results returned. Check folder paths and file naming pattern.")
        st.stop()

ok_results     = [r for r in results if r.get("status") == "OK"]
failed_results = [r for r in results if r.get("status") == "FAILED"]
all_channels   = sorted({r["channel"] for r in results})
channels_display = channels_to_plot if channels_to_plot else all_channels
ch_options = ["All channels"] + [f"Ch{ch}" for ch in channels_display]

#  Summary banner 
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total files", len(results))
c2.metric(" Successful", len(ok_results))
c3.metric(" Failed", len(failed_results))
c4.metric("Channels found", len(all_channels))

st.divider()

metric_cfg = {
    "Peak current (corrected)": ("peak_current",     "Corrected Peak Height (uA)"),
    "Peak current (raw)":       ("peak_current_raw", "Raw Current at Peak (uA)"),
    "Skew":                     ("skew",             "Skew (corrected trace)"),
    "Peak offset (normalized)": ("peak_offset_norm", "Peak offset from bracket center (normalized)"),
    "Wavelet energy":           ("wavelet_energy",   "Wavelet Energy (a.u.)"),
}
if not compute_skew:
    metric_cfg.pop("Skew", None)
    metric_cfg.pop("Peak offset (normalized)", None)
if not compute_wavelet_energy:
    metric_cfg.pop("Wavelet energy", None)

titration_ready = enable_titration_analysis and len(vlines) >= 2

# 
# Tabs
# 
view = st.radio("View", ["Overlays", "Metrics", "Drift", "Failures", "Data Table", "Export"], horizontal=True)



# 
# TAB: Overlays
# 
if view == "Overlays":
    st.subheader("Overlaid traces per channel")

    ov_c1, ov_c2, ov_c3, ov_c4, ov_c5 = st.columns([2, 2, 1, 1, 1])
    trace_type   = ov_c1.radio("Trace type", ["Corrected", "Smoothed Corrected", "Raw", "Smoothed"],
                                horizontal=True, key="overlay_type")
    cmap_name    = ov_c2.selectbox("Colour map",
                                   ["plasma", "viridis", "inferno", "magma", "cividis", "turbo"],
                                   key="overlay_cmap")
    show_anchors = ov_c3.checkbox("Show correction anchors", value=True,
                                  help="Dots mark the two bracketing-minima points used for baseline correction.")
    show_peak_markers = ov_c4.checkbox("Show peak points", value=False,
                                       help="Marks the detected peak on each displayed trace.")
    show_baseline = ov_c5.checkbox("Show 0 baseline", value=True,
                                   help="Draws a dashed horizontal zero-current reference line.")

    key_map = {
        "Corrected": "corrected_current",
        "Smoothed Corrected": "smoothed_corrected_current",
        "Raw": "raw_current",
        "Smoothed": "smoothed_current",
    }
    y_key = key_map[trace_type]

    for ch in channels_display:
        ch_res = [r for r in ok_results if r["channel"] == ch]
        if scan_range:
            ch_res = [r for r in ch_res if scan_range[0] <= r["scan_number"] <= scan_range[1]]
        if not ch_res:
            continue
        with st.expander(f"Channel {ch}  ({len(ch_res)} traces)", expanded=len(channels_display) <= 4):
            fig = plot_overlaid_traces(
                ch_res, y_key=y_key,
                title=f"{trace_type}  Ch{ch}",
                ylabel="Current (uA)",
                colormap_name=cmap_name,
                show_anchors=show_anchors,
                show_peak_markers=show_peak_markers,
                show_zero_baseline=(show_baseline and y_key in ("corrected_current", "smoothed_corrected_current")),
            )
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No plottable traces for this channel.")


# 
# TAB: Metrics
# 
if view == "Metrics":
    st.subheader("Metrics vs scan number")

    m_c1, m_c2 = st.columns([3, 1])
    selected_metrics = m_c1.multiselect(
        "Metrics to display",
        options=list(metric_cfg.keys()),
        default=list(metric_cfg.keys()),
    )
    ch_options   = ["All channels"] + [f"Ch{ch}" for ch in channels_display]
    ch_selection = m_c2.selectbox("Highlight channel", ch_options, key="metric_ch_sel",
                                   help="Selecting one channel dims the others.")
    highlight_ch = None
    if ch_selection != "All channels":
        highlight_ch = int(ch_selection.replace("Ch", ""))

    view_mode = st.radio("View mode", ["Combined", "Individual channels"],
                          horizontal=True, key="metric_view_mode")

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

    for label in selected_metrics:
        metric, ylabel = metric_cfg[label]
        st.markdown(f"**{label}**")

        if view_mode == "Combined":
            fig = plot_metric_vs_scan(
                results, metric=metric, channels=channels_display,
                title=label, ylabel=ylabel, vlines=vlines,
                scan_range=scan_range, highlight_channel=highlight_ch,
            )
            if fig:
                st.pyplot(fig)
                plt.close(fig)
        else:
            cols = st.columns(min(len(channels_display), 3))
            for i, ch in enumerate(channels_display):
                fig = plot_metric_vs_scan(
                    results, metric=metric, channels=[ch],
                    title=f"Ch{ch}", ylabel=ylabel, vlines=vlines,
                    scan_range=scan_range, figsize=(5, 3),
                )
                if fig:
                    with cols[i % min(len(channels_display), 3)]:
                        st.pyplot(fig)
                    plt.close(fig)

        if titration_ready:
            st.caption("Titration plateaus")
            if view_mode == "Combined":
                fig = plot_titration_plateaus(
                    results,
                    metric=metric,
                    channels=channels_display,
                    title=f"{label} | plateau fit",
                    ylabel=ylabel,
                    vlines=vlines,
                    scan_range=scan_range,
                    edge_trim_fraction=titration_edge_trim_fraction,
                    highlight_channel=highlight_ch,
                )
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                cols = st.columns(min(len(channels_display), 3))
                for i, ch in enumerate(channels_display):
                    fig = plot_titration_plateaus(
                        results,
                        metric=metric,
                        channels=[ch],
                        title=f"Ch{ch} | plateau fit",
                        ylabel=ylabel,
                        vlines=vlines,
                        scan_range=scan_range,
                        edge_trim_fraction=titration_edge_trim_fraction,
                        figsize=(5, 3),
                    )
                    if fig:
                        with cols[i % min(len(channels_display), 3)]:
                            st.pyplot(fig)
                        plt.close(fig)

            if fit_titration_langmuir:
                st.caption("Langmuir-style fit of plateau midpoints")
                if view_mode == "Combined":
                    fig = plot_titration_langmuir(
                        results,
                        metric=metric,
                        channels=channels_display,
                        title=f"{label} | Langmuir-style fit",
                        ylabel=ylabel,
                        vlines=vlines,
                        scan_range=scan_range,
                        edge_trim_fraction=titration_edge_trim_fraction,
                        highlight_channel=highlight_ch,
                        fit_langmuir=True,
                    )
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    cols = st.columns(min(len(channels_display), 3))
                    for i, ch in enumerate(channels_display):
                        fig = plot_titration_langmuir(
                            results,
                            metric=metric,
                            channels=[ch],
                            title=f"Ch{ch} | Langmuir-style fit",
                            ylabel=ylabel,
                            vlines=vlines,
                            scan_range=scan_range,
                            edge_trim_fraction=titration_edge_trim_fraction,
                            figsize=(5, 3),
                            fit_langmuir=True,
                        )
                        if fig:
                            with cols[i % min(len(channels_display), 3)]:
                                st.pyplot(fig)
                            plt.close(fig)

        st.divider()


# 
# TAB: Drift
# 
if view == "Drift":
    st.subheader("Drift metrics (relative to each channel's first scan)")
    st.markdown(
        "Both metrics are computed **per channel**  the first valid scan for each channel "
        "is used as the reference (zero line). This lets you compare channels even if they "
        "started at different absolute values."
    )

    dr_c1, dr_c2 = st.columns([3, 1])
    drift_options = {
        "Peak voltage drift (V)": ("peak_voltage_drift", "Peak voltage (V)",
                                   "Shift in peak position  indicates a change in the redox potential."),
        "Skew drift":             ("skew_drift",         "Skew",
                                   "Change in corrected-trace asymmetry  sensitive to baseline shape changes."),
        "Peak offset (normalized) drift": ("peak_offset_norm_drift", "Peak offset (normalized)",
                                   "Shift in peak position relative to bracket center (normalized)."),
    }
    if not compute_skew:
        drift_options.pop("Skew drift", None)
        drift_options.pop("Peak offset (normalized) drift", None)

    selected_drift = dr_c1.multiselect(
        "Drift metrics to display",
        options=list(drift_options.keys()),
        default=list(drift_options.keys()),
    )
    dr_ch_sel = dr_c2.selectbox("Highlight channel", ch_options, key="drift_ch_sel")
    drift_highlight = None
    if dr_ch_sel != "All channels":
        drift_highlight = int(dr_ch_sel.replace("Ch", ""))

    drift_view_mode = st.radio("View mode", ["Combined", "Individual channels"],
                               horizontal=True, key="drift_view_mode")

    for label in selected_drift:
        drift_key, ylabel, caption = drift_options[label]
        st.markdown(f"**{label}**")
        st.caption(f"_{caption}_")

        if drift_view_mode == "Combined":
            fig = plot_drift_vs_scan(
                results, drift_metric=drift_key, channels=channels_display,
                title=label, ylabel=ylabel, vlines=vlines,
                scan_range=scan_range, highlight_channel=drift_highlight,
            )
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning(f"No data available for {label}.")
        else:
            cols = st.columns(min(len(channels_display), 3))
            for i, ch in enumerate(channels_display):
                fig = plot_drift_vs_scan(
                    results, drift_metric=drift_key, channels=[ch],
                    title=f"Ch{ch}", ylabel=ylabel, vlines=vlines,
                    scan_range=scan_range, figsize=(5, 3),
                )
                if fig:
                    with cols[i % min(len(channels_display), 3)]:
                        st.pyplot(fig)
                    plt.close(fig)

        st.divider()


# 
# TAB: Failures
# 
if view == "Failures":
    st.subheader(f"Failed traces  ({len(failed_results)} total)")

    if not failed_results:
        st.success("No failures ")
    else:
        fail_df = pd.DataFrame([
            {"Channel": r["channel"], "Scan #": r["scan_number"],
             "File": r.get("file_name", ""), "Error": r.get("error", "")}
            for r in failed_results
        ])
        st.dataframe(fail_df, use_container_width=True, height=200)
        st.divider()

        for ch in channels_display:
            ch_failed = [r for r in failed_results if r["channel"] == ch]
            if not ch_failed:
                continue
            to_plot = ch_failed[:int(max_failed)]
            with st.expander(f"Ch{ch}  {len(ch_failed)} failures", expanded=False):
                for yk, yl in (
                    ("raw_current",       "Raw Current (uA)"),
                    ("smoothed_current",  "Smoothed Current (uA)"),
                    ("corrected_current", "Corrected Current (uA)"),
                    ("smoothed_corrected_current", "Smoothed Corrected Current (uA)"),
                ):
                    fig = plot_failed_traces(
                        to_plot, y_key=yk, ylabel=yl,
                        title=f"Ch{ch}  {yl}",
                        show_peak_markers=(yk != "raw_current"),
                        show_zero_baseline=(yk in ("corrected_current", "smoothed_corrected_current")),
                        show_local_baselines=(yk == "smoothed_current"),
                        show_minima_candidates=(yk == "smoothed_current"),
                    )
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)

        st.divider()
        st.markdown("####  Single-trace inspector")
        fail_options_map = {
            f"Ch{r['channel']}  Scan {r['scan_number']}  {r.get('file_name','')}": r
            for r in failed_results
        }
        chosen_label = st.selectbox("Pick a failed trace", list(fail_options_map.keys()))
        if chosen_label:
            chosen = fail_options_map[chosen_label]
            st.caption(f"Error: {chosen.get('error', '')}")
            if chosen.get("voltage") is not None:
                fig = plot_single_trace(chosen)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("No trace data available for this file.")


# 
# TAB: Data Table
# 
if view == "Data Table":
    st.subheader("Results table")

    scalar_keys = [
        "channel", "scan_number", "file_name", "status",
        "peak_voltage", "peak_current", "peak_current_raw",
        "skew", "peak_offset_norm", "wavelet_energy",
        "peak_voltage_drift", "skew_drift", "peak_offset_norm_drift", "error",
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

    st.dataframe(filtered_df, use_container_width=True, height=400)
    st.caption(f"{mask.sum()} rows shown")

    if enable_titration_analysis:
        st.divider()
        st.markdown("#### Titration step table")
        if not titration_ready:
            st.info("Add at least two vertical lines inside the active scan range to build titration steps.")
        else:
            default_titration_metrics = (
                ["Peak current (corrected)"]
                if "Peak current (corrected)" in metric_cfg
                else list(metric_cfg.keys())[:1]
            )
            titration_metric_labels = st.multiselect(
                "Titration metrics to tabulate",
                options=list(metric_cfg.keys()),
                default=default_titration_metrics,
                key="table_titration_metrics",
            )
            titration_rows = []
            for label in titration_metric_labels:
                metric_key, ylabel = metric_cfg[label]
                for row in build_titration_step_table(
                    filtered_results,
                    metric=metric_key,
                    vlines=vlines,
                    channels=ch_filter,
                    scan_range=scan_range,
                    edge_trim_fraction=titration_edge_trim_fraction,
                ):
                    titration_rows.append({
                        "Metric": label,
                        "Channel": row["channel"],
                        "Step #": row["step_index"],
                        "Left marker": row["left_vline_label"],
                        "Right marker": row["right_vline_label"],
                        "Step start": row["step_start_scan"],
                        "Step end": row["step_end_scan"],
                        "Midpoint": row["midpoint_scan"],
                        "Plateau value": row["plateau_value"],
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

        if chosen.get("error"):
            st.caption(f"Error: {chosen.get('error')}")

        if chosen.get("voltage") is not None:
            fig = plot_single_trace(chosen)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("No trace data available for this measurement.")


# 
# TAB: Export
# 
if view == "Export":
    st.subheader("Export results")

    st.markdown("####  Results CSV")
    export_keys = [
        "channel", "scan_number", "timestamp", "file_name", "status",
        "peak_voltage", "peak_current", "peak_current_raw",
        "skew", "peak_offset_norm", "wavelet_energy",
        "peak_voltage_drift", "skew_drift", "peak_offset_norm_drift", "error",
    ]
    csv_bytes = pd.DataFrame([{k: r.get(k) for k in export_keys} for r in results])\
                  .to_csv(index=False).encode()
    st.download_button("  Download results.csv", data=csv_bytes,
                       file_name="swv_results.csv", mime="text/csv",
                       use_container_width=True)

    if enable_titration_analysis:
        st.markdown("####  Titration step CSV")
        if not titration_ready:
            st.info("Add at least two vertical lines inside the active scan range to export titration steps.")
        else:
            titration_export_rows = collect_titration_rows(
                results,
                metric_cfg=metric_cfg,
                channels=channels_display,
                vlines=vlines,
                scan_range=scan_range,
                edge_trim_fraction=titration_edge_trim_fraction,
            )
            if titration_export_rows:
                titration_csv = pd.DataFrame(titration_export_rows).to_csv(index=False).encode()
                st.download_button(
                    "  Download titration_steps.csv",
                    data=titration_csv,
                    file_name="swv_titration_steps.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No titration step rows are available for export with the current settings.")

    st.divider()

    st.markdown("####  Report PDF")
    pdf_c1, pdf_c2 = st.columns([2, 1])
    pdf_metric_layout = pdf_c1.radio(
        "Metrics and drift layout",
        ["Combined summary pages", "Individual channel grids"],
        horizontal=True,
        help=(
            "Combined uses one page for all metrics and one page for all drift plots. "
            "Individual channel grids creates per-metric/per-drift channel-grid pages."
        ),
    )
    pdf_cmap = pdf_c2.selectbox(
        "Overlay colour map",
        ["plasma", "viridis", "inferno", "magma", "cividis", "turbo"],
        key="report_pdf_cmap",
    )

    drift_export_items = [
        ("Peak voltage drift (V)", "peak_voltage_drift", "Peak voltage (V)"),
    ]
    if compute_skew:
        drift_export_items.extend([
            ("Skew drift", "skew_drift", "Skew"),
            ("Peak offset (normalized) drift", "peak_offset_norm_drift", "Peak offset (normalized)"),
        ])
    metric_export_items = [
        (label, metric, ylabel)
        for label, (metric, ylabel) in metric_cfg.items()
    ]

    if st.button("  Build report PDF", use_container_width=True):
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            for page_channels, page_num in _chunked(channels_display, 12):
                page_suffix = f" page {page_num}" if len(channels_display) > 12 else ""
                fig = _plot_overlay_grid_page(
                    results,
                    channels=page_channels,
                    y_key="raw_current",
                    title=f"Raw overlays by channel{page_suffix}",
                    colormap_name=pdf_cmap,
                    scan_range=scan_range,
                )
                pdf.savefig(fig)
                plt.close(fig)

            for page_channels, page_num in _chunked(channels_display, 12):
                page_suffix = f" page {page_num}" if len(channels_display) > 12 else ""
                fig = _plot_overlay_grid_page(
                    results,
                    channels=page_channels,
                    y_key="smoothed_corrected_current",
                    title=f"Smoothed fitted overlays by channel{page_suffix}",
                    colormap_name=pdf_cmap,
                    scan_range=scan_range,
                )
                pdf.savefig(fig)
                plt.close(fig)

            if pdf_metric_layout == "Combined summary pages":
                fig = _plot_combined_metric_page(
                    results,
                    metric_items=metric_export_items,
                    channels=channels_display,
                    title="Metrics by scan",
                    vlines=vlines,
                    scan_range=scan_range,
                )
                pdf.savefig(fig)
                plt.close(fig)

                fig = _plot_combined_metric_page(
                    results,
                    metric_items=drift_export_items,
                    channels=channels_display,
                    title="Drift by scan",
                    vlines=vlines,
                    scan_range=scan_range,
                    drift=True,
                )
                pdf.savefig(fig)
                plt.close(fig)
            else:
                _plot_individual_metric_pages(
                    pdf,
                    results,
                    metric_items=metric_export_items,
                    channels=channels_display,
                    title_prefix="Metrics by channel",
                    vlines=vlines,
                    scan_range=scan_range,
                )
                _plot_individual_metric_pages(
                    pdf,
                    results,
                    metric_items=drift_export_items,
                    channels=channels_display,
                    title_prefix="Drift by channel",
                    vlines=vlines,
                    scan_range=scan_range,
                    drift=True,
                )

        pdf_buf.seek(0)
        st.download_button(
            "  Download report.pdf",
            data=pdf_buf,
            file_name="swv_report.pdf",
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

            for title, (metric, ylabel) in metric_cfg.items():
                fig = plot_metric_vs_scan(results, metric=metric, channels=channels_display,
                                          title=title, ylabel=ylabel,
                                          vlines=vlines, scan_range=scan_range)
                if fig:
                    _save(fig, f"metrics/{metric}.{fig_format}")

            if titration_ready:
                for title, (metric, ylabel) in metric_cfg.items():
                    fig = plot_titration_plateaus(
                        results,
                        metric=metric,
                        channels=channels_display,
                        title=f"{title} | plateau fit",
                        ylabel=ylabel,
                        vlines=vlines,
                        scan_range=scan_range,
                        edge_trim_fraction=titration_edge_trim_fraction,
                    )
                    if fig:
                        _save(fig, f"titration/plateaus/{metric}.{fig_format}")

                    if fit_titration_langmuir:
                        fig = plot_titration_langmuir(
                            results,
                            metric=metric,
                            channels=channels_display,
                            title=f"{title} | Langmuir-style fit",
                            ylabel=ylabel,
                            vlines=vlines,
                            scan_range=scan_range,
                            edge_trim_fraction=titration_edge_trim_fraction,
                            fit_langmuir=True,
                        )
                        if fig:
                            _save(fig, f"titration/langmuir/{metric}.{fig_format}")

            for dk, ylabel, title in (
                ("peak_voltage_drift", "Peak voltage (V)", "Peak voltage drift"),
                ("skew_drift",         "Skew",             "Skew drift"),
                ("peak_offset_norm_drift", "Peak offset (normalized)", "Peak offset (normalized) drift"),
            ):
                fig = plot_drift_vs_scan(results, drift_metric=dk, channels=channels_display,
                                         title=title, ylabel=ylabel,
                                         vlines=vlines, scan_range=scan_range)
                if fig:
                    _save(fig, f"drift/{dk}.{fig_format}")

            for ch in channels_display:
                ch_res = [r for r in ok_results if r["channel"] == ch]
                if scan_range:
                    ch_res = [r for r in ch_res if scan_range[0] <= r["scan_number"] <= scan_range[1]]
                for yk, lbl in (
                    ("corrected_current", "corrected"),
                    ("smoothed_corrected_current", "smoothed_corrected"),
                    ("raw_current", "raw"),
                ):
                    fig = plot_overlaid_traces(ch_res, y_key=yk,
                                               title=f"Ch{ch}  {lbl}",
                                               show_anchors=(yk == "corrected_current"))
                    if fig:
                        _save(fig, f"overlays/ch{ch}_{lbl}.{fig_format}")

        zip_buf.seek(0)
        st.download_button("  Download figures.zip", data=zip_buf,
                           file_name="swv_figures.zip", mime="application/zip",
                           use_container_width=True)
