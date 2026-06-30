import re
from typing import Dict, List, Optional, Tuple



import matplotlib.pyplot as plt

import numpy as np

from matplotlib import cm

from matplotlib.colors import Normalize
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeWarning, curve_fit
import warnings

from .processing import find_peak_candidates





# ---- helpers

_CONCENTRATION_UNIT_TO_M = {
    "M": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "µM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
}


def _normalize_concentration_unit(unit: str) -> str:
    unit = (unit or "").strip()
    if unit in ("um", "uM", "µM", "μM"):
        return "uM"
    for known in _CONCENTRATION_UNIT_TO_M:
        if unit.lower() == known.lower():
            return known
    return unit


def _parse_concentration_marker_label(
    label: str,
    default_unit: str = "",
) -> Tuple[Optional[float], str]:
    label = str(label or "").strip()
    if not label:
        return None, ""

    buffer_match = re.match(r"\s*buffer\b", label, flags=re.IGNORECASE)
    if buffer_match:
        note = (label[:buffer_match.start()] + label[buffer_match.end():]).strip()
        note = re.sub(r"^[\s,;:|=-]+|[\s,;:|=-]+$", "", note)
        return 0.0, note or "buffer"

    match = re.match(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(pM|nM|uM|µM|μM|mM|M)?\b",
        label,
    )
    if not match:
        return None, label

    try:
        concentration = float(match.group(1))
    except ValueError:
        return None, label
    if not np.isfinite(concentration) or concentration < 0:
        return None, label

    parsed_unit = _normalize_concentration_unit(match.group(2) or "")
    target_unit = _normalize_concentration_unit(default_unit or parsed_unit)
    if parsed_unit and target_unit and parsed_unit in _CONCENTRATION_UNIT_TO_M and target_unit in _CONCENTRATION_UNIT_TO_M:
        concentration = concentration * _CONCENTRATION_UNIT_TO_M[parsed_unit] / _CONCENTRATION_UNIT_TO_M[target_unit]

    note = (label[:match.start()] + label[match.end():]).strip()
    note = re.sub(r"^[\s,;:|=-]+|[\s,;:|=-]+$", "", note)
    return concentration, note



def _cmap_fig(

    results: List[dict],

    y_key: str,

    title: str,

    ylabel: str,

    colormap_name: str,

    linewidth: float,

    alpha: float,

    show_anchors: bool = False,

    show_peak_markers: bool = False,

    show_zero_baseline: bool = False,

    show_local_baselines: bool = False,

    show_minima_candidates: bool = False,

) -> plt.Figure:

    n = len(results)

    cmap = plt.get_cmap(colormap_name, max(n, 2))

    norm = Normalize(vmin=0, vmax=max(n - 1, 1))



    fig, ax = plt.subplots(figsize=(10, 5))

    if show_zero_baseline:

        ax.axhline(0, color="gray", lw=1.0, linestyle="--", alpha=0.8)

    for i, r in enumerate(results):
        if r.get(y_key) is None or r.get("voltage") is None:

            continue

        color = cmap(norm(i))
        ax.plot(r["voltage"], r[y_key], color=color, lw=linewidth, alpha=alpha)



        if show_local_baselines and y_key == "smoothed_current" and r.get("local_baseline") is not None:

            ax.plot(

                r["voltage"], r["local_baseline"],

                color=color, lw=1.0, linestyle="--", alpha=min(alpha + 0.1, 1.0),

            )



        if show_minima_candidates and y_key == "smoothed_current":

            v = r["voltage"]

            y = r[y_key]

            double_correction_applied = bool(r.get("double_correction_applied")) and (
                r.get("second_pass_corrected_current") is not None
            )
            left_candidates_key = (
                "first_pass_left_local_min_candidates" if double_correction_applied else "left_local_min_candidates"
            )
            right_candidates_key = (
                "first_pass_right_local_min_candidates" if double_correction_applied else "right_local_min_candidates"
            )
            left_idx_key = "first_pass_left_min_idx" if double_correction_applied else "left_min_idx"
            right_idx_key = "first_pass_right_min_idx" if double_correction_applied else "right_min_idx"
            left_candidates = np.asarray(r.get(left_candidates_key, []), dtype=int)

            right_candidates = np.asarray(r.get(right_candidates_key, []), dtype=int)

            if len(left_candidates):

                ax.scatter(

                    v[left_candidates], y[left_candidates],

                    facecolors="none", edgecolors=color, s=18, zorder=5,

                    linewidths=0.8,

                )

            if len(right_candidates):

                ax.scatter(

                    v[right_candidates], y[right_candidates],

                    facecolors="none", edgecolors=color, s=18, zorder=5,

                    linewidths=0.8,

                )

            for idx_key in (left_idx_key, right_idx_key):

                idx = r.get(idx_key)

                if idx is not None and 0 <= idx < len(v):

                    ax.scatter(

                        v[idx], y[idx],

                        color=color, s=22, zorder=6,

                        edgecolors="white", linewidths=0.5,

                    )



        # Correction anchor dots - only meaningful on corrected traces

        if show_anchors and y_key == "corrected_current":

            v = r["voltage"]

            y = r[y_key]

            for idx_key in ("left_min_idx", "right_min_idx"):

                idx = r.get(idx_key)

                if idx is not None and 0 <= idx < len(v):

                    ax.scatter(

                        v[idx], y[idx],

                        color=color, s=18, zorder=5,

                        edgecolors="white", linewidths=0.5,

                    )



        if show_peak_markers:

            v = r["voltage"]

            y = r[y_key]

            peak_idx_key = (
                "peak_idx_corr"
                if y_key in ("corrected_current", "smoothed_corrected_current")
                else "peak_idx"
            )
            peak_idx = r.get(peak_idx_key)

            if peak_idx is not None and 0 <= peak_idx < len(v):

                ax.scatter(

                    v[peak_idx], y[peak_idx],

                    color=color, s=28, zorder=6,

                    edgecolors="white", linewidths=0.8,

                )



    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    sm.set_array([])

    fig.colorbar(sm, ax=ax, pad=0.02).set_label("Time order (earliest -> latest)")

    ax.set_title(title)

    ax.set_xlabel("Voltage (V)")

    ax.set_ylabel(ylabel)

    ax.grid(False)

    fig.tight_layout()

    return fig





def add_scan_vlines(ax, vlines, y_frac: float = 0.85):

    if not vlines:

        return

    for x, label in vlines:

        ax.axvline(x=x, color="gray", linestyle="--", alpha=0.6)

        ax.text(

            x, y_frac, label,

            rotation=90, va="center", ha="center",

            transform=ax.get_xaxis_transform(),

            fontsize=9, fontweight="bold", color="gray",

            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5),

        )





def _filter_titration_vlines(

    vlines: Optional[List[Tuple[float, str]]],

    scan_range: Optional[Tuple[int, int]] = None,

) -> List[Tuple[float, str]]:

    if not vlines:

        return []

    if scan_range:
        start_scan, end_scan = scan_range
        in_range = [
            (float(x), str(label))
            for x, label in vlines
            if start_scan <= x <= end_scan
        ]
        left_candidates = [
            (float(x), str(label))
            for x, label in vlines
            if float(x) < start_scan
        ]
        filtered = ([max(left_candidates, key=lambda item: item[0])] if left_candidates else []) + in_range
    else:
        filtered = [(float(x), str(label)) for x, label in vlines]
    filtered = sorted(filtered, key=lambda item: item[0])

    deduped: List[Tuple[float, str]] = []
    for x, label in filtered:

        if deduped and np.isclose(deduped[-1][0], x):

            continue

        deduped.append((x, label))

    return deduped




def _plateau_slice(n_points: int, edge_trim_fraction: float) -> slice:

    if n_points <= 2 or edge_trim_fraction <= 0:

        return slice(0, n_points)

    trim_n = int(np.floor(n_points * edge_trim_fraction))
    if trim_n <= 0 or (n_points - (2 * trim_n)) < 1:

        return slice(0, n_points)

    return slice(trim_n, n_points - trim_n)


def _scan_window_for_value(
    scan_value: float,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
) -> Optional[Tuple[int, int]]:
    if not scan_windows:
        return None
    for start, end in scan_windows:
        if start <= scan_value <= end:
            return (start, end)
    return None




def build_titration_step_table(

    all_results: List[dict],

    metric: str,

    vlines: Optional[List[Tuple[float, str]]],

    channels: Optional[List[int]] = None,

    scan_windows: Optional[List[Tuple[int, int]]] = None,

    scan_range: Optional[Tuple[int, int]] = None,

    edge_trim_fraction: float = 0.15,
    step_concentrations: Optional[List[float]] = None,
    step_notes: Optional[List[str]] = None,
    concentration_unit: str = "",

) -> List[dict]:

    titration_vlines = _filter_titration_vlines(vlines, scan_range=scan_range)
    if len(titration_vlines) < 2:

        return []

    all_ch = sorted({r["channel"] for r in all_results})
    channels = [ch for ch in channels if ch in all_ch] if channels else all_ch
    if not channels:

        return []

    plot_results = (

        [r for r in all_results if scan_range[0] <= r["scan_number"] <= scan_range[1]]

        if scan_range else all_results

    )

    rows: List[dict] = []
    for ch in channels:

        ch_res = sorted(

            [

                r for r in plot_results

                if r.get("status") == "OK"

                and r["channel"] == ch

                and np.isfinite(r.get(metric, np.nan))

            ],

            key=lambda r: r["scan_number"],

        )
        if not ch_res:

            continue

        for step_index, ((start_scan, left_label), (end_scan, right_label)) in enumerate(

            zip(titration_vlines[:-1], titration_vlines[1:]),

            start=1,

        ):
            if scan_windows:
                start_window = _scan_window_for_value(start_scan, scan_windows=scan_windows)
                end_window = _scan_window_for_value(end_scan, scan_windows=scan_windows)
                if start_window is None or end_window is None or start_window != end_window:
                    continue

            if end_scan <= start_scan:

                continue

            step_results = [

                r for r in ch_res

                if start_scan <= r["scan_number"] < end_scan

            ]
            if not step_results:

                continue

            step_scan_numbers = np.asarray([r["scan_number"] for r in step_results], dtype=float)
            step_values = np.asarray([r.get(metric, np.nan) for r in step_results], dtype=float)
            keep = _plateau_slice(len(step_results), edge_trim_fraction)
            plateau_scan_numbers = step_scan_numbers[keep]
            plateau_values = step_values[keep]
            if plateau_values.size == 0:

                plateau_scan_numbers = step_scan_numbers
                plateau_values = step_values

            plateau_value = float(np.median(plateau_values))
            plateau_mad = float(np.median(np.abs(plateau_values - plateau_value)))
            label_concentration, label_note = _parse_concentration_marker_label(
                left_label,
                default_unit=concentration_unit,
            )
            step_concentration = _concentration_for_step(
                step_index,
                step_concentrations=step_concentrations,
            )
            if step_concentration is None:
                step_concentration = label_concentration
            step_note = label_note
            if step_notes and (step_index - 1) < len(step_notes):
                explicit_note = str(step_notes[step_index - 1]).strip()
                if explicit_note:
                    step_note = explicit_note
            concentration_label = (
                f"{step_concentration:g} {concentration_unit}".strip()
                if step_concentration is not None else ""
            )
            display_bits = [f"Step {step_index}"]
            if concentration_label:
                display_bits.append(concentration_label)
            if step_note:
                display_bits.append(step_note)

            rows.append({

                "channel": ch,

                "metric_key": metric,

                "step_index": step_index,

                "step_label": f"Step {step_index}",
                "step_display_label": " | ".join(display_bits),
                "step_concentration": step_concentration,
                "step_concentration_unit": concentration_unit if step_concentration is not None else "",
                "step_note": step_note,

                "left_vline_label": left_label,

                "right_vline_label": right_label,

                "step_start_scan": float(start_scan),

                "step_end_scan": float(end_scan),

                "midpoint_scan": float((start_scan + end_scan) / 2.0),

                "scan_start_observed": float(step_scan_numbers[0]),

                "scan_end_observed": float(step_scan_numbers[-1]),

                "plateau_scan_start": float(plateau_scan_numbers[0]),

                "plateau_scan_end": float(plateau_scan_numbers[-1]),

                "step_scan_count": int(step_scan_numbers.size),

                "plateau_scan_count": int(plateau_values.size),

                "plateau_value": plateau_value,

                "plateau_mad": plateau_mad,

            })

    return rows


def _langmuir_isotherm(x, baseline, amplitude, kd):
    return baseline + amplitude * (x / (kd + x))


def _fit_langmuir_isotherm(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    if x.size < 3 or np.unique(x).size < 3:
        return None
    if np.any(x < 0):
        return None

    baseline0 = float(y[0])
    amplitude0 = float(y[-1] - y[0])
    if np.isclose(amplitude0, 0.0):
        amplitude0 = float(np.nanmax(y) - np.nanmin(y))
        if np.isclose(amplitude0, 0.0):
            amplitude0 = 1.0

    positive_x = x[x > 0]
    if positive_x.size == 0:
        return None
    kd_floor = float(max(np.nanmin(positive_x) * 1e-9, 1e-12))
    kd0 = float(max(kd_floor, np.nanmedian(positive_x)))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, _ = curve_fit(
                _langmuir_isotherm,
                x,
                y,
                p0=(baseline0, amplitude0, kd0),
                bounds=([-np.inf, -np.inf, kd_floor], [np.inf, np.inf, np.inf]),
                maxfev=20000,
            )
    except Exception:
        return None

    return float(params[0]), float(params[1]), float(params[2])


def _fit_polynomial_segment(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 2,
) -> Optional[Tuple[np.poly1d, int]]:
    unique_x = np.unique(x)
    if x.size < 2 or unique_x.size < 2:
        return None

    degree = min(max_degree, int(unique_x.size - 1))
    if degree < 1:
        return None

    try:
        coeffs = np.polyfit(x, y, deg=degree)
    except Exception:
        return None

    return np.poly1d(coeffs), degree


def _find_saturation_idx(y: np.ndarray) -> int:
    response = np.abs(y - float(y[0]))
    if np.all(~np.isfinite(response)):
        return int(len(y) - 1)
    return int(np.nanargmax(response))


def _build_langmuir_hybrid_fit(x: np.ndarray, y: np.ndarray) -> Optional[dict]:
    if x.size < 2 or y.size < 2:
        return None

    saturation_idx = _find_saturation_idx(y)
    return {
        "saturation_idx": saturation_idx,
        "saturation_x": float(x[saturation_idx]),
        "saturation_y": float(y[saturation_idx]),
        "langmuir_params": _fit_langmuir_isotherm(x[:saturation_idx + 1], y[:saturation_idx + 1]),
        "post_sat_poly": _fit_polynomial_segment(x[saturation_idx:], y[saturation_idx:]),
    }


def _concentration_for_step(
    step_index: int,
    step_concentrations: Optional[List[float]] = None,
) -> Optional[float]:
    if not step_concentrations:
        return None
    idx = int(step_index) - 1
    if idx < 0 or idx >= len(step_concentrations):
        return None
    value = step_concentrations[idx]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value >= 0 else None


def _fit_axis_from_steps(
    ch_steps: List[dict],
    step_concentrations: Optional[List[float]] = None,
) -> Tuple[np.ndarray, str]:
    concentrations = []
    for row in ch_steps:
        concentration = _concentration_for_step(
            row["step_index"],
            step_concentrations=step_concentrations,
        )
        if concentration is None:
            concentration = row.get("step_concentration")
        concentrations.append(concentration)
    if concentrations and all(value is not None for value in concentrations):
        return np.asarray(concentrations, dtype=float), "concentration"
    return np.asarray([row["step_index"] for row in ch_steps], dtype=float), "step_index"


def build_titration_langmuir_summary_table(
    all_results: List[dict],
    metric: str,
    vlines: Optional[List[Tuple[float, str]]],
    channels: Optional[List[int]] = None,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
    edge_trim_fraction: float = 0.15,
    step_concentrations: Optional[List[float]] = None,
    concentration_unit: str = "",
) -> List[dict]:
    step_rows = build_titration_step_table(
        all_results,
        metric=metric,
        vlines=vlines,
        channels=channels,
        scan_windows=scan_windows,
        scan_range=scan_range,
        edge_trim_fraction=edge_trim_fraction,
        step_concentrations=step_concentrations,
        concentration_unit=concentration_unit,
    )
    if not step_rows:
        return []

    rows: List[dict] = []
    for ch in sorted({row["channel"] for row in step_rows}):
        ch_steps = sorted(
            [row for row in step_rows if row["channel"] == ch],
            key=lambda row: row["step_index"],
        )
        if len(ch_steps) < 2:
            continue

        x, fit_axis_kind = _fit_axis_from_steps(
            ch_steps,
            step_concentrations=step_concentrations,
        )
        y = np.asarray([row["plateau_value"] for row in ch_steps], dtype=float)
        hybrid_fit = _build_langmuir_hybrid_fit(x, y)
        if hybrid_fit is None:
            continue

        saturation_idx = hybrid_fit["saturation_idx"]
        saturation_step = ch_steps[saturation_idx]
        langmuir_params = hybrid_fit["langmuir_params"]
        post_sat_poly = hybrid_fit["post_sat_poly"]

        baseline = None
        amplitude = None
        kd = None
        fit_status = "guide_only"
        if langmuir_params is not None and fit_axis_kind == "concentration":
            baseline = float(langmuir_params[0])
            amplitude = float(langmuir_params[1])
            kd = float(langmuir_params[2])
            fit_status = "langmuir_only"
        elif langmuir_params is not None:
            baseline = float(langmuir_params[0])
            amplitude = float(langmuir_params[1])
            fit_status = "step_index_fit_no_kd"

        post_sat_poly_degree = None
        if post_sat_poly is not None and saturation_idx < (len(ch_steps) - 1):
            _, post_sat_poly_degree = post_sat_poly
            if langmuir_params is not None and fit_axis_kind == "concentration":
                fit_status = "langmuir_plus_post_sat_poly"
            elif langmuir_params is not None:
                fit_status = "step_index_fit_plus_post_sat_poly_no_kd"
            else:
                fit_status = "guide_plus_post_sat_poly"

        rows.append({
            "channel": ch,
            "metric_key": metric,
            "fit_axis": "concentration" if fit_axis_kind == "concentration" else "titration_step_index",
            "fit_axis_unit": concentration_unit if fit_axis_kind == "concentration" else "",
            "fit_axis_note": "physical_concentration" if fit_axis_kind == "concentration" else "no_physical_kd",
            "step_count": int(len(ch_steps)),
            "pre_saturation_step_count": int(saturation_idx + 1),
            "post_saturation_step_count": int(len(ch_steps) - saturation_idx - 1),
            "saturation_step_index": float(ch_steps[saturation_idx]["step_index"]),
            "saturation_concentration": (
                float(hybrid_fit["saturation_x"])
                if fit_axis_kind == "concentration" else None
            ),
            "saturation_plateau_value": float(hybrid_fit["saturation_y"]),
            "saturation_left_vline_label": saturation_step["left_vline_label"],
            "saturation_right_vline_label": saturation_step["right_vline_label"],
            "langmuir_fit_used": bool(langmuir_params is not None and fit_axis_kind == "concentration"),
            "langmuir_fit_status": fit_status,
            "langmuir_baseline": baseline,
            "langmuir_amplitude": amplitude,
            "langmuir_kd": kd,
            "langmuir_kd_unit": concentration_unit if kd is not None else "",
            "post_saturation_polynomial_degree": post_sat_poly_degree,
        })

    return rows


# ---- public plot functions



def plot_overlaid_traces(

    results: List[dict],

    y_key: str = "corrected_current",

    title: str = "Overlaid Traces",

    ylabel: str = "Current (uA)",

    colormap_name: str = "plasma",

    linewidth: float = 0.9,

    alpha: float = 0.85,

    show_anchors: bool = False,

    show_peak_markers: bool = False,

    show_zero_baseline: bool = False,

    show_local_baselines: bool = False,

    show_minima_candidates: bool = False,

) -> Optional[plt.Figure]:
    usable = [r for r in results if r.get(y_key) is not None and r.get("voltage") is not None]

    if not usable:

        return None

    return _cmap_fig(usable, y_key, title, ylabel, colormap_name, linewidth, alpha,

                     show_anchors=show_anchors,

                     show_peak_markers=show_peak_markers,

                     show_zero_baseline=show_zero_baseline,

                     show_local_baselines=show_local_baselines,

                     show_minima_candidates=show_minima_candidates)





def plot_failed_traces(

    failed_results: List[dict],

    y_key: str = "raw_current",

    title: str = "Failed Traces",

    ylabel: str = "Current (uA)",

    colormap_name: str = "Reds",

    linewidth: float = 0.9,

    alpha: float = 0.75,

    show_peak_markers: bool = False,

    show_zero_baseline: bool = False,

    show_local_baselines: bool = False,

    show_minima_candidates: bool = False,

) -> Optional[plt.Figure]:

    usable = [r for r in failed_results if r.get(y_key) is not None and r.get("voltage") is not None]

    if not usable:

        return None



    fig = _cmap_fig(usable, y_key, f"{title}\n(n={len(usable)})", ylabel,

                    colormap_name, linewidth, alpha,

                    show_peak_markers=show_peak_markers,

                    show_zero_baseline=show_zero_baseline,

                    show_local_baselines=show_local_baselines,

                    show_minima_candidates=show_minima_candidates)



    counts: Dict[str, int] = {}

    for r in usable:

        key = r.get("error", "unknown").split("\n")[0][:80]

        counts[key] = counts.get(key, 0) + 1

    summary = "\n".join(f"{c} {k}" for k, c in sorted(counts.items(), key=lambda kv: -kv[1])[:6])

    fig.axes[0].text(

        0.02, 0.98, f"Failure reasons:\n{summary}",

        transform=fig.axes[0].transAxes, va="top", ha="left", fontsize=8,

        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=5),

    )

    return fig





def plot_metric_vs_scan(

    all_results: List[dict],

    metric: str,

    channels: Optional[List[int]] = None,

    title: Optional[str] = None,

    ylabel: Optional[str] = None,

    vlines: Optional[List[Tuple[float, str]]] = None,

    vline_y_frac: float = 0.85,

    scan_range: Optional[Tuple[int, int]] = None,

    figsize: Tuple[int, int] = (10, 4),
    xlabel: str = "Scan number",

    highlight_channel: Optional[int] = None,

) -> Optional[plt.Figure]:

    all_ch = sorted({r["channel"] for r in all_results})

    channels = [ch for ch in channels if ch in all_ch] if channels else all_ch

    if not channels:

        return None



    plot_results = (

        [r for r in all_results if scan_range[0] <= r["scan_number"] <= scan_range[1]]

        if scan_range else all_results

    )

    filtered_vlines = (

        [(x, lab) for x, lab in vlines if scan_range[0] <= x <= scan_range[1]]

        if scan_range and vlines else vlines

    )



    cmap = plt.get_cmap("tab10")

    colors = {ch: cmap(i % 10) for i, ch in enumerate(all_ch)}



    fig, ax = plt.subplots(figsize=figsize)

    for ch in channels:

        ch_res = sorted([r for r in plot_results if r["channel"] == ch],

                        key=lambda r: r["scan_number"])

        if not ch_res:

            continue

        x = [r["scan_number"] for r in ch_res]

        y = [r.get(metric, np.nan) for r in ch_res]

        dimmed = highlight_channel is not None and ch != highlight_channel

        ax.plot(x, y, marker="o", ms=3, lw=1.6,

                color=colors[ch],

                alpha=0.15 if dimmed else 0.9,

                label=f"Ch{ch}")



    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel or metric)

    ax.set_title(title or f"{metric} vs Scan")

    ax.grid(False)

    ax.legend(title="Channel", loc="best", fontsize=8)

    add_scan_vlines(ax, filtered_vlines, vline_y_frac)

    if scan_range:

        ax.set_xlim(scan_range)

    fig.tight_layout()

    return fig





def plot_titration_plateaus(

    all_results: List[dict],

    metric: str,

    vlines: Optional[List[Tuple[float, str]]],

    channels: Optional[List[int]] = None,

    title: Optional[str] = None,

    ylabel: Optional[str] = None,

    scan_windows: Optional[List[Tuple[int, int]]] = None,

    scan_range: Optional[Tuple[int, int]] = None,

    edge_trim_fraction: float = 0.15,

    vline_y_frac: float = 0.85,

    figsize: Tuple[int, int] = (10, 4),

    highlight_channel: Optional[int] = None,

) -> Optional[plt.Figure]:

    step_rows = build_titration_step_table(

        all_results,

        metric=metric,

        vlines=vlines,

        channels=channels,

        scan_windows=scan_windows,

        scan_range=scan_range,

        edge_trim_fraction=edge_trim_fraction,

    )
    if not step_rows:

        return None

    all_ch = sorted({r["channel"] for r in all_results})
    channels = sorted({row["channel"] for row in step_rows})
    plot_results = (

        [r for r in all_results if scan_range[0] <= r["scan_number"] <= scan_range[1]]

        if scan_range else all_results

    )
    filtered_vlines = _filter_titration_vlines(vlines, scan_range=scan_range)

    cmap = plt.get_cmap("tab10")
    colors = {ch: cmap(i % 10) for i, ch in enumerate(all_ch)}

    fig, ax = plt.subplots(figsize=figsize)
    for ch in channels:

        ch_res = sorted(

            [

                r for r in plot_results

                if r.get("status") == "OK"

                and r["channel"] == ch

                and np.isfinite(r.get(metric, np.nan))

            ],

            key=lambda r: r["scan_number"],

        )
        ch_steps = [row for row in step_rows if row["channel"] == ch]
        if not ch_res or not ch_steps:

            continue

        dimmed = highlight_channel is not None and ch != highlight_channel
        color = colors[ch]
        x = [r["scan_number"] for r in ch_res]
        y = [r.get(metric, np.nan) for r in ch_res]

        ax.plot(

            x,

            y,

            marker="o",

            ms=2.8,

            lw=1.0,

            color=color,

            alpha=0.08 if dimmed else 0.22,

        )

        step_midpoints = np.asarray([row["midpoint_scan"] for row in ch_steps], dtype=float)
        plateau_values = np.asarray([row["plateau_value"] for row in ch_steps], dtype=float)

        for row in ch_steps:

            ax.hlines(

                row["plateau_value"],

                row["step_start_scan"],

                row["step_end_scan"],

                color=color,

                lw=3.0,

                alpha=0.35 if dimmed else 0.95,

            )

        ax.scatter(

            step_midpoints,

            plateau_values,

            color=color,

            s=28,

            marker="D",

            alpha=0.25 if dimmed else 0.95,

            label=f"Ch{ch}",

            zorder=3,

        )

        if step_midpoints.size >= 2:

            if step_midpoints.size >= 3:

                try:

                    bridge = PchipInterpolator(step_midpoints, plateau_values)
                    x_dense = np.linspace(step_midpoints.min(), step_midpoints.max(), 300)
                    y_dense = bridge(x_dense)
                    ax.plot(

                        x_dense,

                        y_dense,

                        color=color,

                        lw=1.8,

                        linestyle="--",

                        alpha=0.25 if dimmed else 0.75,

                    )
                except Exception:

                    ax.plot(

                        step_midpoints,

                        plateau_values,

                        color=color,

                        lw=1.4,

                        linestyle="--",

                        alpha=0.25 if dimmed else 0.75,

                    )
            else:

                ax.plot(

                    step_midpoints,

                    plateau_values,

                    color=color,

                    lw=1.4,

                    linestyle="--",

                    alpha=0.25 if dimmed else 0.75,

                )

    ax.set_xlabel("Scan number")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} titration plateaus")
    ax.grid(False)
    ax.legend(title="Channel", loc="best", fontsize=8)
    add_scan_vlines(ax, filtered_vlines, vline_y_frac)
    if scan_range:

        ax.set_xlim(scan_range)

    fig.tight_layout()
    return fig


def plot_titration_langmuir(
    all_results: List[dict],
    metric: str,
    vlines: Optional[List[Tuple[float, str]]],
    channels: Optional[List[int]] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
    edge_trim_fraction: float = 0.15,
    figsize: Tuple[int, int] = (8, 4),
    highlight_channel: Optional[int] = None,
    xlabel: str = "Scan number",
    fit_langmuir: bool = True,
    fit_channels: Optional[List[int]] = None,
    step_concentrations: Optional[List[float]] = None,
    concentration_unit: str = "",
) -> Optional[plt.Figure]:
    if metric != "peak_current_selected":
        return None

    step_rows = build_titration_step_table(
        all_results,
        metric=metric,
        vlines=vlines,
        channels=channels,
        scan_windows=scan_windows,
        scan_range=scan_range,
        edge_trim_fraction=edge_trim_fraction,
        step_concentrations=step_concentrations,
        concentration_unit=concentration_unit,
    )
    if not step_rows:
        return None

    all_ch = sorted({r["channel"] for r in all_results})
    channels = sorted({row["channel"] for row in step_rows})

    cmap = plt.get_cmap("tab10")
    colors = {ch: cmap(i % 10) for i, ch in enumerate(all_ch)}

    fit_channel_set = set(fit_channels) if fit_channels is not None else None
    fig, ax = plt.subplots(figsize=figsize)
    plotted_any = False
    xticks = set()
    fit_notes: List[str] = []
    x_axis_kind = "step_index"
    concentration_xmax = None
    langmuir_xmax = None

    for ch in channels:
        ch_steps = sorted(
            [row for row in step_rows if row["channel"] == ch],
            key=lambda row: row["step_index"],
        )
        if not ch_steps:
            continue

        dimmed = highlight_channel is not None and ch != highlight_channel
        color = colors[ch]
        x, fit_axis_kind = _fit_axis_from_steps(
            ch_steps,
            step_concentrations=step_concentrations,
        )
        if fit_axis_kind == "concentration":
            x_axis_kind = "concentration"
            channel_xmax = float(np.nanmax(x)) if x.size else None
            if channel_xmax is not None and np.isfinite(channel_xmax):
                concentration_xmax = (
                    channel_xmax
                    if concentration_xmax is None
                    else max(concentration_xmax, channel_xmax)
                )
        y = np.asarray([row["plateau_value"] for row in ch_steps], dtype=float)
        if fit_axis_kind == "step_index":
            xticks.update(int(v) for v in x)

        ax.scatter(
            x,
            y,
            color=color,
            s=34,
            marker="D",
            alpha=0.25 if dimmed else 0.95,
            label=f"Ch{ch}",
            zorder=3,
        )

        if x.size >= 2:
            ax.plot(
                x,
                y,
                color=color,
                lw=1.2,
                linestyle="--",
                alpha=0.15 if dimmed else 0.45,
            )

            should_fit_channel = fit_langmuir and (
                fit_channel_set is None or ch in fit_channel_set
            )
            if should_fit_channel:
                hybrid_fit = _build_langmuir_hybrid_fit(x, y)
                if hybrid_fit is not None:
                    saturation_idx = hybrid_fit["saturation_idx"]
                    saturation_x = hybrid_fit["saturation_x"]
                    saturation_y = hybrid_fit["saturation_y"]
                    langmuir_params = hybrid_fit["langmuir_params"]
                    post_sat_poly = hybrid_fit["post_sat_poly"]

                    if langmuir_params is not None:
                        x_dense = np.linspace(x.min(), saturation_x, 300)
                        y_dense = _langmuir_isotherm(x_dense, *langmuir_params)
                        ax.plot(
                            x_dense,
                            y_dense,
                            color=color,
                            lw=2.2,
                            alpha=0.25 if dimmed else 0.85,
                        )
                        if fit_axis_kind == "concentration":
                            kd_x = float(langmuir_params[2])
                            if np.nanmin(x) <= kd_x <= np.nanmax(x):
                                kd_y = float(_langmuir_isotherm(kd_x, *langmuir_params))
                                ax.axvline(
                                    kd_x,
                                    color=color,
                                    lw=1.2,
                                    linestyle="--",
                                    alpha=0.18 if dimmed else 0.65,
                                )
                                ax.scatter(
                                    kd_x,
                                    kd_y,
                                    s=72,
                                    marker="o",
                                    facecolors=color,
                                    edgecolors="white",
                                    linewidths=1.1,
                                    alpha=0.35 if dimmed else 0.95,
                                    zorder=6,
                                )
                                unit_suffix = f" {concentration_unit}" if concentration_unit else ""
                                ax.annotate(
                                    f"Kd {kd_x:.3g}{unit_suffix}",
                                    xy=(kd_x, kd_y),
                                    xytext=(8, 10),
                                    textcoords="offset points",
                                    color=color,
                                    fontsize=8,
                                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
                                )
                    elif saturation_idx >= 1:
                        ax.plot(
                            x[:saturation_idx + 1],
                            y[:saturation_idx + 1],
                            color=color,
                            lw=1.8,
                            alpha=0.25 if dimmed else 0.75,
                        )

                    pre_sat_label = "Langmuir <= sat" if langmuir_params is not None else "guide <= sat"
                    sat_step_index = int(ch_steps[saturation_idx]["step_index"])
                    if fit_axis_kind == "concentration":
                        unit_suffix = f" {concentration_unit}" if concentration_unit else ""
                        fit_note = f"Ch{ch}: {pre_sat_label}; sat step {sat_step_index} ({saturation_x:.3g}{unit_suffix})"
                        if np.isfinite(saturation_x):
                            langmuir_xmax = (
                                saturation_x
                                if langmuir_xmax is None
                                else max(langmuir_xmax, saturation_x)
                            )
                    else:
                        fit_note = f"Ch{ch}: {pre_sat_label}; sat step {sat_step_index}"
                    if langmuir_params is not None and fit_axis_kind == "concentration":
                        unit_suffix = f" {concentration_unit}" if concentration_unit else ""
                        fit_note = f"Ch{ch}: Kd = {langmuir_params[2]:.3g}{unit_suffix}; sat step {sat_step_index} ({saturation_x:.3g}{unit_suffix})"
                    elif langmuir_params is not None:
                        fit_note += ", no Kd (missing concentration axis)"
                    fit_notes.append(fit_note)
                    ax.axvline(
                        saturation_x,
                        color=color,
                        lw=1.0,
                        linestyle=":",
                        alpha=0.18 if dimmed else 0.5,
                    )
                    ax.scatter(
                        saturation_x,
                        saturation_y,
                        s=88,
                        facecolors="white",
                        edgecolors=color,
                        linewidths=1.5,
                        zorder=5,
                    )
                    ax.annotate(
                        f"Sat. step {sat_step_index}",
                        xy=(saturation_x, saturation_y),
                        xytext=(8, -16),
                        textcoords="offset points",
                        color=color,
                        fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5),
                    )

        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return None

    if x_axis_kind == "concentration":
        unit_suffix = f" ({concentration_unit})" if concentration_unit else ""
        ax.set_xlabel(f"Ligand concentration{unit_suffix}")
        xmax = langmuir_xmax if langmuir_xmax is not None else concentration_xmax
        if xmax is not None and xmax > 0:
            ax.set_xlim(left=0, right=xmax)
    else:
        ax.set_xlabel("Titration step index")
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or f"{metric} titration isotherm")
    ax.grid(False)
    ax.legend(title="Channel", loc="best", fontsize=8)
    if xticks:
        ax.set_xticks(sorted(xticks))
    if fit_notes:
        ax.text(
            0.02,
            0.98,
            "\n".join(fit_notes),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=4),
        )

    fig.tight_layout()
    return fig




def plot_drift_vs_scan(

    all_results: List[dict],

    drift_metric: str,

    channels: Optional[List[int]] = None,

    title: Optional[str] = None,

    ylabel: Optional[str] = None,

    vlines: Optional[List[Tuple[float, str]]] = None,

    vline_y_frac: float = 0.85,

    scan_range: Optional[Tuple[int, int]] = None,

    highlight_channel: Optional[int] = None,

    figsize: Tuple[int, int] = (10, 4),

    xlabel: str = "Scan number",

) -> Optional[plt.Figure]:

    all_ch = sorted({r["channel"] for r in all_results})

    channels = [ch for ch in channels if ch in all_ch] if channels else all_ch

    if not channels:

        return None



    plot_results = (

        [r for r in all_results if scan_range[0] <= r["scan_number"] <= scan_range[1]]

        if scan_range else all_results

    )

    filtered_vlines = (

        [(x, lab) for x, lab in vlines if scan_range[0] <= x <= scan_range[1]]

        if scan_range and vlines else vlines

    )



    cmap = plt.get_cmap("tab10")

    colors = {ch: cmap(i % 10) for i, ch in enumerate(all_ch)}



    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.5)



    for ch in channels:

        ch_res = sorted([r for r in plot_results if r["channel"] == ch],

                        key=lambda r: r["scan_number"])

        if not ch_res:

            continue

        x = [r["scan_number"] for r in ch_res]

        y = [r.get(drift_metric, np.nan) for r in ch_res]

        if all(np.isnan(v) for v in y):

            continue

        dimmed = highlight_channel is not None and ch != highlight_channel

        ax.plot(x, y, marker="o", ms=3, lw=1.6,

                color=colors[ch],

                alpha=0.15 if dimmed else 0.9,

                label=f"Ch{ch}")



    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel or drift_metric)

    ax.set_title(title or drift_metric)

    ax.grid(False)

    ax.legend(title="Channel", loc="best", fontsize=8)

    add_scan_vlines(ax, filtered_vlines, vline_y_frac)

    if scan_range:

        ax.set_xlim(scan_range)

    fig.tight_layout()

    return fig





def plot_single_trace(result: dict) -> plt.Figure:

    """Single-trace inspector with mode-dependent panel count."""

    v = result["voltage"]
    double_correction_applied = bool(result.get("double_correction_applied")) and (
        result.get("second_pass_corrected_current") is not None
    )
    minima_mode = result.get("first_pass_minima_mode") if double_correction_applied else result.get("minima_mode")
    use_prominent_minima = isinstance(minima_mode, str) and minima_mode.startswith("prominent")

    first_pass_corrected_key = "first_pass_corrected_current" if double_correction_applied else "corrected_current"
    keys = ["raw_current", "smoothed_current"]
    labels = ["Raw", "Smoothed"]
    colors = ["steelblue", "darkorange"]

    if result.get("wavelet_denoised_current") is not None:
        keys.append("wavelet_denoised_current")
        labels.append("Wavelet Denoised")
        colors.append("mediumpurple")

    if use_prominent_minima:
        keys.append("inverted_smoothed_current")
        labels.append("Inverted Smoothed")
        colors.append("firebrick")

    keys.append(first_pass_corrected_key)
    labels.append("Corrected")
    colors.append("seagreen")

    if double_correction_applied:
        keys.append("second_pass_corrected_current")
        labels.append("Corrected x2")
        colors.append("mediumseagreen")

    fig_width = max(14, 4.2 * len(keys))
    fig, axes = plt.subplots(1, len(keys), figsize=(fig_width, 4), sharey=False)
    axes = np.atleast_1d(axes)

    correction_meta = {
        "corrected_current": ("left_min_idx", "right_min_idx", "peak_idx_corr", result.get("minima_mode")),
        "first_pass_corrected_current": (
            "first_pass_left_min_idx", "first_pass_right_min_idx", "first_pass_peak_idx_corr",
            result.get("first_pass_minima_mode"),
        ),
        "second_pass_corrected_current": (
            "second_pass_left_min_idx", "second_pass_right_min_idx", "second_pass_peak_idx_corr",
            result.get("second_pass_minima_mode"),
        ),
    }
    corrected_keys = set(correction_meta.keys())

    for ax, key, label, color in zip(axes, keys, labels, colors):
        if key == "inverted_smoothed_current":
            source = result.get("smoothed_current")
            y = (-np.asarray(source)) if source is not None else None
        else:
            y = result.get(key)

        if y is None:
            ax.set_visible(False)
            continue

        ax.plot(v, y, color=color, lw=1.2)

        if key == "smoothed_current" and result.get("local_baseline") is not None:
            ax.plot(v, result["local_baseline"], color="gray", lw=1, linestyle="--", label="baseline")
            if minima_mode:
                ax.text(
                    0.02, 0.98, f"minima mode: {minima_mode}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=3),
                )

        if key == "inverted_smoothed_current":
            left_candidates_key = "first_pass_left_local_min_candidates" if double_correction_applied else "left_local_min_candidates"
            right_candidates_key = "first_pass_right_local_min_candidates" if double_correction_applied else "right_local_min_candidates"
            left_idx_key = "first_pass_left_min_idx" if double_correction_applied else "left_min_idx"
            right_idx_key = "first_pass_right_min_idx" if double_correction_applied else "right_min_idx"
            left_candidates = np.asarray(result.get(left_candidates_key, []), dtype=int)
            right_candidates = np.asarray(result.get(right_candidates_key, []), dtype=int)
            if len(left_candidates):
                ax.scatter(
                    v[left_candidates], y[left_candidates],
                    facecolors="none", edgecolors="red", s=34, zorder=5,
                    linewidths=1.0, label="left minima as peaks",
                )
                top_two_left = left_candidates[:2]
                left_labels = ("1st left prominent", "2nd left prominent")
                for idx, lbl in zip(top_two_left, left_labels):
                    ax.scatter(
                        v[idx], y[idx],
                        color="red", s=52, zorder=6,
                        edgecolors="white", linewidths=0.8,
                        label=lbl,
                    )
            if len(right_candidates):
                ax.scatter(
                    v[right_candidates], y[right_candidates],
                    facecolors="none", edgecolors="blue", s=34, zorder=5,
                    linewidths=1.0, label="right minima as peaks",
                )
                top_two_right = right_candidates[:2]
                right_labels = ("1st right prominent", "2nd right prominent")
                for idx, lbl in zip(top_two_right, right_labels):
                    ax.scatter(
                        v[idx], y[idx],
                        color="blue", s=52, zorder=6,
                        edgecolors="white", linewidths=0.8,
                        label=lbl,
                    )
            for idx_key, marker_color, marker_label in (
                (left_idx_key, "red", "selected left anchor"),
                (right_idx_key, "blue", "selected right anchor"),
            ):
                idx = result.get(idx_key)
                if idx is not None and 0 <= idx < len(v):
                    ax.scatter(
                        v[idx], y[idx],
                        color=marker_color, s=40, zorder=6,
                        edgecolors="white", linewidths=0.8,
                        label=marker_label,
                    )

        if key == "smoothed_current":
            candidates = find_peak_candidates(y)
            raw_valid_peaks = candidates["raw_valid_peaks"]
            if len(raw_valid_peaks):
                ax.scatter(
                    v[raw_valid_peaks], y[raw_valid_peaks],
                    color="gold", s=28, zorder=5,
                    edgecolors="black", linewidths=0.5,
                    label="pre-prominence find_peaks",
                )

        if key in corrected_keys:
            left_idx_key, right_idx_key, _, panel_minima_mode = correction_meta[key]
            for idx_key, marker_color, marker_label in (
                (left_idx_key, "red", "left anchor"),
                (right_idx_key, "blue", "right anchor"),
            ):
                idx = result.get(idx_key)
                if idx is not None and 0 <= idx < len(v):
                    ax.scatter(
                        v[idx], y[idx],
                        color=marker_color, s=40, zorder=5,
                        edgecolors="white", linewidths=0.8,
                        label=marker_label,
                    )
            if key == "second_pass_corrected_current" and panel_minima_mode:
                ax.text(
                    0.02, 0.98, f"2nd pass minima mode: {panel_minima_mode}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=3),
                )

        peak_idx_key = "peak_idx"
        if key in corrected_keys:
            _, _, peak_idx_key, _ = correction_meta[key]
        peak_idx_for_line = result.get(peak_idx_key)
        if peak_idx_for_line is not None and 0 <= peak_idx_for_line < len(v):
            pi = peak_idx_for_line
            ax.axvline(v[pi], color="red", lw=0.8, linestyle=":")
            if key != "raw_current":
                ax.scatter(
                    v[pi], y[pi],
                    color="crimson", s=55, zorder=6,
                    edgecolors="white", linewidths=0.8,
                    label="selected dominant peak",
                )

        ax.set_title(label)
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (uA)")
        ax.grid(False)
        if key in {"smoothed_current", "inverted_smoothed_current"} | corrected_keys:
            ax.legend(fontsize=7)

    fig.suptitle(result.get("file_name", ""), fontsize=9, y=1.01)
    fig.tight_layout()
    return fig

