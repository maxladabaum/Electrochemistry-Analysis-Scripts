import re
from typing import Any, Dict, List, Optional, Tuple



import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np

from matplotlib import cm

from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
from scipy.optimize import OptimizeWarning, curve_fit
import warnings

from .processing import find_peak_candidates





# ---- helpers

def _channel_sort_key(channel: Any) -> tuple:
    text = str(channel)
    match = re.fullmatch(r"(\d+)(?:\s+group\s+(\d+))?", text, re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), int(match.group(2) or 0), text)
    try:
        return (0, int(channel), 0, text)
    except (TypeError, ValueError):
        return (1, text, 0, text)


def _high_contrast_response_shades(count: int) -> np.ndarray:
    """Alternate light and dark shades so adjacent methods stay distinct."""
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


_SWV_METHOD_BLUE_SHADES = {
    1: 0.90,
    2: 0.40,
}


def _swv_method_blue(channel: Any) -> Optional[Any]:
    """Return the fixed blue assigned to a displayed SWV method, if known."""
    match = re.match(
        r"(?:Ch)?\d+\s+group\s+(\d+)(?:\s*\||\s*$)",
        str(channel).strip(),
        re.IGNORECASE,
    )
    if not match:
        return None
    shade = _SWV_METHOD_BLUE_SHADES.get(int(match.group(1)))
    return plt.get_cmap("Blues")(shade) if shade is not None else None


def _swv_method_trace_label(channel: Any) -> Optional[str]:
    """Return the concise legend label for a displayed two-method SWV trace."""
    match = re.match(
        r"(?:Ch)?\d+\s+group\s+(\d+)(?:\s*\||\s*$)",
        str(channel).strip(),
        re.IGNORECASE,
    )
    if not match:
        return None
    return {
        1: "Optimized Method",
        2: "Manual Method",
    }.get(int(match.group(1)))


def _compact_channel_label(channel: Any) -> str:
    """Keep plot labels readable when a display channel embeds SWV settings."""
    text = str(channel).split("|", 1)[0].strip()
    match = re.fullmatch(r"(?:Ch)?(\d+)\s+group\s+(\d+)", text, re.IGNORECASE)
    if match:
        return f"Channel {match.group(1)} · method {match.group(2)}"
    channel_match = re.fullmatch(r"(?:Ch(?:annel)?)?\s*(\d+)", text, re.IGNORECASE)
    return f"Channel {channel_match.group(1)}" if channel_match else text


def _response_direction_plot_encoding(
    rows: List[dict],
    channels: List[Any],
    response_directions: Optional[Dict[Any, str]] = None,
    channel_colors: Optional[Dict[Any, Any]] = None,
) -> Tuple[Dict[Any, Any], Dict[Any, Tuple[str, str]], Dict[Any, str]]:
    """Assign a distinct direction-family shade to each physical channel."""
    directions: Dict[Any, str] = {
        channel: str(direction).strip().lower()
        for channel, direction in (response_directions or {}).items()
        if channel in channels
        and str(direction).strip().lower() in {"signal-on", "signal-off"}
    }
    for channel in channels:
        if channel in directions:
            continue
        channel_rows = [row for row in rows if row.get("channel") == channel]
        explicit_directions = {
            str(row.get("langmuir_response_direction", "")).strip().lower()
            for row in channel_rows
            if str(row.get("langmuir_response_direction", "")).strip().lower()
            in {"signal-on", "signal-off"}
        }
        if len(explicit_directions) == 1:
            directions[channel] = next(iter(explicit_directions))
            continue
        amplitudes = []
        for row in channel_rows:
            for amplitude_key in ("fit_amplitude", "langmuir_amplitude"):
                try:
                    amplitude = float(row.get(amplitude_key))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(amplitude) and not np.isclose(amplitude, 0.0):
                    amplitudes.append(amplitude)
                    break
        if amplitudes:
            directions[channel] = (
                "signal-off" if float(np.median(amplitudes)) < 0 else "signal-on"
            )

    colors = {}
    for direction, colormap_name in (
        ("signal-on", "Oranges"),
        ("signal-off", "Blues"),
        ("unknown", "Greys"),
    ):
        direction_channels = [
            channel for channel in channels
            if directions.get(channel, "unknown") == direction
        ]
        shades = dict(zip(
            direction_channels,
            _high_contrast_response_shades(len(direction_channels)),
        ))
        colors.update({
            channel: plt.get_cmap(colormap_name)(shades[channel])
            for channel in direction_channels
        })
    colors.update({
        channel: color
        for channel, color in (channel_colors or {}).items()
        if channel in channels
    })

    styles = {channel: ("-", "o") for channel in channels}
    return colors, styles, directions

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


def _concentration_to_doubling_level(
    values: Any,
    minimum_nonzero_concentration: float,
) -> np.ndarray:
    """Map buffer to zero and selected-dose doublings to equal y intervals."""
    concentrations = np.asarray(values, dtype=float)
    levels = np.zeros_like(concentrations, dtype=float)
    linear_region = (
        np.isfinite(concentrations)
        & (concentrations <= minimum_nonzero_concentration)
    )
    levels[linear_region] = (
        concentrations[linear_region] / minimum_nonzero_concentration
    )
    above_minimum = (
        np.isfinite(concentrations)
        & (concentrations > minimum_nonzero_concentration)
    )
    levels[above_minimum] = 1.0 + np.log2(
        concentrations[above_minimum] / minimum_nonzero_concentration
    )
    levels[~np.isfinite(concentrations)] = np.nan
    return levels


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

    normalize_to_peak: bool = False,

    offset_to_baseline: bool = False,

) -> plt.Figure:

    n = len(results)

    cmap = plt.get_cmap(colormap_name, max(n, 2))

    norm = Normalize(vmin=1, vmax=max(n, 2))



    fig, ax = plt.subplots(figsize=(10, 5))

    if show_zero_baseline:

        ax.axhline(0, color="gray", lw=1.0, linestyle="--", alpha=0.8)

    for i, r in enumerate(results, start=1):
        if r.get(y_key) is None or r.get("voltage") is None:

            continue

        y_plot = np.asarray(r[y_key], dtype=float)
        if offset_to_baseline:
            y_plot = _offset_trace_to_anchor_baseline(
                y_plot,
                r.get("left_min_idx"),
                r.get("right_min_idx"),
            )
        peak_idx_key = (
            "peak_idx_corr"
            if y_key in ("corrected_current", "smoothed_corrected_current")
            else "peak_idx"
        )
        if normalize_to_peak:
            peak_idx = r.get(peak_idx_key)
            if peak_idx is None or not 0 <= peak_idx < len(y_plot):
                continue
            peak_height = float(y_plot[peak_idx])
            if not np.isfinite(peak_height) or np.isclose(peak_height, 0.0):
                continue
            y_plot = y_plot / peak_height

        color = cmap(norm(i))
        ax.plot(r["voltage"], y_plot, color=color, lw=linewidth, alpha=alpha)



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



        # Correction anchor dots - meaningful on corrected and offset-raw traces.

        if show_anchors and (y_key == "corrected_current" or offset_to_baseline):

            v = r["voltage"]

            y = y_plot

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

            y = y_plot

            peak_idx = r.get(peak_idx_key)

            if peak_idx is not None and 0 <= peak_idx < len(v):

                ax.scatter(

                    v[peak_idx], y[peak_idx],

                    color=color, s=28, zorder=6,

                    edgecolors="white", linewidths=0.8,

                )



    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    sm.set_array([])

    colorbar = fig.colorbar(sm, ax=ax, pad=0.02)
    if n == 1:
        colorbar.set_ticks([1])
        colorbar.set_ticklabels(["1"])
    else:
        colorbar.set_ticks([1, n])
        colorbar.set_ticklabels(["1", str(n)])
    colorbar.set_label("SWV Measurement Number")

    ax.set_title(title)

    ax.set_xlabel("Voltage (V)")

    ax.set_ylabel(ylabel)

    if normalize_to_peak:

        ax.set_ylim(-0.2, 1.2)

    ax.grid(False)

    fig.tight_layout()

    return fig


def _offset_trace_to_anchor_baseline(
    y: np.ndarray,
    left_idx: Optional[int],
    right_idx: Optional[int],
) -> np.ndarray:
    """Subtract the straight line joining a trace's detected minima anchors."""
    values = np.asarray(y, dtype=float)
    if left_idx is None or right_idx is None:
        raise ValueError("Trace has no detected left/right minima to offset.")
    left = int(left_idx)
    right = int(right_idx)
    if not (
        0 <= left < len(values)
        and 0 <= right < len(values)
        and np.isfinite(values[left])
        and np.isfinite(values[right])
    ):
        raise ValueError("Trace minima are outside the trace or non-finite.")
    if left == right:
        return values - float(values[left])
    baseline = np.interp(
        np.arange(len(values), dtype=float),
        [float(left), float(right)],
        [float(values[left]), float(values[right])],
    )
    return values - baseline





def add_scan_vlines(
    ax,
    vlines,
    y_frac: float = 0.85,
    fontsize: float = 9,
    fontweight: str = "bold",
    bbox_alpha: float = 0.6,
):

    if not vlines:

        return

    for x, label in vlines:

        ax.axvline(x=x, color="gray", linestyle="--", alpha=0.6)

        label_artist = ax.text(

            x, y_frac, label,

            rotation=90, va="center", ha="center",

            transform=ax.get_xaxis_transform(),

            fontsize=fontsize, fontweight=fontweight, color="gray",

            bbox=dict(facecolor="white", edgecolor="none", alpha=bbox_alpha, pad=1.0),

        )
        label_artist._swv_preserve_fontsize = True





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


def _titration_step_selection_keys(
    vlines: List[Tuple[float, str]],
) -> List[str]:
    """Return stable, human-readable keys for each vline interval."""
    bases = [
        "buffer"
        if str(label).strip().lower().startswith("buffer")
        else str(label).strip()
        for _position, label in vlines[:-1]
    ]
    totals = {base: bases.count(base) for base in set(bases)}
    occurrences: Dict[str, int] = {}
    keys = []
    for base in bases:
        occurrences[base] = occurrences.get(base, 0) + 1
        if base == "buffer" or totals[base] > 1:
            keys.append(f"{base}_{occurrences[base]}")
        else:
            keys.append(base)
    return keys


def filter_extreme_titration_outliers(
    all_results: List[dict],
    metric: str,
    vlines: Optional[List[Tuple[float, str]]],
    channels: Optional[List[Any]] = None,
    vlines_by_channel: Optional[Dict[Any, List[Tuple[float, str]]]] = None,
    modified_z_cutoff: float = 5.0,
) -> List[dict]:
    """Remove isolated extreme values within each channel and titration interval."""
    if not all_results:
        return []
    all_channels = sorted({row.get("channel") for row in all_results}, key=_channel_sort_key)
    selected_channels = [ch for ch in channels if ch in all_channels] if channels else all_channels
    excluded_ids = set()
    fallback_vlines = sorted(vlines or [], key=lambda item: item[0])

    def _row_is_in_interval(row: dict, channel: Any, start: float, end: float) -> bool:
        if row.get("channel") != channel or row.get("status") != "OK":
            return False
        try:
            scan_number = float(row.get("scan_number"))
            metric_value = float(row.get(metric))
        except (TypeError, ValueError):
            return False
        return start <= scan_number < end and np.isfinite(metric_value)

    for channel in selected_channels:
        channel_vlines = sorted(
            (vlines_by_channel or {}).get(channel, fallback_vlines),
            key=lambda item: item[0],
        )
        for (start, _left_label), (end, _right_label) in zip(
            channel_vlines[:-1],
            channel_vlines[1:],
        ):
            interval_rows = [
                row for row in all_results
                if _row_is_in_interval(row, channel, start, end)
            ]
            if len(interval_rows) < 3:
                continue
            values = np.asarray([row[metric] for row in interval_rows], dtype=float)
            median = float(np.median(values))
            deviations = np.abs(values - median)
            mad = float(np.median(deviations))
            if mad > np.finfo(float).eps:
                mask = (0.67448975 * deviations / mad) > modified_z_cutoff
            else:
                tolerance = max(abs(median), 1.0) * 1e-12
                differs = deviations > tolerance
                mask = differs if np.count_nonzero(~differs) >= (len(values) // 2 + 1) else np.zeros(len(values), dtype=bool)
            excluded_ids.update(
                id(row) for row, is_outlier in zip(interval_rows, mask) if is_outlier
            )
    return [row for row in all_results if id(row) not in excluded_ids]


def _filter_extreme_accuracy_predictions(
    accuracy_rows: List[dict],
    modified_z_cutoff: float = 5.0,
) -> List[dict]:
    """Remove isolated nonlinear-inversion outliers within each known dose."""
    excluded_ids = set()
    groups: Dict[Tuple[Any, Any], List[dict]] = {}
    for row in accuracy_rows:
        predicted = row.get("predicted_concentration")
        try:
            predicted = float(predicted)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(predicted) or predicted <= 0:
            continue
        group_key = (
            row.get("channel"),
            row.get("step_selection_key", row.get("known_concentration")),
        )
        groups.setdefault(group_key, []).append(row)

    for group_rows in groups.values():
        if len(group_rows) < 3:
            continue
        # Concentration inversion is multiplicative and diverges near the
        # Langmuir asymptote, so robust detection belongs in log space.
        values = np.asarray(
            [np.log10(float(row["predicted_concentration"])) for row in group_rows],
            dtype=float,
        )
        median = float(np.median(values))
        deviations = np.abs(values - median)
        mad = float(np.median(deviations))
        if mad > np.finfo(float).eps:
            mask = (0.67448975 * deviations / mad) > modified_z_cutoff
        else:
            tolerance = max(abs(median), 1.0) * 1e-12
            differs = deviations > tolerance
            mask = (
                differs
                if np.count_nonzero(~differs) >= (len(values) // 2 + 1)
                else np.zeros(len(values), dtype=bool)
            )
        excluded_ids.update(
            id(row) for row, is_outlier in zip(group_rows, mask) if is_outlier
        )
    return [row for row in accuracy_rows if id(row) not in excluded_ids]




def build_titration_step_table(

    all_results: List[dict],

    metric: str,

    vlines: Optional[List[Tuple[float, str]]],

    channels: Optional[List[Any]] = None,
    vlines_by_channel: Optional[Dict[Any, List[Tuple[float, str]]]] = None,

    scan_windows: Optional[List[Tuple[int, int]]] = None,

    scan_range: Optional[Tuple[int, int]] = None,

    edge_trim_fraction: float = 0.15,
    step_concentrations: Optional[List[float]] = None,
    step_notes: Optional[List[str]] = None,
    concentration_unit: str = "",
    baseline_mode: str = "none",
    included_step_labels: Optional[List[str]] = None,
    remove_extreme_outliers: bool = False,

) -> List[dict]:

    titration_vlines = _filter_titration_vlines(vlines, scan_range=scan_range)
    if len(titration_vlines) < 2 and not vlines_by_channel:

        return []

    all_ch = sorted({r["channel"] for r in all_results})
    channels = [ch for ch in channels if ch in all_ch] if channels else all_ch
    if not channels:

        return []

    source_results = (
        filter_extreme_titration_outliers(
            all_results,
            metric=metric,
            vlines=titration_vlines,
            channels=channels,
            vlines_by_channel=vlines_by_channel,
        )
        if remove_extreme_outliers else all_results
    )
    plot_results = (

        [r for r in source_results if scan_range[0] <= r["scan_number"] <= scan_range[1]]

        if scan_range else source_results

    )

    rows: List[dict] = []
    for ch in channels:

        channel_vlines = _filter_titration_vlines(
            (vlines_by_channel or {}).get(ch, titration_vlines),
            scan_range=scan_range,
        )
        if len(channel_vlines) < 2:
            continue

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
        original_channel = ch_res[0].get("original_channel", ch)

        channel_selection_keys = _titration_step_selection_keys(channel_vlines)
        for step_index, ((start_scan, left_label), (end_scan, right_label)) in enumerate(

            zip(channel_vlines[:-1], channel_vlines[1:]),

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
            plateau_std = float(np.std(plateau_values, ddof=1)) if plateau_values.size > 1 else 0.0
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

                "original_channel": original_channel,

                "metric_key": metric,

                "step_index": step_index,

                "step_label": f"Step {step_index}",
                "step_display_label": " | ".join(display_bits),
                "step_concentration": step_concentration,
                "step_concentration_unit": concentration_unit if step_concentration is not None else "",
                "step_note": step_note,
                "step_selection_key": channel_selection_keys[step_index - 1],

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

                "raw_plateau_value": plateau_value,

                "plateau_mad": plateau_mad,

                "plateau_std": plateau_std,

                "baseline_mode": "none",

                "baseline_step_index": None,

                "baseline_value": None,

                "baseline_plateau_std": None,

                "titration_snr": None,

                "snr_noise_std": None,

            })

    included_label_set = (
        {str(label) for label in included_step_labels}
        if included_step_labels is not None
        else None
    )

    def _selected(row: dict) -> bool:
        return (
            included_label_set is None
            or str(row.get("step_selection_key")) in included_label_set
        )

    selected_rows: List[dict] = []
    for ch in channels:
        channel_output_start = len(selected_rows)
        channel_rows = sorted(
            [row for row in rows if row["channel"] == ch],
            key=lambda row: row["step_index"],
        )

        # B is always the closest buffer preceding the first selected target.
        selected_preceding_buffer: Optional[dict] = None
        anchor_buffer: Optional[dict] = None
        for row in channel_rows:
            is_buffer = str(row.get("step_note", "")).strip().lower() == "buffer"
            if is_buffer:
                if _selected(row):
                    selected_preceding_buffer = row
                continue
            if (
                _selected(row)
                and row.get("step_concentration") is not None
                and selected_preceding_buffer is not None
            ):
                anchor_buffer = selected_preceding_buffer
                break

        selected_buffer_stds = [
            float(row["plateau_std"])
            for row in channel_rows
            if str(row.get("step_note", "")).strip().lower() == "buffer"
            and _selected(row)
            and np.isfinite(row.get("plateau_std", np.nan))
            and float(row["plateau_std"]) > 0
        ]

        preceding_buffer = None
        for row in channel_rows:
            is_buffer = str(row.get("step_note", "")).strip().lower() == "buffer"
            if is_buffer:
                preceding_buffer = row
                if baseline_mode != "preceding_buffer" and _selected(row):
                    selected_rows.append(dict(row))
                continue
            if not _selected(row) or row.get("step_concentration") is None:
                continue
            if anchor_buffer is None or preceding_buffer is None:
                if baseline_mode != "preceding_buffer":
                    selected_rows.append(dict(row))
                continue

            adjusted = dict(row)
            adjusted["first_buffer_step_index"] = anchor_buffer["step_index"]
            adjusted["first_buffer_value"] = anchor_buffer["raw_plateau_value"]
            adjusted["anchor_buffer_step_index"] = anchor_buffer["step_index"]
            adjusted["anchor_buffer_value"] = anchor_buffer["raw_plateau_value"]
            adjusted["lod_buffer_stds"] = selected_buffer_stds
            snr_noise_std = (
                float(np.median(selected_buffer_stds))
                if selected_buffer_stds else None
            )
            adjusted["snr_noise_std"] = snr_noise_std
            adjusted["fixed_langmuir_baseline"] = anchor_buffer["raw_plateau_value"]
            if (
                baseline_mode == "preceding_buffer"
            ):
                adjusted["baseline_mode"] = "preceding_buffer"
                adjusted["baseline_step_index"] = preceding_buffer["step_index"]
                adjusted["baseline_value"] = preceding_buffer["raw_plateau_value"]
                adjusted["baseline_plateau_std"] = preceding_buffer["plateau_std"]
                adjusted["plateau_value"] = (
                    row["raw_plateau_value"] - preceding_buffer["raw_plateau_value"]
                    + anchor_buffer["raw_plateau_value"]
                )
                adjusted["step_display_label"] += (
                    f" | drift-corrected with buffer step {preceding_buffer['step_index']}"
                )
            else:
                adjusted["baseline_step_index"] = anchor_buffer["step_index"]
                adjusted["baseline_value"] = anchor_buffer["raw_plateau_value"]
                adjusted["baseline_plateau_std"] = anchor_buffer["plateau_std"]
            if snr_noise_std is not None and snr_noise_std > 0:
                adjusted["titration_snr"] = abs(
                    adjusted["plateau_value"]
                    - adjusted["fixed_langmuir_baseline"]
                ) / snr_noise_std
            selected_rows.append(adjusted)
        if anchor_buffer is not None and selected_buffer_stds:
            channel_noise_std = float(np.median(selected_buffer_stds))
            for selected_row in selected_rows[channel_output_start:]:
                selected_row["snr_noise_std"] = channel_noise_std
                if selected_row.get("titration_snr") is None:
                    selected_row["titration_snr"] = abs(
                        float(selected_row["plateau_value"])
                        - float(anchor_buffer["raw_plateau_value"])
                    ) / channel_noise_std
    return selected_rows


def infer_titration_response_directions(
    all_results: List[dict],
    metric: str,
    vlines: Optional[List[Tuple[float, str]]],
    channels: Optional[List[Any]] = None,
    vlines_by_channel: Optional[Dict[Any, List[Tuple[float, str]]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
    edge_trim_fraction: float = 0.15,
    concentration_unit: str = "",
    included_step_labels: Optional[List[str]] = None,
    remove_extreme_outliers: bool = False,
) -> Dict[Any, str]:
    """Infer signal direction from selected target responses relative to buffer."""
    paired_rows = build_titration_step_table(
        all_results,
        metric=metric,
        vlines=vlines,
        channels=channels,
        vlines_by_channel=vlines_by_channel,
        scan_range=scan_range,
        edge_trim_fraction=edge_trim_fraction,
        concentration_unit=concentration_unit,
        baseline_mode="preceding_buffer",
        included_step_labels=included_step_labels,
        remove_extreme_outliers=remove_extreme_outliers,
    )
    directions: Dict[Any, str] = {}
    for channel in sorted(
        {row.get("channel") for row in paired_rows},
        key=_channel_sort_key,
    ):
        response_changes = []
        response_scale = []
        for row in paired_rows:
            if row.get("channel") != channel:
                continue
            try:
                plateau = float(row.get("plateau_value"))
                baseline = float(row.get("fixed_langmuir_baseline"))
                concentration = float(row.get("step_concentration"))
            except (TypeError, ValueError):
                continue
            if (
                concentration > 0
                and np.isfinite([plateau, baseline, concentration]).all()
            ):
                response_changes.append(plateau - baseline)
                response_scale.extend((plateau, baseline))
        if not response_changes:
            continue
        median_change = float(np.median(response_changes))
        scale = max(
            [abs(value) for value in response_scale if np.isfinite(value)]
            + [1.0]
        )
        if np.isclose(median_change, 0.0, atol=scale * 1e-10, rtol=0.0):
            continue
        directions[channel] = (
            "signal-on" if median_change > 0 else "signal-off"
        )
    return directions


def infer_titration_response_baselines(
    all_results: List[dict],
    metric: str,
    vlines: Optional[List[Tuple[float, str]]],
    channels: Optional[List[Any]] = None,
    vlines_by_channel: Optional[Dict[Any, List[Tuple[float, str]]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
    edge_trim_fraction: float = 0.15,
    concentration_unit: str = "",
    included_step_labels: Optional[List[str]] = None,
    remove_extreme_outliers: bool = False,
) -> Dict[Any, float]:
    """Return each channel's selected anchor-buffer plateau."""
    paired_rows = build_titration_step_table(
        all_results,
        metric=metric,
        vlines=vlines,
        channels=channels,
        vlines_by_channel=vlines_by_channel,
        scan_range=scan_range,
        edge_trim_fraction=edge_trim_fraction,
        concentration_unit=concentration_unit,
        baseline_mode="preceding_buffer",
        included_step_labels=included_step_labels,
        remove_extreme_outliers=remove_extreme_outliers,
    )
    baseline_values: Dict[Any, List[float]] = {}
    for row in paired_rows:
        try:
            baseline = float(row.get("fixed_langmuir_baseline"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(baseline):
            baseline_values.setdefault(row.get("channel"), []).append(baseline)
    return {
        channel: float(np.median(values))
        for channel, values in baseline_values.items()
        if values
    }


def _langmuir_isotherm(x, baseline, amplitude, kd):
    return baseline + amplitude * (x / (kd + x))


def _fit_langmuir_isotherm(
    x: np.ndarray,
    y: np.ndarray,
    fixed_baseline: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    fit_result = _fit_langmuir_isotherm_with_covariance(
        x,
        y,
        fixed_baseline=fixed_baseline,
    )
    return fit_result[0] if fit_result is not None else None


def _fit_langmuir_isotherm_with_covariance(
    x: np.ndarray,
    y: np.ndarray,
    fixed_baseline: Optional[float] = None,
) -> Optional[Tuple[Tuple[float, float, float], np.ndarray]]:
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[finite], dtype=float)
    y = np.asarray(y[finite], dtype=float)
    minimum_points = 2 if fixed_baseline is not None else 3
    if x.size < minimum_points or np.unique(x).size < minimum_points:
        return None
    if np.any(x < 0):
        return None

    baseline0 = float(y[0]) if fixed_baseline is None else float(fixed_baseline)
    amplitude0 = float(y[-1] - baseline0)
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
            if fixed_baseline is None:
                params, covariance = curve_fit(
                    _langmuir_isotherm,
                    x,
                    y,
                    p0=(baseline0, amplitude0, kd0),
                    bounds=([-np.inf, -np.inf, kd_floor], [np.inf, np.inf, np.inf]),
                    maxfev=20000,
                )
            else:
                fitted, fitted_covariance = curve_fit(
                    lambda concentration, amplitude, kd: _langmuir_isotherm(
                        concentration,
                        baseline0,
                        amplitude,
                        kd,
                    ),
                    x,
                    y,
                    p0=(amplitude0, kd0),
                    bounds=([-np.inf, kd_floor], [np.inf, np.inf]),
                    maxfev=20000,
                )
                params = (baseline0, fitted[0], fitted[1])
                covariance = np.full((3, 3), np.nan, dtype=float)
                covariance[1:, 1:] = fitted_covariance
    except Exception:
        return None

    return (
        (float(params[0]), float(params[1]), float(params[2])),
        np.asarray(covariance, dtype=float),
    )


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


def _find_saturation_idx(y: np.ndarray, baseline: Optional[float] = None) -> int:
    reference = float(y[0]) if baseline is None else float(baseline)
    response = np.abs(y - reference)
    if np.all(~np.isfinite(response)):
        return int(len(y) - 1)
    return int(np.nanargmax(response))


def _build_langmuir_hybrid_fit(
    x: np.ndarray,
    y: np.ndarray,
    fixed_baseline: Optional[float] = None,
) -> Optional[dict]:
    if x.size < 2 or y.size < 2:
        return None

    saturation_idx = _find_saturation_idx(y, baseline=fixed_baseline)
    fit_result = _fit_langmuir_isotherm_with_covariance(
        x[:saturation_idx + 1],
        y[:saturation_idx + 1],
        fixed_baseline=fixed_baseline,
    )
    return {
        "saturation_idx": saturation_idx,
        "saturation_x": float(x[saturation_idx]),
        "saturation_y": float(y[saturation_idx]),
        "langmuir_params": fit_result[0] if fit_result is not None else None,
        "langmuir_covariance": fit_result[1] if fit_result is not None else None,
        "post_sat_poly": _fit_polynomial_segment(x[saturation_idx:], y[saturation_idx:]),
    }


def _langmuir_limit_of_detection(
    ch_steps: List[dict],
    langmuir_params: Optional[Tuple[float, float, float]],
) -> Tuple[Optional[float], Optional[float]]:
    """Return (LOD, blank sigma) using 3σ and the fitted zero-dose slope."""
    if langmuir_params is None:
        return None, None
    _baseline, amplitude, kd = langmuir_params
    initial_slope = abs(float(amplitude) / float(kd)) if kd > 0 else 0.0
    if not np.isfinite(initial_slope) or initial_slope <= 0:
        return None, None

    explicit_buffer_sigmas = None
    for row in ch_steps:
        if "lod_buffer_stds" in row:
            explicit_buffer_sigmas = row.get("lod_buffer_stds") or []
            break
    if explicit_buffer_sigmas is not None:
        blank_sigmas = [
            float(value)
            for value in explicit_buffer_sigmas
            if np.isfinite(value) and float(value) > 0
        ]
    else:
        blank_sigmas = []
    for row in ch_steps:
        if explicit_buffer_sigmas is not None:
            break
        value = row.get("baseline_plateau_std")
        if value is None and (
            str(row.get("step_note", "")).strip().lower() == "buffer"
            or row.get("step_concentration") == 0
        ):
            value = row.get("plateau_std")
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            blank_sigmas.append(value)
    if not blank_sigmas:
        return None, None
    blank_sigma = float(np.median(blank_sigmas))
    return float((3.0 * blank_sigma) / initial_slope), blank_sigma


def _langmuir_upper_limit_of_quantification(
    langmuir_params: Optional[Tuple[float, float, float]],
    noise_sigma: Optional[float],
    sigma_multiplier: float = 3.0,
) -> Optional[float]:
    """Return the concentration whose fitted response is 3σ from saturation."""
    if langmuir_params is None or noise_sigma is None:
        return None
    _baseline, amplitude, kd = langmuir_params
    amplitude = abs(float(amplitude))
    kd = float(kd)
    threshold = float(sigma_multiplier) * float(noise_sigma)
    if (
        not np.isfinite(amplitude)
        or not np.isfinite(kd)
        or not np.isfinite(threshold)
        or amplitude <= threshold
        or kd <= 0
        or threshold <= 0
    ):
        return None
    uloq = kd * ((amplitude / threshold) - 1.0)
    return float(uloq) if np.isfinite(uloq) and uloq >= 0 else None


def _langmuir_snr_cutoff_concentration(
    langmuir_params: Optional[Tuple[float, float, float]],
    noise_sigma: Optional[float],
    snr_cutoff: float = 3.0,
) -> Optional[float]:
    """Return the exact concentration where the fitted Langmuir response reaches an SNR cutoff."""
    if langmuir_params is None or noise_sigma is None:
        return None
    _baseline, amplitude, kd = langmuir_params
    amplitude = abs(float(amplitude))
    kd = float(kd)
    threshold_response = float(snr_cutoff) * float(noise_sigma)
    if (
        not np.isfinite(amplitude)
        or not np.isfinite(kd)
        or not np.isfinite(threshold_response)
        or amplitude <= threshold_response
        or kd <= 0
        or threshold_response <= 0
    ):
        return None
    concentration = threshold_response * kd / (amplitude - threshold_response)
    return float(concentration) if np.isfinite(concentration) and concentration >= 0 else None


def _high_concentration_response_noise(
    target_steps: List[dict],
    fallback_sigma: Optional[float],
) -> Tuple[Optional[float], str]:
    """Estimate saturation-side noise from the highest selected target plateaus."""
    concentration_sigmas = []
    for row in target_steps:
        try:
            concentration = float(row.get("step_concentration"))
            sigma = float(row.get("plateau_std"))
        except (TypeError, ValueError):
            continue
        if concentration > 0 and np.isfinite(concentration) and sigma > 0 and np.isfinite(sigma):
            concentration_sigmas.append((concentration, sigma))
    if concentration_sigmas:
        concentration_sigmas.sort(key=lambda item: item[0])
        high_count = max(1, int(np.ceil(len(concentration_sigmas) / 2.0)))
        high_sigmas = [sigma for _concentration, sigma in concentration_sigmas[-high_count:]]
        return float(np.median(high_sigmas)), "median SD of highest selected target plateaus"
    if fallback_sigma is not None and np.isfinite(fallback_sigma) and fallback_sigma > 0:
        return float(fallback_sigma), "selected-buffer SD fallback"
    return None, ""


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


def _prepare_titration_fit_points(
    ch_steps: List[dict],
    step_concentrations: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, str, List[dict]]:
    """Sort dose points and median-collapse repeated concentrations for fitting."""
    x, fit_axis_kind = _fit_axis_from_steps(
        ch_steps,
        step_concentrations=step_concentrations,
    )
    y = np.asarray([row["plateau_value"] for row in ch_steps], dtype=float)
    if fit_axis_kind != "concentration":
        return x, y, fit_axis_kind, list(ch_steps)

    grouped: Dict[float, List[Tuple[float, dict]]] = {}
    for x_value, y_value, row in zip(x, y, ch_steps):
        if np.isfinite(x_value) and np.isfinite(y_value):
            grouped.setdefault(float(x_value), []).append((float(y_value), row))
    sorted_x = sorted(grouped)
    fit_y = []
    representative_rows = []
    for x_value in sorted_x:
        values_and_rows = grouped[x_value]
        values = np.asarray([item[0] for item in values_and_rows], dtype=float)
        median_value = float(np.median(values))
        fit_y.append(median_value)
        representative_rows.append(
            min(values_and_rows, key=lambda item: abs(item[0] - median_value))[1]
        )
    return (
        np.asarray(sorted_x, dtype=float),
        np.asarray(fit_y, dtype=float),
        fit_axis_kind,
        representative_rows,
    )


def _fixed_baseline_from_steps(ch_steps: List[dict]) -> Optional[float]:
    for row in ch_steps:
        try:
            value = float(row.get("fixed_langmuir_baseline"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _langmuir_target_steps(ch_steps: List[dict]) -> List[dict]:
    return [
        row
        for row in ch_steps
        if str(row.get("step_note", "")).strip().lower() != "buffer"
        and row.get("step_concentration") is not None
        and row.get("fixed_langmuir_baseline") is not None
    ]


def build_titration_langmuir_summary_table(
    all_results: List[dict],
    metric: str,
    vlines: Optional[List[Tuple[float, str]]],
    channels: Optional[List[Any]] = None,
    vlines_by_channel: Optional[Dict[Any, List[Tuple[float, str]]]] = None,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
    edge_trim_fraction: float = 0.15,
    step_concentrations: Optional[List[float]] = None,
    concentration_unit: str = "",
    baseline_mode: str = "none",
    included_step_labels: Optional[List[str]] = None,
    remove_extreme_outliers: bool = False,
) -> List[dict]:
    step_rows = build_titration_step_table(
        all_results,
        metric=metric,
        vlines=vlines,
        channels=channels,
        vlines_by_channel=vlines_by_channel,
        scan_windows=scan_windows,
        scan_range=scan_range,
        edge_trim_fraction=edge_trim_fraction,
        step_concentrations=step_concentrations,
        concentration_unit=concentration_unit,
        baseline_mode=baseline_mode,
        included_step_labels=included_step_labels,
        remove_extreme_outliers=remove_extreme_outliers,
    )
    if not step_rows:
        return []

    rows: List[dict] = []
    for ch in sorted({row["channel"] for row in step_rows}):
        ch_steps = sorted(
            [row for row in step_rows if row["channel"] == ch],
            key=lambda row: row["step_index"],
        )
        fit_source_steps = _langmuir_target_steps(ch_steps)
        if len(fit_source_steps) < 2:
            continue

        x, y, fit_axis_kind, fit_steps = _prepare_titration_fit_points(
            fit_source_steps,
            step_concentrations=step_concentrations,
        )
        fixed_baseline = _fixed_baseline_from_steps(fit_source_steps)
        if fixed_baseline is None:
            continue
        hybrid_fit = _build_langmuir_hybrid_fit(
            x,
            y,
            fixed_baseline=fixed_baseline,
        )
        if hybrid_fit is None:
            continue

        saturation_idx = hybrid_fit["saturation_idx"]
        saturation_step = fit_steps[saturation_idx]
        langmuir_params = hybrid_fit["langmuir_params"]
        langmuir_covariance = hybrid_fit.get("langmuir_covariance")
        post_sat_poly = hybrid_fit["post_sat_poly"]
        limit_of_detection, blank_sigma = (
            _langmuir_limit_of_detection(ch_steps, langmuir_params)
            if fit_axis_kind == "concentration"
            else (None, None)
        )
        uloq_sigma, uloq_noise_source = _high_concentration_response_noise(
            fit_source_steps,
            blank_sigma,
        )
        upper_limit_of_quantification = (
            _langmuir_upper_limit_of_quantification(
                langmuir_params,
                uloq_sigma,
            )
            if fit_axis_kind == "concentration"
            else None
        )
        snr_cutoff_concentration = (
            _langmuir_snr_cutoff_concentration(
                langmuir_params,
                blank_sigma,
            )
            if fit_axis_kind == "concentration"
            else None
        )

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
        if post_sat_poly is not None and saturation_idx < (len(fit_steps) - 1):
            _, post_sat_poly_degree = post_sat_poly
            if langmuir_params is not None and fit_axis_kind == "concentration":
                fit_status = "langmuir_plus_post_sat_poly"
            elif langmuir_params is not None:
                fit_status = "step_index_fit_plus_post_sat_poly_no_kd"
            else:
                fit_status = "guide_plus_post_sat_poly"

        rows.append({
            "channel": ch,
            "original_channel": ch_steps[0].get("original_channel", ch),
            "metric_key": metric,
            "fit_axis": "concentration" if fit_axis_kind == "concentration" else "titration_step_index",
            "fit_axis_unit": concentration_unit if fit_axis_kind == "concentration" else "",
            "fit_axis_note": "physical_concentration" if fit_axis_kind == "concentration" else "no_physical_kd",
            "step_count": int(len(fit_source_steps)),
            "fit_point_count": int(len(fit_steps)),
            "pre_saturation_step_count": int(saturation_idx + 1),
            "post_saturation_step_count": int(len(fit_steps) - saturation_idx - 1),
            "saturation_step_index": float(saturation_step["step_index"]),
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
            "langmuir_response_direction": (
                "signal-off"
                if amplitude is not None and amplitude < 0
                else ("signal-on" if amplitude is not None else "")
            ),
            "langmuir_kd": kd,
            "langmuir_amplitude_variance": (
                float(langmuir_covariance[1, 1])
                if langmuir_covariance is not None
                and langmuir_covariance.shape == (3, 3)
                and np.isfinite(langmuir_covariance[1, 1])
                else None
            ),
            "langmuir_kd_variance": (
                float(langmuir_covariance[2, 2])
                if langmuir_covariance is not None
                and langmuir_covariance.shape == (3, 3)
                and np.isfinite(langmuir_covariance[2, 2])
                else None
            ),
            "langmuir_amplitude_kd_covariance": (
                float(langmuir_covariance[1, 2])
                if langmuir_covariance is not None
                and langmuir_covariance.shape == (3, 3)
                and np.isfinite(langmuir_covariance[1, 2])
                else None
            ),
            "langmuir_kd_unit": concentration_unit if kd is not None else "",
            "limit_of_detection": limit_of_detection,
            "limit_of_detection_unit": concentration_unit if limit_of_detection is not None else "",
            "limit_of_detection_method": "3σ blank / Langmuir initial slope" if limit_of_detection is not None else "",
            "upper_limit_of_quantification": upper_limit_of_quantification,
            "upper_limit_of_quantification_unit": (
                concentration_unit
                if upper_limit_of_quantification is not None else ""
            ),
            "upper_limit_of_quantification_method": (
                "Fitted response within 3σ response noise of Langmuir saturation"
                if upper_limit_of_quantification is not None else ""
            ),
            "upper_limit_of_quantification_noise_sigma": uloq_sigma,
            "upper_limit_of_quantification_noise_source": uloq_noise_source,
            "upper_limit_of_quantification_is_extrapolated": bool(
                upper_limit_of_quantification is not None
                and x.size
                and upper_limit_of_quantification > float(np.nanmax(x))
            ),
            "blank_sigma": blank_sigma,
            "snr_3_cutoff_concentration": snr_cutoff_concentration,
            "snr_3_cutoff_concentration_unit": (
                concentration_unit if snr_cutoff_concentration is not None else ""
            ),
            "baseline_mode": baseline_mode,
            "langmuir_baseline_fixed": bool(fixed_baseline is not None),
            "first_buffer_step_index": (
                fit_source_steps[0].get("first_buffer_step_index")
                if fixed_baseline is not None else None
            ),
            "anchor_buffer_step_index": (
                fit_source_steps[0].get("anchor_buffer_step_index")
                if fixed_baseline is not None else None
            ),
            "post_saturation_polynomial_degree": post_sat_poly_degree,
        })

    return rows


def _invert_langmuir_response(
    response: float,
    baseline: float,
    amplitude: float,
    kd: float,
) -> Optional[float]:
    if not all(np.isfinite(value) for value in (response, baseline, amplitude, kd)):
        return None
    if np.isclose(amplitude, 0.0) or kd <= 0:
        return None
    occupancy = (response - baseline) / amplitude
    # Responses below B map to the negative-concentration continuation of the
    # fitted Langmuir curve. Retain those finite estimates instead of clipping
    # them to zero; only the saturation singularity/upper branch is invalid.
    if not np.isfinite(occupancy) or occupancy >= 1:
        return None
    return float(kd * occupancy / (1.0 - occupancy))


def _propagated_langmuir_concentration_std(
    response: float,
    baseline: float,
    amplitude: float,
    kd: float,
    response_sigma: Optional[float],
    amplitude_variance: Optional[float] = None,
    kd_variance: Optional[float] = None,
    amplitude_kd_covariance: Optional[float] = None,
) -> Optional[float]:
    """Delta-method 1σ uncertainty for concentration inferred by Langmuir inversion."""
    predicted = _invert_langmuir_response(response, baseline, amplitude, kd)
    if predicted is None:
        return None
    delta_response = float(response) - float(baseline)
    denominator = float(amplitude) - delta_response
    if np.isclose(denominator, 0.0) or not np.isfinite(denominator):
        return None

    variance = 0.0
    used_component = False
    try:
        response_sigma_value = float(response_sigma)
    except (TypeError, ValueError):
        response_sigma_value = np.nan
    if np.isfinite(response_sigma_value) and response_sigma_value > 0:
        derivative_response = float(kd) * float(amplitude) / denominator ** 2
        variance += (derivative_response * response_sigma_value) ** 2
        used_component = True

    derivative_amplitude = -float(kd) * delta_response / denominator ** 2
    derivative_kd = delta_response / denominator
    covariance_values = (
        amplitude_variance,
        kd_variance,
        amplitude_kd_covariance,
    )
    try:
        amplitude_variance_value, kd_variance_value, covariance_value = (
            float(value) for value in covariance_values
        )
    except (TypeError, ValueError):
        amplitude_variance_value = kd_variance_value = covariance_value = np.nan
    if np.isfinite(amplitude_variance_value) and amplitude_variance_value >= 0:
        variance += derivative_amplitude ** 2 * amplitude_variance_value
        used_component = True
    if np.isfinite(kd_variance_value) and kd_variance_value >= 0:
        variance += derivative_kd ** 2 * kd_variance_value
        used_component = True
    if np.isfinite(covariance_value):
        variance += (
            2.0 * derivative_amplitude * derivative_kd * covariance_value
        )

    if not used_component or not np.isfinite(variance):
        return None
    # A small negative value can occur from floating-point cancellation in the
    # covariance cross-term; a materially negative variance is invalid.
    if variance < 0:
        if variance >= -1e-12:
            variance = 0.0
        else:
            return None
    return float(np.sqrt(variance))


def build_titration_measurement_accuracy_table(
    all_results: List[dict],
    metric: str,
    vlines: Optional[List[Tuple[float, str]]],
    channels: Optional[List[Any]] = None,
    vlines_by_channel: Optional[Dict[Any, List[Tuple[float, str]]]] = None,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
    edge_trim_fraction: float = 0.15,
    concentration_unit: str = "",
    baseline_mode: str = "none",
    included_step_labels: Optional[List[str]] = None,
    remove_extreme_outliers: bool = False,
    include_buffer_measurements: bool = False,
) -> List[dict]:
    """Invert fitted Langmuir curves for every selected target SWV."""
    step_rows = build_titration_step_table(
        all_results,
        metric=metric,
        vlines=vlines,
        channels=channels,
        vlines_by_channel=vlines_by_channel,
        scan_windows=scan_windows,
        scan_range=scan_range,
        edge_trim_fraction=edge_trim_fraction,
        concentration_unit=concentration_unit,
        baseline_mode=baseline_mode,
        included_step_labels=included_step_labels,
        remove_extreme_outliers=remove_extreme_outliers,
    )
    if include_buffer_measurements and baseline_mode == "preceding_buffer":
        uncorrected_steps = build_titration_step_table(
            all_results,
            metric=metric,
            vlines=vlines,
            channels=channels,
            vlines_by_channel=vlines_by_channel,
            scan_windows=scan_windows,
            scan_range=scan_range,
            edge_trim_fraction=edge_trim_fraction,
            concentration_unit=concentration_unit,
            baseline_mode="none",
            included_step_labels=included_step_labels,
            remove_extreme_outliers=remove_extreme_outliers,
        )
        step_rows = step_rows + [
            row for row in uncorrected_steps
            if str(row.get("step_note", "")).strip().lower() == "buffer"
        ]
    summaries = build_titration_langmuir_summary_table(
        all_results,
        metric=metric,
        vlines=vlines,
        channels=channels,
        vlines_by_channel=vlines_by_channel,
        scan_windows=scan_windows,
        scan_range=scan_range,
        edge_trim_fraction=edge_trim_fraction,
        concentration_unit=concentration_unit,
        baseline_mode=baseline_mode,
        included_step_labels=included_step_labels,
        remove_extreme_outliers=remove_extreme_outliers,
    )
    summary_by_channel = {
        row["channel"]: row
        for row in summaries
        if row.get("langmuir_fit_used")
    }
    measurement_results = (
        filter_extreme_titration_outliers(
            all_results,
            metric=metric,
            vlines=vlines,
            channels=channels,
            vlines_by_channel=vlines_by_channel,
        )
        if remove_extreme_outliers else all_results
    )
    rows: List[dict] = []
    for step in step_rows:
        known = step.get("step_concentration")
        summary = summary_by_channel.get(step["channel"])
        is_buffer = (
            str(step.get("step_note", "")).strip().lower() == "buffer"
            or known == 0
        )
        if (
            known is None
            or known < 0
            or (known == 0 and not include_buffer_measurements)
            or summary is None
        ):
            continue
        baseline = summary.get("langmuir_baseline")
        amplitude = summary.get("langmuir_amplitude")
        kd = summary.get("langmuir_kd")
        if baseline is None or amplitude is None or kd is None:
            continue
        channel_measurements = [
            result
            for result in measurement_results
            if result.get("channel") == step["channel"]
            and result.get("status") == "OK"
            and step["step_start_scan"] <= result.get("scan_number", np.nan) < step["step_end_scan"]
            and np.isfinite(result.get(metric, np.nan))
        ]
        for result in channel_measurements:
            raw_response = float(result[metric])
            corrected_response = raw_response
            if baseline_mode == "preceding_buffer" and not is_buffer:
                corrected_response = (
                    raw_response
                    - float(step["baseline_value"])
                    + float(step["fixed_langmuir_baseline"])
                )
            predicted = _invert_langmuir_response(
                corrected_response,
                float(baseline),
                float(amplitude),
                float(kd),
            )
            if predicted is None and is_buffer:
                occupancy = (
                    (corrected_response - float(baseline)) / float(amplitude)
                    if not np.isclose(float(amplitude), 0.0) else np.nan
                )
                if np.isfinite(occupancy) and occupancy <= 0:
                    predicted = 0.0
            unbounded_predicted = predicted
            limit_of_detection = summary.get("limit_of_detection")
            try:
                finite_lod = float(limit_of_detection)
            except (TypeError, ValueError):
                finite_lod = None
            if finite_lod is not None and not np.isfinite(finite_lod):
                finite_lod = None
            concentration_censored_at_lod = bool(
                predicted is not None
                and finite_lod is not None
                and float(predicted) < finite_lod
            )
            if concentration_censored_at_lod:
                predicted = finite_lod
            absolute_error = (
                abs(predicted - float(known)) if predicted is not None else None
            )
            percent_error = (
                100.0 * absolute_error / float(known)
                if absolute_error is not None and known > 0 else None
            )
            signed_percent_error = (
                100.0 * (predicted - float(known)) / float(known)
                if predicted is not None and known > 0 else None
            )
            log10_error = (
                float(np.log10(predicted) - np.log10(float(known)))
                if predicted is not None and predicted > 0 and known > 0 else None
            )
            noise_std = summary.get("blank_sigma")
            measurement_snr = (
                abs(corrected_response - float(baseline)) / float(noise_std)
                if noise_std is not None and float(noise_std) > 0 else None
            )
            uncertainty_response = (
                float(baseline)
                if is_buffer and unbounded_predicted == 0.0
                else corrected_response
            )
            unbounded_predicted_concentration_std = _propagated_langmuir_concentration_std(
                uncertainty_response,
                float(baseline),
                float(amplitude),
                float(kd),
                noise_std,
                amplitude_variance=summary.get("langmuir_amplitude_variance"),
                kd_variance=summary.get("langmuir_kd_variance"),
                amplitude_kd_covariance=summary.get(
                    "langmuir_amplitude_kd_covariance"
                ),
            )
            predicted_concentration_std = (
                None
                if concentration_censored_at_lod
                else unbounded_predicted_concentration_std
            )
            concentration_lower_1sigma = (
                float(predicted) - predicted_concentration_std
                if predicted is not None and predicted_concentration_std is not None
                else None
            )
            concentration_upper_1sigma = (
                float(predicted) + predicted_concentration_std
                if predicted is not None and predicted_concentration_std is not None
                else None
            )
            rows.append({
                "channel": step["channel"],
                "original_channel": step.get("original_channel", step["channel"]),
                "metric_key": metric,
                "scan_number": result.get("scan_number"),
                "source_scan_number": result.get("display_group_source_scan_number", result.get("scan_number")),
                "file_name": result.get("file_name"),
                "measurement_time": result.get("measurement_time"),
                "step_index": step["step_index"],
                "step_selection_key": step.get("step_selection_key"),
                "known_concentration": float(known),
                "predicted_concentration": predicted,
                "unbounded_predicted_concentration": unbounded_predicted,
                "concentration_censored_at_lod": concentration_censored_at_lod,
                "predicted_concentration_std": predicted_concentration_std,
                "unbounded_predicted_concentration_std": (
                    unbounded_predicted_concentration_std
                ),
                "predicted_concentration_lower_1sigma": concentration_lower_1sigma,
                "predicted_concentration_upper_1sigma": concentration_upper_1sigma,
                "predicted_concentration_uncertainty_method": (
                    "Delta-method propagation of selected-buffer response σ and Langmuir A/Kd covariance"
                    if predicted_concentration_std is not None
                    else (
                        "Below LOD; reported at LOD and symmetric uncertainty censored"
                        if concentration_censored_at_lod else None
                    )
                ),
                "concentration_unit": concentration_unit,
                "absolute_concentration_error": absolute_error,
                "signed_percent_error": signed_percent_error,
                "absolute_percent_error": percent_error,
                "log10_concentration_error": log10_error,
                "raw_metric_value": raw_response,
                "fit_metric_value": corrected_response,
                "measurement_snr": measurement_snr,
                "plateau_snr": step.get("titration_snr"),
                "fit_baseline": baseline,
                "fit_amplitude": amplitude,
                "fit_kd": kd,
                "limit_of_detection": limit_of_detection,
                "upper_limit_of_quantification": summary.get(
                    "upper_limit_of_quantification"
                ),
                "upper_limit_of_quantification_is_extrapolated": summary.get(
                    "upper_limit_of_quantification_is_extrapolated",
                    False,
                ),
                "baseline_mode": baseline_mode,
                "anchor_buffer_step_index": step.get("anchor_buffer_step_index"),
                "drift_buffer_step_index": step.get("baseline_step_index"),
            })
    return (
        _filter_extreme_accuracy_predictions(rows)
        if remove_extreme_outliers else rows
    )


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

    normalize_to_peak: bool = False,

    offset_to_baseline: bool = False,

) -> Optional[plt.Figure]:
    usable = [r for r in results if r.get(y_key) is not None and r.get("voltage") is not None]

    if offset_to_baseline:
        offset_usable = []
        for r in usable:
            try:
                _offset_trace_to_anchor_baseline(
                    np.asarray(r[y_key], dtype=float),
                    r.get("left_min_idx"),
                    r.get("right_min_idx"),
                )
            except (TypeError, ValueError):
                continue
            offset_usable.append(r)
        usable = offset_usable

    if normalize_to_peak:
        peak_idx_key = (
            "peak_idx_corr"
            if y_key in ("corrected_current", "smoothed_corrected_current")
            else "peak_idx"
        )
        filtered = []
        for r in usable:
            peak_idx = r.get(peak_idx_key)
            y = np.asarray(r.get(y_key), dtype=float)
            if peak_idx is None or not 0 <= peak_idx < len(y):
                continue
            peak_height = float(y[peak_idx])
            if np.isfinite(peak_height) and not np.isclose(peak_height, 0.0):
                filtered.append(r)
        usable = filtered

    if not usable:

        return None

    return _cmap_fig(usable, y_key, title, ylabel, colormap_name, linewidth, alpha,

                     show_anchors=show_anchors,

                     show_peak_markers=show_peak_markers,

                     show_zero_baseline=show_zero_baseline,

                     show_local_baselines=show_local_baselines,

                     show_minima_candidates=show_minima_candidates,

                     normalize_to_peak=normalize_to_peak,

                     offset_to_baseline=offset_to_baseline)


def plot_grouped_overlaid_traces(
    grouped_results: List[Tuple[str, List[dict], str]],
    y_key: str = "corrected_current",
    title: str = "Grouped Overlaid Traces",
    ylabel: str = "Current (uA)",
    linewidth: float = 0.9,
    alpha: float = 0.85,
    show_anchors: bool = False,
    show_peak_markers: bool = False,
    show_zero_baseline: bool = False,
    normalize_to_peak: bool = False,
    offset_to_baseline: bool = False,
    colorbar_height_fraction: float = 0.85,
    colorbar_side: str = "right",
    show_legend: bool = True,
    show_grid: bool = False,
    outer_margin_fraction: float = 0.04,
) -> Optional[plt.Figure]:
    """Overlay multiple SWV groups, using a separate time-gradient colormap per group."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if show_zero_baseline:
        ax.axhline(0, color="gray", lw=1.0, linestyle="--", alpha=0.8)

    legend_handles = []
    group_colorbars = []
    plotted_trace_count = 0
    for group_number, (group_label, results, colormap_name) in enumerate(
        grouped_results,
        start=1,
    ):
        usable = [
            row for row in results
            if row.get(y_key) is not None and row.get("voltage") is not None
        ]
        if not usable:
            continue
        cmap = plt.get_cmap(colormap_name, max(len(usable), 2))
        norm = Normalize(vmin=1, vmax=max(len(usable), 2))
        group_trace_count = 0
        for index, row in enumerate(usable, start=1):
            voltage = np.asarray(row["voltage"], dtype=float)
            y_plot = np.asarray(row[y_key], dtype=float)
            try:
                if offset_to_baseline:
                    y_plot = _offset_trace_to_anchor_baseline(
                        y_plot,
                        row.get("left_min_idx"),
                        row.get("right_min_idx"),
                    )
                peak_idx_key = (
                    "peak_idx_corr"
                    if y_key in ("corrected_current", "smoothed_corrected_current")
                    else "peak_idx"
                )
                peak_idx = row.get(peak_idx_key)
                if normalize_to_peak:
                    if peak_idx is None or not 0 <= int(peak_idx) < len(y_plot):
                        continue
                    peak_height = float(y_plot[int(peak_idx)])
                    if not np.isfinite(peak_height) or np.isclose(peak_height, 0.0):
                        continue
                    y_plot = y_plot / peak_height
            except (TypeError, ValueError):
                continue

            color = cmap(norm(index))
            ax.plot(voltage, y_plot, color=color, lw=linewidth, alpha=alpha)
            group_trace_count += 1
            plotted_trace_count += 1

            if show_anchors and (y_key == "corrected_current" or offset_to_baseline):
                for idx_key in ("left_min_idx", "right_min_idx"):
                    anchor_idx = row.get(idx_key)
                    if anchor_idx is not None and 0 <= int(anchor_idx) < len(voltage):
                        ax.scatter(
                            voltage[int(anchor_idx)],
                            y_plot[int(anchor_idx)],
                            color=color,
                            s=18,
                            zorder=5,
                            edgecolors="white",
                            linewidths=0.5,
                        )

            if show_peak_markers and peak_idx is not None and 0 <= int(peak_idx) < len(voltage):
                ax.scatter(
                    voltage[int(peak_idx)],
                    y_plot[int(peak_idx)],
                    color=color,
                    s=28,
                    zorder=6,
                    edgecolors="white",
                    linewidths=0.8,
                )

        if group_trace_count:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                color=cmap(0.65),
                    lw=3,
                    label=f"G{group_number} · {group_label} ({group_trace_count})",
                )
            )
            scalar_mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
            scalar_mappable.set_array([])
            group_colorbars.append(
                (scalar_mappable, group_number, group_trace_count)
            )

    if not plotted_trace_count:
        plt.close(fig)
        return None

    ax.set_title(title)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(ylabel)
    if normalize_to_peak:
        ax.set_ylim(-0.2, 1.2)
    if legend_handles and show_legend:
        ax.legend(
            handles=legend_handles,
            title="SWV group",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=min(2, len(legend_handles)),
            fontsize=8,
        )
    if group_colorbars:
        colorbar_side = "left" if str(colorbar_side).lower() == "left" else "right"
        bar_count = len(group_colorbars)
        bar_width = 0.018
        bar_gap = 0.045
        reserved_width = 0.035 + bar_count * (bar_width + bar_gap)
        outer_margin = min(0.20, max(0.01, float(outer_margin_fraction)))
        bottom_margin = max(
            outer_margin,
            0.22 if show_legend and legend_handles else 0.12,
        )
        if colorbar_side == "right":
            fig.subplots_adjust(
                left=max(0.08, outer_margin),
                right=max(0.48, 1.0 - outer_margin - reserved_width),
                bottom=bottom_margin,
                top=min(0.96, 1.0 - outer_margin),
            )
        else:
            fig.subplots_adjust(
                left=min(0.52, outer_margin + reserved_width),
                right=min(0.96, 1.0 - outer_margin),
                bottom=bottom_margin,
                top=min(0.96, 1.0 - outer_margin),
            )
        main_position = ax.get_position()
        height_fraction = min(1.0, max(0.2, float(colorbar_height_fraction)))
        bar_height = main_position.height * height_fraction
        bar_bottom = main_position.y0 + (main_position.height - bar_height) / 2.0
        for bar_index, (scalar_mappable, group_number, group_trace_count) in enumerate(
            group_colorbars
        ):
            if colorbar_side == "right":
                bar_left = (
                    main_position.x1
                    + 0.020
                    + bar_index * (bar_width + bar_gap)
                )
            else:
                bar_left = (
                    main_position.x0
                    - 0.020
                    - bar_width
                    - bar_index * (bar_width + bar_gap)
                )
            colorbar_axis = fig.add_axes([
                bar_left,
                bar_bottom,
                bar_width,
                bar_height,
            ])
            colorbar_axis._swv_colorbar_axis = True
            colorbar = fig.colorbar(scalar_mappable, cax=colorbar_axis)
            colorbar.ax.set_title(f"G{group_number}", fontsize=8, pad=4)
            if group_trace_count == 1:
                colorbar.set_ticks([1])
                colorbar.set_ticklabels(["1"])
            else:
                colorbar.set_ticks([1, group_trace_count])
                colorbar.set_ticklabels(["1", str(group_trace_count)])
            colorbar.set_label(
                "SWV Measurement Number",
                fontsize=7,
                labelpad=2,
            )
            colorbar.ax.tick_params(labelsize=7)
    if show_grid:
        ax.grid(True, alpha=0.2)
    else:
        ax.grid(False)
    fig._swv_manual_layout = True
    return fig





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

    channels: Optional[List[Any]] = None,

    title: Optional[str] = None,

    ylabel: Optional[str] = None,

    vlines: Optional[List[Tuple[float, str]]] = None,

    vline_y_frac: float = 0.85,

    scan_range: Optional[Tuple[int, int]] = None,

    figsize: Tuple[int, int] = (10, 4),
    xlabel: str = "Scan number",

    x_key: str = "scan_number",

    highlight_channel: Optional[Any] = None,

    normalize_per_channel: bool = False,

    percent_change_per_channel: bool = False,

    response_directions: Optional[Dict[Any, str]] = None,

    response_baselines: Optional[Dict[Any, float]] = None,

    offset_to_response_baseline: bool = False,

    response_direction_colors_only: bool = False,

    channel_colors: Optional[Dict[Any, Any]] = None,

) -> Optional[plt.Figure]:

    if normalize_per_channel and percent_change_per_channel:

        raise ValueError("Choose only one per-channel normalization mode.")

    all_ch = sorted({r["channel"] for r in all_results}, key=_channel_sort_key)

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

    if x_key != "scan_number" and filtered_vlines:

        times_by_scan: Dict[float, List[float]] = {}

        for row in plot_results:

            if row.get("scan_number") is None or row.get(x_key) is None:

                continue

            scan_number = float(row["scan_number"])

            times_by_scan.setdefault(scan_number, []).append(

                mdates.date2num(row[x_key])

            )

        ordered_scans = sorted(times_by_scan)

        if ordered_scans:

            ordered_times = [

                float(np.median(times_by_scan[scan])) for scan in ordered_scans

            ]

            filtered_vlines = [

                (

                    mdates.num2date(

                        float(np.interp(x, ordered_scans, ordered_times))

                    ).replace(tzinfo=None),

                    label,

                )

                for x, label in filtered_vlines

                if ordered_scans[0] <= x <= ordered_scans[-1]

            ]

        else:

            filtered_vlines = []



    normalized_directions = {
        channel: str(direction).strip().lower()
        for channel, direction in (response_directions or {}).items()
        if str(direction).strip().lower() in {"signal-on", "signal-off"}
    }
    use_direction_colors = (
        not percent_change_per_channel
        and all(channel in normalized_directions for channel in channels)
    )
    plotted_response_directions = {
        normalized_directions[channel] for channel in channels
    } if use_direction_colors else set()
    use_direction_axes = (
        use_direction_colors
        and not offset_to_response_baseline
        and not response_direction_colors_only
        and plotted_response_directions == {"signal-on", "signal-off"}
    )
    if use_direction_colors:
        signal_on_channels = [
            channel for channel in channels
            if normalized_directions[channel] == "signal-on"
        ]
        signal_off_channels = [
            channel for channel in channels
            if normalized_directions[channel] == "signal-off"
        ]
        signal_on_shades = dict(zip(
            signal_on_channels,
            _high_contrast_response_shades(len(signal_on_channels)),
        ))
        signal_off_shades = dict(zip(
            signal_off_channels,
            _high_contrast_response_shades(len(signal_off_channels)),
        ))
        colors = {
            channel: plt.get_cmap("Oranges")(
                signal_on_shades[channel]
            )
            for channel in signal_on_channels
        }
        colors.update({
            channel: plt.get_cmap("Blues")(
                signal_off_shades[channel]
            )
            for channel in signal_off_channels
        })
        trace_style_by_channel = {
            channel: ("-", "o") for channel in channels
        }
    else:
        cmap = plt.get_cmap("tab10")
        colors = {ch: cmap(i % 10) for i, ch in enumerate(all_ch)}
        trace_style_by_channel = {
            channel: ("-", "o") for channel in channels
        }
    colors.update({
        channel: color
        for channel, color in (channel_colors or {}).items()
        if channel in channels
    })
    # Displayed SWV methods use method identity—not physical channel, signal
    # direction, or a caller palette—for color. This keeps Method 1 and Method 2
    # recognizable in every channel plot.
    colors.update({
        channel: method_color
        for channel in channels
        if (method_color := _swv_method_blue(channel)) is not None
    })



    fig, ax = plt.subplots(figsize=figsize)

    signal_off_ax = ax.twinx() if use_direction_axes else None
    plotted_y_by_axis: Dict[Any, List[np.ndarray]] = {ax: []}
    if signal_off_ax is not None:
        plotted_y_by_axis[signal_off_ax] = []
    if use_direction_axes:
        ax._swv_response_direction = "signal-on"
        signal_off_ax._swv_response_direction = "signal-off"

    for ch in channels:

        ch_res = sorted([r for r in plot_results if r["channel"] == ch and r.get(x_key) is not None],

                        key=lambda r: r[x_key])

        if not ch_res:

            continue

        x = [r[x_key] for r in ch_res]

        y = np.asarray([r.get(metric, np.nan) for r in ch_res], dtype=float)

        if offset_to_response_baseline:
            try:
                response_baseline = float((response_baselines or {}).get(ch))
            except (TypeError, ValueError):
                response_baseline = np.nan
            finite = np.isfinite(y)
            if not np.isfinite(response_baseline) and finite.any():
                response_baseline = float(y[np.flatnonzero(finite)[0]])
            if np.isfinite(response_baseline):
                y[finite] = y[finite] - response_baseline

        if normalize_per_channel:

            finite = np.isfinite(y)

            if finite.any():

                channel_min = float(np.min(y[finite]))

                channel_max = float(np.max(y[finite]))

                channel_span = channel_max - channel_min

                if np.isclose(channel_span, 0.0):

                    y[finite] = 0.0

                else:

                    y[finite] = (y[finite] - channel_min) / channel_span
                    if (
                        use_direction_colors
                        and normalized_directions[ch] == "signal-off"
                    ):
                        y[finite] = y[finite] - 1.0

        elif percent_change_per_channel:

            finite = np.isfinite(y)

            if finite.any():

                first_value = float(y[np.flatnonzero(finite)[0]])

                if np.isclose(first_value, 0.0):

                    y[finite] = np.nan

                else:

                    y[finite] = ((y[finite] - first_value) / abs(first_value)) * 100.0

        dimmed = highlight_channel is not None and ch != highlight_channel

        plot_axis = (
            signal_off_ax
            if use_direction_axes and normalized_directions[ch] == "signal-off"
            else ax
        )
        method_trace_label = _swv_method_trace_label(ch)
        direction_suffix = (
            f" ({normalized_directions[ch]})"
            if use_direction_colors and method_trace_label is None
            else ""
        )

        line_style, marker = trace_style_by_channel[ch]
        line, = plot_axis.plot(x, y, marker=marker, linestyle=line_style, ms=3, lw=1.6,

                color=colors[ch],

                alpha=0.15 if dimmed else 0.9,

                label=(
                    method_trace_label
                    or f"{_compact_channel_label(ch)}{direction_suffix}"
                ))
        line._swv_preserve_color = _swv_method_blue(ch) is not None
        plotted_y_by_axis[plot_axis].append(y[np.isfinite(y)])



    ax.set_xlabel(xlabel)
    ax._swv_force_black_y_axis = True
    if signal_off_ax is not None:
        signal_off_ax._swv_force_black_y_axis = True

    if use_direction_axes:
        axis_label = ylabel or metric
        ax.set_ylabel(f"{axis_label} (signal-on)", color="black")
        signal_off_ax.set_ylabel(
            f"{axis_label} (signal-off)",
            color="black",
        )
        ax.tick_params(axis="y", colors="black")
        signal_off_ax.tick_params(axis="y", colors="black")
        ax.spines["left"].set_color("black")
        signal_off_ax.spines["right"].set_color("black")
    elif use_direction_colors and not response_direction_colors_only:
        axis_label = ylabel or metric
        ax.set_ylabel(axis_label, color="black")
        ax.tick_params(axis="y", colors="black")
        ax.spines["left"].set_color("black")
    else:
        ax.set_ylabel(ylabel or metric)

    # Pad each active metric axis by exactly 10% of its plotted data span.
    # Twin response-direction axes are intentionally calculated independently.
    for plot_axis, value_groups in plotted_y_by_axis.items():
        finite_groups = [values for values in value_groups if values.size]
        if not finite_groups:
            continue
        finite_values = np.concatenate(finite_groups)
        data_min = float(np.min(finite_values))
        data_max = float(np.max(finite_values))
        data_span = data_max - data_min
        if np.isclose(data_span, 0.0):
            padding = max(abs(data_min) * 0.1, 0.1)
        else:
            padding = data_span * 0.1
        plot_axis.set_ylim(data_min - padding, data_max + padding)

    ax.set_title(title or f"{metric} vs Scan")

    if x_key != "scan_number":

        locator = mdates.AutoDateLocator()

        ax.xaxis.set_major_locator(locator)

        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    ax.grid(False)

    if use_direction_axes:
        left_handles, left_labels = ax.get_legend_handles_labels()
        right_handles, right_labels = signal_off_ax.get_legend_handles_labels()
        ax.legend(
            left_handles + right_handles,
            left_labels + right_labels,
            loc="best",
            fontsize=8,
        )
    elif use_direction_colors:
        legend_location = (
            "center left"
            if plotted_response_directions == {"signal-on"}
            else "best"
        )
        ax.legend(
            loc=legend_location,
            fontsize=8,
        )
    else:
        ax.legend(title="Channel", loc="best", fontsize=8)

    add_scan_vlines(ax, filtered_vlines, vline_y_frac)

    if offset_to_response_baseline:
        ax.axhline(0.0, color="gray", lw=0.9, alpha=0.6, zorder=1)

    if scan_range and x_key == "scan_number":

        ax.set_xlim(scan_range)

    fig.tight_layout()

    return fig





def plot_titration_plateaus(

    all_results: List[dict],

    metric: str,

    vlines: Optional[List[Tuple[float, str]]],

    channels: Optional[List[Any]] = None,
    vlines_by_channel: Optional[Dict[Any, List[Tuple[float, str]]]] = None,

    title: Optional[str] = None,

    ylabel: Optional[str] = None,

    scan_windows: Optional[List[Tuple[int, int]]] = None,

    scan_range: Optional[Tuple[int, int]] = None,

    edge_trim_fraction: float = 0.15,

    vline_y_frac: float = 0.85,

    figsize: Tuple[int, int] = (10, 4),

    highlight_channel: Optional[int] = None,
    baseline_mode: str = "none",
    included_step_labels: Optional[List[str]] = None,
    remove_extreme_outliers: bool = False,
    response_directions: Optional[Dict[Any, str]] = None,
    response_baselines: Optional[Dict[Any, float]] = None,
    offset_to_response_baseline: bool = False,
    channel_colors: Optional[Dict[Any, Any]] = None,

) -> Optional[plt.Figure]:

    step_rows = build_titration_step_table(

        all_results,

        metric=metric,

        vlines=vlines,

        channels=channels,
        vlines_by_channel=vlines_by_channel,

        scan_windows=scan_windows,

        scan_range=scan_range,

        edge_trim_fraction=edge_trim_fraction,
        baseline_mode=baseline_mode,
        included_step_labels=included_step_labels,
        remove_extreme_outliers=remove_extreme_outliers,

    )
    if not step_rows:

        return None

    all_ch = sorted({r["channel"] for r in all_results})
    channels = sorted({row["channel"] for row in step_rows})
    source_plot_results = (
        filter_extreme_titration_outliers(
            all_results,
            metric=metric,
            vlines=vlines,
            channels=channels,
            vlines_by_channel=vlines_by_channel,
        )
        if remove_extreme_outliers else all_results
    )
    plot_results = (

        [r for r in source_plot_results if scan_range[0] <= r["scan_number"] <= scan_range[1]]

        if scan_range else source_plot_results

    )
    filtered_vlines = _filter_titration_vlines(vlines, scan_range=scan_range)

    normalized_directions = {
        channel: str(direction).strip().lower()
        for channel, direction in (response_directions or {}).items()
        if str(direction).strip().lower() in {"signal-on", "signal-off"}
    }
    use_direction_colors = all(
        channel in normalized_directions for channel in channels
    )
    if use_direction_colors:
        colors = {}
        for direction, colormap_name in (
            ("signal-on", "Oranges"),
            ("signal-off", "Blues"),
        ):
            direction_channels = [
                channel for channel in channels
                if normalized_directions[channel] == direction
            ]
            shades = dict(zip(
                direction_channels,
                _high_contrast_response_shades(len(direction_channels)),
            ))
            colors.update({
                channel: plt.get_cmap(colormap_name)(shades[channel])
                for channel in direction_channels
            })
        trace_style_by_channel = {
            channel: ("-", "o") for channel in channels
        }
    else:
        cmap = plt.get_cmap("tab10")
        colors = {ch: cmap(i % 10) for i, ch in enumerate(all_ch)}
        trace_style_by_channel = {
            channel: ("-", "o") for channel in channels
        }
    colors.update({
        channel: color
        for channel, color in (channel_colors or {}).items()
        if channel in channels
    })

    fig, ax = plt.subplots(figsize=figsize)
    spread_legend_added = False
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

        if included_step_labels is not None:
            ch_res = [
                row
                for row in ch_res
                if any(
                    step["step_start_scan"] <= row["scan_number"] < step["step_end_scan"]
                    for step in ch_steps
                )
            ]

        dimmed = highlight_channel is not None and ch != highlight_channel
        color = colors[ch]
        line_style, marker = trace_style_by_channel[ch]
        try:
            response_baseline = float((response_baselines or {}).get(ch))
        except (TypeError, ValueError):
            response_baseline = np.nan
        if offset_to_response_baseline and not np.isfinite(response_baseline):
            finite_raw_values = [
                float(row.get(metric))
                for row in ch_res
                if np.isfinite(row.get(metric, np.nan))
            ]
            if finite_raw_values:
                response_baseline = finite_raw_values[0]
        if not offset_to_response_baseline or not np.isfinite(response_baseline):
            response_baseline = 0.0
        if baseline_mode != "preceding_buffer":
            raw_segments = (
                [
                    [
                        row for row in ch_res
                        if step["step_start_scan"] <= row["scan_number"] < step["step_end_scan"]
                    ]
                    for step in ch_steps
                ]
                if included_step_labels is not None
                else [ch_res]
            )
            for raw_segment in raw_segments:
                if not raw_segment:
                    continue
                ax.plot(
                    [row["scan_number"] for row in raw_segment],
                    [
                        row.get(metric, np.nan) - response_baseline
                        for row in raw_segment
                    ],
                    marker=marker,
                    linestyle=line_style,
                    ms=2.8,
                    lw=1.0,
                    color=color,
                    alpha=0.08 if dimmed else 0.22,
                )

        step_midpoints = np.asarray([row["midpoint_scan"] for row in ch_steps], dtype=float)
        plateau_values = np.asarray(
            [row["plateau_value"] - response_baseline for row in ch_steps],
            dtype=float,
        )
        plateau_spreads = np.asarray(
            [row.get("plateau_std", np.nan) for row in ch_steps],
            dtype=float,
        )

        for row in ch_steps:

            ax.hlines(

                row["plateau_value"] - response_baseline,

                row["step_start_scan"],

                row["step_end_scan"],

                color=color,

                lw=3.0,
                linestyle=line_style,

                alpha=0.35 if dimmed else 0.95,

            )

        finite_spread = np.isfinite(plateau_spreads) & (plateau_spreads >= 0)
        if finite_spread.any():
            ax.errorbar(
                step_midpoints[finite_spread],
                plateau_values[finite_spread],
                yerr=plateau_spreads[finite_spread],
                fmt="none",
                ecolor=color,
                elinewidth=1.1,
                capsize=3.0,
                capthick=1.1,
                alpha=0.2 if dimmed else 0.8,
                label=(
                    "Within-plateau ±1 SD"
                    if not spread_legend_added else "_nolegend_"
                ),
                zorder=2.8,
            )
            spread_legend_added = True

        ax.scatter(

            step_midpoints,

            plateau_values,

            color=color,

            s=28,

            marker=marker,

            alpha=0.25 if dimmed else 0.95,

            label=(
                f"{_compact_channel_label(ch)} ({normalized_directions[ch]})"
                if use_direction_colors else f"Channel {ch}"
            ),

            zorder=3,

        )

        contiguous_segments: List[List[dict]] = []
        for step in sorted(ch_steps, key=lambda row: row["step_start_scan"]):
            if (
                not contiguous_segments
                or not np.isclose(
                    contiguous_segments[-1][-1]["step_end_scan"],
                    step["step_start_scan"],
                )
            ):
                contiguous_segments.append([])
            contiguous_segments[-1].append(step)

        for segment in contiguous_segments:
            if len(segment) < 2:
                continue
            segment_x = np.asarray(
                [row["midpoint_scan"] for row in segment],
                dtype=float,
            )
            segment_y = np.asarray(
                [row["plateau_value"] - response_baseline for row in segment],
                dtype=float,
            )
            if len(segment) >= 3:
                try:
                    bridge = PchipInterpolator(segment_x, segment_y)
                    x_dense = np.linspace(segment_x.min(), segment_x.max(), 300)
                    ax.plot(
                        x_dense,
                        bridge(x_dense),
                        color=color,
                        lw=1.8,
                        linestyle="--",
                        alpha=0.25 if dimmed else 0.75,
                    )
                    continue
                except Exception:
                    pass
            ax.plot(
                segment_x,
                segment_y,
                color=color,
                lw=1.4,
                linestyle="--",
                alpha=0.25 if dimmed else 0.75,
            )

    ax.set_xlabel("Scan number")
    plotted_ylabel = ylabel or metric
    if baseline_mode == "preceding_buffer":
        plotted_ylabel = f"Drift-corrected {plotted_ylabel} (B = anchor buffer)"
    ax.set_ylabel(plotted_ylabel)
    ax.set_title(title or f"{metric} titration plateaus")
    ax.grid(False)
    ax.legend(
        title=(
            "Channel and response direction"
            if use_direction_colors else "Channel"
        ),
        loc="best",
        fontsize=8,
    )
    selected_bounds = None
    if included_step_labels is not None and step_rows:
        selected_start = min(row["step_start_scan"] for row in step_rows)
        selected_end = max(row["step_end_scan"] for row in step_rows)
        selected_bounds = (selected_start, selected_end)
        filtered_vlines = [
            (position, label)
            for position, label in filtered_vlines
            if selected_start <= position <= selected_end
        ]
    add_scan_vlines(ax, filtered_vlines, vline_y_frac)
    if offset_to_response_baseline:
        ax.axhline(0.0, color="gray", lw=0.9, alpha=0.6, zorder=1)
    if selected_bounds is not None:
        ax.set_xlim(selected_bounds)
    elif scan_range:

        ax.set_xlim(scan_range)

    fig.tight_layout()
    return fig


def plot_titration_langmuir(
    all_results: List[dict],
    metric: str,
    vlines: Optional[List[Tuple[float, str]]],
    channels: Optional[List[Any]] = None,
    vlines_by_channel: Optional[Dict[Any, List[Tuple[float, str]]]] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    scan_windows: Optional[List[Tuple[int, int]]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
    edge_trim_fraction: float = 0.15,
    figsize: Tuple[int, int] = (8, 4),
    highlight_channel: Optional[int] = None,
    xlabel: str = "Scan number",
    fit_langmuir: bool = True,
    fit_channels: Optional[List[Any]] = None,
    step_concentrations: Optional[List[float]] = None,
    concentration_unit: str = "",
    baseline_mode: str = "none",
    show_legend: bool = True,
    included_step_labels: Optional[List[str]] = None,
    remove_extreme_outliers: bool = False,
    show_uloq: bool = False,
    show_lod: bool = False,
    show_fit_details: bool = False,
    show_errorbar_legend: bool = False,
    response_directions: Optional[Dict[Any, str]] = None,
    channel_colors: Optional[Dict[Any, Any]] = None,
) -> Optional[plt.Figure]:
    if metric not in {"peak_current_selected", "wavelet_energy"}:
        return None

    step_rows = build_titration_step_table(
        all_results,
        metric=metric,
        vlines=vlines,
        channels=channels,
        vlines_by_channel=vlines_by_channel,
        scan_windows=scan_windows,
        scan_range=scan_range,
        edge_trim_fraction=edge_trim_fraction,
        step_concentrations=step_concentrations,
        concentration_unit=concentration_unit,
        baseline_mode=baseline_mode,
        included_step_labels=included_step_labels,
        remove_extreme_outliers=remove_extreme_outliers,
    )
    if not step_rows:
        return None

    channels = sorted({row["channel"] for row in step_rows})
    inferred_directions = {}
    for channel in channels:
        changes = []
        for row in step_rows:
            if row.get("channel") != channel:
                continue
            try:
                concentration = float(row.get("step_concentration"))
                plateau = float(row.get("plateau_value"))
                baseline = float(row.get("fixed_langmuir_baseline"))
            except (TypeError, ValueError):
                continue
            if concentration > 0 and np.isfinite([plateau, baseline]).all():
                changes.append(plateau - baseline)
        if changes and not np.isclose(float(np.median(changes)), 0.0):
            inferred_directions[channel] = (
                "signal-on" if float(np.median(changes)) > 0 else "signal-off"
            )
    inferred_directions.update(response_directions or {})
    colors, styles, plotted_directions = _response_direction_plot_encoding(
        all_results,
        channels,
        response_directions=inferred_directions,
        channel_colors=channel_colors,
    )

    fit_channel_set = set(fit_channels) if fit_channels is not None else None
    fig, ax = plt.subplots(figsize=figsize)
    plotted_any = False
    spread_legend_added = False
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
        line_style, marker = styles[ch]
        raw_x, fit_axis_kind = _fit_axis_from_steps(
            ch_steps,
            step_concentrations=step_concentrations,
        )
        raw_y = np.asarray([row["plateau_value"] for row in ch_steps], dtype=float)
        raw_y_spread = np.asarray(
            [row.get("plateau_std", np.nan) for row in ch_steps],
            dtype=float,
        )
        fit_source_steps = _langmuir_target_steps(ch_steps)
        x, y, fit_axis_kind, fit_steps = _prepare_titration_fit_points(
            fit_source_steps,
            step_concentrations=step_concentrations,
        )
        if fit_axis_kind == "concentration":
            x_axis_kind = "concentration"
            channel_xmax = float(np.nanmax(raw_x)) if raw_x.size else None
            if channel_xmax is not None and np.isfinite(channel_xmax):
                concentration_xmax = (
                    channel_xmax
                    if concentration_xmax is None
                    else max(concentration_xmax, channel_xmax)
                )
        if fit_axis_kind == "step_index":
            xticks.update(int(v) for v in x)

        finite_spread = np.isfinite(raw_y_spread) & (raw_y_spread >= 0)
        if finite_spread.any():
            spread_errorbar = ax.errorbar(
                raw_x[finite_spread],
                raw_y[finite_spread],
                yerr=raw_y_spread[finite_spread],
                fmt="none",
                ecolor=color,
                elinewidth=1.1,
                capsize=7.0,
                capthick=1.4,
                alpha=0.2 if dimmed else 0.8,
                label=(
                    "Within-plateau ±1 SD"
                    if show_errorbar_legend and not spread_legend_added
                    else "_nolegend_"
                ),
                zorder=2.8,
            )
            _data_line, cap_lines, _bar_line_collections = spread_errorbar.lines
            for cap_line in cap_lines:
                cap_line._swv_langmuir_errorbar_cap = True
            if show_errorbar_legend:
                spread_legend_added = True

        method_trace_label = _swv_method_trace_label(ch)
        ax.scatter(
            raw_x,
            raw_y,
            color=color,
            s=34,
            marker=marker,
            alpha=0.25 if dimmed else 0.95,
            label=(
                method_trace_label
                or (
                    f"{_compact_channel_label(ch)} ({plotted_directions[ch]})"
                    if ch in plotted_directions
                    else _compact_channel_label(ch)
                )
            ),
            zorder=3,
        )

        if x.size >= 2:
            if show_fit_details:
                ax.plot(
                    x,
                    y,
                    color=color,
                    lw=1.2,
                    linestyle=line_style,
                    alpha=0.15 if dimmed else 0.45,
                )

            should_fit_channel = fit_langmuir and (
                fit_channel_set is None or ch in fit_channel_set
            )
            if should_fit_channel:
                fixed_baseline = _fixed_baseline_from_steps(fit_source_steps)
                hybrid_fit = (
                    _build_langmuir_hybrid_fit(
                        x,
                        y,
                        fixed_baseline=fixed_baseline,
                    )
                    if fixed_baseline is not None
                    else None
                )
                if hybrid_fit is not None:
                    saturation_idx = hybrid_fit["saturation_idx"]
                    saturation_x = hybrid_fit["saturation_x"]
                    saturation_y = hybrid_fit["saturation_y"]
                    langmuir_params = hybrid_fit["langmuir_params"]
                    post_sat_poly = hybrid_fit["post_sat_poly"]
                    limit_of_detection, _blank_sigma = _langmuir_limit_of_detection(
                        ch_steps,
                        langmuir_params,
                    )
                    uloq_sigma, _uloq_noise_source = _high_concentration_response_noise(
                        fit_source_steps,
                        _blank_sigma,
                    )
                    upper_limit_of_quantification = (
                        _langmuir_upper_limit_of_quantification(
                            langmuir_params,
                            uloq_sigma,
                        )
                        if fit_axis_kind == "concentration"
                        else None
                    )

                    if langmuir_params is not None:
                        measured_fit_max = float(np.nanmax(x))
                        model_curve_end = max(
                            measured_fit_max,
                            (
                                upper_limit_of_quantification
                                if show_uloq else None
                            ) or measured_fit_max,
                        )
                        x_dense_measured = np.linspace(
                            0.0 if fixed_baseline is not None else x.min(),
                            measured_fit_max,
                            300,
                        )
                        y_dense_measured = _langmuir_isotherm(
                            x_dense_measured,
                            *langmuir_params,
                        )
                        ax.plot(
                            x_dense_measured,
                            y_dense_measured,
                            color=color,
                            lw=2.2,
                            linestyle=line_style,
                            alpha=0.25 if dimmed else 0.85,
                        )
                        if model_curve_end > measured_fit_max:
                            x_dense_projected = np.linspace(
                                measured_fit_max,
                                model_curve_end,
                                180,
                            )
                            ax.plot(
                                x_dense_projected,
                                _langmuir_isotherm(
                                    x_dense_projected,
                                    *langmuir_params,
                                ),
                                color=color,
                                lw=1.8,
                                linestyle="--",
                                alpha=0.2 if dimmed else 0.7,
                                label="_nolegend_",
                            )
                        if fit_axis_kind == "concentration":
                            kd_x = float(langmuir_params[2])
                            if (
                                show_fit_details
                                and np.nanmin(x) <= kd_x <= np.nanmax(x)
                            ):
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
                            if show_lod and limit_of_detection is not None:
                                lod_y = float(_langmuir_isotherm(
                                    limit_of_detection,
                                    *langmuir_params,
                                ))
                                channel_label = (
                                    _swv_method_trace_label(ch)
                                    or _compact_channel_label(ch)
                                )
                                unit_suffix = f" {concentration_unit}" if concentration_unit else ""
                                ax.axvline(
                                    limit_of_detection,
                                    color=color,
                                    lw=1.4,
                                    linestyle=":",
                                    alpha=0.25 if dimmed else 0.8,
                                    label=f"{channel_label} LOD",
                                )
                                ax.annotate(
                                    f"LOD {limit_of_detection:.3g}{unit_suffix}",
                                    xy=(limit_of_detection, lod_y),
                                    xytext=(8, -18),
                                    textcoords="offset points",
                                    ha="left",
                                    color=color,
                                    fontsize=8,
                                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
                                )
                            if show_uloq and upper_limit_of_quantification is not None:
                                uloq_y = float(_langmuir_isotherm(
                                    upper_limit_of_quantification,
                                    *langmuir_params,
                                ))
                                channel_label = (
                                    _swv_method_trace_label(ch)
                                    or _compact_channel_label(ch)
                                )
                                unit_suffix = f" {concentration_unit}" if concentration_unit else ""
                                ax.axvline(
                                    upper_limit_of_quantification,
                                    color=color,
                                    lw=1.4,
                                    linestyle="-.",
                                    alpha=0.25 if dimmed else 0.8,
                                    label=f"{channel_label} ULOQ",
                                )
                                uloq_is_projected = (
                                    upper_limit_of_quantification > measured_fit_max
                                )
                                ax.annotate(
                                    f"ULOQ {upper_limit_of_quantification:.3g}{unit_suffix}"
                                    + (" (projected)" if uloq_is_projected else ""),
                                    xy=(upper_limit_of_quantification, uloq_y),
                                    xytext=(-8, -18),
                                    textcoords="offset points",
                                    ha="right",
                                    color=color,
                                    fontsize=8,
                                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
                                )
                                langmuir_xmax = (
                                    upper_limit_of_quantification
                                    if langmuir_xmax is None
                                    else max(langmuir_xmax, upper_limit_of_quantification)
                                )
                    elif show_fit_details and saturation_idx >= 1:
                        ax.plot(
                            x[:saturation_idx + 1],
                            y[:saturation_idx + 1],
                            color=color,
                            lw=1.8,
                            alpha=0.25 if dimmed else 0.75,
                        )

                    pre_sat_label = "Langmuir <= sat" if langmuir_params is not None else "guide <= sat"
                    sat_step_index = int(fit_steps[saturation_idx]["step_index"])
                    channel_label = (
                        _swv_method_trace_label(ch)
                        or _compact_channel_label(ch)
                    )
                    if fit_axis_kind == "concentration":
                        unit_suffix = f" {concentration_unit}" if concentration_unit else ""
                        fit_note = f"{channel_label}: {pre_sat_label}; saturation {saturation_x:.3g}{unit_suffix}"
                        if np.isfinite(saturation_x):
                            langmuir_xmax = (
                                saturation_x
                                if langmuir_xmax is None
                                else max(langmuir_xmax, saturation_x)
                            )
                    else:
                        fit_note = f"{channel_label}: {pre_sat_label}; saturation step {sat_step_index}"
                    if langmuir_params is not None and fit_axis_kind == "concentration":
                        unit_suffix = f" {concentration_unit}" if concentration_unit else ""
                        response_direction = (
                            "signal-off" if langmuir_params[1] < 0 else "signal-on"
                        )
                        fit_note = (
                            f"{channel_label}: {response_direction}; "
                            f"Kd {langmuir_params[2]:.3g}{unit_suffix}"
                        )
                        if fixed_baseline is not None:
                            fit_note += f"; B fixed at {fixed_baseline:.3g}"
                        if show_lod and limit_of_detection is not None:
                            fit_note += f"; LOD = {limit_of_detection:.3g}{unit_suffix}"
                        if show_uloq and upper_limit_of_quantification is not None:
                            fit_note += f"; ULOQ = {upper_limit_of_quantification:.3g}{unit_suffix}"
                            if upper_limit_of_quantification > float(np.nanmax(x)):
                                fit_note += " (projected)"
                    elif langmuir_params is not None:
                        fit_note += ", no Kd (missing concentration axis)"
                    if show_fit_details:
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
        ax.set_xlabel(f"Ligand Concentration{unit_suffix}")
        finite_xmax_candidates = [
            float(value)
            for value in (langmuir_xmax, concentration_xmax)
            if value is not None and np.isfinite(value)
        ]
        xmax = max(finite_xmax_candidates) if finite_xmax_candidates else None
        if xmax is not None and xmax > 0:
            ax.set_xscale("linear")
            ax.set_xlim(left=0, right=xmax * 1.05)
    else:
        ax.set_xlabel("Titration step index")
    plotted_ylabel = ylabel or (
        "Peak Height (uA)"
        if metric == "peak_current_selected"
        else metric
    )
    ax.set_ylabel(plotted_ylabel)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title or f"{metric} titration isotherm")
    ax.grid(False)
    if show_legend:
        ax.legend(loc="best", fontsize=8)
    if xticks:
        ax.set_xticks(sorted(xticks))
    if fit_notes:
        if len(channels) == 1:
            fit_notes = [
                note.split(":", 1)[1].strip() if ":" in note else note
                for note in fit_notes
            ]
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


def plot_titration_snr(
    step_rows: List[dict],
    title: str = "Titration SNR by concentration",
    concentration_unit: str = "",
    figsize: Tuple[int, int] = (8, 4),
    lod_snr_cutoff: float = 3.0,
    fit_summary_rows: Optional[List[dict]] = None,
    show_uloq: bool = True,
    show_lod: bool = True,
    response_directions: Optional[Dict[Any, str]] = None,
    channel_colors: Optional[Dict[Any, Any]] = None,
) -> Optional[plt.Figure]:
    usable_rows = [
        row for row in step_rows
        if row.get("step_concentration") is not None
        and np.isfinite(row.get("step_concentration", np.nan))
        and row.get("titration_snr") is not None
        and np.isfinite(row.get("titration_snr", np.nan))
    ]
    if not usable_rows:
        return None

    channels = sorted(
        {row["channel"] for row in usable_rows},
        key=_channel_sort_key,
    )
    colors, styles, plotted_directions = _response_direction_plot_encoding(
        usable_rows + list(fit_summary_rows or []),
        channels,
        response_directions=response_directions,
        channel_colors=channel_colors,
    )
    fig, ax = plt.subplots(figsize=figsize)
    spread_legend_added = False
    for index, channel in enumerate(channels):
        channel_rows = sorted(
            [row for row in usable_rows if row["channel"] == channel],
            key=lambda row: (row["step_concentration"], row["step_index"]),
        )
        x = np.asarray(
            [row["step_concentration"] for row in channel_rows],
            dtype=float,
        )
        y = np.asarray(
            [row["titration_snr"] for row in channel_rows],
            dtype=float,
        )
        color = colors[channel]
        line_style, marker = styles[channel]
        channel_summary = next(
            (
                row for row in (fit_summary_rows or [])
                if row.get("channel") == channel
            ),
            None,
        )
        summary_noise_sigma = (
            channel_summary.get("blank_sigma")
            if channel_summary is not None else None
        )
        snr_spreads = []
        for row in channel_rows:
            response_spread = row.get("plateau_std")
            noise_sigma = row.get("snr_noise_std")
            if noise_sigma is None:
                noise_sigma = summary_noise_sigma
            try:
                response_spread = float(response_spread)
                noise_sigma = float(noise_sigma)
            except (TypeError, ValueError):
                snr_spreads.append(np.nan)
                continue
            snr_spreads.append(
                response_spread / noise_sigma
                if np.isfinite(response_spread)
                and response_spread >= 0
                and np.isfinite(noise_sigma)
                and noise_sigma > 0
                else np.nan
            )
        snr_spreads = np.asarray(snr_spreads, dtype=float)
        finite_spread = np.isfinite(snr_spreads) & (snr_spreads >= 0)
        if finite_spread.any():
            ax.errorbar(
                x[finite_spread],
                y[finite_spread],
                yerr=snr_spreads[finite_spread],
                fmt="none",
                ecolor=color,
                elinewidth=1.1,
                capsize=3.0,
                capthick=1.1,
                alpha=0.8,
                label=(
                    "Within-plateau ±1 SD"
                    if not spread_legend_added else "_nolegend_"
                ),
                zorder=2.8,
            )
            spread_legend_added = True
        ax.plot(
            x,
            y,
            marker=marker,
            ms=5,
            lw=1.5,
            linestyle=line_style,
            color=color,
            label=(
                f"{_compact_channel_label(channel)} "
                f"({plotted_directions[channel]})"
                if channel in plotted_directions
                else _compact_channel_label(channel)
            ),
        )
        if channel_summary is not None:
            amplitude = channel_summary.get("langmuir_amplitude")
            kd = channel_summary.get("langmuir_kd")
            noise_sigma = channel_summary.get("blank_sigma")
            try:
                amplitude = abs(float(amplitude))
                kd = float(kd)
                noise_sigma = float(noise_sigma)
            except (TypeError, ValueError):
                amplitude = kd = noise_sigma = None
            if (
                amplitude is not None
                and kd is not None
                and noise_sigma is not None
                and amplitude > 0
                and kd > 0
                and noise_sigma > 0
                and np.isfinite([amplitude, kd, noise_sigma]).all()
            ):
                measured_xmax = float(np.nanmax(x))
                summary_uloq = channel_summary.get(
                    "upper_limit_of_quantification"
                )
                projected_xmax = (
                    float(summary_uloq)
                    if show_uloq
                    and summary_uloq is not None
                    and np.isfinite(summary_uloq)
                    and float(summary_uloq) > measured_xmax
                    else measured_xmax
                )
                measured_dense = np.linspace(0.0, measured_xmax, 240)
                measured_snr = (
                    amplitude * measured_dense / (kd + measured_dense)
                ) / noise_sigma
                ax.plot(
                    measured_dense,
                    measured_snr,
                    color=color,
                    lw=2.0,
                    linestyle=line_style,
                    label=f"{_compact_channel_label(channel)} Langmuir SNR fit",
                )
                if projected_xmax > measured_xmax:
                    projected_dense = np.linspace(
                        measured_xmax,
                        projected_xmax,
                        160,
                    )
                    projected_snr = (
                        amplitude * projected_dense / (kd + projected_dense)
                    ) / noise_sigma
                    ax.plot(
                        projected_dense,
                        projected_snr,
                        color=color,
                        lw=1.7,
                        linestyle="--",
                        alpha=0.75,
                        label="_nolegend_",
                    )
        uloq = (
            channel_summary.get("upper_limit_of_quantification")
            if channel_summary is not None else None
        )
        lod = (
            channel_summary.get("snr_3_cutoff_concentration")
            if channel_summary is not None else None
        )
        if show_lod and lod is not None and np.isfinite(lod) and float(lod) >= 0:
            unit_text = f" {concentration_unit}" if concentration_unit else ""
            ax.axvline(
                float(lod),
                color=color,
                lw=1.4,
                linestyle=":",
                label=(
                    f"{_compact_channel_label(channel)} LOD "
                    f"fit SNR={lod_snr_cutoff:g} ({float(lod):.3g}{unit_text})"
                ),
            )
        if show_uloq and uloq is not None and np.isfinite(uloq) and float(uloq) >= 0:
            unit_text = f" {concentration_unit}" if concentration_unit else ""
            projected_text = (
                ", projected"
                if channel_summary.get("upper_limit_of_quantification_is_extrapolated")
                else ""
            )
            ax.axvline(
                float(uloq),
                color=color,
                lw=1.4,
                linestyle="-.",
                label=(
                    f"{_compact_channel_label(channel)} ULOQ "
                    f"({float(uloq):.3g}{unit_text}{projected_text})"
                ),
            )
    unit_suffix = f" ({concentration_unit})" if concentration_unit else ""
    if show_lod and np.isfinite(lod_snr_cutoff) and lod_snr_cutoff > 0:
        ax.axhspan(
            0,
            lod_snr_cutoff,
            color="gray",
            alpha=0.10,
            zorder=0,
        )
        ax.axhline(
            lod_snr_cutoff,
            color="black",
            lw=1.3,
            linestyle="--",
            label=f"LOD cutoff (SNR = {lod_snr_cutoff:g})",
        )
    ax.set_xlabel(f"Target concentration{unit_suffix}")
    ax.set_ylabel("Plateau SNR")
    ax.set_title(title)
    ax.set_xscale("linear")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(False)
    ax.legend(title="Channel / method", fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_titration_concentration_accuracy(
    accuracy_rows: List[dict],
    title: str = "Predicted vs. Known",
    concentration_unit: str = "",
    figsize: Tuple[float, float] = (9, 4.5),
    show_uloq: bool = True,
    show_lod: bool = True,
    percent_error_bound: float = 20.0,
    acceptance_region_alpha: float = 0.10,
    channel_colors: Optional[Dict[Any, Any]] = None,
    response_directions: Optional[Dict[Any, str]] = None,
) -> Optional[plt.Figure]:
    usable_rows = [
        row for row in accuracy_rows
        if row.get("known_concentration") is not None
        and row.get("predicted_concentration") is not None
        and np.isfinite(row.get("known_concentration", np.nan))
        and np.isfinite(row.get("predicted_concentration", np.nan))
        and float(row.get("known_concentration")) > 0
        and float(row.get("predicted_concentration")) > 0
    ]
    if not usable_rows:
        return None

    channels = sorted(
        {row["channel"] for row in usable_rows},
        key=_channel_sort_key,
    )
    measured_concentrations = np.asarray(
        [row["known_concentration"] for row in usable_rows],
        dtype=float,
    )
    measured_min = float(np.nanmin(measured_concentrations))
    measured_max = float(np.nanmax(measured_concentrations))
    if np.isclose(measured_min, measured_max):
        axis_min = measured_min / 1.05
        axis_max = measured_max * 1.05
    else:
        log_min = float(np.log(measured_min))
        log_max = float(np.log(measured_max))
        log_padding = 0.05 * (log_max - log_min)
        axis_min = float(np.exp(log_min - log_padding))
        axis_max = float(np.exp(log_max + log_padding))
    colors, styles, response_directions = _response_direction_plot_encoding(
        usable_rows,
        channels,
        response_directions=response_directions,
        channel_colors=channel_colors,
    )
    fig, ax = plt.subplots(figsize=figsize)
    lod_legend_added = False
    uloq_legend_added = False
    log_errors_by_trace: Dict[str, List[float]] = {}
    annotation_color_by_trace: Dict[str, Any] = {}
    for index, channel in enumerate(channels):
        channel_rows = [
            row for row in usable_rows if row["channel"] == channel
        ]
        known = np.asarray(
            [row["known_concentration"] for row in channel_rows],
            dtype=float,
        )
        predicted = np.asarray(
            [row["predicted_concentration"] for row in channel_rows],
            dtype=float,
        )
        color = colors[channel]
        _line_style, marker = styles[channel]
        method_trace_label = _swv_method_trace_label(channel)
        annotation_label = method_trace_label or _compact_channel_label(channel)
        annotation_color_by_trace.setdefault(annotation_label, color)
        for row in channel_rows:
            known_value = float(row["known_concentration"])
            predicted_value = float(row["predicted_concentration"])
            log_error = float(np.log10(predicted_value / known_value))
            if np.isfinite(log_error):
                log_errors_by_trace.setdefault(annotation_label, []).append(
                    log_error
                )
        uloq = next(
            (
                row.get("upper_limit_of_quantification")
                for row in channel_rows
                if row.get("upper_limit_of_quantification") is not None
                and np.isfinite(row.get("upper_limit_of_quantification"))
            ),
            None,
        )
        lod = next(
            (
                row.get("limit_of_detection")
                for row in channel_rows
                if row.get("limit_of_detection") is not None
                and np.isfinite(row.get("limit_of_detection"))
            ),
            None,
        )
        if show_lod and lod is not None and axis_min <= float(lod) <= axis_max:
            ax.axvline(
                float(lod),
                color=color,
                lw=1.3,
                linestyle=":",
                label="LOD boundaries" if not lod_legend_added else "_nolegend_",
            )
            lod_legend_added = True
            ax.axhline(
                float(lod),
                color=color,
                lw=1.0,
                linestyle=":",
                alpha=0.55,
                label="_nolegend_",
            )
        if (
            show_uloq
            and uloq is not None
            and axis_min <= float(uloq) <= axis_max
        ):
            ax.axvline(
                float(uloq),
                color=color,
                lw=1.3,
                linestyle="-.",
                label="ULOQ boundaries" if not uloq_legend_added else "_nolegend_",
            )
            uloq_legend_added = True
            ax.axhline(
                float(uloq),
                color=color,
                lw=1.0,
                linestyle="-.",
                alpha=0.55,
                label="_nolegend_",
            )
        prediction_points = ax.scatter(
            known,
            predicted,
            s=34,
            alpha=0.30,
            color=color,
            marker=marker,
            label=(
                method_trace_label
                or (
                    f"{_compact_channel_label(channel)} "
                    f"({response_directions[channel]})"
                    if channel in response_directions
                    else _compact_channel_label(channel)
                )
            ),
            zorder=3,
        )
        prediction_points._swv_preserve_alpha = True

    if np.isfinite(percent_error_bound) and 0 < percent_error_bound < 100:
        bound_fraction = float(percent_error_bound) / 100.0
        bound_x = np.geomspace(axis_min, axis_max, 300)
        lower_bound = (1.0 - bound_fraction) * bound_x
        upper_bound = (1.0 + bound_fraction) * bound_x
        acceptance_region = ax.fill_between(
            bound_x,
            lower_bound,
            upper_bound,
            color="tab:green",
            alpha=float(np.clip(acceptance_region_alpha, 0.0, 1.0)),
            label=f"Within ±{percent_error_bound:g}%",
            zorder=0,
        )
        # The Streamlit plot-formatting layer adjusts scatter opacity. Keep
        # this independently controlled fill at its requested alpha.
        acceptance_region._swv_preserve_alpha = True
        ax.plot(
            bound_x,
            lower_bound,
            color="tab:green",
            lw=1.0,
            linestyle=":",
            label="_nolegend_",
            zorder=1,
        )
        ax.plot(
            bound_x,
            upper_bound,
            color="tab:green",
            lw=1.0,
            linestyle=":",
            label="_nolegend_",
            zorder=1,
        )
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color="black",
        lw=1.2,
        linestyle="--",
        label="1:1",
    )
    annotation_order = {
        "Optimized Method": 0,
        "Manual Method": 1,
    }
    for annotation_index, annotation_label in enumerate(sorted(
        log_errors_by_trace,
        key=lambda label: (annotation_order.get(label, 2), label),
    )):
        trace_log_errors = np.asarray(
            log_errors_by_trace[annotation_label],
            dtype=float,
        )
        if trace_log_errors.size:
            log_rmse = float(
                np.sqrt(np.mean(np.square(trace_log_errors)))
            )
            rms_fold_error = float(10.0 ** log_rmse)
        else:
            continue
        compact_annotation_label = annotation_label.removesuffix(" Method")
        ax.text(
            0.03,
            0.97 - (0.075 * annotation_index),
            (
                f"{compact_annotation_label} RMS Fold Error: "
                f"{rms_fold_error:.2f}×"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color=annotation_color_by_trace[annotation_label],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2),
        )
    unit_suffix = f" ({concentration_unit})" if concentration_unit else ""
    ax.set_xlabel(f"Known concentration{unit_suffix}")
    ax.set_ylabel(f"Predicted concentration{unit_suffix}")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    allowed_legend_labels = {
        f"Within ±{percent_error_bound:g}%",
        "1:1",
    }
    limit_handles_and_labels = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if label in allowed_legend_labels
    ]
    ax.legend(
        [item[0] for item in limit_handles_and_labels],
        [item[1] for item in limit_handles_and_labels],
        fontsize=8,
        loc="lower right",
    )
    fig.tight_layout()
    return fig


def plot_titration_concentration_vs_measurement(
    accuracy_rows: List[dict],
    title: str = "Langmuir-predicted concentration by measurement",
    concentration_unit: str = "uM",
    figsize: Tuple[int, int] = (9, 4.5),
    percent_error_bound: float = 20.0,
    acceptance_region_alpha: float = 0.10,
    channel_colors: Optional[Dict[Any, Any]] = None,
    response_directions: Optional[Dict[Any, str]] = None,
    vlines: Optional[List[Tuple[float, str]]] = None,
    vline_y_frac: float = 0.86,
) -> Optional[plt.Figure]:
    """Plot per-SWV Langmuir concentration predictions by measurement number.

    Mapping error bars are shown only when their full ±1σ interval lies inside
    the fitted quantitative range. Outside that range, inverse-Langmuir
    uncertainty is poorly conditioned and an enormous symmetric bar is not a
    meaningful quantitative interval.
    """
    usable_rows = []
    for row in accuracy_rows:
        try:
            known = float(row.get("known_concentration"))
            uncensored_prediction = float(
                row.get("unbounded_predicted_concentration")
            )
        except (TypeError, ValueError):
            try:
                uncensored_prediction = float(row.get("predicted_concentration"))
            except (TypeError, ValueError):
                continue
        if np.isfinite(known) and np.isfinite(uncensored_prediction) and known >= 0:
            plotted_row = dict(row)
            plotted_row["_plot_predicted_concentration"] = uncensored_prediction
            usable_rows.append(plotted_row)
    if not usable_rows:
        return None

    # Each displayed SWV method has its own remapped measurement sequence.
    # The source axis interleaves methods and is therefore 2× too large for a
    # two-method experiment.
    x_key = "scan_number"
    selected_known_concentrations = np.asarray(
        [
            float(row["known_concentration"])
            for row in accuracy_rows
            if row.get("known_concentration") is not None
            and np.isfinite(row.get("known_concentration", np.nan))
            and float(row["known_concentration"]) > 0
        ],
        dtype=float,
    )
    if selected_known_concentrations.size:
        minimum_nonzero_concentration = float(
            np.nanmin(selected_known_concentrations)
        )
    else:
        positive_predictions = np.asarray([
            float(row["_plot_predicted_concentration"])
            for row in usable_rows
            if float(row["_plot_predicted_concentration"]) > 0
        ])
        minimum_nonzero_concentration = (
            float(np.nanmin(positive_predictions))
            if positive_predictions.size else 1.0
        )

    usable_rows = sorted(usable_rows, key=lambda row: row.get(x_key))
    reference_by_x: Dict[Any, List[float]] = {}
    for row in usable_rows:
        reference_concentration = float(row["known_concentration"])
        reference_by_x.setdefault(row.get(x_key), []).append(
            reference_concentration
        )
    reference_x = sorted(reference_by_x)
    reference_y = np.asarray(
        [float(np.median(reference_by_x[value])) for value in reference_x],
        dtype=float,
    )
    reference_levels = _concentration_to_doubling_level(
        reference_y,
        minimum_nonzero_concentration,
    )
    plotted_level_values = list(reference_levels)

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(
        reference_x,
        reference_levels,
        where="mid",
        color="black",
        lw=1.8,
        linestyle="--",
        label="Known Concentration",
        zorder=10,
    )

    channels = sorted(
        {row.get("channel") for row in usable_rows},
        key=_channel_sort_key,
    )
    colors, styles, response_directions = _response_direction_plot_encoding(
        usable_rows,
        channels,
        response_directions=response_directions,
        channel_colors=channel_colors,
    )
    suppressed_uncertainty_count = 0
    for index, channel in enumerate(channels):
        channel_rows = sorted(
            [row for row in usable_rows if row.get("channel") == channel],
            key=lambda row: row.get(x_key),
        )
        x_values = [row.get(x_key) for row in channel_rows]
        predicted = np.asarray(
            [float(row["_plot_predicted_concentration"]) for row in channel_rows],
            dtype=float,
        )
        predicted_levels = _concentration_to_doubling_level(
            predicted,
            minimum_nonzero_concentration,
        )
        plotted_level_values.extend(predicted_levels.tolist())
        color = colors[channel]
        line_style, marker = styles[channel]
        uncertainty_rows = []
        for position, row in zip(x_values, channel_rows):
            uncertainty = row.get("predicted_concentration_std")
            try:
                uncertainty = float(uncertainty)
                predicted_value = float(row["_plot_predicted_concentration"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(uncertainty) or uncertainty < 0:
                continue

            interval_lower = predicted_value - uncertainty
            interval_upper = predicted_value + uncertainty
            uloq = row.get("upper_limit_of_quantification")
            try:
                uloq = float(uloq)
            except (TypeError, ValueError):
                uloq = None
            above_quantifiable_range = (
                uloq is not None
                and np.isfinite(uloq)
                and interval_upper > uloq
            )
            if above_quantifiable_range:
                suppressed_uncertainty_count += 1
                continue
            uncertainty_rows.append((position, row))
        if uncertainty_rows:
            error_x = [position for position, _row in uncertainty_rows]
            error_y = np.asarray(
                [float(row["_plot_predicted_concentration"]) for _position, row in uncertainty_rows],
                dtype=float,
            )
            error_std = np.asarray(
                [float(row["predicted_concentration_std"]) for _position, row in uncertainty_rows],
                dtype=float,
            )
            error_levels = _concentration_to_doubling_level(
                error_y,
                minimum_nonzero_concentration,
            )
            lower_error_levels = _concentration_to_doubling_level(
                error_y - error_std,
                minimum_nonzero_concentration,
            )
            upper_error_levels = _concentration_to_doubling_level(
                error_y + error_std,
                minimum_nonzero_concentration,
            )
            transformed_error = np.vstack((
                error_levels - lower_error_levels,
                upper_error_levels - error_levels,
            ))
            transformed_error = np.maximum(transformed_error, 0.0)
            plotted_level_values.extend(lower_error_levels.tolist())
            plotted_level_values.extend(upper_error_levels.tolist())
            uncertainty_errorbar = ax.errorbar(
                error_x,
                error_levels,
                yerr=transformed_error,
                fmt="none",
                ecolor=color,
                elinewidth=0.9,
                capsize=2.5,
                capthick=0.9,
                alpha=0.7,
                label="_nolegend_",
                zorder=2.8,
            )
            # Keep an explicit tag so callers that change between linear and
            # symlog after figure construction can rebind every error-bar
            # component to the updated concentration-axis transform.
            _data_line, cap_lines, bar_line_collections = uncertainty_errorbar.lines
            for errorbar_artist in (*cap_lines, *bar_line_collections):
                errorbar_artist._swv_concentration_errorbar = True
        ax.plot(
            x_values,
            predicted_levels,
            color=color,
            lw=1.0,
            linestyle=line_style,
            alpha=0.65,
            label="_nolegend_",
            zorder=3,
        )
        method_trace_label = _swv_method_trace_label(channel)
        ax.scatter(
            x_values,
            predicted_levels,
            color=color,
            s=25,
            marker=marker,
            alpha=0.8,
            label=(
                method_trace_label
                or (
                    f"{_compact_channel_label(channel)} "
                    f"({response_directions[channel]})"
                    if channel in response_directions
                    else _compact_channel_label(channel)
                )
            ),
            zorder=4,
        )

    if suppressed_uncertainty_count or any(
        row.get("predicted_concentration_std") is not None
        and np.isfinite(row.get("predicted_concentration_std", np.nan))
        for row in usable_rows
    ):
        uncertainty_note = "Error bars: propagated ±1σ; below-LOD values allowed"
        if suppressed_uncertainty_count:
            uncertainty_note += (
                f"; omitted above ULOQ (n={suppressed_uncertainty_count})"
            )
        ax._swv_uncertainty_note = uncertainty_note
    ax.set_xlabel("SWV Measurement Number")
    concentration_label = (
        f"Predicted Concentration ({concentration_unit})"
        if concentration_unit else "Predicted Concentration"
    )
    ax.set_ylabel(concentration_label)
    ax.set_title(title, pad=12)
    ax.set_yscale("linear")
    displayed_known_concentrations = (
        selected_known_concentrations
        if selected_known_concentrations.size
        else np.asarray([minimum_nonzero_concentration], dtype=float)
    )
    finite_plotted_levels = np.asarray([
        value for value in plotted_level_values if np.isfinite(value)
    ])
    if finite_plotted_levels.size:
        minimum_plotted_level = float(np.nanmin(finite_plotted_levels))
        maximum_plotted_level = float(np.nanmax(finite_plotted_levels))
        level_span = maximum_plotted_level - minimum_plotted_level
        if np.isclose(level_span, 0.0):
            padding = max(abs(minimum_plotted_level) * 0.1, 0.1)
        else:
            padding = level_span * 0.1
        ax.set_ylim(
            minimum_plotted_level - padding,
            maximum_plotted_level + padding,
        )
    tick_concentrations = np.concatenate((
        [0.0],
        np.unique(displayed_known_concentrations),
    ))
    tick_levels = _concentration_to_doubling_level(
        tick_concentrations,
        minimum_nonzero_concentration,
    )
    ax.set_yticks(
        tick_levels,
        labels=[
            f"{value:g}" for value in tick_concentrations
        ],
    )
    ax._swv_concentration_doubling_scale = True
    ax._swv_concentration_doubling_reference = minimum_nonzero_concentration
    ax.axhline(
        0.0,
        color="gray",
        lw=0.8,
        alpha=0.5,
        label="_nolegend_",
        zorder=1,
    )
    finite_x_values = np.asarray([
        float(row.get(x_key))
        for row in usable_rows
        if row.get(x_key) is not None and np.isfinite(row.get(x_key))
    ])
    filtered_vlines = []
    if finite_x_values.size:
        x_min = float(np.nanmin(finite_x_values))
        x_max = float(np.nanmax(finite_x_values))
        filtered_vlines = _filter_titration_vlines(
            vlines,
            scan_range=(x_min, x_max),
        )
        right_boundaries = [
            (float(position), str(label))
            for position, label in (vlines or [])
            if float(position) > x_max
        ]
        if right_boundaries:
            next_boundary = min(right_boundaries, key=lambda item: item[0])
            if not any(np.isclose(position, next_boundary[0]) for position, _ in filtered_vlines):
                filtered_vlines.append(next_boundary)
    add_scan_vlines(
        ax,
        filtered_vlines,
        max(float(vline_y_frac), 0.92),
    )
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.80),
        ncol=1,
        borderaxespad=0.5,
    )
    fig.tight_layout()
    return fig




def plot_drift_vs_scan(

    all_results: List[dict],

    drift_metric: str,

    channels: Optional[List[Any]] = None,

    title: Optional[str] = None,

    ylabel: Optional[str] = None,

    vlines: Optional[List[Tuple[float, str]]] = None,

    vline_y_frac: float = 0.85,

    scan_range: Optional[Tuple[int, int]] = None,

    highlight_channel: Optional[Any] = None,

    figsize: Tuple[int, int] = (10, 4),

    xlabel: str = "Scan number",

    channel_colors: Optional[Dict[Any, Any]] = None,

) -> Optional[plt.Figure]:

    all_ch = sorted({r["channel"] for r in all_results}, key=_channel_sort_key)

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
    colors.update({
        channel: color
        for channel, color in (channel_colors or {}).items()
        if channel in all_ch
    })



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

                label=f"Channel {ch}")



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
