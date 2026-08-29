import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from .io import (
    MeasurementFile,
    collect_cv_csvs_from_folders,
    filter_finite,
    group_by_channel_and_sort,
    load_swv_csv,
)
from .processing import apply_smoothing

NSCANS_RE = re.compile(r"nscans\((?P<n>\d+)\)", re.IGNORECASE)
MEAS_LOOP_CV_RE = re.compile(
    r"meas_loop_cv\s+\S+\s+\S+\s+"
    r"(?P<start>[-\d.]+m)\s+"
    r"(?P<vertex1>[-\d.]+m)\s+"
    r"(?P<vertex2>[-\d.]+m)\s+"
    r"(?P<step>[-\d.]+m)\s+"
    r"(?P<rate>[-\d.]+m)",
    re.IGNORECASE,
)


def _file_signature(filepath: str) -> Tuple[int, int]:
    stat = os.stat(filepath)
    return int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=512)
def _load_filtered_arrays_cached(
    filepath: str,
    voltage_col: str,
    current_col: Optional[str],
    file_mtime_ns: int,
    file_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    del file_mtime_ns, file_size
    v_raw, i_raw = load_swv_csv(filepath, voltage_col=voltage_col, current_col=current_col)
    v_raw, i_raw = filter_finite(v_raw, i_raw)
    return np.asarray(v_raw, dtype=float), np.asarray(i_raw, dtype=float)


def _scan_in_windows(
    scan_number: int,
    scan_windows: Optional[Tuple[Tuple[int, int], ...]],
    scan_range: Optional[Tuple[int, int]],
) -> bool:
    if scan_windows:
        return any(start <= scan_number < end for start, end in scan_windows)
    if scan_range is not None:
        return scan_range[0] <= scan_number <= scan_range[1]
    return True


def _remap_scan_number(
    scan_number: int,
    scan_windows: Optional[Tuple[Tuple[int, int], ...]],
    scan_range: Optional[Tuple[int, int]],
) -> int:
    if scan_windows:
        offset = 0
        for start, end in scan_windows:
            if start <= scan_number < end:
                return offset + (scan_number - start)
            offset += end - start
        raise ValueError(f"Scan {scan_number} is outside selected scan windows.")
    if scan_range is not None:
        return scan_number - scan_range[0]
    return scan_number


def _parse_milli_token(token: Optional[str]) -> Optional[float]:
    if not token:
        return None
    text = token.strip().lower()
    if text.endswith("m"):
        return float(text[:-1]) / 1000.0
    return float(text)


def _infer_method_path(csv_path: str) -> str:
    folder = os.path.dirname(csv_path)
    stem, _ = os.path.splitext(os.path.basename(csv_path))
    return os.path.join(folder, "methods_used", f"{stem}.ms")


def _ec_label_from_nscans(nscans: Optional[int]) -> str:
    if nscans == 50:
        return "EC4"
    if nscans == 3:
        return "EC3"
    if nscans is None:
        return "CV"
    return f"CV-{nscans}"


@lru_cache(maxsize=512)
def load_cv_method_metadata(method_path: str) -> dict:
    meta = {
        "method_path": method_path,
        "method_exists": False,
        "nscans": None,
        "ec_label": "CV",
        "start_voltage": None,
        "vertex1_voltage": None,
        "vertex2_voltage": None,
        "step_voltage": None,
        "scan_rate_v_per_s": None,
    }
    if not os.path.exists(method_path):
        return meta

    meta["method_exists"] = True
    with open(method_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    nscans_match = NSCANS_RE.search(text)
    if nscans_match:
        meta["nscans"] = int(nscans_match.group("n"))
    meta["ec_label"] = _ec_label_from_nscans(meta["nscans"])

    loop_match = MEAS_LOOP_CV_RE.search(text)
    if loop_match:
        meta["start_voltage"] = _parse_milli_token(loop_match.group("start"))
        meta["vertex1_voltage"] = _parse_milli_token(loop_match.group("vertex1"))
        meta["vertex2_voltage"] = _parse_milli_token(loop_match.group("vertex2"))
        meta["step_voltage"] = _parse_milli_token(loop_match.group("step"))
        meta["scan_rate_v_per_s"] = _parse_milli_token(loop_match.group("rate"))

    return meta


def _estimate_turn_idx(voltage: np.ndarray) -> int:
    v = np.asarray(voltage, dtype=float)
    if len(v) < 3:
        raise ValueError("Too few points to determine CV turning point.")

    diffs = np.diff(v)
    nz = diffs[np.isfinite(diffs) & (np.abs(diffs) > 1e-12)]
    if nz.size == 0:
        return int(len(v) // 2)

    initial_dir = 1.0 if float(np.median(nz[: min(25, nz.size)])) >= 0 else -1.0
    turn_idx = int(np.argmax(v) if initial_dir >= 0 else np.argmin(v))
    if turn_idx <= 0 or turn_idx >= len(v) - 1:
        raise ValueError("Failed to locate an interior CV turning point.")
    return turn_idx


def _cycle_boundaries_from_voltage(voltage: np.ndarray, expected_cycles: Optional[int] = None) -> List[Tuple[int, int]]:
    v = np.asarray(voltage, dtype=float)
    if len(v) < 10:
        raise ValueError("Too few points to segment CV cycles.")

    diffs = np.diff(v)
    signs = np.sign(diffs)
    if len(signs) == 0:
        return [(0, len(v) - 1)]

    for idx in range(1, len(signs)):
        if signs[idx] == 0:
            signs[idx] = signs[idx - 1]
    for idx in range(len(signs) - 2, -1, -1):
        if signs[idx] == 0:
            signs[idx] = signs[idx + 1]

    turn_idxs = np.where(signs[:-1] * signs[1:] < 0)[0] + 1
    if len(turn_idxs) < 2:
        return [(0, len(v) - 1)]

    boundaries = [0]
    boundaries.extend(int(idx) for idx in turn_idxs[1::2])
    if boundaries[-1] != len(v) - 1:
        boundaries.append(len(v) - 1)

    spans = [end - start for start, end in zip(boundaries[:-1], boundaries[1:]) if end > start]
    if not spans:
        return [(0, len(v) - 1)]

    median_span = float(np.median(spans))
    cycles: List[Tuple[int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < max(6, int(round(median_span * 0.4))):
            continue
        cycles.append((int(start), int(end)))

    if expected_cycles is not None and len(cycles) > expected_cycles:
        cycles = cycles[:expected_cycles]
    if expected_cycles is not None and expected_cycles > 1 and len(cycles) != expected_cycles:
        approx_span = max((len(v) - 1) / float(expected_cycles), 2.0)
        start_voltage = float(v[0])
        boundary_points = [0]
        last_idx = 0
        search_half_width = max(int(round(approx_span * 0.20)), 3)
        for k in range(1, expected_cycles):
            target = int(round(k * approx_span))
            lo = max(last_idx + 2, target - search_half_width)
            hi = min(len(v) - 2, target + search_half_width)
            if hi > lo:
                candidates = np.arange(lo, hi + 1, dtype=int)
                best = int(candidates[np.argmin(np.abs(v[candidates] - start_voltage))])
            else:
                best = min(max(last_idx + 2, target), len(v) - 2)
            if best <= last_idx + 1:
                best = min(last_idx + max(int(round(approx_span)), 2), len(v) - 2)
            boundary_points.append(best)
            last_idx = best
        boundary_points.append(len(v) - 1)

        rebuilt: List[Tuple[int, int]] = []
        for start, end in zip(boundary_points[:-1], boundary_points[1:]):
            if end - start >= 3:
                rebuilt.append((int(start), int(end)))
        if len(rebuilt) >= max(1, int(round(expected_cycles * 0.8))):
            cycles = rebuilt
    return cycles


def _crop_by_voltage(
    voltage: np.ndarray,
    current: np.ndarray,
    crop_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(voltage, dtype=float)
    i = np.asarray(current, dtype=float)
    lo, hi = float(min(crop_range)), float(max(crop_range))
    mask = (v >= lo) & (v <= hi)
    return v[mask], i[mask]


def _linear_baseline(voltage: np.ndarray, current: np.ndarray) -> np.ndarray:
    v = np.asarray(voltage, dtype=float)
    y = np.asarray(current, dtype=float)
    if len(v) < 2:
        return np.zeros_like(y)
    v0, v1 = float(v[0]), float(v[-1])
    y0, y1 = float(y[0]), float(y[-1])
    denom = (v1 - v0) if abs(v1 - v0) > 1e-12 else 1e-12
    slope = (y1 - y0) / denom
    return slope * v + (y0 - slope * v0)


def _edge_trim_points(length: int, fraction: float, minimum_points: int = 3) -> int:
    if length <= 6:
        return 0
    trim = max(int(round(length * max(float(fraction), 0.0))), int(minimum_points))
    return min(trim, max((length - 3) // 2, 0))


def _dominant_peak_idx(
    signal: np.ndarray,
    kind: str,
    edge_trim_fraction: float,
    min_peak_prominence_uA: Optional[float],
    start_fraction: float = 0.0,
    end_fraction: float = 1.0,
) -> Tuple[int, float, int]:
    y = np.asarray(signal, dtype=float)
    if len(y) < 5:
        raise ValueError("Too few points to determine a CV peak.")

    trim = _edge_trim_points(len(y), edge_trim_fraction)
    start = max(trim, int(round(len(y) * max(float(start_fraction), 0.0))))
    stop = min(len(y) - trim, int(round(len(y) * min(float(end_fraction), 1.0))))
    if stop - start < 3:
        start = 0
        stop = len(y)
        trim = 0

    window = y[start:stop]
    if kind == "max":
        search_signal = window
    elif kind == "min":
        search_signal = -window
    else:
        raise ValueError(f"Unsupported peak kind: {kind}")

    distance = max(3, len(window) // 30)
    prominence = 0.0 if min_peak_prominence_uA is None else float(min_peak_prominence_uA)
    peaks, props = find_peaks(search_signal, prominence=prominence, distance=distance)

    if len(peaks):
        peak_scores = search_signal[peaks]
        best_pos = int(np.argmax(peak_scores))
        peak_local_idx = int(peaks[best_pos])
        peak_idx = start + peak_local_idx
        peak_prominence = float(props.get("prominences", np.zeros(len(peaks), dtype=float))[best_pos])
    else:
        peak_local_idx = int(np.argmax(search_signal))
        peak_idx = start + peak_local_idx
        baseline_level = float(np.nanmedian(search_signal))
        peak_prominence = float(search_signal[peak_local_idx] - baseline_level)

    if min_peak_prominence_uA is not None and peak_prominence < float(min_peak_prominence_uA):
        raise ValueError(
            f"{kind.title()} peak prominence {peak_prominence:.4g} uA "
            f"below cutoff {float(min_peak_prominence_uA):.4g} uA."
        )

    return int(peak_idx), float(peak_prominence), int(trim)


def _combine_cycle_parts(
    forward_voltage: np.ndarray,
    reverse_voltage: np.ndarray,
    forward_current: np.ndarray,
    reverse_current: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if (
        len(forward_voltage)
        and len(reverse_voltage)
        and abs(float(forward_voltage[-1]) - float(reverse_voltage[0])) <= 1e-12
    ):
        v = np.concatenate([forward_voltage, reverse_voltage[1:]])
        y = np.concatenate([forward_current, reverse_current[1:]])
        reverse_offset = len(forward_voltage) - 1
    else:
        v = np.concatenate([forward_voltage, reverse_voltage])
        y = np.concatenate([forward_current, reverse_current])
        reverse_offset = len(forward_voltage)
    return v, y, int(reverse_offset)


def _best_idx_from_candidates(signal: np.ndarray, candidate_idxs: np.ndarray, kind: str) -> Optional[int]:
    idxs = np.asarray(candidate_idxs, dtype=int)
    idxs = idxs[(idxs >= 0) & (idxs < len(signal))]
    if idxs.size == 0:
        return None
    y = np.asarray(signal, dtype=float)
    if kind == "max":
        return int(idxs[np.argmax(y[idxs])])
    if kind == "min":
        return int(idxs[np.argmin(y[idxs])])
    raise ValueError(f"Unsupported kind: {kind}")


def _loop_area_abs(
    forward_voltage: np.ndarray,
    forward_current: np.ndarray,
    reverse_voltage: np.ndarray,
    reverse_current: np.ndarray,
) -> float:
    if len(forward_voltage) < 3 or len(reverse_voltage) < 3:
        return np.nan

    fv = np.asarray(forward_voltage, dtype=float)
    fy = np.asarray(forward_current, dtype=float)
    rv = np.asarray(reverse_voltage, dtype=float)[::-1]
    ry = np.asarray(reverse_current, dtype=float)[::-1]

    lo = max(float(np.min(fv)), float(np.min(rv)))
    hi = min(float(np.max(fv)), float(np.max(rv)))
    if hi <= lo:
        return np.nan

    grid_count = max(200, min(len(fv), len(rv)))
    grid = np.linspace(lo, hi, grid_count)
    f_interp = np.interp(grid, fv, fy)
    r_interp = np.interp(grid, rv, ry)
    return float(np.trapz(np.abs(f_interp - r_interp), grid))


def analyze_cv_arrays(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    crop_range: Tuple[float, float] = (-0.2, 0.9),
    smooth_window: int = 11,
    smooth_polyorder: int = 2,
    edge_trim_fraction: float = 0.05,
    min_peak_prominence_uA: Optional[float] = None,
    file_path: Optional[str] = None,
) -> dict:
    v_full = np.asarray(v_raw, dtype=float)
    i_full = np.asarray(i_raw, dtype=float)
    if len(v_full) < 10:
        raise ValueError("Too few points for CV analysis.")

    turn_idx_full = _estimate_turn_idx(v_full)
    direction = "up_then_down" if float(v_full[turn_idx_full]) >= float(v_full[0]) else "down_then_up"

    forward_voltage_raw = v_full[: turn_idx_full + 1]
    forward_current_raw = i_full[: turn_idx_full + 1]
    reverse_voltage_raw = v_full[turn_idx_full:]
    reverse_current_raw = i_full[turn_idx_full:]

    forward_voltage, forward_current = _crop_by_voltage(
        forward_voltage_raw, forward_current_raw, crop_range=crop_range
    )
    reverse_voltage, reverse_current = _crop_by_voltage(
        reverse_voltage_raw, reverse_current_raw, crop_range=crop_range
    )

    if len(forward_voltage) < 5 or len(reverse_voltage) < 5:
        raise ValueError("Too few forward/reverse points after cropping.")

    forward_smoothed = apply_smoothing(forward_current, smooth_window, smooth_polyorder)
    reverse_smoothed = apply_smoothing(reverse_current, smooth_window, smooth_polyorder)

    forward_baseline = _linear_baseline(forward_voltage, forward_smoothed)
    reverse_baseline = _linear_baseline(reverse_voltage, reverse_smoothed)
    forward_detrended = forward_smoothed - forward_baseline
    reverse_detrended = reverse_smoothed - reverse_baseline

    voltage, raw_current, reverse_offset = _combine_cycle_parts(
        forward_voltage,
        reverse_voltage,
        forward_current,
        reverse_current,
    )
    _, smoothed_current, _ = _combine_cycle_parts(
        forward_voltage,
        reverse_voltage,
        forward_smoothed,
        reverse_smoothed,
    )
    _, baseline_current, _ = _combine_cycle_parts(
        forward_voltage,
        reverse_voltage,
        forward_baseline,
        reverse_baseline,
    )
    _, detrended_current, _ = _combine_cycle_parts(
        forward_voltage,
        reverse_voltage,
        forward_detrended,
        reverse_detrended,
    )

    late_sweep_start_fraction = 0.25
    if direction == "up_then_down":
        ox_idx_local, ox_prominence, ox_trim = _dominant_peak_idx(
            forward_detrended,
            kind="max",
            edge_trim_fraction=edge_trim_fraction,
            min_peak_prominence_uA=min_peak_prominence_uA,
            start_fraction=late_sweep_start_fraction,
        )
        red_idx_local, red_prominence, red_trim = _dominant_peak_idx(
            reverse_detrended,
            kind="min",
            edge_trim_fraction=edge_trim_fraction,
            min_peak_prominence_uA=min_peak_prominence_uA,
            start_fraction=late_sweep_start_fraction,
        )
        oxidation_peak_idx = int(ox_idx_local)
        reduction_peak_idx = int(reverse_offset + red_idx_local)
        oxidation_peak_voltage = float(forward_voltage[ox_idx_local])
        reduction_peak_voltage = float(reverse_voltage[red_idx_local])
        oxidation_peak_current = float(forward_smoothed[ox_idx_local])
        reduction_peak_current = float(reverse_smoothed[red_idx_local])
    else:
        red_idx_local, red_prominence, red_trim = _dominant_peak_idx(
            forward_detrended,
            kind="min",
            edge_trim_fraction=edge_trim_fraction,
            min_peak_prominence_uA=min_peak_prominence_uA,
            start_fraction=late_sweep_start_fraction,
        )
        ox_idx_local, ox_prominence, ox_trim = _dominant_peak_idx(
            reverse_detrended,
            kind="max",
            edge_trim_fraction=edge_trim_fraction,
            min_peak_prominence_uA=min_peak_prominence_uA,
            start_fraction=late_sweep_start_fraction,
        )
        reduction_peak_idx = int(red_idx_local)
        oxidation_peak_idx = int(reverse_offset + ox_idx_local)
        reduction_peak_voltage = float(forward_voltage[red_idx_local])
        oxidation_peak_voltage = float(reverse_voltage[ox_idx_local])
        reduction_peak_current = float(forward_smoothed[red_idx_local])
        oxidation_peak_current = float(reverse_smoothed[ox_idx_local])

    if oxidation_peak_voltage <= reduction_peak_voltage:
        if direction == "up_then_down":
            later_forward_candidates = np.where(forward_voltage > reduction_peak_voltage)[0]
            better_ox = _best_idx_from_candidates(forward_detrended, later_forward_candidates, kind="max")
            if better_ox is not None:
                ox_idx_local = int(better_ox)
                oxidation_peak_idx = int(ox_idx_local)
                oxidation_peak_voltage = float(forward_voltage[ox_idx_local])
                oxidation_peak_current = float(forward_smoothed[ox_idx_local])
                ox_prominence = max(float(ox_prominence), float(forward_detrended[ox_idx_local]))

            earlier_reverse_candidates = np.where(reverse_voltage < oxidation_peak_voltage)[0]
            better_red = _best_idx_from_candidates(reverse_detrended, earlier_reverse_candidates, kind="min")
            if better_red is not None:
                red_idx_local = int(better_red)
                reduction_peak_idx = int(reverse_offset + red_idx_local)
                reduction_peak_voltage = float(reverse_voltage[red_idx_local])
                reduction_peak_current = float(reverse_smoothed[red_idx_local])
                red_prominence = max(float(red_prominence), float(-reverse_detrended[red_idx_local]))
        else:
            later_reverse_candidates = np.where(reverse_voltage > reduction_peak_voltage)[0]
            better_ox = _best_idx_from_candidates(reverse_detrended, later_reverse_candidates, kind="max")
            if better_ox is not None:
                ox_idx_local = int(better_ox)
                oxidation_peak_idx = int(reverse_offset + ox_idx_local)
                oxidation_peak_voltage = float(reverse_voltage[ox_idx_local])
                oxidation_peak_current = float(reverse_smoothed[ox_idx_local])
                ox_prominence = max(float(ox_prominence), float(reverse_detrended[ox_idx_local]))

            earlier_forward_candidates = np.where(forward_voltage < oxidation_peak_voltage)[0]
            better_red = _best_idx_from_candidates(forward_detrended, earlier_forward_candidates, kind="min")
            if better_red is not None:
                red_idx_local = int(better_red)
                reduction_peak_idx = int(red_idx_local)
                reduction_peak_voltage = float(forward_voltage[red_idx_local])
                reduction_peak_current = float(forward_smoothed[red_idx_local])
                red_prominence = max(float(red_prominence), float(-forward_detrended[red_idx_local]))

    peak_separation = float(oxidation_peak_voltage - reduction_peak_voltage)
    denom = abs(reduction_peak_current)
    peak_current_ratio = float(oxidation_peak_current / denom) if denom > 1e-12 else np.nan
    midpoint_voltage = float((oxidation_peak_voltage + reduction_peak_voltage) / 2.0)

    return {
        "file_path": file_path,
        "voltage": voltage,
        "raw_current": raw_current,
        "smoothed_current": smoothed_current,
        "baseline_current": baseline_current,
        "detrended_current": detrended_current,
        "turn_voltage": float(v_full[turn_idx_full]),
        "turn_idx_raw": int(turn_idx_full),
        "cycle_direction": direction,
        "forward_point_count": int(len(forward_voltage)),
        "reverse_point_count": int(len(reverse_voltage)),
        "forward_edge_trim_points": int(ox_trim),
        "reverse_edge_trim_points": int(red_trim),
        "oxidation_peak_idx": oxidation_peak_idx,
        "oxidation_peak_voltage": oxidation_peak_voltage,
        "oxidation_peak_current": oxidation_peak_current,
        "oxidation_peak_prominence": float(ox_prominence),
        "reduction_peak_idx": reduction_peak_idx,
        "reduction_peak_voltage": reduction_peak_voltage,
        "reduction_peak_current": reduction_peak_current,
        "reduction_peak_prominence": float(red_prominence),
        "reduction_peak_abs_current": float(abs(reduction_peak_current)),
        "peak_separation_V": peak_separation,
        "peak_current_ratio": peak_current_ratio,
        "midpoint_voltage": midpoint_voltage,
        "loop_area_abs": _loop_area_abs(
            forward_voltage,
            forward_smoothed,
            reverse_voltage,
            reverse_smoothed,
        ),
        "status": "OK",
    }


def analyze_cv_file(
    filepath: str,
    crop_range: Tuple[float, float] = (-0.2, 0.9),
    voltage_col: str = "Potential (V)",
    current_col: Optional[str] = None,
    smooth_window: int = 11,
    smooth_polyorder: int = 2,
    edge_trim_fraction: float = 0.05,
    min_peak_prominence_uA: Optional[float] = None,
) -> dict:
    file_mtime_ns, file_size = _file_signature(filepath)
    v_raw, i_raw = _load_filtered_arrays_cached(
        filepath=filepath,
        voltage_col=voltage_col,
        current_col=current_col,
        file_mtime_ns=file_mtime_ns,
        file_size=file_size,
    )
    return analyze_cv_arrays(
        v_raw=v_raw,
        i_raw=i_raw,
        crop_range=crop_range,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        edge_trim_fraction=edge_trim_fraction,
        min_peak_prominence_uA=min_peak_prominence_uA,
        file_path=filepath,
    )


def partial_traces_for_failure_arrays(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    crop_range: Tuple[float, float],
    smooth_window: int,
    smooth_polyorder: int,
) -> dict:
    partial = {
        "voltage": None,
        "raw_current": None,
        "smoothed_current": None,
        "baseline_current": None,
        "detrended_current": None,
        "oxidation_peak_idx": None,
        "reduction_peak_idx": None,
        "partial_error": None,
    }
    try:
        v_full = np.asarray(v_raw, dtype=float)
        i_full = np.asarray(i_raw, dtype=float)
        turn_idx_full = _estimate_turn_idx(v_full)

        forward_voltage, forward_current = _crop_by_voltage(
            v_full[: turn_idx_full + 1],
            i_full[: turn_idx_full + 1],
            crop_range=crop_range,
        )
        reverse_voltage, reverse_current = _crop_by_voltage(
            v_full[turn_idx_full:],
            i_full[turn_idx_full:],
            crop_range=crop_range,
        )

        if len(forward_voltage) < 3 or len(reverse_voltage) < 3:
            partial["partial_error"] = "Too few forward/reverse points after cropping."
            return partial

        forward_smoothed = apply_smoothing(forward_current, smooth_window, smooth_polyorder)
        reverse_smoothed = apply_smoothing(reverse_current, smooth_window, smooth_polyorder)
        forward_baseline = _linear_baseline(forward_voltage, forward_smoothed)
        reverse_baseline = _linear_baseline(reverse_voltage, reverse_smoothed)
        forward_detrended = forward_smoothed - forward_baseline
        reverse_detrended = reverse_smoothed - reverse_baseline

        voltage, raw_current, _ = _combine_cycle_parts(
            forward_voltage,
            reverse_voltage,
            forward_current,
            reverse_current,
        )
        _, smoothed_current, _ = _combine_cycle_parts(
            forward_voltage,
            reverse_voltage,
            forward_smoothed,
            reverse_smoothed,
        )
        _, baseline_current, _ = _combine_cycle_parts(
            forward_voltage,
            reverse_voltage,
            forward_baseline,
            reverse_baseline,
        )
        _, detrended_current, _ = _combine_cycle_parts(
            forward_voltage,
            reverse_voltage,
            forward_detrended,
            reverse_detrended,
        )

        partial.update(
            voltage=voltage,
            raw_current=raw_current,
            smoothed_current=smoothed_current,
            baseline_current=baseline_current,
            detrended_current=detrended_current,
        )
        return partial
    except Exception as exc:
        partial["partial_error"] = str(exc)
        return partial


def analyze_cv_cycles_from_arrays(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    crop_range: Tuple[float, float] = (-0.2, 0.9),
    smooth_window: int = 11,
    smooth_polyorder: int = 2,
    edge_trim_fraction: float = 0.05,
    min_peak_prominence_uA: Optional[float] = None,
    file_path: Optional[str] = None,
    expected_cycles: Optional[int] = None,
) -> List[dict]:
    cycle_bounds = _cycle_boundaries_from_voltage(v_raw, expected_cycles=expected_cycles)
    out: List[dict] = []
    for cycle_idx, (start_idx, end_idx) in enumerate(cycle_bounds, start=1):
        v_cycle = np.asarray(v_raw[start_idx:end_idx + 1], dtype=float)
        i_cycle = np.asarray(i_raw[start_idx:end_idx + 1], dtype=float)
        common = {
            "cycle_number": int(cycle_idx),
            "cycle_start_idx_raw": int(start_idx),
            "cycle_end_idx_raw": int(end_idx),
            "cycle_point_count": int(len(v_cycle)),
        }
        try:
            result = analyze_cv_arrays(
                v_raw=v_cycle,
                i_raw=i_cycle,
                crop_range=crop_range,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
                edge_trim_fraction=edge_trim_fraction,
                min_peak_prominence_uA=min_peak_prominence_uA,
                file_path=file_path,
            )
            result.update(common)
            out.append(result)
        except Exception as exc:
            partial = partial_traces_for_failure_arrays(
                v_raw=v_cycle,
                i_raw=i_cycle,
                crop_range=crop_range,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
            )
            out.append({
                **common,
                "status": "FAILED",
                "error": str(exc),
                **partial,
            })
    return out


@lru_cache(maxsize=256)
def _process_cv_file_cached(
    filepath: str,
    voltage_col: str,
    current_col: Optional[str],
    file_mtime_ns: int,
    file_size: int,
    crop_range: Tuple[float, float],
    smooth_window: int,
    smooth_polyorder: int,
    edge_trim_fraction: float,
    min_peak_prominence_uA: Optional[float],
) -> dict:
    v_raw, i_raw = _load_filtered_arrays_cached(
        filepath=filepath,
        voltage_col=voltage_col,
        current_col=current_col,
        file_mtime_ns=file_mtime_ns,
        file_size=file_size,
    )
    try:
        result = analyze_cv_arrays(
            v_raw=v_raw,
            i_raw=i_raw,
            crop_range=crop_range,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            edge_trim_fraction=edge_trim_fraction,
            min_peak_prominence_uA=min_peak_prominence_uA,
            file_path=filepath,
        )
        return {"status": "OK", "result": result, "partial": None, "error": None}
    except Exception as exc:
        partial = partial_traces_for_failure_arrays(
            v_raw=v_raw,
            i_raw=i_raw,
            crop_range=crop_range,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
        )
        return {"status": "FAILED", "result": None, "partial": partial, "error": str(exc)}


def compute_cv_drift_fields(all_results: List[dict]) -> List[dict]:
    ref: Dict[Tuple[int, str, str], dict] = {}
    sorted_results = sorted(
        all_results,
        key=lambda r: (
            r["channel"],
            r.get("ec_label", ""),
            r.get("file_path", ""),
            r["scan_number"],
        ),
    )

    for r in sorted_results:
        key = (r["channel"], r.get("ec_label", ""), r.get("file_path", ""))
        if r.get("status") != "OK":
            r["oxidation_peak_voltage_drift"] = np.nan
            r["reduction_peak_voltage_drift"] = np.nan
            r["peak_separation_drift"] = np.nan
            r["oxidation_peak_current_drift"] = np.nan
            r["reduction_peak_current_drift"] = np.nan
            r["loop_area_abs_drift"] = np.nan
            continue

        if key not in ref:
            ref[key] = r

        r["oxidation_peak_voltage_drift"] = (
            r["oxidation_peak_voltage"] - ref[key]["oxidation_peak_voltage"]
        )
        r["reduction_peak_voltage_drift"] = (
            r["reduction_peak_voltage"] - ref[key]["reduction_peak_voltage"]
        )
        r["peak_separation_drift"] = r["peak_separation_V"] - ref[key]["peak_separation_V"]
        r["oxidation_peak_current_drift"] = (
            r["oxidation_peak_current"] - ref[key]["oxidation_peak_current"]
        )
        r["reduction_peak_current_drift"] = (
            r["reduction_peak_current"] - ref[key]["reduction_peak_current"]
        )
        r["loop_area_abs_drift"] = r["loop_area_abs"] - ref[key]["loop_area_abs"]

    return all_results


def run_cv_batch(
    folders: List[str],
    crop_range: Tuple[float, float] = (-0.2, 0.9),
    voltage_col: str = "Potential (V)",
    current_col: Optional[str] = None,
    smooth_window: int = 11,
    smooth_polyorder: int = 2,
    edge_trim_fraction: float = 0.05,
    min_peak_prominence_uA: Optional[float] = None,
    scan_windows: Optional[Tuple[Tuple[int, int], ...]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
    progress_callback=None,
) -> List[dict]:
    files = collect_cv_csvs_from_folders(folders)
    if not files:
        raise ValueError("No CV CSVs found.")

    by_ch = group_by_channel_and_sort(files)
    all_results: List[dict] = []

    ordered: List[Tuple[int, MeasurementFile]] = [
        (ch, f)
        for ch, flist in sorted(by_ch.items())
        for f in flist
    ]

    total = len(ordered)
    measurement_counters: Dict[int, int] = {}

    for idx, (ch, f) in enumerate(ordered):
        if progress_callback:
            progress_callback(idx + 1, total, os.path.basename(f.path))

        try:
            file_mtime_ns, file_size = _file_signature(f.path)
            v_raw, i_raw = _load_filtered_arrays_cached(
                filepath=f.path,
                voltage_col=voltage_col,
                current_col=current_col,
                file_mtime_ns=file_mtime_ns,
                file_size=file_size,
            )
        except OSError:
            continue
        except Exception as exc:
            all_results.append({
                "channel": ch,
                "channel_label": f"Ch{ch}",
                "timestamp": f.ts,
                "scan_id_from_name": f.scan,
                "original_scan_number": np.nan,
                "scan_number": np.nan,
                "folder_index": f.folder_index,
                "file_path": f.path,
                "file_name": os.path.basename(f.path),
                "status": "FAILED",
                "error": str(exc),
            })
            continue

        measurement_counters[ch] = measurement_counters.get(ch, 0) + 1
        measurement_index = measurement_counters[ch]
        method_meta = load_cv_method_metadata(_infer_method_path(f.path))
        expected_cycles = method_meta.get("nscans")

        try:
            cycle_results = analyze_cv_cycles_from_arrays(
                v_raw=v_raw,
                i_raw=i_raw,
                crop_range=crop_range,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
                edge_trim_fraction=edge_trim_fraction,
                min_peak_prominence_uA=min_peak_prominence_uA,
                file_path=f.path,
                expected_cycles=expected_cycles,
            )
        except Exception as exc:
            partial = partial_traces_for_failure_arrays(
                v_raw=v_raw,
                i_raw=i_raw,
                crop_range=crop_range,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
            )
            all_results.append({
                "channel": ch,
                "channel_label": f"Ch{ch}",
                "timestamp": f.ts,
                "scan_id_from_name": f.scan,
                "original_scan_number": np.nan,
                "scan_number": np.nan,
                "measurement_index": measurement_index,
                "folder_index": f.folder_index,
                "file_path": f.path,
                "file_name": os.path.basename(f.path),
                "method_path": method_meta.get("method_path"),
                "method_nscans": method_meta.get("nscans"),
                "ec_label": method_meta.get("ec_label"),
                "status": "FAILED",
                "error": str(exc),
                **{
                    k: partial.get(k)
                    for k in (
                        "voltage",
                        "raw_current",
                        "smoothed_current",
                        "baseline_current",
                        "detrended_current",
                        "oxidation_peak_idx",
                        "reduction_peak_idx",
                        "partial_error",
                    )
                },
            })
            continue

        cycle_count_in_file = len(cycle_results)
        common_file = dict(
            channel=ch,
            channel_label=f"Ch{ch}",
            timestamp=f.ts,
            scan_id_from_name=f.scan,
            measurement_index=measurement_index,
            folder_index=f.folder_index,
            file_path=f.path,
            file_name=os.path.basename(f.path),
            method_path=method_meta.get("method_path"),
            method_nscans=method_meta.get("nscans"),
            ec_label=method_meta.get("ec_label"),
            start_voltage=method_meta.get("start_voltage"),
            vertex1_voltage=method_meta.get("vertex1_voltage"),
            vertex2_voltage=method_meta.get("vertex2_voltage"),
            step_voltage=method_meta.get("step_voltage"),
            scan_rate_v_per_s=method_meta.get("scan_rate_v_per_s"),
            cycle_count_in_file=cycle_count_in_file,
        )

        for cycle_result in cycle_results:
            cycle_number = int(cycle_result["cycle_number"])
            if not _scan_in_windows(cycle_number, scan_windows=scan_windows, scan_range=scan_range):
                continue
            analysis_cycle_number = _remap_scan_number(
                cycle_number,
                scan_windows=scan_windows,
                scan_range=scan_range,
            )
            r = dict(cycle_result)
            r.update(common_file)
            r["original_scan_number"] = cycle_number
            r["scan_number"] = analysis_cycle_number
            if r.get("status") == "FAILED":
                r.setdefault("oxidation_peak_voltage", np.nan)
                r.setdefault("oxidation_peak_current", np.nan)
                r.setdefault("oxidation_peak_prominence", np.nan)
                r.setdefault("reduction_peak_voltage", np.nan)
                r.setdefault("reduction_peak_current", np.nan)
                r.setdefault("reduction_peak_prominence", np.nan)
                r.setdefault("reduction_peak_abs_current", np.nan)
                r.setdefault("peak_separation_V", np.nan)
                r.setdefault("peak_current_ratio", np.nan)
                r.setdefault("midpoint_voltage", np.nan)
                r.setdefault("loop_area_abs", np.nan)
            all_results.append(r)

    compute_cv_drift_fields(all_results)
    return all_results
