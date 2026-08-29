from typing import Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def apply_smoothing(i: np.ndarray, smooth_window: int, smooth_polyorder: int) -> np.ndarray:
    w = int(smooth_window)
    if w >= len(i):
        w = max(3, (len(i) // 2) * 2 + 1)
    if w % 2 == 0:
        w += 1
    p = int(min(smooth_polyorder, w - 1))
    return savgol_filter(i, window_length=w, polyorder=p)


def find_peak_candidates(
    i_smooth: np.ndarray,
    prominence: float = 0.02,
    distance: int = 5,
    boundary_margin: int = 5,
) -> dict:
    raw_peaks, _ = find_peaks(i_smooth, distance=distance)
    raw_peaks = raw_peaks.astype(int)
    raw_valid_peaks = np.array(
        [p for p in raw_peaks if boundary_margin < p < len(i_smooth) - boundary_margin],
        dtype=int,
    )

    peaks_by_pass = []
    valid_peaks = []

    for prom in (prominence, 0.005):
        peaks, props = find_peaks(i_smooth, prominence=prom, distance=distance)
        peaks = peaks.astype(int)
        valid = np.array(
            [p for p in peaks if boundary_margin < p < len(i_smooth) - boundary_margin],
            dtype=int,
        )
        peaks_by_pass.append({
            "prominence": prom,
            "all_peaks": peaks,
            "valid_peaks": valid,
            "prominences": props.get("prominences"),
        })
        if valid.size:
            valid_peaks = valid
            break

    dominant_idx = None
    if len(valid_peaks):
        dominant_idx = int(valid_peaks[np.argmax(i_smooth[valid_peaks])])
    else:
        idx = int(np.argmax(i_smooth))
        dominant_idx = max(boundary_margin, min(idx, len(i_smooth) - boundary_margin - 1))

    return {
        "raw_peaks": raw_peaks,
        "raw_valid_peaks": raw_valid_peaks,
        "passes": peaks_by_pass,
        "valid_peaks": np.asarray(valid_peaks, dtype=int),
        "dominant_idx": int(dominant_idx),
    }


def detect_dominant_peak(
    i_smooth: np.ndarray,
    prominence: float = 0.02,
    distance: int = 5,
    boundary_margin: int = 5,
) -> int:
    return find_peak_candidates(
        i_smooth,
        prominence=prominence,
        distance=distance,
        boundary_margin=boundary_margin,
    )["dominant_idx"]


def _estimate_point_spacing(voltage: np.ndarray) -> float:
    diffs = np.abs(np.diff(np.asarray(voltage, dtype=float)))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-12)]
    return float(np.median(diffs)) if diffs.size else 1.0


def _support_points_for_window(voltage: np.ndarray, window_V: float, fraction: float = 0.10) -> int:
    dv = _estimate_point_spacing(voltage)
    target_V = max(float(window_V) * float(fraction), dv * 2.0)
    pts = int(round(target_V / max(dv, 1e-12)))
    return max(2, pts)


def _window_indices(
    voltage: np.ndarray,
    peak_idx: int,
    search_window_V: float,
) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(voltage, dtype=float)
    v_peak = float(v[int(peak_idx)])

    left_idxs = np.where((v >= v_peak - search_window_V) & (v < v_peak))[0]
    if left_idxs.size == 0:
        left_idxs = np.arange(0, int(peak_idx))

    right_idxs = np.where((v <= v_peak + search_window_V) & (v > v_peak))[0]
    if right_idxs.size == 0:
        right_idxs = np.arange(int(peak_idx) + 1, len(v))

    return left_idxs.astype(int), right_idxs.astype(int)


def _peak_has_expected_flanks(
    y: np.ndarray,
    peak_idx: int,
    flank_points: int,
) -> bool:
    left_start = int(peak_idx) - int(flank_points)
    right_end = int(peak_idx) + int(flank_points)
    if left_start < 0 or right_end >= len(y):
        return False

    left_vals = y[left_start:int(peak_idx) + 1]
    right_vals = y[int(peak_idx):right_end + 1]
    left_diff = np.diff(left_vals)
    right_diff = np.diff(right_vals)
    if left_diff.size == 0 or right_diff.size == 0:
        return False

    left_rising_frac = float(np.mean(left_diff > 0))
    right_falling_frac = float(np.mean(right_diff < 0))

    # Require the local flanks to mostly move in the expected direction, but
    # keep a weak net-height check so a monotone edge shoulder is not accepted.
    if left_rising_frac < 0.60 or right_falling_frac < 0.60:
        return False

    eps = 0.05 * max(float(np.max(y) - np.min(y)), 1e-12)
    left_gain = float(y[int(peak_idx)] - left_vals[0])
    right_gain = float(y[int(peak_idx)] - right_vals[-1])
    return bool(left_gain > eps and right_gain > eps)


def _candidate_peak_indices(y: np.ndarray, peak_idx: int) -> np.ndarray:
    candidates = find_peak_candidates(y, boundary_margin=0)
    merged = np.concatenate((
        np.asarray([peak_idx], dtype=int),
        np.asarray(candidates.get("valid_peaks", []), dtype=int),
        np.asarray(candidates.get("raw_valid_peaks", []), dtype=int),
    ))
    merged = merged[(merged >= 0) & (merged < len(y))]
    if not merged.size:
        return np.asarray([int(peak_idx)], dtype=int)
    unique = np.unique(merged)
    return unique[np.argsort(y[unique])[::-1]]


def _select_bracketing_peak_idx(
    voltage: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    search_window_V: float,
) -> int:
    flank_points = _support_points_for_window(voltage, search_window_V)

    for candidate_idx in _candidate_peak_indices(y, peak_idx):
        left_idxs, right_idxs = _window_indices(voltage, int(candidate_idx), search_window_V)
        if left_idxs.size == 0 or right_idxs.size == 0:
            continue
        # Only reject candidates that are truly too close to the crop edge.
        if left_idxs.size < 2 or right_idxs.size < 2:
            continue
        if not _peak_has_expected_flanks(y, int(candidate_idx), flank_points):
            continue
        return int(candidate_idx)

    return int(peak_idx)


def _linear_baseline_from_indices(
    voltage: np.ndarray,
    y: np.ndarray,
    left_idx: int,
    right_idx: int,
) -> np.ndarray:
    v = np.asarray(voltage, dtype=float)
    y = np.asarray(y, dtype=float)
    v0, v1 = float(v[left_idx]), float(v[right_idx])
    y0, y1 = float(y[left_idx]), float(y[right_idx])

    denom = (v1 - v0) if abs(v1 - v0) > 1e-12 else 1e-12
    slope = (y1 - y0) / denom
    return slope * v + (y0 - slope * v0)


def rotate_offset_using_bracketing_minima(
    voltage: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    search_window_V: float = 0.12,
) -> dict:
    v = np.asarray(voltage, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(v) < 5:
        raise ValueError("Too few points to compute bracketing minima baseline.")
    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        raise ValueError("Peak index is on/near boundary.")

    peak_idx = _select_bracketing_peak_idx(v, y, int(peak_idx), search_window_V)
    left_idxs, right_idxs = _window_indices(v, peak_idx, search_window_V)
    left_idx = int(left_idxs[np.argmin(y[left_idxs])])
    right_idx = int(right_idxs[np.argmin(y[right_idxs])])

    if right_idx <= left_idx:
        raise ValueError("Failed to find valid left/right minima (indices overlap).")

    baseline = _linear_baseline_from_indices(v, y, left_idx, right_idx)

    return {
        "y_corrected": y - baseline,
        "local_baseline": baseline,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "left_local_min_candidates": np.array([], dtype=int),
        "right_local_min_candidates": np.array([], dtype=int),
        "minima_mode": "argmin_window",
    }


def rotate_offset_using_prominent_bracketing_minima(
    voltage: np.ndarray,
    y: np.ndarray,
    peak_idx: int,
    search_window_V: float = 0.12,
    distance: int = 3,
) -> dict:
    v = np.asarray(voltage, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(v) < 5:
        raise ValueError("Too few points to compute bracketing minima baseline.")
    if peak_idx <= 0 or peak_idx >= len(y) - 1:
        raise ValueError("Peak index is on/near boundary.")

    peak_idx = _select_bracketing_peak_idx(v, y, int(peak_idx), search_window_V)
    left_window_idxs, right_window_idxs = _window_indices(v, peak_idx, search_window_V)

    y_inv = -y
    minima_idxs, props = find_peaks(y_inv, prominence=0, distance=distance)
    minima_idxs = minima_idxs.astype(int)
    prominences = np.asarray(props.get("prominences", np.zeros(len(minima_idxs))), dtype=float)

    if minima_idxs.size == 0:
        fallback = rotate_offset_using_bracketing_minima(v, y, peak_idx, search_window_V)
        fallback.update({
            "left_local_min_candidates": np.array([], dtype=int),
            "right_local_min_candidates": np.array([], dtype=int),
            "minima_mode": "prominent_local_minima_fallback",
        })
        return fallback

    min_peak_separation_pts = max(distance, _support_points_for_window(v, search_window_V, fraction=0.08))

    def _pick_side(window_idxs: np.ndarray, sign: int) -> Tuple[np.ndarray, np.ndarray]:
        in_window = np.isin(minima_idxs, window_idxs)

        for sep_frac in (1.0, 0.5, 0.0):
            sep = int(round(min_peak_separation_pts * sep_frac))
            dist_ok = (sign * (minima_idxs - peak_idx)) >= sep
            mask = in_window & dist_ok
            candidates = minima_idxs[mask]
            prom = prominences[mask]
            if candidates.size:
                return candidates, prom

        return np.array([], dtype=int), np.array([], dtype=float)

    left_candidates, left_prom = _pick_side(left_window_idxs, sign=-1)
    right_candidates, right_prom = _pick_side(right_window_idxs, sign=+1)

    if left_candidates.size == 0 or right_candidates.size == 0:
        fallback = rotate_offset_using_bracketing_minima(v, y, peak_idx, search_window_V)
        fallback.update({
            "left_local_min_candidates": left_candidates,
            "right_local_min_candidates": right_candidates,
            "minima_mode": "prominent_local_minima_fallback",
        })
        return fallback

    left_order = np.argsort(-left_prom)
    right_order = np.argsort(-right_prom)
    left_candidates = left_candidates[left_order]
    right_candidates = right_candidates[right_order]
    left_idx = int(left_candidates[0])
    right_idx = int(right_candidates[0])

    if right_idx <= left_idx:
        fallback = rotate_offset_using_bracketing_minima(v, y, peak_idx, search_window_V)
        fallback.update({
            "left_local_min_candidates": left_candidates,
            "right_local_min_candidates": right_candidates,
            "minima_mode": "prominent_local_minima_fallback",
        })
        return fallback

    baseline = _linear_baseline_from_indices(v, y, left_idx, right_idx)

    return {
        "y_corrected": y - baseline,
        "local_baseline": baseline,
        "left_idx": left_idx,
        "right_idx": right_idx,
        "left_local_min_candidates": left_candidates,
        "right_local_min_candidates": right_candidates,
        "minima_mode": "prominent_local_minima",
    }
