import os
import re
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
from scipy.stats import skew

from .io import (
    SWVFile,
    collect_swv_csvs_from_folders,
    filter_finite,
    group_by_channel_and_sort,
    load_swv_csv,
)
from .processing import (
    apply_smoothing,
    detect_dominant_peak,
    rotate_offset_using_prominent_bracketing_minima,
    rotate_offset_using_bracketing_minima,
)


def is_peak_height_below_cutoff_error(error: object) -> bool:
    """Return whether an error represents an expected minimum-peak rejection."""
    message = str(error or "").strip().lower()
    return "peak height" in message and "below cutoff" in message

_NUMBER_TOKEN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SWV_SETTINGS_RE = re.compile(
    rf"(?P<start>{_NUMBER_TOKEN}m?)[\s,]+"
    rf"(?P<end>{_NUMBER_TOKEN}m?)[\s,]+"
    rf"(?P<step>{_NUMBER_TOKEN}m?)[\s,]+"
    rf"(?P<amplitude>{_NUMBER_TOKEN}m?)[\s,]+"
    rf"(?P<frequency>{_NUMBER_TOKEN})(?=\s|$)",
    re.IGNORECASE,
)


def _file_signature(filepath: str) -> Tuple[int, int]:
    stat = os.stat(filepath)
    return int(stat.st_mtime_ns), int(stat.st_size)


def _method_search_roots(csv_path: str) -> List[str]:
    folder = os.path.dirname(csv_path)
    return [
        os.path.join(folder, "methods_used"),
        os.path.join(folder, "Methods Used"),
        os.path.join(os.path.dirname(folder), "methods_used"),
        os.path.join(os.path.dirname(folder), "Methods Used"),
    ]


def _build_method_file_index(files: List[SWVFile]) -> Dict[str, Dict[str, str]]:
    """Index each possible method directory once for fast batch lookups."""
    search_roots = {
        search_root
        for measurement in files
        for search_root in _method_search_roots(measurement.path)
    }
    indexes: Dict[str, Dict[str, str]] = {}
    for search_root in search_roots:
        names: Dict[str, str] = {}
        try:
            with os.scandir(search_root) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(".ms"):
                        names.setdefault(entry.name.lower(), entry.path)
        except OSError:
            pass
        indexes[search_root] = names
    return indexes


def _infer_method_path(
    csv_path: str,
    method_file_index: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    folder = os.path.dirname(csv_path)
    stem, _ = os.path.splitext(os.path.basename(csv_path))
    search_roots = _method_search_roots(csv_path)
    wanted_names = {f"{stem}.ms".lower(), f"{stem}.csv.ms".lower()}
    for search_root in search_roots:
        if method_file_index is not None:
            indexed_names = method_file_index.get(search_root, {})
            matching_path = next(
                (indexed_names[name] for name in wanted_names if name in indexed_names),
                None,
            )
            if matching_path is not None:
                return matching_path
            continue
        if not os.path.isdir(search_root):
            continue
        try:
            names = os.listdir(search_root)
        except OSError:
            continue
        matching_name = next(
            (name for name in names if name.lower() in wanted_names),
            None,
        )
        if matching_name is not None:
            return os.path.join(search_root, matching_name)
    return os.path.join(folder, "methods_used", f"{stem}.ms")


def _infer_method_path_direct(csv_path: str) -> str:
    """Resolve one method path without listing a potentially large directory."""
    folder = os.path.dirname(csv_path)
    stem, _ = os.path.splitext(os.path.basename(csv_path))
    candidate_names = (f"{stem}.ms", f"{stem}.csv.ms")
    for search_root in _method_search_roots(csv_path):
        for candidate_name in candidate_names:
            candidate_path = os.path.join(search_root, candidate_name)
            if os.path.isfile(candidate_path):
                return candidate_path
    return os.path.join(folder, "methods_used", f"{stem}.ms")


def _parse_voltage_token(token: str) -> float:
    text = token.strip()
    if text.lower().endswith("m"):
        return float(text[:-1]) / 1000.0
    return float(text)


def _format_frequency_label(frequency_hz: Optional[float]) -> str:
    if frequency_hz is None:
        return "Unknown method"
    if float(frequency_hz).is_integer():
        return f"{int(frequency_hz)} Hz"
    return f"{float(frequency_hz):g} Hz"


@lru_cache(maxsize=512)
def _load_swv_method_metadata_cached(
    method_path: str,
    method_mtime_ns: Optional[int],
    method_size: Optional[int],
) -> dict:
    # File metadata is part of the cache key so newly created or updated method
    # files cannot remain stuck behind an earlier "missing" or stale parse.
    del method_mtime_ns, method_size
    meta = {
        "method_path": method_path,
        "method_exists": False,
        "swv_frequency_hz": None,
        "swv_method_group": "Unknown method",
    }
    if not os.path.exists(method_path):
        return meta

    meta["method_exists"] = True
    with open(method_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    settings_match = None
    for line in text.splitlines():
        if "meas_loop_swv" not in line.lower():
            continue
        matches = list(SWV_SETTINGS_RE.finditer(line))
        if matches:
            settings_match = matches[-1]
            break
    if settings_match is None:
        return meta

    frequency_hz = float(settings_match.group("frequency"))
    meta["swv_frequency_hz"] = frequency_hz
    meta["swv_method_group"] = _format_frequency_label(frequency_hz)
    meta["swv_sweep_start_V"] = _parse_voltage_token(settings_match.group("start"))
    meta["swv_sweep_end_V"] = _parse_voltage_token(settings_match.group("end"))
    meta["swv_step_size_V"] = _parse_voltage_token(settings_match.group("step"))
    meta["swv_amplitude_V"] = _parse_voltage_token(settings_match.group("amplitude"))
    return meta


def load_swv_method_metadata(method_path: str) -> dict:
    try:
        stat = os.stat(method_path)
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        signature = (None, None)
    return _load_swv_method_metadata_cached(method_path, *signature)


@lru_cache(maxsize=512)
def _load_filtered_arrays_cached(
    filepath: str,
    voltage_col: str,
    current_col: Optional[str],
    file_mtime_ns: int,
    file_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Include file metadata in the cache key so edits invalidate cached arrays.
    del file_mtime_ns, file_size
    v_raw, i_raw = load_swv_csv(filepath, voltage_col=voltage_col, current_col=current_col)
    v_raw, i_raw = filter_finite(v_raw, i_raw)
    return np.asarray(v_raw, dtype=float), np.asarray(i_raw, dtype=float)


@lru_cache(maxsize=256)
def _process_file_cached(
    filepath: str,
    voltage_col: str,
    current_col: Optional[str],
    file_mtime_ns: int,
    file_size: int,
    crop_range: Tuple[float, float],
    smooth_window: int,
    smooth_polyorder: int,
    minima_search_window_V: float,
    use_prominent_minima: bool,
    use_double_correction: bool,
    min_peak_height_uA: Optional[float],
    compute_skew: bool,
    compute_wavelet_energy: bool,
    compute_wavelet_denoised_trace: bool,
    use_wavelet_for_correction: bool,
) -> dict:
    v_raw, i_raw = _load_filtered_arrays_cached(
        filepath=filepath,
        voltage_col=voltage_col,
        current_col=current_col,
        file_mtime_ns=file_mtime_ns,
        file_size=file_size,
    )
    try:
        result = analyze_swv_arrays(
            v_raw=v_raw,
            i_raw=i_raw,
            crop_range=crop_range,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            minima_search_window_V=minima_search_window_V,
            use_prominent_minima=use_prominent_minima,
            use_double_correction=use_double_correction,
            min_peak_height_uA=min_peak_height_uA,
            compute_skew=compute_skew,
            compute_wavelet_energy=compute_wavelet_energy,
            compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
            use_wavelet_for_correction=use_wavelet_for_correction,
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
            minima_search_window_V=minima_search_window_V,
            use_prominent_minima=use_prominent_minima,
            use_double_correction=use_double_correction,
            compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
            use_wavelet_for_correction=use_wavelet_for_correction,
        )
        return {
            "status": "FAILED",
            "result": None,
            "partial": partial,
            "error": (
                None
                if is_peak_height_below_cutoff_error(exc)
                else str(exc)
            ),
        }


def _run_correction_pass(
    v: np.ndarray,
    y_for_correction: np.ndarray,
    smooth_window: int,
    smooth_polyorder: int,
    minima_search_window_V: float,
    use_prominent_minima: bool,
    peak_source: Optional[np.ndarray] = None,
    peak_idx: Optional[int] = None,
) -> dict:
    y_corr_input = np.asarray(y_for_correction, dtype=float)
    peak_signal = np.asarray(peak_source if peak_source is not None else y_corr_input, dtype=float)
    selected_peak_idx = int(detect_dominant_peak(peak_signal) if peak_idx is None else peak_idx)

    corr = (
        rotate_offset_using_prominent_bracketing_minima(v, y_corr_input, selected_peak_idx, minima_search_window_V)
        if use_prominent_minima
        else rotate_offset_using_bracketing_minima(v, y_corr_input, selected_peak_idx, minima_search_window_V)
    )
    y_corr = np.asarray(corr["y_corrected"], dtype=float)
    y_corr_smooth = (
        apply_smoothing(y_corr, smooth_window, smooth_polyorder)
        if smooth_window > 0 else y_corr.copy()
    )
    left_idx, right_idx = int(corr["left_idx"]), int(corr["right_idx"])
    segment = y_corr_smooth[left_idx:right_idx + 1]
    peak_idx_corr = left_idx + detect_dominant_peak(segment, boundary_margin=0)

    return {
        "peak_idx": selected_peak_idx,
        "peak_idx_corr": int(peak_idx_corr),
        "corrected_current": y_corr,
        "smoothed_corrected_current": y_corr_smooth,
        "local_baseline": np.asarray(corr["local_baseline"], dtype=float),
        "left_idx": left_idx,
        "right_idx": right_idx,
        "left_local_min_candidates": np.asarray(corr.get("left_local_min_candidates", []), dtype=int),
        "right_local_min_candidates": np.asarray(corr.get("right_local_min_candidates", []), dtype=int),
        "minima_mode": corr.get("minima_mode", "argmin_window"),
    }


def _wavelet_denoise_trace(y: np.ndarray) -> np.ndarray:
    signal = np.asarray(y, dtype=float)
    if signal.size < 8:
        return signal.copy()

    pad = max(8, min(signal.size - 1, signal.size // 3))
    padded = np.pad(signal, pad_width=pad, mode="reflect")
    wavelet = "sym4"
    max_level = pywt.dwt_max_level(len(padded), pywt.Wavelet(wavelet).dec_len)
    level = max(1, min(4, max_level))
    coeffs = pywt.wavedec(padded, wavelet=wavelet, mode="symmetric", level=level)
    if len(coeffs) < 2:
        return signal.copy()

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    threshold = float(sigma * np.sqrt(2.0 * np.log(len(padded)))) if sigma > 0 else 0.0
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode="soft"))

    reconstructed = pywt.waverec(denoised_coeffs, wavelet=wavelet, mode="symmetric")
    trimmed = np.asarray(reconstructed[pad:pad + signal.size], dtype=float)
    if trimmed.size != signal.size:
        trimmed = np.resize(trimmed, signal.shape)
    return trimmed


def _compute_minima_bracket_rms_noise(current: np.ndarray, left_idx: int, right_idx: int) -> float:
    """Estimate white noise without treating the trace's DC level as noise."""
    signal = np.asarray(current, dtype=float)
    if signal.size == 0 or left_idx is None or right_idx is None:
        return np.nan
    try:
        lo = max(0, min(int(left_idx), int(right_idx)))
        hi = min(signal.size - 1, max(int(left_idx), int(right_idx)))
    except (TypeError, ValueError):
        return np.nan
    segment = signal[lo:hi + 1]
    segment = segment[np.isfinite(segment)]
    if segment.size < 2:
        return np.nan
    differences = np.diff(segment)
    if differences.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(differences ** 2)) / np.sqrt(2.0))


def _compute_outside_crop_median(i_raw: np.ndarray, crop_mask: np.ndarray) -> float:
    signal = np.asarray(i_raw, dtype=float)
    outside = signal[~np.asarray(crop_mask, dtype=bool)]
    if outside.size == 0:
        return np.nan
    return float(np.median(outside))


def analyze_swv_file(
    filepath: str,
    crop_range: Tuple[float, float] = (-0.6, -0.2),
    voltage_col: str = "Potential (V)",
    current_col: Optional[str] = None,
    smooth_window: int = 9,
    smooth_polyorder: int = 2,
    minima_search_window_V: float = 0.30,
    use_prominent_minima: bool = False,
    use_double_correction: bool = False,
    min_peak_height_uA: Optional[float] = None,
    compute_skew: bool = True,
    compute_wavelet_energy: bool = True,
    compute_wavelet_denoised_trace: bool = False,
    use_wavelet_for_correction: bool = False,
) -> dict:
    file_mtime_ns, file_size = _file_signature(filepath)
    v_raw, i_raw = _load_filtered_arrays_cached(
        filepath=filepath,
        voltage_col=voltage_col,
        current_col=current_col,
        file_mtime_ns=file_mtime_ns,
        file_size=file_size,
    )

    return analyze_swv_arrays(
        v_raw=v_raw,
        i_raw=i_raw,
        crop_range=crop_range,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        minima_search_window_V=minima_search_window_V,
        use_prominent_minima=use_prominent_minima,
        use_double_correction=use_double_correction,
        min_peak_height_uA=min_peak_height_uA,
        compute_skew=compute_skew,
        compute_wavelet_energy=compute_wavelet_energy,
        compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
        use_wavelet_for_correction=use_wavelet_for_correction,
        file_path=filepath,
    )


def analyze_swv_arrays(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    crop_range: Tuple[float, float] = (-0.6, -0.2),
    smooth_window: int = 9,
    smooth_polyorder: int = 2,
    minima_search_window_V: float = 0.30,
    use_prominent_minima: bool = False,
    use_double_correction: bool = False,
    min_peak_height_uA: Optional[float] = None,
    compute_skew: bool = True,
    compute_wavelet_energy: bool = True,
    compute_wavelet_denoised_trace: bool = False,
    use_wavelet_for_correction: bool = False,
    file_path: Optional[str] = None,
) -> dict:
    mask = (v_raw >= crop_range[0]) & (v_raw <= crop_range[1])
    background_median = _compute_outside_crop_median(i_raw, mask)
    v, i = v_raw[mask], i_raw[mask]

    if len(v) < 5:
        raise ValueError("Too few points after cropping.")

    i_smooth = apply_smoothing(i, smooth_window, smooth_polyorder) if smooth_window > 0 else i.copy()
    wavelet_denoised_current = (
        _wavelet_denoise_trace(i)
        if (compute_wavelet_denoised_trace or use_wavelet_for_correction)
        else None
    )
    first_pass_input = (
        wavelet_denoised_current
        if use_wavelet_for_correction and wavelet_denoised_current is not None
        else i_smooth
    )
    first_pass = _run_correction_pass(
        v=v,
        y_for_correction=first_pass_input,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        minima_search_window_V=minima_search_window_V,
        use_prominent_minima=use_prominent_minima,
    )
    final_pass = first_pass
    second_pass = None
    double_correction_error = None
    if use_double_correction:
        try:
            second_pass = _run_correction_pass(
                v=v,
                y_for_correction=first_pass["corrected_current"],
                peak_source=first_pass["smoothed_corrected_current"],
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
                minima_search_window_V=minima_search_window_V,
                use_prominent_minima=use_prominent_minima,
            )
            final_pass = second_pass
        except Exception as exc:
            double_correction_error = str(exc)

    y_corr = final_pass["corrected_current"]
    y_corr_smooth = final_pass["smoothed_corrected_current"]
    left_idx, right_idx = int(final_pass["left_idx"]), int(final_pass["right_idx"])
    peak_idx_corr = int(final_pass["peak_idx_corr"])
    peak_height = float(y_corr[peak_idx_corr])
    background_rms = _compute_minima_bracket_rms_noise(i, left_idx, right_idx)

    if min_peak_height_uA is not None and peak_height < float(min_peak_height_uA):
        raise ValueError(f"Peak height {peak_height:.4g} uA below cutoff {min_peak_height_uA:.4g} uA")

    wavelet_energy = np.nan
    if compute_wavelet_energy:
        coeffs = pywt.wavedec(y_corr, "haar", level=3)
        wavelet_energy = float(sum(np.sum(c**2) for c in coeffs))

    skew_val = float(skew(y_corr)) if compute_skew else np.nan
    peak_offset_norm = np.nan
    v_left = float(v[left_idx])
    v_right = float(v[right_idx])
    bracket_width_V = float(v_right - v_left)
    denom = (v_right - v_left) / 2.0
    if denom != 0:
        peak_offset_norm = float((v[peak_idx_corr] - (v_left + v_right) / 2.0) / denom)

    return {
        "file_path": file_path,
        "background_current_rms": background_rms,
        "background_current_median": background_median,
        "voltage": v,
        "raw_current": i,
        "smoothed_current": i_smooth,
        "wavelet_denoised_current": wavelet_denoised_current,
        "corrected_current": y_corr,
        "smoothed_corrected_current": y_corr_smooth,
        "local_baseline": first_pass["local_baseline"],
        "first_pass_corrected_current": first_pass["corrected_current"] if use_double_correction else None,
        "first_pass_smoothed_corrected_current": first_pass["smoothed_corrected_current"] if use_double_correction else None,
        "first_pass_local_baseline": first_pass["local_baseline"] if use_double_correction else None,
        # Use corrected-trace peak position for peak voltage (and drift downstream)
        "peak_voltage": float(v[peak_idx_corr]),
        "peak_current": peak_height,
        "peak_current_raw": float(i[first_pass["peak_idx"]]),
        "bracket_width_V": bracket_width_V,
        "peak_idx": first_pass["peak_idx"],
        "peak_idx_corr": peak_idx_corr,
        "left_min_idx": left_idx,
        "right_min_idx": right_idx,
        "left_local_min_candidates": np.asarray(final_pass["left_local_min_candidates"], dtype=int),
        "right_local_min_candidates": np.asarray(final_pass["right_local_min_candidates"], dtype=int),
        "minima_mode": final_pass["minima_mode"],
        "first_pass_peak_idx": first_pass["peak_idx"] if use_double_correction else None,
        "first_pass_peak_idx_corr": first_pass["peak_idx_corr"] if use_double_correction else None,
        "first_pass_left_min_idx": first_pass["left_idx"] if use_double_correction else None,
        "first_pass_right_min_idx": first_pass["right_idx"] if use_double_correction else None,
        "first_pass_left_local_min_candidates": (
            np.asarray(first_pass["left_local_min_candidates"], dtype=int) if use_double_correction else np.array([], dtype=int)
        ),
        "first_pass_right_local_min_candidates": (
            np.asarray(first_pass["right_local_min_candidates"], dtype=int) if use_double_correction else np.array([], dtype=int)
        ),
        "first_pass_minima_mode": first_pass["minima_mode"] if use_double_correction else None,
        "second_pass_corrected_current": second_pass["corrected_current"] if second_pass is not None else None,
        "second_pass_smoothed_corrected_current": (
            second_pass["smoothed_corrected_current"] if second_pass is not None else None
        ),
        "second_pass_local_baseline": second_pass["local_baseline"] if second_pass is not None else None,
        "second_pass_peak_idx": second_pass["peak_idx"] if second_pass is not None else None,
        "second_pass_peak_idx_corr": second_pass["peak_idx_corr"] if second_pass is not None else None,
        "second_pass_left_min_idx": second_pass["left_idx"] if second_pass is not None else None,
        "second_pass_right_min_idx": second_pass["right_idx"] if second_pass is not None else None,
        "second_pass_left_local_min_candidates": (
            np.asarray(second_pass["left_local_min_candidates"], dtype=int) if second_pass is not None else np.array([], dtype=int)
        ),
        "second_pass_right_local_min_candidates": (
            np.asarray(second_pass["right_local_min_candidates"], dtype=int) if second_pass is not None else np.array([], dtype=int)
        ),
        "second_pass_minima_mode": second_pass["minima_mode"] if second_pass is not None else None,
        "double_correction_requested": bool(use_double_correction),
        "double_correction_applied": bool(second_pass is not None),
        "wavelet_correction_applied": bool(use_wavelet_for_correction and wavelet_denoised_current is not None),
        "double_correction_error": double_correction_error,
        "correction_passes": 2 if second_pass is not None else 1,
        "skew": skew_val,
        "peak_offset_norm": peak_offset_norm,
        "wavelet_energy": wavelet_energy,
        "status": "OK",
    }

def partial_traces_for_failure_arrays(
    v_raw: np.ndarray,
    i_raw: np.ndarray,
    crop_range: Tuple[float, float],
    smooth_window: int,
    smooth_polyorder: int,
    minima_search_window_V: float,
    use_prominent_minima: bool,
    use_double_correction: bool,
    compute_wavelet_denoised_trace: bool,
    use_wavelet_for_correction: bool,
) -> dict:
    initial_mask = (v_raw >= crop_range[0]) & (v_raw <= crop_range[1])
    base = dict(background_current_rms=np.nan,
                background_current_median=_compute_outside_crop_median(i_raw, initial_mask),
                voltage=None, raw_current=None, smoothed_current=None,
                wavelet_denoised_current=None,
                smoothed_corrected_current=None,
                corrected_current=None, local_baseline=None,
                peak_idx=None, peak_idx_corr=None, left_min_idx=None, right_min_idx=None,
                left_local_min_candidates=np.array([], dtype=int),
                right_local_min_candidates=np.array([], dtype=int),
                minima_mode=None,
                first_pass_corrected_current=None,
                first_pass_smoothed_corrected_current=None,
                first_pass_local_baseline=None,
                first_pass_peak_idx=None,
                first_pass_peak_idx_corr=None,
                first_pass_left_min_idx=None,
                first_pass_right_min_idx=None,
                first_pass_left_local_min_candidates=np.array([], dtype=int),
                first_pass_right_local_min_candidates=np.array([], dtype=int),
                first_pass_minima_mode=None,
                second_pass_corrected_current=None,
                second_pass_smoothed_corrected_current=None,
                second_pass_local_baseline=None,
                second_pass_peak_idx=None,
                second_pass_peak_idx_corr=None,
                second_pass_left_min_idx=None,
                second_pass_right_min_idx=None,
                second_pass_left_local_min_candidates=np.array([], dtype=int),
                second_pass_right_local_min_candidates=np.array([], dtype=int),
                second_pass_minima_mode=None,
                double_correction_requested=bool(use_double_correction),
                double_correction_applied=False,
                wavelet_correction_applied=False,
                double_correction_error=None,
                correction_passes=1)
    try:
        mask = (v_raw >= crop_range[0]) & (v_raw <= crop_range[1])
        v, i = v_raw[mask], i_raw[mask]
        base.update(voltage=v, raw_current=i)

        if len(v) < 5:
            return {**base, "partial_error": "Too few points after cropping."}

        i_smooth = apply_smoothing(i, smooth_window, smooth_polyorder) if smooth_window > 0 else i.copy()
        base["smoothed_current"] = i_smooth
        wavelet_denoised_current = (
            _wavelet_denoise_trace(i)
            if (compute_wavelet_denoised_trace or use_wavelet_for_correction)
            else None
        )
        base["wavelet_denoised_current"] = wavelet_denoised_current
        first_pass_input = (
            wavelet_denoised_current
            if use_wavelet_for_correction and wavelet_denoised_current is not None
            else i_smooth
        )

        first_pass = _run_correction_pass(
            v=v,
            y_for_correction=first_pass_input,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            minima_search_window_V=minima_search_window_V,
            use_prominent_minima=use_prominent_minima,
        )
        final_pass = first_pass
        second_pass = None
        double_correction_error = None
        if use_double_correction:
            try:
                second_pass = _run_correction_pass(
                    v=v,
                    y_for_correction=first_pass["corrected_current"],
                    peak_source=first_pass["smoothed_corrected_current"],
                    smooth_window=smooth_window,
                    smooth_polyorder=smooth_polyorder,
                    minima_search_window_V=minima_search_window_V,
                    use_prominent_minima=use_prominent_minima,
                )
                final_pass = second_pass
            except Exception as exc:
                double_correction_error = str(exc)

        return {
            **base,
            "corrected_current": final_pass["corrected_current"],
            "smoothed_corrected_current": final_pass["smoothed_corrected_current"],
            "local_baseline": first_pass["local_baseline"],
            "peak_idx": first_pass["peak_idx"],
            "peak_idx_corr": final_pass["peak_idx_corr"],
            "left_min_idx": int(final_pass["left_idx"]),
            "right_min_idx": int(final_pass["right_idx"]),
            "left_local_min_candidates": np.asarray(final_pass["left_local_min_candidates"], dtype=int),
            "right_local_min_candidates": np.asarray(final_pass["right_local_min_candidates"], dtype=int),
            "minima_mode": final_pass["minima_mode"],
            "first_pass_corrected_current": first_pass["corrected_current"] if use_double_correction else None,
            "first_pass_smoothed_corrected_current": first_pass["smoothed_corrected_current"] if use_double_correction else None,
            "first_pass_local_baseline": first_pass["local_baseline"] if use_double_correction else None,
            "first_pass_peak_idx": first_pass["peak_idx"] if use_double_correction else None,
            "first_pass_peak_idx_corr": first_pass["peak_idx_corr"] if use_double_correction else None,
            "first_pass_left_min_idx": first_pass["left_idx"] if use_double_correction else None,
            "first_pass_right_min_idx": first_pass["right_idx"] if use_double_correction else None,
            "first_pass_left_local_min_candidates": (
                np.asarray(first_pass["left_local_min_candidates"], dtype=int) if use_double_correction else np.array([], dtype=int)
            ),
            "first_pass_right_local_min_candidates": (
                np.asarray(first_pass["right_local_min_candidates"], dtype=int) if use_double_correction else np.array([], dtype=int)
            ),
            "first_pass_minima_mode": first_pass["minima_mode"] if use_double_correction else None,
            "second_pass_corrected_current": second_pass["corrected_current"] if second_pass is not None else None,
            "second_pass_smoothed_corrected_current": (
                second_pass["smoothed_corrected_current"] if second_pass is not None else None
            ),
            "second_pass_local_baseline": second_pass["local_baseline"] if second_pass is not None else None,
            "second_pass_peak_idx": second_pass["peak_idx"] if second_pass is not None else None,
            "second_pass_peak_idx_corr": second_pass["peak_idx_corr"] if second_pass is not None else None,
            "second_pass_left_min_idx": second_pass["left_idx"] if second_pass is not None else None,
            "second_pass_right_min_idx": second_pass["right_idx"] if second_pass is not None else None,
            "second_pass_left_local_min_candidates": (
                np.asarray(second_pass["left_local_min_candidates"], dtype=int) if second_pass is not None else np.array([], dtype=int)
            ),
            "second_pass_right_local_min_candidates": (
                np.asarray(second_pass["right_local_min_candidates"], dtype=int) if second_pass is not None else np.array([], dtype=int)
            ),
            "second_pass_minima_mode": second_pass["minima_mode"] if second_pass is not None else None,
            "double_correction_applied": bool(second_pass is not None),
            "wavelet_correction_applied": bool(use_wavelet_for_correction and wavelet_denoised_current is not None),
            "double_correction_error": double_correction_error,
            "correction_passes": 2 if second_pass is not None else 1,
            "partial_error": None,
        }
    except Exception as e:
        return {**base, "partial_error": str(e)}


def compute_drift_fields(all_results: List[dict]) -> List[dict]:
    """
    Adds four drift fields to each result (in-place), computed per channel
    relative to each channel's first valid (OK) scan:

      peak_voltage_drift           peak_voltage               - reference peak_voltage  (V)
      bracket_width_drift          bracket_width_V            - reference bracket_width_V  (V)
      skew_drift                   skew                       - reference skew
      peak_offset_norm_drift       peak_offset_norm          - reference peak_offset_norm
    """
    ref: Dict[int, dict] = {}

    # Sort globally so we always pick the lowest scan_number as reference
    sorted_results = sorted(all_results, key=lambda r: (r["channel"], r["scan_number"]))

    for r in sorted_results:
        ch = r["channel"]
        if r.get("status") != "OK":
            r["peak_voltage_drift"] = np.nan
            r["bracket_width_drift"] = np.nan
            r["skew_drift"] = np.nan
            r["peak_offset_norm_drift"] = np.nan
            continue

        if ch not in ref:
            ref[ch] = r  # first OK scan for this channel = reference

        r["peak_voltage_drift"] = r["peak_voltage"] - ref[ch]["peak_voltage"]
        r["bracket_width_drift"] = r["bracket_width_V"] - ref[ch]["bracket_width_V"]
        r["skew_drift"]         = r["skew"]         - ref[ch]["skew"]
        r["peak_offset_norm_drift"] = r["peak_offset_norm"] - ref[ch]["peak_offset_norm"]

    return all_results


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


def _validate_swv_work_item(
    measurement: SWVFile,
    crop_range: Tuple[float, float],
    voltage_col: str,
    current_col: Optional[str],
    min_start_voltage: float,
) -> Optional[Tuple[int, int]]:
    """Return the file signature when a source file is eligible for SWV analysis."""
    try:
        file_mtime_ns, file_size = _file_signature(measurement.path)
        v_check, _ = _load_filtered_arrays_cached(
            filepath=measurement.path,
            voltage_col=voltage_col,
            current_col=current_col,
            file_mtime_ns=file_mtime_ns,
            file_size=file_size,
        )
    except Exception:
        return None

    if len(v_check) == 0 or float(v_check[0]) < float(min_start_voltage):
        return None
    in_crop = (v_check >= crop_range[0]) & (v_check <= crop_range[1])
    if in_crop.sum() < 5:
        return None
    return file_mtime_ns, file_size


def _process_swv_work_item(
    measurement: SWVFile,
    method_path: str,
    crop_range: Tuple[float, float],
    voltage_col: str,
    current_col: Optional[str],
    smooth_window: int,
    smooth_polyorder: int,
    minima_search_window_V: float,
    use_prominent_minima: bool,
    use_double_correction: bool,
    min_peak_height_uA: Optional[float],
    min_start_voltage: float,
    compute_skew: bool,
    compute_wavelet_energy: bool,
    compute_wavelet_denoised_trace: bool,
    use_wavelet_for_correction: bool,
    validated_signature: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[dict, dict]]:
    """Validate and analyze one SWV without assigning its display scan number."""
    signature = validated_signature or _validate_swv_work_item(
        measurement=measurement,
        crop_range=crop_range,
        voltage_col=voltage_col,
        current_col=current_col,
        min_start_voltage=min_start_voltage,
    )
    if signature is None:
        return None
    file_mtime_ns, file_size = signature

    method_meta = load_swv_method_metadata(method_path)
    processed = _process_file_cached(
        filepath=measurement.path,
        voltage_col=voltage_col,
        current_col=current_col,
        file_mtime_ns=file_mtime_ns,
        file_size=file_size,
        crop_range=crop_range,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        minima_search_window_V=minima_search_window_V,
        use_prominent_minima=use_prominent_minima,
        use_double_correction=use_double_correction,
        min_peak_height_uA=min_peak_height_uA,
        compute_skew=compute_skew,
        compute_wavelet_energy=compute_wavelet_energy,
        compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
        use_wavelet_for_correction=use_wavelet_for_correction,
    )
    return method_meta, processed


def run_batch(
    folders: List[str],
    crop_range: Tuple[float, float] = (-0.5, -0.1),
    voltage_col: str = "Potential (V)",
    current_col: Optional[str] = None,
    smooth_window: int = 9,
    smooth_polyorder: int = 2,
    minima_search_window_V: float = 0.30,
    use_prominent_minima: bool = False,
    use_double_correction: bool = False,
    min_peak_height_uA: Optional[float] = None,
    min_start_voltage: float = -0.6,
    scan_windows: Optional[Tuple[Tuple[int, int], ...]] = None,
    scan_range: Optional[Tuple[int, int]] = None,
    time_range: Optional[Tuple[datetime, datetime]] = None,
    compute_skew: bool = True,
    compute_wavelet_energy: bool = True,
    compute_wavelet_denoised_trace: bool = False,
    use_wavelet_for_correction: bool = False,
    parallel_workers: int = 1,
    progress_callback=None,
) -> List[dict]:
    files = collect_swv_csvs_from_folders(folders)
    if not files:
        raise ValueError("No SWV CSVs found.")

    by_ch = group_by_channel_and_sort(files)
    all_results: List[dict] = []

    ordered: List[Tuple[int, SWVFile]] = [
        (ch, f)
        for ch, flist in sorted(by_ch.items())
        for f in flist
    ]

    total = len(ordered)
    scan_counters: Dict[int, int] = {}
    worker_count = min(total, max(1, int(parallel_workers or 1)))
    if time_range is not None and (scan_windows or scan_range is not None):
        raise ValueError("Choose either scan-position windows or a filename-time range, not both.")
    filter_during_analysis = bool(
        scan_windows or scan_range is not None or time_range is not None
    )
    assigned_scans: Dict[int, Tuple[int, int]] = {}

    if filter_during_analysis:
        source_file_counters: Dict[int, int] = {}
        selected_time_counters: Dict[int, int] = {}
        for index, (channel, measurement) in enumerate(ordered):
            source_scan_number = source_file_counters.get(channel, 0)
            source_file_counters[channel] = source_scan_number + 1
            if time_range is not None:
                measurement_time = measurement.measurement_time
                if (
                    measurement_time is None
                    or measurement_time < time_range[0]
                    or measurement_time > time_range[1]
                ):
                    continue
                analysis_scan_number = selected_time_counters.get(channel, 0)
                selected_time_counters[channel] = analysis_scan_number + 1
                assigned_scans[index] = (
                    source_scan_number,
                    analysis_scan_number,
                )
                continue
            if not _scan_in_windows(
                source_scan_number,
                scan_windows=scan_windows,
                scan_range=scan_range,
            ):
                continue
            assigned_scans[index] = (
                source_scan_number,
                _remap_scan_number(
                    source_scan_number,
                    scan_windows=scan_windows,
                    scan_range=scan_range,
                ),
            )

    if filter_during_analysis:
        method_paths = {
            index: _infer_method_path_direct(ordered[index][1].path)
            for index in assigned_scans
        }
    else:
        method_file_index = _build_method_file_index(files)
        method_paths = {
            index: _infer_method_path(measurement.path, method_file_index)
            for index, (_, measurement) in enumerate(ordered)
        }

    def append_processed_result(
        index: int,
        channel: int,
        measurement: SWVFile,
        outcome: Optional[Tuple[dict, dict]],
    ) -> None:
        if outcome is None:
            return
        method_meta, processed = outcome
        if filter_during_analysis:
            assigned_scan = assigned_scans.get(index)
            if assigned_scan is None:
                return
            scan_number, analysis_scan_number = assigned_scan
        else:
            scan_counters[channel] = scan_counters.get(channel, 0) + 1
            scan_number = scan_counters[channel]
            analysis_scan_number = scan_number
        common = dict(
            channel=channel,
            channel_label=f"Ch{channel}",
            timestamp=measurement.ts,
            measurement_time=measurement.measurement_time,
            scan_id_from_name=measurement.scan,
            original_scan_number=scan_number,
            scan_number=analysis_scan_number,
            folder_index=measurement.folder_index,
            file_path=measurement.path,
            file_name=os.path.basename(measurement.path),
        )
        common.update(
            method_path=method_meta.get("method_path"),
            method_exists=method_meta.get("method_exists"),
            swv_frequency_hz=method_meta.get("swv_frequency_hz"),
            swv_method_group=method_meta.get("swv_method_group"),
            swv_sweep_start_V=method_meta.get("swv_sweep_start_V"),
            swv_sweep_end_V=method_meta.get("swv_sweep_end_V"),
            swv_step_size_V=method_meta.get("swv_step_size_V"),
            swv_amplitude_V=method_meta.get("swv_amplitude_V"),
        )
        if processed["status"] == "OK":
            r = dict(processed["result"])
            r.update(common)
            all_results.append(r)
        else:
            partial = dict(processed["partial"])
            all_results.append({
                **common,
                "peak_current": np.nan,
                "peak_current_raw": np.nan,
                "peak_voltage": np.nan,
                "bracket_width_V": np.nan,
                "skew": np.nan,
                "peak_offset_norm": np.nan,
                "wavelet_energy": np.nan,
                "status": "FAILED",
                "error": processed["error"],
                **{k: partial.get(k) for k in (
                    "background_current_rms", "background_current_median",
                    "voltage", "raw_current", "smoothed_current",
                    "wavelet_denoised_current",
                    "corrected_current", "smoothed_corrected_current",
                    "local_baseline", "partial_error",
                    "left_min_idx", "right_min_idx", "peak_idx", "peak_idx_corr",
                    "left_local_min_candidates", "right_local_min_candidates",
                    "minima_mode", "first_pass_corrected_current",
                    "first_pass_smoothed_corrected_current", "first_pass_local_baseline",
                    "first_pass_peak_idx", "first_pass_peak_idx_corr",
                    "first_pass_left_min_idx", "first_pass_right_min_idx",
                    "first_pass_left_local_min_candidates", "first_pass_right_local_min_candidates",
                    "first_pass_minima_mode", "second_pass_corrected_current",
                    "second_pass_smoothed_corrected_current", "second_pass_local_baseline",
                    "second_pass_peak_idx", "second_pass_peak_idx_corr",
                    "second_pass_left_min_idx", "second_pass_right_min_idx",
                    "second_pass_left_local_min_candidates", "second_pass_right_local_min_candidates",
                    "second_pass_minima_mode", "double_correction_requested",
                    "wavelet_correction_applied",
                    "double_correction_applied", "double_correction_error",
                    "correction_passes",
                )},
            })

    def process_index(index: int) -> Optional[Tuple[dict, dict]]:
        _, measurement = ordered[index]
        return _process_swv_work_item(
            measurement=measurement,
            method_path=method_paths[index],
            crop_range=crop_range,
            voltage_col=voltage_col,
            current_col=current_col,
            smooth_window=smooth_window,
            smooth_polyorder=smooth_polyorder,
            minima_search_window_V=minima_search_window_V,
            use_prominent_minima=use_prominent_minima,
            use_double_correction=use_double_correction,
            min_peak_height_uA=min_peak_height_uA,
            min_start_voltage=min_start_voltage,
            compute_skew=compute_skew,
            compute_wavelet_energy=compute_wavelet_energy,
            compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
            use_wavelet_for_correction=use_wavelet_for_correction,
        )

    analysis_indexes = (
        sorted(assigned_scans)
        if filter_during_analysis
        else list(range(total))
    )
    analysis_total = len(analysis_indexes)

    if worker_count == 1:
        for completed, index in enumerate(analysis_indexes, start=1):
            channel, measurement = ordered[index]
            append_processed_result(index, channel, measurement, process_index(index))
            if progress_callback:
                progress_callback(
                    completed,
                    analysis_total,
                    f"Analyzing {os.path.basename(measurement.path)}",
                )
    elif analysis_total:
        # Limit in-flight tasks so very large batches do not allocate one Future
        # per file. Completed work is emitted in source order to preserve the
        # existing per-channel scan numbering exactly.
        max_pending = max(worker_count, worker_count * 4)
        pending = {}
        ready: Dict[int, Optional[Tuple[dict, dict]]] = {}
        next_submit_position = 0
        next_emit_position = 0
        completed = 0

        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="swv-analysis",
        ) as executor:
            while (
                next_submit_position < analysis_total
                and len(pending) < max_pending
            ):
                index = analysis_indexes[next_submit_position]
                future = executor.submit(process_index, index)
                pending[future] = index
                next_submit_position += 1

            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index = pending.pop(future)
                    ready[index] = future.result()
                    completed += 1
                    if progress_callback:
                        progress_callback(
                            completed,
                            analysis_total,
                            f"Analyzing {os.path.basename(ordered[index][1].path)}",
                        )

                while (
                    next_emit_position < analysis_total
                    and analysis_indexes[next_emit_position] in ready
                ):
                    index = analysis_indexes[next_emit_position]
                    channel, measurement = ordered[index]
                    append_processed_result(
                        index,
                        channel,
                        measurement,
                        ready.pop(index),
                    )
                    next_emit_position += 1

                while (
                    next_submit_position < analysis_total
                    and len(pending) + len(ready) < max_pending
                ):
                    index = analysis_indexes[next_submit_position]
                    future = executor.submit(process_index, index)
                    pending[future] = index
                    next_submit_position += 1

        while (
            next_emit_position < analysis_total
            and analysis_indexes[next_emit_position] in ready
        ):
            index = analysis_indexes[next_emit_position]
            channel, measurement = ordered[index]
            append_processed_result(
                index,
                channel,
                measurement,
                ready.pop(index),
            )
            next_emit_position += 1

    # Compute drift relative to each channel's first valid scan
    compute_drift_fields(all_results)

    return all_results
