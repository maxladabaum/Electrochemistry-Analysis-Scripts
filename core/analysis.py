import os
import re
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

SWV_LOOP_RE = re.compile(
    r"meas_loop_swv\s+\S+\s+\S+\s+\S+\s+\S+\s+"
    r"(?P<start>[-\d.]+m)\s+"
    r"(?P<end>[-\d.]+m)\s+"
    r"(?P<step>[-\d.]+m)\s+"
    r"(?P<amplitude>[-\d.]+m)\s+"
    r"(?P<frequency>[-\d.]+)",
    re.IGNORECASE,
)


def _file_signature(filepath: str) -> Tuple[int, int]:
    stat = os.stat(filepath)
    return int(stat.st_mtime_ns), int(stat.st_size)


def _infer_method_path(csv_path: str) -> str:
    folder = os.path.dirname(csv_path)
    stem, _ = os.path.splitext(os.path.basename(csv_path))
    return os.path.join(folder, "methods_used", f"{stem}.ms")


def _format_frequency_label(frequency_hz: Optional[float]) -> str:
    if frequency_hz is None:
        return "Unknown method"
    if float(frequency_hz).is_integer():
        return f"{int(frequency_hz)} Hz"
    return f"{float(frequency_hz):g} Hz"


@lru_cache(maxsize=512)
def load_swv_method_metadata(method_path: str) -> dict:
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

    loop_match = SWV_LOOP_RE.search(text)
    if not loop_match:
        return meta

    frequency_hz = float(loop_match.group("frequency"))
    meta["swv_frequency_hz"] = frequency_hz
    meta["swv_method_group"] = _format_frequency_label(frequency_hz)
    return meta


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
        return {"status": "FAILED", "result": None, "partial": partial, "error": str(exc)}


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
    base = dict(voltage=None, raw_current=None, smoothed_current=None,
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


def run_batch(
    folders: List[str],
    crop_range: Tuple[float, float] = (-0.6, -0.2),
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
    compute_skew: bool = True,
    compute_wavelet_energy: bool = True,
    compute_wavelet_denoised_trace: bool = False,
    use_wavelet_for_correction: bool = False,
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

    for idx, (ch, f) in enumerate(ordered):
        if progress_callback:
            progress_callback(idx + 1, total, os.path.basename(f.path))

        try:
            file_mtime_ns, file_size = _file_signature(f.path)
            v_check, i_check = _load_filtered_arrays_cached(
                filepath=f.path,
                voltage_col=voltage_col,
                current_col=current_col,
                file_mtime_ns=file_mtime_ns,
                file_size=file_size,
            )
        except Exception:
            continue

        if len(v_check) == 0 or float(v_check[0]) < float(min_start_voltage):
            continue

        # Skip files that have no data points within the crop range (e.g. LSV sweeps
        # that cover a completely different voltage window than the SWV crop range).
        in_crop = (v_check >= crop_range[0]) & (v_check <= crop_range[1])
        if in_crop.sum() < 5:
            continue

        scan_counters[ch] = scan_counters.get(ch, 0) + 1
        scan_number = scan_counters[ch]

        # If scan filtering is active, skip analysis+storage for out-of-range scans
        # only after the counter has been incremented so numbering stays consistent
        # with the full dataset.
        if not _scan_in_windows(scan_number, scan_windows=scan_windows, scan_range=scan_range):
            continue

        analysis_scan_number = _remap_scan_number(
            scan_number,
            scan_windows=scan_windows,
            scan_range=scan_range,
        )

        common = dict(
            channel=ch,
            channel_label=f"Ch{ch}",
            timestamp=f.ts,
            scan_id_from_name=f.scan,
            original_scan_number=scan_number,
            scan_number=analysis_scan_number,
            folder_index=f.folder_index,
            file_path=f.path,
            file_name=os.path.basename(f.path),
        )
        method_meta = load_swv_method_metadata(_infer_method_path(f.path))
        common.update(
            method_path=method_meta.get("method_path"),
            method_exists=method_meta.get("method_exists"),
            swv_frequency_hz=method_meta.get("swv_frequency_hz"),
            swv_method_group=method_meta.get("swv_method_group"),
        )

        processed = _process_file_cached(
            filepath=f.path,
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

    # Compute drift relative to each channel's first valid scan
    compute_drift_fields(all_results)

    return all_results
