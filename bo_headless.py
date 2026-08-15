"""
Headless SWV analysis entry point for Bayesian optimization integration.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from core.analysis import run_batch


def _clip01(value: float) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _finite(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if f == f:
            out.append(f)
    return out


def _median(values: Iterable[Any], default: float = 0.0) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return float(default)
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def _mean(values: Iterable[Any], default: float = 0.0) -> float:
    vals = _finite(values)
    if not vals:
        return float(default)
    return float(sum(vals) / len(vals))


def _std(values: Iterable[Any]) -> float:
    vals = _finite(values)
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    return float((sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5)


def _cv(values: Iterable[Any]) -> float:
    vals = [abs(v) for v in _finite(values)]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    if mean <= 1e-12:
        return 0.0
    return _std(vals) / mean


def _score_from_cv(values: Iterable[Any]) -> float:
    return _clip01(1.0 / (1.0 + _cv(values)))


def _pairwise_repeat_snr(
    buffer_peak_currents: Iterable[Any],
    target_peak_currents: Iterable[Any],
) -> Tuple[float, float]:
    """Return (target-minus-buffer mean / pairwise sample STD, pairwise STD)."""
    buffer_values = _finite(buffer_peak_currents)
    target_values = _finite(target_peak_currents)
    if not buffer_values or not target_values:
        return 0.0, 0.0
    pairwise_differences = [
        target_value - buffer_value
        for buffer_value in buffer_values
        for target_value in target_values
    ]
    pairwise_std = _std(pairwise_differences)
    if len(pairwise_differences) < 2 or pairwise_std <= 1e-12:
        return 0.0, pairwise_std
    mean_difference = _mean(target_values) - _mean(buffer_values)
    return mean_difference / pairwise_std, pairwise_std


def _build_channel_metrics(results: List[dict]) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in results:
        channel = row.get("channel")
        if channel is None:
            continue
        key = str(int(channel))
        grouped.setdefault(key, []).append(row)

    metrics: Dict[str, dict] = {}
    for channel, rows in grouped.items():
        total = len(rows)
        ok_rows = [r for r in rows if str(r.get("status", "")).upper() == "OK"]
        success_score = len(ok_rows) / total if total else 0.0
        if not ok_rows:
            metrics[channel] = {
                "snr": 0.0,
                "snr_unadjusted": 0.0,
                "peak_prominence": 0.0,
                "repeat_scan_snr": 0.0,
                "peak_shape_score": 0.0,
                "baseline_stability_score": 0.0,
                "replicate_consistency_score": 0.0,
                "success_score": _clip01(success_score),
                "ok_scan_count": 0,
                "total_scan_count": total,
                "std_peak_current_uA": 0.0,
                "peak_currents_uA": [],
                "std_background_rms_uA": 0.0,
                "repeat_relative_std": 0.0,
            }
            continue

        peak_currents = [abs(r.get("peak_current", 0.0)) for r in ok_rows]
        background_rms = [abs(r.get("background_current_rms", 0.0)) for r in ok_rows]
        offset_scores = [max(0.0, 1.0 - abs(float(r.get("peak_offset_norm", 0.0)))) for r in ok_rows]
        width_scores = _score_from_cv(r.get("bracket_width_V", 0.0) for r in ok_rows)
        baseline_scores = _score_from_cv(background_rms)
        replicate_scores = _score_from_cv(peak_currents)
        mean_peak_current = _mean(peak_currents, 0.0)
        mean_background_rms = _mean(background_rms, 0.0)
        std_peak_current = _std(peak_currents)
        std_background_rms = _std(background_rms)
        repeat_relative_std = 0.5 * (
            _cv(peak_currents) + _cv(background_rms)
        )
        peak_prominence = mean_peak_current / max(mean_background_rms, 1e-12)
        repeat_scan_snr = (
            mean_peak_current / max(std_peak_current, 1e-12)
            if len(peak_currents) > 1
            else 0.0
        )
        metrics[channel] = {
            # Keep the old names so archived automation releases can consume
            # summaries produced by this analysis worker.
            "snr": peak_prominence,
            "snr_unadjusted": peak_prominence,
            "peak_prominence": peak_prominence,
            "repeat_scan_snr": repeat_scan_snr,
            "peak_shape_score": _clip01(0.5 * _mean(offset_scores, 0.0) + 0.5 * width_scores),
            "baseline_stability_score": _clip01(baseline_scores),
            "replicate_consistency_score": _clip01(replicate_scores),
            "success_score": _clip01(success_score),
            "ok_scan_count": len(ok_rows),
            "total_scan_count": total,
            "median_peak_current_uA": _median(peak_currents, 0.0),
            "mean_peak_current_uA": mean_peak_current,
            "std_peak_current_uA": std_peak_current,
            "peak_currents_uA": peak_currents,
            "median_background_rms_uA": _median(background_rms, 0.0),
            "mean_background_rms_uA": mean_background_rms,
            "std_background_rms_uA": std_background_rms,
            "repeat_relative_std": repeat_relative_std,
        }
    return metrics


def _normalize_scan_windows(raw: str) -> Optional[Tuple[Tuple[int, int], ...]]:
    text = (raw or "").strip()
    if not text:
        return None
    windows = []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            raise ValueError(f"Invalid scan window token: {token}")
        start_s, end_s = token.split("-", 1)
        start, end = int(start_s.strip()), int(end_s.strip())
        if end <= start:
            raise ValueError(f"Scan window end must be greater than start: {token}")
        windows.append((start, end))
    return tuple(windows) if windows else None


def _analysis_args_from_request(payload: dict) -> dict:
    analysis = dict(payload.get("analysis") or {})
    crop_min = float(analysis.get("crop_min_v", -0.6))
    crop_max = float(analysis.get("crop_max_v", -0.1))
    require_minima = bool(analysis.get("require_local_minima_on_both_sides", False))
    return {
        "folders": [str(Path(p)) for p in payload.get("folders", [])],
        "crop_range": (crop_min, crop_max),
        "smooth_window": int(analysis.get("smooth_window", 15)),
        "smooth_polyorder": int(analysis.get("smooth_polyorder", 2)),
        "minima_search_window_V": float(analysis.get("minima_search_window_v", 0.30)),
        "use_prominent_minima": bool(analysis.get("use_prominent_minima", False)) or require_minima,
        "use_double_correction": bool(analysis.get("use_double_correction", False)),
        "min_peak_height_uA": (
            None if analysis.get("min_peak_height_ua") in (None, "", "none")
            else float(analysis.get("min_peak_height_ua"))
        ),
        "min_start_voltage": float(analysis.get("min_start_voltage_v", crop_min)),
        "scan_windows": _normalize_scan_windows(str(analysis.get("scan_windows", ""))),
        "scan_range": None,
        "compute_skew": bool(analysis.get("compute_skew", True)),
        "compute_wavelet_energy": bool(analysis.get("compute_wavelet_energy", True)),
        "compute_wavelet_denoised_trace": bool(analysis.get("compute_wavelet_denoised_trace", False)),
        "use_wavelet_for_correction": bool(analysis.get("use_wavelet_for_correction", False)),
    }


def run_request(payload: dict) -> dict:
    args = _analysis_args_from_request(payload)
    folders = args.pop("folders")
    if not folders:
        raise ValueError("At least one folder is required")
    results = run_batch(folders=folders, **args)
    _apply_result_constraints(results, dict(payload.get("analysis") or {}))
    channel_metrics = _build_channel_metrics(results)
    output_dir = Path(payload.get("output_dir") or Path(folders[0]) / "bo_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = str(payload.get("output_stem") or f"bo_analysis_{timestamp}")
    results_csv = output_dir / f"{stem}_results.csv"
    pd.DataFrame(results).to_csv(results_csv, index=False)
    results_json = output_dir / f"{stem}_results.json"
    tmp_results_json = results_json.with_name(f".{results_json.name}.tmp")
    with open(tmp_results_json, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(results), fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_results_json, results_json)
    summary = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folders": folders,
        "analysis": dict(payload.get("analysis") or {}),
        "result_count": len(results),
        "channel_metrics": channel_metrics,
        "results_csv": str(results_csv),
        "results_json": str(results_json),
        "analysis_engine": "Electrochemistry-Analysis-Scripts",
    }
    summary_path = output_dir / f"{stem}.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def _apply_result_constraints(results: List[dict], analysis: dict) -> None:
    peak_min = analysis.get("peak_voltage_min_v")
    peak_max = analysis.get("peak_voltage_max_v")
    peak_min = None if peak_min in (None, "", "none") else float(peak_min)
    peak_max = None if peak_max in (None, "", "none") else float(peak_max)
    require_minima = bool(analysis.get("require_local_minima_on_both_sides", False))
    for row in results:
        if str(row.get("status") or "").upper() != "OK":
            continue
        reasons = []
        peak_voltage = row.get("peak_voltage")
        if peak_min is not None and float(peak_voltage) < peak_min:
            reasons.append(f"peak voltage {float(peak_voltage):g} V is below {peak_min:g} V")
        if peak_max is not None and float(peak_voltage) > peak_max:
            reasons.append(f"peak voltage {float(peak_voltage):g} V is above {peak_max:g} V")
        if require_minima:
            left = row.get("left_local_min_candidates")
            right = row.get("right_local_min_candidates")
            if left is None or len(left) == 0 or right is None or len(right) == 0:
                reasons.append("prominent local minima were not found on both sides")
        if reasons:
            row["status"] = "FAILED"
            row["error"] = "; ".join(reasons)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, float) and value != value:
        return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless BO analysis runner for swv_app.")
    parser.add_argument("--request", required=True, help="Path to request JSON.")
    args = parser.parse_args()
    with open(args.request, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    summary = run_request(payload)
    print(summary["summary_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
