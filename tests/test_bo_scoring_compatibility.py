import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer
import core.analysis as swv_analysis
from bo_headless import (
    _apply_result_constraints,
    _build_channel_metrics,
    _pairwise_repeat_snr,
    run_request,
)
from bo_session_viewer import (
    _add_moving_average_traces,
    _best_q_parameters_by_channel_frame,
    _best_run_q_observation,
    _cache_rescore_profile,
    _classic_q_description,
    _classic_q_equation,
    _clear_rescore_profiles,
    _load_rescore_profiles,
    _metric_impacts_q,
    _next_rescore_label,
    _permanent_rescore_profiles_path,
    _persist_rescore_profile,
    _observation_table,
    _paired_q_equation,
    _paired_q_description,
    _persist_rescored_bo_session,
    _rescore_observations,
    _rescore_default_analysis,
    _rescore_profile_id,
    _reanalyze_observations,
    _resolved_rescore_label,
    _session_with_rescore_profile,
    _simulation_resume_rescore_error,
    _simulation_rescore_token,
    _store_rescore_profiles,
)


def test_headless_worker_enforces_automation_minima_voltage_ranges():
    rows = [{
        "status": "OK",
        "peak_voltage": -0.25,
        "voltage": [-0.4, -0.3, -0.2],
        "left_min_idx": 0,
        "right_min_idx": 2,
        "left_local_min_candidates": [0],
        "right_local_min_candidates": [2],
    }]

    _apply_result_constraints(rows, {
        "peak_voltage_min_v": -0.3,
        "peak_voltage_max_v": -0.2,
        "left_min_voltage_min_v": -0.35,
        "right_min_voltage_max_v": -0.15,
    })

    assert rows[0]["status"] == "FAILED"
    assert "left minimum voltage -0.4 V is below -0.35 V" in rows[0]["error"]


def test_rescore_cache_identity_includes_reanalysis_settings():
    scoring = {"mode": "classic"}

    saved_id = _rescore_profile_id(scoring)
    first_id = _rescore_profile_id(
        scoring,
        reanalyze_swv=True,
        analysis={"use_double_correction": False},
    )
    second_id = _rescore_profile_id(
        scoring,
        reanalyze_swv=True,
        analysis={"use_double_correction": True},
    )

    assert len({saved_id, first_id, second_id}) == 3


def test_rescore_cache_identity_includes_channel_scope():
    scoring = {"mode": "classic"}

    channel_one = _rescore_profile_id(
        scoring,
        rescore_scope={"group_ids": [1], "channels": ["1"]},
    )
    channel_two = _rescore_profile_id(
        scoring,
        rescore_scope={"group_ids": [1], "channels": ["2"]},
    )
    group_two = _rescore_profile_id(
        scoring,
        rescore_scope={"group_ids": [2], "channels": ["1"]},
    )

    assert len({channel_one, channel_two, group_two}) == 3


def test_simulation_rescore_token_changes_with_active_profile():
    assert _simulation_rescore_token("original", None) == "original"
    assert _simulation_rescore_token(
        "profile-a", {"id": "profile-a"}
    ) == "rescore:profile-a"
    assert _simulation_rescore_token(
        "profile-b", {"id": "profile-b"}
    ) == "rescore:profile-b"


def test_simulation_resume_rejects_a_different_rescore_profile():
    saved = {
        "rescore_profile_id": "profile-a",
        "rescore_profile_label": "Rescore A",
    }

    assert _simulation_resume_rescore_error(
        saved,
        "Interpolate loaded real observations",
        {"id": "profile-a", "label": "Rescore A"},
    ) is None
    mismatch = _simulation_resume_rescore_error(
        saved,
        "Interpolate loaded real observations",
        {"id": "profile-b", "label": "Rescore B"},
    )
    assert "Rescore A" in mismatch
    assert "Rescore B" in mismatch


def test_rescore_reanalysis_exposes_every_automation_analysis_setting():
    assert set(_rescore_default_analysis({})) == {
        "crop_min_v",
        "crop_max_v",
        "smooth_window",
        "smooth_polyorder",
        "minima_search_window_v",
        "min_peak_height_ua",
        "peak_voltage_min_v",
        "peak_voltage_max_v",
        "left_min_voltage_min_v",
        "left_min_voltage_max_v",
        "right_min_voltage_min_v",
        "right_min_voltage_max_v",
        "min_start_voltage_v",
        "scan_windows",
        "use_prominent_minima",
        "require_local_minima_on_both_sides",
        "use_double_correction",
        "use_triple_correction",
        "compute_skew",
        "compute_wavelet_energy",
        "compute_wavelet_denoised_trace",
        "use_wavelet_for_correction",
    }


def test_active_reanalysis_triple_correction_reaches_displayed_swv_trace(
    monkeypatch,
    tmp_path,
):
    trace_path = tmp_path / "measurement_ch1.csv"
    trace_path.touch()
    captured = {}

    def fake_corrected_arrays(*args):
        captured["use_double_correction"] = args[9]
        captured["use_triple_correction"] = args[10]
        return np.array([0.0]), np.array([1.0]), 0, 0, 0

    monkeypatch.setattr(
        viewer,
        "_cached_corrected_swv_arrays",
        fake_corrected_arrays,
    )
    analysis = viewer._bo_analysis_settings({
        "analysis": {
            "use_double_correction": True,
            "use_triple_correction": True,
        }
    })

    viewer._swv_trace_arrays(trace_path, True, analysis)

    assert captured == {
        "use_double_correction": True,
        "use_triple_correction": True,
    }


def test_reanalysis_uses_automation_worker_and_profile_applies_new_metrics(
    monkeypatch,
    tmp_path,
):
    buffer_path = tmp_path / "buffer_ch1.csv"
    target_path = tmp_path / "target_ch1.csv"
    buffer_path.touch()
    target_path.touch()
    observation = {
        "method_id": "paired_1",
        "group_id": 1,
        "iteration": 1,
        "objective": "paired_response",
        "channels": [1],
        "Q_run": 1.0,
        "quality": {"Q_run": 1.0},
        "buffer_channel_metrics": {"1": {"mean_peak_current_uA": 1.0}},
        "target_channel_metrics": {"1": {"mean_peak_current_uA": 2.0}},
    }
    session = {
        "root": tmp_path,
        "config": {"scoring": {}},
        "state": {"observations": [observation]},
        "observations": [observation],
        "history": pd.DataFrame([{"iteration": 1, "group_id": 1, "Q_run": 1.0}]),
    }
    monkeypatch.setattr(viewer, "_trace_paths", lambda _session, _observation: [
        {"phase": "buffer", "channel": "1", "path": buffer_path},
        {"phase": "target", "channel": "1", "path": target_path},
    ])
    requests = []
    detail_progress = []

    def fake_worker(request):
        requests.append(request)
        phase = "buffer" if request["output_stem"].endswith("_buffer") else "target"
        request["progress_callback"](1, 1, f"{phase}_ch1.csv")
        peak = 3.0 if phase == "buffer" else 8.0
        summary_path = tmp_path / f"{phase}_summary.json"
        results_path = tmp_path / f"{phase}_results.json"
        return {
            "summary_path": str(summary_path),
            "results_json": str(results_path),
            "analysis_engine": "Electrochemistry-Analysis-Scripts",
            "channel_metrics": {"1": {
                "mean_peak_current_uA": peak,
                "peak_currents_uA": [peak, peak],
                "std_peak_current_uA": 0.0,
                "mean_background_rms_uA": 0.1,
                "success_score": 1.0,
                "ok_scan_count": 2,
                "total_scan_count": 2,
            }},
        }

    monkeypatch.setattr(viewer, "run_bo_analysis_request", fake_worker)
    analysis = {
        "use_double_correction": True,
        "use_triple_correction": True,
        "crop_min_v": -0.5,
        "crop_max_v": 0.0,
    }
    rebuilt = _reanalyze_observations(
        [observation],
        session,
        analysis,
        tmp_path / "reanalysis",
        detail_progress_callback=lambda fraction, text: detail_progress.append(
            (fraction, text)
        ),
    )

    assert len(requests) == 2
    assert all(request["analysis"]["use_double_correction"] for request in requests)
    assert all(request["analysis"]["use_triple_correction"] for request in requests)
    assert any("Buffer trace 1 of 1" in text for _, text in detail_progress)
    assert any("Target trace 1 of 1" in text for _, text in detail_progress)
    assert detail_progress[-1][0] == pytest.approx(1.0)
    assert rebuilt[0]["buffer_channel_metrics"]["1"]["mean_peak_current_uA"] == 3.0
    assert rebuilt[0]["target_channel_metrics"]["1"]["mean_peak_current_uA"] == 8.0

    rebuilt[0]["quality"] = {"Q_run": 7.0, "Q_channels": {"1": 7.0}}
    rebuilt[0]["Q_run"] = 7.0
    profile = _cache_rescore_profile(
        session,
        {"mode": "classic"},
        rebuilt,
        "Reanalyzed",
        reanalyze_swv=True,
        analysis=analysis,
    )
    active = _session_with_rescore_profile(session, profile)

    assert active["observations"][0]["buffer_channel_metrics"]["1"]["mean_peak_current_uA"] == 3.0
    assert active["observations"][0]["target_channel_metrics"]["1"]["mean_peak_current_uA"] == 8.0
    assert active["config"]["analysis"] == analysis


def test_reanalysis_filters_selected_channels_and_preserves_other_metrics(
    monkeypatch,
    tmp_path,
):
    channel_one_path = tmp_path / "measurement_ch1.csv"
    channel_two_path = tmp_path / "measurement_ch2.csv"
    channel_one_path.touch()
    channel_two_path.touch()
    observation = {
        "method_id": "method_1",
        "group_id": 1,
        "iteration": 1,
        "objective": "classic",
        "channels": [1, 2],
        "channel_metrics": {
            "1": {"mean_peak_current_uA": 1.0},
            "2": {"mean_peak_current_uA": 2.0},
        },
    }
    session = {
        "root": tmp_path,
        "config": {"channel_groups": [{"id": 1, "channels": [1, 2]}]},
        "observations": [observation],
    }
    monkeypatch.setattr(viewer, "_trace_paths", lambda *_args: [
        {"phase": "measurement", "channel": "1", "path": channel_one_path},
        {"phase": "measurement", "channel": "2", "path": channel_two_path},
    ])
    requests = []

    def fake_worker(request):
        requests.append(request)
        return {
            "summary_path": str(tmp_path / "summary.json"),
            "results_json": str(tmp_path / "results.json"),
            "results_csv": str(tmp_path / "results.csv"),
            "analysis_engine": "test",
            "channel_metrics": {"1": {"mean_peak_current_uA": 11.0}},
        }

    monkeypatch.setattr(viewer, "run_bo_analysis_request", fake_worker)
    rebuilt = _reanalyze_observations(
        [observation],
        session,
        {},
        tmp_path / "reanalysis",
        rescore_scope={"group_ids": [1], "channels": ["1"]},
    )

    assert requests[0]["folders"] == [str(channel_one_path.resolve())]
    assert rebuilt[0]["channel_metrics"]["1"]["mean_peak_current_uA"] == 11.0
    assert rebuilt[0]["channel_metrics"]["2"]["mean_peak_current_uA"] == 2.0


def test_triple_correction_runs_three_sequential_minima_correction_passes(monkeypatch):
    calls = []

    def fake_correction_pass(*, v, y_for_correction, peak_source=None, **_kwargs):
        pass_number = len(calls) + 1
        calls.append({
            "input": np.asarray(y_for_correction, dtype=float).copy(),
            "peak_source": (
                None if peak_source is None
                else np.asarray(peak_source, dtype=float).copy()
            ),
        })
        corrected = np.asarray(y_for_correction, dtype=float) + pass_number
        return {
            "peak_idx": 3,
            "peak_idx_corr": 3,
            "corrected_current": corrected,
            "smoothed_corrected_current": corrected.copy(),
            "local_baseline": np.zeros_like(corrected),
            "left_idx": 1,
            "right_idx": 5,
            "left_local_min_candidates": np.asarray([1]),
            "right_local_min_candidates": np.asarray([5]),
            "minima_mode": "test",
        }

    monkeypatch.setattr(swv_analysis, "_run_correction_pass", fake_correction_pass)
    voltage = np.linspace(-0.6, 0.0, 7)
    current = np.asarray([0.0, 0.1, 0.4, 1.0, 0.4, 0.1, 0.0])

    result = swv_analysis.analyze_swv_arrays(
        voltage,
        current,
        crop_range=(-0.6, 0.0),
        smooth_window=0,
        use_triple_correction=True,
        compute_skew=False,
        compute_wavelet_energy=False,
    )

    assert len(calls) == 3
    assert np.array_equal(calls[1]["input"], result["first_pass_corrected_current"])
    assert np.array_equal(calls[2]["input"], result["second_pass_corrected_current"])
    assert result["triple_correction_requested"] is True
    assert result["triple_correction_applied"] is True
    assert result["correction_passes"] == 3
    assert np.array_equal(result["corrected_current"], result["third_pass_corrected_current"])


def test_rescore_default_labels_advance_and_changed_settings_do_not_reuse_name():
    profiles = {
        "profile-a": {"id": "profile-a", "label": "Rescore 1"},
        "profile-b": {"id": "profile-b", "label": "Rescore 2"},
    }

    assert _next_rescore_label(profiles) == "Rescore 3"
    assert _resolved_rescore_label(
        "Rescore 2",
        "new-settings",
        profiles,
        profiles["profile-b"],
    ) == "Rescore 3"
    assert _resolved_rescore_label(
        "Renamed profile",
        "profile-b",
        profiles,
        profiles["profile-b"],
    ) == "Renamed profile"


def test_loading_cache_repairs_duplicate_automatic_rescore_labels(tmp_path):
    session = {"root": tmp_path}
    _store_rescore_profiles(session, {
        "profile-a": {"id": "profile-a", "label": "Rescore 1"},
        "profile-b": {"id": "profile-b", "label": "Rescore 1"},
        "profile-c": {"id": "profile-c", "label": "My comparison"},
    })

    profiles = _load_rescore_profiles(session)

    assert profiles["profile-a"]["label"] == "Rescore 1"
    assert profiles["profile-b"]["label"] == "Rescore 2"
    assert profiles["profile-c"]["label"] == "My comparison"
    _clear_rescore_profiles(session)


def test_best_run_q_prefers_paired_response_observations():
    observations = [
        {"objective": "classic", "Q_run": 100.0, "iteration": 1},
        {"objective": "paired_response", "Q_run": 4.0, "iteration": 1},
        {"objective": "paired_response", "Q_run": 7.0, "iteration": 2},
    ]

    best = _best_run_q_observation(observations)

    assert best["objective"] == "paired_response"
    assert best["Q_run"] == pytest.approx(7.0)


def test_observation_table_keeps_top_level_paired_q_run():
    session = {
        "history": pd.DataFrame(),
        "observations": [{
            "iteration": 1,
            "group_id": 1,
            "objective": "paired_response",
            "Q_run": 8.5,
            "quality": {"Q_run": 91.0},
        }],
    }

    history = _observation_table(session)

    assert history.loc[0, "Q_run"] == pytest.approx(8.5)


def test_history_trend_moving_average_is_applied_per_trace():
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[1, 2, 3],
        y=[1.0, 3.0, 5.0],
        mode="lines+markers",
        name="Channel 1",
    ))
    figure.add_trace(go.Scatter(
        x=[1, 2, 3],
        y=[10.0, 20.0, 30.0],
        mode="lines+markers",
        name="Channel 2",
    ))

    _add_moving_average_traces(figure, 2)

    assert len(figure.data) == 4
    assert list(figure.data[2].y) == pytest.approx([1.0, 2.0, 4.0])
    assert list(figure.data[3].y) == pytest.approx([10.0, 15.0, 25.0])
    assert "2-point moving average" in figure.data[2].name


def test_q_descriptions_follow_active_rescore_weights_and_snr_definition():
    config = {
        "scoring": {
            "mode": "classic",
            "channel_weights": {
                "peak_prominence": 0.0,
                "repeat_scan_snr": 0.0,
                "peak_height": 2.0,
                "peak_shape": 0.0,
                "baseline": 0.0,
                "replicate_consistency": 0.0,
                "success": 0.0,
            },
            "paired_response_weights": {
                "buffer_classic_Q": 0.0,
                "target_classic_Q": 0.0,
                "peak_prominence": 0.0,
                "repeat_scan_snr": 1.5,
                "repeat_scan_snr_definition": "pairwise",
            },
            "run_weights": {
                "lambda_variability": 0.0,
                "lambda_failed": 0.0,
                "lambda_low": 0.0,
            },
        }
    }

    classic_text = _classic_q_description(config)
    paired_text = _paired_q_description(config)

    assert "peak height (2)" in classic_text
    assert "peak shape" not in classic_text
    assert "pairwise repeat-scan SNR" in paired_text
    assert "positive direction is target > buffer" in paired_text


def test_rescore_cache_can_be_cleared_for_a_new_page_session(tmp_path):
    session = {"root": tmp_path}
    profile = _cache_rescore_profile(
        session,
        {"mode": "classic"},
        [{"iteration": 1, "group_id": 1, "Q_run": 2.0, "quality": {}}],
        "Temporary rescore",
    )
    assert profile["id"] in _load_rescore_profiles(session)

    _clear_rescore_profiles(session)

    assert _load_rescore_profiles(session) == {}
    assert not (tmp_path / "bo_rescore_cache.json").exists()


def test_permanent_rescore_survives_page_cache_clear_and_preserves_original(tmp_path):
    original = {
        "method_id": "method_1",
        "iteration": 1,
        "group_id": 1,
        "Q_run": 1.0,
        "quality": {"Q_run": 1.0},
    }
    rescored = {
        **original,
        "Q_run": 7.0,
        "quality": {"Q_run": 7.0, "Q_channels": {"1": 7.0}},
    }
    session = {
        "root": tmp_path,
        "config": {"scoring": {}},
        "observations": [original],
        "history": pd.DataFrame([{
            "iteration": 1,
            "group_id": 1,
            "Q_run": 1.0,
        }]),
    }
    profile = _cache_rescore_profile(
        session,
        {"mode": "classic"},
        [rescored],
        "Permanent comparison",
    )

    saved, path = _persist_rescore_profile(session, profile["id"])
    _clear_rescore_profiles(session)
    reloaded = _load_rescore_profiles(session)
    rescored_session = _session_with_rescore_profile(
        session,
        reloaded[profile["id"]],
    )

    assert path == _permanent_rescore_profiles_path(session)
    assert path.is_file()
    assert saved["permanent"] is True
    assert reloaded[profile["id"]]["label"] == "Permanent comparison"
    assert reloaded[profile["id"]]["permanent"] is True
    assert rescored_session["observations"][0]["Q_run"] == pytest.approx(7.0)
    assert session["observations"][0]["Q_run"] == pytest.approx(1.0)


def test_permanent_reanalysis_restores_swv_correction_settings_after_reload(
    monkeypatch,
    tmp_path,
):
    original = {
        "method_id": "method_1",
        "iteration": 1,
        "group_id": 1,
        "Q_run": 1.0,
        "quality": {"Q_run": 1.0},
    }
    session = {
        "root": tmp_path,
        "config": {
            "scoring": {},
            "analysis": {
                "use_double_correction": False,
                "use_triple_correction": False,
            },
        },
        "observations": [original],
        "history": pd.DataFrame([{
            "method_id": "method_1",
            "iteration": 1,
            "group_id": 1,
            "Q_run": 1.0,
        }]),
    }
    saved_analysis = {
        "crop_min_v": -0.52,
        "crop_max_v": -0.18,
        "smooth_window": 11,
        "smooth_polyorder": 2,
        "minima_search_window_v": 0.24,
        "use_prominent_minima": True,
        "use_double_correction": True,
        "use_triple_correction": True,
        "min_peak_height_ua": 0.002,
        "compute_wavelet_denoised_trace": False,
        "use_wavelet_for_correction": False,
    }
    profile = _cache_rescore_profile(
        session,
        {"mode": "classic"},
        [original],
        "Saved reanalysis",
        reanalyze_swv=True,
        analysis=saved_analysis,
    )
    _persist_rescore_profile(session, profile["id"])

    # Simulate a new app page: discard memory profiles and reload from disk.
    _clear_rescore_profiles(session)
    reloaded_profile = _load_rescore_profiles(session)[profile["id"]]
    active_session = _session_with_rescore_profile(session, reloaded_profile)
    displayed_analysis = viewer._bo_analysis_settings(active_session["config"])

    trace_path = tmp_path / "archived_ch1.csv"
    trace_path.touch()
    captured = {}

    def fake_corrected_arrays(*args):
        captured.update({
            "crop_min_v": args[3],
            "crop_max_v": args[4],
            "smooth_window": args[5],
            "use_prominent_minima": args[8],
            "use_double_correction": args[9],
            "use_triple_correction": args[10],
        })
        return np.array([0.0]), np.array([1.0]), 0, 0, 0

    monkeypatch.setattr(
        viewer,
        "_cached_corrected_swv_arrays",
        fake_corrected_arrays,
    )
    viewer._swv_trace_arrays(trace_path, True, displayed_analysis)

    assert reloaded_profile["reanalyze_swv"] is True
    assert reloaded_profile["analysis"] == saved_analysis
    assert active_session["config"]["analysis"] == saved_analysis
    assert captured == {
        "crop_min_v": -0.52,
        "crop_max_v": -0.18,
        "smooth_window": 11,
        "use_prominent_minima": True,
        "use_double_correction": True,
        "use_triple_correction": True,
    }


def _repeat_rows():
    base = {
        "channel": 1,
        "status": "OK",
        "peak_offset_norm": 0.0,
        "bracket_width_V": 0.2,
    }
    return [
        {**base, "peak_current": 1.0, "background_current_rms": 0.1},
        {**base, "peak_current": 1.4, "background_current_rms": 0.3},
    ]


def test_headless_metrics_match_repeat_aware_automation_scoring_inputs():
    metrics = _build_channel_metrics(_repeat_rows())["1"]

    assert metrics["mean_peak_current_uA"] == pytest.approx(1.2)
    assert metrics["mean_background_rms_uA"] == pytest.approx(0.2)
    assert metrics["peak_prominence"] == pytest.approx(6.0)
    assert metrics["snr"] == pytest.approx(metrics["peak_prominence"])
    assert metrics["std_peak_current_uA"] == pytest.approx(0.4 / math.sqrt(2))
    assert metrics["repeat_scan_snr"] == pytest.approx(
        1.2 / metrics["std_peak_current_uA"]
    )
    assert metrics["repeat_relative_std"] > 0.0


def test_headless_reanalysis_serializes_timestamps_and_saves_nonzero_peak_metrics(
    tmp_path,
):
    voltage = np.linspace(-0.7, -0.1, 121)
    current = (
        0.05 * (voltage + 0.4) ** 2
        + np.exp(-((voltage + 0.32) / 0.05) ** 2)
    )
    paths = []
    for scan in range(1, 4):
        path = tmp_path / (
            f"swv_ch1_abcd_meas_20260816_120{scan}_{scan}_ch1.csv"
        )
        pd.DataFrame({
            "Potential (V)": voltage,
            "Current Diff (uA)": current,
        }).to_csv(path, index=False)
        paths.append(str(path))

    progress_events = []
    summary = run_request({
        "folders": paths,
        "output_dir": str(tmp_path / "output"),
        "output_stem": "timestamp_regression",
        "analysis": {
            "crop_min_v": -0.61,
            "crop_max_v": -0.2,
            "smooth_window": 15,
            "smooth_polyorder": 2,
            "minima_search_window_v": 0.3,
            "min_peak_height_ua": None,
            "min_start_voltage_v": -0.7,
            "compute_wavelet_energy": False,
        },
        "progress_callback": lambda completed, total, text: progress_events.append(
            (completed, total, text)
        ),
    })

    metrics = summary["channel_metrics"]["1"]
    saved_rows = json.loads(Path(summary["results_json"]).read_text(encoding="utf-8"))
    assert metrics["mean_peak_current_uA"] > 0.0
    assert metrics["peak_currents_uA"] == pytest.approx([
        metrics["mean_peak_current_uA"]
    ] * 3)
    assert saved_rows[0]["peak_current"] > 0.0
    assert saved_rows[0]["measurement_time"].startswith("2026-08-16T12:01")
    assert [event[0] for event in progress_events] == [1, 2, 3]
    assert all(event[1] == 3 for event in progress_events)


def test_reanalysis_worker_error_preserves_recorded_metrics_instead_of_zeroing(
    monkeypatch,
    tmp_path,
):
    raw_path = tmp_path / "swv_ch1_abcd_meas_20260816_1200_1_ch1.csv"
    raw_path.touch()
    original = {
        "method_id": "method_1",
        "group_id": 1,
        "iteration": 1,
        "objective": "classic",
        "channels": [1],
        "Q_run": 4.2,
        "quality": {"Q_run": 4.2},
        "channel_metrics": {"1": {
            "mean_peak_current_uA": 1.5,
            "peak_currents_uA": [1.4, 1.5, 1.6],
            "success_score": 1.0,
        }},
    }
    session = {"root": tmp_path, "config": {}, "observations": [original]}
    monkeypatch.setattr(viewer, "_trace_paths", lambda *_args: [{
        "phase": "measurement",
        "channel": "1",
        "path": raw_path,
    }])
    monkeypatch.setattr(
        viewer,
        "run_bo_analysis_request",
        lambda _request: (_ for _ in ()).throw(TypeError("storage failure")),
    )

    rebuilt = _reanalyze_observations(
        [original], session, {}, tmp_path / "reanalysis"
    )
    rescored = _rescore_observations(rebuilt, {}, {"channel_weights": {}})

    assert rebuilt[0]["_viewer_rescore_skipped"] is True
    assert rebuilt[0]["channel_metrics"] == original["channel_metrics"]
    assert rescored[0]["Q_run"] == pytest.approx(4.2)
    assert rescored[0]["channel_metrics"]["1"]["mean_peak_current_uA"] == 1.5


def test_one_scan_has_no_repeat_scan_snr():
    metrics = _build_channel_metrics(_repeat_rows()[:1])["1"]

    assert metrics["peak_prominence"] == pytest.approx(10.0)
    assert metrics["repeat_scan_snr"] == 0.0


def test_pairwise_repeat_snr_uses_target_minus_buffer_sample_std():
    snr, pairwise_std = _pairwise_repeat_snr(
        [1.0, 2.0, 3.0],
        [4.0, 6.0, 8.0],
    )
    differences = [
        target_value - buffer_value
        for buffer_value in (1.0, 2.0, 3.0)
        for target_value in (4.0, 6.0, 8.0)
    ]
    expected_std = (
        sum((value - sum(differences) / len(differences)) ** 2 for value in differences)
        / (len(differences) - 1)
    ) ** 0.5

    assert pairwise_std == pytest.approx(expected_std)
    assert snr == pytest.approx((6.0 - 2.0) / expected_std)


def test_q_equations_use_new_metrics_and_keep_legacy_weight_fallbacks():
    config = {
        "scoring": {
            "channel_weights": {"snr": 0.7, "repeat_scan_snr": 0.2},
            "run_weights": {"lambda_repeat_std": 1.5},
            "paired_response_weights": {
                "delta_peak": 2.5,
                "repeat_scan_snr": 0.3,
                "lambda_repeat_std": 3.0,
            },
        }
    }

    classic = _classic_q_equation(config)
    paired = _paired_q_equation(config)
    assert "0.7·Peak prominence" in classic
    assert "0.2·Repeat-scan SNR" in classic
    assert "1.5·mean_repeat_relative_STD" in classic
    assert "2.5·Peak prominence" in paired
    assert "0.3·Repeat-scan SNR" in paired
    assert "Δpeak / (buffer peak STD + target peak STD)" in paired
    assert "3·mean_repeat_relative_STD" in paired
    assert "directional_penalty" in classic
    assert "directional_penalty" in paired


def test_q_relevance_uses_separate_prominence_repeat_and_penalty_weights():
    config = {
        "scoring": {
            "channel_weights": {
                "peak_prominence": 0.0,
                "repeat_scan_snr": 2.0,
            },
            "run_weights": {"lambda_repeat_std": 3.0},
            "paired_response_weights": {
                "peak_prominence": 4.0,
                "repeat_scan_snr": 0.0,
                "lambda_repeat_std": 0.0,
            },
        }
    }

    assert not _metric_impacts_q("Peak prominence", config, False)
    assert not _metric_impacts_q("Normalized peak prominence", config, False)
    assert _metric_impacts_q("Repeat-scan SNR", config, False)
    assert _metric_impacts_q("Repeat relative STD", config, False)
    assert _metric_impacts_q("Peak prominence", config, True)
    assert not _metric_impacts_q("Repeat-scan SNR", config, True)
    assert not _metric_impacts_q("Repeat relative STD", config, True)


def test_viewer_rescores_classic_observations_from_editable_weights():
    observation = {
        "iteration": 1,
        "group_id": 1,
        "objective": "classic",
        "channel_metrics": {
            "1": {
                "peak_prominence": 5.0,
                "success_score": 1.0,
                "ok_scan_count": 3,
            }
        },
        "Q_run": 1.0,
    }
    scoring = {
        "mode": "classic",
        "channel_weights": {
            "peak_prominence": 2.0,
            "peak_shape": 0.0,
            "baseline": 0.0,
            "replicate_consistency": 0.0,
            "success": 0.0,
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    progress_updates = []
    rescored = _rescore_observations(
        [observation],
        {"acquisition": {"optimization_direction": "maximize"}},
        scoring,
        progress_callback=lambda completed, total: progress_updates.append(
            (completed, total)
        ),
    )[0]

    assert rescored["Q_run"] == pytest.approx(10.0)
    assert rescored["quality"]["Q_channels"]["1"] == pytest.approx(10.0)
    assert progress_updates == [(1, 1)]


def test_viewer_rescores_only_selected_group_and_channels():
    observations = [
        {
            "iteration": 1,
            "group_id": 1,
            "objective": "classic",
            "channels": [1, 2],
            "channel_metrics": {
                "1": {"peak_prominence": 2.0},
                "2": {"peak_prominence": 10.0},
            },
            "Q_run": 1.0,
        },
        {
            "iteration": 1,
            "group_id": 2,
            "objective": "classic",
            "channels": [3],
            "channel_metrics": {"3": {"peak_prominence": 50.0}},
            "Q_run": 99.0,
        },
    ]
    config = {
        "channel_groups": [
            {"id": 1, "channels": [1, 2]},
            {"id": 2, "channels": [3]},
        ],
        "acquisition": {"optimization_direction": "maximize"},
    }
    scoring = {
        "mode": "classic",
        "channel_weights": {
            "peak_prominence": 1.0,
            "peak_shape": 0.0,
            "baseline": 0.0,
            "replicate_consistency": 0.0,
            "success": 0.0,
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations(
        observations,
        config,
        scoring,
        rescore_scope={"group_ids": [1], "channels": ["2"]},
    )

    assert rescored[0]["Q_run"] == pytest.approx(10.0)
    assert set(rescored[0]["quality"]["Q_channels"]) == {"2"}
    assert rescored[1]["Q_run"] == pytest.approx(99.0)
    assert rescored[1]["_viewer_rescore_skipped"] is True


def test_scoped_rescore_preserves_unselected_channel_q_for_simulation():
    observation = {
        "method_id": "method-1",
        "iteration": 1,
        "group_id": 1,
        "objective": "paired_response",
        "channels": [1, 2],
        "params": {
            "frequency": 100.0,
            "amplitude": 0.05,
            "step_potential": 0.002,
        },
        "Q_run": 3.0,
        "quality": {
            "Q_run": 3.0,
            "Q_channels": {"1": 2.0, "2": 4.0},
            "channel_components": {
                "1": {"Q_channel": 2.0, "paired_Q_channel": 2.0},
                "2": {"Q_channel": 4.0, "paired_Q_channel": 4.0},
            },
        },
    }
    session = {
        "root": "/tmp/test-session",
        "config": {"scoring": {}},
        "observations": [observation],
        "history": pd.DataFrame(),
    }
    profile = {
        "id": "scoped-profile",
        "scoring": {},
        "rescore_scope": {"group_ids": [1], "channels": ["2"]},
        "results": [{
            "method_id": "method-1",
            "group_id": 1,
            "iteration": 1,
            "Q_run": 10.0,
            "quality": {
                "Q_run": 10.0,
                "Q_channels": {"2": 10.0},
                "channel_components": {
                    "2": {"Q_channel": 10.0, "paired_Q_channel": 10.0},
                },
            },
            "skipped": False,
        }],
    }

    active = _session_with_rescore_profile(session, profile)
    active_observation = active["observations"][0]
    assert active_observation["Q_run"] == pytest.approx(10.0)
    assert active_observation["quality"]["Q_channels"] == {
        "1": 2.0,
        "2": 10.0,
    }
    points = viewer._real_metric_points(
        active["observations"],
        "Paired Q",
        "measurement",
        ["1", "2"],
        average_channels=False,
    )
    assert dict(zip(points["channel"], points["value"])) == {
        "1": pytest.approx(2.0),
        "2": pytest.approx(10.0),
    }


def test_per_channel_paired_q_points_exclude_other_channels_run_q():
    observations = [
        {
            "method_id": f"method-{channel}",
            "group_id": channel,
            "iteration": 1,
            "channels": [channel],
            "params": {
                "frequency": 100.0 * channel,
                "amplitude": 0.01 * channel,
                "step_potential": 0.001 * channel,
            },
            "Q_run": float(channel * 10),
            "quality": {
                "Q_run": float(channel * 10),
                "Q_channels": {str(channel): float(channel * 10)},
                "channel_components": {
                    str(channel): {
                        "Q_channel": float(channel * 10),
                        "paired_Q_channel": float(channel * 10),
                    }
                },
            },
        }
        for channel in (1, 2)
    ]

    channel_one = viewer._real_metric_points(
        observations,
        "Paired Q",
        "measurement",
        ["1"],
        average_channels=False,
    )
    channel_two = viewer._real_metric_points(
        observations,
        "Paired Q",
        "measurement",
        ["2"],
        average_channels=False,
    )

    assert channel_one[["channel", "group_id", "value"]].to_dict("records") == [{
        "channel": "1",
        "group_id": 1,
        "value": 10.0,
    }]
    assert channel_two[["channel", "group_id", "value"]].to_dict("records") == [{
        "channel": "2",
        "group_id": 2,
        "value": 20.0,
    }]


def test_rescored_per_channel_paired_q_points_remain_channel_specific():
    observations = [
        {
            "method_id": f"method-{channel}",
            "group_id": channel,
            "iteration": 1,
            "channels": [channel],
            "params": {
                "frequency": 100.0 * channel,
                "amplitude": 0.01 * channel,
                "step_potential": 0.001 * channel,
            },
            "Q_run": float(channel),
            "quality": {
                "Q_run": float(channel),
                "Q_channels": {str(channel): float(channel)},
                "channel_components": {
                    str(channel): {"Q_channel": float(channel)}
                },
            },
        }
        for channel in (1, 2)
    ]
    session = {
        "root": "/tmp/test-session",
        "config": {"scoring": {}},
        "observations": observations,
        "history": pd.DataFrame(),
    }
    profile = {
        "id": "per-channel-profile",
        "scoring": {},
        "results": [
            {
                "method_id": f"method-{channel}",
                "group_id": channel,
                "iteration": 1,
                "Q_run": float(channel * 100),
                "quality": {
                    "Q_run": float(channel * 100),
                    "Q_channels": {str(channel): float(channel * 100)},
                    "channel_components": {
                        str(channel): {
                            "Q_channel": float(channel * 100),
                            "paired_Q_channel": float(channel * 100),
                        }
                    },
                },
                "skipped": False,
            }
            for channel in (1, 2)
        ],
    }

    active = _session_with_rescore_profile(session, profile)
    channel_one = viewer._real_metric_points(
        active["observations"],
        "Paired Q",
        "measurement",
        ["1"],
        average_channels=False,
    )
    channel_two = viewer._real_metric_points(
        active["observations"],
        "Paired Q",
        "measurement",
        ["2"],
        average_channels=False,
    )

    assert channel_one[["channel", "group_id", "value"]].to_dict("records") == [{
        "channel": "1",
        "group_id": 1,
        "value": 100.0,
    }]
    assert channel_two[["channel", "group_id", "value"]].to_dict("records") == [{
        "channel": "2",
        "group_id": 2,
        "value": 200.0,
    }]


def test_paired_q_run_fallback_remains_available_for_legacy_unknown_channel():
    observation = {
        "group_id": 1,
        "iteration": 1,
        "params": {
            "frequency": 100.0,
            "amplitude": 0.05,
            "step_potential": 0.002,
        },
        "Q_run": 3.5,
        "quality": {"Q_run": 3.5},
    }

    points = viewer._real_metric_points(
        [observation],
        "Paired Q",
        "measurement",
        ["1"],
        average_channels=False,
    )

    assert points[["channel", "value"]].to_dict("records") == [{
        "channel": "Run",
        "value": 3.5,
    }]


def test_paired_q_run_fallback_does_not_stand_in_for_multichannel_scores():
    observation = {
        "group_id": 1,
        "iteration": 1,
        "channels": [1, 2],
        "params": {
            "frequency": 100.0,
            "amplitude": 0.05,
            "step_potential": 0.002,
        },
        "Q_run": 3.5,
        "quality": {"Q_run": 3.5},
    }

    points = viewer._real_metric_points(
        [observation],
        "Paired Q",
        "measurement",
        ["1"],
        average_channels=False,
    )

    assert points.empty


def test_trace_realistic_simulation_uses_active_classic_scoring(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_trace_realistic_channel_measurements",
        lambda *_args, **_kwargs: ({
            "1": {
                "peak_prominence": 5.0,
                "success_score": 1.0,
            },
        }, []),
    )
    session = {
        "config": {
            "scoring": {
                "mode": "classic",
                "channel_weights": {
                    "peak_prominence": 2.0,
                    "peak_shape": 0.0,
                    "baseline": 0.0,
                    "replicate_consistency": 0.0,
                    "success": 0.0,
                },
                "run_weights": {
                    "lambda_variability": 0.0,
                    "lambda_failed": 0.0,
                    "lambda_low": 0.0,
                },
            },
        },
    }

    q_run, metadata = viewer._trace_realistic_measurement_value(
        session,
        [{"group_id": 1, "objective": "classic"}],
        {},
        ["1"],
        analysis={},
        nearest_count=1,
    )

    assert q_run == pytest.approx(10.0)
    assert metadata["channel_measurements"]["1"]["Q_channel"] == pytest.approx(10.0)


def test_trace_realistic_paired_simulation_keeps_phases_separate(monkeypatch):
    phases = []

    def fake_measurements(*_args, phase=None, **_kwargs):
        phases.append(phase)
        peak = 1.0 if phase == "buffer" else 5.0
        return ({"1": {
            "mean_peak_current_uA": peak,
            "mean_background_rms_uA": 1.0,
            "success_score": 1.0,
            "ok_scan_count": 2,
            "total_scan_count": 2,
        }}, [])

    monkeypatch.setattr(
        viewer,
        "_trace_realistic_channel_measurements",
        fake_measurements,
    )
    session = {
        "config": {
            "scoring": {
                "mode": "classic",
                "channel_weights": {
                    "peak_prominence": 0.0,
                    "peak_shape": 0.0,
                    "baseline": 0.0,
                    "replicate_consistency": 0.0,
                    "success": 0.0,
                },
                "paired_response_weights": {
                    "buffer_classic_Q": 0.0,
                    "target_classic_Q": 0.0,
                    "peak_prominence": 1.0,
                    "repeat_scan_snr": 0.0,
                },
                "run_weights": {
                    "lambda_variability": 0.0,
                    "lambda_failed": 0.0,
                    "lambda_low": 0.0,
                },
            },
        },
    }

    q_run, metadata = viewer._trace_realistic_measurement_value(
        session,
        [{"group_id": 1, "objective": "paired_response"}],
        {},
        ["1"],
        analysis={},
        nearest_count=1,
    )

    assert phases == ["buffer", "target"]
    assert q_run == pytest.approx(2.0)
    assert metadata["quality"]["Q_channels"]["1"] == pytest.approx(2.0)


def test_viewer_rescores_paired_observations_with_pairwise_snr():
    phase_common = {
        "peak_shape_score": 1.0,
        "baseline_stability_score": 1.0,
        "replicate_consistency_score": 1.0,
        "success_score": 1.0,
        "ok_scan_count": 2,
    }
    observation = {
        "iteration": 1,
        "group_id": 1,
        "objective": "paired_response",
        "buffer_channel_metrics": {
            "1": {
                **phase_common,
                "mean_peak_current_uA": 2.0,
                "std_peak_current_uA": .25,
                "mean_background_rms_uA": .5,
                "peak_prominence": 4.0,
            }
        },
        "target_channel_metrics": {
            "1": {
                **phase_common,
                "mean_peak_current_uA": 8.0,
                "std_peak_current_uA": .75,
                "mean_background_rms_uA": 1.5,
                "peak_prominence": 4.0,
            }
        },
    }
    scoring = {
        "mode": "classic",
        "channel_weights": {
            "peak_prominence": 0.0,
            "peak_shape": 0.0,
            "baseline": 0.0,
            "replicate_consistency": 0.0,
            "success": 1.0,
        },
        "paired_response_weights": {
            "buffer_classic_Q": 0.0,
            "target_classic_Q": 0.0,
            "peak_prominence": 0.0,
            "repeat_scan_snr": 1.0,
            "repeat_scan_snr_definition": "pairwise",
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations(
        [observation],
        {"acquisition": {"optimization_direction": "survey"}},
        scoring,
    )[0]

    pairwise_std = (1.25 / 3.0) ** .5
    rms_floor = (0.5 + 1.5) / 2.0
    regularized_std = math.hypot(pairwise_std, rms_floor)
    component = rescored["quality"]["channel_components"]["1"]
    assert component["pairwise_baseline_rms_floor_uA"] == pytest.approx(rms_floor)
    assert component["pairwise_std_floor_uA"] == pytest.approx(rms_floor)
    assert component["repeat_scan_snr"] == pytest.approx(6.0 / regularized_std)
    assert rescored["Q_run"] == pytest.approx(6.0 / regularized_std)


def test_paired_rescore_zero_pads_failed_fit_for_std_without_changing_mean():
    observation = {
        "iteration": 1,
        "group_id": 1,
        "objective": "paired_response",
        "buffer_channel_metrics": {"1": {
            "peak_currents_uA": [1.0, 1.1],
            "mean_peak_current_uA": 1.05,
            "std_peak_current_uA": 0.05,
            "success_score": 2.0 / 3.0,
            "ok_scan_count": 2,
            "total_scan_count": 3,
        }},
        "target_channel_metrics": {"1": {
            "peak_currents_uA": [2.0, 2.1, 2.2],
            "mean_peak_current_uA": 2.1,
            "std_peak_current_uA": 0.1,
            "success_score": 1.0,
            "ok_scan_count": 3,
            "total_scan_count": 3,
        }},
    }
    scoring = {
        "channel_weights": {"success": 1.0},
        "paired_response_weights": {
            "buffer_classic_Q": 0.5,
            "target_classic_Q": 0.5,
            "peak_prominence": 0.0,
            "repeat_scan_snr": 0.0,
            "repeat_scan_snr_definition": "pairwise",
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations(
        [observation],
        {"acquisition": {"optimization_direction": "maximize"}},
        scoring,
    )[0]
    quality = rescored["quality"]
    component = quality["channel_components"]["1"]

    assert component["buffer_all_peaks_identified"] is False
    assert component["target_all_peaks_identified"] is True
    assert component["buffer_ok_scan_count"] == 2
    assert component["buffer_total_scan_count"] == 3
    assert component["buffer_failure_adjusted_peak_currents_uA"] == pytest.approx(
        [1.0, 1.1, 0.0]
    )
    assert component["target_failure_adjusted_peak_currents_uA"] == pytest.approx(
        [2.0, 2.1, 2.2]
    )
    assert component["buffer_peak_std_uA"] == pytest.approx(
        0.6082762530298219
    )
    assert component["delta_peak_height_uA"] == pytest.approx(1.05)
    assert component["minimum_phase_peaks_identified"] is True
    assert component["paired_Q_channel"] > 0.0
    assert component["success_score"] == pytest.approx(2.0 / 3.0)
    assert quality["Q_run"] > 0.0
    assert quality["peak_completeness_gate_applied"] is False
    assert quality["incomplete_peak_channels"] == ["1"]


def test_one_partially_successful_channel_does_not_zero_entire_rescore():
    complete = {
        "mean_peak_current_uA": 1.0,
        "success_score": 1.0,
        "ok_scan_count": 3,
        "total_scan_count": 3,
    }
    observation = {
        "iteration": 1,
        "group_id": 1,
        "objective": "paired_response",
        "buffer_channel_metrics": {
            "1": complete,
            "2": {
                **complete,
                "success_score": 2.0 / 3.0,
                "ok_scan_count": 2,
            },
        },
        "target_channel_metrics": {
            "1": {**complete, "mean_peak_current_uA": 2.0},
            "2": {**complete, "mean_peak_current_uA": 2.0},
        },
    }
    scoring = {
        "channel_weights": {"success": 1.0},
        "paired_response_weights": {
            "buffer_classic_Q": 0.5,
            "target_classic_Q": 0.5,
            "peak_prominence": 0.0,
            "repeat_scan_snr": 0.0,
        },
        "run_weights": {
            "low_channel_threshold": 0.0,
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations(
        [observation],
        {"acquisition": {"optimization_direction": "maximize"}},
        scoring,
    )[0]
    quality = rescored["quality"]

    assert quality["channel_components"]["1"]["paired_Q_channel"] > 0.0
    assert quality["channel_components"]["2"]["paired_Q_channel"] > 0.0
    assert quality["all_phase_peaks_identified"] is False
    assert quality["incomplete_peak_channels"] == ["2"]
    assert quality["Q_run"] > 0.0


def test_paired_rescore_zeros_channel_with_fewer_than_two_peaks_in_a_phase():
    observation = {
        "objective": "paired_response",
        "buffer_channel_metrics": {"1": {
            "peak_currents_uA": [1.0],
            "mean_peak_current_uA": 1.0,
            "success_score": 1.0 / 3.0,
            "ok_scan_count": 1,
            "total_scan_count": 3,
        }},
        "target_channel_metrics": {"1": {
            "peak_currents_uA": [2.0, 2.1, 2.2],
            "mean_peak_current_uA": 2.1,
            "success_score": 1.0,
            "ok_scan_count": 3,
            "total_scan_count": 3,
        }},
    }
    scoring = {
        "channel_weights": {"success": 1.0},
        "paired_response_weights": {
            "buffer_classic_Q": 0.5,
            "target_classic_Q": 0.5,
            "peak_prominence": 0.0,
            "repeat_scan_snr": 1.0,
            "repeat_scan_snr_definition": "pairwise",
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations([observation], {}, scoring)[0]
    quality = rescored["quality"]
    component = quality["channel_components"]["1"]

    assert component["buffer_failure_adjusted_peak_currents_uA"] == [1.0, 0.0, 0.0]
    assert component["minimum_phase_peaks_identified"] is False
    assert component["paired_Q_channel"] == 0.0
    assert quality["peak_completeness_gate_applied"] is True
    assert quality["insufficient_peak_channels"] == ["1"]


def test_paired_only_rescore_does_not_require_nonzero_classic_q_weights():
    observation = {
        "iteration": 1,
        "group_id": 1,
        "objective": "paired_response",
        "buffer_channel_metrics": {"1": {
            "peak_currents_uA": [1.0, 2.0, 3.0],
            "mean_peak_current_uA": 2.0,
            "std_peak_current_uA": 1.0,
            "success_score": 1.0,
            "ok_scan_count": 3,
        }},
        "target_channel_metrics": {"1": {
            "peak_currents_uA": [8.0, 9.0, 10.0],
            "mean_peak_current_uA": 9.0,
            "std_peak_current_uA": 1.0,
            "success_score": 1.0,
            "ok_scan_count": 3,
        }},
    }
    scoring = {
        "channel_weights": {
            "peak_prominence": 0.0,
            "repeat_scan_snr": 0.0,
            "peak_height": 0.0,
            "peak_shape": 0.0,
            "baseline": 0.0,
            "replicate_consistency": 0.0,
            "success": 0.0,
            "noise_penalty": 0.0,
        },
        "paired_response_weights": {
            "buffer_classic_Q": 0.0,
            "target_classic_Q": 0.0,
            "peak_prominence": 0.0,
            "repeat_scan_snr": 1.0,
            "repeat_scan_snr_definition": "pairwise",
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations(
        [observation],
        {"acquisition": {"optimization_direction": "maximize"}},
        scoring,
    )[0]
    component = rescored["quality"]["channel_components"]["1"]

    assert component["buffer_classic_Q"] == pytest.approx(0.0)
    assert component["target_classic_Q"] == pytest.approx(0.0)
    assert component["valid_source_pair"] is True
    assert component["pairwise_peak_differences_uA"] == pytest.approx([
        7.0, 8.0, 9.0,
        6.0, 7.0, 8.0,
        5.0, 6.0, 7.0,
    ])
    assert component["repeat_scan_snr"] > 0
    assert rescored["Q_run"] == pytest.approx(component["repeat_scan_snr"])


def test_pairwise_std_floor_regularizes_near_zero_denominators():
    observation = {
        "objective": "paired_response",
        "buffer_channel_metrics": {"1": {
            "peak_currents_uA": [0.999, 1.0, 1.001],
            "mean_peak_current_uA": 1.0,
            "std_peak_current_uA": 0.001,
            "success_score": 1.0,
            "ok_scan_count": 3,
            "mean_background_rms_uA": 0.008,
        }},
        "target_channel_metrics": {"1": {
            "peak_currents_uA": [1.299, 1.3, 1.301],
            "mean_peak_current_uA": 1.3,
            "std_peak_current_uA": 0.001,
            "success_score": 1.0,
            "ok_scan_count": 3,
            "mean_background_rms_uA": 0.012,
        }},
    }
    scoring = {
        "channel_weights": {"success": 1.0, "peak_prominence": 0.0},
        "paired_response_weights": {
            "buffer_classic_Q": 0.0,
            "target_classic_Q": 0.0,
            "peak_prominence": 0.0,
            "repeat_scan_snr": 1.0,
            "repeat_scan_snr_definition": "pairwise",
            "pairwise_std_floor_uA": 0.002,
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations([observation], {}, scoring)[0]
    component = rescored["quality"]["channel_components"]["1"]

    assert component["pairwise_configured_std_floor_uA"] == pytest.approx(0.002)
    assert component["pairwise_baseline_rms_floor_uA"] == pytest.approx(0.01)
    assert component["pairwise_std_floor_uA"] == pytest.approx(0.01)
    assert component["pairwise_regularized_std_uA"] == pytest.approx(
        math.hypot(component["pairwise_peak_difference_std_uA"], 0.01)
    )
    assert component["pairwise_regularized_std_uA"] > component[
        "pairwise_peak_difference_std_uA"
    ]
    assert component["repeat_scan_snr"] < component[
        "unregularized_repeat_scan_snr"
    ]
    assert component["repeat_scan_snr"] == pytest.approx(
        0.3 / component["pairwise_regularized_std_uA"]
    )


def test_pairwise_rescore_does_not_replace_missing_saved_inputs_with_new_analysis():
    observation = {
        "objective": "paired_response",
        "buffer_channel_metrics": {"1": {"success_score": 1.0}},
        "target_channel_metrics": {"1": {"success_score": 1.0}},
    }
    scoring = {
        "channel_weights": {
            "peak_prominence": 0.0,
            "repeat_scan_snr": 0.0,
            "peak_height": 0.0,
            "peak_shape": 0.0,
            "baseline": 0.0,
            "replicate_consistency": 0.0,
            "success": 0.0,
        },
        "paired_response_weights": {
            "buffer_classic_Q": 0.0,
            "target_classic_Q": 0.0,
            "peak_prominence": 0.0,
            "repeat_scan_snr": 1.0,
            "repeat_scan_snr_definition": "pairwise",
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations([observation], {}, scoring)[0]
    component = rescored["quality"]["channel_components"]["1"]

    assert component["pairwise_inputs_available"] is False
    assert component["pairwise_peak_differences_uA"] == []
    assert component["repeat_scan_snr"] == 0.0
    assert component["repeat_scan_snr_contribution"] == 0.0
    assert rescored["Q_run"] == 0.0


def test_rescore_excludes_stale_metrics_outside_observation_channels():
    valid_phase = {
        "mean_peak_current_uA": 1.0,
        "std_peak_current_uA": 0.1,
        "success_score": 1.0,
        "ok_scan_count": 3,
    }
    observation = {
        "objective": "paired_response",
        "channels": ["1"],
        "buffer_channel_metrics": {
            "1": valid_phase,
            "2": {**valid_phase, "success_score": 0.0},
        },
        "target_channel_metrics": {
            "1": {**valid_phase, "mean_peak_current_uA": 3.0},
            "2": {**valid_phase, "success_score": 0.0},
        },
    }
    scoring = {
        "channel_weights": {"success": 1.0, "peak_prominence": 0.0},
        "paired_response_weights": {
            "buffer_classic_Q": 0.0,
            "target_classic_Q": 0.0,
            "peak_prominence": 1.0,
            "repeat_scan_snr": 0.0,
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations([observation], {}, scoring)[0]

    assert set(rescored["quality"]["channel_components"]) == {"1"}
    assert rescored["quality"]["mean_Q_channel"] == pytest.approx(
        rescored["quality"]["channel_components"]["1"]["Q_channel"]
    )


def test_viewer_rescore_defaults_to_original_paired_snr_definition():
    phase_common = {"success_score": 1.0, "ok_scan_count": 2}
    observation = {
        "iteration": 1,
        "objective": "paired_response",
        "buffer_channel_metrics": {
            "1": {**phase_common, "mean_peak_current_uA": 2.0, "std_peak_current_uA": .25}
        },
        "target_channel_metrics": {
            "1": {**phase_common, "mean_peak_current_uA": 8.0, "std_peak_current_uA": .75}
        },
    }
    scoring = {
        "channel_weights": {"success": 1.0, "peak_prominence": 0.0},
        "paired_response_weights": {
            "buffer_classic_Q": 0.0,
            "target_classic_Q": 0.0,
            "peak_prominence": 0.0,
            "repeat_scan_snr": 1.0,
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations(
        [observation],
        {"acquisition": {"optimization_direction": "survey"}},
        scoring,
    )[0]

    component = rescored["quality"]["channel_components"]["1"]
    assert component["combined_peak_std_uA"] == pytest.approx(1.0)
    assert component["repeat_scan_snr"] == pytest.approx(6.0)


def test_viewer_saves_rescored_session_with_backups(tmp_path):
    observation = {
        "iteration": 1,
        "group_id": 1,
        "method_id": "method_1",
        "Q_run": 3.0,
        "quality": {"Q_run": 3.0},
        "channel_metrics": {},
    }
    state = {
        "observations": [{**observation, "Q_run": 1.0}],
        "suggestions": [{"method_id": "method_1", "Q_run": 1.0}],
    }
    config = {"acquisition": {"optimization_direction": "maximize"}}
    (tmp_path / "bo_state.json").write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "bo_config_snapshot.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    pd.DataFrame([{"iteration": 1, "group_id": 1, "Q_run": 1.0}]).to_csv(
        tmp_path / "history.csv", index=False
    )
    session = {
        "root": tmp_path,
        "state": state,
        "config": config,
        "observations": state["observations"],
        "history": pd.read_csv(tmp_path / "history.csv"),
    }

    backups = _persist_rescored_bo_session(
        session,
        [observation],
        {"mode": "classic", "channel_weights": {}, "run_weights": {}},
    )

    saved_state = json.loads((tmp_path / "bo_state.json").read_text(encoding="utf-8"))
    saved_history = pd.read_csv(tmp_path / "history.csv")
    assert len(backups) == 3
    assert saved_state["observations"][0]["Q_run"] == pytest.approx(3.0)
    assert saved_state["suggestions"][0]["Q_run"] == pytest.approx(3.0)
    assert saved_history["Q_run"].iloc[0] == pytest.approx(3.0)


def test_viewer_caches_distinct_rescores_and_applies_them_globally(tmp_path):
    original = {
        "iteration": 1,
        "group_id": 1,
        "method_id": "method_1",
        "objective": "classic",
        "Q_run": 1.0,
        "quality": {"Q_run": 1.0},
        "channel_metrics": {"1": {"peak_prominence": 2.0}},
    }
    session = {
        "root": tmp_path,
        "config": {"scoring": {}},
        "state": {"observations": [original]},
        "observations": [original],
        "history": pd.DataFrame([{"iteration": 1, "group_id": 1, "Q_run": 1.0}]),
    }
    rescored = [{
        **original,
        "Q_run": 7.0,
        "quality": {"Q_run": 7.0, "Q_channels": {"1": 7.0}},
    }]
    scoring_a = {"mode": "classic", "channel_weights": {"peak_prominence": 1.0}}
    scoring_b = {"mode": "classic", "channel_weights": {"peak_prominence": 2.0}}

    profile_a = _cache_rescore_profile(session, scoring_a, rescored, "First")
    same_profile = _cache_rescore_profile(session, scoring_a, rescored, "Renamed")
    _cache_rescore_profile(session, scoring_b, rescored, "Second")
    profiles = _load_rescore_profiles(session)
    active_session = _session_with_rescore_profile(session, same_profile)

    assert profile_a["id"] == same_profile["id"]
    assert len(profiles) == 2
    assert profiles[profile_a["id"]]["label"] == "Renamed"
    assert active_session["observations"][0]["Q_run"] == pytest.approx(7.0)
    assert active_session["history"]["Q_run"].iloc[0] == pytest.approx(7.0)
    assert active_session["config"]["scoring"] == scoring_a
    assert session["observations"][0]["Q_run"] == pytest.approx(1.0)


def test_active_paired_rescore_updates_history_channel_scores_and_best_parameters(tmp_path):
    original = {
        "iteration": 1,
        "group_id": 1,
        "method_id": "paired_1",
        "objective": "paired_response",
        "params": {"frequency": 200.0, "amplitude": 0.04, "step_potential": 0.002},
        "Q_run": 99.0,
        "channel_metrics": {"1": {"Q_channel": 99.0}},
        "buffer_channel_metrics": {"1": {"Q_channel": 88.0, "classic_Q": 88.0}},
        "target_channel_metrics": {"1": {"Q_channel": 77.0, "classic_Q": 77.0}},
        "quality": {"Q_run": 99.0},
    }
    quality = {
        "Q_run": 5.0,
        "Q_channels": {"1": 5.0},
        "channel_components": {"1": {
            "Q_channel": 5.0,
            "paired_Q_channel": 5.0,
            "buffer_classic_components": {"Q_channel": 0.0},
            "target_classic_components": {"Q_channel": 0.0},
        }},
    }
    session = {
        "root": tmp_path,
        "config": {"scoring": {}},
        "observations": [original],
        "history": pd.DataFrame([{
            "iteration": 1,
            "group_id": 1,
            "objective": "paired_response",
            "Q_run": 99.0,
            "ch1_Q_channel": 99.0,
            "frequency": 200.0,
            "amplitude": 0.04,
            "step_potential": 0.002,
        }]),
    }
    profile = {
        "id": "paired-profile",
        "label": "Paired only",
        "scoring": {},
        "results": [{
            "method_id": "paired_1",
            "group_id": 1,
            "iteration": 1,
            "Q_run": 5.0,
            "quality": quality,
            "skipped": False,
        }],
    }

    active = _session_with_rescore_profile(session, profile)
    history = active["history"]
    best_parameters = _best_q_parameters_by_channel_frame(history)

    assert active["observations"][0]["Q_run"] == pytest.approx(5.0)
    assert active["observations"][0]["channel_metrics"]["1"]["Q_channel"] == pytest.approx(5.0)
    assert active["observations"][0]["buffer_channel_metrics"]["1"]["Q_channel"] == pytest.approx(0.0)
    assert history.loc[0, "Q_run"] == pytest.approx(5.0)
    assert history.loc[0, "ch1_Q_channel"] == pytest.approx(5.0)
    assert history.loc[0, "ch1_paired_Q_channel"] == pytest.approx(5.0)
    assert best_parameters.loc[0, "Highest Q"] == pytest.approx(5.0)
    assert best_parameters.loc[0, "Lowest Q"] == pytest.approx(5.0)


def test_channel_q_summary_includes_parameters_for_both_extrema():
    history = pd.DataFrame([
        {
            "iteration": 1,
            "frequency": 100.0,
            "amplitude": 0.05,
            "step_potential": 0.001,
            "Q_ch1": -2.0,
        },
        {
            "iteration": 2,
            "frequency": 300.0,
            "amplitude": 0.10,
            "step_potential": 0.003,
            "Q_ch1": 7.0,
        },
    ])

    summary = _best_q_parameters_by_channel_frame(history)

    assert summary.loc[0, "Highest Q iteration"] == 2
    assert summary.loc[0, "Highest Q"] == pytest.approx(7.0)
    assert summary.loc[0, "Frequency at highest Q (Hz)"] == pytest.approx(300.0)
    assert summary.loc[0, "Lowest Q iteration"] == 1
    assert summary.loc[0, "Lowest Q"] == pytest.approx(-2.0)
    assert summary.loc[0, "Frequency at lowest Q (Hz)"] == pytest.approx(100.0)


def test_compact_channel_q_summary_includes_highest_and_lowest_runs():
    history = pd.DataFrame([
        {
            "iteration": 4,
            "ground_truth_channel": "2",
            "observed_value": 8.0,
            "frequency": 400.0,
            "amplitude": 0.12,
            "step_potential": 0.004,
            "run_label": "maximize",
        },
        {
            "iteration": 7,
            "ground_truth_channel": "2",
            "observed_value": -3.0,
            "frequency": 150.0,
            "amplitude": 0.06,
            "step_potential": 0.002,
            "run_label": "minimize",
        },
    ])

    summary = _best_q_parameters_by_channel_frame(history)

    assert summary.loc[0, "Highest Q"] == pytest.approx(8.0)
    assert summary.loc[0, "Highest Q run"] == "maximize"
    assert summary.loc[0, "Lowest Q"] == pytest.approx(-3.0)
    assert summary.loc[0, "Lowest Q run"] == "minimize"
