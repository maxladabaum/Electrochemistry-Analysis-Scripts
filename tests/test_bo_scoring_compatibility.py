import json
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bo_headless import _build_channel_metrics, _pairwise_repeat_snr
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
    _observation_table,
    _paired_q_equation,
    _paired_q_description,
    _persist_rescored_bo_session,
    _rescore_observations,
    _resolved_rescore_label,
    _session_with_rescore_profile,
    _store_rescore_profiles,
)


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
    component = rescored["quality"]["channel_components"]["1"]
    assert component["repeat_scan_snr"] == pytest.approx(6.0 / pairwise_std)
    assert rescored["Q_run"] == pytest.approx(6.0 / pairwise_std)


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
        }},
        "target_channel_metrics": {"1": {
            "peak_currents_uA": [1.299, 1.3, 1.301],
            "mean_peak_current_uA": 1.3,
            "std_peak_current_uA": 0.001,
            "success_score": 1.0,
            "ok_scan_count": 3,
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
            "pairwise_std_floor_uA": 0.01,
        },
        "run_weights": {
            "lambda_variability": 0.0,
            "lambda_failed": 0.0,
            "lambda_low": 0.0,
        },
    }

    rescored = _rescore_observations([observation], {}, scoring)[0]
    component = rescored["quality"]["channel_components"]["1"]

    assert component["pairwise_regularized_std_uA"] > component[
        "pairwise_peak_difference_std_uA"
    ]
    assert component["repeat_scan_snr"] < component[
        "unregularized_repeat_scan_snr"
    ]
    assert component["repeat_scan_snr"] == pytest.approx(
        0.3 / component["pairwise_regularized_std_uA"]
    )


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
    assert best_parameters.loc[0, "Best Q"] == pytest.approx(5.0)
