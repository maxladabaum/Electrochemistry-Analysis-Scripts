import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bo_session_viewer import (
    _compact_simulation_completion_summary,
    _finalize_incremental_compact_simulated_sweep,
    _hyperparameter_response_frame,
    _incremental_compact_completed_run_indices,
    _prepare_landscape_candidate_records,
    _parallel_simulation_progress_snapshot,
    _run_landscape_bo_simulation,
    _simulation_directional_value,
    _simulation_objective_value,
    _simulation_optimum_reference_from_frame,
    _simulation_sweep_run_counts,
    _starting_point_response_frame,
    _starting_point_response_grouped,
    _upgrade_incremental_compact_run_indices,
    load_bo_session,
)


def test_prepared_candidates_preserve_simulation_results():
    ground_truth = pd.DataFrame({
        "frequency": [100.0, 100.0, 200.0, 200.0],
        "amplitude": [0.03, 0.04, 0.03, 0.04],
        "step_potential": [0.002, 0.002, 0.003, 0.003],
        "fitness": [1.0, 2.0, 3.0, 4.0],
    })
    config = {"acquisition": {"use_gp": False}}
    prepared = _prepare_landscape_candidate_records(
        ground_truth,
        "frequency",
        "amplitude",
        "step_potential",
        "fitness",
        config,
    )
    kwargs = dict(
        config=config,
        iterations=4,
        initial_random_points=2,
        candidate_pool_size=4,
        force_initial_point=False,
        seed=17,
    )

    ordinary, _ = _run_landscape_bo_simulation(
        ground_truth,
        "frequency",
        "amplitude",
        "step_potential",
        "fitness",
        **kwargs,
    )
    cached, _ = _run_landscape_bo_simulation(
        ground_truth,
        "frequency",
        "amplitude",
        "step_potential",
        "fitness",
        prepared_candidates=prepared,
        **kwargs,
    )

    pd.testing.assert_frame_equal(ordinary, cached)


def test_incremental_resume_recovers_out_of_order_legacy_runs_by_seed(tmp_path):
    pd.DataFrame({
        "run_index": [1, 2, 3, 4, 5],
        "seed": [101, 102, 103, 104, 105],
    }).to_csv(tmp_path / "simulation_run_manifest.csv", index=False)
    pd.DataFrame({
        "group_id": [1, 2, 3],
        "seed": [101, 104, 105],
    }).to_csv(tmp_path / "simulation_run_summary.csv", index=False)

    assert _incremental_compact_completed_run_indices(tmp_path) == {1, 4, 5}


def test_incremental_resume_upgrades_legacy_csv_columns(tmp_path):
    pd.DataFrame({
        "run_index": [1, 2, 3, 4],
        "seed": [101, 102, 103, 104],
    }).to_csv(tmp_path / "simulation_run_manifest.csv", index=False)
    pd.DataFrame({
        "group_id": [1, 2],
        "seed": [104, 102],
        "best_fitness": [4.0, 2.0],
    }).to_csv(tmp_path / "simulation_run_summary.csv", index=False)
    pd.DataFrame({
        "group_id": [1, 1, 2, 2],
        "iteration": [1, 2, 1, 2],
    }).to_csv(tmp_path / "history.csv", index=False)

    _upgrade_incremental_compact_run_indices(tmp_path)

    summary = pd.read_csv(tmp_path / "simulation_run_summary.csv")
    history = pd.read_csv(tmp_path / "history.csv")
    assert summary["run_index"].tolist() == [4, 2]
    assert history["run_index"].tolist() == [4, 4, 2, 2]


def test_incremental_resume_prefers_persisted_run_indices(tmp_path):
    pd.DataFrame({
        "run_index": [7, 2, 11],
        "seed": [1, 1, 1],
    }).to_csv(tmp_path / "simulation_run_summary.csv", index=False)

    assert _incremental_compact_completed_run_indices(tmp_path) == {2, 7, 11}


def test_incremental_checkpoint_does_not_duplicate_groups_in_json(tmp_path):
    context = {
        "root": tmp_path,
        "timestamp": "20260816_120000",
        "config": {},
        "source_session": "source",
        "value_label": "Q",
        "axes": ("frequency", "amplitude", "step_potential"),
        "simulation_settings": {},
        "total_runs": 1250,
        "completed_runs": 1250,
        "completed_run_indices": set(range(1, 1251)),
        "completed_channels": {str(channel) for channel in range(1, 11)},
        "first_initial_parameters": {"frequency": 200.0},
        "best_summary": None,
    }

    _finalize_incremental_compact_simulated_sweep(context, status="running")

    import json

    state = json.loads((tmp_path / "bo_state.json").read_text())
    config = json.loads((tmp_path / "bo_config_snapshot.json").read_text())
    assert state["channel_groups"] == []
    assert config["channel_groups"] == []
    assert len(state["simulation_metadata"]["completed_run_indices"]) == 1250
    assert (tmp_path / "bo_state.json").stat().st_size < 25_000


def test_per_channel_sweep_count_reports_fully_expanded_runs():
    base, expanded = _simulation_sweep_run_counts(
        25,
        5,
        per_channel=True,
        channel_count=10,
        runs_per_channel=1,
    )

    assert base == 125
    assert expanded == 1250


def test_parallel_progress_uses_true_overall_run_total():
    snapshot = _parallel_simulation_progress_snapshot(
        disk_completed_runs=0,
        completed_parallel_runs=89,
        total_runs=1250,
        worker_iteration_progress={
            90: (50, 50),
            91: (50, 50),
            92: (50, 50),
            93: (49, 50),
        },
        completed_parallel_indices=set(),
    )

    assert snapshot["completed_runs"] == 89
    assert snapshot["active_runs"] == 4
    assert snapshot["equivalent_completed_runs"] == pytest.approx(
        89 + 3 + 49 / 50
    )
    assert snapshot["overall_fraction"] == pytest.approx(
        (89 + 3 + 49 / 50) / 1250
    )


def test_compact_simulation_reports_runs_separately_from_iteration_rows():
    observations = [{"iteration": iteration} for iteration in range(4550)]
    session = {
        "state": {
            "simulation_metadata": {
                "compact_export": True,
                "run_count": 91,
                "planned_run_count": 1250,
            }
        },
        "history": pd.DataFrame({"group_id": [1, 2]}),
    }

    summary = _compact_simulation_completion_summary(session, observations)

    assert summary["saved_runs"] == 91
    assert summary["planned_runs"] == 1250
    assert summary["history_rows"] == 4550
    assert summary["metric_value"] == "91 / 1,250"


def test_compact_session_load_reconstructs_normalized_observations(tmp_path):
    (tmp_path / "bo_state.json").write_text(json.dumps({
        "observations": [],
        "simulation_metadata": {"compact_export": True, "run_count": 2},
    }))
    (tmp_path / "bo_config_snapshot.json").write_text(json.dumps({
        "records": {
            "simulated_session": True,
            "compact_simulation_export": True,
        },
        "channel_groups": [],
    }))
    pd.DataFrame({
        "iteration": [2, 1],
        "group_id": [2, 1],
        "Q_run": [4.0, 3.0],
        "ground_truth_channel": ["2.0", "1.0"],
        "frequency": [200.0, 100.0],
    }).to_csv(tmp_path / "history.csv", index=False)

    session = load_bo_session(tmp_path)

    assert [obs["group_id"] for obs in session["observations"]] == [1, 2]
    assert [obs["channels"] for obs in session["observations"]] == [[1], [2]]
    assert session["observations"][0]["quality"]["channel_components"]["1"][
        "Q_channel"
    ] == pytest.approx(3.0)
    assert session["history"]["ground_truth_channel"].tolist() == ["2", "1"]
    assert pd.isna(session["history"]["Q_ch1"].iloc[0])
    assert session["history"]["Q_ch1"].iloc[1] == pytest.approx(3.0)


def test_hyperparameter_response_can_be_filtered_to_one_simulation_channel():
    rows = []
    for channel, offset in (("1", 0.0), ("2", 10.0)):
        for exploration in (0.2, 0.8):
            for falloff in (0.3, 0.7):
                for iteration in (1, 2):
                    rows.append({
                        "group_id": len(rows) // 2 + 1,
                        "iteration": iteration,
                        "ground_truth_channel": channel,
                        "exploration": exploration,
                        "gp_falloff_value": falloff,
                        "Q_run": offset + exploration + falloff + iteration,
                    })
    history = pd.DataFrame(rows)
    channel_two_history = history.loc[
        history["ground_truth_channel"] == "2"
    ].copy()

    response = _hyperparameter_response_frame(
        channel_two_history,
        "Q_run",
        "Final iteration",
    )

    assert set(response["ground_truth_channel"].astype(str)) == {"2"}
    assert set(response["response_channels"]) == {"Ch 2"}
    assert len(response) == 4
    assert response["metric_value"].min() > 10.0


@pytest.mark.parametrize(
    ("direction", "raw_value", "clipped_value", "objective_value"),
    [
        ("maximize", -2.5, 0.0, 0.0),
        ("maximize", 2.5, 2.5, 2.5),
        ("minimize", 2.5, 0.0, 0.0),
        ("minimize", -2.5, -2.5, 2.5),
    ],
)
def test_simulation_direction_clips_disallowed_fitness_sign(
    direction,
    raw_value,
    clipped_value,
    objective_value,
):
    config = {"acquisition": {"optimization_direction": direction}}

    assert _simulation_directional_value(raw_value, config) == clipped_value
    assert _simulation_objective_value(raw_value, config) == objective_value


@pytest.mark.parametrize(
    ("direction", "expected_values", "expected_best"),
    [
        ("maximize", {0.0, 2.0}, 2.0),
        ("minimize", {-3.0, 0.0}, -3.0),
    ],
)
def test_landscape_optimizer_supports_signed_maximize_and_minimize(
    direction,
    expected_values,
    expected_best,
):
    ground_truth = pd.DataFrame({
        "frequency": [100.0, 200.0],
        "amplitude": [0.03, 0.04],
        "step_potential": [0.002, 0.003],
        "fitness": [-3.0, 2.0],
    })
    config = {"acquisition": {"optimization_direction": direction}}

    history, _candidates = _run_landscape_bo_simulation(
        ground_truth,
        "frequency",
        "amplitude",
        "step_potential",
        "fitness",
        config=config,
        iterations=2,
        initial_random_points=2,
        candidate_pool_size=2,
        force_initial_point=False,
        seed=7,
    )

    assert set(history["observed_value"]) == expected_values
    assert history["best_so_far"].iloc[-1] == expected_best

    optimum = _simulation_optimum_reference_from_frame(
        ground_truth.rename(columns={"fitness": "ground_truth_value"}),
        config=config,
    )
    assert optimum["Q_run"] == expected_best


def _starting_point_history():
    return pd.DataFrame({
        "run_index": [1, 1, 2, 2, 3, 3],
        "group_id": [11, 11, 12, 12, 13, 13],
        "iteration": [1, 2, 1, 2, 1, 2],
        # Later rows deliberately move away from each run's starting point.
        "frequency": [100.0, 500.0, 200.0, 600.0, 100.0, 700.0],
        "amplitude": [0.10, 0.50, 0.20, 0.60, 0.10, 0.70],
        "step_potential": [0.001, 0.005, 0.002, 0.006, 0.001, 0.007],
        "Q_run": [1.0, 4.0, -2.0, -5.0, 3.0, 6.0],
        "ground_truth_channel": [1, 1, 1, 1, 1, 1],
    })


def test_starting_point_map_uses_first_coordinates_and_requested_iteration():
    response = _starting_point_response_frame(
        _starting_point_history(),
        "Q_run",
        "Value at iteration",
        iteration=2,
    )

    first = response.loc[response["run_index"] == 1].iloc[0]
    assert first["frequency"] == 100.0
    assert first["amplitude"] == pytest.approx(0.10)
    assert first["step_potential"] == pytest.approx(0.001)
    assert first["metric_value"] == 4.0
    assert first["source_iteration"] == 2


@pytest.mark.parametrize(
    ("direction", "expected"),
    [("maximize", -2.0), ("minimize", -5.0)],
)
def test_starting_point_best_achieved_follows_optimization_direction(
    direction,
    expected,
):
    response = _starting_point_response_frame(
        _starting_point_history(),
        "Q_run",
        "Best achieved",
        optimization_direction=direction,
    )

    assert response.loc[response["run_index"] == 2, "metric_value"].iloc[0] == expected


def test_starting_point_map_aggregates_duplicate_starts_and_counts_runs():
    response = _starting_point_response_frame(
        _starting_point_history(),
        "Q_run",
        "Final value",
    )
    grouped = _starting_point_response_grouped(
        response,
        axes=("frequency", "amplitude", "step_potential"),
        aggregate="Mean",
    )

    duplicate = grouped.loc[grouped["frequency"] == 100.0].iloc[0]
    assert duplicate["runs"] == 2
    assert duplicate["metric_value"] == pytest.approx(5.0)
