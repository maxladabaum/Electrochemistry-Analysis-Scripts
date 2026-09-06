import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer


def test_channel_metrics_keep_constant_and_single_point_channels():
    history = pd.DataFrame({
        "iteration": [1, 2, 3],
        "Q_ch1": [0.2, 0.4, 0.8],
        "Q_ch2": [0.0, 0.0, 0.0],
        "ch3_Q_channel": [None, 0.3, None],
        "ch4_Q_channel": [None, None, None],
    })

    channels = viewer._channel_metric_columns(history)["Q_channel"]

    assert channels == {"1": "Q_ch1", "2": "Q_ch2", "3": "ch3_Q_channel"}
    figure = viewer._plot_channel_trend(
        history, "Q_channel", channels, ["2"], "Overlay selected channels"
    )
    assert list(figure.data[0].y) == [0.0, 0.0, 0.0]


def test_separate_simulation_channel_plot_excludes_other_channel_rows():
    history = pd.DataFrame({
        "iteration": [1, 2, 1, 2],
        "ground_truth_channel": ["1", "1", "2", "2"],
        "frequency": [10.0, 20.0, 80.0, 90.0],
    })

    figure = viewer._plot_channel_trend(
        history,
        "frequency",
        {"1": "frequency", "2": "frequency"},
        ["2"],
        "Overlay selected channels",
    )

    assert list(figure.data[0].x) == [1, 2]
    assert list(figure.data[0].y) == [80.0, 90.0]


def test_expanded_analysis_channels_are_available_from_saved_history():
    session = {
        "history": pd.DataFrame(),
        "observations": [
            {
                "iteration": iteration,
                "channels": [1],
                "Q_run": 0.5,
                "quality": {"channel_components": {
                    "1": {"Q_channel": increasing},
                    "2": {"Q_channel": decreasing},
                }},
            }
            for iteration, increasing, decreasing in [(1, 0.2, 0.8), (2, 0.8, 0.2)]
        ],
    }
    history = viewer._observation_table(session)
    channels = viewer._channel_metric_columns(history)["Q_channel"]

    assert set(channels) == {"1", "2"}
    for channel, expected in [("1", [0.2, 0.8]), ("2", [0.8, 0.2])]:
        figure = viewer._plot_channel_trend(
            history, "Q_channel", channels, [channel], "Overlay selected channels"
        )
        assert list(figure.data[0].y) == expected


def test_shared_parameter_history_preserves_expanded_channel_membership():
    history = pd.DataFrame({
        "iteration": [1, 2, 3],
        "channels": ["1", "1", "3"],
        "frequency": [10.0, 20.0, 90.0],
        "ch1_Q_channel": [0.2, 0.4, None],
        "ch2_Q_channel": [0.8, 0.6, None],
        "ch3_Q_channel": [None, None, 0.5],
    })
    frame, metric, columns, shared = viewer._history_channel_series(
        history, "frequency", viewer._channel_metric_columns(history), ["2"]
    )
    assert shared
    assert set(columns) == {"2"}
    figure = viewer._plot_channel_trend(
        frame, metric, columns, ["2"], "Overlay selected channels"
    )
    assert list(figure.data[0].y) == [10.0, 20.0]


def test_q_run_channel_view_uses_individual_scores():
    history = pd.DataFrame({
        "iteration": [1, 2],
        "Q_run": [0.5, 0.5],
        "ch1_Q_channel": [0.2, 0.4],
        "ch2_Q_channel": [0.8, 0.6],
    })
    frame, metric, columns, shared = viewer._history_channel_series(
        history, "Q_run", viewer._channel_metric_columns(history), ["1", "2"]
    )
    assert not shared
    assert metric == "Q_channel"
    assert list(frame[columns["2"]]) == [0.8, 0.6]


def test_history_channel_display_is_visible_for_run_level_metric():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_string('''
import streamlit as st
from bo_session_viewer import _history_channel_display_control
layout = _history_channel_display_control(
    "history_display", global_metric=True, has_channels=True,
)
st.text(layout)
''').run()

    assert not app.exception
    assert app.radio[0].label == "Channel display"
    assert not app.radio[0].disabled
    assert app.radio[0].value == "Run-level series"
    app.radio[0].set_value("Separate plots").run()
    assert not app.exception
    assert app.text[0].value == "Separate plots"


def _mixed_direction_session():
    observations = [
        {
            "iteration": iteration,
            "group_id": 1,
            "channels": [1],
            "method_id": f"{direction}-{iteration}",
            "optimization_direction": direction,
            "Q_run": value,
            "quality": {"channel_components": {"1": {"Q_channel": value}}},
        }
        for iteration, direction, value in [
            (1, "minimize", 0.8), (1, "maximize", 0.2),
            (2, "minimize", 0.6), (2, "maximize", 0.4),
        ]
    ]
    return {
        "observations": observations,
        # Deliberately reverse direction order relative to observations.
        "history": pd.DataFrame([
            {key: observation[key] for key in (
                "iteration", "group_id", "method_id", "optimization_direction", "Q_run",
            )}
            for observation in reversed(observations)
        ]),
    }


def test_history_merge_preserves_both_directions_at_same_iteration():
    session = _mixed_direction_session()
    history = viewer._observation_table(session)
    assert len(history) == 4
    for observation in session["observations"]:
        row = history.loc[history["method_id"] == observation["method_id"]].iloc[0]
        assert row["ch1_Q_channel"] == observation["Q_run"]
        assert row["optimization_direction"] == observation["optimization_direction"]


def test_history_merge_keeps_direction_missing_from_csv():
    session = _mixed_direction_session()
    session["history"] = session["history"].loc[
        session["history"]["optimization_direction"] == "minimize"
    ].reset_index(drop=True)
    history = viewer._observation_table(session)
    assert len(history) == 4
    assert history["optimization_direction"].value_counts().to_dict() == {
        "minimize": 2, "maximize": 2,
    }


def test_channel_plot_keeps_direction_traces_and_moving_averages_separate():
    history = viewer._observation_table(_mixed_direction_session())
    figure = viewer._plot_channel_trend(
        history, "Q_channel", viewer._channel_metric_columns(history)["Q_channel"],
        ["1"], "Separate plots", moving_average_window=2,
    )
    raw = [trace for trace in figure.data if not trace.meta]
    assert len(raw) == 2
    traces = {trace.name: trace for trace in raw}
    assert list(traces["Minimize"].y) == [0.8, 0.6]
    assert list(traces["Maximize"].y) == [0.2, 0.4]
    assert traces["Minimize"].line.dash == "dash"
    assert traces["Maximize"].line.dash == "solid"
    assert len(figure.data) == 4


def test_averaging_channels_and_groups_does_not_average_opposite_directions():
    history = viewer._observation_table(_mixed_direction_session())
    second_group = history.assign(group_id=2, group_name="Group 2")
    history = pd.concat([history, second_group], ignore_index=True)
    history["ch2_Q_channel"] = history["ch1_Q_channel"]
    for group_layout, metadata in [
        ("Average groups together", None),
        ("Plot groups overlaid", {1: 0.1, 2: 0.1}),
    ]:
        figure = viewer._plot_channel_trend(
            history, "Q_channel", viewer._channel_metric_columns(history)["Q_channel"],
            ["1", "2"], "Average selected channels", group_layout,
            group_average_values=metadata,
        )
        traces = [trace for trace in figure.data if trace.mode == "lines+markers"]
        assert len(traces) == 2
        assert sorted([list(trace.y) for trace in traces]) == [[0.2, 0.4], [0.8, 0.6]]


def test_direction_pairs_can_be_selected_and_plotted_as_separate_channels():
    history = viewer._observation_table(_mixed_direction_session())
    history["ch2_Q_channel"] = history["ch1_Q_channel"] * 2
    frame, columns = viewer._history_channels_by_direction(
        history, viewer._channel_metric_columns(history)["Q_channel"],
    )
    assert set(columns) == {
        "1 · Minimize", "1 · Maximize", "2 · Minimize", "2 · Maximize",
    }
    selected = viewer._plot_channel_trend(
        frame, "Q_channel", columns, ["1 · Minimize", "2 · Maximize"], "Separate plots",
    )
    assert len(selected.data) == 2
    assert list(selected.data[0].y) == [0.8, 0.6]
    assert list(selected.data[1].y) == [0.4, 0.8]
    assert selected.data[0].xaxis != selected.data[1].xaxis
    overlay = viewer._plot_channel_trend(
        frame, "Q_channel", columns, list(columns), "Overlay selected channels",
    )
    assert {trace.name for trace in overlay.data} == {f"Ch {channel}" for channel in columns}


def test_split_direction_channels_retains_records_without_direction():
    history = pd.DataFrame({
        "optimization_direction": ["minimize", None],
        "Q_ch1": [0.2, 0.4],
    })
    frame, columns = viewer._history_channels_by_direction(history, {"1": "Q_ch1"})
    assert frame[columns["1 · Unspecified direction"]].dropna().tolist() == [0.4]
    legacy = history.drop(columns="optimization_direction")
    _, columns = viewer._history_channels_by_direction(legacy, {"1": "Q_ch1"})
    assert columns == {"1": "Q_ch1"}
