import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.analysis import is_peak_height_below_cutoff_error
import bo_session_viewer as viewer
from bo_session_viewer import (
    REAL_DATA_METRICS,
    _add_real_heatmap_history_columns,
    _apply_real_heatmap_metadata_filters,
    _compact_simulation_observations_from_history,
    _iteration_trace_q_score_lines,
    _observation_with_plotted_peak_replicates,
    _offset_hyperparameter_response_channels,
    _overlay_trace_scoring_blocks,
    _pair_buffer_target_traces,
    _paired_trend_differences,
    _paired_pairwise_repeat_snr,
    _plot_real_channel_iteration_heatmap,
    _plot_paired_phase_trend,
    _plot_iteration_trace_overlay,
    _plot_traces,
    _real_metric_phase_independent,
    _real_heatmap_full_export_height,
    _real_metric_points,
    _recorded_file_index,
    _resolve_recorded_path,
    _session_candidate_count,
    _trace_entries_by_iteration,
    _trace_paths,
)


def test_recorded_path_fallback_skips_sibling_bo_sessions(tmp_path):
    selected = tmp_path / "experiment" / "bo_sessions" / "selected"
    sibling = tmp_path / "experiment" / "bo_sessions" / "large_simulation"
    archive = tmp_path / "experiment" / "archived_measurements"
    selected.mkdir(parents=True)
    sibling.mkdir(parents=True)
    archive.mkdir(parents=True)
    filename = "swv_ch1_relocated.csv"
    (sibling / filename).write_text("wrong sibling", encoding="utf-8")
    expected = archive / filename
    expected.write_text("archived trace", encoding="utf-8")
    _recorded_file_index.cache_clear()

    resolved = _resolve_recorded_path(
        selected,
        f"/old/computer/path/{filename}",
    )

    assert resolved == expected


def test_channel_iteration_heatmap_keeps_repeats_with_shared_run_labels():
    rows = []
    group_id = 0
    for channel in ("1", "2"):
        for _repeat in range(3):
            group_id += 1
            for iteration in (1, 2):
                rows.append({
                    "channel": channel,
                    "iteration": iteration,
                    "value": float(group_id + iteration),
                    "group_id": group_id,
                    "group_name": "Fixed hyperparameters",
                    "run_label": "Fixed hyperparameters",
                })
    points = pd.DataFrame(rows)

    figure = _plot_real_channel_iteration_heatmap(
        points,
        metric_label="Q",
        phase="measurement",
    )

    heatmap = figure.data[-1]
    assert len(heatmap.y) == 6
    assert len(heatmap.x) == 2
    assert sum(
        int(cell[0])
        for row in heatmap.customdata
        for cell in row
    ) == 12


def test_channel_iteration_heatmap_preserves_legacy_unidentified_repeats():
    points = pd.DataFrame({
        "channel": ["1", "1", "1"],
        "iteration": [1, 1, 1],
        "value": [1.0, 2.0, 3.0],
    })

    figure = _plot_real_channel_iteration_heatmap(
        points,
        metric_label="Q",
        phase="measurement",
    )

    heatmap = figure.data[-1]
    assert len(heatmap.y) == 3
    assert sum(int(row[0][0]) for row in heatmap.customdata) == 3


def test_channel_iteration_heatmap_can_group_runs_by_channel():
    rows = []
    for channel, group_id, exploration in (
        ("1", 1, 0.1),
        ("1", 2, 0.2),
        ("2", 3, 0.1),
        ("2", 4, 0.2),
    ):
        for iteration in (1, 2):
            rows.append({
                "channel": channel,
                "iteration": iteration,
                "value": float(group_id + iteration),
                "group_id": group_id,
                "run_label": f"explore={exploration}",
                "exploration": exploration,
            })

    figure = _plot_real_channel_iteration_heatmap(
        pd.DataFrame(rows),
        metric_label="Q",
        phase="measurement",
        group_runs_by_channel=True,
    )

    heatmap = figure.data[-1]
    row_channels = [
        str(row_label).split("|", 1)[0].strip()
        for row_label in heatmap.y
    ]
    assert row_channels == ["1", "1", "2", "2"]
    assert any(
        shape.y0 == pytest.approx(1.5)
        and shape.line.width == pytest.approx(3.2)
        for shape in figure.layout.shapes
    )
    separator_positions = [float(shape.y0) for shape in figure.layout.shapes]
    assert len(separator_positions) == len(set(separator_positions))


def test_full_heatmap_export_height_preserves_each_row_without_display_cap():
    assert _real_heatmap_full_export_height(
        5_000,
        2,
        1_000,
    ) == 5_240
    assert _real_heatmap_full_export_height(
        5_000,
        4,
        1_000,
    ) == 10_240


def test_channel_iteration_heatmap_limits_only_interactive_display_rows():
    points = pd.DataFrame({
        "channel": ["1"] * 24,
        "iteration": [1, 2] * 12,
        "value": [float(value) for value in range(24)],
        "run_index": [run for run in range(12) for _ in range(2)],
    })

    preview = _plot_real_channel_iteration_heatmap(
        points,
        metric_label="Q",
        phase="measurement",
        max_display_rows=5,
    )
    complete = _plot_real_channel_iteration_heatmap(
        points,
        metric_label="Q",
        phase="measurement",
        max_display_rows=None,
    )

    assert len(preview.data[-1].y) == 5
    assert preview.layout.meta["bo_heatmap_displayed_rows"] == 5
    assert preview.layout.meta["bo_heatmap_total_rows"] == 12
    assert len(complete.data[-1].y) == 12


def test_hyperparameter_channel_offset_zeros_each_channel_surface_minimum():
    response = pd.DataFrame({
        "ground_truth_channel": ["1", "1", "1", "1", "2", "2", "2", "2"],
        "x": [0, 0, 1, 1, 0, 0, 1, 1],
        "y": [0] * 8,
        "metric_value": [0.1, 0.3, 0.7, 0.9, 10.4, 10.6, 11.4, 11.6],
    })

    offset = _offset_hyperparameter_response_channels(
        response,
        group_axes=["x", "y"],
        aggregate="Mean",
    )

    channel_surfaces = (
        offset.groupby(["ground_truth_channel", "x", "y"])["metric_value"]
        .mean()
    )
    assert channel_surfaces.groupby(level=0).min().to_dict() == pytest.approx({
        "1": 0.0,
        "2": 0.0,
    })
    assert offset["channel_q_baseline"].drop_duplicates().tolist() == pytest.approx([
        0.2,
        10.5,
    ])


def test_heatmap_history_join_does_not_mix_channels_with_reused_group_ids():
    points = {
        "measurement": pd.DataFrame({
            "channel": ["1", "2"],
            "group_id": [1, 1],
            "iteration": [1, 1],
            "value": [1.0, 2.0],
        })
    }
    history = pd.DataFrame({
        "ground_truth_channel": ["1", "2"],
        "group_id": [1, 1],
        "iteration": [1, 1],
        "Q_run": [10.0, 20.0],
    })

    enriched = _add_real_heatmap_history_columns(points, [], history)["measurement"]

    assert dict(zip(enriched["channel"], enriched["Q_run"])) == {
        "1": pytest.approx(10.0),
        "2": pytest.approx(20.0),
    }


def test_channel_iteration_heatmap_metadata_filters_apply_each_level():
    points = pd.DataFrame({
        "channel": ["1", "1", "2", "2"],
        "exploration": [0.7, 0.7, 0.2, 0.7],
        "gp_falloff_value": [0.2, 0.3, 0.2, 0.2],
        "initial_random_points": [0, 5, 0, 0],
        "value": [1.0, 2.0, 3.0, 4.0],
    })

    filtered = _apply_real_heatmap_metadata_filters(
        points,
        {
            "exploration": (0.7,),
            "gp_falloff": (0.2,),
            "initial": (0.0,),
        },
    )

    assert filtered["value"].tolist() == [1.0, 4.0]


def test_empty_heatmap_metadata_selection_excludes_all_rows():
    points = pd.DataFrame({
        "initial_random_points": [0, 5, 10],
        "value": [1.0, 2.0, 3.0],
    })

    filtered = _apply_real_heatmap_metadata_filters(
        points,
        {"initial": ()},
    )

    assert filtered.empty


def test_compact_simulation_metadata_reaches_real_heatmap_points():
    history = pd.DataFrame({
        "group_id": [1],
        "iteration": [1],
        "Q_run": [4.2],
        "objective": ["compact_simulation"],
        "ground_truth_channel": ["3"],
        "run_label": ["fixed settings"],
        "exploration": [0.7],
        "initial_random_points": [0],
        "gp_falloff_parameter": ["all"],
        "gp_falloff_value": [0.2],
        "frequency": [200.0],
    })
    observations = _compact_simulation_observations_from_history(history)

    points = _real_metric_points(
        observations,
        "Classic Q",
        "measurement",
        ["3"],
        average_channels=False,
    )

    assert points.loc[0, "exploration"] == pytest.approx(0.7)
    assert points.loc[0, "initial_random_points"] == 0
    assert points.loc[0, "gp_falloff_value"] == pytest.approx(0.2)
    assert points.loc[0, "gp_falloff_parameter"] == "all"


def test_visible_swv_lines_backfill_missing_phase_replicate_peaks():
    observation = {
        "buffer_channel_metrics": {"1": {}},
        "target_channel_metrics": {"1": {}},
    }
    trace_items = [
        *[{"phase": "buffer", "channel": "1"} for _ in range(3)],
        *[{"phase": "target", "channel": "1"} for _ in range(3)],
    ]
    figure, axis = plt.subplots()
    for peak in (1.0, 1.1, 1.2, 2.0, 2.1, 2.2):
        axis.plot([0.0, 1.0, 2.0], [0.0, peak, 0.0])

    enriched = _observation_with_plotted_peak_replicates(
        observation,
        trace_items,
        figure,
    )
    plt.close(figure)

    assert len(enriched["buffer_channel_metrics"]["1"]["peak_currents_uA"]) == 3
    assert len(enriched["target_channel_metrics"]["1"]["peak_currents_uA"]) == 3
    assert enriched["buffer_channel_metrics"]["1"]["peak_currents_source"] == (
        "reconstructed from displayed SWV traces"
    )


def test_separate_iteration_score_panel_lists_q_inputs_and_replicate_peaks():
    observation = {
        "objective": "paired_response",
        "Q_run": 4.5,
        "buffer_channel_metrics": {
            "1": {
                "peak_currents_uA": [1.0, 2.0, 3.0],
                "mean_peak_current_uA": 2.0,
                "std_peak_current_uA": 1.0,
                "mean_background_rms_uA": 0.2,
                "peak_prominence": 10.0,
                "repeat_scan_snr": 2.0,
            }
        },
        "target_channel_metrics": {
            "1": {
                "peak_currents_uA": [4.0, 6.0, 8.0],
                "mean_peak_current_uA": 6.0,
                "std_peak_current_uA": 2.0,
                "mean_background_rms_uA": 0.3,
                "peak_prominence": 20.0,
                "repeat_scan_snr": 3.0,
            }
        },
        "quality": {
            "channel_components": {
                "1": {
                    "paired_Q_channel": 4.5,
                    "buffer_classic_Q": 1.2,
                    "target_classic_Q": 2.4,
                    "delta_peak_height_uA": 4.0,
                    "combined_channel_noise": 0.5,
                    "peak_prominence": 8.0,
                    "repeat_scan_snr": -2.0,
                    "pairwise_peak_difference_std_uA": 2.0,
                    "repeat_scan_snr_contribution": -3.0,
                }
            }
        },
    }
    config = {
        "scoring": {
            "channel_weights": {
                "peak_prominence": 0.0,
                "repeat_scan_snr": 0.0,
                "peak_height": 0.0,
                "peak_shape": 0.0,
                "baseline": 0.0,
                "replicate_consistency": 0.0,
                "success": 1.0,
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

    lines = _iteration_trace_q_score_lines(observation, ["1"], config)
    text = "\n".join(lines)

    assert "Only nonzero-weight terms" in text
    assert "Buffer peak replicates µA: [1, 2, 3]" in text
    assert "Target peak replicates µA: [4, 6, 8]" in text
    assert "[3, 5, 7, 2, 4, 6, 1, 3, 5]" in text
    assert "Mean pairwise peak difference=4 µA" in text
    assert "sample pairwise STD=" in text
    assert "Regularized pairwise STD" in text
    assert "Pairwise repeat SNR = mean pairwise difference / regularized STD" in text
    assert "weight=1.5; contribution=" in text
    assert "Paired prominence" not in text
    assert "Classic-Q paired term" not in text
    assert "Derived Paired Q run=" in text
    assert "Run channels included in mean: Ch 1=" in text


def test_separate_iteration_score_panel_uses_saved_pairwise_differences_without_replicates():
    observation = {
        "objective": "paired_response",
        "buffer_channel_metrics": {"1": {
            "mean_peak_current_uA": 1.0,
            "std_peak_current_uA": 0.1,
            "ok_scan_count": 3,
            "success_score": 1.0,
        }},
        "target_channel_metrics": {"1": {
            "mean_peak_current_uA": 3.0,
            "std_peak_current_uA": 0.2,
            "ok_scan_count": 3,
            "success_score": 1.0,
        }},
        "quality": {"channel_components": {"1": {
            "pairwise_peak_differences_uA": [1.7, 2.0, 2.3],
        }}},
    }
    config = {"scoring": {
        "channel_weights": {"success": 1.0, "peak_prominence": 0.0},
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
    }}

    text = "\n".join(_iteration_trace_q_score_lines(observation, ["1"], config))

    assert "All pairwise target−buffer peak differences µA: [1.7, 2, 2.3]" in text
    assert "Mean pairwise peak difference=2 µA" in text


def test_paired_trend_difference_is_target_minus_buffer_and_skips_missing_pairs():
    series = {
        "iteration": [1, 2, 3],
        "buffer": [2.0, None, 8.0],
        "target": [5.5, 7.0, 3.0],
    }

    iterations, differences = _paired_trend_differences(series)

    assert iterations == [1, 3]
    assert differences == pytest.approx([3.5, -5.0])


def test_pairwise_repeat_snr_uses_all_buffer_target_combinations():
    result = _paired_pairwise_repeat_snr(
        {"peak_currents_uA": [1.0, 2.0, 3.0]},
        {"peak_currents_uA": [4.0, 6.0, 8.0]},
    )

    pairwise_differences = [
        target_value - buffer_value
        for buffer_value in (1.0, 2.0, 3.0)
        for target_value in (4.0, 6.0, 8.0)
    ]
    expected_std = pd.Series(pairwise_differences).std(ddof=1)
    assert result is not None
    assert result[0] == pytest.approx((6.0 - 2.0) / expected_std)
    assert result[1] == pytest.approx(expected_std)


def test_pairwise_repeat_snr_can_be_reconstructed_from_phase_summaries():
    result = _paired_pairwise_repeat_snr(
        {
            "mean_peak_current_uA": 2.0,
            "std_peak_current_uA": 1.0,
            "ok_scan_count": 3,
        },
        {
            "mean_peak_current_uA": 6.0,
            "std_peak_current_uA": 2.0,
            "ok_scan_count": 3,
        },
    )

    expected_std = (30.0 / 8.0) ** 0.5
    assert result is not None
    assert result[0] == pytest.approx(4.0 / expected_std)
    assert result[1] == pytest.approx(expected_std)


def test_paired_difference_checkbox_adds_trace_per_channel_to_overlay():
    figure = _plot_paired_phase_trend(
        {
            "1": {"iteration": [1, 2], "buffer": [1.0, 3.0], "target": [4.0, 2.0]},
            "2": {"iteration": [1, 2], "buffer": [5.0, 2.0], "target": [6.0, 8.0]},
        },
        "Peak height (µA)",
        ["1", "2"],
        "Overlay selected channels",
        True,
    )

    assert [trace.name for trace in figure.data] == [
        "Ch 1 buffer",
        "Ch 1 target",
        "Ch 1 target − buffer",
        "Ch 2 buffer",
        "Ch 2 target",
        "Ch 2 target − buffer",
    ]
    assert list(figure.data[2].y) == pytest.approx([3.0, -1.0])
    assert list(figure.data[5].y) == pytest.approx([1.0, 6.0])
    assert figure.layout.yaxis.title.text == "Peak height (µA)"


def test_paired_difference_checkbox_works_with_average_display():
    figure = _plot_paired_phase_trend(
        {
            "1": {"iteration": [1, 2], "buffer": [1.0, 3.0], "target": [4.0, 2.0]},
            "2": {"iteration": [1, 2], "buffer": [5.0, 2.0], "target": [6.0, 8.0]},
        },
        "Peak height (µA)",
        ["1", "2"],
        "Average selected channels",
        True,
    )

    assert [trace.name for trace in figure.data] == [
        "Buffer",
        "Target",
        "Target − buffer",
    ]
    assert list(figure.data[0].y) == pytest.approx([3.0, 2.5])
    assert list(figure.data[1].y) == pytest.approx([5.0, 5.0])
    assert list(figure.data[2].y) == pytest.approx([2.0, 2.5])


def test_buffer_target_trend_supports_shared_matplotlib_plot_settings():
    figure = _plot_paired_phase_trend(
        {
            "1": {
                "iteration": [1, 2],
                "buffer": [1.0, 3.0],
                "target": [4.0, 2.0],
            },
        },
        "Peak height (µA)",
        ["1"],
        "Overlay selected channels",
    )
    settings = {
        "width": 1000,
        "height": 600,
        "text_size": 18.0,
        "tick_size": 14.0,
        "line_scale": 2.0,
        "margin": 60,
        "perimeter_width": 1.5,
        "perimeter_color": "#123456",
        "show_legend": True,
        "show_grid": False,
        "legend_side": "Left",
        "legend_text_size": 16.0,
        "line_color": "",
        "moving_average_width": 3.0,
        "moving_average_color": "",
        "override_legend_text": True,
        "legend_title": "Phase",
        "legend_labels": "Baseline\nAnalyte",
        "override_text": True,
        "override_title_size": True,
        "title_size": 24.0,
        "title": "Custom paired trend",
        "xlabel": "Iteration",
        "ylabel": "Response",
    }

    viewer._apply_individual_plotly_style(figure, settings)
    matplotlib_figure = viewer._history_plotly_to_matplotlib(figure, settings)
    axis = matplotlib_figure.axes[0]
    legend = axis.get_legend()

    assert tuple(matplotlib_figure.get_size_inches()) == pytest.approx((10, 6))
    assert matplotlib_figure._suptitle.get_text() == "Custom paired trend"
    assert matplotlib_figure._suptitle.get_fontsize() == pytest.approx(24)
    assert axis.get_xlabel() == "Iteration"
    assert axis.get_ylabel() == "Response"
    assert not any(line.get_visible() for line in axis.get_xgridlines())
    assert axis.spines["left"].get_linewidth() == pytest.approx(1.5)
    assert legend._loc == 2
    assert legend.get_title().get_text() == "Phase"
    assert [text.get_text() for text in legend.get_texts()] == [
        "Baseline",
        "Analyte",
    ]
    assert all(text.get_fontsize() == pytest.approx(16) for text in legend.get_texts())
    assert all(line.get_linewidth() == pytest.approx(4) for line in axis.lines)
    plt.close(matplotlib_figure)


def test_peak_height_cutoff_is_classified_as_expected_rejection():
    assert is_peak_height_below_cutoff_error(
        ValueError("Peak height 0.0002 uA below cutoff 0.001 uA")
    )
    assert not is_peak_height_below_cutoff_error(
        ValueError("Trace has no points inside the selected voltage crop.")
    )


def _write_analysis_record(root: Path, phase: str) -> str:
    rows = []
    for replicate in range(1, 4):
        trace_path = root / f"{phase}_ch1_replicate_{replicate}.csv"
        trace_path.write_text("voltage,current\n0,0\n", encoding="utf-8")
        rows.append({"file_path": str(trace_path), "channel": 1})
    results_path = root / f"{phase}_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    record_path = root / f"{phase}_analysis_record.json"
    record_path.write_text(
        json.dumps({"results_csv": str(results_path)}),
        encoding="utf-8",
    )
    return str(record_path)


def test_trace_paths_preserves_all_same_channel_phase_replicates(tmp_path):
    observation = {
        "objective": "paired_response",
        "buffer_analysis_record": _write_analysis_record(tmp_path, "buffer"),
        "target_analysis_record": _write_analysis_record(tmp_path, "target"),
    }

    traces = _trace_paths({"root": tmp_path}, observation)

    assert len(traces) == 6
    assert [trace["phase"] for trace in traces].count("buffer") == 3
    assert [trace["phase"] for trace in traces].count("target") == 3
    assert {trace["channel"] for trace in traces} == {"1"}


def test_trace_paths_preserve_per_channel_settings_and_saved_bo_directions(tmp_path):
    rows = []
    for channel, frequency, amplitude, step in (
        (1, 36.0, 0.11, 0.005),
        (2, 322.0, 0.06, 0.004),
    ):
        trace_path = tmp_path / f"buffer_ch{channel}.csv"
        trace_path.write_text("voltage,current\n0,0\n", encoding="utf-8")
        rows.append({
            "file_path": str(trace_path),
            "channel": channel,
            "frequency_hz": frequency,
            "swv_amplitude_V": amplitude,
            "swv_step_size_V": step,
        })
    results_path = tmp_path / "buffer_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    record_path = tmp_path / "buffer_analysis_record.json"
    record_path.write_text(
        json.dumps({"results_csv": str(results_path)}),
        encoding="utf-8",
    )
    current = {
        "group_id": 1,
        "iteration": 5,
        "objective": "paired_response",
        "buffer_analysis_record": str(record_path),
        "params": {
            "frequency": 36.0,
            "amplitude": 0.11,
            "step_potential": 0.005,
        },
    }
    maximize_observation = {
        "group_id": 2,
        "iteration": 5,
        "params": {
            "frequency": 322.0,
            "amplitude": 0.06,
            "step_potential": 0.004,
        },
    }
    session = {
        "root": tmp_path,
        "config": {
            "acquisition": {"optimization_direction": "maximize"},
            "channel_groups": [
                {"id": 1, "optimization_direction": "minimize"},
                {"id": 2, "optimization_direction": "maximize"},
            ],
        },
        "state": {},
        "observations": [current, maximize_observation],
    }

    traces = _trace_paths(session, current)

    assert [trace["frequency_hz"] for trace in traces] == [36.0, 322.0]
    assert [trace["optimization_direction"] for trace in traces] == [
        "minimize",
        "maximize",
    ]


def test_trace_direction_prefers_own_saved_observation_over_global_default():
    params = {
        "frequency": 86.0,
        "amplitude": 0.1,
        "step_potential": 0.005,
    }
    minimize_observation = {
        "group_id": 1,
        "iteration": 41,
        "params": params,
        "quality": {"optimization_direction": "minimize"},
    }
    maximize_observation = {
        "group_id": 1,
        "iteration": 41,
        "params": params,
        "quality": {"optimization_direction": "maximize"},
    }
    session = {
        "config": {"acquisition": {"optimization_direction": "maximize"}},
        "state": {},
        "observations": [minimize_observation, maximize_observation],
    }
    trace = {
        "channel": "2",
        "frequency_hz": 86.0,
        "swv_amplitude_V": 0.1,
        "swv_step_size_V": 0.005,
    }

    assert viewer._saved_trace_optimization_direction(
        session,
        minimize_observation,
        trace,
    ) == "minimize"
    assert viewer._saved_trace_optimization_direction(
        session,
        maximize_observation,
        trace,
    ) == "maximize"


def test_swv_trace_title_and_legend_show_each_method_and_direction(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_swv_trace_arrays",
        lambda *_args, **_kwargs: (
            pd.Series([-0.5, -0.4, -0.3]).to_numpy(),
            pd.Series([0.0, 1.0, 0.0]).to_numpy(),
            1,
            0,
            2,
        ),
    )
    observation = {"iteration": 5, "params": {}}
    traces = [
        {
            "phase": phase,
            "channel": channel,
            "path": Path(f"{phase}_ch{channel}.csv"),
            "frequency_hz": frequency,
            "swv_amplitude_V": amplitude,
            "swv_step_size_V": step,
            "optimization_direction": direction,
        }
        for phase in ("buffer", "target")
        for channel, frequency, amplitude, step, direction in (
            ("1", 36.0, 0.11, 0.005, "minimize"),
            ("2", 322.0, 0.06, 0.004, "maximize"),
        )
    ]

    figure, errors = _plot_traces(
        {"root": Path(".")},
        observation,
        False,
        ["1", "2"],
        {},
        "session settings",
        True,
        traces,
    )

    assert not errors
    title = figure.axes[0].get_title()
    assert "Ch 1 · Minimize: 36 Hz, step 0.005 V, amplitude 0.11 V" in title
    assert "Ch 2 · Maximize: 322 Hz, step 0.004 V, amplitude 0.06 V" in title
    legend_labels = [
        text.get_text() for text in figure.axes[0].get_legend().get_texts()
    ]
    assert legend_labels == [
        "Buffer Ch 1 · Minimize",
        "Buffer Ch 2 · Maximize",
        "Target Ch 1 · Minimize",
        "Target Ch 2 · Maximize",
    ]
    assert [line.get_linestyle() for line in figure.axes[0].lines] == [
        "--",
        "-",
        "--",
        "-",
    ]
    plt.close(figure)


def test_pair_buffer_target_traces_preserves_every_replicate():
    traces = [
        {"phase": phase, "channel": "1", "path": Path(f"{phase}_{index}.csv")}
        for phase in ("buffer", "target")
        for index in range(3)
    ]

    pairs = _pair_buffer_target_traces(traces)

    assert len(pairs) == 3
    assert all(channel == "1" for channel, _pair in pairs)
    assert all(
        [trace["phase"] for trace in pair] == ["buffer", "target"]
        for _channel, pair in pairs
    )


def test_trace_entries_group_all_same_iteration_swvs_together():
    iteration_two = {"group_id": 1, "iteration": 2}
    iteration_one = {"group_id": 1, "iteration": 1}
    entries = [
        (iteration_two, {"path": Path("target_2.csv"), "channel": "1"}),
        (iteration_one, {"path": Path("buffer_1a.csv"), "channel": "1"}),
        (iteration_one, {"path": Path("buffer_1b.csv"), "channel": "1"}),
        (iteration_one, {"path": Path("target_1.csv"), "channel": "1"}),
    ]

    grouped = _trace_entries_by_iteration(entries)

    assert [(group_id, iteration) for group_id, iteration, _obs, _traces in grouped] == [
        (1, 1),
        (1, 2),
    ]
    assert [trace["path"].name for trace in grouped[0][3]] == [
        "buffer_1a.csv",
        "buffer_1b.csv",
        "target_1.csv",
    ]


def test_overlay_score_details_group_visible_channels_by_iteration(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_observation_with_trace_peak_replicates",
        lambda observation, _traces, _analysis: dict(observation),
    )
    monkeypatch.setattr(
        viewer,
        "_iteration_trace_q_score_lines",
        lambda observation, channels, _config: [
            f"score={observation['Q_run']}; channels={','.join(channels)}"
        ],
    )
    first = {"group_id": 1, "iteration": 1, "Q_run": 1.25}
    second = {"group_id": 1, "iteration": 2, "Q_run": 2.5}
    entries = [
        (first, {"phase": "buffer", "channel": "1"}),
        (first, {"phase": "target", "channel": "1"}),
        (first, {"phase": "target", "channel": "2"}),
        (second, {"phase": "buffer", "channel": "1"}),
    ]

    blocks = _overlay_trace_scoring_blocks(entries, ["1"], {}, {})

    assert blocks == [
        ("Iteration 1", "score=1.25; channels=1"),
        ("Iteration 2", "score=2.5; channels=1"),
    ]


def test_paired_swv_trace_plot_uses_distinct_phase_colors(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_swv_trace_arrays",
        lambda *_args, **_kwargs: (
            pd.Series([-0.5, -0.4, -0.3]).to_numpy(),
            pd.Series([0.0, 1.0, 0.0]).to_numpy(),
            1,
            0,
            2,
        ),
    )
    observation = {
        "iteration": 1,
        "Q_run": 2.0,
        "params": {
            "frequency": 100.0,
            "amplitude": 0.05,
            "step_potential": 0.002,
        },
    }
    traces = [
        {"phase": "buffer", "channel": "1", "path": Path("buffer.csv")},
        {"phase": "target", "channel": "1", "path": Path("target.csv")},
    ]

    figure, errors = _plot_traces(
        {"root": Path(".")},
        observation,
        False,
        ["1"],
        {},
        "session settings",
        True,
        traces,
    )

    assert not errors
    assert [line.get_color() for line in figure.axes[0].lines] == [
        viewer.SWV_PHASE_COLORS["buffer"],
        viewer.SWV_PHASE_COLORS["target"],
    ]
    plt.close(figure)


def test_paired_trend_metrics_include_saved_zero_weight_measurements():
    observations = [{
        "iteration": 1,
        "buffer_channel_metrics": {
            "1": {
                "mean_peak_current_uA": 1.2,
                "repeat_scan_snr": 3.0,
            }
        },
        "target_channel_metrics": {
            "1": {
                "mean_peak_current_uA": 2.4,
                "repeat_scan_snr": 4.0,
            }
        },
    }]

    available = viewer._available_paired_trend_metrics(observations)

    assert "Peak height (µA)" in available
    assert "Repeat-scan SNR" in available
    assert "Peak shape score" not in available


def test_individual_history_plot_style_does_not_use_global_plot_settings():
    figure = go.Figure(
        go.Scatter(
            x=[1, 2],
            y=[3, 4],
            mode="lines",
            name="Original series",
            line={"width": 1},
        )
    )
    figure.update_layout(
        title="Original title",
        xaxis_title="Original X",
        yaxis_title="Original Y",
    )
    settings = {
        "width": 1100,
        "height": 600,
        "text_size": 18.0,
        "tick_size": 14.0,
        "line_scale": 2.0,
        "margin": 70,
        "perimeter_width": 1.5,
        "perimeter_color": "#123456",
        "show_legend": False,
        "show_grid": False,
        "legend_side": "Right",
        "legend_text_size": 17.0,
        "line_color": "crimson",
        "moving_average_width": 5.0,
        "moving_average_color": "navy",
        "override_legend_text": True,
        "legend_title": "Custom legend",
        "legend_labels": "Custom series",
        "override_text": True,
        "override_title_size": True,
        "title_size": 26.0,
        "title": "Custom title",
        "xlabel": "Custom X",
        "ylabel": "Custom Y",
    }

    viewer._apply_individual_plotly_style(figure, settings)

    assert figure.layout.width == 1100
    assert figure.layout.height == 600
    assert figure.layout.title.text == (
        '<span style="font-weight:400">Custom title</span>'
    )
    assert figure.layout.title.font.size == pytest.approx(26)
    assert figure.layout.title.font.weight == "normal"
    assert figure.layout.title.x == pytest.approx(0.5)
    assert figure.layout.title.xanchor == "center"
    assert figure.layout.xaxis.title.text == "Custom X"
    assert figure.layout.yaxis.title.text == "Custom Y"
    assert figure.layout.xaxis.tickfont.size == pytest.approx(14)
    assert figure.layout.font.color == "#000000"
    assert figure.layout.title.font.color == "#000000"
    assert figure.layout.xaxis.title.font.color == "#000000"
    assert figure.layout.xaxis.tickfont.color == "#000000"
    assert figure.layout.xaxis.ticks == "outside"
    assert figure.layout.xaxis.ticklen == 6
    assert figure.layout.yaxis.ticks == "outside"
    assert figure.layout.legend.font.color == "#000000"
    assert figure.layout.legend.x == pytest.approx(0.98)
    assert figure.layout.legend.xanchor == "right"
    assert figure.layout.legend.y == pytest.approx(0.98)
    assert figure.layout.legend.yanchor == "top"
    assert figure.layout.legend.orientation == "v"
    assert figure.layout.xaxis.showgrid is False
    assert figure.layout.showlegend is False
    assert figure.data[0].line.width == pytest.approx(2)
    assert figure.data[0].line.color == "crimson"
    assert figure.layout.legend.title.text == "Custom legend"
    assert figure.data[0].name == "Custom series"

    matplotlib_figure = viewer._history_plotly_to_matplotlib(
        figure,
        settings,
    )
    assert len(matplotlib_figure.axes) == 1
    assert matplotlib_figure._suptitle.get_text() == "Custom title"
    assert matplotlib_figure._suptitle.get_fontweight() == "normal"
    assert matplotlib_figure.axes[0].get_xlabel() == "Custom X"
    assert matplotlib_figure.axes[0].get_ylabel() == "Custom Y"
    assert not any(
        gridline.get_visible()
        for gridline in matplotlib_figure.axes[0].get_xgridlines()
    )
    plt.close(matplotlib_figure)


def test_individual_history_plot_preserves_default_axis_labels():
    figure = go.Figure(go.Scatter(x=[1, 2], y=[3, 4], mode="lines"))
    figure.update_layout(
        title="Trend",
        xaxis_title="BO iteration",
        yaxis_title="Peak height (µA)",
    )
    settings = {
        "width": 900,
        "height": 420,
        "text_size": 28.0,
        "tick_size": 20.0,
        "line_scale": 1.0,
        "margin": 20,
        "perimeter_width": 0.8,
        "perimeter_color": "#222222",
        "show_legend": False,
        "show_grid": True,
        "legend_side": "Right",
        "legend_text_size": 8.0,
        "line_color": "",
        "moving_average_width": 3.0,
        "moving_average_color": "",
        "override_legend_text": False,
        "legend_title": "",
        "legend_labels": "",
        "override_text": False,
        "override_title_size": False,
        "title_size": 12.0,
        "title": "Trend",
        "xlabel": "BO iteration",
        "ylabel": "Peak height (µA)",
    }

    viewer._apply_individual_plotly_style(figure, settings)
    matplotlib_figure = viewer._history_plotly_to_matplotlib(figure, settings)
    axis = matplotlib_figure.axes[0]

    assert axis.get_xlabel() == "BO iteration"
    assert axis.get_ylabel() == "Peak height (µA)"
    assert axis.get_position().x0 > settings["margin"] / settings["width"]
    assert axis.get_position().y0 > settings["margin"] / settings["height"]
    matplotlib_figure.canvas.draw()
    renderer = matplotlib_figure.canvas.get_renderer()
    tight_box = axis.get_tightbbox(renderer)
    canvas_width, canvas_height = renderer.width, renderer.height
    assert tight_box.x0 >= 19.5
    assert tight_box.y0 >= 19.5
    assert tight_box.x1 <= canvas_width - 19.5
    assert tight_box.y1 <= canvas_height - 19.5
    plt.close(matplotlib_figure)


def test_individual_plot_settings_can_be_copied_between_plot_types():
    trend_settings = {
        "width": 1200,
        "height": 640,
        "text_size": 21.0,
        "tick_size": 16.0,
        "line_scale": 2.5,
        "legend_side": "Left",
        "title": "Copied title",
        "xlabel": "Copied x",
        "ylabel": "Copied y",
        "_has_legend_entries": True,
    }
    clipboard = viewer._copyable_individual_plot_settings(trend_settings)
    target_state = {
        "paired_width": 800,
        "paired_height": 400,
        "unrelated": "preserved",
    }

    viewer._apply_copied_individual_plot_settings(
        target_state,
        "paired",
        clipboard,
    )

    assert "_has_legend_entries" not in clipboard
    assert target_state["paired_width"] == 1200
    assert target_state["paired_height"] == 640
    assert target_state["paired_text_size"] == pytest.approx(21)
    assert target_state["paired_tick_size"] == pytest.approx(16)
    assert target_state["paired_line_scale"] == pytest.approx(2.5)
    assert target_state["paired_legend_side"] == "Left"
    assert target_state["paired_title"] == "Copied title"
    assert target_state["paired_xlabel"] == "Copied x"
    assert target_state["paired_ylabel"] == "Copied y"
    assert target_state["unrelated"] == "preserved"


def test_manual_optimum_marker_supports_3d_and_matching_2d_slices():
    reference = {
        "frequency": 250.0,
        "amplitude": 0.075,
        "step_potential": 0.004,
    }
    tensor = go.Figure()

    viewer._add_global_optimum_marker(
        tensor,
        reference,
        "frequency",
        "amplitude",
        "step_potential",
        label="Manual optimum",
    )

    assert len(tensor.data) == 1
    assert isinstance(tensor.data[0], go.Scatter3d)
    assert tensor.data[0].name == "Manual optimum"
    assert list(tensor.data[0].x) == pytest.approx([250.0])
    assert list(tensor.data[0].y) == pytest.approx([0.075])
    assert list(tensor.data[0].z) == pytest.approx([0.004])
    assert tensor.data[0].marker.symbol == "diamond"

    matching_slice = go.Figure()
    viewer._add_global_optimum_marker(
        matching_slice,
        reference,
        "frequency",
        "amplitude",
        label="Manual optimum",
        slice_axis="step_potential",
        slice_value=0.004,
    )
    assert len(matching_slice.data) == 1
    assert isinstance(matching_slice.data[0], go.Scatter)
    assert matching_slice.data[0].name == "Manual optimum"

    other_slice = go.Figure()
    viewer._add_global_optimum_marker(
        other_slice,
        reference,
        "frequency",
        "amplitude",
        label="Manual optimum",
        slice_axis="step_potential",
        slice_value=0.007,
    )
    assert not other_slice.data


def test_parallel_coordinates_marks_optimum_on_every_parameter_axis():
    points = pd.DataFrame({
        "iteration": [1, 2],
        "channel": ["1", "1"],
        "phase": ["target", "target"],
        "frequency": [100.0, 200.0],
        "amplitude": [0.05, 0.05],
        "step_potential": [0.004, 0.004],
        "value": [1.0, 2.0],
    })

    figure = viewer._plot_real_data_parallel_coordinates(
        points,
        metric_label="Q",
        phase="target",
        parameter_columns=["frequency", "amplitude", "step_potential"],
        optimum_reference={
            "frequency": 400.0,
            "amplitude": 0.075,
            "step_potential": 0.004,
        },
        optimum_label="Manual optimum",
        optimum_marker_size=27.0,
        axis_line_width=4.25,
    )

    optimum_trace = next(
        trace for trace in figure.data
        if trace.name == "Manual optimum"
    )
    assert optimum_trace.mode == "lines+markers"
    assert list(optimum_trace.x) == [1, 2, 3]
    assert len(optimum_trace.y) == 3
    assert all(0.0 <= float(value) <= 1.0 for value in optimum_trace.y)
    assert optimum_trace.marker.symbol == "diamond"
    assert optimum_trace.marker.size == pytest.approx(27)
    assert optimum_trace.line.dash == "dash"
    assert optimum_trace.marker.showscale is False
    assert figure.layout.font.color == "#000000"
    assert figure.layout.title.font.color == "#000000"
    parallel_annotations = [
        annotation for annotation in figure.layout.annotations
        if str(annotation.name).startswith("bo_parallel_tick_label")
        or str(annotation.name).startswith("bo_parallel_axis_label")
    ]
    assert parallel_annotations
    assert all(
        annotation.font.color == "#000000"
        for annotation in parallel_annotations
    )

    viewer._apply_plotly_annotation_text_style(
        figure,
        10.0,
        8.0,
        x_label_override="Custom frequency",
        y_label_override="Custom amplitude",
        z_label_override="Custom step size",
        x_label_override_size=19.0,
        y_label_override_size=20.0,
        z_label_override_size=21.0,
    )
    axis_annotations = {
        annotation.name: annotation
        for annotation in figure.layout.annotations
        if str(annotation.name).startswith("bo_parallel_axis_label_")
    }
    assert axis_annotations["bo_parallel_axis_label_x"].text == (
        "Custom frequency"
    )
    assert axis_annotations["bo_parallel_axis_label_x"].font.size == pytest.approx(19)
    assert axis_annotations["bo_parallel_axis_label_y"].text == (
        "Custom amplitude"
    )
    assert axis_annotations["bo_parallel_axis_label_y"].font.size == pytest.approx(20)
    assert axis_annotations["bo_parallel_axis_label_z"].text == (
        "Custom step size"
    )
    assert axis_annotations["bo_parallel_axis_label_z"].font.size == pytest.approx(21)

    viewer._apply_plotly_colorbar_height(figure)

    assert optimum_trace.marker.colorbar.to_plotly_json() == {}
    metric_colorbar_trace = next(
        trace for trace in figure.data
        if getattr(trace.marker, "showscale", None) is True
    )
    assert metric_colorbar_trace.marker.colorbar.title.text == "Q"
    assert metric_colorbar_trace.marker.colorbar.title.font.color == "#000000"
    assert metric_colorbar_trace.marker.colorbar.tickfont.color == "#000000"
    coordinate_axis_shapes = [
        shape for shape in figure.layout.shapes
        if float(shape.x0) == float(shape.x1)
        and float(shape.y0) == pytest.approx(0.0)
        and float(shape.y1) == pytest.approx(1.0)
    ]
    assert len(coordinate_axis_shapes) == 4
    assert all(
        shape.line.width == pytest.approx(4.25)
        for shape in coordinate_axis_shapes
    )


def test_parallel_coordinates_apply_xyz_tick_label_overrides(monkeypatch):
    tick_overrides = {
        "x": ([100.0, 400.0], ["Low frequency", "Optimal frequency"]),
        "y": ([0.05, 0.075], ["Low amplitude", "Optimal amplitude"]),
        "z": ([0.004], ["Optimal step"]),
    }
    monkeypatch.setattr(
        viewer,
        "_plot_tick_label_update",
        lambda axis_key, **_kwargs: tick_overrides.get(axis_key),
    )
    points = pd.DataFrame({
        "iteration": [1, 2],
        "channel": ["1", "1"],
        "phase": ["target", "target"],
        "frequency": [100.0, 200.0],
        "amplitude": [0.05, 0.06],
        "step_potential": [0.004, 0.005],
        "value": [1.0, 2.0],
    })
    figure = viewer._plot_real_data_parallel_coordinates(
        points,
        metric_label="Q",
        phase="target",
        parameter_columns=["frequency", "amplitude", "step_potential"],
        optimum_reference={
            "frequency": 400.0,
            "amplitude": 0.075,
            "step_potential": 0.004,
        },
    )

    tick_text_by_axis = {
        axis_key: [
            annotation.text
            for annotation in figure.layout.annotations
            if annotation.name == f"bo_parallel_tick_label_{axis_key}"
        ]
        for axis_key in "xyz"
    }
    assert tick_text_by_axis == {
        "x": ["Low frequency", "Optimal frequency"],
        "y": ["Low amplitude", "Optimal amplitude"],
        "z": ["Optimal step"],
    }


def test_manual_optimum_does_not_change_2d_colorbar_layout():
    def landscape(include_optimum: bool) -> go.Figure:
        figure = go.Figure(go.Heatmap(
            x=[100.0, 200.0],
            y=[0.05, 0.10],
            z=[[1.0, 2.0], [3.0, 4.0]],
            colorbar={"title": "Q"},
        ))
        if include_optimum:
            viewer._add_global_optimum_marker(
                figure,
                {"frequency": 150.0, "amplitude": 0.075},
                "frequency",
                "amplitude",
                label="Manual optimum",
            )
        viewer._apply_plotly_2d_slice_aspect(
            figure,
            width=1000,
            height=600,
        )
        return figure

    without_optimum = landscape(False)
    with_optimum = landscape(True)
    optimum_trace = with_optimum.data[-1]

    assert list(with_optimum.layout.xaxis.domain) == pytest.approx(
        list(without_optimum.layout.xaxis.domain)
    )
    assert with_optimum.data[0].colorbar.x == pytest.approx(
        without_optimum.data[0].colorbar.x
    )
    assert optimum_trace.meta["bo_trace_role"] == "optimum_marker"
    assert optimum_trace.marker.showscale is False
    assert optimum_trace.marker.colorbar.to_plotly_json() == {}


def test_real_3d_legend_is_kept_away_from_right_side_colorbars():
    points = pd.DataFrame({
        "iteration": [1, 2],
        "group_id": [1, 1],
        "group_name": ["Group 1", "Group 1"],
        "channel": ["5", "5"],
        "frequency": [100.0, 500.0],
        "amplitude": [0.05, 0.20],
        "step_potential": [0.001, 0.010],
        "value": [1.0, 2.0],
    })
    figure = viewer._plot_real_data_landscape(
        points,
        "Paired Q",
        "measurement",
        "3D tensor",
        "step_potential",
        "amplitude",
        "frequency",
        iteration_path=points,
        show_iteration_path=True,
    )
    viewer._add_global_optimum_marker(
        figure,
        {
            "frequency": 500.0,
            "amplitude": 0.10,
            "step_potential": 0.004,
        },
        "step_potential",
        "amplitude",
        "frequency",
        label="Manual optimum",
    )

    assert figure.layout.legend.x == pytest.approx(0.015)
    assert figure.layout.legend.xanchor == "left"
    assert figure.layout.legend.y == pytest.approx(0.985)
    assert figure.layout.legend.yanchor == "top"
    assert figure.layout.legend.orientation == "v"
    assert figure.layout.legend.bgcolor == "rgba(255,255,255,0.78)"
    optimum_trace = next(
        trace for trace in figure.data
        if trace.name == "Manual optimum"
    )
    assert optimum_trace.showlegend is True


def test_metric_and_iteration_colorbars_use_separate_overrides():
    metric_trace = go.Scatter(
        marker={
            "color": [1.0, 2.0],
            "showscale": True,
            "colorbar": {"title": "Paired Q"},
        },
    )
    iteration_trace = go.Scatter(
        marker={
            "color": [1.0, 2.0],
            "showscale": True,
            "colorbar": {"title": "Iteration"},
        },
        meta={"bo_trace_role": "iteration_colorbar"},
    )

    metric_override = viewer._plotly_colorbar_override_settings(
        metric_trace.marker.colorbar,
        metric_trace,
        metric_text="Score",
        metric_size=18.0,
        iteration_text="Optimization step",
        iteration_size=14.0,
    )
    iteration_override = viewer._plotly_colorbar_override_settings(
        iteration_trace.marker.colorbar,
        iteration_trace,
        metric_text="Score",
        metric_size=18.0,
        iteration_text="Optimization step",
        iteration_size=14.0,
    )

    assert metric_override == ("Score", 18.0)
    assert iteration_override == ("Optimization step", 14.0)


def test_iteration_path_colorbar_is_tagged_for_independent_override():
    figure = go.Figure()
    viewer._add_plotly_iteration_path(
        figure,
        [0.0, 1.0],
        [0.0, 1.0],
        [1, 2],
    )

    colorbar_trace = next(
        trace for trace in figure.data
        if getattr(trace.marker, "showscale", None) is True
    )
    assert viewer._plotly_trace_role(colorbar_trace) == "iteration_colorbar"
    assert colorbar_trace.marker.colorbar.title.text == "Iteration"


def test_individual_history_plot_can_style_moving_average_separately():
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[1, 2],
        y=[2, 4],
        mode="lines+markers",
        line={"color": "orange", "width": 1},
    ))
    viewer._add_moving_average_traces(figure, 2)
    settings = {
        "width": 900,
        "height": 420,
        "text_size": 10.0,
        "tick_size": 10.0,
        "line_scale": 2.0,
        "margin": 50,
        "perimeter_width": 0.8,
        "perimeter_color": "#222222",
        "show_legend": True,
        "show_grid": True,
        "legend_side": "Left",
        "legend_text_size": 8.0,
        "line_color": "",
        "moving_average_width": 6.0,
        "moving_average_color": "purple",
        "override_legend_text": False,
        "legend_title": "",
        "legend_labels": "",
        "override_text": False,
        "override_title_size": False,
        "title_size": 12.0,
        "title": "",
        "xlabel": "",
        "ylabel": "",
    }

    viewer._apply_individual_plotly_style(figure, settings)

    assert figure.data[0].line.width == pytest.approx(2.0)
    assert figure.data[0].line.color == "orange"
    assert figure.data[1].meta["bo_trace_role"] == "moving_average"
    assert figure.data[1].line.width == pytest.approx(6.0)
    assert figure.data[1].line.color == "purple"
    assert figure.layout.legend.x == pytest.approx(0.02)
    assert figure.layout.legend.xanchor == "left"
    matplotlib_figure = viewer._history_plotly_to_matplotlib(
        figure,
        settings,
    )
    legend = matplotlib_figure.axes[0].get_legend()
    assert legend.get_texts()[0].get_fontsize() == pytest.approx(8)
    plt.close(matplotlib_figure)


def test_replicate_swv_traces_use_distinct_phase_shades(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_swv_trace_arrays",
        lambda *_args, **_kwargs: (
            pd.Series([-0.5, -0.4, -0.3]).to_numpy(),
            pd.Series([0.0, 1.0, 0.0]).to_numpy(),
            1,
            0,
            2,
        ),
    )
    observation = {"iteration": 1, "params": {}}
    traces = [
        {
            "phase": phase,
            "channel": "1",
            "path": Path(f"{phase}_{replicate}.csv"),
        }
        for phase in ("buffer", "target")
        for replicate in range(1, 4)
    ]

    figure, errors = _plot_traces(
        {"root": Path(".")},
        observation,
        False,
        ["1"],
        {},
        "session settings",
        True,
        traces,
    )

    assert not errors
    colors = [line.get_color() for line in figure.axes[0].lines]
    buffer_colors, target_colors = colors[:3], colors[3:]
    assert len(set(buffer_colors)) == 3
    assert len(set(target_colors)) == 3
    assert all(to_rgb(color)[2] > to_rgb(color)[0] for color in buffer_colors)
    assert all(to_rgb(color)[0] > to_rgb(color)[2] for color in target_colors)
    legend = figure.axes[0].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        "Buffer 1",
        "Buffer 2",
        "Buffer 3",
        "Target 1",
        "Target 2",
        "Target 3",
    ]
    assert [handle.get_color() for handle in legend.legend_handles] == [
        *buffer_colors,
        *target_colors,
    ]
    plt.close(figure)


def test_filtered_target_replicates_stay_orange(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_swv_trace_arrays",
        lambda *_args, **_kwargs: (
            pd.Series([-0.5, -0.4, -0.3]).to_numpy(),
            pd.Series([0.0, 1.0, 0.0]).to_numpy(),
            1,
            0,
            2,
        ),
    )
    traces = [
        {
            "phase": "target",
            "channel": "1",
            "path": Path(f"target_{replicate}.csv"),
        }
        for replicate in range(1, 4)
    ]

    figure, errors = _plot_traces(
        {"root": Path(".")},
        {"iteration": 1, "params": {}},
        False,
        ["1"],
        {},
        "session settings",
        True,
        traces,
    )

    assert not errors
    colors = [line.get_color() for line in figure.axes[0].lines]
    assert len(set(colors)) == 3
    assert all(to_rgb(color)[0] > to_rgb(color)[2] for color in colors)
    plt.close(figure)


def test_per_plot_text_override_preserves_styled_font_sizes():
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], label="Original entry")
    axis.set_title("Original title", fontsize=25)
    axis.set_xlabel("Original x", fontsize=19)
    axis.set_ylabel("Original y", fontsize=17)
    legend = axis.legend(title="Original legend")
    legend.get_title().set_fontsize(15)
    legend.get_texts()[0].set_fontsize(13)

    viewer._apply_per_plot_matplotlib_text_override(
        axis,
        legend,
        title="Custom title",
        xlabel="Custom x",
        ylabel="Custom y",
        legend_title="Custom legend",
        legend_labels=["Custom entry"],
    )

    assert axis.get_title() == "Custom title"
    assert axis.get_xlabel() == "Custom x"
    assert axis.get_ylabel() == "Custom y"
    assert axis.title.get_fontsize() == pytest.approx(25)
    assert axis.xaxis.label.get_fontsize() == pytest.approx(19)
    assert axis.yaxis.label.get_fontsize() == pytest.approx(17)
    assert legend.get_title().get_text() == "Custom legend"
    assert legend.get_title().get_fontsize() == pytest.approx(15)
    assert legend.get_texts()[0].get_text() == "Custom entry"
    assert legend.get_texts()[0].get_fontsize() == pytest.approx(13)
    plt.close(figure)


def test_per_plot_axis_text_sizes_can_be_overridden_independently():
    figure, axis = plt.subplots()
    axis.set_title("Title", fontsize=12)
    axis.set_xlabel("X", fontsize=10)
    axis.set_ylabel("Y", fontsize=10)

    viewer._apply_per_plot_matplotlib_text_override(
        axis,
        None,
        title="Ignored",
        xlabel="Ignored",
        ylabel="Ignored",
        legend_title="",
        legend_labels=[],
        override_plot_text=False,
        override_legend_text=False,
        title_text_size=24.0,
        xlabel_text_size=18.0,
        ylabel_text_size=20.0,
    )

    assert axis.get_title() == "Title"
    assert axis.get_xlabel() == "X"
    assert axis.get_ylabel() == "Y"
    assert axis.title.get_fontsize() == pytest.approx(24)
    assert axis.xaxis.label.get_fontsize() == pytest.approx(18)
    assert axis.yaxis.label.get_fontsize() == pytest.approx(20)
    plt.close(figure)


def test_per_plot_override_replaces_figure_level_axis_labels_in_place():
    figure, axis = plt.subplots()
    figure_xlabel = figure.text(0.5, 0.02, "Original X", fontsize=9)
    figure_ylabel = figure.text(0.98, 0.5, "Original Y", fontsize=9)

    viewer._apply_per_plot_matplotlib_text_override(
        axis,
        None,
        title="Title",
        xlabel="Custom X",
        ylabel="Custom Y",
        legend_title="",
        legend_labels=[],
        override_legend_text=False,
        xlabel_artist=figure_xlabel,
        ylabel_artist=figure_ylabel,
        xlabel_text_size=18.0,
        ylabel_text_size=20.0,
    )

    assert axis.get_xlabel() == ""
    assert axis.get_ylabel() == ""
    assert figure_xlabel.get_text() == "Custom X"
    assert figure_ylabel.get_text() == "Custom Y"
    assert figure_xlabel.get_fontsize() == pytest.approx(18)
    assert figure_ylabel.get_fontsize() == pytest.approx(20)
    plt.close(figure)


def test_per_plot_legend_overrides_are_independent_from_axis_text():
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], label="Original entry")
    axis.set_title("Original title")
    legend = axis.legend(title="Original legend")

    viewer._apply_per_plot_matplotlib_text_override(
        axis,
        legend,
        title="Ignored title",
        xlabel="Ignored x",
        ylabel="Ignored y",
        legend_title="Custom legend",
        legend_labels=["Custom entry"],
        override_plot_text=False,
        override_legend_text=True,
        legend_text_size=21.0,
    )

    assert axis.get_title() == "Original title"
    assert legend.get_title().get_text() == "Custom legend"
    assert legend.get_texts()[0].get_text() == "Custom entry"
    assert legend.get_title().get_fontsize() == pytest.approx(21)
    assert legend.get_texts()[0].get_fontsize() == pytest.approx(21)
    plt.close(figure)


@pytest.mark.parametrize(
    ("side", "expected_location"),
    [("Left", 2), ("Right", 1)],
)
def test_per_plot_legend_can_be_positioned_on_either_side(
    side,
    expected_location,
):
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1], label="Trace")
    legend = axis.legend()

    viewer._apply_per_plot_matplotlib_text_override(
        axis,
        legend,
        title="",
        xlabel="",
        ylabel="",
        legend_title="",
        legend_labels=[],
        override_plot_text=False,
        override_legend_text=False,
        legend_side=side,
    )

    assert legend._loc == expected_location
    plt.close(figure)


def test_chronological_stack_right_legend_is_inset_from_canvas_edge():
    figure, axis = plt.subplots()
    axis._bo_chronological_swv_stack = True
    axis.plot([0, 1], [0, 1], label="Trace")
    legend = axis.legend()

    viewer._apply_per_plot_matplotlib_text_override(
        axis,
        legend,
        title="",
        xlabel="",
        ylabel="",
        legend_title="",
        legend_labels=[],
        override_plot_text=False,
        override_legend_text=False,
        legend_side="Right",
    )

    anchor = legend.get_bbox_to_anchor()._bbox
    assert anchor.x1 == pytest.approx(0.86)
    assert anchor.y1 == pytest.approx(0.98)
    plt.close(figure)


def test_chronological_stack_axis_uses_the_exact_stack_offset_direction():
    axis_dx, axis_dy = viewer._chronological_swv_stack_axis_vector(
        [2, 5, 8],
        x_step=0.015,
        y_step=0.4,
        fallback_x_span=2.0,
    )

    assert axis_dx == pytest.approx(6 * 0.015)
    assert axis_dy == pytest.approx(6 * 0.4)
    assert axis_dx * 0.4 - axis_dy * 0.015 == pytest.approx(0.0)


def test_chronological_order_label_is_marked_for_plot_customization(monkeypatch):
    loaded = [
        {
            "iteration": iteration,
            "stack_index": iteration - 1,
            "phase": "buffer",
            "channel": "1",
            "voltage": pd.Series([-0.5, -0.4, -0.3]).to_numpy(),
            "current": pd.Series([0.0, 1.0, 0.0]).to_numpy(),
        }
        for iteration in (1, 2)
    ]
    entries = [
        ({"iteration": row["iteration"]}, {"channel": "1"})
        for row in loaded
    ]
    monkeypatch.setattr(
        viewer,
        "_chronological_swv_stack_entries",
        lambda *_args, **_kwargs: (loaded, [], entries),
    )

    figure, errors = viewer._plot_chronological_swv_stack(
        [],
        False,
        ["1"],
        {},
        {},
        "session settings",
        "All iterations",
    )

    assert not errors
    labels = [
        text
        for text in figure.axes[0].texts
        if getattr(text, "_bo_chronological_order_label", False)
    ]
    assert len(labels) == 1
    assert labels[0].get_text() == "chronological order"
    axis_start = labels[0]._bo_axis_start
    axis_end = labels[0]._bo_axis_end
    midpoint_display = figure.axes[0].transData.transform((
        (axis_start[0] + axis_end[0]) / 2,
        (axis_start[1] + axis_end[1]) / 2,
    ))
    label_display = figure.axes[0].transData.transform(labels[0].get_position())
    assert label_display[1] < midpoint_display[1]
    assert float(
        ((label_display - midpoint_display) ** 2).sum() ** 0.5
    ) == pytest.approx(30.0)
    plt.close(figure)


def test_paired_multi_iteration_overlay_uses_phase_colors(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_swv_trace_arrays",
        lambda *_args, **_kwargs: (
            pd.Series([-0.5, -0.4, -0.3]).to_numpy(),
            pd.Series([0.0, 1.0, 0.0]).to_numpy(),
            1,
            0,
            2,
        ),
    )
    observation = {"iteration": 1}
    entries = [
        (
            observation,
            {"phase": "buffer", "channel": "1", "path": Path("buffer.csv")},
        ),
        (
            observation,
            {"phase": "target", "channel": "1", "path": Path("target.csv")},
        ),
    ]

    figure, errors = _plot_iteration_trace_overlay(
        entries,
        False,
        ["1"],
        {},
        "session settings",
        "All iterations",
    )

    assert not errors
    assert [line.get_color() for line in figure.axes[0].lines] == [
        viewer.SWV_PHASE_COLORS["buffer"],
        viewer.SWV_PHASE_COLORS["target"],
    ]
    legend = figure.axes[0].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        "Buffer 1",
        "Target 1",
    ]
    plt.close(figure)


def test_session_candidate_count_prefers_effective_group_metadata():
    session = {
        "state": {
            "candidate_count": 600,
            "candidate_counts_by_group": {"1": 1000},
        },
        "config": {},
    }

    assert _session_candidate_count(session, 1) == 1000
    assert _session_candidate_count(session) == 1000


def test_legacy_session_candidate_count_uses_group_override():
    session = {
        "state": {"candidate_count": 600},
        "config": {
            "channel_groups": [
                {"id": 1, "candidate_pool_size": 1000},
            ]
        },
    }

    assert _session_candidate_count(session, 1) == 1000
    assert _session_candidate_count(session) == 1000


def test_all_group_candidate_count_shows_range_only_when_groups_differ():
    session = {
        "state": {
            "candidate_count": 600,
            "candidate_counts_by_group": {"1": 1000, "2": 800},
        },
        "config": {},
    }

    assert _session_candidate_count(session) == "800–1000"


def test_configured_candidate_target_beats_stale_legacy_global_count():
    session = {
        "state": {"candidate_count": 600},
        "config": {"acquisition": {"candidate_pool_size": 1000}},
    }

    assert _session_candidate_count(session) == 1000


def test_real_data_landscapes_expose_q_inputs_and_paired_derivatives():
    observation = {
        "iteration": 1,
        "group_id": 1,
        "params": {"frequency": 100.0, "amplitude": 0.04},
        "buffer_channel_metrics": {
            "1": {
                "mean_peak_current_uA": 1.0,
                "median_peak_current_uA": 0.9,
                "std_peak_current_uA": 0.1,
                "mean_background_rms_uA": 0.2,
                "median_background_rms_uA": 0.18,
                "std_background_rms_uA": 0.02,
                "peak_prominence": 5.0,
                "repeat_scan_snr": 10.0,
                "repeat_relative_std": 0.1,
                "ok_scan_count": 3,
                "total_scan_count": 3,
            }
        },
        "target_channel_metrics": {
            "1": {
                "mean_peak_current_uA": 3.0,
                "median_peak_current_uA": 2.9,
                "std_peak_current_uA": 0.2,
                "mean_background_rms_uA": 0.3,
                "median_background_rms_uA": 0.28,
                "std_background_rms_uA": 0.03,
                "peak_prominence": 10.0,
                "repeat_scan_snr": 15.0,
                "repeat_relative_std": 0.08,
                "ok_scan_count": 3,
                "total_scan_count": 3,
            }
        },
        "quality": {"channel_components": {"1": {}}},
    }

    required_metrics = {
        "Peak-height STD (µA)",
        "Mean background RMS (µA)",
        "Median background RMS (µA)",
        "Background RMS STD (µA)",
        "Successful scan count",
        "Total scan count",
        "Target − buffer peak height (µA)",
        "Combined buffer + target RMS (µA)",
        "Paired peak prominence",
        "Combined buffer + target peak STD (µA)",
        "Paired repeat-scan SNR",
    }
    assert required_metrics.issubset(REAL_DATA_METRICS)

    expected_paired_values = {
        "Target − buffer peak height (µA)": 2.0,
        "Combined buffer + target RMS (µA)": 0.5,
        "Paired peak prominence": 4.0,
        "Combined buffer + target peak STD (µA)": 0.3,
        "Paired repeat-scan SNR": 2.0 / 0.3,
    }
    for metric, expected in expected_paired_values.items():
        points = _real_metric_points(
            [observation], metric, "measurement", ["1"], False
        )
        assert points["value"].iloc[0] == pytest.approx(expected)
        assert _real_metric_phase_independent(metric)

    noise_points = _real_metric_points(
        [observation], "Mean background RMS (µA)", "target", ["1"], False
    )
    assert noise_points["value"].iloc[0] == pytest.approx(0.3)
