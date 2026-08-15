import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.analysis import is_peak_height_below_cutoff_error
from bo_session_viewer import (
    REAL_DATA_METRICS,
    _iteration_trace_q_score_lines,
    _observation_with_plotted_peak_replicates,
    _pair_buffer_target_traces,
    _paired_trend_differences,
    _paired_pairwise_repeat_snr,
    _plot_paired_phase_trend,
    _real_metric_phase_independent,
    _real_metric_points,
    _session_candidate_count,
    _trace_entries_by_iteration,
    _trace_paths,
)


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
