import numpy as np
import sys
from pathlib import Path
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.plotting import (
    _langmuir_isotherm,
    _invert_langmuir_response,
    _prepare_titration_fit_points,
    _propagated_langmuir_concentration_std,
    _filter_extreme_accuracy_predictions,
    build_titration_langmuir_summary_table,
    build_titration_measurement_accuracy_table,
    build_titration_step_table,
    filter_extreme_titration_outliers,
    infer_titration_response_directions,
    infer_titration_response_baselines,
    plot_metric_vs_scan,
    plot_titration_langmuir,
    plot_titration_concentration_accuracy,
    plot_titration_concentration_vs_measurement,
    plot_titration_plateaus,
    plot_titration_snr,
)


def test_response_direction_inference_does_not_require_langmuir_fit():
    vlines = [
        (1, "buffer"),
        (4, "10 uM"),
        (7, "buffer"),
        (10, "20 uM"),
        (13, "end"),
    ]
    rows = []
    for channel, plateaus in ((1, [1.0, 2.0, 1.0, 3.0]), (2, [10.0, 8.0, 10.0, 6.0])):
        for step_index, plateau in enumerate(plateaus):
            start_scan = 1 + (3 * step_index)
            rows.extend([
                _result(channel, start_scan + offset, plateau)
                for offset in range(3)
            ])

    directions = infer_titration_response_directions(
        rows,
        metric="peak_current_selected",
        vlines=vlines,
        edge_trim_fraction=0,
        concentration_unit="uM",
    )

    assert directions == {1: "signal-on", 2: "signal-off"}
    baselines = infer_titration_response_baselines(
        rows,
        metric="peak_current_selected",
        vlines=vlines,
        edge_trim_fraction=0,
        concentration_unit="uM",
    )
    assert baselines == {1: 1.0, 2: 10.0}


def test_langmuir_and_snr_use_distinct_direction_family_shades():
    vlines = [
        (1, "buffer"),
        (4, "10 uM"),
        (7, "buffer"),
        (10, "20 uM"),
        (13, "end"),
    ]
    plateau_sets = {
        1: [1.0, 2.0, 1.0, 3.0],
        2: [10.0, 8.0, 10.0, 6.0],
        3: [2.0, 4.0, 2.0, 7.0],
        4: [12.0, 9.0, 12.0, 5.0],
    }
    results = []
    for channel, plateaus in plateau_sets.items():
        for step_index, plateau in enumerate(plateaus):
            start_scan = 1 + (3 * step_index)
            results.extend(
                _result(channel, start_scan + offset, plateau)
                for offset in range(3)
            )

    clean_langmuir_figure = plot_titration_langmuir(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
    )
    clean_annotations = "\n".join(
        text.get_text() for text in clean_langmuir_figure.axes[0].texts
    )
    assert clean_langmuir_figure.axes[0].get_xlabel() == "Ligand Concentration (uM)"
    assert clean_langmuir_figure.axes[0].get_ylabel() == "Peak Height (uA)"
    assert not any(
        detail in clean_annotations
        for detail in ("Kd", "LOD", "ULOQ", "Sat.", "saturation")
    )
    assert not any(
        label.endswith((" LOD", " ULOQ"))
        for label in clean_langmuir_figure.axes[0].get_legend_handles_labels()[1]
    )

    langmuir_figure = plot_titration_langmuir(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        edge_trim_fraction=0,
        concentration_unit="uM",
        baseline_mode="preceding_buffer",
    )
    langmuir_axis = langmuir_figure.axes[0]
    langmuir_colors = {
        collection.get_label(): collection.get_facecolors()[0]
        for collection in langmuir_axis.collections
        if collection.get_label().startswith("Channel ")
    }
    assert langmuir_colors["Channel 1 (signal-on)"][0] > langmuir_colors["Channel 1 (signal-on)"][2]
    assert langmuir_colors["Channel 2 (signal-off)"][2] > langmuir_colors["Channel 2 (signal-off)"][0]
    assert not np.allclose(
        langmuir_colors["Channel 1 (signal-on)"],
        langmuir_colors["Channel 3 (signal-on)"],
    )
    assert not np.allclose(
        langmuir_colors["Channel 2 (signal-off)"],
        langmuir_colors["Channel 4 (signal-off)"],
    )

    snr_rows = [
        {
            "channel": channel,
            "step_index": step_index,
            "step_concentration": concentration,
            "titration_snr": snr,
            "plateau_std": np.nan,
        }
        for channel in plateau_sets
        for step_index, (concentration, snr) in enumerate(
            ((10.0, 2.0), (20.0, 4.0)), start=1
        )
    ]
    fit_summary_rows = [
        {
            "channel": channel,
            "langmuir_amplitude": amplitude,
            "langmuir_kd": 10.0,
            "blank_sigma": 1.0,
        }
        for channel, amplitude in ((1, 5.0), (2, -5.0), (3, 8.0), (4, -8.0))
    ]
    snr_figure = plot_titration_snr(
        snr_rows,
        fit_summary_rows=fit_summary_rows,
    )
    snr_colors = {
        line.get_label(): line.get_color()
        for line in snr_figure.axes[0].lines
        if line.get_label().startswith("Channel ") and "fit" not in line.get_label()
    }
    assert snr_colors["Channel 1 (signal-on)"][0] > snr_colors["Channel 1 (signal-on)"][2]
    assert snr_colors["Channel 2 (signal-off)"][2] > snr_colors["Channel 2 (signal-off)"][0]
    assert not np.allclose(
        snr_colors["Channel 1 (signal-on)"], snr_colors["Channel 3 (signal-on)"]
    )
    assert not np.allclose(
        snr_colors["Channel 2 (signal-off)"], snr_colors["Channel 4 (signal-off)"]
    )


def test_mixed_response_peak_plot_uses_direction_specific_axes_and_colors():
    rows = []
    for channel, values in ((1, [1.0, 2.0, 4.0]), (2, [10.0, 7.0, 4.0])):
        rows.extend([
            {
                "channel": channel,
                "scan_number": scan_number,
                "peak_current_selected": value,
            }
            for scan_number, value in enumerate(values, start=1)
        ])

    figure = plot_metric_vs_scan(
        rows,
        metric="peak_current_selected",
        channels=[1, 2],
        ylabel="Peak current (uA)",
        response_directions={1: "signal-on", 2: "signal-off"},
    )

    assert figure is not None
    assert len(figure.axes) == 2
    signal_on_axis, signal_off_axis = figure.axes
    assert signal_on_axis._swv_response_direction == "signal-on"
    assert signal_off_axis._swv_response_direction == "signal-off"
    assert np.allclose(signal_on_axis.get_ylim(), [0.7, 4.3])
    assert np.allclose(signal_off_axis.get_ylim(), [3.4, 10.6])
    assert "signal-on" in signal_on_axis.get_ylabel()
    assert "signal-off" in signal_off_axis.get_ylabel()
    signal_on_color = signal_on_axis.lines[0].get_color()
    signal_off_color = signal_off_axis.lines[0].get_color()
    assert signal_on_color[0] > signal_on_color[2]
    assert signal_off_color[2] > signal_off_color[0]
    legend_labels = [
        text.get_text() for text in signal_on_axis.get_legend().get_texts()
    ]
    assert "Channel 1 (signal-on)" in legend_labels
    assert "Channel 2 (signal-off)" in legend_labels


def test_explicit_channel_palette_is_stable_when_plotting_a_subset():
    rows = [
        {
            "channel": channel,
            "scan_number": scan_number,
            "peak_current_selected": value,
        }
        for channel, values in ((1, [1.0, 2.0]), (2, [8.0, 6.0]))
        for scan_number, value in enumerate(values, start=1)
    ]
    directions = {1: "signal-on", 2: "signal-off"}
    palette = {
        1: (0.95, 0.45, 0.10, 1.0),
        2: (0.10, 0.35, 0.80, 1.0),
    }
    combined = plot_metric_vs_scan(
        rows,
        metric="peak_current_selected",
        channels=[1, 2],
        response_directions=directions,
        channel_colors=palette,
    )
    subset = plot_metric_vs_scan(
        rows,
        metric="peak_current_selected",
        channels=[2],
        response_directions=directions,
        channel_colors=palette,
    )

    assert np.allclose(combined.axes[0].lines[0].get_color(), palette[1])
    assert np.allclose(combined.axes[1].lines[0].get_color(), palette[2])
    assert np.allclose(subset.axes[0].lines[0].get_color(), palette[2])


def test_mixed_response_plot_uses_shades_not_styles_for_swv_settings():
    method_1 = "1 group 1 | optimized"
    method_2 = "1 group 2 | standard"
    rows = []
    for channel, original_channel, values in (
        (method_1, 1, [1.0, 2.0, 3.0]),
        (method_2, 1, [1.2, 2.1, 3.2]),
        (2, 2, [10.0, 8.0, 6.0]),
    ):
        rows.extend([
            {
                "channel": channel,
                "original_channel": original_channel,
                "scan_number": scan_number,
                "peak_current_selected": value,
            }
            for scan_number, value in enumerate(values, start=1)
        ])

    figure = plot_metric_vs_scan(
        rows,
        metric="peak_current_selected",
        channels=[method_1, method_2, 2],
        response_directions={
            method_1: "signal-on",
            method_2: "signal-on",
            2: "signal-off",
        },
    )

    signal_on_lines = figure.axes[0].lines
    assert len(signal_on_lines) == 2
    assert signal_on_lines[0].get_color() != signal_on_lines[1].get_color()
    assert np.linalg.norm(
        np.asarray(signal_on_lines[0].get_color())[:3]
        - np.asarray(signal_on_lines[1].get_color())[:3]
    ) > 0.35
    assert signal_on_lines[0].get_linestyle() == signal_on_lines[1].get_linestyle()
    assert signal_on_lines[0].get_marker() == signal_on_lines[1].get_marker()
    assert signal_on_lines[0].get_color()[2] > signal_on_lines[0].get_color()[0]
    assert signal_on_lines[1].get_color()[2] > signal_on_lines[1].get_color()[0]
    assert figure.axes[1].lines[0].get_color()[2] > figure.axes[1].lines[0].get_color()[0]


def test_same_direction_swv_settings_use_distinct_same_family_shades():
    method_1 = "1 group 1 | optimized"
    method_2 = "1 group 2 | standard"
    rows = [
        {
            "channel": channel,
            "original_channel": 1,
            "scan_number": scan_number,
            "peak_current_selected": value,
        }
        for channel, values in (
            (method_1, [1.0, 2.0, 3.0]),
            (method_2, [1.1, 2.1, 3.1]),
        )
        for scan_number, value in enumerate(values, start=1)
    ]

    figure = plot_metric_vs_scan(
        rows,
        metric="peak_current_selected",
        channels=[method_1, method_2],
        ylabel="Change in Peak Height (uA)",
        xlabel="SWV Measurement Number",
        response_directions={method_1: "signal-on", method_2: "signal-on"},
    )

    assert len(figure.axes) == 1
    lines = figure.axes[0].lines
    assert [line.get_label() for line in lines] == [
        "Optimized Method",
        "Manual Method",
    ]
    assert figure.axes[0].get_legend().get_title().get_text() == ""
    assert figure.axes[0].get_ylabel() == "Change in Peak Height (uA)"
    assert figure.axes[0].get_xlabel() == "SWV Measurement Number"
    assert lines[0].get_color() != lines[1].get_color()
    assert np.linalg.norm(
        np.asarray(lines[0].get_color())[:3]
        - np.asarray(lines[1].get_color())[:3]
    ) > 0.35
    assert lines[0].get_color()[2] > lines[0].get_color()[0]
    assert lines[1].get_color()[2] > lines[1].get_color()[0]
    assert lines[0].get_linestyle() == lines[1].get_linestyle()
    assert lines[0].get_marker() == lines[1].get_marker()
    assert figure.axes[0].get_legend()._loc == 6  # center left


def test_method_colors_are_fixed_across_physical_channels():
    channels = [f"{physical} group {method}" for physical in range(1, 5) for method in (1, 2)]
    rows = [
        {
            "channel": channel,
            "original_channel": int(channel.split()[0]),
            "scan_number": scan_number,
            "peak_current_selected": float(scan_number),
        }
        for channel in channels
        for scan_number in (1, 2)
    ]
    figure = plot_metric_vs_scan(
        rows,
        metric="peak_current_selected",
        channels=channels,
        response_directions={channel: "signal-on" for channel in channels},
    )
    line_colors = [np.asarray(line.get_color())[:3] for line in figure.axes[0].lines]

    method_1_colors = line_colors[::2]
    method_2_colors = line_colors[1::2]
    for method_1_color, method_2_color in zip(method_1_colors, method_2_colors):
        assert np.linalg.norm(method_1_color - method_2_color) > 0.65
        assert method_1_color[2] > method_1_color[0]
        assert method_2_color[2] > method_2_color[0]
    assert all(np.allclose(color, method_1_colors[0]) for color in method_1_colors)
    assert all(np.allclose(color, method_2_colors[0]) for color in method_2_colors)
    assert np.mean(method_1_colors[0]) < np.mean(method_2_colors[0])


def test_metric_y_axes_stay_black_for_response_colored_channels():
    rows = [
        {
            "channel": channel,
            "scan_number": scan_number,
            "peak_current_selected": value,
        }
        for channel, values in ((1, [1.0, 2.0]), (2, [8.0, 6.0]))
        for scan_number, value in enumerate(values, start=1)
    ]
    figure = plot_metric_vs_scan(
        rows,
        metric="peak_current_selected",
        channels=[1, 2],
        response_directions={1: "signal-on", 2: "signal-off"},
    )

    for axis, spine_name in zip(figure.axes, ("left", "right")):
        assert axis.yaxis.label.get_color() == "black"
        assert axis.spines[spine_name].get_edgecolor()[:3] == (0.0, 0.0, 0.0)
        assert all(tick.get_color() == "black" for tick in axis.get_yticklabels())


def test_normalized_mixed_response_plot_retains_direction_axes_and_colors():
    rows = [
        {
            "channel": channel,
            "scan_number": scan_number,
            "peak_current_selected": value,
        }
        for channel, values in (
            (1, [2.0, 3.0, 6.0]),
            (2, [12.0, 9.0, 3.0]),
        )
        for scan_number, value in enumerate(values, start=1)
    ]

    figure = plot_metric_vs_scan(
        rows,
        metric="peak_current_selected",
        channels=[1, 2],
        ylabel="Normalized Peak Current Change",
        normalize_per_channel=True,
        response_directions={1: "signal-on", 2: "signal-off"},
    )

    signal_on_axis, signal_off_axis = figure.axes
    assert np.allclose(signal_on_axis.get_ylim(), [-0.1, 1.1])
    assert np.allclose(signal_off_axis.get_ylim(), [-1.1, 0.1])
    assert np.allclose(signal_on_axis.lines[0].get_ydata(), [0.0, 0.25, 1.0])
    assert np.allclose(signal_off_axis.lines[0].get_ydata(), [0.0, -1.0 / 3.0, -1.0])
    assert signal_on_axis.lines[0].get_color()[0] > signal_on_axis.lines[0].get_color()[2]
    assert signal_off_axis.lines[0].get_color()[2] > signal_off_axis.lines[0].get_color()[0]


def test_buffer_offset_mixed_response_plot_uses_one_signed_axis():
    rows = [
        {
            "channel": channel,
            "scan_number": scan_number,
            "peak_current_selected": value,
        }
        for channel, values in (
            (1, [2.0, 3.0, 6.0]),
            (2, [12.0, 9.0, 3.0]),
        )
        for scan_number, value in enumerate(values, start=1)
    ]

    figure = plot_metric_vs_scan(
        rows,
        metric="peak_current_selected",
        channels=[1, 2],
        ylabel="Peak Current Change from Buffer (uA)",
        response_directions={1: "signal-on", 2: "signal-off"},
        response_baselines={1: 2.0, 2: 12.0},
        offset_to_response_baseline=True,
    )

    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert np.allclose(axis.lines[0].get_ydata(), [0.0, 1.0, 4.0])
    assert np.allclose(axis.lines[1].get_ydata(), [0.0, -3.0, -9.0])
    assert np.allclose(axis.get_ylim(), [-10.3, 5.3])
    assert axis.lines[0].get_color()[0] > axis.lines[0].get_color()[2]
    assert axis.lines[1].get_color()[2] > axis.lines[1].get_color()[0]
    assert any(
        len(line.get_ydata()) == 2
        and np.allclose(line.get_ydata(), [0.0, 0.0])
        for line in axis.lines
    )


def test_other_metric_uses_direction_colors_without_axis_manipulation():
    rows = [
        {
            "channel": channel,
            "scan_number": scan_number,
            "skew": value,
        }
        for channel, values in (
            (1, [0.2, 0.4, 0.3]),
            (2, [-0.1, -0.3, -0.2]),
        )
        for scan_number, value in enumerate(values, start=1)
    ]

    figure = plot_metric_vs_scan(
        rows,
        metric="skew",
        channels=[1, 2],
        ylabel="Skew",
        response_directions={1: "signal-on", 2: "signal-off"},
        response_direction_colors_only=True,
    )

    assert len(figure.axes) == 1
    axis = figure.axes[0]
    assert axis.get_ylabel() == "Skew"
    assert np.allclose(axis.lines[0].get_ydata(), [0.2, 0.4, 0.3])
    assert np.allclose(axis.lines[1].get_ydata(), [-0.1, -0.3, -0.2])
    assert np.allclose(axis.get_ylim(), [-0.37, 0.47])
    assert axis.lines[0].get_color()[0] > axis.lines[0].get_color()[2]
    assert axis.lines[1].get_color()[2] > axis.lines[1].get_color()[0]


def test_wavelet_energy_supports_buffer_relative_signed_translation():
    rows = [
        {
            "channel": channel,
            "scan_number": scan_number,
            "wavelet_energy": value,
        }
        for channel, values in (
            (1, [5.0, 7.0, 9.0]),
            (2, [20.0, 16.0, 11.0]),
        )
        for scan_number, value in enumerate(values, start=1)
    ]

    figure = plot_metric_vs_scan(
        rows,
        metric="wavelet_energy",
        channels=[1, 2],
        ylabel="Wavelet Energy Change from Buffer (a.u.)",
        response_directions={1: "signal-on", 2: "signal-off"},
        response_baselines={1: 5.0, 2: 20.0},
        offset_to_response_baseline=True,
    )

    assert len(figure.axes) == 1
    assert np.allclose(figure.axes[0].lines[0].get_ydata(), [0.0, 2.0, 4.0])
    assert np.allclose(figure.axes[0].lines[1].get_ydata(), [0.0, -4.0, -9.0])


def test_plateau_plot_respects_response_colors_and_buffer_translation():
    vlines = [
        (1, "buffer"),
        (4, "10 uM"),
        (7, "buffer"),
        (10, "20 uM"),
        (13, "end"),
    ]
    rows = []
    for channel, plateaus in (
        (1, [2.0, 3.0, 2.0, 5.0]),
        (2, [12.0, 9.0, 12.0, 6.0]),
    ):
        for step_index, plateau in enumerate(plateaus):
            start_scan = 1 + (3 * step_index)
            rows.extend([
                _result(channel, start_scan + offset, plateau)
                for offset in range(3)
            ])

    figure = plot_titration_plateaus(
        rows,
        metric="peak_current_selected",
        channels=[1, 2],
        vlines=vlines,
        edge_trim_fraction=0,
        ylabel="Peak Current Change from Buffer (uA)",
        response_directions={1: "signal-on", 2: "signal-off"},
        response_baselines={1: 2.0, 2: 12.0},
        offset_to_response_baseline=True,
    )

    assert figure is not None
    axis = figure.axes[0]
    signal_on_points = next(
        collection for collection in axis.collections
        if collection.get_label() == "Channel 1 (signal-on)"
    )
    signal_off_points = next(
        collection for collection in axis.collections
        if collection.get_label() == "Channel 2 (signal-off)"
    )
    assert np.allclose(signal_on_points.get_offsets()[:, 1], [0.0, 1.0, 0.0, 3.0])
    assert np.allclose(signal_off_points.get_offsets()[:, 1], [0.0, -3.0, 0.0, -6.0])
    assert signal_on_points.get_facecolors()[0, 0] > signal_on_points.get_facecolors()[0, 2]
    assert signal_off_points.get_facecolors()[0, 2] > signal_off_points.get_facecolors()[0, 0]
    assert any(
        len(line.get_ydata()) == 2
        and np.allclose(line.get_ydata(), [0.0, 0.0])
        for line in axis.lines
    )


def test_concentration_diagnostics_use_response_direction_colors():
    rows = []
    for channel, amplitude, predictions in (
        (1, 5.0, [11.0, 19.0]),
        (2, -5.0, [9.0, 21.0]),
        (3, 4.0, [10.5, 20.5]),
        (4, -4.0, [9.5, 19.5]),
    ):
        for scan_number, (known, predicted) in enumerate(
            zip([10.0, 20.0], predictions),
            start=1,
        ):
            rows.append({
                "channel": channel,
                "original_channel": channel,
                "scan_number": scan_number,
                "source_scan_number": scan_number,
                "known_concentration": known,
                "predicted_concentration": predicted,
                "predicted_concentration_std": None,
                "concentration_censored_at_lod": False,
                "absolute_percent_error": 100.0 * abs(predicted - known) / known,
                "fit_amplitude": amplitude,
                "limit_of_detection": None,
                "upper_limit_of_quantification": None,
            })

    accuracy_figure = plot_titration_concentration_accuracy(rows)
    accuracy_axis = accuracy_figure.axes[0]
    accuracy_on = next(
        collection for collection in accuracy_axis.collections
        if collection.get_label() == "Channel 1 (signal-on)"
    )
    accuracy_off = next(
        collection for collection in accuracy_axis.collections
        if collection.get_label() == "Channel 2 (signal-off)"
    )
    accuracy_on_2 = next(
        collection for collection in accuracy_axis.collections
        if collection.get_label() == "Channel 3 (signal-on)"
    )
    accuracy_off_2 = next(
        collection for collection in accuracy_axis.collections
        if collection.get_label() == "Channel 4 (signal-off)"
    )
    assert all(
        np.isclose(collection.get_alpha(), 0.30)
        and collection._swv_preserve_alpha is True
        for collection in (
            accuracy_on,
            accuracy_off,
            accuracy_on_2,
            accuracy_off_2,
        )
    )
    assert accuracy_on.get_facecolors()[0, 0] > accuracy_on.get_facecolors()[0, 2]
    assert accuracy_off.get_facecolors()[0, 2] > accuracy_off.get_facecolors()[0, 0]
    assert not np.allclose(
        accuracy_on.get_facecolors()[0], accuracy_on_2.get_facecolors()[0]
    )
    assert not np.allclose(
        accuracy_off.get_facecolors()[0], accuracy_off_2.get_facecolors()[0]
    )

    measurement_figure = plot_titration_concentration_vs_measurement(rows)
    measurement_axis = measurement_figure.axes[0]
    assert measurement_axis.get_ylabel() == "Predicted Concentration (uM)"
    assert measurement_axis.get_legend()._loc == 2
    measurement_on = next(
        collection for collection in measurement_axis.collections
        if collection.get_label() == "Channel 1 (signal-on)"
    )
    measurement_off = next(
        collection for collection in measurement_axis.collections
        if collection.get_label() == "Channel 2 (signal-off)"
    )
    measurement_on_2 = next(
        collection for collection in measurement_axis.collections
        if collection.get_label() == "Channel 3 (signal-on)"
    )
    measurement_off_2 = next(
        collection for collection in measurement_axis.collections
        if collection.get_label() == "Channel 4 (signal-off)"
    )
    assert measurement_on.get_facecolors()[0, 0] > measurement_on.get_facecolors()[0, 2]
    assert measurement_off.get_facecolors()[0, 2] > measurement_off.get_facecolors()[0, 0]
    assert not np.allclose(
        measurement_on.get_facecolors()[0], measurement_on_2.get_facecolors()[0]
    )
    assert not np.allclose(
        measurement_off.get_facecolors()[0], measurement_off_2.get_facecolors()[0]
    )


def test_accuracy_plot_reports_color_coded_rms_fold_error_by_method():
    rows = []
    for channel, predictions in (
        ("2 group 1 | optimized", [100.0, 260.0]),
        ("2 group 2 | manual", [110.0, 190.0]),
    ):
        for scan_number, (known, predicted) in enumerate(
            zip([100.0, 200.0], predictions),
            start=1,
        ):
            rows.append({
                "channel": channel,
                "original_channel": 2,
                "scan_number": scan_number,
                "known_concentration": known,
                "predicted_concentration": predicted,
                "absolute_percent_error": 100.0 * abs(predicted - known) / known,
                "fit_amplitude": 1.0,
                "limit_of_detection": None,
                "upper_limit_of_quantification": None,
            })

    figure = plot_titration_concentration_accuracy(rows, concentration_unit="uM")
    axis = figure.axes[0]
    annotation_by_label = {
        text.get_text().split(" RMS Fold Error", 1)[0]: text
        for text in axis.texts
        if " RMS Fold Error:" in text.get_text()
    }
    assert annotation_by_label["Optimized"].get_text() == (
        "Optimized RMS Fold Error: 1.20×"
    )
    assert annotation_by_label["Manual"].get_text() == (
        "Manual RMS Fold Error: 1.08×"
    )
    for compact_method_label, annotation in annotation_by_label.items():
        method_label = f"{compact_method_label} Method"
        method_points = next(
            collection for collection in axis.collections
            if collection.get_label() == method_label
        )
        assert np.allclose(
            np.asarray(annotation.get_color())[:3],
            method_points.get_facecolors()[0, :3],
        )
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "Within ±20%",
        "1:1",
    ]


def test_propagated_langmuir_uncertainty_grows_toward_saturation():
    baseline, amplitude, kd = 1.0, 10.0, 100.0
    low_response = _langmuir_isotherm(10.0, baseline, amplitude, kd)
    high_response = _langmuir_isotherm(1000.0, baseline, amplitude, kd)

    low_std = _propagated_langmuir_concentration_std(
        low_response, baseline, amplitude, kd, response_sigma=0.1
    )
    high_std = _propagated_langmuir_concentration_std(
        high_response, baseline, amplitude, kd, response_sigma=0.1
    )

    assert low_std is not None
    assert high_std is not None
    assert high_std > low_std


def test_langmuir_inversion_retains_negative_below_baseline_estimates():
    predicted = _invert_langmuir_response(
        response=0.5,
        baseline=1.0,
        amplitude=10.0,
        kd=100.0,
    )

    assert predicted is not None
    assert predicted < 0


def _has_nonzero_vertical_errorbar(axis):
    return any(
        len(segment) == 2
        and np.isclose(segment[0][0], segment[1][0])
        and not np.isclose(segment[0][1], segment[1][1])
        for collection in axis.collections
        if isinstance(collection, LineCollection)
        for segment in collection.get_segments()
    )


def test_extreme_titration_outlier_filter_is_local_to_interval():
    rows = [
        {
            "channel": 1,
            "status": "OK",
            "scan_number": scan,
            "wavelet_energy": value,
        }
        for scan, value in enumerate([1.0, 0.9, 1.1, 100.0, 7.0, 7.1, 6.9], start=1)
    ]
    filtered = filter_extreme_titration_outliers(
        rows,
        metric="wavelet_energy",
        vlines=[(1, "10 uM"), (5, "20 uM"), (8, "end")],
    )

    assert [row["scan_number"] for row in filtered] == [1, 2, 3, 5, 6, 7]

    step_rows = build_titration_step_table(
        rows,
        metric="wavelet_energy",
        vlines=[(1, "10 uM"), (5, "20 uM"), (8, "end")],
        edge_trim_fraction=0.0,
        remove_extreme_outliers=True,
    )
    assert np.isclose(step_rows[0]["plateau_value"], 1.0)


def test_extreme_prediction_created_by_langmuir_inversion_is_removed():
    rows = [
        {
            "channel": 1,
            "step_selection_key": "80 uM",
            "known_concentration": 80.0,
            "predicted_concentration": predicted,
        }
        for predicted in (75.0, 125.0, 11975.0)
    ]

    filtered = _filter_extreme_accuracy_predictions(rows)

    assert [row["predicted_concentration"] for row in filtered] == [75.0, 125.0]


def _result(channel, scan, value):
    return {
        "channel": channel,
        "scan_number": scan,
        "status": "OK",
        "peak_current_selected": value,
        "wavelet_energy": value,
    }


def test_channel_specific_vlines_keep_swv_groups_independent():
    results = [
        _result("1 group 1", 1, 10.0),
        _result("1 group 1", 2, 20.0),
        _result("1 group 2", 1, 100.0),
        _result("1 group 2", 2, 200.0),
    ]
    for result in results:
        result["original_channel"] = 1
    by_channel = {
        "1 group 1": [(1, "10 uM"), (2, "20 uM"), (3, "end")],
        "1 group 2": [(1, "10 uM"), (2, "20 uM"), (3, "end")],
    }

    rows = build_titration_step_table(
        results,
        metric="peak_current_selected",
        vlines=[],
        vlines_by_channel=by_channel,
        concentration_unit="uM",
        edge_trim_fraction=0,
    )

    assert [(row["channel"], row["plateau_value"]) for row in rows] == [
        ("1 group 1", 10.0),
        ("1 group 1", 20.0),
        ("1 group 2", 100.0),
        ("1 group 2", 200.0),
    ]
    assert all(row["original_channel"] == 1 for row in rows)

    figure = plot_titration_langmuir(
        results,
        metric="peak_current_selected",
        vlines=[],
        vlines_by_channel=by_channel,
        channels=["1 group 1", "1 group 2"],
        concentration_unit="uM",
        edge_trim_fraction=0,
    )
    assert figure is not None
    assert figure.axes[0].get_ylim()[0] == 0.0
    legend = figure.axes[0].get_legend()
    assert legend.get_title().get_text() == ""
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == ["Optimized Method", "Manual Method"]


def test_langmuir_axis_keeps_highest_concentration_after_earlier_saturation():
    labels = [
        "buffer", "10 uM", "buffer", "20 uM",
        "buffer", "40 uM", "buffer", "80 uM", "end",
    ]
    vlines = [(1 + (3 * index), label) for index, label in enumerate(labels)]
    plateaus = [5.0, 7.0, 5.0, 9.0, 5.0, 12.0, 5.0, 10.0]
    results = []
    for step_index, plateau in enumerate(plateaus):
        start = 1 + (3 * step_index)
        results.extend([
            _result(1, start, plateau - 0.1),
            _result(1, start + 1, plateau),
            _result(1, start + 2, plateau + 0.1),
        ])

    figure = plot_titration_langmuir(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
        show_uloq=False,
    )

    assert figure is not None
    assert figure.axes[0].get_xlim()[1] > 80.0


def test_repeated_buffer_doses_are_collapsed_and_sorted_for_fit_display():
    steps = [
        {"step_index": 1, "step_concentration": 40.0, "plateau_value": 4.0},
        {"step_index": 2, "step_concentration": 0.0, "plateau_value": 1.0},
        {"step_index": 3, "step_concentration": 80.0, "plateau_value": 7.0},
        {"step_index": 4, "step_concentration": 0.0, "plateau_value": 3.0},
    ]

    x, y, axis_kind, _fit_steps = _prepare_titration_fit_points(steps)

    assert axis_kind == "concentration"
    assert np.allclose(x, [0.0, 40.0, 80.0])
    assert np.allclose(y, [2.0, 4.0, 7.0])


def test_preceding_buffer_mode_subtracts_drifting_buffer_and_omits_buffers():
    labels = ["buffer", "40 uM", "buffer", "80 uM", "end"]
    vlines = [(1 + 2 * index, label) for index, label in enumerate(labels)]
    plateaus = [10.0, 14.0, 12.0, 21.0]
    results = []
    for step, plateau in enumerate(plateaus):
        start = 1 + 2 * step
        results.extend([
            _result(1, start, plateau - 0.1),
            _result(1, start + 1, plateau + 0.1),
        ])
    # Give buffer_2 much higher noise while preserving its median.
    results[4]["peak_current_selected"] = 11.0
    results[4]["wavelet_energy"] = 11.0
    results[5]["peak_current_selected"] = 13.0
    results[5]["wavelet_energy"] = 13.0

    rows = build_titration_step_table(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
    )

    assert [row["step_concentration"] for row in rows] == [40.0, 80.0]
    assert np.allclose([row["plateau_value"] for row in rows], [14.0, 19.0])
    assert [row["baseline_step_index"] for row in rows] == [1, 3]
    assert [row["fixed_langmuir_baseline"] for row in rows] == [10.0, 10.0]

    selected_rows = build_titration_step_table(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
        included_step_labels=["buffer_2", "80 uM"],
    )

    assert [row["step_concentration"] for row in selected_rows] == [80.0]
    assert np.isclose(selected_rows[0]["plateau_value"], 21.0)
    assert selected_rows[0]["baseline_step_index"] == 3
    assert selected_rows[0]["fixed_langmuir_baseline"] == 12.0

    uncorrected_selected_rows = build_titration_step_table(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="none",
        included_step_labels=["buffer_2", "80 uM"],
    )

    uncorrected_target = next(
        row for row in uncorrected_selected_rows
        if row["step_concentration"] == 80.0
    )
    assert np.isclose(uncorrected_target["plateau_value"], 21.0)
    assert uncorrected_target["fixed_langmuir_baseline"] == 12.0
    assert uncorrected_target["first_buffer_step_index"] == 3

    all_uncorrected_rows = build_titration_step_table(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="none",
    )
    assert [
        row["step_selection_key"]
        for row in all_uncorrected_rows
        if row["step_note"] == "buffer"
    ] == ["buffer_1", "buffer_2"]

    corrected_with_deselected_buffer = build_titration_step_table(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
        included_step_labels=["buffer_1", "80 uM"],
    )
    assert len(corrected_with_deselected_buffer) == 1
    corrected_target = corrected_with_deselected_buffer[0]
    assert np.isclose(corrected_target["plateau_value"], 19.0)
    assert corrected_target["baseline_step_index"] == 3
    assert corrected_target["fixed_langmuir_baseline"] == 10.0
    assert np.allclose(
        corrected_target["lod_buffer_stds"],
        [np.sqrt(0.02)],
    )
    assert np.isclose(corrected_target["baseline_plateau_std"], np.sqrt(2.0))

    fallback_anchor_target = build_titration_step_table(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="none",
        included_step_labels=["buffer_1", "80 uM"],
    )
    target_row = next(
        row for row in fallback_anchor_target if row["step_concentration"] == 80.0
    )
    assert target_row["fixed_langmuir_baseline"] == 10.0

    plateau_figure = plot_titration_plateaus(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        edge_trim_fraction=0,
        baseline_mode="none",
        included_step_labels=["buffer_2", "80 uM"],
    )

    assert plateau_figure is not None
    assert np.allclose(plateau_figure.axes[0].get_xlim(), [5.0, 9.0])
    assert _has_nonzero_vertical_errorbar(plateau_figure.axes[0])


def test_langmuir_summary_reports_kd_and_lod_for_buffer_baselined_targets():
    concentrations = [10.0, 20.0, 40.0, 80.0]
    kd = 25.0
    amplitude = 12.0
    labels = []
    plateau_specs = []
    for index, concentration in enumerate(concentrations):
        buffer = 5.0 + index
        response = amplitude * concentration / (kd + concentration)
        labels.extend(["buffer", f"{concentration:g} uM"])
        plateau_specs.extend([buffer, buffer + response])
    labels.append("end")
    vlines = [(1 + 3 * index, label) for index, label in enumerate(labels)]
    results = []
    for step, plateau in enumerate(plateau_specs):
        start = 1 + 3 * step
        results.extend([
            _result(1, start, plateau - 0.2),
            _result(1, start + 1, plateau),
            _result(1, start + 2, plateau + 0.2),
        ])

    summary = build_titration_langmuir_summary_table(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
    )

    assert len(summary) == 1
    assert np.isclose(summary[0]["langmuir_kd"], kd, rtol=0.05)
    assert np.isclose(summary[0]["langmuir_baseline"], 5.0)
    assert summary[0]["langmuir_baseline_fixed"] is True
    assert summary[0]["limit_of_detection"] is not None
    assert summary[0]["limit_of_detection"] > 0
    assert summary[0]["upper_limit_of_quantification"] is not None
    assert summary[0]["upper_limit_of_quantification"] > summary[0]["limit_of_detection"]
    expected_uloq = kd * ((amplitude / (3.0 * 0.2)) - 1.0)
    assert np.isclose(
        summary[0]["upper_limit_of_quantification"],
        expected_uloq,
        rtol=0.05,
    )
    assert summary[0]["upper_limit_of_quantification"] > max(concentrations)
    assert summary[0]["upper_limit_of_quantification_is_extrapolated"] is True
    assert "target plateaus" in summary[0]["upper_limit_of_quantification_noise_source"]
    expected_snr_3 = (3.0 * 0.2 * kd) / (amplitude - (3.0 * 0.2))
    assert np.isclose(summary[0]["snr_3_cutoff_concentration"], expected_snr_3, rtol=0.05)
    assert summary[0]["baseline_mode"] == "preceding_buffer"

    peak_steps = build_titration_step_table(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
    )
    assert all(row["titration_snr"] is not None for row in peak_steps)
    assert all(row["titration_snr"] > 0 for row in peak_steps)

    accuracy_rows = build_titration_measurement_accuracy_table(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
    )
    assert len(accuracy_rows) == len(concentrations) * 3
    assert all("predicted_concentration" in row for row in accuracy_rows)
    propagated_uncertainties = [
        row["predicted_concentration_std"]
        for row in accuracy_rows
        if row.get("predicted_concentration_std") is not None
    ]
    assert propagated_uncertainties
    assert all(value >= 0 for value in propagated_uncertainties)
    assert all(
        row["predicted_concentration_lower_1sigma"]
        <= row["predicted_concentration"]
        <= row["predicted_concentration_upper_1sigma"]
        for row in accuracy_rows
        if row.get("predicted_concentration_std") is not None
    )
    median_absolute_error = np.median([
        row["absolute_percent_error"]
        for row in accuracy_rows
        if row["absolute_percent_error"] is not None
    ])
    assert median_absolute_error < 10.0

    measurement_rows = build_titration_measurement_accuracy_table(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
        include_buffer_measurements=True,
    )
    buffer_measurement_rows = [
        row for row in measurement_rows
        if row["known_concentration"] == 0
    ]
    assert len(buffer_measurement_rows) == len(concentrations) * 3
    assert all(
        row["predicted_concentration"] is not None
        for row in buffer_measurement_rows
    )
    assert any(
        row["unbounded_predicted_concentration"] < 0
        for row in buffer_measurement_rows
    )
    lod_censored_rows = [
        row for row in buffer_measurement_rows
        if row["concentration_censored_at_lod"]
    ]
    assert lod_censored_rows
    assert all(
        np.isclose(row["predicted_concentration"], row["limit_of_detection"])
        for row in lod_censored_rows
    )
    assert all(
        row["predicted_concentration_std"] is None
        for row in lod_censored_rows
    )

    snr_figure = plot_titration_snr(
        peak_steps,
        concentration_unit="uM",
        fit_summary_rows=summary,
    )
    assert snr_figure is not None
    assert snr_figure.axes[0].get_xscale() == "linear"
    assert snr_figure.axes[0].get_ylabel() == "Plateau SNR"
    lod_cutoff_lines = [
        line for line in snr_figure.axes[0].lines
        if line.get_label() == "LOD cutoff (SNR = 3)"
    ]
    assert len(lod_cutoff_lines) == 1
    assert np.allclose(lod_cutoff_lines[0].get_ydata(), [3.0, 3.0])
    assert any(
        " LOD fit SNR=3" in line.get_label()
        for line in snr_figure.axes[0].lines
    )
    assert any(
        "Langmuir SNR fit" in line.get_label()
        for line in snr_figure.axes[0].lines
    )
    assert any("ULOQ" in line.get_label() for line in snr_figure.axes[0].lines)
    assert _has_nonzero_vertical_errorbar(snr_figure.axes[0])

    accuracy_figure = plot_titration_concentration_accuracy(
        accuracy_rows,
        concentration_unit="uM",
    )
    assert accuracy_figure is not None
    assert accuracy_figure.axes[0].get_xscale() == "log"
    assert accuracy_figure.axes[0].get_yscale() == "log"
    assert accuracy_figure.axes[0].get_title() == "Predicted vs. Known"
    assert accuracy_figure.axes[0].get_aspect() == 1.0
    assert not any(
        bool(getattr(collection, "_swv_concentration_errorbar", False))
        for collection in accuracy_figure.axes[0].collections
    )
    accuracy_axis_position = accuracy_figure.axes[0].get_position()
    assert np.isclose(
        accuracy_axis_position.width * accuracy_figure.get_figwidth(),
        accuracy_axis_position.height * accuracy_figure.get_figheight(),
    )
    expected_log_margin = 0.05 * np.log(80.0 / 10.0)
    expected_limits = [
        10.0 / np.exp(expected_log_margin),
        80.0 * np.exp(expected_log_margin),
    ]
    assert np.allclose(accuracy_figure.axes[0].get_xlim(), expected_limits)
    assert np.allclose(accuracy_figure.axes[0].get_ylim(), expected_limits)
    assert not any(line.get_label() == "LOD boundaries" for line in accuracy_figure.axes[0].lines)
    assert not any(line.get_label() == "ULOQ boundaries" for line in accuracy_figure.axes[0].lines)
    accuracy_handles, accuracy_labels = accuracy_figure.axes[0].get_legend_handles_labels()
    assert "Within ±20%" in accuracy_labels
    acceptance_regions = [
        collection for collection in accuracy_figure.axes[0].collections
        if collection.get_label() == "Within ±20%"
    ]
    assert len(acceptance_regions) == 1
    assert np.isclose(acceptance_regions[0].get_alpha(), 0.10)
    assert acceptance_regions[0]._swv_preserve_alpha is True
    error_bound_lines = [
        line for line in accuracy_figure.axes[0].lines
        if line.get_color() == "tab:green" and line.get_linestyle() == ":"
    ]
    assert len(error_bound_lines) == 2
    assert np.isclose(
        error_bound_lines[0].get_ydata()[-1]
        / error_bound_lines[0].get_xdata()[-1],
        0.8,
    )
    assert np.isclose(
        error_bound_lines[1].get_ydata()[-1]
        / error_bound_lines[1].get_xdata()[-1],
        1.2,
    )
    accuracy_legend = accuracy_figure.axes[0].get_legend()
    assert accuracy_legend._loc == 4
    assert accuracy_legend.get_title().get_text() == ""
    assert [text.get_text() for text in accuracy_legend.get_texts()] == [
        "Within ±20%",
        "1:1",
    ]
    accuracy_annotation = "\n".join(
        text.get_text() for text in accuracy_figure.axes[0].texts
    )
    assert "RMS Fold Error:" in accuracy_annotation
    assert "% RMSE:" not in accuracy_annotation
    assert "within ±20%:" not in accuracy_annotation
    assert "Median |error|" not in accuracy_annotation
    expected_log_rmse = np.sqrt(np.mean([
        np.log10(
            row["predicted_concentration"] / row["known_concentration"]
        ) ** 2
        for row in accuracy_rows
    ]))
    expected_rms_fold_error = 10.0 ** expected_log_rmse
    assert (
        f"RMS Fold Error: {expected_rms_fold_error:.2f}×"
        in accuracy_annotation
    )

    custom_alpha_figure = plot_titration_concentration_accuracy(
        accuracy_rows,
        concentration_unit="uM",
        acceptance_region_alpha=0.35,
    )
    custom_region = next(
        collection for collection in custom_alpha_figure.axes[0].collections
        if collection.get_label() == "Within ±20%"
    )
    assert np.isclose(custom_region.get_alpha(), 0.35)

    interleaved_source_measurement_rows = [
        {
            **row,
            "source_scan_number": 2 * row["scan_number"],
        }
        for row in measurement_rows
    ]
    time_figure = plot_titration_concentration_vs_measurement(
        interleaved_source_measurement_rows,
        concentration_unit="uM",
        vlines=vlines,
    )
    assert time_figure is not None
    assert np.isclose(
        accuracy_figure.get_figheight(),
        time_figure.get_figheight(),
    )
    assert time_figure.axes[0].get_xlabel() == "SWV Measurement Number"
    assert time_figure.axes[0].get_ylabel() == "Predicted Concentration (uM)"
    assert not bool(getattr(time_figure, "_swv_manual_layout", False))
    assert time_figure.axes[0].get_yscale() == "linear"
    assert time_figure.axes[0].get_position().width > 0.7
    assert time_figure.axes[0]._swv_concentration_doubling_scale is True
    assert np.isclose(
        time_figure.axes[0]._swv_concentration_doubling_reference,
        min(concentrations),
    )
    assert [tick.get_text() for tick in time_figure.axes[0].get_yticklabels()] == [
        "0", "10", "20", "40", "80",
    ]
    time_labels = time_figure.axes[0].get_legend_handles_labels()[1]
    assert "Known Concentration" in time_labels
    assert "Known Concentration ±20%" not in time_labels
    assert time_figure.axes[0].get_legend().get_title().get_text() == ""
    titration_annotation_labels = {
        text.get_text() for text in time_figure.axes[0].texts
    }
    assert set(labels).issubset(titration_annotation_labels)
    titration_boundary_lines = [
        line for line in time_figure.axes[0].lines
        if line.get_color() == "gray" and line.get_linestyle() == "--"
    ]
    assert len(titration_boundary_lines) == len(vlines)
    vline_label_artists = [
        text for text in time_figure.axes[0].texts
        if text.get_text() in labels
    ]
    assert vline_label_artists
    assert all(text.get_fontsize() == 9 for text in vline_label_artists)
    known_reference = next(
        line for line in time_figure.axes[0].lines
        if line.get_label() == "Known Concentration"
    )
    known_reference_levels = np.asarray(known_reference.get_ydata(), dtype=float)
    assert max(known_reference.get_xdata()) == max(
        row["scan_number"] for row in measurement_rows
    )
    assert max(known_reference.get_xdata()) * 2 == max(
        row["source_scan_number"]
        for row in interleaved_source_measurement_rows
    )
    assert known_reference.get_zorder() > max(
        collection.get_zorder()
        for collection in time_figure.axes[0].collections
    )
    assert np.isclose(np.min(known_reference_levels), 0.0)
    assert set(np.unique(known_reference_levels)) == {0.0, 1.0, 2.0, 3.0, 4.0}
    prediction_collections = [
        collection for collection in time_figure.axes[0].collections
        if collection.get_label().startswith("Channel ")
    ]
    assert prediction_collections
    plotted_prediction_levels = np.concatenate([
        np.asarray(collection.get_offsets())[:, 1]
        for collection in prediction_collections
    ])
    errorbar_levels = np.concatenate([
        np.asarray(segment)[:, 1]
        for collection in time_figure.axes[0].collections
        if hasattr(collection, "get_segments")
        for segment in collection.get_segments()
        if np.asarray(segment).size
    ])
    all_displayed_levels = np.concatenate((
        known_reference_levels,
        plotted_prediction_levels,
        errorbar_levels,
    ))
    displayed_minimum = float(np.nanmin(all_displayed_levels))
    displayed_maximum = float(np.nanmax(all_displayed_levels))
    displayed_span = displayed_maximum - displayed_minimum
    assert np.allclose(
        time_figure.axes[0].get_ylim(),
        [
            displayed_minimum - (0.1 * displayed_span),
            displayed_maximum + (0.1 * displayed_span),
        ],
    )
    assert np.nanmin(plotted_prediction_levels) < 0
    assert time_figure.axes[0].get_ylim()[0] < np.nanmin(plotted_prediction_levels)
    measurement_annotation = "\n".join(
        text.get_text() for text in time_figure.axes[0].texts
    )
    assert "All displayed predictions vs known" not in measurement_annotation
    assert "Nonzero targets within ±20%" not in measurement_annotation
    time_figure.set_size_inches(8, 4, forward=True)
    time_figure.axes[0].yaxis.label.set_fontsize(24)
    time_figure.tight_layout(pad=4 / 3)
    time_figure.canvas.draw()
    renderer = time_figure.canvas.get_renderer()
    ylabel_box = time_figure.axes[0].yaxis.label.get_window_extent(renderer)
    assert ylabel_box.x0 >= time_figure.bbox.x0
    time_legend = time_figure.axes[0].get_legend()
    legend_box = time_legend.get_window_extent(renderer)
    axes_box = time_figure.axes[0].get_window_extent(renderer)
    vline_label_boxes = [
        artist.get_window_extent(renderer)
        for artist in vline_label_artists
    ]
    assert time_legend._ncols == 1
    assert legend_box.x0 >= axes_box.x0
    assert legend_box.y1 <= axes_box.y1
    assert legend_box.y1 < min(box.y0 for box in vline_label_boxes)
    assert any(
        isinstance(collection, LineCollection)
        for collection in time_figure.axes[0].collections
    )
    concentration_errorbars = [
        collection
        for collection in time_figure.axes[0].collections
        if getattr(collection, "_swv_concentration_errorbar", False)
    ]
    assert concentration_errorbars
    assert all(
        collection.get_transform() == time_figure.axes[0].transData
        for collection in concentration_errorbars
    )
    assert "propagated ±1σ; below-LOD values allowed" in (
        time_figure.axes[0]._swv_uncertainty_note
    )
    assert "Error bars:" not in measurement_annotation
    assert "below-LOD points reported at LOD" not in measurement_annotation

    wavelet_summary = build_titration_langmuir_summary_table(
        results,
        metric="wavelet_energy",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
    )

    assert len(wavelet_summary) == 1
    assert np.isclose(wavelet_summary[0]["langmuir_kd"], kd, rtol=0.05)
    assert np.isclose(wavelet_summary[0]["langmuir_baseline"], 5.0)
    assert wavelet_summary[0]["limit_of_detection"] > 0

    for selected_baseline_mode in ("none", "preceding_buffer"):
        selected_summary = build_titration_langmuir_summary_table(
            results,
            metric="peak_current_selected",
            vlines=vlines,
            concentration_unit="uM",
            edge_trim_fraction=0,
            baseline_mode=selected_baseline_mode,
            included_step_labels=[
                "buffer_3",
                "40 uM",
                "buffer_4",
                "80 uM",
            ],
        )

        assert len(selected_summary) == 1
        assert np.isclose(selected_summary[0]["langmuir_baseline"], 7.0)
        assert selected_summary[0]["anchor_buffer_step_index"] == 5
        assert selected_summary[0]["langmuir_baseline_fixed"] is True

    langmuir_figure = plot_titration_langmuir(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
        show_lod=True,
        show_uloq=True,
    )
    assert langmuir_figure is not None
    assert langmuir_figure.axes[0].get_xscale() == "linear"
    assert _has_nonzero_vertical_errorbar(langmuir_figure.axes[0])
    langmuir_errorbar_caps = [
        line for line in langmuir_figure.axes[0].lines
        if bool(getattr(line, "_swv_langmuir_errorbar_cap", False))
    ]
    assert langmuir_errorbar_caps
    assert all(cap.get_markersize() >= 14.0 for cap in langmuir_errorbar_caps)
    assert any(line.get_label().endswith(" LOD") for line in langmuir_figure.axes[0].lines)
    assert any("ULOQ" in line.get_label() for line in langmuir_figure.axes[0].lines)

    hidden_uloq_snr = plot_titration_snr(
        peak_steps,
        concentration_unit="uM",
        fit_summary_rows=summary,
        show_uloq=False,
    )
    assert hidden_uloq_snr is not None
    assert not any("ULOQ" in line.get_label() for line in hidden_uloq_snr.axes[0].lines)

    hidden_uloq_accuracy = plot_titration_concentration_accuracy(
        accuracy_rows,
        concentration_unit="uM",
        show_uloq=False,
    )
    assert hidden_uloq_accuracy is not None
    assert not any("ULOQ" in line.get_label() for line in hidden_uloq_accuracy.axes[0].lines)

    hidden_uloq_langmuir = plot_titration_langmuir(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
        show_uloq=False,
    )
    assert hidden_uloq_langmuir is not None
    assert not any("ULOQ" in line.get_label() for line in hidden_uloq_langmuir.axes[0].lines)

    hidden_lod_snr = plot_titration_snr(
        peak_steps,
        concentration_unit="uM",
        fit_summary_rows=summary,
        show_lod=False,
    )
    assert hidden_lod_snr is not None
    assert not any("LOD" in line.get_label() for line in hidden_lod_snr.axes[0].lines)

    hidden_lod_langmuir = plot_titration_langmuir(
        results,
        metric="peak_current_selected",
        vlines=vlines,
        concentration_unit="uM",
        edge_trim_fraction=0,
        baseline_mode="preceding_buffer",
        show_lod=False,
    )
    assert hidden_lod_langmuir is not None
    assert not any("LOD" in line.get_label() for line in hidden_lod_langmuir.axes[0].lines)


def test_signal_off_langmuir_fit_and_inversion_are_supported():
    concentrations = [10.0, 20.0, 40.0, 80.0]
    baseline = 10.0
    amplitude = -6.0
    kd = 30.0
    labels = []
    plateau_values = []
    for concentration in concentrations:
        response = amplitude * concentration / (kd + concentration)
        labels.extend(["buffer", f"{concentration:g} uM"])
        plateau_values.extend([baseline, baseline + response])
    labels.append("end")
    vlines = [(1 + 3 * index, label) for index, label in enumerate(labels)]
    results = []
    for step_index, plateau in enumerate(plateau_values):
        start = 1 + 3 * step_index
        for scan_offset, deviation in enumerate((-0.1, 0.0, 0.1)):
            result = _result(1, start + scan_offset, plateau + deviation)
            result["wavelet_energy"] = plateau + deviation
            results.append(result)

    for metric in ("peak_current_selected", "wavelet_energy"):
        summary = build_titration_langmuir_summary_table(
            results,
            metric=metric,
            vlines=vlines,
            concentration_unit="uM",
            edge_trim_fraction=0,
            baseline_mode="preceding_buffer",
        )
        assert len(summary) == 1
        assert summary[0]["langmuir_response_direction"] == "signal-off"
        assert summary[0]["langmuir_amplitude"] < 0
        assert np.isclose(summary[0]["langmuir_amplitude"], amplitude, rtol=0.05)
        assert np.isclose(summary[0]["langmuir_kd"], kd, rtol=0.05)
        assert summary[0]["limit_of_detection"] > 0
        assert summary[0]["upper_limit_of_quantification"] > 0

        accuracy_rows = build_titration_measurement_accuracy_table(
            results,
            metric=metric,
            vlines=vlines,
            concentration_unit="uM",
            edge_trim_fraction=0,
            baseline_mode="preceding_buffer",
        )
        uncensored_rows = [
            row for row in accuracy_rows
            if not row["concentration_censored_at_lod"]
        ]
        assert uncensored_rows
        assert all(row["predicted_concentration"] > 0 for row in uncensored_rows)
        assert np.median([
            row["absolute_percent_error"] for row in uncensored_rows
        ]) < 10.0

        figure = plot_titration_langmuir(
            results,
            metric=metric,
            vlines=vlines,
            concentration_unit="uM",
            edge_trim_fraction=0,
            baseline_mode="preceding_buffer",
            show_fit_details=True,
        )
        assert figure is not None
        assert "signal-off" in "\n".join(
            text.get_text() for text in figure.axes[0].texts
        )
