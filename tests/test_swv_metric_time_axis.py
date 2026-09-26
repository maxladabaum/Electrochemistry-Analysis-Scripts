"""Check alignment of measurement numbers and filename times on SWV metrics."""
from datetime import datetime, timedelta
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.plotting import plot_metric_vs_scan


def test_dual_axis_uses_selected_channels_range_and_median_times():
    start = datetime(2026, 9, 25, 23, 55)
    rows = [
        dict(channel=ch, scan_number=scan, peak=scan,
             measurement_time=start + timedelta(minutes=minutes + offset))
        for ch, offset in [(1, 0), (2, 2), (3, 300)]
        for scan, minutes in [(1, 0), (2, 0), (3, 8), (4, 60)]
    ]
    fig = plot_metric_vs_scan(
        rows, metric="peak", channels=[1, 2], scan_range=(2, 3),
        show_time_axis=True, xlabel="SWV Measurement Number",
        response_directions={1: "signal-on", 2: "signal-off"},
    )
    try:
        ax = fig.axes[0]
        top = ax.child_axes[0]
        fig.canvas.draw()
        np.testing.assert_allclose(top.get_xticks(), [2, 3])
        assert [tick.get_text() for tick in top.get_xticklabels()] == [
            "2026-09-25\n23:56", "2026-09-26\n00:04",
        ]
        np.testing.assert_allclose(top.get_xlim(), ax.get_xlim())
        assert ax.get_xlabel() == "SWV Measurement Number"
        assert top.get_xlabel() == "Measurement time"
        output = io.BytesIO()
        fig.savefig(output, format="pdf")
        assert output.getvalue().startswith(b"%PDF")
    finally:
        plt.close(fig)


def test_missing_timestamps_keep_metric_points_and_default_has_one_x_axis():
    rows = [dict(channel=1, scan_number=i, peak=i) for i in range(1, 4)]
    for enabled in (False, True):
        fig = plot_metric_vs_scan(rows, metric="peak", show_time_axis=enabled)
        try:
            assert not fig.axes[0].child_axes
            np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [1, 2, 3])
        finally:
            plt.close(fig)

    rows[1]["measurement_time"] = datetime(2026, 9, 25, 12)
    fig = plot_metric_vs_scan(rows, metric="peak", show_time_axis=True)
    try:
        np.testing.assert_allclose(fig.axes[0].child_axes[0].get_xticks(), [2])
        np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), [1, 2, 3])
    finally:
        plt.close(fig)


def test_measurement_annotations_follow_each_channels_time_and_plotted_value():
    import matplotlib.dates as mdates

    start = datetime(2026, 9, 25, 12)
    rows = [
        dict(channel=ch, scan_number=scan, peak=baseline + scan,
             measurement_time=start + timedelta(minutes=scan * rate))
        for ch, rate, baseline in [(1, 1, 10), (2, 5, 20)]
        for scan in range(1, 5)
    ]
    rows.append(dict(channel=1, scan_number=5, peak=15, measurement_time=None))
    fig = plot_metric_vs_scan(
        rows, metric="peak", x_key="measurement_time",
        annotate_measurement_numbers=True, annotation_every=2,
        offset_to_response_baseline=True, response_baselines={1: 10, 2: 20},
    )
    try:
        ax = fig.axes[0]
        labels = [text for text in ax.texts if text.get_text() in {"1", "3"}]
        assert [text.get_text() for text in labels] == ["1", "3", "1", "3"]
        for label, (scan, rate) in zip(labels, [(1, 1), (3, 1), (1, 5), (3, 5)]):
            np.testing.assert_allclose(label.xy, [
                mdates.date2num(start + timedelta(minutes=scan * rate)), scan,
            ])
        assert labels[0].get_color() == ax.lines[0].get_color()
        assert labels[2].get_color() == ax.lines[1].get_color()
        assert not ax.child_axes
        output = io.BytesIO()
        fig.savefig(output, format="pdf")
        assert output.getvalue().startswith(b"%PDF")
    finally:
        plt.close(fig)


def test_legend_rates_are_channel_specific_and_handle_missing_or_equal_times():
    start = datetime(2026, 9, 25, 12)
    rows = [
        dict(channel=ch, scan_number=scan, peak=scan,
             measurement_time=start + timedelta(minutes=scan * rate))
        for ch, rate in [(1, 1), (2, 5), (3, 0)]
        for scan in range(1, 5)
    ]
    rows.append(dict(channel=4, scan_number=1, peak=1))
    fig = plot_metric_vs_scan(rows, metric="peak", show_measurement_rate=True)
    try:
        labels = fig.axes[0].get_legend_handles_labels()[1]
        assert labels[0].endswith("1.00 scans/min (avg)")
        assert labels[1].endswith("0.20 scans/min (avg)")
        assert labels[2].endswith("rate unavailable")
        assert labels[3].endswith("rate unavailable")
    finally:
        plt.close(fig)


def test_elapsed_minutes_share_origin_and_align_annotations_and_boundaries():
    start = datetime(2026, 9, 25, 23, 55)
    rows = [
        dict(channel=ch, scan_number=scan, peak=scan,
             measurement_time=start + timedelta(minutes=offset + (scan - 1) * 5))
        for ch, offset in [(1, 0), (2, 2), (3, -60)] for scan in [1, 2]
    ]
    fig = plot_metric_vs_scan(
        rows, metric="peak", channels=[1, 2], x_key="measurement_time",
        elapsed_time=True, annotate_measurement_numbers=True,
        vlines=[(2, "boundary")],
    )
    try:
        ax = fig.axes[0]
        np.testing.assert_allclose(ax.lines[0].get_xdata(), [0, 5], atol=1e-7)
        np.testing.assert_allclose(ax.lines[1].get_xdata(), [2, 7], atol=1e-7)
        np.testing.assert_allclose(ax.lines[2].get_xdata(), [6, 6], atol=1e-7)
        np.testing.assert_allclose([text.xy[0] for text in ax.texts[:4]], [0, 5, 2, 7], atol=1e-7)
        assert ax.get_xlabel() == "Elapsed time (min)"
        assert ax.get_xlim()[0] == 0
        fig.canvas.draw()
    finally:
        plt.close(fig)

    fig = plot_metric_vs_scan(rows, metric="peak", channels=[1], show_time_axis=True, elapsed_time=True)
    try:
        top = fig.axes[0].child_axes[0]
        assert top.get_xlabel() == "Elapsed time (min)"
        assert [text.get_text() for text in top.get_xticklabels()] == ["0", "5"]
    finally:
        plt.close(fig)


def test_annotation_intervals_are_independent_per_channel():
    start = datetime(2026, 9, 25, 12)
    rows = [
        dict(channel=ch, scan_number=scan, peak=scan + ch,
             measurement_time=start + timedelta(minutes=scan * ch))
        for ch in [2, 6, 7] for scan in range(1, 7)
    ]
    fig = plot_metric_vs_scan(
        rows, metric="peak", x_key="measurement_time", elapsed_time=True,
        annotate_measurement_numbers=True, annotation_every=5,
        annotation_every_by_channel={2: 2, 6: 3},
    )
    try:
        ax = fig.axes[0]
        assert [label.get_text() for label in ax.texts] == ["1", "3", "5", "1", "4", "1", "6"]
        for label in ax.texts:
            assert label.get_position() == (0, 10)
            assert label.get_zorder() > max(line.get_zorder() for line in ax.lines)
        fig.canvas.draw()
    finally:
        plt.close(fig)
