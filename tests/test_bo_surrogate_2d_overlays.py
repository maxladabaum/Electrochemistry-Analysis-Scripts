from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer


def test_plotly_axis_labels_and_ticks_accept_independent_text_sizes():
    axis_style = viewer._plotly_axis_font_update(18.5, tick_size=11.0)

    assert axis_style["title"]["font"]["size"] == 19.43
    assert axis_style["tickfont"]["size"] == 11.0


def test_plotly_3d_tick_spacing_uses_outward_tick_length():
    assert viewer._plotly_3d_tick_spacing_update(14) == {
        "ticks": "outside",
        "ticklen": 14,
    }
    assert viewer._plotly_3d_tick_spacing_update(0) == {
        "ticks": "",
        "ticklen": 0,
    }


def test_surrogate_2d_colorbar_label_is_above_bar():
    fig, ax = plt.subplots()
    image = ax.imshow(np.arange(4).reshape(2, 2))
    colorbar = fig.colorbar(image, ax=ax)

    viewer._set_surrogate_2d_colorbar_title(
        colorbar,
        "Predicted mean Q",
        fontsize=8,
    )

    assert colorbar.ax.get_title() == "Predicted mean Q"
    assert colorbar.ax.get_ylabel() == ""
    assert colorbar.ax._bo_colorbar_label_above is True
    plt.close(fig)


def test_pyplot_renderer_returns_the_exact_displayed_preview(monkeypatch):
    class FakeContainer:
        def __init__(self):
            self.image_data = None
            self.image_width = None

        def empty(self):
            return self

        def image(self, data, *, width=None, **_kwargs):
            self.image_data = data
            self.image_width = width

        def markdown(self, *_args, **_kwargs):
            return None

    container = FakeContainer()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([0.0, 1.0], [0.0, 1.0])
    styled_widths = []
    monkeypatch.setattr(
        viewer,
        "_apply_global_plot_style",
        lambda figure: styled_widths.append(float(figure.get_size_inches()[0])),
    )

    preview = viewer._render_downloadable_pyplot(
        container,
        fig,
        key="test-preview",
        file_stem="test-preview",
        width_percent=1000,
    )

    assert styled_widths == [10.0]
    assert preview == container.image_data
    assert container.image_width == 1000


def test_surrogate_2d_axes_follow_selected_canvas_aspect(monkeypatch):
    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    mesh = ax.pcolormesh(np.arange(3), np.arange(3), np.arange(4).reshape(2, 2))
    fig.colorbar(mesh, ax=ax)
    fig._bo_follow_canvas_aspect = True
    monkeypatch.setattr(viewer, "_plot_height_px", lambda _kind: 1000)

    viewer._apply_matplotlib_global_plot_style(fig)

    assert viewer._matplotlib_plot_kind(fig) == "2d"
    assert fig.get_size_inches().tolist() == [10.0, 10.0]
    assert ax.get_box_aspect() == 1.0
    fig.canvas.draw()
    rendered_axes = ax.get_window_extent()
    assert rendered_axes.width == pytest.approx(rendered_axes.height)
    plt.close(fig)


def test_surrogate_shared_color_range_combines_slices_and_ignores_nonfinite():
    first_slice = pd.DataFrame({"predicted_mean_Q": [-3.0, np.nan, 1.0]})
    second_slice = pd.DataFrame({"predicted_mean_Q": [2.0, np.inf, 8.0]})

    color_range = viewer._surrogate_shared_color_range(
        [first_slice, second_slice],
        "predicted_mean_Q",
    )

    assert color_range == (-3.0, 8.0)


def test_surrogate_2d_control_style_uses_locked_color_range(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_surrogate_slice_base_params",
        lambda _session, _iteration: {},
    )
    monkeypatch.setattr(
        viewer,
        "_surrogate_regular_2d_grid",
        lambda *_args, **_kwargs: (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([[-1.0, 0.0], [1.0, 2.0]]),
        ),
    )
    monkeypatch.setattr(viewer, "_observed_points", lambda *_args, **_kwargs: [])
    fig, ax = plt.subplots()

    mesh = viewer._plot_surrogate_2d_control_style(
        ax,
        {"observations": []},
        pd.DataFrame(),
        1,
        "predicted_mean_Q",
        "frequency",
        "amplitude",
        pd.DataFrame(),
        color_range=(-5.0, 10.0),
    )

    assert mesh.norm.vmin == -5.0
    assert mesh.norm.vmax == 10.0
    plt.close(fig)


def test_automation_surrogate_2d_style_uses_locked_color_range(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_surrogate_parameter_context",
        lambda _session, _iteration: "test context",
    )
    monkeypatch.setattr(
        viewer,
        "_surrogate_slice_base_params",
        lambda _session, _iteration: {},
    )
    monkeypatch.setattr(
        viewer,
        "_surrogate_regular_2d_grid",
        lambda *_args, **_kwargs: (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([[-1.0, 0.0], [1.0, 2.0]]),
        ),
    )
    monkeypatch.setattr(viewer, "_observed_points", lambda *_args, **_kwargs: [])
    frame = pd.DataFrame({
        "frequency": [0.0, 1.0],
        "amplitude": [0.0, 1.0],
        "predicted_mean_Q": [-1.0, 2.0],
    })

    fig = viewer._plot_surrogate(
        {"observations": []},
        frame,
        1,
        "predicted_mean_Q",
        "2D map",
        "frequency",
        "amplitude",
        None,
        surrogate_2d_style="Automation smooth heatmap",
        color_range=(-4.0, 12.0),
    )

    mesh = fig.axes[0].collections[0]
    assert mesh.norm.vmin == -4.0
    assert mesh.norm.vmax == 12.0
    plt.close(fig)


def test_slice_perimeter_survives_global_style_and_skips_colorbar(monkeypatch):
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(np.arange(3), np.arange(3), np.arange(4).reshape(2, 2))
    colorbar = fig.colorbar(mesh, ax=ax)
    slice_color = (0.2, 0.4, 0.8)
    viewer._apply_slice_perimeter_style(
        fig,
        slice_color,
        thickness=4,
    )
    colorbar_spines_before = {
        name: (spine.get_linewidth(), spine.get_edgecolor())
        for name, spine in colorbar.ax.spines.items()
    }
    monkeypatch.setattr(viewer, "_plot_perimeter_color", lambda: "#ff0000")
    monkeypatch.setattr(viewer, "_plot_perimeter_width", lambda: 1.0)

    viewer._apply_matplotlib_global_plot_style(fig)

    for spine in ax.spines.values():
        assert spine.get_edgecolor()[:3] == pytest.approx(slice_color)
        assert spine.get_linewidth() == pytest.approx(4.0)
    assert not hasattr(colorbar.ax, "_bo_slice_perimeter_color")
    assert {
        name: (spine.get_linewidth(), spine.get_edgecolor())
        for name, spine in colorbar.ax.spines.items()
    } == colorbar_spines_before
    plt.close(fig)


def test_voltage_slice_plane_edges_are_offset_inward_by_point_zero_one_mv():
    assert viewer._slice_plane_display_value(
        "amplitude",
        0.05,
        (0.05, 0.20),
    ) == pytest.approx(0.05001)
    assert viewer._slice_plane_display_value(
        "step_potential",
        0.01,
        (0.001, 0.01),
    ) == pytest.approx(0.00999)
    assert viewer._slice_plane_display_value(
        "step_potential",
        10.0,
        (1.0, 10.0),
    ) == pytest.approx(9.99)
    assert viewer._slice_plane_display_value(
        "frequency",
        500.0,
        (100.0, 500.0),
    ) == pytest.approx(500.0)


def test_highlighted_slice_sweep_excludes_single_slice_value():
    assert viewer._highlighted_slice_values(
        0.004,
        [0.001, 0.007, 0.010],
    ) == [0.001, 0.007, 0.010]
    assert viewer._highlighted_slice_values(0.004, []) == [0.004]

    figure = viewer.go.Figure()
    points = pd.DataFrame({
        "frequency": [100.0, 500.0],
        "amplitude": [0.05, 0.20],
        "step_potential": [0.001, 0.01],
    })
    viewer._add_plotly_slice_plane(
        figure,
        points,
        x_axis="amplitude",
        y_axis="frequency",
        z_axis="step_potential",
        slice_axis="step_potential",
        slice_value=0.004,
        slice_sweep_values=[0.001, 0.01],
    )

    outlines = [
        trace
        for trace in figure.data
        if viewer._plotly_trace_role(trace) == "highlighted_slice_outline"
    ]
    assert len(outlines) == 2
    assert list(outlines[0].z) == pytest.approx([0.00101] * 5)
    assert list(outlines[1].z) == pytest.approx([0.00999] * 5)


def test_real_data_slice_plane_uses_offset_but_keeps_requested_label(monkeypatch):
    figure = viewer.go.Figure()
    points = pd.DataFrame({
        "frequency": [100.0, 500.0],
        "amplitude": [0.05, 0.20],
        "step_potential": [0.001, 0.01],
    })

    viewer._add_plotly_slice_plane(
        figure,
        points,
        x_axis="amplitude",
        y_axis="frequency",
        z_axis="step_potential",
        slice_axis="step_potential",
        slice_value=0.01,
    )

    outline = figure.data[-1]
    assert list(outline.z) == pytest.approx([0.00999] * 5)
    assert "0.01" in outline.hovertemplate
    slice_color = outline.line.color
    monkeypatch.setattr(viewer, "_plot_perimeter_color", lambda: "#000000")
    monkeypatch.setattr(viewer, "_plot_perimeter_width", lambda: 5.0)

    viewer._apply_plotly_global_plot_style(figure)

    assert outline.line.color == slice_color
    assert outline.line.width == pytest.approx(5.0)


def test_plotly_2d_slice_perimeter_keeps_slice_color(monkeypatch):
    figure = viewer.go.Figure(
        viewer.go.Heatmap(z=[[0.0, 1.0], [1.0, 0.0]])
    )
    slice_color = "rgba(80,120,220,1)"
    viewer._apply_plotly_slice_perimeter_style(
        figure,
        slice_color,
        thickness=3,
    )
    monkeypatch.setattr(viewer, "_plot_perimeter_color", lambda: "#000000")
    monkeypatch.setattr(viewer, "_plot_perimeter_width", lambda: 6.0)

    viewer._apply_plotly_global_plot_style(figure)

    perimeter = next(
        shape
        for shape in figure.layout.shapes
        if shape.name == "bo_highlighted_slice_perimeter"
    )
    assert perimeter.line.color == slice_color
    assert perimeter.line.width == pytest.approx(6.0)


def test_surrogate_edge_slice_plane_uses_inward_offset(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_surrogate_parameter_context",
        lambda _session, _iteration: "test context",
    )
    monkeypatch.setattr(viewer, "_observed_points", lambda *_args, **_kwargs: [])
    frame = pd.DataFrame({
        "amplitude": [0.05, 0.20],
        "frequency": [100.0, 500.0],
        "step_potential": [1.0, 10.0],
        "predicted_mean_Q": [-1.0, 2.0],
    })

    figure = viewer._plot_surrogate(
        {"observations": []},
        frame,
        1,
        "predicted_mean_Q",
        "3D tensor",
        "amplitude",
        "frequency",
        "step_potential",
        slice_axis="step_potential",
        slice_value=10.0,
    )

    outline = next(
        trace
        for trace in figure.data
        if viewer._plotly_trace_role(trace) == "highlighted_slice_outline"
    )
    assert list(outline.z) == pytest.approx([9.99] * 5)
    assert "10" in outline.hovertemplate


def test_current_iteration_marker_follows_show_observed_points(monkeypatch):
    monkeypatch.setattr(
        viewer,
        "_surrogate_slice_base_params",
        lambda _session, _iteration: {},
    )
    monkeypatch.setattr(
        viewer,
        "_surrogate_regular_2d_grid",
        lambda *_args, **_kwargs: (
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([[0.0, 0.5], [0.5, 1.0]]),
        ),
    )
    monkeypatch.setattr(
        viewer,
        "_observed_points",
        lambda *_args, **_kwargs: [
            {"iteration": 2, "params": {"frequency": 0.5, "amplitude": 0.5}}
        ],
    )

    for show_observed_points, expected_marker_count in ((False, 0), (True, 1)):
        fig, ax = plt.subplots()
        viewer._plot_surrogate_2d_control_style(
            ax,
            {},
            pd.DataFrame(),
            2,
            "predicted_mean_Q",
            "frequency",
            "amplitude",
            pd.DataFrame(),
            show_iteration_path=False,
            show_observed_points=show_observed_points,
        )

        current_markers = [
            collection
            for collection in ax.collections
            if collection.get_label() == "current iteration 2"
        ]
        assert len(current_markers) == expected_marker_count
        plt.close(fig)
