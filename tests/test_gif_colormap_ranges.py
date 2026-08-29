import sys
from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bo_session_viewer import _fit_y_axis_to_figure, _lock_gif_colormap_ranges


def test_locks_matplotlib_heatmap_range_across_frames():
    figures = []
    for values in ([[0.0, 1.0], [2.0, 3.0]], [[-5.0, 0.0], [5.0, 10.0]]):
        figure, axis = plt.subplots()
        image = axis.imshow(values, cmap="viridis")
        figure.colorbar(image, ax=axis, label="Acquisition value")
        figures.append(figure)

    _lock_gif_colormap_ranges(figures)

    assert [figure.axes[0].images[0].get_clim() for figure in figures] == [
        (-5.0, 10.0),
        (-5.0, 10.0),
    ]
    for figure in figures:
        plt.close(figure)


def test_locks_plotly_heatmap_range_across_frames():
    figures = [
        go.Figure(go.Heatmap(
            z=[[0.0, 2.0]],
            colorscale="Viridis",
            colorbar={"title": "Acquisition value"},
        )),
        go.Figure(go.Heatmap(
            z=[[-4.0, 8.0]],
            colorscale="Viridis",
            colorbar={"title": "Acquisition value"},
        )),
    ]

    _lock_gif_colormap_ranges(figures)

    assert [(figure.data[0].zmin, figure.data[0].zmax) for figure in figures] == [
        (-4.0, 8.0),
        (-4.0, 8.0),
    ]


def test_keeps_distinct_plotly_colorbars_independently_scaled():
    figures = []
    for acquisition, iteration in (([1.0, 2.0], [1.0, 2.0]), ([-3.0, 9.0], [1.0, 7.0])):
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="markers",
            marker={
                "color": acquisition,
                "colorscale": "Viridis",
                "colorbar": {"title": "Acquisition value"},
            },
        ))
        figure.add_trace(go.Scatter(
            x=[0, 1],
            y=[1, 0],
            mode="markers",
            marker={
                "color": iteration,
                "colorscale": "Viridis",
                "colorbar": {"title": "Iteration"},
            },
        ))
        figures.append(figure)

    _lock_gif_colormap_ranges(figures)

    assert [(figure.data[0].marker.cmin, figure.data[0].marker.cmax) for figure in figures] == [
        (-3.0, 9.0),
        (-3.0, 9.0),
    ]
    assert [(figure.data[1].marker.cmin, figure.data[1].marker.cmax) for figure in figures] == [
        (1.0, 7.0),
        (1.0, 7.0),
    ]


def test_locks_shared_plotly_coloraxis_across_frames():
    figures = []
    for values in ([0.0, 4.0], [-6.0, 2.0]):
        figure = go.Figure(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="markers",
            marker={"color": values, "coloraxis": "coloraxis"},
        ))
        figure.update_layout(coloraxis={
            "colorscale": "Plasma",
            "colorbar": {"title": "Measured value"},
        })
        figures.append(figure)

    _lock_gif_colormap_ranges(figures)

    assert [(figure.layout.coloraxis.cmin, figure.layout.coloraxis.cmax) for figure in figures] == [
        (-6.0, 4.0),
        (-6.0, 4.0),
    ]


def test_fits_each_independent_figure_y_axis_to_its_own_values():
    low_figure = go.Figure(go.Scatter(y=[0.0, 10.0]))
    high_figure = go.Figure(go.Scatter(y=[100.0, 101.0]))

    _fit_y_axis_to_figure(low_figure)
    _fit_y_axis_to_figure(high_figure)

    assert tuple(low_figure.layout.yaxis.range) == (-0.5, 10.5)
    assert tuple(high_figure.layout.yaxis.range) == (99.95, 101.05)
    assert low_figure.layout.yaxis.matches is None
    assert high_figure.layout.yaxis.matches is None
