import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
import pytest
from matplotlib import pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bo_session_viewer as viewer


@pytest.mark.parametrize("layout", ["Overlay selected channels", "Separate plots"])
def test_history_png_preserves_data_after_plotly_json_roundtrip(layout):
    history = pd.DataFrame({
        "iteration": [1, 2, 3, 4],
        "Q_ch2_min": [0.0, -0.8, 0.2, -0.4],
        "Q_ch2_max": [-0.9, 0.1, 0.5, 0.0],
    })
    source = viewer._plot_channel_trend(
        history, "Q_channel", {"2_min": "Q_ch2_min", "2_max": "Q_ch2_max"},
        ["2_min", "2_max"], layout,
    )
    restored = pio.from_json(source.to_json())
    settings = {
        "width": 1200, "height": 500, "margin": 50,
        "text_size": 10, "tick_size": 10, "title_size": 12,
        "perimeter_width": 0.8, "perimeter_color": "#222222",
        "show_grid": True, "show_legend": True,
    }
    assert viewer._figure_y_bounds(restored) == pytest.approx((-0.9, 0.5))
    original_export = viewer._history_plotly_to_matplotlib(source, settings)
    restored_export = viewer._history_plotly_to_matplotlib(restored, settings)
    try:
        assert len(restored_export.axes) == len(original_export.axes)
        for expected_axis, actual_axis in zip(original_export.axes, restored_export.axes):
            assert len(actual_axis.lines) == len(expected_axis.lines) > 0
            for expected, actual in zip(expected_axis.lines, actual_axis.lines):
                np.testing.assert_array_equal(actual.get_xdata(), expected.get_xdata())
                np.testing.assert_allclose(actual.get_ydata(), expected.get_ydata())
        images = [
            np.asarray(Image.open(BytesIO(viewer._matplotlib_png_bytes(
                figure, apply_global_style=False,
            ))))
            for figure in (original_export, restored_export)
        ]
        np.testing.assert_array_equal(*images)
    finally:
        plt.close(original_export)
        plt.close(restored_export)


def test_3d_download_does_not_require_server_png_export(monkeypatch):
    import plotly.graph_objects as go

    component_calls = []
    container = object()
    monkeypatch.setattr(viewer, "_sized_plot_container", lambda *_args: container)
    monkeypatch.setattr(viewer, "_apply_plotly_colorbar_height", lambda fig: fig)
    monkeypatch.setattr(viewer, "_apply_global_plot_style", lambda fig: fig)
    monkeypatch.setattr(
        viewer, "_render_camera_persistent_plotly",
        lambda *args, **kwargs: component_calls.append(kwargs),
    )

    def unavailable_png_export(*args, **kwargs):
        pytest.fail("Interactive 3D downloads must not require server PNG export")

    monkeypatch.setattr(viewer, "_plotly_png_bytes", unavailable_png_export)
    fig = go.Figure(go.Scatter3d(x=[1], y=[2], z=[3]))
    viewer._render_downloadable_plotly(
        container, fig, key="real_landscape", file_stem="real_landscape",
        width_percent=1200, export_width=1200, export_height=620,
        camera_storage_key="real_landscape_camera",
    )
    assert len(component_calls) == 1
    assert component_calls[0]["show_download"] is True
    assert component_calls[0]["file_stem"] == "real_landscape"
    assert component_calls[0]["export_width"] == 1200
    assert component_calls[0]["export_height"] == 620


@pytest.mark.parametrize("plot_type", ["scatter", "heatmap", "parcoords"])
@pytest.mark.parametrize("container_type", ["streamlit", "column"])
def test_2d_download_does_not_require_server_png_export(monkeypatch, plot_type, container_type):
    from contextlib import contextmanager
    import plotly.graph_objects as go

    component_calls = []
    active_containers = []

    @contextmanager
    def column():
        active_containers.append("column")
        try:
            yield
        finally:
            active_containers.pop()

    container = viewer.st if container_type == "streamlit" else column()
    monkeypatch.setattr(viewer, "_apply_plotly_colorbar_height", lambda fig: fig)
    monkeypatch.setattr(viewer, "_apply_global_plot_style", lambda fig: fig)
    def capture_component(**kwargs):
        assert active_containers == (["column"] if container_type == "column" else [])
        component_calls.append(kwargs)

    monkeypatch.setattr(viewer, "_plotly_camera_capture", capture_component)

    def unavailable_png_export(*args, **kwargs):
        pytest.fail("Browser downloads must not require server PNG export")

    monkeypatch.setattr(viewer, "_plotly_png_bytes", unavailable_png_export)
    traces = {
        "scatter": go.Scatter(x=[1, 2], y=[3, 4]),
        "heatmap": go.Heatmap(z=[[1, 2], [3, 4]]),
        "parcoords": go.Parcoords(dimensions=[
            dict(label="Frequency", values=[10, 20]),
            dict(label="Amplitude", values=[0.1, 0.2]),
        ]),
    }
    viewer._render_downloadable_plotly(
        container, go.Figure(traces[plot_type]),
        key=plot_type, file_stem=f"real_{plot_type}",
        width_percent=1200, export_width=1200, export_height=560,
    )
    assert len(component_calls) == 1
    args = component_calls[0]
    assert args["show_download"] is True
    assert args["show_cache_view"] is False
    assert args["camera_enabled"] is False
    assert args["figure"]["data"][0]["type"] == plot_type
    assert args["download_file_stem"] == f"real_{plot_type}"
    assert args["download_width"] == 1200
    assert args["download_height"] == 560


def test_series_styles_follow_selected_run_and_survive_legend_rename():
    import plotly.graph_objects as go

    first = go.Scatter(
        x=[1, 2], y=[1, 2], name="Simulation (run 1)",
        mode="lines+markers", line={"color": "blue", "width": 2},
        marker={"size": 5},
    )
    second = go.Scatter(
        x=[1, 2], y=[3, 4], name="Simulation (run 3)",
        mode="lines+markers", line={"color": "orange", "width": 2},
        marker={"size": 5},
    )
    figure = go.Figure([first, second])
    token = viewer._individual_plot_series(figure)[1][0]
    # Removing another selected run must not move the style to a different run.
    assert viewer._individual_plot_series(go.Figure([second]))[0][0] == token
    settings = {
        "width": 1000, "height": 600, "text_size": 10, "tick_size": 10,
        "margin": 50, "perimeter_width": 1, "perimeter_color": "black",
        "show_legend": True, "show_grid": True, "override_text": False,
        "line_scale": 2, "marker_scale": 2,
        "override_legend_text": True, "legend_labels": "First\nSecond",
        f"series_{token}_color": "purple",
        f"series_{token}_width": 7.0,
        f"series_{token}_marker_size": 12.0,
    }
    viewer._apply_individual_plotly_style(figure, settings)
    assert figure.data[0].line.color == "blue"
    assert figure.data[0].line.width == 4
    assert figure.data[0].marker.size == 10
    assert figure.data[1].name == "Second"
    assert figure.data[1].line.color == "#800080"
    assert figure.data[1].marker.color == "#800080"
    assert figure.data[1].line.width == 7
    assert figure.data[1].marker.size == 12
    exported = viewer._history_plotly_to_matplotlib(figure, settings)
    try:
        line = exported.axes[0].lines[1]
        assert line.get_color() == "#800080"
        assert line.get_linewidth() == 7
        assert line.get_markersize() == 12
    finally:
        plt.close(exported)


def test_plot_settings_form_applies_color_in_single_submission():
    from streamlit.testing.v1 import AppTest

    # Streamlit runs the script on a worker thread; macOS GUI backends cannot.
    plt.switch_backend("Agg")
    app = AppTest.from_string("""
import streamlit as st
import plotly.graph_objects as go
import bo_session_viewer as viewer
st.session_state["render_count"] = st.session_state.get("render_count", 0) + 1
figure = go.Figure(go.Scatter(
    x=[1, 2], y=[2, 3], name="Run 1", mode="lines+markers",
    line={"color": "blue"}, marker={"color": [1, 2], "coloraxis": "coloraxis"},
))
viewer._render_downloadable_plotly(
    st, figure, key="test_plot", file_stem="test", width_percent=800,
    individual_plot_settings=True,
)
st.session_state["rendered_color"] = figure.data[0].line.color
st.session_state["marker_coloraxis"] = figure.data[0].marker.coloraxis
""", default_timeout=30).run()
    assert not app.exception
    for widget_type in ("text_input", "text_area", "slider", "checkbox", "number_input", "selectbox"):
        for widget in getattr(app, widget_type):
            assert widget.proto.form_id == "test_plot_individual_plot_form"
    before = app.session_state["render_count"]
    color = next(widget for widget in app.text_input if widget.label == "Series color")
    color.set_value("rgb(255, 0, 0)")
    next(button for button in app.button if button.label == "Update settings").click()
    app.run()
    assert not app.exception
    assert app.session_state["render_count"] == before + 1
    assert app.session_state["rendered_color"] == "#ff0000"
    assert app.session_state["marker_coloraxis"] is None

    next(widget for widget in app.text_input if widget.label == "Series color").set_value("bad color")
    next(button for button in app.button if button.label == "Update settings").click()
    app.run()
    assert not app.exception
    assert any("Invalid color for Run 1" in warning.value for warning in app.warning)


def test_selected_runs_settings_have_their_own_fragment():
    from streamlit.testing.v1 import AppTest

    plt.switch_backend("Agg")
    app = AppTest.from_string("""
import streamlit as st
import plotly.graph_objects as go
from streamlit.runtime.scriptrunner import get_script_run_ctx
import bo_session_viewer as viewer

@st.fragment
def history():
    st.session_state["history_fragment"] = get_script_run_ctx().current_fragment_id
    source = go.Figure(go.Scatter(x=[1, 2], y=[2, 3], name="Run 1", line={"width": 2}))
    real_render = viewer._render_downloadable_plotly
    def capture(*args, **kwargs):
        st.session_state["overlay_fragment"] = get_script_run_ctx().current_fragment_id
        return real_render(*args, **kwargs)
    viewer._render_downloadable_plotly = capture
    try:
        viewer._render_selected_simulation_runs_plot(
            source, key="isolated_overlay", metric="Q_run",
            width_percent=800, observations=[],
        )
    finally:
        viewer._render_downloadable_plotly = real_render
    st.session_state["source_width"] = source.data[0].line.width
history()
""", default_timeout=30).run()
    assert not app.exception
    assert app.session_state["overlay_fragment"]
    assert app.session_state["overlay_fragment"] != app.session_state["history_fragment"]
    assert app.session_state["source_width"] == 2
    next(widget for widget in app.slider if widget.label == "Line thickness").set_value(2.0)
    next(button for button in app.button if button.label == "Update settings").click()
    app.run()
    assert not app.exception
    assert app.session_state["source_width"] == 2
