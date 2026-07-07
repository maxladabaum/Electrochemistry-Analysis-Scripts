"""Read-only Streamlit viewer for experiment_automation BO session folders."""

from __future__ import annotations

import json
import itertools
import base64
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from scipy.special import ndtr
import streamlit as st
import streamlit.components.v1 as components

from core.analysis import analyze_swv_arrays
from core.io import load_swv_csv


PARAMETERS = (
    "begin_potential", "end_potential", "step_potential", "amplitude",
    "frequency", "conditioning_potential", "conditioning_time",
)
SURROGATE_VALUES = ("predicted_mean_Q", "predicted_std_Q", "acquisition_value")
SURROGATE_VALUE_LABELS = {
    "predicted_mean_Q": "Predicted Q",
    "predicted_std_Q": "Predicted Q std.",
    "acquisition_value": "Acquisition value",
}
OBSERVED_PATH_COLORS = ("#e31a1c", "#000000")
OBSERVED_PATH_CMAP = LinearSegmentedColormap.from_list(
    "observed_iteration", OBSERVED_PATH_COLORS
)
PAIRED_TREND_METRICS = {
    "Peak height (µA)": ("channel", "mean_peak_current_uA", "median_peak_current_uA"),
    "Raw SNR": ("channel", "snr_unadjusted", "snr"),
    "Peak shape score": ("channel", "peak_shape_score", None),
    "Baseline stability score": ("channel", "baseline_stability_score", None),
    "Replicate consistency score": ("channel", "replicate_consistency_score", None),
    "Success score": ("channel", "success_score", None),
    "Classic Q": ("component", "classic_Q", None),
    "SNR score": ("component", "snr_score", None),
}
REAL_DATA_METRICS = {
    "Peak height (µA)": ("channel", "mean_peak_current_uA", "median_peak_current_uA"),
    "Raw SNR": ("channel", "snr_unadjusted", "snr"),
    "Peak shape score": ("channel", "peak_shape_score", None),
    "Baseline stability score": ("channel", "baseline_stability_score", None),
    "Replicate consistency score": ("channel", "replicate_consistency_score", None),
    "Success score": ("channel", "success_score", None),
    "Classic Q": ("component", "classic_Q", None),
    "SNR score": ("component", "snr_score", None),
}


def _pick_session_folder() -> str:
    """Open the platform-native folder picker outside the Streamlit thread."""
    if sys.platform == "darwin":
        script = 'POSIX path of (choose folder with prompt "Select BO session folder")'
        return subprocess.check_output(["osascript", "-e", script], text=True).strip()
    if sys.platform.startswith("win"):
        code = (
            "import tkinter as tk\n"
            "from tkinter import filedialog\n"
            "root=tk.Tk()\n"
            "root.withdraw()\n"
            "root.wm_attributes('-topmost', True)\n"
            "p=filedialog.askdirectory(title='Select BO session folder')\n"
            "root.destroy()\n"
            "print(p or '')\n"
        )
        return subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    return ""


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def load_bo_session(folder: str | Path) -> dict:
    """Load a BO record folder without depending on experiment_automation."""
    root = Path(folder).expanduser().resolve()
    state_path = root / "bo_state.json"
    history_path = root / "history.csv"
    if not root.is_dir():
        raise ValueError(f"BO session folder does not exist: {root}")
    if not state_path.is_file():
        raise ValueError(f"Not a BO session: {state_path.name} is missing")
    state = _read_json(state_path)
    observations = state.get("observations") or []
    if not isinstance(observations, list):
        raise ValueError("bo_state.json observations must be a list")
    history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
    config_path = root / "bo_config_snapshot.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    return {
        "root": root,
        "state": state,
        "config": config,
        "observations": observations,
        "history": history,
    }


def _session_channel_groups(session: dict) -> list[dict]:
    """Return the persisted groups that have observations in this session."""
    observations = session["observations"]
    observed_ids = {
        int(obs.get("group_id", 1))
        for obs in observations
        if obs.get("group_id", 1) is not None
    }
    configured = (
        session["state"].get("channel_groups")
        or session["config"].get("channel_groups")
        or []
    )
    groups = []
    for raw in configured:
        try:
            group_id = int(raw.get("id", 1))
        except (AttributeError, TypeError, ValueError):
            continue
        if observed_ids and group_id not in observed_ids:
            continue
        groups.append({
            "id": group_id,
            "name": str(raw.get("name") or f"Group {group_id}"),
            "channels": list(raw.get("channels") or []),
        })
    known_ids = {group["id"] for group in groups}
    for obs in observations:
        group_id = int(obs.get("group_id", 1))
        if group_id not in known_ids:
            groups.append({
                "id": group_id,
                "name": str(obs.get("group_name") or f"Group {group_id}"),
                "channels": list(obs.get("channels") or []),
            })
            known_ids.add(group_id)
    return sorted(groups, key=lambda group: group["id"])


def _session_for_channel_group(session: dict, group_id: int) -> dict:
    """Create a group-scoped view while leaving the loaded session untouched."""
    scoped = dict(session)
    scoped["observations"] = [
        obs for obs in session["observations"]
        if int(obs.get("group_id", 1)) == int(group_id)
    ]
    history = session["history"]
    if not history.empty and "group_id" in history.columns:
        ids = pd.to_numeric(history["group_id"], errors="coerce")
        scoped["history"] = history.loc[ids == int(group_id)].reset_index(drop=True)
    scoped["selected_group_id"] = int(group_id)
    return scoped


def _channel_group_optimization_metadata(
    session: dict,
    groups: list[dict],
) -> list[dict]:
    """Merge global acquisition defaults with persisted per-group overrides."""
    config = session.get("config") or {}
    acquisition = config.get("acquisition") or {}
    def groups_by_id(raw_groups) -> dict[int, dict]:
        result = {}
        for raw_group in raw_groups or []:
            try:
                result[int(raw_group["id"])] = raw_group
            except (KeyError, TypeError, ValueError):
                continue
        return result

    config_groups = groups_by_id(config.get("channel_groups"))
    state_groups = groups_by_id(
        (session.get("state") or {}).get("channel_groups")
    )
    metadata = []
    for group in groups:
        group_id = int(group["id"])
        settings = {
            "exploration": acquisition.get(
                "exploration", config.get("exploration")
            ),
            "n_initial_points": config.get("n_initial_points"),
            "candidate_pool_size": acquisition.get(
                "candidate_pool_size", config.get("candidate_pool_size")
            ),
            "local_candidate_pool_size": acquisition.get(
                "local_candidate_pool_size",
                config.get("local_candidate_pool_size"),
            ),
            "initial_point_mode": acquisition.get(
                "initial_point_mode", config.get("initial_point_mode")
            ),
            "use_gp": acquisition.get("use_gp", config.get("use_gp")),
            "gp_optimizer_restarts": acquisition.get(
                "gp_optimizer_restarts", config.get("gp_optimizer_restarts")
            ),
            "gp_falloff_fractions": acquisition.get("gp_falloff_fractions") or {},
            "gp_length_scales": acquisition.get("gp_length_scales") or {},
            "initial_parameters": config.get("initial_parameters") or {},
        }
        settings.update(config_groups.get(group_id, {}))
        settings.update(state_groups.get(group_id, {}))
        exploration = settings.get("exploration")
        try:
            exploration = float(exploration)
            exploitation = (
                1.0 - exploration if 0.0 <= exploration <= 1.0 else None
            )
        except (TypeError, ValueError):
            exploitation = None
        metadata.append({
            "id": group_id,
            "name": str(settings.get("name") or group["name"]),
            "channels": list(settings.get("channels") or group["channels"]),
            "exploration": exploration,
            "exploitation": exploitation,
            "n_initial_points": settings.get("n_initial_points"),
            "candidate_pool_size": settings.get("candidate_pool_size"),
            "local_candidate_pool_size": settings.get(
                "local_candidate_pool_size"
            ),
            "initial_point_mode": settings.get("initial_point_mode"),
            "use_gp": settings.get("use_gp"),
            "gp_optimizer_restarts": settings.get("gp_optimizer_restarts"),
            "gp_falloff_fractions": dict(
                settings.get("gp_falloff_fractions") or {}
            ),
            "gp_length_scales": dict(settings.get("gp_length_scales") or {}),
            "initial_parameters": dict(settings.get("initial_parameters") or {}),
        })
    return metadata


def _observation_table(session: dict) -> pd.DataFrame:
    history = session["history"].copy()
    rows = []
    for obs in session["observations"]:
        row = {
            "iteration": obs.get("iteration"),
            "group_id": obs.get("group_id", 1),
            "group_name": obs.get("group_name", "Group 1"),
            "channels": ",".join(str(channel) for channel in obs.get("channels", [])),
            "Q_run": obs.get("Q_run"),
            "objective": obs.get("objective"),
            "completed_at": obs.get("completed_at"),
        }
        row.update(obs.get("params") or {})
        for key, value in (obs.get("quality") or {}).items():
            if np.isscalar(value) and not isinstance(value, (str, bytes)):
                row[key] = value
        for source_name, prefix in (
            ("channel_metrics", ""),
            ("buffer_channel_metrics", "buffer_"),
            ("target_channel_metrics", "target_"),
        ):
            for channel, metrics in (obs.get(source_name) or {}).items():
                for key, value in (metrics or {}).items():
                    if np.isscalar(value) and not isinstance(value, (str, bytes)):
                        row[f"ch{channel}_{prefix}{key}"] = value
        components = (obs.get("quality") or {}).get("channel_components") or {}
        for channel, metrics in components.items():
            for key, value in (metrics or {}).items():
                if np.isscalar(value) and not isinstance(value, (str, bytes)):
                    row.setdefault(f"ch{channel}_{key}", value)
        rows.append(row)
    observation_frame = pd.DataFrame(rows)
    if history.empty:
        return observation_frame
    if observation_frame.empty:
        return history

    missing_columns = [
        column for column in observation_frame.columns
        if column not in history.columns
    ]
    if missing_columns:
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    np.nan,
                    index=history.index,
                    columns=missing_columns,
                ),
            ],
            axis=1,
        )
    history_group = pd.to_numeric(
        history.get("group_id", pd.Series(1, index=history.index)),
        errors="coerce",
    ).fillna(1).astype(int)
    history_iteration = pd.to_numeric(history["iteration"], errors="coerce")
    for _, row in observation_frame.iterrows():
        mask = (
            (history_group == int(row.get("group_id", 1)))
            & (history_iteration == int(row["iteration"]))
        )
        matching = history.index[mask]
        if matching.empty:
            continue
        index = matching[0]
        for column, value in row.items():
            if pd.notna(value):
                history.at[index, column] = value
    return history


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "iteration", "group_id", "paired_cycle", "paired_batch_index",
        "buffer_trace_number", "target_trace_number",
    }
    return [
        column for column in frame.columns
        if (
            column not in excluded
            and pd.to_numeric(frame[column], errors="coerce").nunique(dropna=True) > 1
        )
    ]


def _channel_metric_columns(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return metric -> channel -> history-column mappings."""
    metrics: dict[str, dict[str, str]] = {}
    for column in frame.columns:
        q_match = re.fullmatch(r"Q_ch(\d+)", str(column), re.IGNORECASE)
        component_match = re.fullmatch(r"ch(\d+)_(.+)", str(column), re.IGNORECASE)
        if q_match:
            channel, metric = q_match.group(1), "Q_channel"
        elif component_match:
            channel, metric = component_match.group(1), component_match.group(2)
        else:
            continue
        if pd.to_numeric(frame[column], errors="coerce").nunique(dropna=True) > 1:
            metrics.setdefault(metric, {})[channel] = column
    return metrics


def _metric_label(metric: str) -> str:
    replacements = {
        "Q_channel": "Channel Q",
        "Q_run": "Q run",
        "mean_peak_current_uA": "Peak height (µA)",
        "median_peak_current_uA": "Peak height (µA)",
        "snr_unadjusted": "Raw SNR",
        "uA": "µA",
    }
    if metric in replacements:
        return replacements[metric]
    return metric.replace("_", " ").strip().title().replace("Ua", "µA").replace("Snr", "SNR")


def _metadata_group_color(
    group_id: Any,
    values: dict[int, float] | None,
) -> str | None:
    if not values:
        return None
    try:
        value = float(values[int(group_id)])
    except (KeyError, TypeError, ValueError):
        return None
    minimum = min(values.values())
    maximum = max(values.values())
    fraction = (
        (value - minimum) / (maximum - minimum)
        if maximum > minimum else .5
    )
    red, green, blue, _alpha = plt.get_cmap("viridis")(fraction)
    return f"rgb({red * 255:.0f},{green * 255:.0f},{blue * 255:.0f})"


def _add_metadata_colorbar(
    fig: go.Figure,
    values: dict[int, float] | None,
    label: str | None,
) -> None:
    if not values or not label:
        return
    minimum = min(values.values())
    maximum = max(values.values())
    if minimum == maximum:
        maximum = minimum + 1.0
    fig.add_trace(go.Scatter(
        x=[None, None],
        y=[None, None],
        mode="markers",
        marker={
            "color": [minimum, maximum],
            "colorscale": "Viridis",
            "cmin": minimum,
            "cmax": maximum,
            "showscale": True,
            "colorbar": {"title": label},
        },
        hoverinfo="skip",
        showlegend=False,
    ))


def _plot_trend(
    frame: pd.DataFrame,
    metric: str,
    group_layout: str = "Plot groups overlaid",
    group_color_values: dict[int, float] | None = None,
    group_color_label: str | None = None,
):
    if metric not in frame.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No changing trend metrics are available.",
            x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
        )
        fig.update_layout(height=340)
        return fig
    values = pd.to_numeric(frame[metric], errors="coerce")
    x = pd.to_numeric(frame.get("iteration", pd.Series(range(1, len(frame) + 1))), errors="coerce")
    valid = values.notna() & x.notna()
    fig = go.Figure()
    grouped = "group_id" in frame.columns and frame.loc[valid, "group_id"].nunique() > 1
    if grouped and group_layout == "Average groups together":
        averaged = pd.DataFrame({
            "iteration": x[valid].astype(int),
            "value": values[valid],
        }).groupby("iteration", as_index=False)["value"].mean()
        fig.add_trace(go.Scatter(
            x=averaged["iteration"],
            y=averaged["value"],
            mode="lines+markers",
            name="Group average",
            marker={"size": 8},
            customdata=averaged["iteration"],
            hovertemplate=(
                "Iteration %{x}<br>Group average: %{y:.4g}<extra></extra>"
            ),
        ))
        fig.update_layout(
            title=f"{metric} over BO iterations — average across groups",
            xaxis_title="BO iteration",
            yaxis_title=metric,
            height=340,
            margin={"l": 65, "r": 20, "t": 50, "b": 55},
            clickmode="event+select",
        )
        return fig
    if grouped and group_layout == "Plot groups separately":
        grouped_rows = list(frame.loc[valid].groupby("group_id", sort=True))
        columns = 2 if len(grouped_rows) > 1 else 1
        rows_count = max(1, (len(grouped_rows) + columns - 1) // columns)
        titles = [
            (
                str(rows["group_name"].iloc[0])
                if "group_name" in rows.columns
                else f"Group {int(group_id)}"
            )
            for group_id, rows in grouped_rows
        ]
        fig = make_subplots(
            rows=rows_count,
            cols=columns,
            subplot_titles=titles,
        )
        for index, ((group_id, rows), group_name) in enumerate(
            zip(grouped_rows, titles)
        ):
            subplot_row, subplot_column = divmod(index, columns)
            rows = rows.sort_values("iteration")
            iterations = pd.to_numeric(
                rows["iteration"],
                errors="coerce",
            ).astype(int)
            row_values = pd.to_numeric(rows[metric], errors="coerce")
            group_color = _metadata_group_color(
                group_id,
                group_color_values,
            )
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=row_values,
                    mode="lines+markers",
                    name=group_name,
                    marker={"color": group_color} if group_color else None,
                    line={"color": group_color} if group_color else None,
                    customdata=iterations,
                    hovertemplate=(
                        f"{group_name}<br>Iteration %{{x}}<br>"
                        f"{metric}: %{{y:.4g}}<extra></extra>"
                    ),
                ),
                row=subplot_row + 1,
                col=subplot_column + 1,
            )
            fig.update_xaxes(
                title_text="BO iteration",
                row=subplot_row + 1,
                col=subplot_column + 1,
            )
            fig.update_yaxes(
                title_text=metric,
                matches="y",
                row=subplot_row + 1,
                col=subplot_column + 1,
            )
        _add_metadata_colorbar(
            fig,
            group_color_values,
            group_color_label,
        )
        fig.update_layout(
            title=f"{metric} over BO iterations — separate groups",
            height=max(
                420 if group_color_values else 360,
                280 * rows_count,
            ),
            margin={
                "l": 65,
                "r": 80 if group_color_values else 20,
                "t": 65,
                "b": 55,
            },
            showlegend=False,
            clickmode="event+select",
        )
        return fig
    group_series = (
        frame.loc[valid].groupby("group_id", sort=True)
        if grouped
        else [(None, frame.loc[valid])]
    )
    for group_id, rows in group_series:
        rows = rows.sort_values("iteration")
        row_values = pd.to_numeric(rows[metric], errors="coerce")
        iterations = pd.to_numeric(rows["iteration"], errors="coerce").astype(int)
        group_name = (
            str(rows["group_name"].iloc[0])
            if grouped and "group_name" in rows.columns
            else metric
        )
        group_color = _metadata_group_color(group_id, group_color_values)
        fig.add_trace(go.Scatter(
            x=iterations,
            y=row_values,
            mode="lines+markers",
            name=group_name,
            marker={
                "size": 8,
                **({"color": group_color} if group_color else {}),
            },
            line={"color": group_color} if group_color else None,
            customdata=iterations,
            hovertemplate=(
                f"{group_name}<br>Iteration %{{x}}<br>"
                f"{metric}: %{{y:.4g}}<extra></extra>"
            ),
        ))
    if grouped:
        _add_metadata_colorbar(fig, group_color_values, group_color_label)
    fig.update_layout(
        title=f"{metric} over BO iterations",
        xaxis_title="BO iteration",
        yaxis_title=metric,
        height=420 if group_color_values else 340,
        margin={
            "l": 65,
            "r": 80 if group_color_values else 20,
            "t": 50,
            "b": 115 if group_color_values else 55,
        },
        legend=(
            {
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": -0.22,
                "yanchor": "top",
            }
            if group_color_values else None
        ),
        clickmode="event+select",
    )
    return fig


def _plot_channel_trend(
    frame: pd.DataFrame,
    metric: str,
    channel_columns: dict[str, str],
    selected_channels: list[str],
    layout: str,
    group_layout: str = "Plot groups overlaid",
    group_color_values: dict[int, float] | None = None,
    group_color_label: str | None = None,
):
    iterations = pd.to_numeric(
        frame.get("iteration", pd.Series(range(1, len(frame) + 1))),
        errors="coerce",
    )
    label = _metric_label(metric)
    group_ids = (
        frame["group_id"]
        if "group_id" in frame.columns
        else pd.Series(1, index=frame.index)
    )
    group_names = (
        frame["group_name"].fillna("").astype(str)
        if "group_name" in frame.columns
        else pd.Series("", index=frame.index)
    )
    records = []
    for channel in selected_channels:
        values = pd.to_numeric(frame[channel_columns[channel]], errors="coerce")
        valid = iterations.notna() & values.notna()
        for index in frame.index[valid]:
            group_id = group_ids.loc[index]
            group_name = group_names.loc[index] or f"Group {group_id}"
            records.append({
                "iteration": int(iterations.loc[index]),
                "value": float(values.loc[index]),
                "group_id": group_id,
                "group_name": group_name,
                "channel": str(channel),
            })
    data = pd.DataFrame(records)
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No values are available for the selected channels.",
            x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
        )
        fig.update_layout(height=340)
        return fig

    # Normalize duplicate rows before applying either display-level average.
    data = data.groupby(
        ["group_id", "group_name", "channel", "iteration"],
        as_index=False,
        dropna=False,
    )["value"].mean()
    multiple_groups = data["group_id"].nunique(dropna=False) > 1

    if multiple_groups and group_layout == "Average groups together":
        data = data.groupby(
            ["channel", "iteration"],
            as_index=False,
        )["value"].mean()
        data["group_id"] = "__average__"
        data["group_name"] = "Group average"

    if layout == "Average selected channels":
        data = data.groupby(
            ["group_id", "group_name", "iteration"],
            as_index=False,
            dropna=False,
        )["value"].mean()
        data["channel"] = "average"

    separate_groups = (
        multiple_groups
        and group_layout == "Plot groups separately"
    )
    separate_channels = layout == "Separate plots"
    if separate_groups and separate_channels:
        facets = [
            ((group_id, channel), f"{group_name} — Channel {channel}")
            for (group_id, group_name, channel), _rows
            in data.groupby(["group_id", "group_name", "channel"], sort=True)
        ]
    elif separate_groups:
        facets = [
            (group_id, group_name)
            for (group_id, group_name), _rows
            in data.groupby(["group_id", "group_name"], sort=True)
        ]
    elif separate_channels:
        facets = [
            (channel, f"Channel {channel}")
            for channel in sorted(data["channel"].unique(), key=_channel_sort_key)
        ]
    else:
        facets = [(None, "")]

    subplot_columns = 2 if len(facets) > 1 else 1
    subplot_rows = max(1, (len(facets) + subplot_columns - 1) // subplot_columns)
    faceted = len(facets) > 1 or facets[0][0] is not None
    if faceted:
        fig = make_subplots(
            rows=subplot_rows,
            cols=subplot_columns,
            subplot_titles=[title for _key, title in facets],
        )
    else:
        fig = go.Figure()

    for facet_index, (facet_key, _facet_title) in enumerate(facets):
        facet_data = data
        if separate_groups and separate_channels:
            facet_data = data[
                (data["group_id"] == facet_key[0])
                & (data["channel"] == facet_key[1])
            ]
        elif separate_groups:
            facet_data = data[data["group_id"] == facet_key]
        elif separate_channels:
            facet_data = data[data["channel"] == facet_key]

        series_columns = []
        if not separate_groups and facet_data["group_id"].nunique(dropna=False) > 1:
            series_columns.extend(["group_id", "group_name"])
        if not separate_channels and facet_data["channel"].nunique() > 1:
            series_columns.append("channel")
        series = (
            facet_data.groupby(series_columns, sort=True, dropna=False)
            if series_columns
            else [(None, facet_data)]
        )
        for _series_key, rows in series:
            rows = rows.sort_values("iteration")
            group_name = str(rows["group_name"].iloc[0])
            channel = str(rows["channel"].iloc[0])
            name_parts = []
            if group_name != "Group average" and (
                not separate_groups or not separate_channels
            ):
                if multiple_groups:
                    name_parts.append(group_name)
            elif group_name == "Group average":
                name_parts.append(group_name)
            if channel == "average":
                name_parts.append("channel average")
            elif not separate_channels:
                name_parts.append(f"Ch {channel}")
            trace_name = " · ".join(name_parts) or (
                f"Ch {channel}" if channel != "average" else "Channel average"
            )
            group_color = _metadata_group_color(
                rows["group_id"].iloc[0],
                group_color_values,
            )
            trace = go.Scatter(
                x=rows["iteration"],
                y=rows["value"],
                mode="lines+markers",
                name=trace_name,
                marker={"color": group_color} if group_color else None,
                line={"color": group_color} if group_color else None,
                customdata=rows["iteration"],
                hovertemplate=(
                    f"{trace_name}<br>Iteration %{{x}}<br>"
                    f"{label}: %{{y:.4g}}<extra></extra>"
                ),
            )
            if faceted:
                subplot_row, subplot_column = divmod(facet_index, subplot_columns)
                fig.add_trace(
                    trace,
                    row=subplot_row + 1,
                    col=subplot_column + 1,
                )
            else:
                fig.add_trace(trace)

    if multiple_groups:
        _add_metadata_colorbar(fig, group_color_values, group_color_label)

    if faceted:
        for facet_index in range(len(facets)):
            subplot_row, subplot_column = divmod(facet_index, subplot_columns)
            fig.update_xaxes(
                title_text="BO iteration",
                row=subplot_row + 1,
                col=subplot_column + 1,
            )
            fig.update_yaxes(
                title_text=label,
                matches="y",
                row=subplot_row + 1,
                col=subplot_column + 1,
            )
    title = label
    if layout == "Average selected channels":
        title += ": average across selected channels"
    elif layout == "Overlay selected channels":
        title += ": selected channels overlaid"
    else:
        title += " by channel"
    if multiple_groups and group_layout == "Average groups together":
        title += " — average across groups"
    elif separate_groups:
        title += " — separate groups"
    fig.update_layout(
        title=title,
        xaxis_title=None if faceted else "BO iteration",
        yaxis_title=None if faceted else label,
        height=max(420 if group_color_values else 340, 270 * subplot_rows),
        margin={
            "l": 65,
            "r": 80 if group_color_values else 20,
            "t": 65,
            "b": 115 if group_color_values else 55,
        },
        legend=(
            {
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": -0.22,
                "yanchor": "top",
            }
            if group_color_values else None
        ),
        clickmode="event+select",
    )
    return fig


def _paired_trend_values(observations: list[dict], metric_label: str) -> dict[str, dict[str, list]]:
    """Build channel-wise buffer/target series from paired BO observations."""
    source, primary_key, fallback_key = PAIRED_TREND_METRICS[metric_label]
    result: dict[str, dict[str, list]] = {}
    for observation in observations:
        iteration = int(observation.get("iteration", 0))
        quality = observation.get("quality") or {}
        components = quality.get("channel_components") or {}
        phase_metrics = {
            "buffer": observation.get("buffer_channel_metrics") or {},
            "target": observation.get("target_channel_metrics") or {},
        }
        channels = set(phase_metrics["buffer"]) | set(phase_metrics["target"]) | set(components)
        for channel in channels:
            series = result.setdefault(
                str(channel),
                {"iteration": [], "buffer": [], "target": []},
            )
            buffer_value = target_value = None
            if source == "channel":
                for phase in ("buffer", "target"):
                    metrics = phase_metrics[phase].get(str(channel), {}) or phase_metrics[phase].get(channel, {}) or {}
                    value = metrics.get(primary_key)
                    if value is None and fallback_key:
                        value = metrics.get(fallback_key)
                    if phase == "buffer":
                        buffer_value = value
                    else:
                        target_value = value
            else:
                component = components.get(str(channel), {}) or components.get(channel, {}) or {}
                buffer_value = component.get(f"buffer_{primary_key}")
                target_value = component.get(f"target_{primary_key}")
            series["iteration"].append(iteration)
            series["buffer"].append(buffer_value)
            series["target"].append(target_value)
    return result


def _plot_paired_phase_trend(
    series_by_channel: dict[str, dict[str, list]],
    metric_label: str,
    selected_channels: list[str],
    layout: str,
):
    phase_colors = {"buffer": "#1f77b4", "target": "#ff7f0e"}
    if layout == "Separate plots":
        columns = 2 if len(selected_channels) > 1 else 1
        rows = max(1, (len(selected_channels) + columns - 1) // columns)
        fig = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=[f"Channel {channel}" for channel in selected_channels],
        )
        for index, channel in enumerate(selected_channels):
            row, column = divmod(index, columns)
            series = series_by_channel[channel]
            for phase in ("buffer", "target"):
                values = pd.to_numeric(pd.Series(series[phase]), errors="coerce")
                iterations = pd.Series(series["iteration"])
                valid = values.notna()
                fig.add_trace(
                    go.Scatter(
                        x=iterations[valid],
                        y=values[valid],
                        mode="lines+markers",
                        name=phase.title(),
                        legendgroup=phase,
                        showlegend=index == 0,
                        line={"color": phase_colors[phase]},
                        customdata=iterations[valid],
                        hovertemplate=(
                            f"Iteration %{{x}}<br>{phase.title()} {metric_label}: "
                            "%{y:.4g}<extra></extra>"
                        ),
                    ),
                    row=row + 1,
                    col=column + 1,
                )
            fig.update_xaxes(title_text="BO iteration", row=row + 1, col=column + 1)
            fig.update_yaxes(
                title_text=metric_label,
                matches="y",
                row=row + 1,
                col=column + 1,
            )
        fig.update_layout(
            title=f"Buffer vs target {metric_label}",
            height=max(340, 270 * rows),
            clickmode="event+select",
        )
        return fig

    fig = go.Figure()
    if layout == "Average selected channels":
        all_iterations = sorted({
            iteration
            for channel in selected_channels
            for iteration in series_by_channel[channel]["iteration"]
        })
        for phase in ("buffer", "target"):
            values = []
            for iteration in all_iterations:
                iteration_values = []
                for channel in selected_channels:
                    series = series_by_channel[channel]
                    for idx, recorded_iteration in enumerate(series["iteration"]):
                        if recorded_iteration == iteration and pd.notna(series[phase][idx]):
                            iteration_values.append(float(series[phase][idx]))
                values.append(float(np.mean(iteration_values)) if iteration_values else None)
            fig.add_trace(go.Scatter(
                x=all_iterations,
                y=values,
                mode="lines+markers",
                name=phase.title(),
                line={"color": phase_colors[phase]},
                customdata=all_iterations,
                hovertemplate=(
                    f"Iteration %{{x}}<br>Average {phase} {metric_label}: "
                    "%{y:.4g}<extra></extra>"
                ),
            ))
        title = f"Buffer vs target {metric_label}: selected-channel average"
    else:
        use_generic_phase_legend = len(selected_channels) > 4
        for channel_index, channel in enumerate(selected_channels):
            series = series_by_channel[channel]
            for phase in ("buffer", "target"):
                values = pd.to_numeric(pd.Series(series[phase]), errors="coerce")
                iterations = pd.Series(series["iteration"])
                valid = values.notna()
                fig.add_trace(go.Scatter(
                    x=iterations[valid],
                    y=values[valid],
                    mode="lines+markers",
                    name=phase.title() if use_generic_phase_legend else f"Ch {channel} {phase}",
                    legendgroup=phase,
                    showlegend=not use_generic_phase_legend or channel_index == 0,
                    line={"color": phase_colors[phase]},
                    customdata=iterations[valid],
                    hovertemplate=(
                        f"Iteration %{{x}}<br>Ch {channel} {phase} {metric_label}: "
                        "%{y:.4g}<extra></extra>"
                    ),
                ))
        title = f"Buffer vs target {metric_label}: selected channels"
    fig.update_layout(
        title=title,
        xaxis_title="BO iteration",
        yaxis_title=metric_label,
        height=360,
        clickmode="event+select",
    )
    return fig


def _chronological_points(
    observations: list[dict],
    config: dict,
    metric_label: str,
    selected_channels: list[str],
    average_channels: bool,
) -> tuple[pd.DataFrame, list[tuple[float, str]]]:
    """Expand paired observations into their actual buffer-then-target order."""
    source, primary_key, fallback_key = PAIRED_TREND_METRICS[metric_label]
    batch_size = max(1, int((config or {}).get("paired_batch_size", 1) or 1))
    grouped: dict[int, list[dict]] = {}
    for observation in observations:
        iteration = int(observation.get("iteration", 0) or 0)
        cycle = observation.get("paired_cycle")
        if cycle in (None, ""):
            cycle = ((iteration - 1) // batch_size) + 1
        grouped.setdefault(int(cycle), []).append(observation)

    rows = []
    transitions: list[tuple[float, str]] = []
    position = 0
    cycles = sorted(grouped)
    for cycle_index, cycle in enumerate(cycles):
        cycle_observations = grouped[cycle]
        for phase in ("buffer", "target"):
            trace_key = f"{phase}_trace_number"

            def order_key(observation):
                iteration = int(observation.get("iteration", 0) or 0)
                batch = observation.get("paired_batch_index")
                if batch in (None, ""):
                    batch = ((iteration - 1) % batch_size) + 1
                trace = observation.get(trace_key)
                try:
                    trace_value = int(trace)
                except (TypeError, ValueError):
                    trace_value = int(batch)
                return trace_value, int(batch), iteration

            for observation in sorted(cycle_observations, key=order_key):
                position += 1
                iteration = int(observation.get("iteration", 0) or 0)
                batch = observation.get("paired_batch_index")
                if batch in (None, ""):
                    batch = ((iteration - 1) % batch_size) + 1
                phase_metrics = observation.get(f"{phase}_channel_metrics") or {}
                components = (observation.get("quality") or {}).get("channel_components") or {}
                event_rows = []
                for channel in selected_channels:
                    if source == "channel":
                        metrics = phase_metrics.get(channel, {}) or {}
                        value = metrics.get(primary_key)
                        if value is None and fallback_key:
                            value = metrics.get(fallback_key)
                    else:
                        component = components.get(channel, {}) or {}
                        value = component.get(f"{phase}_{primary_key}")
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    event_rows.append({
                        "position": position,
                        "cycle": cycle,
                        "batch": int(batch),
                        "iteration": iteration,
                        "phase": phase,
                        "channel": channel,
                        "value": numeric_value,
                    })
                if average_channels and event_rows:
                    averaged = dict(event_rows[0])
                    averaged["channel"] = "Average"
                    averaged["value"] = float(np.mean([row["value"] for row in event_rows]))
                    rows.append(averaged)
                else:
                    rows.extend(event_rows)
            if phase == "buffer":
                transitions.append((position + .5, "Buffer → Target"))
        if cycle_index < len(cycles) - 1:
            transitions.append((position + .5, "Target → Buffer"))
    return pd.DataFrame(rows), transitions


def _plot_chronological(
    points: pd.DataFrame,
    transitions: list[tuple[float, str]],
    metric_label: str,
    layout: str,
):
    colors = {"buffer": "#1f77b4", "target": "#ff7f0e"}
    channels = list(dict.fromkeys(points["channel"].astype(str)))
    if layout == "Separate plots":
        columns = 2 if len(channels) > 1 else 1
        rows = max(1, (len(channels) + columns - 1) // columns)
        fig = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=[
                f"Channel {channel}" if channel != "Average" else "Channel average"
                for channel in channels
            ],
        )
        for channel_index, channel in enumerate(channels):
            row_index, column_index = divmod(channel_index, columns)
            row, column = row_index + 1, column_index + 1
            channel_points = points[
                points["channel"].astype(str) == channel
            ].sort_values("position")
            fig.add_trace(
                go.Scatter(
                    x=channel_points["position"],
                    y=channel_points["value"],
                    mode="lines",
                    line={"color": "rgba(90,90,90,0.28)", "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=column,
            )
            for phase in ("buffer", "target"):
                phase_points = channel_points[channel_points["phase"] == phase]
                if phase_points.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=phase_points["position"],
                        y=phase_points["value"],
                        mode="markers",
                        name=phase.title(),
                        legendgroup=phase,
                        showlegend=channel_index == 0,
                        marker={"color": colors[phase], "size": 8},
                        customdata=phase_points[["iteration", "cycle", "batch", "channel"]],
                        hovertemplate=(
                            f"{phase.title()} {metric_label}: %{{y:.4g}}<br>"
                            "Iteration %{customdata[0]} | Cycle %{customdata[1]} | "
                            "Batch %{customdata[2]} | Channel %{customdata[3]}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=column,
                )
            for x_position, _label in transitions:
                fig.add_vline(
                    x=x_position,
                    line_dash="dash",
                    line_color="#5a6b84",
                    line_width=1.2,
                    row=row,
                    col=column,
                )
            fig.update_xaxes(
                title_text="Chronological order",
                row=row,
                col=column,
            )
            fig.update_yaxes(
                title_text=metric_label,
                matches="y",
                row=row,
                col=column,
            )
        fig.update_layout(
            title=f"Chronological buffer/target measurements | {metric_label}",
            height=max(400, 280 * rows),
            margin={"l": 65, "r": 30, "t": 60, "b": 55},
        )
        return fig

    fig = go.Figure()
    generic_legend = len(channels) > 4
    for channel_index, channel in enumerate(channels):
        channel_points = points[points["channel"].astype(str) == channel].sort_values("position")
        fig.add_trace(go.Scatter(
            x=channel_points["position"],
            y=channel_points["value"],
            mode="lines",
            line={"color": "rgba(90,90,90,0.28)", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        ))
        for phase in ("buffer", "target"):
            phase_points = channel_points[channel_points["phase"] == phase]
            if phase_points.empty:
                continue
            if channel == "Average":
                name = phase.title()
                showlegend = True
            elif generic_legend:
                name = phase.title()
                showlegend = channel_index == 0
            else:
                name = f"Ch {channel} {phase}"
                showlegend = True
            fig.add_trace(go.Scatter(
                x=phase_points["position"],
                y=phase_points["value"],
                mode="markers",
                name=name,
                legendgroup=phase if generic_legend else f"{channel}_{phase}",
                showlegend=showlegend,
                marker={"color": colors[phase], "size": 8},
                customdata=phase_points[["iteration", "cycle", "batch", "channel"]],
                hovertemplate=(
                    f"{phase.title()} {metric_label}: %{{y:.4g}}<br>"
                    "Iteration %{customdata[0]} | Cycle %{customdata[1]} | "
                    "Batch %{customdata[2]} | Channel %{customdata[3]}<extra></extra>"
                ),
            ))
    for x_position, label in transitions:
        color = "#2ca02c" if label.startswith("Buffer") else "#9467bd"
        fig.add_vline(x=x_position, line_dash="dash", line_color=color, line_width=1.5)
        fig.add_annotation(
            x=x_position,
            y=1,
            yref="paper",
            text=label,
            showarrow=False,
            textangle=-90,
            xanchor="left",
            yanchor="top",
            font={"size": 10, "color": color},
        )
    fig.update_layout(
        title=f"Chronological buffer/target measurements | {metric_label}",
        xaxis_title="Chronological measurement order",
        yaxis_title=metric_label,
        height=400,
        margin={"l": 65, "r": 30, "t": 60, "b": 55},
    )
    return fig


def _real_data_channels(observations: list[dict]) -> list[str]:
    channels = set()
    for observation in observations:
        channels.update(str(key) for key in (observation.get("buffer_channel_metrics") or {}))
        channels.update(str(key) for key in (observation.get("target_channel_metrics") or {}))
        channels.update(str(key) for key in (observation.get("channel_metrics") or {}))
        channels.update(
            str(key)
            for key in ((observation.get("quality") or {}).get("channel_components") or {})
        )
    return sorted(channels, key=_channel_sort_key)


def _real_metric_points(
    observations: list[dict],
    metric_label: str,
    phase: str,
    selected_channels: list[str],
    average_channels: bool,
) -> pd.DataFrame:
    source, primary_key, fallback_key = REAL_DATA_METRICS[metric_label]
    rows = []
    for observation in observations:
        params = observation.get("params") or {}
        per_iteration = []
        phase_metrics = (
            observation.get("channel_metrics") or {}
            if phase == "measurement"
            else observation.get(f"{phase}_channel_metrics") or {}
        )
        components = (observation.get("quality") or {}).get("channel_components") or {}
        for channel in selected_channels:
            if source == "channel":
                metrics = phase_metrics.get(channel, {}) or {}
                value = metrics.get(primary_key)
                if value is None and fallback_key:
                    value = metrics.get(fallback_key)
            else:
                component = components.get(channel, {}) or {}
                if phase == "measurement" and primary_key == "classic_Q":
                    value = component.get("Q_channel")
                elif phase == "measurement" and primary_key == "snr_score":
                    value = component.get("normalized_SNR")
                else:
                    value = component.get(f"{phase}_{primary_key}")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            row = {
                "iteration": int(observation.get("iteration", 0)),
                "channel": channel,
                "group_id": int(observation.get("group_id", 1)),
                "group_name": str(
                    observation.get("group_name")
                    or f"Group {observation.get('group_id', 1)}"
                ),
                "value": numeric_value,
            }
            row.update({
                name: float(params[name])
                for name in PARAMETERS
                if params.get(name) is not None
            })
            per_iteration.append(row)
        if average_channels and per_iteration:
            averaged = dict(per_iteration[0])
            averaged["channel"] = "Average"
            averaged["value"] = float(np.mean([row["value"] for row in per_iteration]))
            rows.append(averaged)
        else:
            rows.extend(per_iteration)
    return pd.DataFrame(rows)


def _real_group_metric_points(
    observations: list[dict],
    metric_label: str,
    phase: str,
    selected_groups: list[dict],
) -> pd.DataFrame:
    """Average member channels into one real-data series per channel group."""
    frames = []
    for group in selected_groups:
        group_id = int(group["id"])
        group_observations = [
            observation for observation in observations
            if int(observation.get("group_id", 1)) == group_id
        ]
        points = _real_metric_points(
            group_observations,
            metric_label,
            phase,
            [str(channel) for channel in group.get("channels", [])],
            average_channels=True,
        )
        if not points.empty:
            points["channel"] = str(group.get("name") or f"Group {group_id}")
            points["group_id"] = group_id
            frames.append(points)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _real_count_occurrences(
    observations: list[dict],
    selected_channels: list[str],
) -> pd.DataFrame:
    """Return one row per completed observation and participating channel."""
    rows = []
    selected = set(map(str, selected_channels))
    for observation in observations:
        params = observation.get("params") or {}
        channels = selected.intersection(_real_data_channels([observation]))
        for channel in channels:
            row = {
                "iteration": int(observation.get("iteration", 0)),
                "channel": channel,
                "group_id": int(observation.get("group_id", 1)),
                "group_name": str(
                    observation.get("group_name")
                    or f"Group {observation.get('group_id', 1)}"
                ),
                "value": 1.0,
            }
            row.update({
                name: float(params[name])
                for name in PARAMETERS
                if params.get(name) is not None
            })
            rows.append(row)
    return pd.DataFrame(rows)


def _bin_real_count_points(
    points: pd.DataFrame,
    dimensions: list[str],
    bin_sizes: dict[str, float],
    average_channels: bool,
) -> pd.DataFrame:
    """Aggregate observation occurrences into parameter-space bins."""
    if points.empty:
        return points
    binned = points.copy()
    for dimension in dimensions:
        size = float(bin_sizes[dimension])
        values = pd.to_numeric(binned[dimension], errors="coerce")
        binned[dimension] = (
            np.floor(values / size) * size + size / 2.0
        ).round(12)
    group_columns = ["group_id", "group_name", "channel", *dimensions]
    counts = (
        binned.dropna(subset=dimensions)
        .groupby(group_columns, as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "value"})
    )
    bin_values = [
        list(np.round(np.arange(
            float(binned[dimension].min()),
            float(binned[dimension].max()) + bin_sizes[dimension] * 0.5,
            bin_sizes[dimension],
        ), 12))
        for dimension in dimensions
    ]
    series_rows = binned[
        ["group_id", "group_name", "channel"]
    ].drop_duplicates()
    complete_rows = []
    for _, series in series_rows.iterrows():
        for coordinates in itertools.product(*bin_values):
            complete_rows.append({
                "group_id": series["group_id"],
                "group_name": series["group_name"],
                "channel": series["channel"],
                **dict(zip(dimensions, coordinates)),
            })
    if complete_rows:
        counts = pd.DataFrame(complete_rows).merge(
            counts,
            on=group_columns,
            how="left",
        )
        counts["value"] = counts["value"].fillna(0.0)
    counts["iteration"] = counts["value"]
    if average_channels and not counts.empty:
        counts = (
            counts.groupby(
                ["group_id", "group_name", *dimensions],
                as_index=False,
                dropna=False,
            )["value"]
            .mean()
        )
        counts["channel"] = "Average"
        counts["iteration"] = counts["value"]
    return counts


def _add_plotly_iteration_path(
    fig: go.Figure,
    x,
    y,
    iterations,
    z=None,
    *,
    show_colorbar: bool = True,
) -> None:
    """Add chronological red-to-black line segments to a Plotly figure."""
    x_values = list(x)
    y_values = list(y)
    z_values = list(z) if z is not None else None
    iteration_values = np.asarray(list(iterations), dtype=float)
    if len(iteration_values) < 2:
        return
    norm = _iteration_norm(iteration_values)
    is_3d = z_values is not None
    scatter_type = go.Scatter3d if is_3d else go.Scatter
    for index in range(1, len(iteration_values)):
        color = OBSERVED_PATH_CMAP(norm(iteration_values[index]))
        trace_kwargs = {
            "x": x_values[index - 1:index + 1],
            "y": y_values[index - 1:index + 1],
            "mode": "lines",
            "line": {
                "color": (
                    f"rgb({color[0] * 255:.0f},"
                    f"{color[1] * 255:.0f},"
                    f"{color[2] * 255:.0f})"
                ),
                "width": 4,
            },
            "hoverinfo": "skip",
            "showlegend": False,
        }
        if is_3d:
            trace_kwargs["z"] = z_values[index - 1:index + 1]
        fig.add_trace(scatter_type(**trace_kwargs))
    if show_colorbar:
        marker = {
            "color": [norm.vmin, norm.vmax],
            "colorscale": [
                [0, OBSERVED_PATH_COLORS[0]],
                [1, OBSERVED_PATH_COLORS[1]],
            ],
            "cmin": norm.vmin,
            "cmax": norm.vmax,
            "showscale": True,
            "colorbar": {
                "title": "Iteration",
                "x": 1.12,
                "len": 0.78,
            },
        }
        invisible_kwargs = {
            "x": [None, None],
            "y": [None, None],
            "mode": "markers",
            "marker": marker,
            "hoverinfo": "skip",
            "showlegend": False,
        }
        if is_3d:
            invisible_kwargs["z"] = [None, None]
        fig.add_trace(scatter_type(**invisible_kwargs))


def _plot_real_data_landscape(
    points: pd.DataFrame,
    metric_label: str,
    phase: str,
    view: str,
    x_name: str,
    y_name: str | None,
    z_name: str | None,
    tensor_height: int = 480,
    dot_size: int = 6,
    log_frequency: bool = False,
    color_by: str = "Measured value",
    group_color_values: dict[int, float] | None = None,
    group_color_label: str | None = None,
    iteration_path: pd.DataFrame | None = None,
    show_iteration_path: bool = True,
    count_bin_sizes: dict[str, float] | None = None,
    axis_ranges: dict[str, tuple[float, float]] | None = None,
    value_range: tuple[float, float] | None = None,
):
    def series_label(value: Any) -> str:
        text = str(value)
        if text == "Average":
            return "Channel average"
        if text.lower().startswith("group"):
            return text
        return f"Ch {text}"

    hover_data = ["iteration", "channel"]
    series_columns = ["group_id", "group_name", "channel"]
    multiple_groups = points["group_id"].nunique() > 1
    multiple_series = len(points[series_columns].drop_duplicates()) > 1
    move_legend_below = multiple_series and (
        color_by == "Measured value" or bool(group_color_values)
    )

    def colored_series():
        grouped = list(points.groupby(series_columns, sort=True))
        channel_values = sorted(points["channel"].astype(str).unique(), key=_channel_sort_key)
        group_values = sorted(points["group_id"].astype(int).unique())
        for (group_id, group_name, channel), group in grouped:
            if color_by == "Channel":
                color_index = channel_values.index(str(channel))
                color = plt.get_cmap("tab10")(color_index % 10)
            elif color_by == "Group":
                color_index = group_values.index(int(group_id))
                color = plt.get_cmap("tab10")(color_index % 10)
            else:
                color = _metadata_group_color(group_id, group_color_values)
                yield group_id, group_name, channel, group, color
                continue
            yield (
                group_id,
                group_name,
                channel,
                group,
                f"rgb({color[0] * 255:.0f},{color[1] * 255:.0f},{color[2] * 255:.0f})",
            )

    def trace_label(group_name, channel) -> str:
        channel_label = series_label(channel)
        return (
            f"{group_name} · {channel_label}"
            if multiple_groups else channel_label
        )

    if metric_label == "Count" and view == "3D tensor":
        fig = go.Figure()
        bin_sizes = count_bin_sizes or {}
        value_min = (
            float(value_range[0])
            if value_range is not None
            else float(points["value"].min())
        )
        value_max = (
            float(value_range[1])
            if value_range is not None
            else float(points["value"].max())
        )
        if value_min == value_max:
            value_max = value_min + 1.0
        cube_faces = (
            (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
            (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
        )
        for trace_index, (
            _group_id,
            group_name,
            channel,
            group,
            _series_color,
        ) in enumerate(colored_series()):
            vertices_x, vertices_y, vertices_z = [], [], []
            face_i, face_j, face_k = [], [], []
            intensities, hover_text = [], []
            half_x = float(bin_sizes.get(x_name, 1.0)) / 2.0
            half_y = float(bin_sizes.get(y_name, 1.0)) / 2.0
            half_z = float(bin_sizes.get(z_name, 1.0)) / 2.0
            for _, row in group.loc[group["value"] > 0].iterrows():
                base = len(vertices_x)
                x0, x1 = row[x_name] - half_x, row[x_name] + half_x
                y0, y1 = row[y_name] - half_y, row[y_name] + half_y
                z0, z1 = row[z_name] - half_z, row[z_name] + half_z
                vertices_x.extend((x0, x1, x1, x0, x0, x1, x1, x0))
                vertices_y.extend((y0, y0, y1, y1, y0, y0, y1, y1))
                vertices_z.extend((z0, z0, z0, z0, z1, z1, z1, z1))
                intensities.extend([float(row["value"])] * 8)
                hover_text.extend([
                    f"Count: {float(row['value']):g}<br>"
                    f"{x_name}: {float(row[x_name]):g}<br>"
                    f"{y_name}: {float(row[y_name]):g}<br>"
                    f"{z_name}: {float(row[z_name]):g}"
                ] * 8)
                for first, second, third in cube_faces:
                    face_i.append(base + first)
                    face_j.append(base + second)
                    face_k.append(base + third)
            fig.add_trace(go.Mesh3d(
                x=vertices_x,
                y=vertices_y,
                z=vertices_z,
                i=face_i,
                j=face_j,
                k=face_k,
                intensity=intensities,
                colorscale="Viridis",
                cmin=value_min,
                cmax=value_max,
                showscale=trace_index == 0,
                colorbar={"title": "Count"},
                opacity=.82,
                flatshading=True,
                name=trace_label(group_name, channel),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            ))
        if (
            show_iteration_path
            and iteration_path is not None
            and not iteration_path.empty
        ):
            path_colorbar_added = False
            for _group_id, group_path in iteration_path.groupby("group_id"):
                ordered_path = group_path.dropna(
                    subset=[x_name, y_name, z_name]
                ).sort_values("iteration")
                _add_plotly_iteration_path(
                    fig,
                    ordered_path[x_name],
                    ordered_path[y_name],
                    ordered_path["iteration"],
                    z=ordered_path[z_name],
                    show_colorbar=not path_colorbar_added,
                )
                path_colorbar_added = (
                    path_colorbar_added or len(ordered_path) >= 2
                )
        fig.update_layout(
            scene={
                "xaxis": {
                    "title": x_name,
                    "type": "log" if log_frequency and x_name == "frequency" else "linear",
                },
                "yaxis": {
                    "title": y_name,
                    "type": "log" if log_frequency and y_name == "frequency" else "linear",
                },
                "zaxis": {
                    "title": z_name,
                    "type": "log" if log_frequency and z_name == "frequency" else "linear",
                },
            },
            height=tensor_height,
        )
    elif metric_label == "Count" and view == "2D map":
        fig = go.Figure()
        heatmap_points = (
            points.groupby([x_name, y_name], as_index=False)["value"].sum()
        )
        count_grid = heatmap_points.pivot(
            index=y_name,
            columns=x_name,
            values="value",
        ).sort_index().sort_index(axis=1).fillna(0.0)
        fig.add_trace(go.Heatmap(
            x=count_grid.columns.to_numpy(),
            y=count_grid.index.to_numpy(),
            z=count_grid.to_numpy(),
            colorscale="Viridis",
            zmin=value_range[0] if value_range is not None else None,
            zmax=value_range[1] if value_range is not None else None,
            colorbar={"title": "Count"},
            hovertemplate=(
                f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}"
                "<br>Count: %{z:g}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        ))
        if (
            show_iteration_path
            and iteration_path is not None
            and not iteration_path.empty
        ):
            path_colorbar_added = False
            for _group_id, group_path in iteration_path.groupby("group_id"):
                ordered_path = group_path.dropna(
                    subset=[x_name, y_name]
                ).sort_values("iteration")
                _add_plotly_iteration_path(
                    fig,
                    ordered_path[x_name],
                    ordered_path[y_name],
                    ordered_path["iteration"],
                    show_colorbar=not path_colorbar_added,
                )
                path_colorbar_added = (
                    path_colorbar_added or len(ordered_path) >= 2
                )
        fig.update_layout(
            xaxis_title=x_name,
            yaxis_title=y_name,
            xaxis_type="log" if log_frequency and x_name == "frequency" else "linear",
            yaxis_type="log" if log_frequency and y_name == "frequency" else "linear",
            height=420,
        )
    elif view == "3D tensor":
        fig = go.Figure()
        value_min = (
            float(value_range[0])
            if value_range is not None
            else float(points["value"].min())
        )
        value_max = (
            float(value_range[1])
            if value_range is not None
            else float(points["value"].max())
        )
        if value_min == value_max:
            value_max = value_min + 1.0
        for group_id, group_name, channel, group, series_color in colored_series():
            marker = {"size": dot_size}
            if color_by == "Measured value":
                marker.update({
                    "color": group["value"],
                    "colorscale": "Viridis",
                    "cmin": value_min,
                    "cmax": value_max,
                    "showscale": len(fig.data) == 0,
                    "colorbar": {"title": metric_label},
                })
            else:
                marker["color"] = series_color
            fig.add_trace(go.Scatter3d(
                x=group[x_name],
                y=group[y_name],
                z=group[z_name],
                mode="markers",
                name=trace_label(group_name, channel),
                marker=marker,
                customdata=group[hover_data],
                hovertemplate=(
                    f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}<br>"
                    f"{z_name}: %{{z:.4g}}<br>{metric_label}: %{{marker.color:.4g}}"
                    "<br>Iteration %{customdata[0]}<br>Series %{customdata[1]}<extra></extra>"
                ),
            ))
        if (
            show_iteration_path
            and iteration_path is not None
            and not iteration_path.empty
        ):
            path_colorbar_added = False
            for _group_id, group_path in iteration_path.groupby("group_id"):
                ordered_path = group_path.dropna(
                    subset=[x_name, y_name, z_name]
                ).sort_values("iteration")
                _add_plotly_iteration_path(
                    fig,
                    ordered_path[x_name],
                    ordered_path[y_name],
                    ordered_path["iteration"],
                    z=ordered_path[z_name],
                    show_colorbar=not path_colorbar_added,
                )
                path_colorbar_added = (
                    path_colorbar_added or len(ordered_path) >= 2
                )
        if group_color_values:
            _add_metadata_colorbar(fig, group_color_values, group_color_label)
        fig.update_layout(
            scene={
                "xaxis": {
                    "title": x_name,
                    "type": "log" if log_frequency and x_name == "frequency" else "linear",
                },
                "yaxis": {
                    "title": y_name,
                    "type": "log" if log_frequency and y_name == "frequency" else "linear",
                },
                "zaxis": {
                    "title": z_name,
                    "type": "log" if log_frequency and z_name == "frequency" else "linear",
                },
            },
            height=tensor_height,
        )
    elif view == "2D map":
        valid, grid = _interpolated_2d_grid(points, x_name, y_name)
        fig = go.Figure()
        if grid is not None and color_by == "Measured value":
            grid_x, grid_y, grid_values = grid
            fig.add_trace(go.Heatmap(
                x=grid_x,
                y=grid_y,
                z=grid_values,
                colorscale="Viridis",
                zmin=value_range[0] if value_range is not None else None,
                zmax=value_range[1] if value_range is not None else None,
                colorbar={"title": metric_label},
                hovertemplate=(
                    f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}<br>"
                    f"Interpolated {metric_label}: %{{z:.4g}}<extra></extra>"
                ),
                connectgaps=False,
            ))
        if color_by == "Measured value":
            fig.add_trace(go.Scatter(
                x=valid[x_name], y=valid[y_name], mode="markers",
                marker={
                    "size": 8, "color": valid["value"], "colorscale": "Viridis",
                    "cmin": value_range[0] if value_range is not None else None,
                    "cmax": value_range[1] if value_range is not None else None,
                    "showscale": grid is None, "colorbar": {"title": metric_label},
                    "line": {"color": "white", "width": 1},
                },
                name="measured points",
                customdata=valid[["iteration", "channel", "value"]],
                hovertemplate=(
                    f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}<br>"
                    f"{metric_label}: %{{customdata[2]:.4g}}<br>"
                    "Iteration %{customdata[0]}<br>Channel %{customdata[1]}<extra></extra>"
                ),
            ))
        else:
            for _group_id, group_name, channel, group, series_color in colored_series():
                fig.add_trace(go.Scatter(
                    x=group[x_name], y=group[y_name], mode="markers",
                    marker={"size": 8, "color": series_color},
                    name=trace_label(group_name, channel),
                    customdata=group[["iteration", "channel", "value"]],
                    hovertemplate=(
                        f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}<br>"
                        f"{metric_label}: %{{customdata[2]:.4g}}<extra></extra>"
                    ),
                ))
            if group_color_values:
                _add_metadata_colorbar(fig, group_color_values, group_color_label)
        if (
            show_iteration_path
            and iteration_path is not None
            and not iteration_path.empty
        ):
            path_colorbar_added = False
            for _group_id, group_path in iteration_path.groupby("group_id"):
                ordered_path = group_path.dropna(
                    subset=[x_name, y_name]
                ).sort_values("iteration")
                _add_plotly_iteration_path(
                    fig,
                    ordered_path[x_name],
                    ordered_path[y_name],
                    ordered_path["iteration"],
                    show_colorbar=not path_colorbar_added,
                )
                path_colorbar_added = (
                    path_colorbar_added or len(ordered_path) >= 2
                )
        fig.update_layout(
            xaxis_title=x_name,
            yaxis_title=y_name,
            xaxis_type="log" if log_frequency and x_name == "frequency" else "linear",
            yaxis_type="log" if log_frequency and y_name == "frequency" else "linear",
            height=420,
        )
    else:
        fig = go.Figure()
        for _group_id, group_name, channel, group, series_color in colored_series():
            ordered = group.sort_values(x_name)
            line_marker = (
                {"color": series_color}
                if color_by != "Measured value" else {}
            )
            fig.add_trace(go.Scatter(
                x=ordered[x_name],
                y=ordered["value"],
                mode="lines+markers",
                name=trace_label(group_name, channel),
                marker={"size": dot_size, **line_marker},
                line=line_marker or None,
                customdata=ordered[["iteration", "channel"]],
                hovertemplate=(
                    f"{x_name}: %{{x:.4g}}<br>{metric_label}: %{{y:.4g}}<br>"
                    "Iteration %{customdata[0]}<br>Channel %{customdata[1]}<extra></extra>"
                ),
            ))
        if group_color_values:
            _add_metadata_colorbar(fig, group_color_values, group_color_label)
        if show_iteration_path and metric_label != "Count":
            path_colorbar_added = False
            for (
                _group_id,
                _group_name,
                _channel,
                group,
                _series_color,
            ) in colored_series():
                ordered_path = group.sort_values("iteration")
                if len(ordered_path) < 2:
                    continue
                _add_plotly_iteration_path(
                    fig,
                    ordered_path[x_name],
                    ordered_path["value"],
                    ordered_path["iteration"],
                    show_colorbar=not path_colorbar_added,
                )
                path_colorbar_added = True
        fig.update_layout(
            xaxis_title=x_name,
            yaxis_title=metric_label,
            xaxis_type="log" if log_frequency and x_name == "frequency" else "linear",
            height=360,
        )
    fig.update_layout(
        title=f"Measured {phase} {metric_label} | {view}",
        margin={
            "l": 65,
            "r": (
                150
                if show_iteration_path and (
                    iteration_path is not None or metric_label != "Count"
                )
                else 85 if move_legend_below else 30
            ),
            "t": 55,
            "b": 135 if move_legend_below else 55,
        },
        legend=(
            {
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": -0.24,
                "yanchor": "top",
            }
            if move_legend_below else None
        ),
        height=(
            int(fig.layout.height or 400) + 100
            if move_legend_below else fig.layout.height
        ),
    )
    if axis_ranges:
        def display_range(name: str) -> list[float] | None:
            bounds = axis_ranges.get(name)
            if bounds is None:
                return None
            if log_frequency and name == "frequency":
                return [
                    float(np.log10(max(bounds[0], 1e-12))),
                    float(np.log10(max(bounds[1], 1e-12))),
                ]
            return [float(bounds[0]), float(bounds[1])]

        if view == "3D tensor":
            fig.update_scenes(
                xaxis_range=display_range(x_name),
                yaxis_range=display_range(y_name),
                zaxis_range=display_range(z_name),
            )
        else:
            fig.update_xaxes(range=display_range(x_name))
            if view == "2D map":
                fig.update_yaxes(range=display_range(y_name))
            elif value_range is not None:
                fig.update_yaxes(range=list(map(float, value_range)))
    return fig


def _plot_real_data_both_1d(
    buffer_points: pd.DataFrame,
    target_points: pd.DataFrame,
    metric_label: str,
    x_name: str,
    dot_size: int = 6,
    log_frequency: bool = False,
    color_by: str = "Measured value",
    group_color_values: dict[int, float] | None = None,
    group_color_label: str | None = None,
    show_iteration_path: bool = True,
    axis_ranges: dict[str, tuple[float, float]] | None = None,
    value_range: tuple[float, float] | None = None,
):
    fig = go.Figure()
    colors = {"buffer": "#1f77b4", "target": "#ff7f0e"}
    for phase, points in (("buffer", buffer_points), ("target", target_points)):
        if points.empty:
            continue
        channel_values = sorted(
            points["channel"].astype(str).unique(),
            key=_channel_sort_key,
        )
        group_values = sorted(points["group_id"].astype(int).unique())
        for channel_index, (
            (group_id, group_name, channel),
            group,
        ) in enumerate(points.groupby(
            ["group_id", "group_name", "channel"],
            sort=False,
        )):
            ordered = group.sort_values(x_name)
            series_color = colors[phase]
            if color_by == "Channel":
                rgba = plt.get_cmap("tab10")(
                    channel_values.index(str(channel)) % 10
                )
                series_color = (
                    f"rgb({rgba[0] * 255:.0f},{rgba[1] * 255:.0f},"
                    f"{rgba[2] * 255:.0f})"
                )
            elif color_by == "Group":
                rgba = plt.get_cmap("tab10")(
                    group_values.index(int(group_id)) % 10
                )
                series_color = (
                    f"rgb({rgba[0] * 255:.0f},{rgba[1] * 255:.0f},"
                    f"{rgba[2] * 255:.0f})"
                )
            elif group_color_values:
                series_color = _metadata_group_color(
                    group_id,
                    group_color_values,
                )
            fig.add_trace(go.Scatter(
                x=ordered[x_name],
                y=ordered["value"],
                mode="lines+markers",
                name=f"{phase.title()} · {group_name} · Ch {channel}",
                legendgroup=phase,
                showlegend=True,
                line={
                    "color": series_color,
                    "dash": "dash" if phase == "buffer" else "solid",
                },
                marker={"color": series_color, "size": dot_size},
                customdata=ordered[["iteration", "channel"]],
                hovertemplate=(
                    f"{phase.title()}<br>{x_name}: %{{x:.4g}}<br>"
                    f"{metric_label}: %{{y:.4g}}<br>"
                    "Iteration %{customdata[0]}<br>Channel %{customdata[1]}<extra></extra>"
                ),
            ))
    if group_color_values:
        _add_metadata_colorbar(fig, group_color_values, group_color_label)
    if show_iteration_path:
        path_colorbar_added = False
        for _phase, points in (("buffer", buffer_points), ("target", target_points)):
            if points.empty:
                continue
            for _series, group in points.groupby(
                ["group_id", "group_name", "channel"],
                sort=False,
            ):
                ordered_path = group.sort_values("iteration")
                if len(ordered_path) < 2:
                    continue
                _add_plotly_iteration_path(
                    fig,
                    ordered_path[x_name],
                    ordered_path["value"],
                    ordered_path["iteration"],
                    show_colorbar=not path_colorbar_added,
                )
                path_colorbar_added = True
    fig.update_layout(
        title=f"Measured buffer vs target {metric_label} | 1D slice",
        xaxis_title=x_name,
        yaxis_title=metric_label,
        xaxis_type="log" if log_frequency and x_name == "frequency" else "linear",
        height=480 if group_color_values else 380,
        margin={
            "l": 65,
            "r": 150 if show_iteration_path else 85,
            "t": 55,
            "b": 135 if group_color_values else 55,
        },
        legend=(
            {
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": -0.24,
                "yanchor": "top",
            }
            if group_color_values else None
        ),
    )
    if axis_ranges and x_name in axis_ranges:
        x_range = list(map(float, axis_ranges[x_name]))
        if log_frequency and x_name == "frequency":
            x_range = [
                float(np.log10(max(x_range[0], 1e-12))),
                float(np.log10(max(x_range[1], 1e-12))),
            ]
        fig.update_xaxes(range=x_range)
    if value_range is not None:
        fig.update_yaxes(range=list(map(float, value_range)))
    return fig


def _clicked_iteration(event: Any) -> int | None:
    """Extract an iteration from Streamlit's Plotly selection event."""
    try:
        points = event.selection.points
    except (AttributeError, KeyError, TypeError):
        return None
    if not points:
        return None
    point = points[-1]
    raw = point.get("customdata", point.get("x"))
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _pdf_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(.08, .94, title, fontsize=20, weight="bold", va="top")
    fig.text(.08, .89, "\n".join(lines), fontsize=10, va="top", family="monospace", wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_text_pages(
    pdf: PdfPages,
    title: str,
    lines: list[str],
    lines_per_page: int = 44,
) -> None:
    chunks = [
        lines[index:index + lines_per_page]
        for index in range(0, max(len(lines), 1), lines_per_page)
    ]
    for index, chunk in enumerate(chunks):
        page_title = title if len(chunks) == 1 else f"{title} ({index + 1}/{len(chunks)})"
        _pdf_text_page(pdf, page_title, chunk)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "calculating..."
    seconds = int(round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _pdf_save(pdf: PdfPages, fig, title: str | None = None) -> None:
    if title and fig.axes:
        fig.suptitle(title, fontsize=13, y=.995)
    parameter_context = getattr(pdf, "_bo_parameter_context", "")
    if parameter_context:
        fig.text(
            .01, .023,
            f"BO parameter context: {parameter_context}",
            fontsize=5,
            va="bottom",
            wrap=True,
        )
    fig.tight_layout(rect=(0, .05, 1, .97) if title else (0, .05, 1, 1))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_equation_footer(fig, label: str, equation: str) -> None:
    compact = " | ".join(line.strip() for line in equation.splitlines() if line.strip())
    fig.text(.01, .008, f"{label}: {compact}", fontsize=4.5, va="bottom", wrap=True)


def _has_numeric_variation(values) -> bool:
    return pd.to_numeric(pd.Series(values), errors="coerce").dropna().nunique() > 1


def _interpolated_2d_grid(
    points: pd.DataFrame,
    x_name: str,
    y_name: str,
    resolution: int = 120,
) -> tuple[pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    """Interpolate measured points and fill the complete rectangular grid."""
    columns = [x_name, y_name, "value"]
    metadata = [name for name in ("iteration", "channel") if name in points.columns]
    valid = points[columns + metadata].copy()
    valid[columns] = valid[columns].apply(pd.to_numeric, errors="coerce")
    valid = valid.dropna(subset=columns)

    unique = valid.groupby([x_name, y_name], as_index=False)["value"].mean()
    if (
        len(unique) < 3
        or unique[x_name].nunique() < 2
        or unique[y_name].nunique() < 2
    ):
        return valid, None

    grid_x = np.linspace(unique[x_name].min(), unique[x_name].max(), resolution)
    grid_y = np.linspace(unique[y_name].min(), unique[y_name].max(), resolution)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    try:
        coordinates = unique[[x_name, y_name]].to_numpy()
        values = unique["value"].to_numpy()
        grid_values = griddata(
            coordinates,
            values,
            (mesh_x, mesh_y),
            method="linear",
            rescale=True,
        )
        if not np.isfinite(grid_values).all():
            nearest_values = griddata(
                coordinates,
                values,
                (mesh_x, mesh_y),
                method="nearest",
                rescale=True,
            )
            grid_values = np.where(
                np.isfinite(grid_values),
                grid_values,
                nearest_values,
            )
    except Exception:
        try:
            grid_values = griddata(
                unique[[x_name, y_name]].to_numpy(),
                unique["value"].to_numpy(),
                (mesh_x, mesh_y),
                method="nearest",
                rescale=True,
            )
        except Exception:
            return valid, None
    if not np.isfinite(grid_values).any():
        return valid, None
    return valid, (grid_x, grid_y, grid_values)


def _session_parameter_context(observations: list[dict]) -> str:
    labels = {
        "begin_potential": "begin",
        "end_potential": "end",
        "step_potential": "step",
        "amplitude": "amp",
        "frequency": "freq",
        "conditioning_potential": "cond E",
        "conditioning_time": "cond t",
    }
    units = {
        "begin_potential": "V",
        "end_potential": "V",
        "step_potential": "V",
        "amplitude": "V",
        "frequency": "Hz",
        "conditioning_potential": "V",
        "conditioning_time": "s",
    }
    parts = []
    for parameter in PARAMETERS:
        values = sorted({
            float((observation.get("params") or {})[parameter])
            for observation in observations
            if (observation.get("params") or {}).get(parameter) is not None
        })
        if not values:
            continue
        value_text = (
            f"{values[0]:g}"
            if len(values) == 1
            else f"{values[0]:g}–{values[-1]:g}"
        )
        parts.append(f"{labels[parameter]}={value_text} {units[parameter]}")
    return " | ".join(parts)


def _pdf_metric_landscapes(
    pdf: PdfPages,
    points: pd.DataFrame,
    metric: str,
    phase: str,
    dimensions: list[str],
    q_equation: str | None = None,
) -> None:
    metric_varies = (
        not points.empty
        and any(
            _has_numeric_variation(group["value"])
            for _channel, group in points.groupby("channel")
        )
    )
    if not metric_varies or not dimensions:
        return
    # One page of measured 1D projections.
    columns = 2
    rows = (len(dimensions) + 1) // 2
    fig, axes = plt.subplots(rows, columns, figsize=(8.5, max(4, 3.2 * rows)), squeeze=False)
    for ax, parameter in zip(axes.flat, dimensions):
        for channel, group in points.groupby("channel"):
            ordered = group.sort_values(parameter)
            ax.plot(ordered[parameter], ordered["value"], marker="o", label=str(channel))
        ax.set(xlabel=parameter, ylabel=metric, title=f"{metric} vs {parameter}")
        ax.grid(alpha=.25)
    for ax in axes.flat[len(dimensions):]:
        ax.axis("off")
    if points["channel"].nunique() <= 8:
        axes.flat[0].legend(fontsize=7, title="Channel")
    if q_equation:
        _pdf_equation_footer(fig, "Classic Q equation", q_equation)
    _pdf_save(pdf, fig, f"Measured {phase} data — 1D projections")

    # Every available 2D parameter map, grouped four per page.
    pairs = list(itertools.combinations(dimensions, 2))
    for start in range(0, len(pairs), 4):
        page_pairs = pairs[start:start + 4]
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 8), squeeze=False)
        for ax, (x_name, y_name) in zip(axes.flat, page_pairs):
            valid, grid = _interpolated_2d_grid(points, x_name, y_name)
            if grid is not None:
                grid_x, grid_y, grid_values = grid
                scatter = ax.contourf(
                    grid_x, grid_y, grid_values,
                    levels=14, cmap="viridis",
                )
                ax.scatter(
                    valid[x_name], valid[y_name],
                    facecolors="none", edgecolors="white",
                    linewidths=.65, s=20,
                )
            else:
                scatter = ax.scatter(
                    valid[x_name], valid[y_name], c=valid["value"],
                    cmap="viridis", s=28,
                )
            ax.set(xlabel=x_name, ylabel=y_name, title=f"{metric}")
            fig.colorbar(scatter, ax=ax, shrink=.75)
        for ax in axes.flat[len(page_pairs):]:
            ax.axis("off")
        if q_equation:
            _pdf_equation_footer(fig, "Classic Q equation", q_equation)
        _pdf_save(pdf, fig, f"Measured {phase} data — 2D maps")

    # Every available 3D parameter tensor.
    for x_name, y_name, z_name in itertools.combinations(dimensions, 3):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(
            points[x_name], points[y_name], points[z_name],
            c=points["value"], cmap="viridis", s=22,
        )
        ax.set(xlabel=x_name, ylabel=y_name, zlabel=z_name)
        fig.colorbar(scatter, ax=ax, label=metric, shrink=.7)
        if q_equation:
            _pdf_equation_footer(fig, "Classic Q equation", q_equation)
        _pdf_save(
            pdf, fig,
            f"Measured {phase} {metric} — 3D tensor",
        )


def _pdf_chronological_section(
    pdf: PdfPages,
    observations: list[dict],
    config: dict,
    channels: list[str],
    classic_equation: str,
) -> None:
    for metric in _q_relevant_metrics(
        PAIRED_TREND_METRICS, config, paired_objective=True,
    ):
        chronological, transitions = _chronological_points(
            observations, config, metric, channels, average_channels=False,
        )
        if chronological.empty or not _has_numeric_variation(chronological["value"]):
            continue

        summary_fig, (average_ax, overlay_ax) = plt.subplots(1, 2, figsize=(10, 4))
        averaged = (
            chronological.groupby(["position", "phase"], as_index=False)["value"]
            .mean()
            .sort_values("position")
        )
        average_ax.plot(averaged["position"], averaged["value"], color="#777777", alpha=.35)
        for phase, color in (("buffer", "#1f77b4"), ("target", "#ff7f0e")):
            phase_rows = averaged[averaged["phase"] == phase]
            average_ax.scatter(
                phase_rows["position"], phase_rows["value"],
                color=color, s=22, label=phase.title(),
            )
        for channel in channels:
            group = chronological[chronological["channel"] == channel]
            overlay_ax.plot(group["position"], group["value"], color="#777777", alpha=.20)
            for phase, color in (("buffer", "#1f77b4"), ("target", "#ff7f0e")):
                phase_rows = group[group["phase"] == phase]
                overlay_ax.scatter(
                    phase_rows["position"], phase_rows["value"],
                    color=color, s=16,
                )
        for x_value, _label in transitions:
            for ax in (average_ax, overlay_ax):
                ax.axvline(x_value, linestyle="--", color="#5a6b84", linewidth=1)
        average_ax.set_title("Mean across channels")
        overlay_ax.set_title("Channels overlaid")
        average_ax.legend(fontsize=7)
        for ax in (average_ax, overlay_ax):
            ax.set(xlabel="Chronological measurement order", ylabel=metric)
            ax.grid(alpha=.2)
        if metric == "Classic Q":
            _pdf_equation_footer(summary_fig, "Classic Q equation", classic_equation)
        _pdf_save(pdf, summary_fig, f"Chronological {metric} — summary")

        rows = max(1, (len(channels) + 1) // 2)
        individual_fig, individual_axes = plt.subplots(
            rows, 2, figsize=(8.5, max(4, 3 * rows)), squeeze=False,
        )
        for ax, channel in zip(individual_axes.flat, channels):
            group = chronological[
                chronological["channel"] == channel
            ].sort_values("position")
            ax.plot(group["position"], group["value"], color="#777777", alpha=.3)
            for phase, color in (("buffer", "#1f77b4"), ("target", "#ff7f0e")):
                phase_rows = group[group["phase"] == phase]
                ax.scatter(
                    phase_rows["position"], phase_rows["value"],
                    color=color, s=18, label=phase.title(),
                )
            for x_value, _label in transitions:
                ax.axvline(x_value, linestyle="--", color="#5a6b84", linewidth=1)
            ax.set(
                xlabel="Chronological order", ylabel=metric,
                title=f"Channel {channel}",
            )
            ax.grid(alpha=.2)
        for ax in individual_axes.flat[len(channels):]:
            ax.axis("off")
        if channels:
            individual_axes.flat[0].legend(fontsize=7)
        if metric == "Classic Q":
            _pdf_equation_footer(
                individual_fig, "Classic Q equation", classic_equation,
            )
        _pdf_save(
            pdf, individual_fig,
            f"Chronological {metric} — individual channels",
        )


def _pdf_surrogate_iteration(
    pdf: PdfPages,
    artifact_iteration: int,
    path: Path,
    objective_equation_label: str,
    objective_equation: str,
    observations: list[dict],
) -> None:
    predictions = pd.read_csv(path)
    current_observation = next(
        (
            observation for observation in observations
            if int(observation.get("iteration", 0)) == artifact_iteration
        ),
        None,
    )
    parameter_text = _iteration_parameter_text(current_observation)
    surrogate_session = {
        "observations": observations,
        "state": {},
        "config": {},
    }
    tested = [
        observation for observation in observations
        if int(observation.get("iteration", 0)) <= artifact_iteration
    ]
    dimensions = [
        name for name in PARAMETERS
        if name in predictions.columns and predictions[name].nunique(dropna=True) > 1
    ]
    for value_key in SURROGATE_VALUES:
        if (
            value_key not in predictions.columns
            or not _has_numeric_variation(predictions[value_key])
        ):
            continue
        fig, axes = plt.subplots(
            max(1, (len(dimensions) + 1) // 2), 2,
            figsize=(8.5, max(4, 3 * ((len(dimensions) + 1) // 2))),
            squeeze=False,
        )
        for ax, parameter in zip(axes.flat, dimensions):
            ax.scatter(predictions[parameter], predictions[value_key], s=7, alpha=.3)
            grouped = predictions.groupby(parameter)[value_key].median()
            ax.plot(grouped.index, grouped.values, color="#d67b32")
            tested_for_axis = [
                observation for observation in tested
                if (observation.get("params") or {}).get(parameter) is not None
            ]
            if tested_for_axis:
                ax.plot(
                    [observation["params"][parameter] for observation in tested_for_axis],
                    [observation.get("Q_run", np.nan) for observation in tested_for_axis],
                    color="#d67b32",
                    marker="o",
                    linewidth=1.2,
                    label="tested parameter sets",
                )
            ax.set(xlabel=parameter, ylabel=value_key)
            ax.grid(alpha=.2)
        for ax in axes.flat[len(dimensions):]:
            ax.axis("off")
        _pdf_equation_footer(fig, objective_equation_label, objective_equation)
        _pdf_save(
            pdf, fig,
            f"Surrogate iteration {artifact_iteration} — {value_key}\n{parameter_text}",
        )
        pairs = list(itertools.combinations(dimensions, 2))
        for start in range(0, len(pairs), 4):
            page_pairs = pairs[start:start + 4]
            map_fig, map_axes = plt.subplots(2, 2, figsize=(8.5, 8), squeeze=False)
            for ax, (x_name, y_name) in zip(map_axes.flat, page_pairs):
                selected_next = next(
                    (
                        observation for observation in observations
                        if int(observation.get("iteration", 0)) == int(artifact_iteration) + 1
                    ),
                    None,
                )
                selected_next_frame = pd.DataFrame(
                    [selected_next["params"]]
                    if selected_next is not None and selected_next.get("params")
                    else []
                )
                scatter = _plot_surrogate_2d_control_style(
                    ax,
                    surrogate_session,
                    predictions,
                    artifact_iteration,
                    value_key,
                    x_name,
                    y_name,
                    selected_next_frame,
                    dot_size=5,
                    show_iteration_path=True,
                )
                if scatter is not None:
                    colorbar = map_fig.colorbar(scatter, ax=ax, shrink=.75)
                    colorbar.ax.tick_params(labelsize=7)
            for ax in map_axes.flat[len(page_pairs):]:
                ax.axis("off")
            _pdf_equation_footer(
                map_fig, objective_equation_label, objective_equation,
            )
            _pdf_save(
                pdf,
                map_fig,
                f"Surrogate iteration {artifact_iteration} — {value_key} — 2D maps\n"
                f"{parameter_text}",
            )
        for x_name, y_name, z_name in itertools.combinations(dimensions, 3):
            tensor_fig = plt.figure(figsize=(8, 6))
            tensor_ax = tensor_fig.add_subplot(111, projection="3d")
            tensor_scatter = tensor_ax.scatter(
                predictions[x_name],
                predictions[y_name],
                predictions[z_name],
                c=predictions[value_key],
                cmap="viridis",
                s=7,
                alpha=.4,
            )
            tensor_ax.set(xlabel=x_name, ylabel=y_name, zlabel=z_name)
            tested_for_axes = [
                observation for observation in tested
                if all(
                    (observation.get("params") or {}).get(axis) is not None
                    for axis in (x_name, y_name, z_name)
                )
            ]
            if tested_for_axes:
                tensor_ax.plot(
                    [observation["params"][x_name] for observation in tested_for_axes],
                    [observation["params"][y_name] for observation in tested_for_axes],
                    [observation["params"][z_name] for observation in tested_for_axes],
                    color="#d67b32",
                    marker="o",
                    linewidth=1.3,
                    markersize=4,
                )
            tensor_fig.colorbar(
                tensor_scatter,
                ax=tensor_ax,
                label=value_key,
                shrink=.7,
            )
            _pdf_equation_footer(
                tensor_fig, objective_equation_label, objective_equation,
            )
            _pdf_save(
                pdf,
                tensor_fig,
                f"Surrogate iteration {artifact_iteration} — {value_key} — 3D tensor\n"
                f"{parameter_text}",
            )


def _pdf_metadata_lines(
    session: dict,
    observations: list[dict],
    paired: bool,
    best: dict,
    surrogate_iteration_mode: str = "final",
    swv_iteration_mode: str = "milestones",
) -> list[str]:
    def bounded_json_lines(
        value,
        max_lines: int = 120,
        max_line_chars: int = 180,
    ) -> list[str]:
        raw_lines = json.dumps(value, indent=2, default=str).splitlines()
        clipped = [
            line[:max_line_chars] + (" ..." if len(line) > max_line_chars else "")
            for line in raw_lines[:max_lines]
        ]
        if len(raw_lines) > max_lines:
            clipped.append(
                f"... truncated {len(raw_lines) - max_lines} additional metadata lines"
            )
        return clipped

    state = session.get("state") or {}
    config = session.get("config") or {}
    groups = _session_channel_groups(session)
    lines = [
        f"Session: {state.get('session_id', session['root'].name)}",
        f"Root: {session['root']}",
        f"Objective: {'Paired response' if paired else 'Standard quality'}",
        f"Completed observations: {len(observations)}",
        f"Candidate count: {state.get('candidate_count', 'unknown')}",
        (
            f"Best observation: {best.get('group_name', f'Group {best.get('group_id', 1)}')} "
            f"iteration {best.get('iteration')}"
        ),
        f"Best Q_run: {float(best.get('Q_run', 0)):.6g}",
        f"BO parameter context: {_session_parameter_context(observations) or 'none'}",
        "",
        "Report order:",
        "  1. Metadata",
        "  2. History and scores metrics",
        "  3. Real data landscapes: 3D tensors",
        (
            "  4. Surrogate landscapes: 3D tensors and 2D maps "
            f"({'every artifact' if surrogate_iteration_mode == 'all' else 'final artifact per group'})"
        ),
        (
            "  5. Per-iteration SWV traces "
            f"({'every iteration' if swv_iteration_mode == 'all' else 'first/middle/final per group'})"
        ),
        "",
        "Channel groups:",
    ]
    if groups:
        for group in groups:
            channels = ", ".join(str(channel) for channel in group.get("channels", [])) or "unknown"
            lines.append(f"  Group {group['id']}: {group['name']} | channels: {channels}")
    else:
        lines.append("  none")
    lines.extend([
        "",
        "State:",
        *bounded_json_lines(state),
        "",
        "Config:",
        *bounded_json_lines(config),
    ])
    return lines


def _history_metric_specs(
    history: pd.DataFrame,
) -> list[tuple[str, str, dict[str, str] | None]]:
    channel_metrics = _channel_metric_columns(history)
    channel_column_names = {
        column for columns in channel_metrics.values() for column in columns.values()
    }
    global_metrics = [
        column for column in _numeric_columns(history)
        if column not in channel_column_names
    ]
    specs = [("global", metric, None) for metric in global_metrics]
    specs.extend(
        ("channel", metric, columns)
        for metric, columns in sorted(channel_metrics.items())
    )
    return specs


def _pdf_history_metric_page(
    pdf: PdfPages,
    history: pd.DataFrame,
    metric_kind: str,
    metric: str,
    channel_columns: dict[str, str] | None = None,
) -> None:
    if history.empty:
        return
    iterations = pd.to_numeric(
        history.get("iteration", pd.Series(range(1, len(history) + 1), index=history.index)),
        errors="coerce",
    )
    group_ids = (
        pd.to_numeric(history["group_id"], errors="coerce").fillna(1).astype(int)
        if "group_id" in history.columns
        else pd.Series(1, index=history.index)
    )
    group_names = (
        history["group_name"].fillna("").astype(str)
        if "group_name" in history.columns
        else pd.Series("", index=history.index)
    )
    rows = []
    if metric_kind == "channel" and channel_columns:
        for channel, column in channel_columns.items():
            values = pd.to_numeric(history[column], errors="coerce")
            valid = iterations.notna() & values.notna()
            for index in history.index[valid]:
                rows.append({
                    "iteration": int(iterations.loc[index]),
                    "value": float(values.loc[index]),
                    "group_id": int(group_ids.loc[index]),
                    "group_name": group_names.loc[index] or f"Group {int(group_ids.loc[index])}",
                    "channel": str(channel),
                })
    else:
        if metric not in history.columns:
            return
        values = pd.to_numeric(history[metric], errors="coerce")
        valid = iterations.notna() & values.notna()
        for index in history.index[valid]:
            rows.append({
                "iteration": int(iterations.loc[index]),
                "value": float(values.loc[index]),
                "group_id": int(group_ids.loc[index]),
                "group_name": group_names.loc[index] or f"Group {int(group_ids.loc[index])}",
                "channel": "run",
            })
    data = pd.DataFrame(rows)
    if data.empty or not _has_numeric_variation(data["value"]):
        return
    data = (
        data.groupby(["group_id", "group_name", "channel", "iteration"], as_index=False)["value"]
        .mean()
        .sort_values(["group_id", "channel", "iteration"])
    )
    groups = list(data.groupby(["group_id", "group_name"], sort=True))
    bottom_count = max(1, len(groups))
    fig = plt.figure(figsize=(8.5, max(7.5, 3.2 + 2.45 * bottom_count)))
    grid = fig.add_gridspec(bottom_count + 1, 1, height_ratios=[1.25] + [1] * bottom_count)
    overlay_ax = fig.add_subplot(grid[0, 0])
    label = _metric_label(metric)

    for (group_id, group_name, channel), series in data.groupby(["group_id", "group_name", "channel"], sort=True):
        series = series.sort_values("iteration")
        series_label = str(group_name)
        if metric_kind == "channel":
            series_label = f"{series_label} · Ch {channel}"
        overlay_ax.plot(series["iteration"], series["value"], marker="o", linewidth=1.0, label=series_label)
    overlay_ax.set(title=f"{label} — groups overlaid", xlabel="Iteration", ylabel=label)
    overlay_ax.grid(alpha=.25)
    if data[["group_id", "channel"]].drop_duplicates().shape[0] <= 14:
        overlay_ax.legend(fontsize=6, ncol=2)

    for axis_index, ((_group_id, group_name), group) in enumerate(groups):
        ax = fig.add_subplot(grid[axis_index + 1, 0])
        for channel, series in group.groupby("channel", sort=True):
            series = series.sort_values("iteration")
            line_label = f"Ch {channel}" if metric_kind == "channel" else str(group_name)
            ax.plot(series["iteration"], series["value"], marker="o", linewidth=1.0, label=line_label)
        ax.set(title=f"{label} — {group_name}", xlabel="Iteration", ylabel=label)
        ax.grid(alpha=.25)
        if metric_kind == "channel" and group["channel"].nunique() <= 12:
            ax.legend(fontsize=6, ncol=3)
    _pdf_save(pdf, fig, f"History and scores — {label}")


def _pdf_real_data_3d_tensors(
    pdf: PdfPages,
    observations: list[dict],
    config: dict,
    paired: bool,
    channels: list[str],
    status_callback=None,
) -> None:
    del config
    phases = ("buffer", "target") if paired else ("measurement",)
    for phase in phases:
        for metric in REAL_DATA_METRICS:
            points = _real_metric_points(
                observations,
                metric,
                phase,
                channels,
                average_channels=False,
            )
            if points.empty or not _has_numeric_variation(points["value"]):
                continue
            dimensions = [
                name for name in PARAMETERS
                if name in points.columns and points[name].nunique(dropna=True) > 1
            ]
            for x_name, y_name, z_name in itertools.combinations(dimensions, 3):
                if status_callback is not None:
                    status_callback(
                        f"Rendering real data 3D tensor: {phase} {metric} "
                        f"({x_name}, {y_name}, {z_name})..."
                    )
                valid = points[[x_name, y_name, z_name, "value", "channel", "group_name"]].copy()
                valid[[x_name, y_name, z_name, "value"]] = valid[[x_name, y_name, z_name, "value"]].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                valid = valid.dropna(subset=[x_name, y_name, z_name, "value"])
                if valid.empty:
                    continue
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                scatter = ax.scatter(
                    valid[x_name],
                    valid[y_name],
                    valid[z_name],
                    c=valid["value"],
                    cmap="viridis",
                    s=22,
                    alpha=.75,
                )
                ax.set(xlabel=x_name, ylabel=y_name, zlabel=z_name)
                fig.colorbar(scatter, ax=ax, label=metric, shrink=.7)
                _pdf_save(
                    pdf,
                    fig,
                    f"Real data landscape — {phase} — {metric} — 3D tensor",
                )


def _pdf_surrogate_tensor_maps(
    pdf: PdfPages,
    surrogate_session: dict,
    artifact_iteration: int,
    path: Path,
    objective_equation_label: str,
    objective_equation: str,
    status_callback=None,
) -> None:
    if status_callback is not None:
        status_callback(f"Reading surrogate artifact for iteration {artifact_iteration}...")
    predictions = pd.read_csv(path)
    dimensions = [
        name for name in PARAMETERS
        if name in predictions.columns and predictions[name].nunique(dropna=True) > 1
    ]
    parameter_text = _iteration_parameter_text(next(
        (
            observation for observation in surrogate_session["observations"]
            if int(observation.get("iteration", 0)) == int(artifact_iteration)
        ),
        None,
    ))
    for value_key in SURROGATE_VALUES:
        if value_key not in predictions.columns or not _has_numeric_variation(predictions[value_key]):
            continue
        for x_name, y_name, z_name in itertools.combinations(dimensions, 3):
            if status_callback is not None:
                status_callback(
                    f"Rendering surrogate 3D tensor: iteration {artifact_iteration}, "
                    f"{value_key} ({x_name}, {y_name}, {z_name})..."
                )
            valid = predictions[[x_name, y_name, z_name, value_key]].apply(
                pd.to_numeric,
                errors="coerce",
            ).dropna()
            if valid.empty:
                continue
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                valid[x_name],
                valid[y_name],
                valid[z_name],
                c=valid[value_key],
                cmap="viridis",
                s=8,
                alpha=.45,
            )
            observed = _observed_points(
                surrogate_session,
                artifact_iteration,
                [x_name, y_name, z_name],
            )
            if observed:
                ax.plot(
                    [obs["params"][x_name] for obs in observed],
                    [obs["params"][y_name] for obs in observed],
                    [obs["params"][z_name] for obs in observed],
                    color="#d67b32",
                    marker="o",
                    linewidth=1.3,
                    markersize=4,
                    label="observed path",
                )
                ax.legend(fontsize=7)
            ax.set(xlabel=x_name, ylabel=y_name, zlabel=z_name)
            fig.colorbar(
                scatter,
                ax=ax,
                label=SURROGATE_VALUE_LABELS.get(value_key, value_key),
                shrink=.7,
            )
            _pdf_equation_footer(fig, objective_equation_label, objective_equation)
            _pdf_save(
                pdf,
                fig,
                f"Surrogate iteration {artifact_iteration} — {value_key} — 3D tensor\n{parameter_text}",
            )
        pairs = list(itertools.combinations(dimensions, 2))
        for x_name, y_name in pairs:
            if status_callback is not None:
                status_callback(
                    f"Rendering surrogate 2D map: iteration {artifact_iteration}, "
                    f"{value_key} ({x_name} vs {y_name})..."
                )
            fig = _plot_surrogate(
                surrogate_session,
                predictions,
                artifact_iteration,
                value_key,
                "2D map",
                x_name,
                y_name,
                None,
                show_iteration_path=True,
            )
            _pdf_equation_footer(fig, objective_equation_label, objective_equation)
            _pdf_save(
                pdf,
                fig,
                f"Surrogate iteration {artifact_iteration} — {value_key} — 2D {x_name} vs {y_name}\n{parameter_text}",
            )


def _pdf_iteration_swv_overlays(
    pdf: PdfPages,
    session: dict,
    observations: list[dict],
    observation: dict,
    trace_analysis: dict,
    correction_label: str,
    status_callback=None,
) -> None:
    group_id = int(observation.get("group_id", 1))
    iteration = int(observation.get("iteration", 0))
    cumulative_observations = [
        previous for previous in observations
        if int(previous.get("group_id", 1)) == group_id
        and int(previous.get("iteration", 0)) <= iteration
    ]
    trace_entries = [
        (previous, trace)
        for previous in cumulative_observations
        for trace in _trace_paths(session, previous)
    ]
    trace_channels = sorted(
        {trace["channel"] for _previous, trace in trace_entries},
        key=_channel_sort_key,
    )
    if not trace_channels:
        return
    group_name = str(observation.get("group_name") or f"Group {group_id}")
    selection_label = f"{group_name} — through iteration {iteration}"
    trace_specs = [
        ("Raw", False, False, "smoothed_corrected_current"),
        ("Corrected", True, False, "smoothed_corrected_current"),
        ("Normalized corrected", True, True, "smoothed_corrected_current"),
        ("Normalized raw", True, True, "corrected_current"),
    ]
    for label, corrected, normalize_to_peak, corrected_trace_key in trace_specs:
        if status_callback is not None:
            status_callback(
                f"Rendering SWV {label.lower()} traces for {group_name}, "
                f"iteration {iteration}..."
            )
        fig, errors = _plot_iteration_trace_overlay(
            trace_entries,
            corrected,
            trace_channels,
            trace_analysis,
            correction_label,
            selection_label,
            normalize_to_peak,
            corrected_trace_key,
        )
        if errors:
            fig.text(.02, .01, " | ".join(errors[:3]), fontsize=6)
        _pdf_save(pdf, fig)


def _pdf_milestone_observation_keys(
    observations_by_key: dict[tuple[int, int], dict],
) -> list[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()
    by_group: dict[int, list[int]] = {}
    for group_id, iteration in observations_by_key:
        by_group.setdefault(int(group_id), []).append(int(iteration))
    for group_id, iterations in by_group.items():
        ordered = sorted(set(iterations))
        if not ordered:
            continue
        candidate_iterations = {
            ordered[0],
            ordered[len(ordered) // 2],
            ordered[-1],
        }
        selected.update((group_id, iteration) for iteration in candidate_iterations)
    return sorted(selected)


def _pdf_final_artifact_keys(
    artifact_entries: dict[tuple[int, int], tuple[dict, Path]],
) -> list[tuple[int, int]]:
    by_group: dict[int, list[int]] = {}
    for group_id, iteration in artifact_entries:
        by_group.setdefault(int(group_id), []).append(int(iteration))
    return sorted(
        (group_id, max(iterations))
        for group_id, iterations in by_group.items()
        if iterations
    )


def build_bo_session_pdf(
    session: dict,
    trace_analysis: dict,
    correction_label: str,
    progress_callback=None,
    surrogate_iteration_mode: str = "final",
    swv_iteration_mode: str = "milestones",
) -> bytes:
    """Build an exhaustive, shareable report from persisted BO artifacts."""
    observations = session["observations"]
    if progress_callback is not None:
        progress_callback(0.01, "Building history and scores table...")
    history = _observation_table(session)
    config = session["config"]
    if progress_callback is not None:
        progress_callback(0.02, "Indexing objective metadata...")
    paired = any(
        str(obs.get("objective", "")).lower() == "paired_response"
        for obs in observations
    )
    classic_equation = _classic_q_equation(config)
    objective_equation = _paired_q_equation(config) if paired else classic_equation
    objective_equation_label = "Paired Q equation" if paired else "Classic Q equation"
    q_values = [float(obs.get("Q_run", np.nan)) for obs in observations]
    best = observations[int(np.nanargmax(q_values))]
    if progress_callback is not None:
        progress_callback(0.025, "Indexing real-data channels...")
    channels = _real_data_channels(observations)
    if progress_callback is not None:
        progress_callback(0.03, "Indexing history and score metrics...")
    metric_specs = _history_metric_specs(history)
    if progress_callback is not None:
        progress_callback(0.035, "Indexing channel groups...")
    multiple_groups = len({
        int(observation.get("group_id", 1))
        for observation in observations
    }) > 1
    groups = _session_channel_groups(session)
    if progress_callback is not None:
        progress_callback(0.04, "Scanning surrogate artifacts...")
    ungrouped_surrogates, grouped_surrogates = _pdf_surrogate_file_index(session["root"])
    artifact_entries: dict[tuple[int, int], tuple[dict, Path]] = {}
    if multiple_groups:
        for group in groups:
            if progress_callback is not None:
                progress_callback(
                    0.045,
                    f"Indexing surrogate artifacts for {group['name']}...",
                )
            group_session = _session_for_channel_group(session, group["id"])
            for iteration, path in grouped_surrogates.get(int(group["id"]), {}).items():
                artifact_entries[(int(group["id"]), int(iteration))] = (group_session, path)
    else:
        selected_group_id = session.get("selected_group_id")
        source_surrogates = (
            grouped_surrogates.get(int(selected_group_id), {})
            if selected_group_id is not None
            else ungrouped_surrogates
        )
        for iteration, path in source_surrogates.items():
            artifact_entries[(1, int(iteration))] = (session, path)
    if progress_callback is not None:
        progress_callback(0.05, "Indexing completed observations...")
    observations_by_key = {
        (int(observation.get("group_id", 1)), int(observation.get("iteration", 0))): observation
        for observation in observations
    }
    artifact_report_keys = (
        sorted(artifact_entries)
        if surrogate_iteration_mode == "all"
        else _pdf_final_artifact_keys(artifact_entries)
    )
    swv_report_keys = (
        sorted(observations_by_key)
        if swv_iteration_mode == "all"
        else _pdf_milestone_observation_keys(observations_by_key)
    )
    total_steps = (
        7
        + len(metric_specs)
        + max(1, len(artifact_report_keys))
        + max(1, len(swv_report_keys))
    )
    completed_steps = max(1, int(total_steps * 0.05))

    def report_progress(text: str, increment: int = 1) -> None:
        nonlocal completed_steps
        completed_steps += increment
        if progress_callback is not None:
            progress_callback(
                min(completed_steps / max(total_steps, 1), 0.99),
                text,
            )

    output = BytesIO()
    with PdfPages(output) as pdf:
        pdf._bo_parameter_context = _session_parameter_context(observations)
        if progress_callback is not None:
            progress_callback(0.06, "Rendering metadata pages...")
        _pdf_text_pages(
            pdf,
            "1. Metadata",
            _pdf_metadata_lines(
                session,
                observations,
                paired,
                best,
                surrogate_iteration_mode,
                swv_iteration_mode,
            ),
        )
        report_progress("Rendered metadata.")
        _pdf_text_page(
            pdf,
            "Q Definition",
            (_paired_q_equation(config) if paired else _classic_q_equation(config)).splitlines(),
        )
        if paired:
            _pdf_text_page(
                pdf,
                "Classic Q Definition (inputs to Paired Q)",
                _classic_q_equation(config).splitlines(),
            )
        report_progress("Rendered Q definitions.")

        _pdf_text_page(pdf, "2. History and Scores Metrics", [
            "Every changing numeric metric from the History & scores tab is plotted.",
            "For each metric, groups are overlaid first; separate group plots appear below.",
        ])
        if not metric_specs:
            _pdf_text_page(pdf, "History and Scores Metrics Not Available", [
                "No changing numeric metrics were found in the history table.",
            ])
        for metric_kind, metric, channel_columns in metric_specs:
            _pdf_history_metric_page(
                pdf,
                history,
                metric_kind,
                metric,
                channel_columns,
            )
            report_progress(f"Rendered history metric: {_metric_label(metric)}.")

        _pdf_text_page(pdf, "3. Real Data Landscapes", [
            "This section includes 3D tensor plots for every available real-data metric",
            "and every varied 3-parameter combination.",
        ])
        _pdf_real_data_3d_tensors(
            pdf,
            observations,
            config,
            paired,
            channels,
            status_callback=lambda text: report_progress(text, 0),
        )
        report_progress("Rendered real data landscapes.")

        _pdf_text_page(pdf, "4. Surrogate Landscapes", [
            "This section includes surrogate 3D tensor plots and 2D maps for every",
            "available surrogate artifact, value, and varied parameter combination.",
        ])
        if not artifact_entries:
            _pdf_text_page(pdf, "Surrogate Landscapes Not Available", [
                "No persisted surrogate candidate prediction files were found.",
            ])
            report_progress("No surrogate landscapes available.")
        for group_id, iteration in artifact_report_keys:
            surrogate_session, artifact_path = artifact_entries[(group_id, iteration)]
            _pdf_surrogate_tensor_maps(
                pdf,
                surrogate_session,
                iteration,
                artifact_path,
                objective_equation_label,
                objective_equation,
                status_callback=lambda text: report_progress(text, 0),
            )
            report_progress(f"Rendered surrogate group {group_id}, iteration {iteration}.")

        _pdf_text_page(pdf, "5. Per-Iteration SWV Traces", [
            "For each completed iteration, cumulative SWV overlays include raw, corrected,",
            "normalized corrected, and normalized raw traces through that iteration.",
        ])
        if not observations_by_key:
            report_progress("No SWV iterations available.")
        for report_key in swv_report_keys:
            observation = observations_by_key[report_key]
            _pdf_iteration_swv_overlays(
                pdf,
                session,
                observations,
                observation,
                trace_analysis,
                correction_label,
                status_callback=lambda text: report_progress(text, 0),
            )
            report_progress(
                f"Rendered SWV traces for group {observation.get('group_id', 1)}, "
                f"iteration {observation.get('iteration', 0)}."
            )
        report_progress("Finalized PDF.")
    if progress_callback is not None:
        progress_callback(1.0, "PDF ready.")
    return output.getvalue()


def _channel_table(observation: dict) -> pd.DataFrame:
    quality = observation.get("quality") or {}
    components = quality.get("channel_components") or {}
    metrics = observation.get("channel_metrics") or {}
    channels = sorted(set(components) | set(metrics), key=lambda value: int(value) if str(value).isdigit() else str(value))
    rows = []
    for channel in channels:
        component = components.get(channel, {}) or {}
        metric = metrics.get(channel, {}) or {}
        rows.append({
            "Channel": channel,
            "Q": (quality.get("Q_channels") or {}).get(channel, component.get("Q_channel")),
            "Peak uA": metric.get("mean_peak_current_uA", metric.get("median_peak_current_uA")),
            "Raw SNR": metric.get("snr_unadjusted", metric.get("snr")),
            "SNR Score": component.get("snr_score", component.get("target_snr_score")),
            "Shape": metric.get("peak_shape_score"),
            "Baseline": metric.get("baseline_stability_score"),
            "Replicate": metric.get("replicate_consistency_score"),
            "Success": metric.get("success_score"),
        })
    return pd.DataFrame(rows)


@lru_cache(maxsize=8192)
def _resolve_recorded_path_cached(
    root_text: str,
    raw_text: str,
) -> str | None:
    root = Path(root_text)
    direct = Path(raw_text)
    try:
        if direct.is_file():
            return direct
    except OSError:
        # A path recorded on another OS may be interpreted as one invalid local
        # filename (for example, a long Windows path loaded on macOS).
        pass
    name = PureWindowsPath(raw_text).name
    if not name:
        return None

    def find_recorded_file(search_root: Path) -> Path | None:
        match = _recorded_file_index(str(search_root)).get(name)
        return Path(match) if match is not None else None

    match = find_recorded_file(root)
    if match is not None:
        return str(match)
    # Archived measurements commonly sit beside bo_sessions in the experiment folder.
    for parent in list(root.parents)[:3]:
        match = find_recorded_file(parent)
        if match is not None:
            return str(match)
    return None


@lru_cache(maxsize=12)
def _recorded_file_index(root_text: str) -> dict[str, str]:
    """Index a search root once instead of recursively scanning per trace."""
    files: dict[str, str] = {}

    def ignore_walk_error(_error: OSError) -> None:
        return None

    for directory, _subdirectories, names in os.walk(
        root_text,
        onerror=ignore_walk_error,
    ):
        for name in names:
            files.setdefault(name, str(Path(directory) / name))
    return files


def _resolve_recorded_path(root: Path, raw: Any) -> Path | None:
    if not raw:
        return None
    resolved = _resolve_recorded_path_cached(str(root), str(raw))
    return Path(resolved) if resolved is not None else None


def _channel_from_path(path: Path) -> str:
    match = re.search(r"(?:^|[_-])ch(?:annel)?[_-]?(\d+)(?:[_\-.]|$)", path.name, re.IGNORECASE)
    return match.group(1) if match else "Unknown"


def _phase_from_path(raw: Any) -> str | None:
    text = str(raw).replace("\\", "/").lower()
    parts = [part for part in text.split("/") if part]
    for part in reversed(parts):
        if "buffer" in part:
            return "buffer"
        if "target" in part:
            return "target"
    return None


def _trace_paths(session: dict, observation: dict) -> list[dict]:
    paths: list[dict] = []
    root = session["root"]

    def add(raw_path: Any, phase: str | None = None, channel: Any = None) -> None:
        path = _resolve_recorded_path(root, raw_path)
        if not path or not path.is_file() or path.suffix.lower() != ".csv":
            return
        resolved_phase = phase or _phase_from_path(raw_path) or _phase_from_path(path) or "unknown"
        has_channel = channel is not None and pd.notna(channel) and str(channel).strip()
        resolved_channel = str(channel).strip() if has_channel else _channel_from_path(path)
        if resolved_channel.endswith(".0") and resolved_channel[:-2].isdigit():
            resolved_channel = resolved_channel[:-2]
        existing = next((item for item in paths if item["path"] == path), None)
        if existing is not None:
            # Explicit analysis-record metadata is more reliable than path inference.
            if phase in ("buffer", "target"):
                existing["phase"] = phase
            if resolved_channel != "Unknown":
                existing["channel"] = resolved_channel
            return
        paths.append({
            "phase": resolved_phase,
            "path": path,
            "channel": resolved_channel,
        })

    # Paired records explicitly identify buffer versus target. Standard BO
    # stores one analysis_record for the measurement.
    paired = str(observation.get("objective") or "").lower() == "paired_response"
    record_specs = (
        [
            ("buffer", observation.get("buffer_analysis_record")),
            ("target", observation.get("target_analysis_record")),
        ]
        if paired
        else [("measurement", observation.get("analysis_record"))]
    )
    for phase, raw_record in record_specs:
        record = _resolve_recorded_path(root, raw_record)
        if not record:
            continue
        try:
            payload = _read_json(record)
        except Exception:
            continue
        results_csv = _resolve_recorded_path(root, payload.get("results_csv"))
        if results_csv:
            try:
                results = pd.read_csv(results_csv)
                for _, row in results.iterrows():
                    recorded_path = row.get("file_path")
                    if recorded_path is None or pd.isna(recorded_path) or not str(recorded_path).strip():
                        recorded_path = row.get("file_name")
                    add(
                        recorded_path,
                        phase=phase,
                        channel=row.get("channel"),
                    )
            except Exception:
                pass
        for raw in payload.get("folders") or []:
            add(raw, phase=phase)

    for raw in observation.get("archived_measurements") or []:
        add(raw, phase=_phase_from_path(raw))

    return paths


def _channel_sort_key(channel: str):
    return (0, int(channel)) if str(channel).isdigit() else (1, str(channel))


def _trace_channel_key(item: dict) -> str:
    return str(item.get("display_channel") or item["channel"])


def _trace_channel_label(channel: Any) -> str:
    text = str(channel)
    if text == "Unknown":
        return "Unknown channel"
    if " · " in text:
        return text
    return f"Ch {text}"


def _group_qualified_trace(
    observation: dict,
    trace: dict,
    include_group: bool,
) -> dict:
    if not include_group:
        return trace
    group_name = str(
        observation.get("group_name")
        or f"Group {observation.get('group_id', 1)}"
    )
    qualified = dict(trace)
    qualified["display_channel"] = (
        f"{group_name} · {_trace_channel_label(trace['channel'])}"
    )
    return qualified


@st.cache_data(show_spinner=False)
def _cached_swv_arrays(
    path_text: str,
    modified_ns: int,
) -> tuple[np.ndarray, np.ndarray, int | None, int | None, int | None]:
    """Load an unchanged SWV file once per Streamlit process."""
    del modified_ns  # Included in the cache key to invalidate changed files.
    return load_swv_csv(path_text)


@st.cache_data(show_spinner=False)
def _cached_corrected_swv_arrays(
    path_text: str,
    modified_ns: int,
    corrected_trace_key: str,
    crop_min_v: float,
    crop_max_v: float,
    smooth_window: int,
    smooth_polyorder: int,
    minima_search_window_v: float,
    use_prominent_minima: bool,
    use_double_correction: bool,
    min_peak_height_uA: float | None,
    compute_wavelet_denoised_trace: bool,
    use_wavelet_for_correction: bool,
) -> tuple[np.ndarray, np.ndarray, int | None, int | None, int | None]:
    voltage, current = _cached_swv_arrays(path_text, modified_ns)
    result = analyze_swv_arrays(
        voltage,
        current,
        crop_range=(crop_min_v, crop_max_v),
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
        minima_search_window_V=minima_search_window_v,
        use_prominent_minima=use_prominent_minima,
        use_double_correction=use_double_correction,
        min_peak_height_uA=min_peak_height_uA,
        compute_wavelet_denoised_trace=compute_wavelet_denoised_trace,
        use_wavelet_for_correction=use_wavelet_for_correction,
    )
    corrected_voltage = result.get(
        "cropped_voltage",
        result.get("voltage", voltage),
    )
    corrected_current = result.get(corrected_trace_key)
    if corrected_current is None:
        corrected_current = result.get(
            "smoothed_corrected_current",
            result.get("corrected_current"),
        )
    return (
        corrected_voltage,
        corrected_current,
        result.get("peak_idx_corr"),
        result.get("left_min_idx"),
        result.get("right_min_idx"),
    )


def _swv_trace_arrays(
    path: Path,
    corrected: bool,
    analysis: dict,
    corrected_trace_key: str = "smoothed_corrected_current",
) -> tuple[np.ndarray, np.ndarray, int | None, int | None, int | None]:
    modified_ns = path.stat().st_mtime_ns
    if not corrected:
        voltage, current = _cached_swv_arrays(str(path), modified_ns)
        return voltage, current, None, None, None
    minimum_peak = analysis.get("min_peak_height_uA")
    return _cached_corrected_swv_arrays(
        str(path),
        modified_ns,
        corrected_trace_key,
        float(analysis.get("crop_min_v", -.45)),
        float(analysis.get("crop_max_v", 0)),
        int(analysis.get("smooth_window", 3)),
        int(analysis.get("smooth_polyorder", 2)),
        float(analysis.get("minima_search_window_v", .3)),
        bool(analysis.get("use_prominent_minima", False)),
        bool(analysis.get("use_double_correction", True)),
        float(minimum_peak) if minimum_peak is not None else None,
        bool(analysis.get("compute_wavelet_denoised_trace", False)),
        bool(analysis.get("use_wavelet_for_correction", False)),
    )


def _normalize_trace_to_peak(
    y: np.ndarray,
    peak_idx: int | None = None,
    left_idx: int | None = None,
    right_idx: int | None = None,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if left_idx is not None and right_idx is not None:
        left = max(0, int(left_idx))
        right = min(len(y) - 1, int(right_idx))
        if left <= right:
            peak_region = y[left:right + 1]
            finite_region = peak_region[np.isfinite(peak_region)]
            if len(finite_region):
                peak_height = float(np.nanmax(finite_region))
            else:
                peak_height = np.nan
        else:
            peak_height = np.nan
    elif peak_idx is not None and 0 <= int(peak_idx) < len(y):
        peak_height = float(y[int(peak_idx)])
    else:
        raise ValueError("Trace has no detected peak index to normalize.")
    if not np.isfinite(peak_height) or np.isclose(peak_height, 0.0):
        raise ValueError("Trace peak height is zero or non-finite.")
    return y / peak_height


def _swv_global_y_limits(
    trace_observations: list[dict],
    session: dict,
    selected_channels: list[str],
    corrected: bool,
    analysis: dict,
    normalize_to_peak: bool,
    corrected_trace_key: str,
) -> tuple[float, float] | None:
    if normalize_to_peak:
        return (-0.2, 1.2)
    values = []
    selected_channel_set = set(selected_channels)
    for observation in trace_observations:
        for item in _trace_paths(session, observation):
            if item["channel"] not in selected_channel_set:
                continue
            try:
                _voltage, y, _peak_idx, _left_idx, _right_idx = _swv_trace_arrays(
                    item["path"],
                    corrected,
                    analysis,
                    corrected_trace_key,
                )
            except Exception:
                continue
            y = np.asarray(y, dtype=float)
            finite = y[np.isfinite(y)]
            if len(finite):
                values.append(finite)
    if not values:
        return None
    merged = np.concatenate(values)
    y_min = float(np.nanmin(merged))
    y_max = float(np.nanmax(merged))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None
    if np.isclose(y_min, y_max):
        padding = max(abs(y_min) * 0.05, 0.1)
    else:
        padding = (y_max - y_min) * 0.05
    return y_min - padding, y_max + padding


def _swv_trace_kind_label(
    corrected: bool,
    normalize_to_peak: bool,
    corrected_trace_key: str,
) -> str:
    if not corrected:
        return "raw"
    if normalize_to_peak and corrected_trace_key == "corrected_current":
        return "normalized raw"
    if normalize_to_peak:
        return "normalized corrected"
    if corrected_trace_key == "corrected_current":
        return "raw corrected"
    return "corrected"


def _plot_traces(
    session: dict,
    observation: dict,
    corrected: bool,
    selected_channels: list[str],
    analysis: dict,
    correction_label: str,
    overlaid: bool,
    trace_items: list[dict] | None = None,
    normalize_to_peak: bool = False,
    corrected_trace_key: str = "smoothed_corrected_current",
):
    traces = [
        item for item in (
            trace_items
            if trace_items is not None
            else _trace_paths(session, observation)
        )
        if _trace_channel_key(item) in selected_channels
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    errors = []
    trace_colors = plt.get_cmap("turbo")(
        np.linspace(.03, .97, max(len(traces), 2))
    )
    phase_styles = {
        "buffer": "--",
        "target": "-",
        "measurement": "-",
        "unknown": ":",
    }
    for trace_index, item in enumerate(traces):
        phase, path, channel = item["phase"], item["path"], item["channel"]
        try:
            voltage, y, peak_idx, left_idx, right_idx = _swv_trace_arrays(
                path,
                corrected,
                analysis,
                corrected_trace_key,
            )
            if normalize_to_peak:
                y = _normalize_trace_to_peak(y, peak_idx, left_idx, right_idx)
            channel_label = _trace_channel_label(_trace_channel_key(item))
            trace_label = f"{phase} {channel_label}: {path.stem}"
            ax.plot(
                voltage,
                y,
                linewidth=1.1,
                color=trace_colors[trace_index],
                linestyle=phase_styles.get(
                    str(phase).lower(),
                    phase_styles["unknown"],
                ),
                label=trace_label,
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if not traces:
        message = (
            "No traces match the selected channels."
            if selected_channels else
            "Select at least one channel to display traces."
        )
        ax.text(.5, .5, message, ha="center", va="center")
        ax.set_axis_off()
    else:
        channel_title = ""
        if not overlaid:
            channel_title = " | " + ", ".join(
                _trace_channel_label(channel)
                for channel in selected_channels
            )
        params = observation.get("params") or {}
        parameter_text = (
            f"Frequency={float(params['frequency']):g} Hz"
            if params.get("frequency") is not None else "Frequency=unknown"
        )
        parameter_text += (
            f" | Step size={float(params['step_potential']):g} V"
            if params.get("step_potential") is not None else " | Step size=unknown"
        )
        parameter_text += (
            f" | Amplitude={float(params['amplitude']):g} V"
            if params.get("amplitude") is not None else " | Amplitude=unknown"
        )
        ax.set(
            xlabel="Voltage (V)",
            ylabel="Normalized current (peak = 1)" if normalize_to_peak else "Current (uA)",
            title=(
                f"Iteration {observation.get('iteration')} "
                f"{_swv_trace_kind_label(corrected, normalize_to_peak, corrected_trace_key)} SWV traces"
                f"{channel_title}"
                f"{f' ({correction_label})' if corrected else ''}"
                f"\n{parameter_text}"
            ),
        )
        if normalize_to_peak:
            ax.set_ylim(-0.2, 1.2)
        ax.grid(alpha=.25)
        if len(traces) <= 16:
            ax.legend(fontsize=7)
    # Keep raw and corrected plots geometrically identical. ``tight_layout``
    # otherwise assigns different axes sizes when their titles or tick labels
    # require different amounts of space.
    fig.subplots_adjust(left=.12, right=.97, bottom=.16, top=.76)
    return fig, errors


def _plot_iteration_trace_overlay(
    trace_entries: list[tuple[dict, dict]],
    corrected: bool,
    selected_channels: list[str],
    analysis: dict,
    correction_label: str,
    selection_label: str,
    normalize_to_peak: bool = False,
    corrected_trace_key: str = "smoothed_corrected_current",
):
    """Overlay traces from many observations with channel-aware colors."""
    entries = [
        (observation, item)
        for observation, item in trace_entries
        if _trace_channel_key(item) in selected_channels
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    errors = []
    iterations = [
        int(observation.get("iteration", 0))
        for observation, _item in entries
    ]
    plotted_channels = sorted(
        {_trace_channel_key(item) for _observation, item in entries},
        key=_channel_sort_key,
    )
    single_channel = len(plotted_channels) <= 1
    if single_channel:
        iteration_minimum = min(iterations, default=0)
        iteration_maximum = max(iterations, default=0)
        iteration_norm = plt.Normalize(
            vmin=(
                iteration_minimum
                if iteration_maximum > iteration_minimum
                else iteration_minimum - .5
            ),
            vmax=(
                iteration_maximum
                if iteration_maximum > iteration_minimum
                else iteration_maximum + .5
            ),
        )
        iteration_cmap = plt.get_cmap("viridis")
        channel_colors = {}
    else:
        categorical_colors = plt.get_cmap("turbo")(
            np.linspace(.03, .97, max(len(plotted_channels), 2))
        )
        channel_colors = {
            channel: categorical_colors[index]
            for index, channel in enumerate(plotted_channels)
        }
        iteration_norm = None
        iteration_cmap = None
    phase_styles = {
        "buffer": "--",
        "target": "-",
        "measurement": "-",
        "unknown": ":",
    }
    for observation, item in entries:
        phase, path, channel = item["phase"], item["path"], _trace_channel_key(item)
        iteration = int(observation.get("iteration", 0))
        try:
            voltage, y, peak_idx, left_idx, right_idx = _swv_trace_arrays(
                path,
                corrected,
                analysis,
                corrected_trace_key,
            )
            if normalize_to_peak:
                y = _normalize_trace_to_peak(y, peak_idx, left_idx, right_idx)
            ax.plot(
                voltage,
                y,
                linewidth=1.0,
                alpha=.72,
                color=(
                    iteration_cmap(iteration_norm(iteration))
                    if single_channel
                    else channel_colors[channel]
                ),
                linestyle=phase_styles.get(
                    str(phase).lower(),
                    phase_styles["unknown"],
                ),
                label=(
                    f"Iter {iteration} · {str(phase).title()} · "
                    f"{_trace_channel_label(channel)}"
                    f": {path.stem}"
                ),
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    if not entries:
        ax.text(
            .5,
            .5,
            "No traces match the selected channels.",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
    else:
        channel_text = ", ".join(
            _trace_channel_label(channel)
            for channel in selected_channels
        )
        ax.set(
            xlabel="Voltage (V)",
            ylabel="Normalized current (peak = 1)" if normalize_to_peak else "Current (µA)",
            title=(
                f"{selection_label}\n"
                f"{_swv_trace_kind_label(corrected, normalize_to_peak, corrected_trace_key).capitalize()} SWV traces"
                f"{f' ({correction_label})' if corrected else ''}"
                f" | {channel_text}"
            ),
        )
        if normalize_to_peak:
            ax.set_ylim(-0.2, 1.2)
        ax.grid(alpha=.25)
        if single_channel:
            unique_iterations = sorted(set(iterations))
            colorbar_ticks = (
                unique_iterations
                if len(unique_iterations) <= 10
                else sorted({
                    int(round(value))
                    for value in np.linspace(
                        iteration_minimum,
                        iteration_maximum,
                        8,
                    )
                })
            )
            scalar_map = plt.cm.ScalarMappable(
                norm=iteration_norm,
                cmap=iteration_cmap,
            )
            scalar_map.set_array([])
            fig.colorbar(
                scalar_map,
                ax=ax,
                label="BO iteration",
                ticks=colorbar_ticks,
            )
        if not single_channel:
            channel_handles = [
                Line2D(
                    [0],
                    [0],
                    color=channel_colors[channel],
                    linewidth=2,
                    label=(
                        f"Ch {channel}"
                        if channel != "Unknown"
                        else "Unknown channel"
                    ),
                )
                for channel in plotted_channels
            ]
            ax.legend(
                handles=channel_handles,
                title="Channel color",
                fontsize=7,
                title_fontsize=8,
            )
        elif len(entries) <= 16:
            ax.legend(fontsize=7)
    fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.78)
    return fig, errors


def _bo_analysis_settings(config: dict) -> dict:
    analysis = dict((config or {}).get("analysis") or {})
    return {
        "crop_min_v": float(analysis.get("crop_min_v", -.45)),
        "crop_max_v": float(analysis.get("crop_max_v", 0.0)),
        "smooth_window": int(analysis.get("smooth_window", 3)),
        "smooth_polyorder": int(analysis.get("smooth_polyorder", 2)),
        "minima_search_window_v": float(analysis.get("minima_search_window_v", .3)),
        "use_prominent_minima": bool(analysis.get("use_prominent_minima", False)),
        "use_double_correction": bool(analysis.get("use_double_correction", True)),
        "min_peak_height_uA": analysis.get("min_peak_height_ua"),
        "compute_wavelet_denoised_trace": bool(
            analysis.get("compute_wavelet_denoised_trace", False)
        ),
        "use_wavelet_for_correction": bool(
            analysis.get("use_wavelet_for_correction", False)
        ),
    }


def _native_analysis_settings_sidebar() -> dict:
    st.markdown("##### Native SWV processing")
    crop_columns = st.columns(2)
    crop_min = crop_columns[0].number_input(
        "Crop min (V)", value=-0.47, step=0.01, format="%.3f",
        key="bo_native_crop_min",
    )
    crop_max = crop_columns[1].number_input(
        "Crop max (V)", value=-0.15, step=0.01, format="%.3f",
        key="bo_native_crop_max",
    )
    smooth_window = st.slider(
        "Savitzky-Golay window", 3, 31, 15, 2,
        key="bo_native_smooth_window",
    )
    smooth_polyorder = st.slider(
        "Polynomial order", 1, 5, 2,
        key="bo_native_smooth_polyorder",
    )
    minima_window = st.number_input(
        "Minima search window (V)", value=0.30, step=0.01, format="%.3f",
        key="bo_native_minima_window",
    )
    double_correction = st.checkbox(
        "Double baseline correction", value=True,
        key="bo_native_double_correction",
    )
    enforce_peak = st.checkbox(
        "Enforce min peak height", value=True,
        key="bo_native_enforce_peak",
    )
    min_peak = (
        st.number_input(
            "Min peak height (uA)", value=0.001, step=0.001, format="%.3f",
            key="bo_native_min_peak",
        )
        if enforce_peak else None
    )
    with st.expander("Experimental", expanded=False):
        prominent_minima = st.checkbox(
            "Use prominent local minima for bracketing",
            value=False,
            key="bo_native_prominent_minima",
        )
        wavelet_trace = st.checkbox(
            "Compute wavelet-denoised trace",
            value=False,
            key="bo_native_wavelet_trace",
        )
        wavelet_correction = st.checkbox(
            "Use wavelet-denoised trace for baseline correction",
            value=False,
            disabled=not wavelet_trace,
            key="bo_native_wavelet_correction",
        )
    return {
        "crop_min_v": crop_min,
        "crop_max_v": crop_max,
        "smooth_window": smooth_window,
        "smooth_polyorder": smooth_polyorder,
        "minima_search_window_v": minima_window,
        "use_prominent_minima": prominent_minima,
        "use_double_correction": double_correction,
        "min_peak_height_uA": min_peak,
        "compute_wavelet_denoised_trace": wavelet_trace,
        "use_wavelet_for_correction": wavelet_correction,
    }


def _classic_q_equation(config: dict) -> str:
    scoring = dict((config or {}).get("scoring") or {})
    mode = str(scoring.get("mode", "classic") or "classic").strip().lower()
    weights = dict(scoring.get("channel_weights") or {})
    run = dict(scoring.get("run_weights") or {})
    if mode == "signal_priority_unbounded":
        terms = (
            ("log1p(Raw SNR)", float(weights.get("snr", .45))),
            ("log1p(Peak µA)", float(weights.get("peak_height", .35))),
            ("Baseline", float(weights.get("baseline", .12))),
            ("Shape", float(weights.get("peak_shape", .05))),
            ("Replicate", float(weights.get("replicate_consistency", .03))),
            ("Success", float(weights.get("success", 0))),
        )
        total = max(sum(weight for _name, weight in terms), 1e-12)
        numerator = " + ".join(
            f"{weight:g}·{name}" for name, weight in terms if weight
        ) or "0"
        channel_equation = f"Classic Q_channel = ({numerator}) / {total:g}"
    else:
        terms = (
            ("Raw SNR", float(weights.get("snr", .35))),
            ("Peak µA", float(weights.get("peak_height", 0))),
            ("Shape", float(weights.get("peak_shape", .20))),
            ("Baseline", float(weights.get("baseline", .20))),
            ("Replicate", float(weights.get("replicate_consistency", .15))),
            ("Success", float(weights.get("success", .10))),
        )
        numerator = " + ".join(
            f"{weight:g}·{name}" for name, weight in terms if weight
        ) or "0"
        noise = float(weights.get("noise_penalty", 0))
        channel_equation = (
            f"Classic Q_channel = max(0, {numerator} - {noise:g}·Noise µA)"
        )
    run_equation = (
        "Classic Q_run = mean(Q_channel)"
        f" - {float(run.get('lambda_variability', .20)):g}·std(Q_channel)"
        f" - {float(run.get('lambda_failed', .40)):g}·failed_fraction"
        f" - {float(run.get('lambda_low', .20)):g}·"
        f"fraction(Q_channel < {float(run.get('low_channel_threshold', .50)):g})"
    )
    return f"{channel_equation}\n{run_equation}"


def _paired_q_equation(config: dict) -> str:
    scoring = dict((config or {}).get("scoring") or {})
    weights = dict(scoring.get("paired_response_weights") or {})
    if (
        "standard_quality" in weights
        and "buffer_classic_Q" not in weights
        and "target_classic_Q" not in weights
    ):
        legacy = max(0.0, float(weights.get("standard_quality", 0) or 0))
        weights["buffer_classic_Q"] = legacy / 2
        weights["target_classic_Q"] = legacy / 2
    buffer_weight = float(weights.get("buffer_classic_Q", .25))
    target_weight = float(weights.get("target_classic_Q", .25))
    delta_weight = float(weights.get("delta_peak", 1))
    delta_scale = max(float(weights.get("delta_scale_uA", 1)), 1e-12)
    total = max(buffer_weight + target_weight + delta_weight, 1e-12)
    run = dict(scoring.get("run_weights") or {})
    return (
        "Δpeak = target_peak_height_µA - buffer_peak_height_µA\n"
        f"Δpeak_score = log1p(|Δpeak| / {delta_scale:g})\n"
        "Paired Q_channel = "
        f"({buffer_weight:g}·buffer_classic_Q + "
        f"{target_weight:g}·target_classic_Q + "
        f"{delta_weight:g}·Δpeak_score) / {total:g}\n"
        "If buffer_classic_Q ≤ 0 or target_classic_Q ≤ 0, Paired Q_channel = 0\n"
        "Paired Q_run = max(0, mean(Paired Q_channel)"
        f" - {float(run.get('lambda_variability', .20)):g}·std(Paired Q_channel)"
        f" - {float(run.get('lambda_failed', .40)):g}·failed_fraction"
        f" - {float(run.get('lambda_low', .20)):g}·"
        f"fraction(Paired Q_channel < {float(run.get('low_channel_threshold', .50)):g}))"
    )


def _render_q_equation(config: dict, kind: str) -> None:
    if kind == "paired":
        st.markdown("**Paired Q equation**")
        st.code(_paired_q_equation(config), language=None)
    elif kind == "classic":
        st.markdown("**Classic Q equation**")
        st.code(_classic_q_equation(config), language=None)


def _metric_q_kind(metric: str, paired_objective: bool) -> str | None:
    normalized = str(metric).lower()
    if "contribution" in normalized and "q" in normalized:
        return "paired"
    if "classic_q" in normalized or "classic q" in normalized:
        return "classic"
    if "q" in normalized:
        return "paired" if paired_objective else "classic"
    return None


def _q_weight_context(config: dict) -> dict:
    scoring = dict((config or {}).get("scoring") or {})
    channel = dict(scoring.get("channel_weights") or {})
    paired = dict(scoring.get("paired_response_weights") or {})
    if (
        "standard_quality" in paired
        and "buffer_classic_Q" not in paired
        and "target_classic_Q" not in paired
    ):
        legacy = max(0.0, float(paired.get("standard_quality", 0) or 0))
        paired["buffer_classic_Q"] = legacy / 2
        paired["target_classic_Q"] = legacy / 2
    return {
        "snr": float(channel.get("snr", .35)),
        "peak": float(channel.get("peak_height", 0)),
        "shape": float(channel.get("peak_shape", .20)),
        "baseline": float(channel.get("baseline", .20)),
        "replicate": float(channel.get("replicate_consistency", .15)),
        "success": float(channel.get("success", .10)),
        "noise": float(channel.get("noise_penalty", 0)),
        "buffer_classic": float(paired.get("buffer_classic_Q", .25)),
        "target_classic": float(paired.get("target_classic_Q", .25)),
        "delta_peak": float(paired.get("delta_peak", 1)),
    }


def _metric_impacts_q(
    metric_label: str,
    config: dict,
    paired_objective: bool,
    phase: str | None = None,
) -> bool:
    weights = _q_weight_context(config)
    normalized = metric_label.lower().replace("_", " ")
    if "classic q" in normalized:
        if not paired_objective:
            return True
        if phase == "buffer":
            return weights["buffer_classic"] != 0
        if phase == "target":
            return weights["target_classic"] != 0
        return (
            weights["buffer_classic"] != 0
            or weights["target_classic"] != 0
        )
    if "q" in normalized:
        return True
    classic_enabled = True
    if paired_objective:
        if phase == "buffer":
            classic_enabled = weights["buffer_classic"] != 0
        elif phase == "target":
            classic_enabled = weights["target_classic"] != 0
        else:
            classic_enabled = (
                weights["buffer_classic"] != 0
                or weights["target_classic"] != 0
            )
    if "fractional delta" in normalized:
        return False
    if "delta peak" in normalized:
        return paired_objective and weights["delta_peak"] != 0
    if "peak height" in normalized or normalized in {"peak ua", "peak µa"}:
        return (
            paired_objective and weights["delta_peak"] != 0
        ) or (classic_enabled and weights["peak"] != 0)
    if "raw snr" in normalized or "snr raw" in normalized or normalized == "snr":
        return classic_enabled and weights["snr"] != 0
    if "snr score" in normalized:
        return False  # Display-only normalization; Classic Q uses raw SNR.
    if "shape" in normalized:
        return classic_enabled and weights["shape"] != 0
    if "baseline" in normalized:
        return classic_enabled and weights["baseline"] != 0
    if "replicate" in normalized:
        return classic_enabled and weights["replicate"] != 0
    if "success" in normalized:
        return classic_enabled and weights["success"] != 0
    if "noise" in normalized:
        return classic_enabled and weights["noise"] != 0
    return False


def _history_metric_impacts_q(
    metric: str,
    config: dict,
    paired_objective: bool,
) -> bool:
    if metric in PARAMETERS:
        return True
    normalized = str(metric).lower()
    if paired_objective and "classic_q" in normalized:
        weights = _q_weight_context(config)
        if "buffer_classic_q" in normalized:
            return weights["buffer_classic"] != 0
        if "target_classic_q" in normalized:
            return weights["target_classic"] != 0
        return (
            weights["buffer_classic"] != 0
            or weights["target_classic"] != 0
        )
    if "q" in normalized:
        return True
    return _metric_impacts_q(normalized, config, paired_objective)


def _q_relevant_metrics(
    metrics: dict[str, tuple],
    config: dict,
    paired_objective: bool,
    phase: str | None = None,
) -> list[str]:
    return [
        metric
        for metric in metrics
        if _metric_impacts_q(metric, config, paired_objective, phase)
    ]


def _surrogate_files(root: Path, group_id: int | None = None) -> dict[int, Path]:
    result = {}
    surrogate_dir = root / "surrogate"
    pattern = (
        f"group_{group_id:02d}_iter_*_candidate_predictions.csv"
        if group_id is not None
        else "iter_*_candidate_predictions.csv"
    )
    for path in surrogate_dir.glob(pattern):
        try:
            iteration_index = 3 if group_id is not None else 1
            result[int(path.name.split("_")[iteration_index])] = path
        except (ValueError, IndexError):
            continue
    return result


def _pdf_surrogate_file_index(root: Path) -> tuple[dict[int, Path], dict[int, dict[int, Path]]]:
    ungrouped: dict[int, Path] = {}
    grouped: dict[int, dict[int, Path]] = {}
    surrogate_dir = root / "surrogate"
    if not surrogate_dir.is_dir():
        return ungrouped, grouped
    for path in surrogate_dir.iterdir():
        if not path.is_file() or not path.name.endswith("_candidate_predictions.csv"):
            continue
        group_match = re.fullmatch(
            r"group_(\d+)_iter_(\d+)_candidate_predictions\.csv",
            path.name,
        )
        if group_match:
            group_id = int(group_match.group(1))
            iteration = int(group_match.group(2))
            grouped.setdefault(group_id, {})[iteration] = path
            continue
        iteration_match = re.fullmatch(
            r"iter_(\d+)_candidate_predictions\.csv",
            path.name,
        )
        if iteration_match:
            ungrouped[int(iteration_match.group(1))] = path
    return ungrouped, grouped


def _recompute_group_surrogate(
    session: dict,
    frame: pd.DataFrame,
    iteration: int,
) -> pd.DataFrame:
    """Rebuild group-specific GP predictions and acquisition values."""
    corrected = frame.copy()
    observations = [
        observation
        for observation in session["observations"]
        if int(observation.get("iteration", 0)) <= int(iteration)
    ]
    if not observations or corrected.empty:
        return corrected
    selected_next = next(
        (
            observation
            for observation in session["observations"]
            if int(observation.get("iteration", 0)) == int(iteration) + 1
        ),
        None,
    )
    corrected["selected_next"] = False
    if selected_next is not None:
        selected_key = tuple(
            round(float(selected_next["params"][name]), 9)
            for name in PARAMETERS
        )
        candidate_keys = {
            tuple(round(float(row[name]), 9) for name in PARAMETERS)
            for _, row in corrected.iterrows()
        }
        if selected_key not in candidate_keys:
            appended = {
                name: selected_next["params"][name]
                for name in PARAMETERS
            }
            appended["selected_next"] = True
            corrected = pd.concat(
                [corrected, pd.DataFrame([appended])],
                ignore_index=True,
            )
        else:
            selected_mask = [
                tuple(round(float(row[name]), 9) for name in PARAMETERS)
                == selected_key
                for _, row in corrected.iterrows()
            ]
            corrected.loc[selected_mask, "selected_next"] = True
    config = session.get("config") or {}
    group_id = session.get("selected_group_id")
    if group_id is None:
        observed_ids = {
            int(observation.get("group_id", 1))
            for observation in observations
        }
        group_id = next(iter(observed_ids)) if len(observed_ids) == 1 else 1
    group_stub = next(
        (
            group for group in _session_channel_groups(session)
            if int(group["id"]) == int(group_id)
        ),
        {
            "id": int(group_id),
            "name": f"Group {group_id}",
            "channels": [],
        },
    )
    metadata = _channel_group_optimization_metadata(
        session,
        [group_stub],
    )[0]
    exploration = float(metadata.get("exploration") or 0.0)
    parameter_config = config.get("parameters") or {}

    def encode(values) -> np.ndarray:
        encoded = []
        for name in PARAMETERS:
            definition = parameter_config.get(name) or {}
            numeric = np.asarray(values[name], dtype=float)
            lower = float(definition.get("min", np.nanmin(numeric)))
            upper = float(definition.get("max", np.nanmax(numeric)))
            scale = str(
                definition.get(
                    "scale",
                    definition.get("encoding", ""),
                )
            ).lower()
            if scale in ("log", "log10"):
                numeric = np.log10(np.maximum(numeric, 1e-12))
                lower = np.log10(max(lower, 1e-12))
                upper = np.log10(max(upper, 1e-12))
            encoded.append(
                (numeric - lower) / (upper - lower + 1e-12)
            )
        return np.column_stack(encoded)

    best_q = max(float(observation["Q_run"]) for observation in observations)
    means = pd.to_numeric(
        corrected.get("predicted_mean_Q"),
        errors="coerce",
    ).to_numpy(dtype=float)
    stds = pd.to_numeric(
        corrected.get("predicted_std_Q"),
        errors="coerce",
    ).to_numpy(dtype=float)
    if len(observations) >= 2:
        try:
            training_frame = pd.DataFrame([
                observation["params"] for observation in observations
            ])
            x_train = encode(training_frame)
            y_train = np.asarray(
                [float(observation["Q_run"]) for observation in observations],
                dtype=float,
            )
            x_candidates = encode(corrected)
            falloffs = metadata.get("gp_falloff_fractions") or {}
            length_scales = [
                max(1e-9, float(falloffs.get(name, 1.0)))
                for name in PARAMETERS
            ]
            acquisition_config = config.get("acquisition") or {}
            noise = max(
                1e-10,
                float(acquisition_config.get("gp_noise_level", 1e-4) or 1e-4),
            )

            def matern_kernel(left, right):
                scaled = (
                    left[:, None, :] - right[None, :, :]
                ) / np.asarray(length_scales)[None, None, :]
                distance = np.sqrt(np.sum(scaled * scaled, axis=2))
                root5_distance = np.sqrt(5.0) * distance
                return (
                    1.0
                    + root5_distance
                    + root5_distance ** 2 / 3.0
                ) * np.exp(-root5_distance)

            y_mean = float(y_train.mean())
            y_scale = float(y_train.std())
            if y_scale < 1e-12:
                y_scale = 1.0
            normalized_y = (y_train - y_mean) / y_scale
            covariance = matern_kernel(x_train, x_train)
            identity = np.eye(len(x_train), dtype=float)
            jitter = noise
            cholesky = None
            for _attempt in range(8):
                try:
                    cholesky = np.linalg.cholesky(
                        covariance + identity * jitter
                    )
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10.0
            if cholesky is None:
                raise np.linalg.LinAlgError(
                    "GP covariance is not positive definite"
                )
            alpha = np.linalg.solve(
                cholesky.T,
                np.linalg.solve(cholesky, normalized_y),
            )
            cross_covariance = matern_kernel(x_candidates, x_train)
            means = y_mean + y_scale * (cross_covariance @ alpha)
            projected = np.linalg.solve(
                cholesky,
                cross_covariance.T,
            )
            normalized_variance = np.maximum(
                1.0 + jitter - np.sum(projected * projected, axis=0),
                1e-12,
            )
            stds = y_scale * np.sqrt(normalized_variance)
        except Exception:
            # Retain saved predictions, but still correct the acquisition blend.
            pass
    else:
        x_observed = encode(pd.DataFrame([
            observations[0]["params"]
        ]))[0]
        x_candidates = encode(corrected)
        distances = np.sqrt(
            np.sum((x_candidates - x_observed[None, :]) ** 2, axis=1)
        )
        means = np.full(
            len(corrected),
            float(observations[0]["Q_run"]),
        )
        stds = np.maximum(distances, 0.05)
    stds = np.maximum(np.asarray(stds, dtype=float), 1e-12)
    means = np.asarray(means, dtype=float)
    improvement = means - best_q - 0.01
    z_score = improvement / stds
    expected_improvement = (
        improvement * ndtr(z_score)
        + stds * np.exp(-0.5 * z_score ** 2) / np.sqrt(2.0 * np.pi)
    )
    corrected["predicted_mean_Q"] = means
    corrected["predicted_std_Q"] = stds
    corrected["acquisition_value"] = (
        (1.0 - exploration) * (means + 0.25 * expected_improvement)
        + exploration * stds
    )
    corrected["best_observed_Q"] = best_q
    tested = {
        tuple(round(float(observation["params"][name]), 9) for name in PARAMETERS)
        for observation in observations
    }
    corrected["already_tested"] = [
        tuple(round(float(row[name]), 9) for name in PARAMETERS) in tested
        for _, row in corrected.iterrows()
    ]
    return corrected


def _observed_points(session: dict, iteration: int, axes: list[str]):
    """Return history through the next observation chosen from an artifact."""
    rows = []
    for obs in session["observations"]:
        if int(obs.get("iteration", 0)) > iteration + 1:
            continue
        params = obs.get("params") or {}
        if all(params.get(axis) is not None for axis in axes):
            rows.append(obs)
    return sorted(rows, key=lambda obs: int(obs.get("iteration", 0)))


def _observed_iterations(observed: list[dict]) -> np.ndarray:
    return np.asarray([int(obs.get("iteration", 0)) for obs in observed], dtype=float)


def _iteration_norm(iterations: np.ndarray) -> Normalize:
    minimum = float(np.min(iterations))
    maximum = float(np.max(iterations))
    if minimum == maximum:
        maximum = minimum + 1.0
    return Normalize(vmin=minimum, vmax=maximum)


def _plot_observed_path(
    ax,
    x,
    y,
    observed: list[dict],
    *,
    label: str,
    value_norm: Normalize | None,
    marker_size: float = 28,
    show_iteration_path: bool = True,
) -> None:
    iterations = _observed_iterations(observed)
    norm = _iteration_norm(iterations)
    measured = np.asarray([float(obs["Q_run"]) for obs in observed])
    points = np.column_stack((x, y))
    if show_iteration_path and len(points) > 1:
        segments = np.stack((points[:-1], points[1:]), axis=1)
        lines = LineCollection(
            segments,
            cmap=OBSERVED_PATH_CMAP,
            norm=norm,
            linewidth=1.6,
            zorder=4,
        )
        lines.set_array(iterations[1:])
        ax.add_collection(lines)
    marker_colors = (
        {"c": measured, "cmap": "viridis", "norm": value_norm}
        if value_norm is not None
        else {"color": "#35589a"}
    )
    ax.scatter(
        x, y, **marker_colors, s=marker_size, label=label, zorder=3
    )
    if show_iteration_path:
        iteration_map = plt.cm.ScalarMappable(
            norm=norm,
            cmap=OBSERVED_PATH_CMAP,
        )
        ax.figure.colorbar(iteration_map, ax=ax, label="Iteration")


def _surrogate_slice_base_params(session: dict, iteration: int) -> dict:
    observations = [
        observation for observation in session["observations"]
        if int(observation.get("iteration", 0)) <= int(iteration) + 1
    ]
    selected = next(
        (
            observation for observation in observations
            if int(observation.get("iteration", 0)) == int(iteration) + 1
        ),
        None,
    )
    if selected is None and observations:
        selected = max(
            observations,
            key=lambda observation: float(observation.get("Q_run", 0.0)),
        )
    base = dict((selected or {}).get("params") or {})
    initial = dict((session.get("config") or {}).get("initial_parameters") or {})
    for name in PARAMETERS:
        if base.get(name) is None and initial.get(name) is not None:
            base[name] = initial[name]
    return base


def _parameter_grid_values(
    session: dict,
    frame: pd.DataFrame,
    name: str,
    grid_size: int,
) -> np.ndarray | None:
    definition = ((session.get("config") or {}).get("parameters") or {}).get(name) or {}
    values = pd.to_numeric(frame.get(name), errors="coerce").dropna()
    try:
        lower = float(definition.get("min"))
        upper = float(definition.get("max"))
    except (TypeError, ValueError):
        if values.empty:
            return None
        lower = float(values.min())
        upper = float(values.max())
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    if np.isclose(lower, upper):
        if values.nunique() > 1:
            lower = float(values.min())
            upper = float(values.max())
        else:
            return None
    scale = str(
        definition.get(
            "scale",
            definition.get("encoding", ""),
        )
    ).lower()
    if scale in ("log", "log10"):
        lower = max(lower, 1e-12)
        upper = max(upper, lower * (1.0 + 1e-9))
        return np.logspace(np.log10(lower), np.log10(upper), grid_size)
    return np.linspace(lower, upper, grid_size)


def _base_parameter_value(
    session: dict,
    frame: pd.DataFrame,
    base_params: dict,
    name: str,
) -> float:
    try:
        return float(base_params.get(name))
    except (TypeError, ValueError):
        pass
    initial = (session.get("config") or {}).get("initial_parameters") or {}
    try:
        return float(initial.get(name))
    except (TypeError, ValueError):
        pass
    values = pd.to_numeric(frame.get(name), errors="coerce").dropna()
    if not values.empty:
        return float(values.median())
    return 0.0


def _surrogate_regular_2d_grid(
    session: dict,
    frame: pd.DataFrame,
    iteration: int,
    value: str,
    x_name: str,
    y_name: str,
    base_params: dict,
    grid_size: int = 75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if x_name == y_name or value not in SURROGATE_VALUES:
        return None
    size = max(20, min(160, int(grid_size)))
    x_values = _parameter_grid_values(session, frame, x_name, size)
    y_values = _parameter_grid_values(session, frame, y_name, size)
    if x_values is None or y_values is None:
        return None
    mesh_x, mesh_y = np.meshgrid(x_values, y_values)
    grid_frame = pd.DataFrame({
        name: np.full(mesh_x.size, _base_parameter_value(session, frame, base_params, name))
        for name in PARAMETERS
    })
    grid_frame[x_name] = mesh_x.ravel()
    grid_frame[y_name] = mesh_y.ravel()
    for surrogate_value in SURROGATE_VALUES:
        grid_frame[surrogate_value] = np.nan
    try:
        predicted = _recompute_group_surrogate(
            session,
            grid_frame,
            iteration,
        ).iloc[:len(grid_frame)]
        z_values = pd.to_numeric(
            predicted[value],
            errors="coerce",
        ).to_numpy(dtype=float).reshape(mesh_x.shape)
    except Exception:
        return None
    if not np.isfinite(z_values).any():
        return None
    return x_values, y_values, z_values


def _plot_surrogate_2d_control_style(
    ax,
    session: dict,
    display_frame: pd.DataFrame,
    iteration: int,
    value: str,
    x_name: str,
    y_name: str,
    selected_next_frame: pd.DataFrame,
    *,
    dot_size: int = 6,
    log_frequency: bool = False,
    show_iteration_path: bool = True,
):
    """Render 2D surrogate maps with the control-software slice style."""
    base_params = _surrogate_slice_base_params(session, iteration)
    grid = _surrogate_regular_2d_grid(
        session,
        display_frame,
        iteration,
        value,
        x_name,
        y_name,
        base_params,
    )
    if grid is not None:
        x_values, y_values, z_values = grid
        mesh = ax.pcolormesh(
            x_values,
            y_values,
            np.ma.masked_invalid(z_values),
            cmap="viridis",
            shading="auto",
        )
    else:
        valid = display_frame[[x_name, y_name, value]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        if valid.empty:
            ax.text(.5, .5, "No plottable surrogate rows", ha="center", va="center")
            ax.set_axis_off()
            return None
        mesh = ax.scatter(
            valid[x_name],
            valid[y_name],
            c=valid[value],
            cmap="viridis",
            s=max(10, dot_size ** 2),
            alpha=.8,
        )
    observed = _observed_points(session, iteration, [x_name, y_name])
    if observed:
        observed_x = [float(obs["params"][x_name]) for obs in observed]
        observed_y = [float(obs["params"][y_name]) for obs in observed]
        if show_iteration_path and len(observed) > 1:
            ax.plot(
                observed_x,
                observed_y,
                color="#d67b32",
                linewidth=1.4,
                alpha=.95,
                label="observed path",
                zorder=4,
            )
        ax.scatter(
            observed_x,
            observed_y,
            color="#d67b32",
            s=max(18, dot_size ** 2),
            zorder=5,
        )
    if (
        not selected_next_frame.empty
        and x_name in selected_next_frame.columns
        and y_name in selected_next_frame.columns
    ):
        selected_row = selected_next_frame.iloc[0]
        if pd.notna(selected_row[x_name]) and pd.notna(selected_row[y_name]):
            ax.scatter(
                [selected_row[x_name]],
                [selected_row[y_name]],
                color="#ffd166",
                edgecolors="black",
                linewidths=1.0,
                s=(dot_size + 8) ** 2,
                zorder=6,
                label=f"selected iteration {iteration + 1}",
            )
    ax.set_xlabel(x_name, fontsize=9, labelpad=4)
    ax.set_ylabel(y_name, fontsize=9, labelpad=4)
    if log_frequency and x_name == "frequency":
        ax.set_xscale("log")
    if log_frequency and y_name == "frequency":
        ax.set_yscale("log")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=.2)
    return mesh


def _iteration_parameter_text(observation: dict | None) -> str:
    params = (observation or {}).get("params") or {}
    parts = []
    for label, key, unit in (
        ("Frequency", "frequency", "Hz"),
        ("Step size", "step_potential", "V"),
        ("Amplitude", "amplitude", "V"),
    ):
        value = params.get(key)
        try:
            parts.append(f"{label}={float(value):g} {unit}")
        except (TypeError, ValueError):
            parts.append(f"{label}=unknown")
    return " | ".join(parts)


def _plot_surrogate(session: dict, frame: pd.DataFrame, iteration: int, value: str, view: str,
                    x_name: str, y_name: str | None, z_name: str | None,
                    tensor_height: int = 620, dot_size: int = 6,
                    log_frequency: bool = False,
                    show_iteration_path: bool = True):
    selected_observation = next(
        (
            observation for observation in session["observations"]
            if int(observation.get("iteration", 0)) == int(iteration) + 1
        ),
        None,
    )
    artifact_observation = next(
        (
            observation for observation in session["observations"]
            if int(observation.get("iteration", 0)) == int(iteration)
        ),
        None,
    )
    if selected_observation is not None:
        parameter_context = (
            f"selected iteration {iteration + 1}: "
            f"{_iteration_parameter_text(selected_observation)}"
        )
    else:
        parameter_context = (
            f"no completed iteration {iteration + 1}; "
            f"artifact iteration {iteration}: "
            f"{_iteration_parameter_text(artifact_observation)}"
        )
    title = (
        f"{view} | {value} | artifact iteration {iteration}<br>"
        f"{parameter_context}"
    )
    display_frame = frame
    if value == "acquisition_value" and "already_tested" in frame.columns:
        eligible = ~frame["already_tested"].fillna(False).astype(bool)
        if "selected_next" in frame.columns:
            eligible |= frame["selected_next"].fillna(False).astype(bool)
        display_frame = frame.loc[eligible].copy()
    selected_next_frame = (
        display_frame.loc[
            display_frame["selected_next"].fillna(False).astype(bool)
        ]
        if "selected_next" in display_frame.columns
        else pd.DataFrame()
    )
    if view == "3D tensor":
        valid = display_frame[[x_name, y_name, z_name, value]].apply(pd.to_numeric, errors="coerce").dropna()
        prediction_min = float(valid[value].min())
        prediction_max = float(valid[value].max())
        if prediction_min == prediction_max:
            prediction_max = prediction_min + 1.0
        fig = go.Figure(go.Scatter3d(
            x=valid[x_name],
            y=valid[y_name],
            z=valid[z_name],
            mode="markers",
            name="Candidate predictions",
            marker={
                "size": dot_size,
                "color": valid[value],
                "colorscale": "Viridis",
                "cmin": prediction_min,
                "cmax": prediction_max,
                "opacity": 0.55,
                "showscale": True,
                "colorbar": {
                    "title": SURROGATE_VALUE_LABELS.get(value, value),
                    "x": 0.84,
                    "y": 0.48,
                    "len": 0.68,
                },
            },
            customdata=valid[value],
            hovertemplate=(
                f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}<br>"
                f"{z_name}: %{{z:.4g}}<br>{value}: %{{customdata:.4g}}"
                "<extra></extra>"
            ),
        ))
        if not selected_next_frame.empty:
            selected_row = selected_next_frame.iloc[0]
            fig.add_trace(go.Scatter3d(
                x=[selected_row[x_name]],
                y=[selected_row[y_name]],
                z=[selected_row[z_name]],
                mode="markers",
                name=f"Selected iteration {iteration + 1}",
                marker={
                    "size": dot_size + 4,
                    "color": "#ff2da1",
                    "symbol": "diamond",
                    "line": {"color": "white", "width": 2},
                },
                hovertemplate=(
                    f"Selected iteration {iteration + 1}<br>"
                    f"{value}: {float(selected_row[value]):.4g}"
                    "<extra></extra>"
                ),
            ))
        observed = _observed_points(session, iteration, [x_name, y_name, z_name])
        if observed:
            observed_x = [float(obs["params"][x_name]) for obs in observed]
            observed_y = [float(obs["params"][y_name]) for obs in observed]
            observed_z = [float(obs["params"][z_name]) for obs in observed]
            observed_iterations = _observed_iterations(observed)
            observed_values = np.asarray([float(obs["Q_run"]) for obs in observed])
            norm = _iteration_norm(observed_iterations)
            if show_iteration_path:
                for index in range(1, len(observed)):
                    color = OBSERVED_PATH_CMAP(norm(observed_iterations[index]))
                    fig.add_trace(go.Scatter3d(
                        x=observed_x[index - 1:index + 1],
                        y=observed_y[index - 1:index + 1],
                        z=observed_z[index - 1:index + 1],
                        mode="lines",
                        line={"color": f"rgb({color[0] * 255:.0f},{color[1] * 255:.0f},{color[2] * 255:.0f})", "width": 5},
                        hoverinfo="skip",
                        showlegend=False,
                    ))
                fig.add_trace(go.Scatter3d(
                    x=[None, None],
                    y=[None, None],
                    z=[None, None],
                    mode="markers",
                    marker={
                        "color": [norm.vmin, norm.vmax],
                        "colorscale": [
                            [0, OBSERVED_PATH_COLORS[0]],
                            [1, OBSERVED_PATH_COLORS[1]],
                        ],
                        "cmin": norm.vmin,
                        "cmax": norm.vmax,
                        "showscale": True,
                        "colorbar": {
                            "title": "Iteration",
                            "x": 0.98,
                            "y": 0.48,
                            "len": 0.68,
                        },
                    },
                    hoverinfo="skip",
                    showlegend=False,
                ))
            fig.add_trace(go.Scatter3d(
                x=observed_x,
                y=observed_y,
                z=observed_z,
                mode="markers",
                name="Observed path",
                marker={
                    "color": observed_values,
                    "colorscale": "Viridis",
                    "cmin": prediction_min,
                    "cmax": prediction_max,
                    "size": dot_size,
                    "showscale": False,
                },
                customdata=np.column_stack((observed_iterations, observed_values)),
                hovertemplate=(
                    f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}<br>"
                    f"{z_name}: %{{z:.4g}}<br>Iteration: %{{customdata[0]:.0f}}"
                    "<br>Measured Q: %{customdata[1]:.4g}<extra>Observed</extra>"
                ),
            ))
        fig.update_layout(
            title={
                "text": title,
                "x": 0.01,
                "xanchor": "left",
                "y": 0.98,
                "yanchor": "top",
                "font": {"size": 16},
            },
            scene={
                "xaxis": {
                    "title": x_name,
                    "type": "log" if log_frequency and x_name == "frequency" else "linear",
                },
                "yaxis": {
                    "title": y_name,
                    "type": "log" if log_frequency and y_name == "frequency" else "linear",
                },
                "zaxis": {
                    "title": z_name,
                    "type": "log" if log_frequency and z_name == "frequency" else "linear",
                },
                "domain": {"x": [0, 0.76], "y": [0.14, 0.82]},
            },
            legend={
                "orientation": "h",
                "x": 0,
                "xanchor": "left",
                "y": 0.02,
                "yanchor": "bottom",
            },
            height=tensor_height,
            margin={"l": 10, "r": 10, "t": 105, "b": 75},
        )
        return fig
    elif view == "2D map":
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        mesh = _plot_surrogate_2d_control_style(
            ax,
            session,
            display_frame,
            iteration,
            value,
            x_name,
            y_name,
            selected_next_frame,
            dot_size=dot_size,
            log_frequency=log_frequency,
            show_iteration_path=show_iteration_path,
        )
        if mesh is not None:
            colorbar = fig.colorbar(
                mesh,
                ax=ax,
                label=SURROGATE_VALUE_LABELS.get(value, value),
            )
            colorbar.ax.tick_params(labelsize=8)
            colorbar.set_label(
                SURROGATE_VALUE_LABELS.get(value, value),
                fontsize=8,
            )
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="best", fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(6.4, 3.5))
        valid = display_frame[[x_name, value]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.scatter(
            valid[x_name],
            valid[value],
            color="#155e63",
            s=dot_size ** 2,
            alpha=.35,
            label="candidate predictions",
        )
        observed = _observed_points(session, iteration, [x_name])
        if observed:
            observed_x = [obs["params"][x_name] for obs in observed]
            observed_q = [obs["Q_run"] for obs in observed]
            _plot_observed_path(
                ax,
                observed_x,
                observed_q,
                observed,
                label="tested parameter sets",
                value_norm=None,
                marker_size=dot_size ** 2,
                show_iteration_path=show_iteration_path,
            )
        if not selected_next_frame.empty:
            selected_row = selected_next_frame.iloc[0]
            ax.scatter(
                [selected_row[x_name]],
                [selected_row[value]],
                marker="*",
                s=(dot_size + 8) ** 2,
                color="#ff2da1",
                edgecolor="white",
                linewidth=1,
                label=f"selected iteration {iteration + 1}",
                zorder=6,
            )
        ax.set(xlabel=x_name, ylabel=value)
        if log_frequency and x_name == "frequency":
            ax.set_xscale("log")
        ax.legend()
    ax.set_title(
        f"{view} | {value} | artifact iteration {iteration}\n"
        f"{parameter_context}"
    )
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


def _sized_plot_container(container, width_percent: int):
    """Create a centered container whose width is known before plot rendering."""
    if width_percent >= 100:
        return container
    side_width = (100 - width_percent) / 2
    _left, plot_column, _right = container.columns(
        [side_width, width_percent, side_width]
    )
    return plot_column


def _preserve_valid_widget_value(
    key: str,
    options: list,
    default=None,
) -> None:
    """Keep a widget selection across reruns unless it is no longer valid."""
    if not options:
        return
    current = st.session_state.get(key)
    if current not in options:
        st.session_state[key] = (
            default if default in options else options[0]
        )


def _figures_to_gif(
    figures,
    duration_ms: int,
    *,
    plotly_width: int = 1000,
    plotly_height: int = 700,
    total_frames: int | None = None,
    progress_callback=None,
) -> bytes:
    """Render Matplotlib/Plotly figures and encode an animated GIF."""
    from PIL import Image

    frames = []
    for frame_index, figure in enumerate(figures, start=1):
        buffer = BytesIO()
        if isinstance(figure, go.Figure):
            try:
                png = figure.to_image(
                    format="png",
                    width=plotly_width,
                    height=plotly_height,
                    scale=1,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Plotly GIF export requires kaleido==0.2.1. "
                    "Install the updated requirements and restart the app."
                ) from exc
            buffer.write(png)
            buffer.seek(0)
        else:
            figure.savefig(
                buffer,
                format="png",
                dpi=110,
            )
            buffer.seek(0)
            plt.close(figure)
        with Image.open(buffer) as image:
            frames.append(image.convert("RGB").copy())
        if progress_callback is not None:
            progress_callback(frame_index, total_frames or frame_index)
    if not frames:
        raise ValueError("No compatible iteration frames were available.")
    maximum_width = max(frame.width for frame in frames)
    maximum_height = max(frame.height for frame in frames)
    if any(
        frame.size != (maximum_width, maximum_height)
        for frame in frames
    ):
        padded_frames = []
        for frame in frames:
            canvas = Image.new(
                "RGB",
                (maximum_width, maximum_height),
                "white",
            )
            canvas.paste(
                frame,
                (
                    (maximum_width - frame.width) // 2,
                    (maximum_height - frame.height) // 2,
                ),
            )
            padded_frames.append(canvas)
        frames = padded_frames
    frames = [
        frame.quantize(
            colors=128,
            method=Image.Quantize.FASTOCTREE,
        )
        for frame in frames
    ]
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    return output.getvalue()


def render_bo_session_app() -> None:
    """Render the complete BO viewer. Called after Analysis mode is set to BO."""
    st.title("⚡ Bayesian Optimization Session Analysis")
    with st.sidebar:
        st.subheader("BO Session")
        browse_label = "Browse (macOS)" if sys.platform == "darwin" else "Browse (Windows)"
        if st.button(
            browse_label,
            use_container_width=True,
            disabled=not (sys.platform == "darwin" or sys.platform.startswith("win")),
        ):
            try:
                picked = _pick_session_folder()
                if picked:
                    st.session_state.bo_session_folder = picked
            except subprocess.CalledProcessError:
                # Native pickers return a non-zero status when the user cancels.
                pass
            except Exception as exc:
                st.error(f"Folder picker failed: {exc}")
        folder = st.text_input(
            "Session folder",
            key="bo_session_folder",
            placeholder=".../bo_session_<name>",
            help="Choose a folder containing bo_state.json.",
        )

    if not folder:
        st.info("Enter a BO session folder in the sidebar to load its results.")
        return
    try:
        session = load_bo_session(folder)
    except Exception as exc:
        st.error(str(exc))
        return

    full_session = session
    groups = _session_channel_groups(session)
    with st.sidebar:
        st.divider()
        if len(groups) > 1:
            group_ids = [group["id"] for group in groups]
            selected_group_scope = st.selectbox(
                "Channel groups",
                ["all", *group_ids],
                format_func=lambda group_id: next(
                    (
                        f"{group['name']} (channels "
                        f"{', '.join(map(str, group['channels']))})"
                        for group in groups
                        if group["id"] == group_id
                    ),
                    "All channel groups" if group_id == "all" else f"Group {group_id}",
                ),
                key="bo_channel_group_scope",
            )
            if selected_group_scope != "all":
                session = _session_for_channel_group(session, selected_group_scope)
        correction_source = st.radio(
            "Corrected trace processing",
            ["BO session settings", "Analysis app settings"],
            help=(
                "Both options process the raw SWV CSVs with the analysis app's "
                "native algorithm. Choose whether to use the settings saved by "
                "the BO run or editable SWV-analysis settings."
            ),
        )
        if correction_source == "Analysis app settings":
            trace_analysis = _native_analysis_settings_sidebar()
            correction_label = "analysis app settings"
        else:
            trace_analysis = _bo_analysis_settings(session["config"])
            correction_label = "BO session settings"
            st.caption(
                "Using the analysis parameters recorded in bo_config_snapshot.json."
            )
        st.divider()
        plot_width_percent = st.slider(
            "Plot width",
            min_value=40,
            max_value=100,
            value=80,
            step=5,
            format="%d%%",
            help="Width of plots relative to the space available in the current tab or column.",
            key="bo_plot_width_percent",
        )
        plot_3d_height = st.slider(
            "3D plot height",
            min_value=400,
            max_value=1500,
            value=620,
            step=20,
            format="%d px",
            help="Canvas height used for 3D tensor plots.",
            key="bo_plot_3d_height",
        )
        plot_dot_size = st.slider(
            "1D/3D dot size",
            min_value=2,
            max_value=20,
            value=6,
            step=1,
            help="Marker diameter for 1D slice and 3D tensor plots.",
            key="bo_plot_dot_size",
        )

    observations = session["observations"]
    history = _observation_table(session)
    if not observations:
        st.warning("This channel group has no completed observations.")
        return
    selected_group_id = session.get("selected_group_id")
    if selected_group_id is not None:
        selected_group = next(
            group for group in groups if group["id"] == selected_group_id
        )
        st.caption(
            f"Showing {selected_group['name']} — channels "
            f"{', '.join(map(str, selected_group['channels']))}"
        )
    paired_objective = any(
        str(obs.get("objective", "")).lower() == "paired_response"
        for obs in observations
    )
    q_values = [float(obs.get("Q_run", np.nan)) for obs in observations]
    best_index = int(np.nanargmax(q_values))
    best = observations[best_index]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed observations", len(observations))
    c2.metric("Best Q", f"{best.get('Q_run', 0):.4g}")
    best_group_name = best.get("group_name") or f"Group {best.get('group_id', 1)}"
    best_label = (
        f"{best_group_name} iter {best.get('iteration')}"
        if len(groups) > 1 and selected_group_id is None
        else best.get("iteration")
    )
    c3.metric("Best observation", best_label)
    c4.metric("Candidates", session["state"].get("candidate_count", "—"))

    all_groups_scope = len(groups) > 1 and selected_group_id is None
    observation_group_ids = sorted({
        int(obs.get("group_id", 1)) for obs in observations
    })
    observation_selector_columns = st.columns(2)
    with observation_selector_columns[0]:
        selected_observation_group_scope = st.selectbox(
            "Observation group",
            [*observation_group_ids, "all"],
            format_func=lambda group_id: next(
                (
                    f"{group['name']} (channels "
                    f"{', '.join(map(str, group['channels']))})"
                    for group in groups
                    if group["id"] == group_id
                ),
                "All groups" if group_id == "all" else f"Group {group_id}",
            ),
            key="bo_observation_group_scope",
        )
    group_scoped_observations = [
        obs for obs in observations
        if (
            selected_observation_group_scope == "all"
            or int(obs.get("group_id", 1)) == int(selected_observation_group_scope)
        )
    ]
    scoped_iterations = sorted({
        int(obs["iteration"]) for obs in group_scoped_observations
    })
    with observation_selector_columns[1]:
        selected_iteration_scope = st.selectbox(
            "Observation iteration",
            [*scoped_iterations, "all"],
            index=max(0, len(scoped_iterations) - 1),
            format_func=lambda iteration: (
                "All iterations" if iteration == "all" else f"Iteration {iteration}"
            ),
            key=f"bo_observation_iteration_{selected_observation_group_scope}",
        )
    selected_observations = [
        obs for obs in group_scoped_observations
        if (
            selected_iteration_scope == "all"
            or int(obs["iteration"]) == int(selected_iteration_scope)
        )
    ]
    observation = selected_observations[-1]
    selected_observation_group = int(observation.get("group_id", 1))
    selected_iteration = int(observation["iteration"])
    observation_is_single = len(selected_observations) == 1
    iteration_options = scoped_iterations
    iteration_state_key = "bo_requested_observation_iteration"
    overview, metadata_tab, traces, real_data, surrogate, gifs, pdf_export = st.tabs(
        [
            "History & scores",
            "Optimization metadata",
            "SWV traces",
            "Real data landscapes",
            "Surrogate",
            "GIFs",
            "PDF Export",
        ]
    )

    with metadata_tab:
        st.subheader("Channel-group optimization metadata")
        optimization_metadata = _channel_group_optimization_metadata(
            full_session,
            groups,
        )
        metadata_summary = pd.DataFrame([
            {
                "Group": item["name"],
                "Channels": ", ".join(map(str, item["channels"])),
                "Explore weight": item["exploration"],
                "Exploit weight": item["exploitation"],
                "Initial-point mode": item["initial_point_mode"],
                "Initial random points": item["n_initial_points"],
                "Candidate pool": item["candidate_pool_size"],
                "Local candidate pool": item["local_candidate_pool_size"],
                "Use GP": item["use_gp"],
                "GP optimizer restarts": item["gp_optimizer_restarts"],
            }
            for item in optimization_metadata
        ])
        st.dataframe(
            metadata_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Explore weight": st.column_config.NumberColumn(format="%.3f"),
                "Exploit weight": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        parameter_config = (full_session.get("config") or {}).get("parameters") or {}
        falloff_parameters = [
            name for name in PARAMETERS
            if any(
                name in item["gp_falloff_fractions"]
                for item in optimization_metadata
            )
        ]
        st.markdown("#### GP falloff fractions")
        if falloff_parameters:
            falloff_rows = []
            for item in optimization_metadata:
                row = {
                    "Group": item["name"],
                    "Channels": ", ".join(map(str, item["channels"])),
                }
                row.update({
                    (parameter_config.get(name) or {}).get("label") or name:
                    item["gp_falloff_fractions"].get(name)
                    for name in falloff_parameters
                })
                falloff_rows.append(row)
            st.table(pd.DataFrame(falloff_rows).set_index("Group"))
            st.caption(
                "Each value is the saved GP falloff fraction for that parameter "
                "and channel group."
            )
        else:
            st.info("No GP falloff fractions were saved for this session.")

        st.markdown("#### Starting points and parameter details")
        for item in optimization_metadata:
            with st.expander(
                f"{item['name']} parameter metadata "
                f"(channels {', '.join(map(str, item['channels']))})"
            ):
                parameter_names = [
                    name for name in PARAMETERS
                    if (
                        name in item["initial_parameters"]
                        or name in item["gp_falloff_fractions"]
                        or name in item["gp_length_scales"]
                    )
                ]
                parameter_rows = []
                for name in parameter_names:
                    definition = parameter_config.get(name) or {}
                    parameter_rows.append({
                        "Parameter": definition.get("label") or name,
                        "Mode": definition.get("mode"),
                        "Start": item["initial_parameters"].get(name),
                        "Unit": definition.get("unit"),
                        "Minimum": definition.get("min"),
                        "Maximum": definition.get("max"),
                        "GP falloff fraction": item[
                            "gp_falloff_fractions"
                        ].get(name),
                        "GP length scale": item["gp_length_scales"].get(name),
                    })
                if parameter_rows:
                    st.dataframe(
                        pd.DataFrame(parameter_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption(
                        "No parameter-level starting-point or GP metadata was saved."
                    )

        st.divider()
        st.markdown("#### How explore vs exploit affects candidate selection")
        st.write(
            "For every candidate x, the optimizer predicts mean mu(x) and "
            "standard deviation sigma(x). With exploration weight alpha, "
            "it maximizes:"
        )
        st.code(
            "A(x) = (1 - alpha) * [mu(x) + 0.25 * EI(x)]"
            " + alpha * sigma(x)\n"
            "EI(x) = I(x) * Phi(z) + sigma(x) * phi(z)\n"
            "I(x) = mu(x) - Q_best - 0.01\n"
            "z = I(x) / sigma(x)",
            language=None,
        )
        st.write(
            "alpha is the explore weight and (1 - alpha) is the exploit "
            "weight. alpha = 0 favors high predicted Q plus expected "
            "improvement. alpha = 1 favors high GP uncertainty. Phi and phi "
            "are the standard-normal CDF and PDF."
        )
        st.markdown("#### How GP falloff fractions affect the surrogate")
        st.write(
            "Each parameter is mapped to a normalized coordinate from 0 to 1. "
            "Its falloff fraction ell_j is the fixed Matérn-5/2 length scale:"
        )
        st.code(
            "r(x, x') = sqrt(sum_j(((x_j - x'_j) / ell_j)^2))\n"
            "k(x, x') = (1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)",
            language=None,
        )
        st.write(
            "A smaller ell_j makes correlation fall off quickly when that "
            "parameter changes, allowing a more rapidly varying surrogate. "
            "A larger ell_j makes the GP smoother and extends the influence "
            "of observations along that parameter. Because coordinates are "
            "normalized, ell_j = 0.2 is roughly one fifth of the configured "
            "parameter range."
        )

    with overview:
        trend_history = history
        if (
            selected_observation_group_scope != "all"
            and "group_id" in history.columns
        ):
            history_group_ids = pd.to_numeric(
                history["group_id"],
                errors="coerce",
            )
            trend_history = history.loc[
                history_group_ids == int(selected_observation_group_scope)
            ].reset_index(drop=True)
        trend_scope_key = str(selected_observation_group_scope)
        available_trend_iterations = pd.to_numeric(
            trend_history.get("iteration", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna().astype(int)
        if available_trend_iterations.empty:
            iteration_start = iteration_end = 0
            plot_observations = []
        else:
            minimum_iteration = int(available_trend_iterations.min())
            maximum_iteration = int(available_trend_iterations.max())
            range_columns = st.columns(2)
            range_key = (
                f"{session['state'].get('session_id', 'session')}_"
                f"{trend_scope_key}"
            )
            iteration_start = int(range_columns[0].number_input(
                "Plot iteration start",
                min_value=minimum_iteration,
                max_value=maximum_iteration,
                value=minimum_iteration,
                step=1,
                key=f"bo_trend_iteration_start_{range_key}",
            ))
            iteration_end = int(range_columns[1].number_input(
                "Plot iteration end",
                min_value=minimum_iteration,
                max_value=maximum_iteration,
                value=maximum_iteration,
                step=1,
                key=f"bo_trend_iteration_end_{range_key}",
            ))
            if iteration_start > iteration_end:
                st.warning(
                    "Iteration start is greater than iteration end; "
                    "the bounds are being applied in ascending order."
                )
                iteration_start, iteration_end = iteration_end, iteration_start
            trend_iterations = pd.to_numeric(
                trend_history["iteration"],
                errors="coerce",
            )
            trend_history = trend_history.loc[
                trend_iterations.between(iteration_start, iteration_end)
            ].reset_index(drop=True)
            plot_observations = [
                observation
                for observation in group_scoped_observations
                if iteration_start
                <= int(observation.get("iteration", 0))
                <= iteration_end
            ]
        channel_metrics = _channel_metric_columns(trend_history)
        if selected_observation_group_scope != "all":
            selected_trend_group = next(
                (
                    group for group in groups
                    if group["id"] == int(selected_observation_group_scope)
                ),
                None,
            )
            configured_channels = {
                str(channel)
                for channel in (
                    selected_trend_group.get("channels", [])
                    if selected_trend_group is not None
                    else []
                )
            }
            if configured_channels:
                channel_metrics = {
                    metric_name: {
                        channel: column
                        for channel, column in columns.items()
                        if channel in configured_channels
                    }
                    for metric_name, columns in channel_metrics.items()
                }
                channel_metrics = {
                    metric_name: columns
                    for metric_name, columns in channel_metrics.items()
                    if columns
                }
        channel_column_names = {
            column for column in trend_history.columns
            if (
                re.fullmatch(r"Q_ch\d+", str(column), re.IGNORECASE)
                or re.fullmatch(r"ch\d+_.+", str(column), re.IGNORECASE)
            )
        }
        global_metrics = [
            metric for metric in _numeric_columns(trend_history)
            if metric not in channel_column_names
        ]
        metric_options = (
            [f"global::{metric}" for metric in global_metrics]
            + [f"channel::{metric}" for metric in channel_metrics]
        )
        if metric_options:
            default_metric = (
                "global::Q_run"
                if "Q_run" in global_metrics
                else metric_options[0]
            )
            metric_choice = st.selectbox(
                "Trend metric",
                metric_options,
                index=metric_options.index(default_metric),
                format_func=lambda choice: _metric_label(choice.split("::", 1)[1]),
                key=f"bo_trend_metric_{trend_scope_key}",
            )
        else:
            st.info("No trend metrics change within the current group scope.")
            metric_choice = "global::__no_changing_metric__"
        metric_kind, metric = metric_choice.split("::", 1)
        plot_metric_kind = metric_kind
        plot_metric = metric
        q_run_channel_view = None
        if (
            metric_kind == "global"
            and metric == "Q_run"
            and "Q_channel" in channel_metrics
        ):
            q_run_channel_view = st.radio(
                "Q display",
                [
                    "Run-level Q",
                    "Average channel Q",
                    "Overlay channel Q",
                    "Separate channel Q plots",
                ],
                horizontal=True,
                key=f"bo_q_run_display_{trend_scope_key}",
            )
            if q_run_channel_view != "Run-level Q":
                plot_metric_kind = "channel"
                plot_metric = "Q_channel"
        group_layout = "Plot groups overlaid"
        group_color_values = None
        group_color_label = None
        has_multiple_trend_groups = (
            metric in trend_history.columns
            and
            "group_id" in trend_history.columns
            and trend_history["group_id"].nunique(dropna=True) > 1
        )
        if has_multiple_trend_groups:
            group_layout = st.radio(
                "Group display",
                [
                    "Plot groups overlaid",
                    "Plot groups separately",
                    "Average groups together",
                ],
                horizontal=True,
                key=f"bo_trend_group_layout_{trend_scope_key}_{metric}",
            )
            if group_layout != "Average groups together":
                trend_group_metadata = _channel_group_optimization_metadata(
                    full_session,
                    groups,
                )
                color_options: dict[str, tuple[dict[int, float] | None, str | None]] = {
                    "Default categorical colors": (None, None),
                    "Exploration weight": (
                        {
                            item["id"]: float(item["exploration"])
                            for item in trend_group_metadata
                            if item["exploration"] is not None
                        },
                        "Exploration",
                    ),
                    "Exploitation weight": (
                        {
                            item["id"]: float(item["exploitation"])
                            for item in trend_group_metadata
                            if item["exploitation"] is not None
                        },
                        "Exploitation",
                    ),
                }
                parameter_config = (
                    (full_session.get("config") or {}).get("parameters") or {}
                )
                for parameter in PARAMETERS:
                    values = {
                        item["id"]: float(
                            item["gp_falloff_fractions"][parameter]
                        )
                        for item in trend_group_metadata
                        if parameter in item["gp_falloff_fractions"]
                    }
                    if values:
                        parameter_label = (
                            (parameter_config.get(parameter) or {}).get("label")
                            or parameter
                        )
                        option_label = f"GP falloff — {parameter_label}"
                        color_options[option_label] = (
                            values,
                            f"{parameter_label} GP falloff",
                        )
                group_color_choice = st.selectbox(
                    "Group colormap metadata metric",
                    list(color_options),
                    key=(
                        f"bo_trend_group_colormap_{trend_scope_key}_{metric}"
                    ),
                )
                group_color_values, group_color_label = color_options[
                    group_color_choice
                ]
        chart_key_suffix = metric
        if plot_metric_kind == "channel":
            available_metric_channels = sorted(
                channel_metrics[plot_metric],
                key=_channel_sort_key,
            )
            trend_channels = st.multiselect(
                "Trend channels",
                available_metric_channels,
                default=available_metric_channels,
                format_func=lambda channel: f"Ch {channel}",
                key=f"bo_trend_channels_{trend_scope_key}_{plot_metric}",
            )
            if q_run_channel_view is None:
                channel_layout = st.radio(
                    "Channel display",
                    ["Overlay selected channels", "Separate plots", "Average selected channels"],
                    horizontal=True,
                    key=f"bo_trend_layout_{trend_scope_key}_{plot_metric}",
                )
            else:
                channel_layout = {
                    "Average channel Q": "Average selected channels",
                    "Overlay channel Q": "Overlay selected channels",
                    "Separate channel Q plots": "Separate plots",
                }[q_run_channel_view]
            if trend_channels:
                trend_figure = _plot_channel_trend(
                    trend_history,
                    plot_metric,
                    channel_metrics[plot_metric],
                    trend_channels,
                    channel_layout,
                    group_layout,
                    group_color_values,
                    group_color_label,
                )
            else:
                trend_figure = go.Figure()
                trend_figure.add_annotation(
                    text="Select at least one trend channel.",
                    x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
                )
                trend_figure.update_layout(height=340)
            chart_key_suffix = (
                f"{metric}_{plot_metric}_{channel_layout}_"
                f"{group_layout}_{group_color_label or 'categorical'}_"
                f"{'_'.join(trend_channels) or 'none'}"
            )
        else:
            trend_figure = _plot_trend(
                trend_history,
                metric,
                group_layout,
                group_color_values,
                group_color_label,
            )
            chart_key_suffix = (
                f"{metric}_{group_layout}_{group_color_label or 'categorical'}"
            )
        chart_key_suffix = (
            f"{trend_scope_key}_{iteration_start}_{iteration_end}_"
            f"{chart_key_suffix}"
        )
        trend_events = [_sized_plot_container(st, plot_width_percent).plotly_chart(
            trend_figure,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key=f"bo_trend_{chart_key_suffix}",
        )]
        trend_q_kind = _metric_q_kind(metric, paired_objective)
        if trend_q_kind:
            _render_q_equation(session["config"], trend_q_kind)
        clicked_iteration = next(
            (
                iteration
                for iteration in (_clicked_iteration(event) for event in trend_events)
                if iteration is not None
            ),
            None,
        )
        click_state_key = (
            f"bo_last_trend_click_{session['state'].get('session_id', 'session')}_"
            f"{selected_group_id}_{chart_key_suffix}"
        )
        last_clicked_iteration = st.session_state.get(click_state_key)
        if clicked_iteration is not None and clicked_iteration != last_clicked_iteration:
            st.session_state[click_state_key] = clicked_iteration
        is_new_plot_click = (
            clicked_iteration is not None
            and clicked_iteration != last_clicked_iteration
        )
        if (
            not all_groups_scope
            and
            is_new_plot_click
            and clicked_iteration in iteration_options
            and clicked_iteration != selected_iteration
        ):
            st.session_state[iteration_state_key] = (
                f"g{selected_observation_group}:i{clicked_iteration}"
            )
            st.rerun()

        if any(
            str(obs.get("objective", "")).lower() == "paired_response"
            for obs in plot_observations
        ):
            st.divider()
            st.subheader("Buffer vs target trends")
            paired_metric = st.selectbox(
                "Buffer/target metric",
                _q_relevant_metrics(
                    PAIRED_TREND_METRICS,
                    session["config"],
                    paired_objective=True,
                ),
                key="bo_paired_trend_metric",
            )
            paired_series = _paired_trend_values(
                plot_observations,
                paired_metric,
            )
            paired_channels = sorted(paired_series, key=_channel_sort_key)
            selected_paired_channels = st.multiselect(
                "Buffer/target channels",
                paired_channels,
                default=paired_channels,
                format_func=lambda channel: f"Ch {channel}",
                key=f"bo_paired_channels_{paired_metric}",
            )
            paired_layout = st.radio(
                "Buffer/target display",
                ["Overlay selected channels", "Separate plots", "Average selected channels"],
                horizontal=True,
                key=f"bo_paired_layout_{paired_metric}",
            )
            if selected_paired_channels:
                paired_figure = _plot_paired_phase_trend(
                    paired_series,
                    paired_metric,
                    selected_paired_channels,
                    paired_layout,
                )
            else:
                paired_figure = go.Figure()
                paired_figure.add_annotation(
                    text="Select at least one buffer/target channel.",
                    x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
                )
                paired_figure.update_layout(height=360)
            paired_chart_suffix = (
                f"{iteration_start}_{iteration_end}_{paired_metric}_{paired_layout}_"
                f"{'_'.join(selected_paired_channels) or 'none'}"
            )
            paired_events = []
            if (
                paired_layout == "Average selected channels"
                and selected_paired_channels
            ):
                average_column, overlay_column = st.columns(2)
                paired_events.append(_sized_plot_container(
                    average_column, plot_width_percent
                ).plotly_chart(
                    paired_figure,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points",
                    key=f"bo_paired_trend_{paired_chart_suffix}_average",
                ))
                paired_events.append(_sized_plot_container(
                    overlay_column, plot_width_percent
                ).plotly_chart(
                    _plot_paired_phase_trend(
                        paired_series,
                        paired_metric,
                        selected_paired_channels,
                        "Overlay selected channels",
                    ),
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points",
                    key=f"bo_paired_trend_{paired_chart_suffix}_overlay",
                ))
                paired_events.append(_sized_plot_container(
                    st, plot_width_percent
                ).plotly_chart(
                    _plot_paired_phase_trend(
                        paired_series,
                        paired_metric,
                        selected_paired_channels,
                        "Separate plots",
                    ),
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points",
                    key=f"bo_paired_trend_{paired_chart_suffix}_separate",
                ))
            else:
                paired_events.append(_sized_plot_container(
                    st, plot_width_percent
                ).plotly_chart(
                    paired_figure,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points",
                    key=f"bo_paired_trend_{paired_chart_suffix}",
                ))
            if paired_metric == "Classic Q":
                _render_q_equation(session["config"], "classic")
            paired_clicked_iteration = next(
                (
                    iteration
                    for iteration in (
                        _clicked_iteration(event) for event in paired_events
                    )
                    if iteration is not None
                ),
                None,
            )
            paired_click_key = (
                f"bo_last_paired_click_{session['state'].get('session_id', 'session')}_"
                f"{selected_group_id}_{paired_chart_suffix}"
            )
            last_paired_click = st.session_state.get(paired_click_key)
            if (
                paired_clicked_iteration is not None
                and paired_clicked_iteration != last_paired_click
            ):
                st.session_state[paired_click_key] = paired_clicked_iteration
                if (
                    not all_groups_scope
                    and
                    paired_clicked_iteration in iteration_options
                    and paired_clicked_iteration != selected_iteration
                ):
                    st.session_state[iteration_state_key] = (
                        f"g{selected_observation_group}:i{paired_clicked_iteration}"
                    )
                    st.rerun()

            st.divider()
            st.subheader("Chronological plot")
            chronological_metric = st.selectbox(
                "Chronological metric",
                _q_relevant_metrics(
                    PAIRED_TREND_METRICS,
                    session["config"],
                    paired_objective=True,
                ),
                key="bo_chronological_metric",
            )
            chronological_channels = _real_data_channels(
                plot_observations
            )
            selected_chronological_channels = st.multiselect(
                "Chronological channels",
                chronological_channels,
                default=chronological_channels,
                format_func=lambda channel: f"Ch {channel}",
                key=f"bo_chronological_channels_{chronological_metric}",
            )
            chronological_mode = st.radio(
                "Chronological display",
                ["Overlay selected channels", "Separate plots", "Average selected channels"],
                horizontal=True,
                key=f"bo_chronological_mode_{chronological_metric}",
            )
            chronological_points, phase_transitions = _chronological_points(
                plot_observations,
                session["config"],
                chronological_metric,
                selected_chronological_channels,
                average_channels=chronological_mode == "Average selected channels",
            )
            chronological_raw_points = None
            if (
                chronological_mode == "Average selected channels"
                and selected_chronological_channels
            ):
                chronological_raw_points, _ = _chronological_points(
                    plot_observations,
                    session["config"],
                    chronological_metric,
                    selected_chronological_channels,
                    average_channels=False,
                )
            if chronological_points.empty:
                st.info("No chronological values are available for this selection.")
            else:
                chronological_suffix = (
                    f"{iteration_start}_{iteration_end}_"
                    f"{chronological_metric}_{chronological_mode}_"
                    f"{'_'.join(selected_chronological_channels)}"
                )
                chronological_events = []
                if chronological_raw_points is not None:
                    average_column, overlay_column = st.columns(2)
                    chronological_events.append(_sized_plot_container(
                        average_column, plot_width_percent
                    ).plotly_chart(
                        _plot_chronological(
                            chronological_points,
                            phase_transitions,
                            chronological_metric,
                            "Average selected channels",
                        ),
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="points",
                        key=f"bo_chronological_{chronological_suffix}_average",
                    ))
                    chronological_events.append(_sized_plot_container(
                        overlay_column, plot_width_percent
                    ).plotly_chart(
                        _plot_chronological(
                            chronological_raw_points,
                            phase_transitions,
                            chronological_metric,
                            "Overlay selected channels",
                        ),
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="points",
                        key=f"bo_chronological_{chronological_suffix}_overlay",
                    ))
                    chronological_events.append(_sized_plot_container(
                        st, plot_width_percent
                    ).plotly_chart(
                        _plot_chronological(
                            chronological_raw_points,
                            phase_transitions,
                            chronological_metric,
                            "Separate plots",
                        ),
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="points",
                        key=f"bo_chronological_{chronological_suffix}_separate",
                    ))
                else:
                    chronological_events.append(_sized_plot_container(
                        st, plot_width_percent
                    ).plotly_chart(
                        _plot_chronological(
                            chronological_points,
                            phase_transitions,
                            chronological_metric,
                            chronological_mode,
                        ),
                        use_container_width=True,
                        on_select="rerun",
                        selection_mode="points",
                        key=f"bo_chronological_{chronological_suffix}",
                    ))
                if chronological_metric == "Classic Q":
                    _render_q_equation(session["config"], "classic")
                chronological_iteration = next(
                    (
                        iteration
                        for iteration in (
                            _clicked_iteration(event)
                            for event in chronological_events
                        )
                        if iteration is not None
                    ),
                    None,
                )
                chronological_click_key = (
                    f"bo_last_chronological_click_"
                    f"{session['state'].get('session_id', 'session')}_"
                    f"{selected_group_id}_{chronological_suffix}"
                )
                last_chronological_click = st.session_state.get(chronological_click_key)
                if (
                    chronological_iteration is not None
                    and chronological_iteration != last_chronological_click
                ):
                    st.session_state[chronological_click_key] = chronological_iteration
                    if (
                        not all_groups_scope
                        and
                        chronological_iteration in iteration_options
                        and chronological_iteration != selected_iteration
                    ):
                        st.session_state[iteration_state_key] = (
                            f"g{selected_observation_group}:i{chronological_iteration}"
                        )
                        st.rerun()

        st.subheader("History")
        st.dataframe(history, use_container_width=True, hide_index=True)
        if not observation_is_single:
            st.info(
                "Select one group and one iteration to inspect per-channel "
                "scores for a single observation."
            )
        else:
            st.subheader(
                f"{observation.get('group_name', f'Group {selected_observation_group}')} "
                f"iteration {selected_iteration} per-channel scores"
            )
            channel_frame = _channel_table(observation)
            if channel_frame.empty:
                st.info("No per-channel scores were recorded for this iteration.")
            else:
                st.dataframe(channel_frame, use_container_width=True, hide_index=True)

    with traces:
        trace_observations = selected_observations
        trace_all_mode = not observation_is_single
        selected_group_label = (
            "All groups"
            if selected_observation_group_scope == "all"
            else next(
                (
                    group["name"] for group in groups
                    if group["id"] == selected_observation_group_scope
                ),
                f"Group {selected_observation_group_scope}",
            )
        )
        selected_iteration_label = (
            "all iterations"
            if selected_iteration_scope == "all"
            else f"iteration {selected_iteration_scope}"
        )
        trace_selection_label = (
            f"{selected_group_label} — {selected_iteration_label}"
        )
        qualify_trace_channels = (
            selected_observation_group_scope == "all"
            and len({
                int(item.get("group_id", 1))
                for item in trace_observations
            }) > 1
        )
        with st.spinner("Locating SWV traces..."):
            trace_entries = [
                (
                    trace_observation,
                    _group_qualified_trace(
                        trace_observation,
                        trace,
                        qualify_trace_channels,
                    ),
                )
                for trace_observation in trace_observations
                for trace in _trace_paths(full_session, trace_observation)
            ]
        available_traces = [trace for _observation, trace in trace_entries]
        available_channels = sorted(
            {_trace_channel_key(item) for item in available_traces},
            key=_channel_sort_key,
        )
        if not available_traces:
            st.info(
                "No locally accessible raw SWV files were found for this selection. "
                "The recorded CSVs must remain inside or beside the experiment folder."
            )
        else:
            trace_channels_key = (
                f"bo_trace_channels_{selected_observation_group_scope}_"
                f"{selected_iteration_scope}"
            )
            current_trace_channels = st.session_state.get(trace_channels_key)
            if current_trace_channels and any(
                channel not in available_channels
                for channel in current_trace_channels
            ):
                st.session_state[trace_channels_key] = available_channels
            selected_channels = st.multiselect(
                "Channels to display",
                available_channels,
                default=available_channels,
                format_func=_trace_channel_label,
                key=trace_channels_key,
            )
            trace_layout = st.radio(
                "Plot layout",
                ["Overlay selected channels", "Separate plot per channel"],
                horizontal=True,
                key=(
                    f"bo_trace_layout_{selected_observation_group_scope}_"
                    f"{selected_iteration_scope}"
                ),
            )
            selected_trace_type = st.radio(
                "Trace type",
                ["Raw", "Corrected", "Normalized corrected", "Normalized raw"],
                horizontal=True,
                key=(
                    f"bo_trace_type_{selected_observation_group_scope}_"
                    f"{selected_iteration_scope}"
                ),
            )
            corrected = selected_trace_type != "Raw"
            normalize_to_peak = selected_trace_type in {
                "Normalized corrected",
                "Normalized raw",
            }
            corrected_trace_key = (
                "corrected_current"
                if selected_trace_type == "Normalized raw"
                else "smoothed_corrected_current"
            )
            trace_gif_duration = st.slider(
                "SWV GIF frame duration (ms)",
                min_value=100,
                max_value=1500,
                value=350,
                step=50,
                key=(
                    f"bo_trace_gif_duration_{selected_observation_group_scope}_"
                    f"{selected_iteration_scope}"
                ),
                disabled=not trace_all_mode,
            )

            channel_groups = (
                [[channel] for channel in selected_channels]
                if trace_layout == "Separate plot per channel"
                else [selected_channels]
            )
            if not selected_channels:
                st.info("Select at least one channel to display traces.")
            for channel_group in channel_groups:
                if trace_layout == "Separate plot per channel":
                    channel = channel_group[0]
                    st.markdown(
                        f"#### {_trace_channel_label(channel)}"
                    )
                with st.spinner(
                    "Processing corrected traces..."
                    if corrected
                    else "Loading raw traces..."
                ):
                    if trace_all_mode:
                        figure, errors = _plot_iteration_trace_overlay(
                            trace_entries,
                            corrected,
                            channel_group,
                            trace_analysis,
                            correction_label,
                            trace_selection_label,
                            normalize_to_peak,
                            corrected_trace_key,
                        )
                    else:
                        figure, errors = _plot_traces(
                            session,
                            observation,
                            corrected,
                            channel_group,
                            trace_analysis,
                            correction_label,
                            trace_layout == "Overlay selected channels",
                            available_traces,
                            normalize_to_peak,
                            corrected_trace_key,
                        )
                _sized_plot_container(st, plot_width_percent).pyplot(
                    figure,
                    clear_figure=True,
                    use_container_width=True,
                )
                for error in errors:
                    st.warning(error)

            if trace_all_mode and selected_channels:
                trace_gif_key = (
                    f"bo_trace_gif_{selected_observation_group_scope}_"
                    f"{selected_iteration_scope}_{trace_layout}_{selected_trace_type}_"
                    f"{'_'.join(selected_channels)}"
                )
                if st.button(
                    "Generate SWV GIF",
                    key=f"{trace_gif_key}_button",
                ):
                    with st.spinner("Rendering SWV trace frames..."):
                        try:
                            trace_progress = st.progress(
                                0.0,
                                text="Preparing SWV frames...",
                            )
                            frame_observations = sorted(
                                trace_observations,
                                key=lambda item: (
                                    int(item.get("group_id", 1)),
                                    int(item.get("iteration", 0)),
                                ),
                            )
                            trace_progress.progress(
                                0.02,
                                text="Calculating shared SWV y-axis range...",
                            )
                            selected_raw_channels = sorted(
                                {
                                    trace["channel"]
                                    for _trace_observation, trace in trace_entries
                                    if _trace_channel_key(trace) in selected_channels
                                },
                                key=_channel_sort_key,
                            )
                            gif_y_limits = _swv_global_y_limits(
                                frame_observations,
                                full_session,
                                selected_raw_channels,
                                corrected,
                                trace_analysis,
                                normalize_to_peak,
                                corrected_trace_key,
                            )

                            def swv_frames():
                                for frame_observation in frame_observations:
                                    frame_iteration = int(
                                        frame_observation.get("iteration", 0)
                                    )
                                    frame_group = str(
                                        frame_observation.get("group_name")
                                        or f"Group {frame_observation.get('group_id', 1)}"
                                    )
                                    frame_traces = _trace_paths(
                                        full_session,
                                        frame_observation,
                                    )
                                    frame_traces = [
                                        _group_qualified_trace(
                                            frame_observation,
                                            trace,
                                            qualify_trace_channels,
                                        )
                                        for trace in frame_traces
                                    ]
                                    if not frame_traces:
                                        continue
                                    if trace_layout == "Separate plot per channel":
                                        figures = []
                                        for single_channel in selected_channels:
                                            fig, errors = _plot_traces(
                                                full_session,
                                                frame_observation,
                                                corrected,
                                                [single_channel],
                                                trace_analysis,
                                                correction_label,
                                                overlaid=False,
                                                trace_items=frame_traces,
                                                normalize_to_peak=normalize_to_peak,
                                                corrected_trace_key=corrected_trace_key,
                                            )
                                            if errors:
                                                fig.text(
                                                    .02,
                                                    .01,
                                                    " | ".join(errors[:3]),
                                                    fontsize=6,
                                                )
                                            if gif_y_limits is not None and fig.axes:
                                                fig.axes[0].set_ylim(*gif_y_limits)
                                            figures.append((single_channel, fig))
                                        if not figures:
                                            continue
                                        rows = len(figures)
                                        combined, axes = plt.subplots(
                                            rows,
                                            1,
                                            figsize=(8, max(3.2, 3.0 * rows)),
                                            squeeze=False,
                                        )
                                        for ax, (single_channel, fig) in zip(axes.flat, figures):
                                            source_ax = fig.axes[0] if fig.axes else None
                                            if source_ax is not None:
                                                for line in source_ax.get_lines():
                                                    ax.plot(
                                                        line.get_xdata(),
                                                        line.get_ydata(),
                                                        color=line.get_color(),
                                                        linewidth=line.get_linewidth(),
                                                        linestyle=line.get_linestyle(),
                                                        alpha=line.get_alpha(),
                                                    )
                                                ax.set_xlim(source_ax.get_xlim())
                                                ax.set_ylim(
                                                        *(gif_y_limits or source_ax.get_ylim())
                                                    )
                                                ax.set_xlabel(source_ax.get_xlabel())
                                                ax.set_ylabel(source_ax.get_ylabel())
                                            ax.set_title(
                                                _trace_channel_label(single_channel)
                                            )
                                            ax.grid(alpha=.25)
                                            plt.close(fig)
                                        combined.suptitle(
                                            f"{frame_group} iteration {frame_iteration} — {selected_trace_type}",
                                            fontsize=12,
                                        )
                                        combined.tight_layout()
                                        yield combined
                                    else:
                                        fig, errors = _plot_traces(
                                            full_session,
                                            frame_observation,
                                            corrected,
                                            selected_channels,
                                            trace_analysis,
                                            correction_label,
                                            overlaid=True,
                                            trace_items=frame_traces,
                                            normalize_to_peak=normalize_to_peak,
                                            corrected_trace_key=corrected_trace_key,
                                        )
                                        if errors:
                                            fig.text(
                                                .02,
                                                .01,
                                                " | ".join(errors[:3]),
                                                fontsize=6,
                                            )
                                        if gif_y_limits is not None and fig.axes:
                                            fig.axes[0].set_ylim(*gif_y_limits)
                                        yield fig

                            gif_bytes = _figures_to_gif(
                                swv_frames(),
                                trace_gif_duration,
                                total_frames=len(frame_observations),
                                progress_callback=(
                                    lambda current, total:
                                    trace_progress.progress(
                                        min(0.95, 0.95 * current / max(total, 1)),
                                        text=(
                                            f"Rendering SWV frame {current}/"
                                            f"{max(total, current)}"
                                        ),
                                    )
                                ),
                            )
                            trace_progress.progress(
                                1.0,
                                text=f"SWV GIF complete ({len(frame_observations)} frames)",
                            )
                            st.session_state[trace_gif_key] = gif_bytes
                        except Exception as exc:
                            st.session_state.pop(trace_gif_key, None)
                            st.error(f"SWV GIF generation failed: {exc}")
                trace_gif_bytes = st.session_state.get(trace_gif_key)
                if trace_gif_bytes:
                    st.image(trace_gif_bytes)
                    safe_trace_name = re.sub(
                        r"[^A-Za-z0-9._-]+",
                        "_",
                        f"{selected_trace_type}_{trace_layout}",
                    ).strip("_")
                    st.download_button(
                        "Download SWV GIF",
                        data=trace_gif_bytes,
                        file_name=f"swv_traces_{safe_trace_name}.gif",
                        mime="image/gif",
                        key=f"{trace_gif_key}_download",
                    )

    with real_data:
        st.caption(
            "These plots use completed experimental observations only; no surrogate predictions are shown."
        )
        all_real_observations = group_scoped_observations
        real_observations = [
            observation
            for observation in all_real_observations
            if (
                selected_iteration_scope == "all"
                or int(observation.get("iteration", 0))
                <= int(selected_iteration_scope)
            )
        ]
        real_iteration_path = pd.DataFrame([
            {
                "iteration": int(observation.get("iteration", 0)),
                "group_id": int(observation.get("group_id", 1)),
                **{
                    parameter: float(value)
                    for parameter, value in (
                        observation.get("params") or {}
                    ).items()
                    if parameter in PARAMETERS and value is not None
                },
            }
            for observation in real_observations
        ])
        real_axis_ranges = {}
        for parameter in PARAMETERS:
            parameter_values = [
                float((observation.get("params") or {})[parameter])
                for observation in all_real_observations
                if (observation.get("params") or {}).get(parameter) is not None
            ]
            if parameter_values:
                real_axis_ranges[parameter] = (
                    min(parameter_values),
                    max(parameter_values),
                )
        real_scope_key = str(selected_observation_group_scope)
        real_metric_options = [
            *_q_relevant_metrics(
                REAL_DATA_METRICS,
                session["config"],
                paired_objective,
                phase=None,
            ),
            "Count",
        ]
        _preserve_valid_widget_value(
            "bo_real_metric",
            real_metric_options,
            real_metric_options[0],
        )
        real_metric = st.selectbox(
            "Measured metric",
            real_metric_options,
            key="bo_real_metric",
        )
        count_mode = real_metric == "Count"
        if count_mode:
            real_phase = "measurement"
            st.caption(
                "Count uses completed observation occurrences and is independent "
                "of buffer/target measurement phase."
            )
            bin_columns = st.columns(3)
            count_bin_sizes = {
                "frequency": float(bin_columns[0].number_input(
                    "Frequency bin size (Hz)",
                    min_value=0.001,
                    value=50.0,
                    format="%.3f",
                    key="bo_count_frequency_bin",
                )),
                "amplitude": float(bin_columns[1].number_input(
                    "Amplitude bin size (V)",
                    min_value=0.000001,
                    value=0.005,
                    format="%.6f",
                    key="bo_count_amplitude_bin",
                )),
                "step_potential": float(bin_columns[2].number_input(
                    "Step-size bin (V)",
                    min_value=0.000001,
                    value=0.001,
                    format="%.6f",
                    key="bo_count_step_bin",
                )),
            }
        else:
            real_phase_options = (
                ["buffer", "target", "both"]
                if paired_objective
                else ["measurement"]
            )
            _preserve_valid_widget_value(
                "bo_real_phase",
                real_phase_options,
                real_phase_options[0],
            )
            real_phase = st.radio(
                "Measurement phase",
                real_phase_options,
                horizontal=True,
                format_func=str.title,
                key="bo_real_phase",
            )
            count_bin_sizes = {}
        observed_group_ids = {
            int(observation.get("group_id", 1))
            for observation in real_observations
        }
        real_groups = [
            group for group in groups if group["id"] in observed_group_ids
        ]
        handling_options = [
            "Average selected channels",
            "Overlay selected channels",
            "Plot channels individually",
        ]
        if real_groups and not count_mode:
            handling_options.append("Plot channel groups")
        real_channel_mode = st.radio(
            "Channel handling",
            handling_options,
            horizontal=True,
            key=f"bo_real_channel_mode_{real_metric}_{real_phase}",
        )
        selected_real_groups = []
        selected_real_channels = []
        if real_channel_mode == "Plot channel groups":
            selected_group_ids = st.multiselect(
                "Metric channel groups",
                [group["id"] for group in real_groups],
                default=[group["id"] for group in real_groups],
                format_func=lambda group_id: next(
                    (
                        f"{group['name']} (channels "
                        f"{', '.join(map(str, group['channels']))})"
                        for group in real_groups
                        if group["id"] == group_id
                    ),
                    f"Group {group_id}",
                ),
                key=f"bo_real_groups_{real_scope_key}_{real_metric}_{real_phase}",
            )
            selected_real_groups = [
                group for group in real_groups
                if group["id"] in selected_group_ids
            ]
        else:
            real_channels = _real_data_channels(all_real_observations)
            selected_real_channels = st.multiselect(
                "Metric channels",
                real_channels,
                default=real_channels,
                format_func=lambda channel: f"Ch {channel}",
                key=f"bo_real_channels_{real_scope_key}_{real_metric}_{real_phase}",
            )
        real_color_by = "Measured value"
        real_group_color_values = None
        real_group_color_label = None
        if real_channel_mode == "Overlay selected channels":
            real_metadata = _channel_group_optimization_metadata(
                full_session,
                groups,
            )
            real_color_options: dict[
                str,
                tuple[dict[int, float] | None, str | None],
            ] = {
                "Measured value": (None, None),
                "Channel": (None, None),
                "Group": (None, None),
                "Exploration weight": (
                    {
                        item["id"]: float(item["exploration"])
                        for item in real_metadata
                        if item["exploration"] is not None
                    },
                    "Exploration",
                ),
                "Exploitation weight": (
                    {
                        item["id"]: float(item["exploitation"])
                        for item in real_metadata
                        if item["exploitation"] is not None
                    },
                    "Exploitation",
                ),
            }
            parameter_config = (
                (full_session.get("config") or {}).get("parameters") or {}
            )
            for parameter in PARAMETERS:
                metadata_values = {
                    item["id"]: float(item["gp_falloff_fractions"][parameter])
                    for item in real_metadata
                    if parameter in item["gp_falloff_fractions"]
                }
                if metadata_values:
                    parameter_label = (
                        (parameter_config.get(parameter) or {}).get("label")
                        or parameter
                    )
                    real_color_options[f"GP falloff — {parameter_label}"] = (
                        metadata_values,
                        f"{parameter_label} GP falloff",
                    )
            real_color_by = st.selectbox(
                "Overlay color",
                list(real_color_options),
                key=f"bo_real_overlay_color_{real_metric}_{real_phase}",
            )
            real_group_color_values, real_group_color_label = (
                real_color_options[real_color_by]
            )
        requested_phases = ("buffer", "target") if real_phase == "both" else (real_phase,)
        if count_mode:
            count_channels = (
                [
                    str(channel)
                    for group in selected_real_groups
                    for channel in group.get("channels", [])
                ]
                if real_channel_mode == "Plot channel groups"
                else selected_real_channels
            )
            real_points_by_phase = {
                "measurement": _real_count_occurrences(
                    real_observations,
                    count_channels,
                )
            }
        else:
            real_points_by_phase = {
                phase: (
                    _real_group_metric_points(
                        real_observations,
                        real_metric,
                        phase,
                        selected_real_groups,
                    )
                    if real_channel_mode == "Plot channel groups"
                    else _real_metric_points(
                        real_observations,
                        real_metric,
                        phase,
                        selected_real_channels,
                        average_channels=real_channel_mode == "Average selected channels",
                    )
                )
                for phase in requested_phases
            }
        real_movie_source_by_phase = {
            phase: points.copy()
            for phase, points in real_points_by_phase.items()
        }
        combined_real_points = pd.concat(
            [
                points.assign(phase=phase)
                for phase, points in real_points_by_phase.items()
                if not points.empty
            ],
            ignore_index=True,
        ) if any(not points.empty for points in real_points_by_phase.values()) else pd.DataFrame()
        if combined_real_points.empty:
            st.info(
                "No recorded values are available for this metric, phase, "
                "and channel or group selection."
            )
        else:
            real_dimensions = [
                parameter
                for parameter in PARAMETERS
                if parameter in combined_real_points.columns
                and combined_real_points[parameter].nunique(dropna=True) > 1
                and (
                    not count_mode
                    or parameter in (
                        "frequency",
                        "amplitude",
                        "step_potential",
                    )
                )
            ]
            if not real_dimensions:
                st.info("No experimental parameter varied in the recorded observations.")
            else:
                real_view_options = ["1D slice"]
                if len(real_dimensions) >= 2:
                    real_view_options.append("2D map")
                if len(real_dimensions) >= 3:
                    real_view_options.append("3D tensor")
                _preserve_valid_widget_value(
                    "bo_real_view",
                    real_view_options,
                    "1D slice",
                )
                real_view = st.radio(
                    "Real-data view",
                    real_view_options,
                    horizontal=True,
                    key="bo_real_view",
                )
                _preserve_valid_widget_value(
                    "bo_real_x",
                    real_dimensions,
                    real_dimensions[0],
                )
                real_x = st.selectbox("Real-data X", real_dimensions, key="bo_real_x")
                real_y_options = [name for name in real_dimensions if name != real_x]
                if real_view != "1D slice":
                    _preserve_valid_widget_value(
                        "bo_real_y",
                        real_y_options,
                        real_y_options[0],
                    )
                real_y = (
                    st.selectbox("Real-data Y", real_y_options, key="bo_real_y")
                    if real_view != "1D slice" else None
                )
                real_z_options = [
                    name for name in real_dimensions if name not in (real_x, real_y)
                ]
                if real_view == "3D tensor":
                    _preserve_valid_widget_value(
                        "bo_real_z",
                        real_z_options,
                        real_z_options[0],
                    )
                real_z = (
                    st.selectbox("Real-data Z", real_z_options, key="bo_real_z")
                    if real_view == "3D tensor" else None
                )
                if count_mode:
                    count_dimensions = [
                        dimension
                        for dimension in (real_x, real_y, real_z)
                        if dimension is not None
                    ]
                    real_points_by_phase = {
                        phase: _bin_real_count_points(
                            points,
                            count_dimensions,
                            count_bin_sizes,
                            average_channels=(
                                real_channel_mode
                                == "Average selected channels"
                            ),
                        )
                        for phase, points in real_points_by_phase.items()
                    }
                    combined_real_points = pd.concat(
                        [
                            points.assign(phase=phase)
                            for phase, points in real_points_by_phase.items()
                            if not points.empty
                        ],
                        ignore_index=True,
                    )
                if count_mode:
                    full_range_points = _real_count_occurrences(
                        all_real_observations,
                        count_channels,
                    )
                    full_range_points = _bin_real_count_points(
                        full_range_points,
                        count_dimensions,
                        count_bin_sizes,
                        average_channels=(
                            real_channel_mode == "Average selected channels"
                        ),
                    )
                    for dimension in count_dimensions:
                        if not full_range_points.empty:
                            half_bin = count_bin_sizes[dimension] / 2.0
                            real_axis_ranges[dimension] = (
                                float(full_range_points[dimension].min())
                                - half_bin,
                                float(full_range_points[dimension].max())
                                + half_bin,
                            )
                    full_value_frames = [full_range_points]
                else:
                    full_value_frames = []
                    for phase in requested_phases:
                        full_points = (
                            _real_group_metric_points(
                                all_real_observations,
                                real_metric,
                                phase,
                                selected_real_groups,
                            )
                            if real_channel_mode == "Plot channel groups"
                            else _real_metric_points(
                                all_real_observations,
                                real_metric,
                                phase,
                                selected_real_channels,
                                average_channels=(
                                    real_channel_mode
                                    == "Average selected channels"
                                ),
                            )
                        )
                        if not full_points.empty:
                            full_value_frames.append(full_points)
                full_values = pd.concat(
                    full_value_frames,
                    ignore_index=True,
                )["value"] if full_value_frames else pd.Series(dtype=float)
                real_value_range = (
                    (float(full_values.min()), float(full_values.max()))
                    if not full_values.empty
                    else None
                )
                real_log_frequency = st.checkbox(
                    "Log-scale frequency axis",
                    value=False,
                    disabled="frequency" not in (real_x, real_y, real_z),
                    help="Uses a logarithmic axis whenever frequency is a displayed dimension.",
                    key="bo_real_log_frequency",
                )
                real_show_iteration_path = st.checkbox(
                    "Show iteration path",
                    value=True,
                    key="bo_real_show_iteration_path",
                    help="Shows the chronological red-to-black path and iteration colorbar.",
                )
                if real_phase == "both" and real_view == "1D slice":
                    if real_channel_mode == "Plot channels individually":
                        individual_channels = sorted(
                            set(real_points_by_phase["buffer"].get("channel", []))
                            | set(real_points_by_phase["target"].get("channel", [])),
                            key=_channel_sort_key,
                        )
                        for channel in individual_channels:
                            st.markdown(f"#### Ch {channel}")
                            _sized_plot_container(
                                st, plot_width_percent
                            ).plotly_chart(
                                _plot_real_data_both_1d(
                                    real_points_by_phase["buffer"].loc[
                                        real_points_by_phase["buffer"]["channel"] == channel
                                    ],
                                    real_points_by_phase["target"].loc[
                                        real_points_by_phase["target"]["channel"] == channel
                                    ],
                                    real_metric,
                                    real_x,
                                    dot_size=plot_dot_size,
                                    log_frequency=real_log_frequency,
                                    color_by=real_color_by,
                                    group_color_values=real_group_color_values,
                                    group_color_label=real_group_color_label,
                                    show_iteration_path=real_show_iteration_path,
                                    axis_ranges=real_axis_ranges,
                                    value_range=real_value_range,
                                ),
                                use_container_width=True,
                                key=(
                                    f"bo_real_plot_both_{channel}_{real_metric}_{real_x}_"
                                    f"{real_scope_key}"
                                ),
                            )
                    else:
                        _sized_plot_container(
                            st, plot_width_percent
                        ).plotly_chart(
                            _plot_real_data_both_1d(
                                real_points_by_phase["buffer"],
                                real_points_by_phase["target"],
                                real_metric,
                                real_x,
                                dot_size=plot_dot_size,
                                log_frequency=real_log_frequency,
                                color_by=real_color_by,
                                group_color_values=real_group_color_values,
                                group_color_label=real_group_color_label,
                                show_iteration_path=real_show_iteration_path,
                                axis_ranges=real_axis_ranges,
                                value_range=real_value_range,
                            ),
                            use_container_width=True,
                            key=(
                                f"bo_real_plot_both_{real_metric}_{real_x}_"
                                f"{real_channel_mode}_{real_scope_key}"
                            ),
                        )
                elif real_phase == "both":
                    buffer_column, target_column = st.columns(2)
                    for column, phase in (
                        (buffer_column, "buffer"),
                        (target_column, "target"),
                    ):
                        column.subheader(phase.title())
                        if real_points_by_phase[phase].empty:
                            column.info(f"No {phase} values are available.")
                        else:
                            phase_points = real_points_by_phase[phase]
                            plot_series = (
                                list(phase_points.groupby("channel", sort=False))
                                if real_channel_mode in (
                                    "Plot channels individually",
                                    "Plot channel groups",
                                )
                                else [(None, phase_points)]
                            )
                            for series_name, series_points in plot_series:
                                if series_name is not None:
                                    series_heading = (
                                        f"Ch {series_name}"
                                        if real_channel_mode == "Plot channels individually"
                                        else str(series_name)
                                    )
                                    column.markdown(f"**{series_heading}**")
                                _sized_plot_container(
                                    column, plot_width_percent
                                ).plotly_chart(
                                    _plot_real_data_landscape(
                                        series_points,
                                        real_metric,
                                        phase,
                                        real_view,
                                        real_x,
                                        real_y,
                                        real_z,
                                        tensor_height=plot_3d_height,
                                        dot_size=plot_dot_size,
                                        log_frequency=real_log_frequency,
                                        color_by=real_color_by,
                                        group_color_values=real_group_color_values,
                                        group_color_label=real_group_color_label,
                                        iteration_path=real_iteration_path,
                                        show_iteration_path=real_show_iteration_path,
                                        count_bin_sizes=count_bin_sizes,
                                        axis_ranges=real_axis_ranges,
                                        value_range=real_value_range,
                                    ),
                                    use_container_width=True,
                                    key=(
                                        f"bo_real_plot_{phase}_{series_name}_{real_metric}_"
                                        f"{real_view}_{real_x}_{real_y}_{real_z}_"
                                        f"{real_channel_mode}_{real_scope_key}"
                                    ),
                                )
                else:
                    phase_points = real_points_by_phase[real_phase]
                    plot_series = (
                        list(phase_points.groupby("channel", sort=False))
                        if (
                            real_channel_mode in (
                                "Plot channels individually",
                                "Plot channel groups",
                            )
                        )
                        else [(None, phase_points)]
                    )
                    for series_name, series_points in plot_series:
                        if series_name is not None:
                            series_heading = (
                                f"Ch {series_name}"
                                if real_channel_mode == "Plot channels individually"
                                else str(series_name)
                            )
                            st.markdown(f"#### {series_heading}")
                        _sized_plot_container(
                            st, plot_width_percent
                        ).plotly_chart(
                            _plot_real_data_landscape(
                                series_points,
                                real_metric,
                                real_phase,
                                real_view,
                                real_x,
                                real_y,
                                real_z,
                                tensor_height=plot_3d_height,
                                dot_size=plot_dot_size,
                                log_frequency=real_log_frequency,
                                color_by=real_color_by,
                                group_color_values=real_group_color_values,
                                group_color_label=real_group_color_label,
                                iteration_path=real_iteration_path,
                                show_iteration_path=real_show_iteration_path,
                                count_bin_sizes=count_bin_sizes,
                                axis_ranges=real_axis_ranges,
                                value_range=real_value_range,
                            ),
                            use_container_width=True,
                            key=(
                                f"bo_real_plot_{series_name}_{real_metric}_{real_phase}_"
                                f"{real_view}_{real_x}_{real_y}_{real_z}_"
                                f"{real_channel_mode}_{real_scope_key}"
                            ),
                        )

                st.divider()
                st.markdown("#### Real-data landscape animation")
                real_gif_duration = st.slider(
                    "Real-data GIF frame duration",
                    min_value=100,
                    max_value=2000,
                    value=400,
                    step=100,
                    format="%d ms",
                    key="bo_real_gif_duration",
                )
                real_gif_key = (
                    f"bo_real_gif_{real_scope_key}_{real_metric}_{real_phase}_"
                    f"{real_channel_mode}_{real_view}_{real_x}_{real_y}_{real_z}"
                )
                if st.button(
                    "Generate real-data GIF",
                    key=f"{real_gif_key}_button",
                ):
                    with st.spinner("Rendering cumulative real-data frames..."):
                        try:
                            movie_targets = []
                            for phase, source_points in (
                                real_movie_source_by_phase.items()
                            ):
                                if source_points.empty:
                                    continue
                                if real_channel_mode in (
                                    "Plot channels individually",
                                    "Plot channel groups",
                                ):
                                    movie_targets.extend([
                                        (
                                            f"{phase.title()} · Ch {channel}",
                                            phase,
                                            channel_points,
                                        )
                                        for channel, channel_points
                                        in source_points.groupby(
                                            "channel",
                                            sort=False,
                                        )
                                    ])
                                else:
                                    movie_targets.append((
                                        phase.title(),
                                        phase,
                                        source_points,
                                    ))
                            generated_gifs = {}
                            movie_dimensions = [
                                dimension
                                for dimension in (real_x, real_y, real_z)
                                if dimension is not None
                            ]
                            target_frame_counts = [
                                len(pd.to_numeric(
                                    source_points["iteration"],
                                    errors="coerce",
                                ).dropna().astype(int).unique())
                                for _name, _phase, source_points in movie_targets
                            ]
                            total_movie_frames = max(
                                1,
                                sum(target_frame_counts),
                            )
                            completed_movie_frames = 0
                            movie_progress = st.progress(
                                0.0,
                                text="Preparing real-data frames...",
                            )
                            for (
                                target_name,
                                phase,
                                source_points,
                            ), target_frame_count in zip(
                                movie_targets,
                                target_frame_counts,
                            ):
                                frame_iterations = sorted(
                                    pd.to_numeric(
                                        source_points["iteration"],
                                        errors="coerce",
                                    ).dropna().astype(int).unique()
                                )

                                def real_frames():
                                    for frame_iteration in frame_iterations:
                                        frame_points = source_points.loc[
                                            pd.to_numeric(
                                                source_points["iteration"],
                                                errors="coerce",
                                            ) <= frame_iteration
                                        ].copy()
                                        if count_mode:
                                            frame_points = _bin_real_count_points(
                                                frame_points,
                                                movie_dimensions,
                                                count_bin_sizes,
                                                average_channels=(
                                                    real_channel_mode
                                                    == "Average selected channels"
                                                ),
                                            )
                                        frame_path = real_iteration_path.loc[
                                            real_iteration_path["iteration"]
                                            <= frame_iteration
                                        ]
                                        yield _plot_real_data_landscape(
                                            frame_points,
                                            real_metric,
                                            phase,
                                            real_view,
                                            real_x,
                                            real_y,
                                            real_z,
                                            tensor_height=plot_3d_height,
                                            dot_size=plot_dot_size,
                                            log_frequency=real_log_frequency,
                                            color_by=real_color_by,
                                            group_color_values=(
                                                real_group_color_values
                                            ),
                                            group_color_label=(
                                                real_group_color_label
                                            ),
                                            iteration_path=frame_path,
                                            show_iteration_path=(
                                                real_show_iteration_path
                                            ),
                                            count_bin_sizes=count_bin_sizes,
                                            axis_ranges=real_axis_ranges,
                                            value_range=real_value_range,
                                        )

                                generated_gifs[target_name] = _figures_to_gif(
                                    real_frames(),
                                    real_gif_duration,
                                    plotly_width=max(
                                        500,
                                        int(1200 * plot_width_percent / 100),
                                    ),
                                    plotly_height=plot_3d_height,
                                    total_frames=target_frame_count,
                                    progress_callback=(
                                        lambda current, _total,
                                        offset=completed_movie_frames,
                                        name=target_name:
                                        movie_progress.progress(
                                            min(
                                                0.95,
                                                0.95
                                                * (offset + current)
                                                / total_movie_frames,
                                            ),
                                            text=(
                                                f"Rendering {name}: frame "
                                                f"{current}/{target_frame_count}"
                                            ),
                                        )
                                    ),
                                )
                                completed_movie_frames += target_frame_count
                            movie_progress.progress(
                                1.0,
                                text=(
                                    f"Real-data GIF complete "
                                    f"({completed_movie_frames} frames)"
                                ),
                            )
                            st.session_state[real_gif_key] = generated_gifs
                        except Exception as exc:
                            st.session_state.pop(real_gif_key, None)
                            st.error(f"Real-data GIF generation failed: {exc}")
                for gif_name, gif_bytes in st.session_state.get(
                    real_gif_key,
                    {},
                ).items():
                    st.caption(f"{gif_name} — {real_view}")
                    st.image(gif_bytes)
                    safe_gif_name = re.sub(
                        r"[^A-Za-z0-9._-]+",
                        "_",
                        gif_name,
                    ).strip("_")
                    st.download_button(
                        f"Download {gif_name} GIF",
                        data=gif_bytes,
                        file_name=(
                            f"real_data_{safe_gif_name}_{real_view}"
                            ".gif"
                        ),
                        mime="image/gif",
                        key=f"{real_gif_key}_download_{safe_gif_name}",
                    )
                if real_metric == "Classic Q":
                    _render_q_equation(session["config"], "classic")
                st.dataframe(combined_real_points, use_container_width=True, hide_index=True)

    with surrogate:
        grouped_surrogate_files = {
            group["id"]: _surrogate_files(full_session["root"], group["id"])
            for group in groups
        }
        available_surrogate_groups = [
            group for group in groups if grouped_surrogate_files[group["id"]]
        ]
        surrogate_state_prefix = (
            f"bo_surrogate_"
            f"{session['state'].get('session_id', session['root'].name)}"
        )
        if available_surrogate_groups:
            surrogate_group_options = [
                group["id"] for group in available_surrogate_groups
            ]
            default_surrogate_groups = (
                [int(selected_observation_group_scope)]
                if (
                    selected_observation_group_scope != "all"
                    and int(selected_observation_group_scope) in surrogate_group_options
                )
                else surrogate_group_options
            )
            surrogate_groups_key = f"{surrogate_state_prefix}_groups"
            stored_surrogate_groups = st.session_state.get(
                surrogate_groups_key,
                default_surrogate_groups,
            )
            st.session_state[surrogate_groups_key] = [
                group_id for group_id in stored_surrogate_groups
                if group_id in surrogate_group_options
            ] or default_surrogate_groups
            selected_surrogate_groups = st.multiselect(
                "Surrogate channel groups",
                surrogate_group_options,
                default=default_surrogate_groups,
                format_func=lambda group_id: next(
                    (
                        f"{group['name']} (channels "
                        f"{', '.join(map(str, group['channels']))})"
                        for group in available_surrogate_groups
                        if group["id"] == group_id
                    ),
                    f"Group {group_id}",
                ),
                key=surrogate_groups_key,
            )
            selected_surrogates = [
                (
                    group,
                    _session_for_channel_group(full_session, group["id"]),
                    grouped_surrogate_files[group["id"]],
                )
                for group in available_surrogate_groups
                if group["id"] in selected_surrogate_groups
            ]
        else:
            ungrouped_files = _surrogate_files(full_session["root"])
            selected_surrogates = (
                [(None, session, ungrouped_files)] if ungrouped_files else []
            )
        if not selected_surrogates:
            if len(groups) > 1:
                st.info(
                    "Select at least one channel group with saved surrogate artifacts."
                )
            else:
                st.info("No candidate-prediction artifacts were saved for this session.")
        else:
            artifact_iterations = sorted({
                iteration
                for _group, _surrogate_session, files in selected_surrogates
                for iteration in files
            })
            surrogate_iteration_key = (
                f"{surrogate_state_prefix}_artifact_iteration"
                if available_surrogate_groups
                else "bo_surrogate_ungrouped_artifact_iteration"
            )
            _preserve_valid_widget_value(
                surrogate_iteration_key,
                artifact_iterations,
                artifact_iterations[-1],
            )
            artifact_iteration = st.selectbox(
                "Artifact iteration",
                artifact_iterations,
                key=surrogate_iteration_key,
            )
            available_at_iteration = [
                (
                    group,
                    surrogate_session,
                    _recompute_group_surrogate(
                        surrogate_session,
                        pd.read_csv(files[artifact_iteration]),
                        artifact_iteration,
                    ),
                )
                for group, surrogate_session, files in selected_surrogates
                if artifact_iteration in files
            ]
            if len(available_at_iteration) < len(selected_surrogates):
                missing_names = [
                    group["name"] if group is not None else "Ungrouped"
                    for group, _surrogate_session, files in selected_surrogates
                    if artifact_iteration not in files
                ]
                st.caption(
                    "No artifact at this iteration for: " + ", ".join(missing_names)
                )
            predictions_frames = [
                predictions
                for _group, _surrogate_session, predictions in available_at_iteration
            ]
            dimensions = [
                name for name in PARAMETERS
                if all(
                    name in predictions.columns
                    and predictions[name].nunique(dropna=True) > 1
                    for predictions in predictions_frames
                )
            ]
            values = [
                name for name in SURROGATE_VALUES
                if all(name in predictions.columns for predictions in predictions_frames)
            ]
            surrogate_value_key = f"{surrogate_state_prefix}_value"
            _preserve_valid_widget_value(
                surrogate_value_key,
                values,
                values[0],
            )
            value = st.selectbox("Value", values, key=surrogate_value_key)
            view_options = ["1D slice"]
            if len(dimensions) >= 2:
                view_options.append("2D map")
            if len(dimensions) >= 3:
                view_options.append("3D tensor")
            surrogate_view_key = f"{surrogate_state_prefix}_view"
            _preserve_valid_widget_value(
                surrogate_view_key,
                view_options,
                "1D slice",
            )
            view = st.radio(
                "View",
                view_options,
                horizontal=True,
                key=surrogate_view_key,
            )
            surrogate_x_key = f"{surrogate_state_prefix}_x"
            _preserve_valid_widget_value(
                surrogate_x_key,
                dimensions,
                dimensions[0],
            )
            x_name = st.selectbox("X", dimensions, key=surrogate_x_key)
            y_options = [name for name in dimensions if name != x_name]
            surrogate_y_key = f"{surrogate_state_prefix}_y"
            if view != "1D slice":
                _preserve_valid_widget_value(
                    surrogate_y_key,
                    y_options,
                    y_options[0],
                )
            y_name = st.selectbox(
                "Y",
                y_options,
                key=surrogate_y_key,
            ) if view != "1D slice" else None
            z_options = [name for name in dimensions if name not in (x_name, y_name)]
            surrogate_z_key = f"{surrogate_state_prefix}_z"
            if view == "3D tensor":
                _preserve_valid_widget_value(
                    surrogate_z_key,
                    z_options,
                    z_options[0],
                )
            z_name = st.selectbox(
                "Z",
                z_options,
                key=surrogate_z_key,
            ) if view == "3D tensor" else None
            surrogate_log_frequency = st.checkbox(
                "Log-scale frequency axis",
                value=False,
                disabled="frequency" not in (x_name, y_name, z_name),
                help="Uses a logarithmic axis whenever frequency is a displayed dimension.",
                key=f"{surrogate_state_prefix}_log_frequency",
            )
            surrogate_show_iteration_path = st.checkbox(
                "Show iteration path",
                value=True,
                key=f"{surrogate_state_prefix}_show_iteration_path",
                help="Shows the chronological red-to-black path and iteration colorbar.",
            )
            surrogate_gif_duration = st.slider(
                "Surrogate GIF frame duration",
                min_value=100,
                max_value=2000,
                value=400,
                step=100,
                format="%d ms",
                key=f"{surrogate_state_prefix}_gif_duration",
            )
            st.caption(
                "Surrogate predictions and acquisition values are reconstructed "
                "with each group’s saved observations, exploration weight, GP "
                "falloff fractions, and recorded NumPy GP equations. Tested "
                "candidates are excluded from acquisition surfaces."
            )
            for group, surrogate_session, predictions in available_at_iteration:
                surrogate_group_id = group["id"] if group is not None else "ungrouped"
                group_files = next(
                    files
                    for selected_group, _group_session, files
                    in selected_surrogates
                    if (
                        selected_group is None and group is None
                        or (
                            selected_group is not None
                            and group is not None
                            and selected_group["id"] == group["id"]
                        )
                    )
                )
                group_name = (
                    group["name"] if group is not None else "Session"
                )
                if len(available_at_iteration) > 1 and group is not None:
                    st.markdown(
                        f"#### {group['name']} "
                        f"(channels {', '.join(map(str, group['channels']))})"
                    )
                surrogate_figure = _plot_surrogate(
                    surrogate_session, predictions, artifact_iteration, value, view,
                    x_name, y_name, z_name,
                    tensor_height=plot_3d_height,
                    dot_size=plot_dot_size,
                    log_frequency=surrogate_log_frequency,
                    show_iteration_path=surrogate_show_iteration_path,
                )
                if view == "3D tensor":
                    _sized_plot_container(
                        st, plot_width_percent
                    ).plotly_chart(
                        surrogate_figure,
                        use_container_width=True,
                        key=(
                            f"bo_surrogate_3d_{surrogate_group_id}_"
                            f"{artifact_iteration}_{value}_{x_name}_{y_name}_{z_name}"
                        ),
                    )
                else:
                    _sized_plot_container(st, plot_width_percent).pyplot(
                        surrogate_figure,
                        clear_figure=True,
                        use_container_width=True,
                    )
                if (
                    value == "acquisition_value"
                    and "selected_next" in predictions.columns
                    and predictions["selected_next"].any()
                ):
                    eligible_predictions = predictions.loc[
                        ~predictions["already_tested"].astype(bool)
                    ].sort_values(
                        "acquisition_value",
                        ascending=False,
                    )
                    selected_index = predictions.index[
                        predictions["selected_next"].astype(bool)
                    ][0]
                    selected_acquisition = float(
                        predictions.loc[selected_index, "acquisition_value"]
                    )
                    selected_rank = int(
                        (
                            eligible_predictions["acquisition_value"]
                            > selected_acquisition
                        ).sum()
                        + 1
                    )
                    st.caption(
                        f"Selected iteration {artifact_iteration + 1}: "
                        f"eligible acquisition rank {selected_rank} of "
                        f"{len(eligible_predictions)}."
                    )
                with st.expander(
                    f"Candidate predictions — "
                    f"{group['name'] if group is not None else 'session'}"
                ):
                    st.dataframe(predictions, use_container_width=True, hide_index=True)
                st.markdown(f"##### {group_name} animation")
                surrogate_gif_key = (
                    f"{surrogate_state_prefix}_gif_{surrogate_group_id}_"
                    f"{view}_{value}_{x_name}_{y_name}_{z_name}"
                )
                if st.button(
                    f"Generate {group_name} GIF",
                    key=f"{surrogate_gif_key}_button",
                ):
                    with st.spinner(
                        f"Rendering {group_name} surrogate frames..."
                    ):
                        try:
                            surrogate_progress = st.progress(
                                0.0,
                                text=f"Preparing {group_name} frames...",
                            )

                            def surrogate_frames():
                                for frame_iteration in sorted(group_files):
                                    frame_predictions = pd.read_csv(
                                        group_files[frame_iteration]
                                    )
                                    frame_predictions = (
                                        _recompute_group_surrogate(
                                            surrogate_session,
                                            frame_predictions,
                                            frame_iteration,
                                        )
                                    )
                                    required_columns = [
                                        name for name in (
                                            x_name,
                                            y_name,
                                            z_name,
                                            value,
                                        )
                                        if name is not None
                                    ]
                                    if not all(
                                        name in frame_predictions.columns
                                        for name in required_columns
                                    ):
                                        continue
                                    yield _plot_surrogate(
                                        surrogate_session,
                                        frame_predictions,
                                        frame_iteration,
                                        value,
                                        view,
                                        x_name,
                                        y_name,
                                        z_name,
                                        tensor_height=plot_3d_height,
                                        dot_size=plot_dot_size,
                                        log_frequency=surrogate_log_frequency,
                                        show_iteration_path=(
                                            surrogate_show_iteration_path
                                        ),
                                    )

                            gif_bytes = _figures_to_gif(
                                surrogate_frames(),
                                surrogate_gif_duration,
                                plotly_width=max(
                                    500,
                                    int(1200 * plot_width_percent / 100),
                                ),
                                plotly_height=plot_3d_height,
                                total_frames=len(group_files),
                                progress_callback=(
                                    lambda current, _total:
                                    surrogate_progress.progress(
                                        min(
                                            0.95,
                                            0.95 * current / len(group_files),
                                        ),
                                        text=(
                                            f"Rendering {group_name}: frame "
                                            f"{current}/{len(group_files)}"
                                        ),
                                    )
                                ),
                            )
                            surrogate_progress.progress(
                                1.0,
                                text=(
                                    f"{group_name} GIF complete "
                                    f"({len(group_files)} frames)"
                                ),
                            )
                            st.session_state[surrogate_gif_key] = gif_bytes
                        except Exception as exc:
                            st.session_state.pop(surrogate_gif_key, None)
                            st.error(
                                f"{group_name} GIF generation failed: {exc}"
                            )
                gif_bytes = st.session_state.get(surrogate_gif_key)
                if gif_bytes:
                    st.image(gif_bytes)
                    safe_gif_name = re.sub(
                        r"[^A-Za-z0-9._-]+",
                        "_",
                        group_name,
                    ).strip("_")
                    st.download_button(
                        f"Download {group_name} GIF",
                        data=gif_bytes,
                        file_name=(
                            f"surrogate_{safe_gif_name}_{view}.gif"
                        ),
                        mime="image/gif",
                        key=(
                            f"{surrogate_gif_key}_download_"
                            f"{safe_gif_name}"
                        ),
                    )
                st.divider()
            _render_q_equation(
                session["config"],
                "paired" if paired_objective else "classic",
            )

    with gifs:
        st.subheader("Synchronized GIF set")
        if not groups:
            st.info("No observation groups are available for GIF generation.")
        else:
            gif_group_options = [group["id"] for group in groups]
            default_gif_group = (
                int(selected_observation_group_scope)
                if (
                    selected_observation_group_scope != "all"
                    and int(selected_observation_group_scope) in gif_group_options
                )
                else gif_group_options[0]
            )
            selected_gif_group = st.selectbox(
                "Observation group",
                gif_group_options,
                index=gif_group_options.index(default_gif_group),
                format_func=lambda group_id: next(
                    (
                        f"{group['name']} (channels {', '.join(map(str, group['channels']))})"
                        for group in groups
                        if group["id"] == group_id
                    ),
                    f"Group {group_id}",
                ),
                key="bo_gif_observation_group",
            )
            gif_group = next(
                group for group in groups if group["id"] == selected_gif_group
            )
            gif_group_session = _session_for_channel_group(
                full_session,
                selected_gif_group,
            )
            gif_group_files = _surrogate_files(
                full_session["root"],
                selected_gif_group,
            )
            if not gif_group_files:
                gif_group_files = _surrogate_files(full_session["root"])
            gif_observations = sorted(
                gif_group_session["observations"],
                key=lambda item: int(item.get("iteration", 0)),
            )
            gif_observations_by_iteration = {
                int(observation.get("iteration", 0)): observation
                for observation in gif_observations
            }
            gif_iterations = [
                iteration for iteration in sorted(gif_group_files)
                if iteration in gif_observations_by_iteration
            ]
            gif_duration = st.slider(
                "GIF frame duration (ms)",
                min_value=100,
                max_value=1500,
                value=350,
                step=50,
                key=f"bo_gifs_duration_{selected_gif_group}",
            )
            if not gif_group_files:
                st.info("No surrogate artifacts were found for this group.")
            elif not gif_iterations:
                st.info(
                    "No shared iterations have both a surrogate artifact and an observation."
                )
            else:
                first_predictions = pd.read_csv(gif_group_files[gif_iterations[-1]])
                preferred_dimensions = [
                    name for name in ("step_potential", "amplitude", "frequency")
                    if (
                        name in first_predictions.columns
                        and first_predictions[name].nunique(dropna=True) > 1
                    )
                ]
                fallback_dimensions = [
                    name for name in PARAMETERS
                    if (
                        name in first_predictions.columns
                        and first_predictions[name].nunique(dropna=True) > 1
                        and name not in preferred_dimensions
                    )
                ]
                gif_dimensions = (preferred_dimensions + fallback_dimensions)[:3]
                gif_pairs = list(itertools.combinations(gif_dimensions, 2))
                trace_entries_for_group = [
                    trace
                    for iteration in gif_iterations
                    for trace in _trace_paths(
                        full_session,
                        gif_observations_by_iteration[iteration],
                    )
                ]
                group_channel_set = {str(channel) for channel in gif_group["channels"]}
                gif_channels = sorted(
                    {
                        trace["channel"]
                        for trace in trace_entries_for_group
                        if str(trace["channel"]) in group_channel_set
                    },
                    key=_channel_sort_key,
                )
                st.caption(
                    f"Frames: {len(gif_iterations)} shared iterations. "
                    f"Parameters: {', '.join(gif_dimensions) or 'not enough varied parameters'}."
                )
                if len(gif_dimensions) < 3:
                    st.warning(
                        "Surrogate tensor and 2D map GIFs require at least three varied parameters."
                    )
                if not gif_channels:
                    st.warning("No locally accessible SWV traces were found for this group.")

                batch_key = (
                    f"bo_synced_gifs_{selected_gif_group}_"
                    f"{len(gif_iterations)}_{gif_duration}"
                )
                if st.button(
                    "Generate all GIFs",
                    type="primary",
                    key=f"{batch_key}_button",
                    disabled=(len(gif_dimensions) < 3 or not gif_channels),
                ):
                    try:
                        generated_gifs: dict[str, bytes] = {}
                        gif_targets = [
                            "Raw SWVs",
                            "Raw normalized SWVs",
                            "Surrogate tensor Q",
                            "Surrogate tensor acquisition",
                            *[
                                f"2D Q {x_name} vs {y_name}"
                                for x_name, y_name in gif_pairs
                            ],
                            *[
                                f"2D acquisition {x_name} vs {y_name}"
                                for x_name, y_name in gif_pairs
                            ],
                        ]
                        total_target_frames = max(1, len(gif_targets) * len(gif_iterations))
                        completed_target_frames = 0
                        batch_progress = st.progress(
                            0.0,
                            text="Preparing synchronized GIFs...",
                        )

                        def update_batch_progress(name: str, current: int, total: int) -> None:
                            nonlocal completed_target_frames
                            fraction = (
                                completed_target_frames + current
                            ) / total_target_frames
                            batch_progress.progress(
                                min(0.98, fraction),
                                text=f"Rendering {name}: frame {current}/{total}",
                            )

                        raw_y_limits = _swv_global_y_limits(
                            [gif_observations_by_iteration[it] for it in gif_iterations],
                            full_session,
                            gif_channels,
                            corrected=False,
                            analysis=trace_analysis,
                            normalize_to_peak=False,
                            corrected_trace_key="smoothed_corrected_current",
                        )
                        normalized_y_limits = (-0.2, 1.2)

                        def swv_gif_frames(
                            normalized_raw: bool,
                            y_limits: tuple[float, float] | None,
                        ):
                            for iteration in gif_iterations:
                                frame_observation = gif_observations_by_iteration[iteration]
                                frame_traces = _trace_paths(full_session, frame_observation)
                                fig, errors = _plot_traces(
                                    full_session,
                                    frame_observation,
                                    corrected=normalized_raw,
                                    selected_channels=gif_channels,
                                    analysis=trace_analysis,
                                    correction_label=correction_label,
                                    overlaid=True,
                                    trace_items=frame_traces,
                                    normalize_to_peak=normalized_raw,
                                    corrected_trace_key=(
                                        "corrected_current"
                                        if normalized_raw
                                        else "smoothed_corrected_current"
                                    ),
                                )
                                if y_limits is not None and fig.axes:
                                    fig.axes[0].set_ylim(*y_limits)
                                if errors:
                                    fig.text(.02, .01, " | ".join(errors[:3]), fontsize=6)
                                yield fig

                        for name, normalized_raw, y_limits in (
                            ("Raw SWVs", False, raw_y_limits),
                            ("Raw normalized SWVs", True, normalized_y_limits),
                        ):
                            generated_gifs[name] = _figures_to_gif(
                                swv_gif_frames(normalized_raw, y_limits),
                                gif_duration,
                                total_frames=len(gif_iterations),
                                progress_callback=lambda current, total, target=name: update_batch_progress(
                                    target,
                                    current,
                                    total,
                                ),
                            )
                            completed_target_frames += len(gif_iterations)

                        def surrogate_frames(
                            value_key: str,
                            view: str,
                            x_name: str,
                            y_name: str | None,
                            z_name: str | None = None,
                        ):
                            for iteration in gif_iterations:
                                predictions = pd.read_csv(gif_group_files[iteration])
                                required = [
                                    name for name in (x_name, y_name, z_name, value_key)
                                    if name is not None
                                ]
                                if not all(name in predictions.columns for name in required):
                                    continue
                                yield _plot_surrogate(
                                    gif_group_session,
                                    predictions,
                                    iteration,
                                    value_key,
                                    view,
                                    x_name,
                                    y_name,
                                    z_name,
                                    tensor_height=plot_3d_height,
                                    dot_size=plot_dot_size,
                                    show_iteration_path=True,
                                )

                        x_name, y_name, z_name = gif_dimensions[:3]
                        for name, value_key in (
                            ("Surrogate tensor Q", "predicted_mean_Q"),
                            ("Surrogate tensor acquisition", "acquisition_value"),
                        ):
                            generated_gifs[name] = _figures_to_gif(
                                surrogate_frames(value_key, "3D tensor", x_name, y_name, z_name),
                                gif_duration,
                                plotly_width=max(500, int(1200 * plot_width_percent / 100)),
                                plotly_height=plot_3d_height,
                                total_frames=len(gif_iterations),
                                progress_callback=lambda current, total, target=name: update_batch_progress(
                                    target,
                                    current,
                                    total,
                                ),
                            )
                            completed_target_frames += len(gif_iterations)

                        for label_prefix, value_key in (
                            ("2D Q", "predicted_mean_Q"),
                            ("2D acquisition", "acquisition_value"),
                        ):
                            for pair_x, pair_y in gif_pairs:
                                name = f"{label_prefix} {pair_x} vs {pair_y}"
                                generated_gifs[name] = _figures_to_gif(
                                    surrogate_frames(value_key, "2D map", pair_x, pair_y),
                                    gif_duration,
                                    total_frames=len(gif_iterations),
                                    progress_callback=lambda current, total, target=name: update_batch_progress(
                                        target,
                                        current,
                                        total,
                                    ),
                                )
                                completed_target_frames += len(gif_iterations)

                        batch_progress.progress(
                            1.0,
                            text=f"Generated {len(generated_gifs)} synchronized GIFs.",
                        )
                        st.session_state[batch_key] = generated_gifs
                    except Exception as exc:
                        st.session_state.pop(batch_key, None)
                        st.error(f"Synchronized GIF generation failed: {exc}")

                generated_gifs = st.session_state.get(batch_key, {})
                if generated_gifs:
                    encoded = [
                        (
                            name,
                            base64.b64encode(gif_bytes).decode("ascii"),
                        )
                        for name, gif_bytes in generated_gifs.items()
                    ]
                    cards = "\n".join(
                        (
                            "<div class='gif-card'>"
                            f"<div class='gif-title'>{name}</div>"
                            f"<img data-gif-src='data:image/gif;base64,{encoded_gif}' />"
                            "</div>"
                        )
                        for name, encoded_gif in encoded
                    )
                    components.html(
                        f"""
                        <style>
                        .gif-grid {{
                            display: grid;
                            grid-template-columns: repeat(2, minmax(0, 1fr));
                            gap: 14px;
                            align-items: start;
                        }}
                        .gif-card {{
                            border: 1px solid #ddd;
                            padding: 8px;
                            background: white;
                        }}
                        .gif-title {{
                            font: 600 14px sans-serif;
                            margin-bottom: 6px;
                        }}
                        .gif-card img {{
                            width: 100%;
                            display: block;
                        }}
                        </style>
                        <div class="gif-grid">{cards}</div>
                        <script>
                        const images = Array.from(document.querySelectorAll("img[data-gif-src]"));
                        window.requestAnimationFrame(() => {{
                            images.forEach((image) => {{
                                image.src = image.dataset.gifSrc;
                            }});
                        }});
                        </script>
                        """,
                        height=max(520, 360 * ((len(encoded) + 1) // 2)),
                        scrolling=True,
                    )
                    for name, gif_bytes in generated_gifs.items():
                        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
                        st.download_button(
                            f"Download {name}",
                            data=gif_bytes,
                            file_name=f"{gif_group['name']}_{safe_name}.gif",
                            mime="image/gif",
                            key=f"{batch_key}_download_{safe_name}",
                        )

    with pdf_export:
        st.subheader("Exhaustive BO session report")
        st.write(
            "Build a shareable PDF that explains the experiment in acquisition order, "
            "shows objective improvement and parameter movement, and includes measured "
            "landscapes, surrogate evolution, and every locally available SWV trace."
        )
        st.caption(
            f"Corrected traces will use {correction_label}. Large sessions can produce "
            "long reports and may take several minutes to render."
        )
        pdf_c1, pdf_c2 = st.columns(2)
        surrogate_pdf_scope = pdf_c1.selectbox(
            "Surrogate pages",
            ["Final artifact per group (fast)", "Every artifact (slow)"],
            key="bo_pdf_surrogate_scope",
            help=(
                "Final artifact keeps the last saved surrogate per group. "
                "Every artifact renders all saved surrogate iterations."
            ),
        )
        swv_pdf_scope = pdf_c2.selectbox(
            "SWV pages",
            ["First/middle/final per group (fast)", "Every iteration (slow)"],
            key="bo_pdf_swv_scope",
            help=(
                "The fast option keeps representative cumulative SWV overlays. "
                "Every iteration can be very slow because cumulative overlays grow each page."
            ),
        )
        report_key = (
            f"bo_pdf_bytes_{session['state'].get('session_id', session['root'].name)}_"
            f"{selected_group_id}_{surrogate_pdf_scope}_{swv_pdf_scope}"
        )
        if st.button(
            "Build exhaustive PDF",
            type="primary",
            key=f"build_{report_key}",
        ):
            try:
                pdf_started_at = time.monotonic()
                pdf_status = st.empty()
                pdf_progress = st.progress(
                    0.0,
                    text="0% | ETA calculating... | Preparing PDF report...",
                )
                last_pdf_fraction = {"value": 0.0}

                def update_pdf_progress(fraction: float, text: str) -> None:
                    fraction = max(
                        last_pdf_fraction["value"],
                        min(max(float(fraction), 0.0), 1.0),
                    )
                    last_pdf_fraction["value"] = fraction
                    percent = int(round(fraction * 100))
                    elapsed = time.monotonic() - pdf_started_at
                    eta = None
                    if 0.0 < fraction < 1.0:
                        eta = elapsed * (1.0 - fraction) / fraction
                    elif fraction >= 1.0:
                        eta = 0.0
                    status_text = (
                        f"{percent}% | ETA {_format_duration(eta)} | {text}"
                    )
                    pdf_status.info(status_text)
                    pdf_progress.progress(fraction, text=status_text)

                st.session_state[report_key] = build_bo_session_pdf(
                    session,
                    trace_analysis,
                    correction_label,
                    progress_callback=update_pdf_progress,
                    surrogate_iteration_mode=(
                        "all" if surrogate_pdf_scope == "Every artifact (slow)" else "final"
                    ),
                    swv_iteration_mode=(
                        "all" if swv_pdf_scope == "Every iteration (slow)" else "milestones"
                    ),
                )
                update_pdf_progress(1.0, "PDF ready.")
            except Exception as exc:
                st.session_state.pop(report_key, None)
                st.error(f"PDF generation failed: {exc}")
        report_bytes = st.session_state.get(report_key)
        if report_bytes:
            st.success(f"PDF ready ({len(report_bytes) / (1024 * 1024):.1f} MB).")
            st.download_button(
                "Download BO session report",
                data=report_bytes,
                file_name=f"{session['root'].name}_report.pdf",
                mime="application/pdf",
                key=f"download_{report_key}",
            )
