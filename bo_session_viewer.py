"""Read-only Streamlit viewer for experiment_automation BO session folders."""

from __future__ import annotations

import json
import itertools
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path, PureWindowsPath
import re
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
import streamlit as st

from core.analysis import analyze_swv_arrays
from core.io import load_swv_csv


PARAMETERS = (
    "begin_potential", "end_potential", "step_potential", "amplitude",
    "frequency", "conditioning_potential", "conditioning_time",
)
SURROGATE_VALUES = ("predicted_mean_Q", "predicted_std_Q", "acquisition_value")
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


def _plot_trend(
    frame: pd.DataFrame,
    metric: str,
    group_layout: str = "Plot groups overlaid",
):
    if metric not in frame.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No changing trend metrics are available.",
            x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
        )
        fig.update_layout(width=650, height=340)
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
        if metric == "Q_run":
            fig.add_trace(go.Scatter(
                x=averaged["iteration"],
                y=averaged["value"].cummax(),
                mode="lines",
                name="Group-average best",
                line={"dash": "dot"},
                hoverinfo="skip",
            ))
        fig.update_layout(
            title=f"{metric} over BO iterations — average across groups",
            xaxis_title="BO iteration",
            yaxis_title=metric,
            width=650,
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
        for index, ((_group_id, rows), group_name) in enumerate(
            zip(grouped_rows, titles)
        ):
            subplot_row, subplot_column = divmod(index, columns)
            rows = rows.sort_values("iteration")
            iterations = pd.to_numeric(
                rows["iteration"],
                errors="coerce",
            ).astype(int)
            row_values = pd.to_numeric(rows[metric], errors="coerce")
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=row_values,
                    mode="lines+markers",
                    name=group_name,
                    customdata=iterations,
                    hovertemplate=(
                        f"{group_name}<br>Iteration %{{x}}<br>"
                        f"{metric}: %{{y:.4g}}<extra></extra>"
                    ),
                ),
                row=subplot_row + 1,
                col=subplot_column + 1,
            )
            if metric == "Q_run":
                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=row_values.cummax(),
                        mode="lines",
                        name=f"{group_name} best",
                        line={"dash": "dot"},
                        hoverinfo="skip",
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
        fig.update_layout(
            title=f"{metric} over BO iterations — separate groups",
            width=800,
            height=max(360, 280 * rows_count),
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
        fig.add_trace(go.Scatter(
            x=iterations,
            y=row_values,
            mode="lines+markers",
            name=group_name,
            marker={"size": 8},
            customdata=iterations,
            hovertemplate=(
                f"{group_name}<br>Iteration %{{x}}<br>"
                f"{metric}: %{{y:.4g}}<extra></extra>"
            ),
        ))
        if metric == "Q_run":
            fig.add_trace(go.Scatter(
                x=iterations,
                y=row_values.cummax(),
                mode="lines",
                name=f"{group_name} best",
                line={"dash": "dot"},
                hoverinfo="skip",
            ))
    fig.update_layout(
        title=f"{metric} over BO iterations",
        xaxis_title="BO iteration",
        yaxis_title=metric,
        width=650,
        height=340,
        margin={"l": 65, "r": 20, "t": 50, "b": 55},
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
        fig.update_layout(width=650, height=340)
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
            trace = go.Scatter(
                x=rows["iteration"],
                y=rows["value"],
                mode="lines+markers",
                name=trace_name,
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
        width=800 if faceted else 650,
        height=max(340, 270 * subplot_rows),
        margin={"l": 65, "r": 20, "t": 65, "b": 55},
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
            width=760,
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
        width=700,
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
            width=800,
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
        width=780,
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


def _plot_real_data_landscape(
    points: pd.DataFrame,
    metric_label: str,
    phase: str,
    view: str,
    x_name: str,
    y_name: str | None,
    z_name: str | None,
):
    def series_label(value: Any) -> str:
        text = str(value)
        if text == "Average":
            return "Channel average"
        if text.lower().startswith("group"):
            return text
        return f"Ch {text}"

    hover_data = ["iteration", "channel"]
    if view == "3D tensor":
        fig = go.Figure()
        for channel, group in points.groupby("channel", sort=False):
            fig.add_trace(go.Scatter3d(
                x=group[x_name],
                y=group[y_name],
                z=group[z_name],
                mode="markers",
                name=series_label(channel),
                marker={
                    "size": 6,
                    "color": group["value"],
                    "colorscale": "Viridis",
                    "showscale": len(fig.data) == 0,
                    "colorbar": {"title": metric_label},
                },
                customdata=group[hover_data],
                hovertemplate=(
                    f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}<br>"
                    f"{z_name}: %{{z:.4g}}<br>{metric_label}: %{{marker.color:.4g}}"
                    "<br>Iteration %{customdata[0]}<br>Series %{customdata[1]}<extra></extra>"
                ),
            ))
        fig.update_layout(
            scene={
                "xaxis_title": x_name,
                "yaxis_title": y_name,
                "zaxis_title": z_name,
            },
            width=680,
            height=480,
        )
    elif view == "2D map":
        valid, grid = _interpolated_2d_grid(points, x_name, y_name)
        fig = go.Figure()
        if grid is not None:
            grid_x, grid_y, grid_values = grid
            fig.add_trace(go.Heatmap(
                x=grid_x,
                y=grid_y,
                z=grid_values,
                colorscale="Viridis",
                colorbar={"title": metric_label},
                hovertemplate=(
                    f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}<br>"
                    f"Interpolated {metric_label}: %{{z:.4g}}<extra></extra>"
                ),
                connectgaps=False,
            ))
        fig.add_trace(go.Scatter(
            x=valid[x_name],
            y=valid[y_name],
            mode="markers",
            marker={
                "size": 8,
                "color": valid["value"],
                "colorscale": "Viridis",
                "showscale": grid is None,
                "colorbar": {"title": metric_label},
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
        fig.update_layout(
            xaxis_title=x_name,
            yaxis_title=y_name,
            width=650,
            height=420,
        )
    else:
        fig = go.Figure()
        for channel, group in points.groupby("channel", sort=False):
            ordered = group.sort_values(x_name)
            fig.add_trace(go.Scatter(
                x=ordered[x_name],
                y=ordered["value"],
                mode="lines+markers",
                name=series_label(channel),
                customdata=ordered[["iteration", "channel"]],
                hovertemplate=(
                    f"{x_name}: %{{x:.4g}}<br>{metric_label}: %{{y:.4g}}<br>"
                    "Iteration %{customdata[0]}<br>Channel %{customdata[1]}<extra></extra>"
                ),
            ))
        fig.update_layout(
            xaxis_title=x_name,
            yaxis_title=metric_label,
            width=650,
            height=360,
        )
    fig.update_layout(
        title=f"Measured {phase} {metric_label} | {view}",
        margin={"l": 65, "r": 30, "t": 55, "b": 55},
    )
    return fig


def _plot_real_data_both_1d(
    buffer_points: pd.DataFrame,
    target_points: pd.DataFrame,
    metric_label: str,
    x_name: str,
):
    fig = go.Figure()
    colors = {"buffer": "#1f77b4", "target": "#ff7f0e"}
    for phase, points in (("buffer", buffer_points), ("target", target_points)):
        if points.empty:
            continue
        for channel_index, (channel, group) in enumerate(points.groupby("channel", sort=False)):
            ordered = group.sort_values(x_name)
            fig.add_trace(go.Scatter(
                x=ordered[x_name],
                y=ordered["value"],
                mode="lines+markers",
                name=phase.title(),
                legendgroup=phase,
                showlegend=channel_index == 0,
                line={"color": colors[phase]},
                marker={"color": colors[phase]},
                customdata=ordered[["iteration", "channel"]],
                hovertemplate=(
                    f"{phase.title()}<br>{x_name}: %{{x:.4g}}<br>"
                    f"{metric_label}: %{{y:.4g}}<br>"
                    "Iteration %{customdata[0]}<br>Channel %{customdata[1]}<extra></extra>"
                ),
            ))
    fig.update_layout(
        title=f"Measured buffer vs target {metric_label} | 1D slice",
        xaxis_title=x_name,
        yaxis_title=metric_label,
        width=680,
        height=380,
        margin={"l": 65, "r": 30, "t": 55, "b": 55},
    )
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
    """Clean measured points and linearly interpolate inside their convex hull."""
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
        grid_values = griddata(
            unique[[x_name, y_name]].to_numpy(),
            unique["value"].to_numpy(),
            (mesh_x, mesh_y),
            method="linear",
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
                valid = predictions[[x_name, y_name, value_key]].apply(
                    pd.to_numeric,
                    errors="coerce",
                ).dropna()
                try:
                    scatter = ax.tricontourf(
                        valid[x_name],
                        valid[y_name],
                        valid[value_key],
                        levels=14,
                        cmap="viridis",
                    )
                except Exception:
                    scatter = ax.scatter(
                        valid[x_name],
                        valid[y_name],
                        c=valid[value_key],
                        cmap="viridis",
                        s=8,
                        alpha=.55,
                    )
                tested_for_axes = [
                    observation for observation in tested
                    if all(
                        (observation.get("params") or {}).get(axis) is not None
                        for axis in (x_name, y_name)
                    )
                ]
                if tested_for_axes:
                    ax.plot(
                        [observation["params"][x_name] for observation in tested_for_axes],
                        [observation["params"][y_name] for observation in tested_for_axes],
                        color="#d67b32",
                        marker="o",
                        linewidth=1.3,
                        markersize=4,
                    )
                ax.set(xlabel=x_name, ylabel=y_name)
                map_fig.colorbar(scatter, ax=ax, shrink=.75)
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


def build_bo_session_pdf(
    session: dict,
    trace_analysis: dict,
    correction_label: str,
) -> bytes:
    """Build an exhaustive, shareable report from persisted BO artifacts."""
    observations = session["observations"]
    history = _observation_table(session)
    config = session["config"]
    paired = any(
        str(obs.get("objective", "")).lower() == "paired_response"
        for obs in observations
    )
    classic_equation = _classic_q_equation(config)
    objective_equation = _paired_q_equation(config) if paired else classic_equation
    objective_equation_label = "Paired Q equation" if paired else "Classic Q equation"
    q_values = [float(obs.get("Q_run", np.nan)) for obs in observations]
    best = observations[int(np.nanargmax(q_values))]
    multiple_groups = len({
        int(obs.get("group_id", 1)) for obs in observations
    }) > 1

    def plot_grouped_history(ax, frame: pd.DataFrame, column: str) -> None:
        if multiple_groups and "group_id" in frame.columns:
            for _group_id, rows in frame.groupby("group_id", sort=True):
                label = (
                    str(rows["group_name"].iloc[0])
                    if "group_name" in rows.columns
                    else f"Group {int(_group_id)}"
                )
                ax.plot(
                    rows["iteration"],
                    pd.to_numeric(rows[column], errors="coerce"),
                    marker="o",
                    label=label,
                )
            ax.legend()
        else:
            ax.plot(
                frame["iteration"],
                pd.to_numeric(frame[column], errors="coerce"),
                marker="o",
            )

    output = BytesIO()
    with PdfPages(output) as pdf:
        pdf._bo_parameter_context = _session_parameter_context(observations)
        _pdf_text_page(pdf, "Bayesian Optimization Session Report", [
            f"Session: {session['state'].get('session_id', session['root'].name)}",
            f"Objective: {'Paired response' if paired else 'Standard quality'}",
            f"Completed observations: {len(observations)}",
            f"Candidate count: {session['state'].get('candidate_count', 'unknown')}",
            (
                f"Best observation: {best.get('group_name', f'Group {best.get('group_id', 1)}')} "
                f"iteration {best.get('iteration')}"
                if multiple_groups
                else f"Best iteration: {best.get('iteration')}"
            ),
            f"Best Q_run: {float(best.get('Q_run', 0)):.6g}",
            "",
            "Report order:",
            "1. Chronological acquisition view",
            "2. Global trends and parameter evolution",
            "3. Per-channel data and measured landscapes",
            "4. Per-iteration surrogate and SWV data",
        ])
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

        channels = _real_data_channels(observations)
        _pdf_text_page(pdf, "1. Chronological View", [
            "Measurements are shown in acquisition order.",
            "Vertical markers identify Buffer → Target and Target → Buffer transitions.",
            "Each mean view is followed by overlaid channels and individual-channel plots.",
        ])
        if paired:
            _pdf_chronological_section(
                pdf, observations, config, channels, classic_equation,
            )
        else:
            _pdf_text_page(pdf, "Chronological View Not Applicable", [
                "This is a standard BO session with one measurement phase per iteration.",
                "Iteration-ordered performance is shown in Global Trends.",
            ])

        _pdf_text_page(pdf, "2. Global Trends", [
            "This section shows overall objective improvement and optimizer parameter movement.",
            "Only quantities that vary and affect Q are included.",
        ])

        # Objective evolution.
        values = [float(obs.get("Q_run", np.nan)) for obs in observations]
        if _has_numeric_variation(values):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            plot_grouped_history(ax, history, "Q_run")
            ax.set(xlabel="BO iteration", ylabel="Q", title="Objective improvement over time")
            ax.grid(alpha=.25)
            _pdf_equation_footer(fig, objective_equation_label, objective_equation)
            _pdf_save(pdf, fig)

        # Parameter movement.
        parameter_columns = [
            name for name in PARAMETERS
            if name in history.columns and _has_numeric_variation(history[name])
        ]
        if parameter_columns:
            fig, axes = plt.subplots(
                (len(parameter_columns) + 1) // 2, 2,
                figsize=(8.5, 3 * ((len(parameter_columns) + 1) // 2)),
                squeeze=False,
            )
            for ax, name in zip(axes.flat, parameter_columns):
                plot_grouped_history(ax, history, name)
                ax.set(xlabel="Iteration", ylabel=name, title=name)
                ax.grid(alpha=.25)
            for ax in axes.flat[len(parameter_columns):]:
                ax.axis("off")
            _pdf_save(pdf, fig, "Experimental parameter evolution")

        # Every numeric Q history series.
        q_columns = [
            column for column in _numeric_columns(history)
            if (
                "q" in str(column).lower()
                and column != "Q_run"
                and _history_metric_impacts_q(column, config, paired)
                and _has_numeric_variation(history[column])
            )
        ]
        global_q_columns = [
            column for column in q_columns
            if not re.fullmatch(r"(?:Q_ch\d+|ch\d+_.+)", str(column), re.IGNORECASE)
        ]
        channel_q_columns = [
            column for column in q_columns if column not in global_q_columns
        ]
        for start in range(0, len(global_q_columns), 6):
            page_columns = global_q_columns[start:start + 6]
            fig, axes = plt.subplots(3, 2, figsize=(8.5, 10), squeeze=False)
            for ax, column in zip(axes.flat, page_columns):
                plot_grouped_history(ax, history, column)
                ax.set(xlabel="Iteration", ylabel=column, title=column)
                ax.grid(alpha=.25)
            for ax in axes.flat[len(page_columns):]:
                ax.axis("off")
            _pdf_equation_footer(fig, objective_equation_label, objective_equation)
            _pdf_save(pdf, fig, "Global Q components")

        _pdf_text_page(pdf, "3. Per-Channel Data", [
            "This section shows channel-resolved Q behavior, paired phase trends,",
            "and measured parameter landscapes.",
        ])
        for start in range(0, len(channel_q_columns), 6):
            page_columns = channel_q_columns[start:start + 6]
            fig, axes = plt.subplots(3, 2, figsize=(8.5, 10), squeeze=False)
            for ax, column in zip(axes.flat, page_columns):
                ax.plot(
                    history["iteration"],
                    pd.to_numeric(history[column], errors="coerce"),
                    marker="o",
                )
                ax.set(xlabel="Iteration", ylabel=column, title=column)
                ax.grid(alpha=.25)
            for ax in axes.flat[len(page_columns):]:
                ax.axis("off")
            _pdf_equation_footer(fig, objective_equation_label, objective_equation)
            _pdf_save(pdf, fig, "Per-channel Q components")

        if paired:
            # Paired phase trends and fully chronological views for every metric.
            for metric in _q_relevant_metrics(
                PAIRED_TREND_METRICS, config, paired_objective=True,
            ):
                series = _paired_trend_values(observations, metric)
                selected = sorted(series, key=_channel_sort_key)
                paired_values = [
                    value
                    for channel in selected
                    for phase in ("buffer", "target")
                    for value in series[channel][phase]
                ]
                if not _has_numeric_variation(paired_values):
                    continue
                if selected:
                    summary_fig, summary_axes = plt.subplots(
                        1, 2, figsize=(10, 4), squeeze=False,
                    )
                    average_ax, overlay_ax = summary_axes.flat
                    all_iterations = sorted({
                        iteration
                        for channel in selected
                        for iteration in series[channel]["iteration"]
                    })
                    for phase, color in (
                        ("buffer", "#1f77b4"),
                        ("target", "#ff7f0e"),
                    ):
                        averages = []
                        for iteration in all_iterations:
                            iteration_values = []
                            for channel in selected:
                                channel_series = series[channel]
                                for index, recorded in enumerate(channel_series["iteration"]):
                                    value = channel_series[phase][index]
                                    if recorded == iteration and pd.notna(value):
                                        iteration_values.append(float(value))
                            averages.append(
                                float(np.mean(iteration_values))
                                if iteration_values else np.nan
                            )
                        average_ax.plot(
                            all_iterations, averages, marker="o",
                            color=color, label=phase.title(),
                        )
                        for channel in selected:
                            channel_series = series[channel]
                            overlay_ax.plot(
                                channel_series["iteration"],
                                channel_series[phase],
                                marker="o",
                                color=color,
                                alpha=.35,
                            )
                    average_ax.set_title("Mean across channels")
                    overlay_ax.set_title("Channels overlaid")
                    for ax in (average_ax, overlay_ax):
                        ax.set(xlabel="Iteration", ylabel=metric)
                        ax.grid(alpha=.25)
                    average_ax.legend()
                    if metric == "Classic Q":
                        _pdf_equation_footer(
                            summary_fig, "Classic Q equation", classic_equation,
                        )
                    _pdf_save(
                        pdf, summary_fig,
                        f"Buffer vs target — {metric} — summary",
                    )

                    rows = (len(selected) + 1) // 2
                    mpl_fig, axes = plt.subplots(
                        rows, 2, figsize=(8.5, max(4, 3 * rows)), squeeze=False,
                    )
                    for ax, channel in zip(axes.flat, selected):
                        channel_series = series[channel]
                        ax.plot(
                            channel_series["iteration"], channel_series["buffer"],
                            marker="o", color="#1f77b4", label="Buffer",
                        )
                        ax.plot(
                            channel_series["iteration"], channel_series["target"],
                            marker="o", color="#ff7f0e", label="Target",
                        )
                        ax.set(
                            xlabel="Iteration", ylabel=metric,
                            title=f"Channel {channel}",
                        )
                        ax.grid(alpha=.25)
                    for ax in axes.flat[len(selected):]:
                        ax.axis("off")
                    axes.flat[0].legend()
                    if metric == "Classic Q":
                        _pdf_equation_footer(
                            mpl_fig, "Classic Q equation", classic_equation,
                        )
                    _pdf_save(pdf, mpl_fig, f"Buffer vs target — {metric}")
        # Real measured landscapes: every metric, phase, and varied parameter combination.
        phases = ("buffer", "target") if paired else ("measurement",)
        for phase in phases:
            for metric in _q_relevant_metrics(
                REAL_DATA_METRICS,
                config,
                paired,
                phase=phase,
            ):
                points = _real_metric_points(
                    observations, metric, phase, channels, average_channels=False,
                )
                dimensions = [
                    name for name in PARAMETERS
                    if name in points.columns and points[name].nunique(dropna=True) > 1
                ]
                _pdf_metric_landscapes(
                    pdf,
                    points,
                    metric,
                    phase,
                    dimensions,
                    q_equation=classic_equation if metric == "Classic Q" else None,
                )

        _pdf_text_page(pdf, "4. Per-Iteration Data", [
            "This section follows model and trace artifacts iteration by iteration.",
            "It includes every varying surrogate view and every locally available",
            "raw and corrected SWV trace.",
        ])

        # Keep each iteration together: trace overlays first, surrogate second.
        multiple_groups = len({
            int(observation.get("group_id", 1))
            for observation in observations
        }) > 1
        artifacts = (
            {}
            if multiple_groups
            else _surrogate_files(
                session["root"], session.get("selected_group_id"),
            )
        )
        observations_by_key = {
            (
                (int(observation.get("group_id", 1)), int(observation.get("iteration", 0)))
                if multiple_groups
                else int(observation.get("iteration", 0))
            ): observation
            for observation in observations
        }
        report_keys = sorted(observations_by_key)
        if not multiple_groups:
            report_keys = sorted(set(report_keys) | set(artifacts))
        for report_key in report_keys:
            iteration = report_key[1] if multiple_groups else report_key
            observation = observations_by_key.get(report_key)
            if observation is not None:
                trace_rows = _trace_paths(session, observation)
                trace_channels = sorted(
                    {row["channel"] for row in trace_rows},
                    key=_channel_sort_key,
                )
                if trace_channels:
                    # Raw overlay is intentionally the first plot for the iteration.
                    for corrected in (False, True):
                        fig, errors = _plot_traces(
                            session,
                            observation,
                            corrected,
                            trace_channels,
                            trace_analysis,
                            correction_label,
                            overlaid=True,
                        )
                        if errors:
                            fig.text(.02, .01, " | ".join(errors[:3]), fontsize=6)
                        _pdf_save(pdf, fig)
            artifact_path = artifacts.get(iteration)
            if artifact_path is not None:
                _pdf_surrogate_iteration(
                    pdf,
                    iteration,
                    artifact_path,
                    objective_equation_label,
                    objective_equation,
                    observations,
                )
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


@st.cache_data(show_spinner=False)
def _cached_swv_arrays(
    path_text: str,
    modified_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an unchanged SWV file once per Streamlit process."""
    del modified_ns  # Included in the cache key to invalidate changed files.
    return load_swv_csv(path_text)


@st.cache_data(show_spinner=False)
def _cached_corrected_swv_arrays(
    path_text: str,
    modified_ns: int,
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
) -> tuple[np.ndarray, np.ndarray]:
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
    corrected_current = result.get(
        "smoothed_corrected_current",
        result.get("corrected_current"),
    )
    return corrected_voltage, corrected_current


def _swv_trace_arrays(
    path: Path,
    corrected: bool,
    analysis: dict,
) -> tuple[np.ndarray, np.ndarray]:
    modified_ns = path.stat().st_mtime_ns
    if not corrected:
        return _cached_swv_arrays(str(path), modified_ns)
    minimum_peak = analysis.get("min_peak_height_uA")
    return _cached_corrected_swv_arrays(
        str(path),
        modified_ns,
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


def _plot_traces(
    session: dict,
    observation: dict,
    corrected: bool,
    selected_channels: list[str],
    analysis: dict,
    correction_label: str,
    overlaid: bool,
    trace_items: list[dict] | None = None,
):
    traces = [
        item for item in (
            trace_items
            if trace_items is not None
            else _trace_paths(session, observation)
        )
        if item["channel"] in selected_channels
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
            voltage, y = _swv_trace_arrays(path, corrected, analysis)
            channel_label = f"Ch {channel}" if channel != "Unknown" else "Unknown channel"
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
                f"Ch {channel}" if channel != "Unknown" else "Unknown channel"
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
            xlabel="Voltage (V)", ylabel="Current (uA)",
            title=(
                f"Iteration {observation.get('iteration')} "
                f"{'smoothed corrected' if corrected else 'raw'} SWV traces"
                f"{channel_title}"
                f"{f' ({correction_label})' if corrected else ''}"
                f"\n{parameter_text}"
            ),
        )
        ax.grid(alpha=.25)
        if len(traces) <= 16:
            ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, errors


def _plot_iteration_trace_overlay(
    trace_entries: list[tuple[dict, dict]],
    corrected: bool,
    selected_channels: list[str],
    analysis: dict,
    correction_label: str,
    selection_label: str,
):
    """Overlay traces from many observations with channel-aware colors."""
    entries = [
        (observation, item)
        for observation, item in trace_entries
        if item["channel"] in selected_channels
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    errors = []
    iterations = [
        int(observation.get("iteration", 0))
        for observation, _item in entries
    ]
    plotted_channels = sorted(
        {item["channel"] for _observation, item in entries},
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
        phase, path, channel = item["phase"], item["path"], item["channel"]
        iteration = int(observation.get("iteration", 0))
        try:
            voltage, y = _swv_trace_arrays(path, corrected, analysis)
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
                    f"{'Ch ' + channel if channel != 'Unknown' else 'Unknown channel'}"
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
            f"Ch {channel}" if channel != "Unknown" else "Unknown"
            for channel in selected_channels
        )
        ax.set(
            xlabel="Voltage (V)",
            ylabel="Current (µA)",
            title=(
                f"{selection_label}\n"
                f"{'Smoothed corrected' if corrected else 'Raw'} SWV traces"
                f"{f' ({correction_label})' if corrected else ''}"
                f" | {channel_text}"
            ),
        )
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
    fig.tight_layout()
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


def _observed_points(session: dict, iteration: int, axes: list[str]):
    rows = []
    for obs in session["observations"]:
        if int(obs.get("iteration", 0)) > iteration:
            continue
        params = obs.get("params") or {}
        if all(params.get(axis) is not None for axis in axes):
            rows.append(obs)
    return rows


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
                    x_name: str, y_name: str | None, z_name: str | None):
    current_observation = next(
        (
            observation for observation in session["observations"]
            if int(observation.get("iteration", 0)) == int(iteration)
        ),
        None,
    )
    title = (
        f"{view} | {value} | artifact iteration {iteration}<br>"
        f"{_iteration_parameter_text(current_observation)}"
    )
    if view == "3D tensor":
        valid = frame[[x_name, y_name, z_name, value]].apply(pd.to_numeric, errors="coerce").dropna()
        fig = go.Figure(go.Scatter3d(
            x=valid[x_name],
            y=valid[y_name],
            z=valid[z_name],
            mode="markers",
            name="Candidate predictions",
            marker={
                "size": 4,
                "color": valid[value],
                "colorscale": "Viridis",
                "opacity": 0.55,
                "showscale": True,
                "colorbar": {"title": value},
            },
            customdata=valid[value],
            hovertemplate=(
                f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}<br>"
                f"{z_name}: %{{z:.4g}}<br>{value}: %{{customdata:.4g}}"
                "<extra></extra>"
            ),
        ))
        observed = _observed_points(session, iteration, [x_name, y_name, z_name])
        if observed:
            fig.add_trace(go.Scatter3d(
                x=[float(obs["params"][x_name]) for obs in observed],
                y=[float(obs["params"][y_name]) for obs in observed],
                z=[float(obs["params"][z_name]) for obs in observed],
                mode="lines+markers",
                name="Observed path",
                line={"color": "#d67b32", "width": 5},
                marker={"color": "#d67b32", "size": 5},
                hovertemplate=(
                    f"{x_name}: %{{x:.4g}}<br>{y_name}: %{{y:.4g}}<br>"
                    f"{z_name}: %{{z:.4g}}<extra>Observed</extra>"
                ),
            ))
        fig.update_layout(
            title=title,
            scene={
                "xaxis_title": x_name,
                "yaxis_title": y_name,
                "zaxis_title": z_name,
            },
            width=760,
            height=620,
            margin={"l": 0, "r": 0, "t": 80, "b": 0},
        )
        return fig
    elif view == "2D map":
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        valid = frame[[x_name, y_name, value]].apply(pd.to_numeric, errors="coerce").dropna()
        try:
            mesh = ax.tricontourf(valid[x_name], valid[y_name], valid[value], levels=14, cmap="viridis")
        except Exception:
            mesh = ax.scatter(valid[x_name], valid[y_name], c=valid[value], cmap="viridis")
        observed = _observed_points(session, iteration, [x_name, y_name])
        if observed:
            ax.plot(
                [obs["params"][x_name] for obs in observed],
                [obs["params"][y_name] for obs in observed],
                color="#d67b32", marker="o", label="observed path",
            )
            ax.legend()
        ax.set(xlabel=x_name, ylabel=y_name)
        fig.colorbar(mesh, ax=ax, label=value)
    else:
        fig, ax = plt.subplots(figsize=(6.4, 3.5))
        valid = frame[[x_name, value]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.scatter(valid[x_name], valid[value], color="#155e63", s=14, alpha=.35, label="candidate predictions")
        grouped = valid.groupby(x_name)[value].median()
        if len(grouped) > 1:
            ax.plot(grouped.index, grouped.values, color="#d67b32", label="median at X")
        observed = _observed_points(session, iteration, [x_name])
        if observed:
            observed_x = [obs["params"][x_name] for obs in observed]
            observed_q = [obs["Q_run"] for obs in observed]
            ax.plot(
                observed_x, observed_q,
                color="#d67b32", marker="o", linewidth=1.4,
                label="tested parameter sets",
            )
        ax.set(xlabel=x_name, ylabel=value)
        ax.legend()
    ax.set_title(
        f"{view} | {value} | artifact iteration {iteration}\n"
        f"{_iteration_parameter_text(current_observation)}"
    )
    ax.grid(alpha=.2)
    fig.tight_layout()
    return fig


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
    overview, traces, real_data, surrogate, pdf_export = st.tabs(
        [
            "History & scores",
            "SWV traces",
            "Real data landscapes",
            "Surrogate",
            "PDF Export",
        ]
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
        if (
            metric in trend_history.columns
            and
            "group_id" in trend_history.columns
            and trend_history["group_id"].nunique(dropna=True) > 1
        ):
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
                )
            else:
                trend_figure = go.Figure()
                trend_figure.add_annotation(
                    text="Select at least one trend channel.",
                    x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
                )
                trend_figure.update_layout(width=650, height=340)
            chart_key_suffix = (
                f"{metric}_{plot_metric}_{channel_layout}_"
                f"{group_layout}_{'_'.join(trend_channels) or 'none'}"
            )
        else:
            trend_figure = _plot_trend(trend_history, metric, group_layout)
            chart_key_suffix = f"{metric}_{group_layout}"
        chart_key_suffix = f"{trend_scope_key}_{chart_key_suffix}"
        trend_events = [st.plotly_chart(
            trend_figure,
            use_container_width=False,
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
            for obs in group_scoped_observations
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
                group_scoped_observations,
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
                paired_figure.update_layout(width=700, height=360)
            paired_chart_suffix = (
                f"{paired_metric}_{paired_layout}_"
                f"{'_'.join(selected_paired_channels) or 'none'}"
            )
            paired_events = []
            if (
                paired_layout == "Average selected channels"
                and selected_paired_channels
            ):
                average_column, overlay_column = st.columns(2)
                paired_events.append(average_column.plotly_chart(
                    paired_figure,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points",
                    key=f"bo_paired_trend_{paired_chart_suffix}_average",
                ))
                paired_events.append(overlay_column.plotly_chart(
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
                paired_events.append(st.plotly_chart(
                    _plot_paired_phase_trend(
                        paired_series,
                        paired_metric,
                        selected_paired_channels,
                        "Separate plots",
                    ),
                    use_container_width=False,
                    on_select="rerun",
                    selection_mode="points",
                    key=f"bo_paired_trend_{paired_chart_suffix}_separate",
                ))
            else:
                paired_events.append(st.plotly_chart(
                    paired_figure,
                    use_container_width=False,
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
                group_scoped_observations
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
                group_scoped_observations,
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
                    group_scoped_observations,
                    session["config"],
                    chronological_metric,
                    selected_chronological_channels,
                    average_channels=False,
                )
            if chronological_points.empty:
                st.info("No chronological values are available for this selection.")
            else:
                chronological_suffix = (
                    f"{chronological_metric}_{chronological_mode}_"
                    f"{'_'.join(selected_chronological_channels)}"
                )
                chronological_events = []
                if chronological_raw_points is not None:
                    average_column, overlay_column = st.columns(2)
                    chronological_events.append(average_column.plotly_chart(
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
                    chronological_events.append(overlay_column.plotly_chart(
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
                    chronological_events.append(st.plotly_chart(
                        _plot_chronological(
                            chronological_raw_points,
                            phase_transitions,
                            chronological_metric,
                            "Separate plots",
                        ),
                        use_container_width=False,
                        on_select="rerun",
                        selection_mode="points",
                        key=f"bo_chronological_{chronological_suffix}_separate",
                    ))
                else:
                    chronological_events.append(st.plotly_chart(
                        _plot_chronological(
                            chronological_points,
                            phase_transitions,
                            chronological_metric,
                            chronological_mode,
                        ),
                        use_container_width=False,
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
        with st.spinner("Locating SWV traces..."):
            trace_entries = [
                (trace_observation, trace)
                for trace_observation in trace_observations
                for trace in _trace_paths(full_session, trace_observation)
            ]
        available_traces = [trace for _observation, trace in trace_entries]
        available_channels = sorted(
            {item["channel"] for item in available_traces},
            key=_channel_sort_key,
        )
        if not available_traces:
            st.info(
                "No locally accessible raw SWV files were found for this selection. "
                "The recorded CSVs must remain inside or beside the experiment folder."
            )
        else:
            selected_channels = st.multiselect(
                "Channels to display",
                available_channels,
                default=available_channels,
                format_func=lambda channel: f"Ch {channel}" if channel != "Unknown" else "Unknown channel",
                key=(
                    f"bo_trace_channels_{selected_observation_group_scope}_"
                    f"{selected_iteration_scope}"
                ),
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
                        f"#### {'Channel ' + channel if channel != 'Unknown' else 'Unknown channel'}"
                    )
                raw_column, corrected_column = st.columns(2)
                for column, corrected, heading in (
                    (raw_column, False, "Raw SWV traces"),
                    (corrected_column, True, "Smoothed corrected traces"),
                ):
                    column.subheader(heading)
                    with column:
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
                                )
                    column.pyplot(
                        figure,
                        clear_figure=True,
                        use_container_width=True,
                    )
                    for error in errors:
                        column.warning(error)

    with real_data:
        st.caption(
            "These plots use completed experimental observations only; no surrogate predictions are shown."
        )
        real_phase_options = (
            ["buffer", "target", "both"]
            if paired_objective
            else ["measurement"]
        )
        real_phase = st.radio(
            "Measurement phase",
            real_phase_options,
            horizontal=True,
            format_func=str.title,
            key="bo_real_phase",
        )
        real_metric = st.selectbox(
            "Measured metric",
            _q_relevant_metrics(
                REAL_DATA_METRICS,
                session["config"],
                paired_objective,
                phase=None if real_phase == "both" else real_phase,
            ),
            key="bo_real_metric",
        )
        observed_group_ids = {
            int(observation.get("group_id", 1)) for observation in observations
        }
        real_groups = [
            group for group in groups if group["id"] in observed_group_ids
        ]
        handling_options = [
            "Average selected channels",
            "Plot channels individually",
        ]
        if real_groups:
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
                key=f"bo_real_groups_{real_metric}_{real_phase}",
            )
            selected_real_groups = [
                group for group in real_groups
                if group["id"] in selected_group_ids
            ]
        else:
            real_channels = _real_data_channels(observations)
            selected_real_channels = st.multiselect(
                "Metric channels",
                real_channels,
                default=real_channels,
                format_func=lambda channel: f"Ch {channel}",
                key=f"bo_real_channels_{real_metric}_{real_phase}",
            )
        requested_phases = ("buffer", "target") if real_phase == "both" else (real_phase,)
        real_points_by_phase = {
            phase: (
                _real_group_metric_points(
                    observations,
                    real_metric,
                    phase,
                    selected_real_groups,
                )
                if real_channel_mode == "Plot channel groups"
                else _real_metric_points(
                    observations,
                    real_metric,
                    phase,
                    selected_real_channels,
                    average_channels=real_channel_mode == "Average selected channels",
                )
            )
            for phase in requested_phases
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
            ]
            if not real_dimensions:
                st.info("No experimental parameter varied in the recorded observations.")
            else:
                real_view_options = ["1D slice"]
                if len(real_dimensions) >= 2:
                    real_view_options.append("2D map")
                if len(real_dimensions) >= 3:
                    real_view_options.append("3D tensor")
                real_view = st.radio(
                    "Real-data view",
                    real_view_options,
                    horizontal=True,
                    key="bo_real_view",
                )
                real_x = st.selectbox("Real-data X", real_dimensions, key="bo_real_x")
                real_y_options = [name for name in real_dimensions if name != real_x]
                real_y = (
                    st.selectbox("Real-data Y", real_y_options, key="bo_real_y")
                    if real_view != "1D slice" else None
                )
                real_z_options = [
                    name for name in real_dimensions if name not in (real_x, real_y)
                ]
                real_z = (
                    st.selectbox("Real-data Z", real_z_options, key="bo_real_z")
                    if real_view == "3D tensor" else None
                )
                if real_phase == "both" and real_view == "1D slice":
                    st.plotly_chart(
                        _plot_real_data_both_1d(
                            real_points_by_phase["buffer"],
                            real_points_by_phase["target"],
                            real_metric,
                            real_x,
                        ),
                        use_container_width=False,
                        key=(
                            f"bo_real_plot_both_{real_metric}_{real_x}_"
                            f"{real_channel_mode}"
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
                                if real_channel_mode == "Plot channel groups"
                                else [(None, phase_points)]
                            )
                            for series_name, series_points in plot_series:
                                if series_name is not None:
                                    column.markdown(f"**{series_name}**")
                                column.plotly_chart(
                                    _plot_real_data_landscape(
                                        series_points,
                                        real_metric,
                                        phase,
                                        real_view,
                                        real_x,
                                        real_y,
                                        real_z,
                                    ),
                                    use_container_width=True,
                                    key=(
                                        f"bo_real_plot_{phase}_{series_name}_{real_metric}_"
                                        f"{real_view}_{real_x}_{real_y}_{real_z}_"
                                        f"{real_channel_mode}"
                                    ),
                                )
                else:
                    phase_points = real_points_by_phase[real_phase]
                    plot_series = (
                        list(phase_points.groupby("channel", sort=False))
                        if (
                            real_channel_mode == "Plot channel groups"
                            and real_view != "1D slice"
                        )
                        else [(None, phase_points)]
                    )
                    for series_name, series_points in plot_series:
                        if series_name is not None:
                            st.markdown(f"#### {series_name}")
                        st.plotly_chart(
                            _plot_real_data_landscape(
                                series_points,
                                real_metric,
                                real_phase,
                                real_view,
                                real_x,
                                real_y,
                                real_z,
                            ),
                            use_container_width=False,
                            key=(
                                f"bo_real_plot_{series_name}_{real_metric}_{real_phase}_"
                                f"{real_view}_{real_x}_{real_y}_{real_z}_"
                                f"{real_channel_mode}"
                            ),
                        )
                if real_metric == "Classic Q":
                    _render_q_equation(session["config"], "classic")
                st.dataframe(combined_real_points, use_container_width=True, hide_index=True)

    with surrogate:
        surrogate_session = session
        surrogate_group_id = session.get("selected_group_id")
        grouped_surrogate_files = {
            group["id"]: _surrogate_files(full_session["root"], group["id"])
            for group in groups
        }
        available_surrogate_groups = [
            group for group in groups if grouped_surrogate_files[group["id"]]
        ]
        if available_surrogate_groups:
            surrogate_group_options = [
                group["id"] for group in available_surrogate_groups
            ]
            surrogate_group_key = (
                f"bo_surrogate_group_"
                f"{session['state'].get('session_id', session['root'].name)}"
            )
            requested_surrogate_group = st.session_state.get(
                surrogate_group_key,
                surrogate_group_id,
            )
            if requested_surrogate_group not in surrogate_group_options:
                requested_surrogate_group = surrogate_group_options[0]
            st.session_state[surrogate_group_key] = requested_surrogate_group
            surrogate_group_id = st.selectbox(
                "Surrogate channel group",
                surrogate_group_options,
                format_func=lambda group_id: next(
                    (
                        f"{group['name']} (channels "
                        f"{', '.join(map(str, group['channels']))})"
                        for group in available_surrogate_groups
                        if group["id"] == group_id
                    ),
                    f"Group {group_id}",
                ),
                key=surrogate_group_key,
            )
            surrogate_session = _session_for_channel_group(
                full_session,
                surrogate_group_id,
            )
            files = grouped_surrogate_files[surrogate_group_id]
        else:
            files = _surrogate_files(full_session["root"])
        if not files:
            if len(groups) > 1:
                st.info(
                    "No group-keyed candidate-prediction artifacts were saved for "
                    "these channel groups. Older grouped sessions reused artifact "
                    "filenames between groups, so those ambiguous files are not shown."
                )
            else:
                st.info("No candidate-prediction artifacts were saved for this session.")
        else:
            artifact_iterations = sorted(files)
            artifact_iteration = st.selectbox(
                "Artifact iteration",
                artifact_iterations,
                index=len(artifact_iterations) - 1,
                key=f"bo_surrogate_iteration_group_{surrogate_group_id}",
            )
            predictions = pd.read_csv(files[artifact_iteration])
            dimensions = [name for name in PARAMETERS if name in predictions.columns and predictions[name].nunique(dropna=True) > 1]
            value = st.selectbox("Value", [name for name in SURROGATE_VALUES if name in predictions.columns])
            view_options = ["1D slice"]
            if len(dimensions) >= 2:
                view_options.append("2D map")
            if len(dimensions) >= 3:
                view_options.append("3D tensor")
            view = st.radio("View", view_options, horizontal=True)
            x_name = st.selectbox("X", dimensions, index=0)
            y_name = st.selectbox("Y", [name for name in dimensions if name != x_name], index=0) if view != "1D slice" else None
            z_options = [name for name in dimensions if name not in (x_name, y_name)]
            z_name = st.selectbox("Z", z_options, index=0) if view == "3D tensor" else None
            surrogate_figure = _plot_surrogate(
                surrogate_session, predictions, artifact_iteration, value, view,
                x_name, y_name, z_name,
            )
            if view == "3D tensor":
                st.plotly_chart(
                    surrogate_figure,
                    use_container_width=True,
                    key=(
                        f"bo_surrogate_3d_{surrogate_group_id}_"
                        f"{artifact_iteration}_{value}_{x_name}_{y_name}_{z_name}"
                    ),
                )
            else:
                st.pyplot(
                    surrogate_figure,
                    clear_figure=True,
                    use_container_width=False,
                )
            _render_q_equation(
                session["config"],
                "paired" if paired_objective else "classic",
            )
            st.dataframe(predictions, use_container_width=True, hide_index=True)

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
        report_key = (
            f"bo_pdf_bytes_{session['state'].get('session_id', session['root'].name)}_"
            f"{selected_group_id}"
        )
        if st.button(
            "Build exhaustive PDF",
            type="primary",
            key=f"build_{report_key}",
        ):
            with st.spinner("Rendering all BO plots into PDF…"):
                try:
                    st.session_state[report_key] = build_bo_session_pdf(
                        session,
                        trace_analysis,
                        correction_label,
                    )
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
