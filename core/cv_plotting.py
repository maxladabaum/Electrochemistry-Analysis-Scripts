from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_cv_overlaid_cycles(
    results: List[dict],
    y_key: str = "smoothed_current",
    title: Optional[str] = None,
    ylabel: str = "Current (uA)",
    colormap_name: str = "plasma",
    linewidth: float = 1.2,
    alpha: float = 0.9,
    show_peak_markers: bool = False,
    show_zero_baseline: bool = False,
    show_baseline: bool = False,
    show_peak_reference_vlines: bool = False,
) -> Optional[plt.Figure]:
    usable = [r for r in results if r.get("voltage") is not None and r.get(y_key) is not None]
    if not usable:
        return None

    n = len(usable)
    cmap = plt.get_cmap(colormap_name, max(n, 2))
    norm = Normalize(vmin=0, vmax=max(n - 1, 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    if show_zero_baseline:
        ax.axhline(0, color="gray", lw=1.0, linestyle="--", alpha=0.8)

    for i, r in enumerate(usable):
        color = cmap(norm(i))
        v = np.asarray(r["voltage"], dtype=float)
        y = np.asarray(r[y_key], dtype=float)
        ax.plot(v, y, color=color, lw=linewidth, alpha=alpha)

        if show_baseline and r.get("baseline_current") is not None and y_key == "smoothed_current":
            ax.plot(
                v,
                np.asarray(r["baseline_current"], dtype=float),
                color=color,
                lw=0.9,
                linestyle="--",
                alpha=min(alpha + 0.05, 1.0),
            )

        if show_peak_markers:
            for idx_key, marker, label in (
                ("oxidation_peak_idx", "^", "oxidation"),
                ("reduction_peak_idx", "v", "reduction"),
            ):
                idx = r.get(idx_key)
                if idx is None or not (0 <= int(idx) < len(v)):
                    continue
                ax.scatter(
                    v[int(idx)],
                    y[int(idx)],
                    color=color,
                    s=34,
                    marker=marker,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5,
                    label=label if i == 0 else None,
                )

    if show_peak_reference_vlines:
        reference_specs = (
            ("oxidation_peak_voltage", "crimson", "Ox", [("initial", ":"), ("average", "--"), ("final", "-.")]),
            ("reduction_peak_voltage", "royalblue", "Red", [("initial", ":"), ("average", "--"), ("final", "-.")]),
        )
        ordered = sorted(
            usable,
            key=lambda r: (
                float(r.get("scan_number", np.nan)) if np.isfinite(r.get("scan_number", np.nan)) else np.inf,
                str(r.get("file_name", "")),
            ),
        )
        for metric_key, color, short_label, stat_specs in reference_specs:
            vals = np.asarray(
                [r.get(metric_key, np.nan) for r in ordered],
                dtype=float,
            )
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            stat_values = {
                "initial": float(vals[0]),
                "average": float(np.mean(vals)),
                "final": float(vals[-1]),
            }
            for stat_name, linestyle in stat_specs:
                ax.axvline(
                    stat_values[stat_name],
                    color=color,
                    lw=1.1 if stat_name != "average" else 1.4,
                    linestyle=linestyle,
                    alpha=0.75,
                    label=f"{short_label} {stat_name}",
                )

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.02).set_label("Time order (earliest -> latest)")
    ax.set_title(title or "CV cycles")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(ylabel)
    ax.grid(False)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_cv_trace(result: dict) -> Optional[plt.Figure]:
    if result.get("voltage") is None or result.get("raw_current") is None:
        return None

    v = np.asarray(result["voltage"], dtype=float)
    raw = np.asarray(result["raw_current"], dtype=float)
    smooth = np.asarray(result.get("smoothed_current"), dtype=float) if result.get("smoothed_current") is not None else None
    baseline = np.asarray(result.get("baseline_current"), dtype=float) if result.get("baseline_current") is not None else None
    detrended = np.asarray(result.get("detrended_current"), dtype=float) if result.get("detrended_current") is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=False)

    axes[0].plot(v, raw, color="steelblue", lw=1.2)
    axes[0].set_title("Raw cycle")

    if smooth is not None:
        axes[1].plot(v, smooth, color="darkorange", lw=1.2, label="smoothed")
    if baseline is not None:
        axes[1].plot(v, baseline, color="gray", lw=1.0, linestyle="--", label="baseline")
    axes[1].set_title("Smoothed + baseline")
    if smooth is not None or baseline is not None:
        axes[1].legend(fontsize=8)

    if detrended is not None:
        axes[2].axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.6)
        axes[2].plot(v, detrended, color="seagreen", lw=1.2)
        for idx_key, color, label, marker in (
            ("oxidation_peak_idx", "crimson", "oxidation peak", "^"),
            ("reduction_peak_idx", "royalblue", "reduction peak", "v"),
        ):
            idx = result.get(idx_key)
            if idx is not None and 0 <= int(idx) < len(v):
                axes[2].scatter(
                    v[int(idx)],
                    detrended[int(idx)],
                    color=color,
                    s=46,
                    marker=marker,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=5,
                    label=label,
                )
        axes[2].legend(fontsize=8)
    axes[2].set_title("Detrended peaks")

    for ax in axes:
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (uA)")
        ax.grid(False)

    fig.suptitle(result.get("file_name", ""), fontsize=9, y=1.02)
    fig.tight_layout()
    return fig
