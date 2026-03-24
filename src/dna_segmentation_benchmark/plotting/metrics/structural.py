"""Plotting functions for STRUCTURAL_COHERENCE metrics.

Provides visualisations for gap chain analysis, transcript match
classification, and segment count delta results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import DEFAULT_FIG_SIZE, PlotMetadata
from ..utils import _save_figure, _add_pictogram_panel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transcript match classification stacked bar
# ---------------------------------------------------------------------------


def plot_transcript_match_distribution(
    df_sc: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: Optional[PlotMetadata] = None,
) -> Optional[plt.Figure]:
    """Stacked bar chart of transcript match class distribution per method.

    Parameters
    ----------
    df_sc : pd.DataFrame
        Long-format DataFrame filtered to STRUCTURAL_COHERENCE rows.
    class_name : str
        Human-readable class name.
    save_path : Path | None
        If provided, the figure is saved to this path.
    metadata : PlotMetadata | None
        If provided, a pictogram panel is added to the figure.

    Returns
    -------
    Figure | None
    """
    rows = []
    for _, row in df_sc.iterrows():
        if row["metric_key"] == "transcript_match_distribution" and isinstance(row["value"], dict):
            for match_class, count in row["value"].items():
                rows.append({
                    "method_name": row["method_name"],
                    "match_class": match_class,
                    "count": count,
                })

    if not rows:
        return None

    plot_df = pd.DataFrame(rows)

    # Pivot for stacked bar
    pivot = plot_df.pivot_table(
        index="method_name", columns="match_class",
        values="count", fill_value=0, aggfunc="sum",
    )

    # Normalise to fractions
    pivot = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_title(f"{class_name} — Transcript Match Classification")
    ax.set_xlabel("Method")
    ax.set_ylabel("Fraction")
    ax.legend(title="Match Class", loc="center left", fontsize=8)
    fig.tight_layout()

    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig


# ---------------------------------------------------------------------------
# Segment count delta bar chart
# ---------------------------------------------------------------------------


def plot_segment_count_delta(
    df_sc: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: Optional[PlotMetadata] = None,
) -> Optional[plt.Figure]:
    """Bar chart of mean segment count delta per method.

    Positive values indicate over-segmentation, negative values indicate
    under-segmentation.

    Parameters
    ----------
    df_sc : pd.DataFrame
        Long-format DataFrame filtered to STRUCTURAL_COHERENCE rows.
    class_name : str
        Human-readable class name.
    save_path : Path | None
        If provided, the figure is saved to this path.
    metadata : PlotMetadata | None
        If provided, a pictogram panel is added to the figure.

    Returns
    -------
    Figure | None
    """
    rows = []
    for _, row in df_sc.iterrows():
        if row["metric_key"] == "segment_count_delta" and isinstance(row["value"], dict):
            rows.append({
                "method_name": row["method_name"],
                "mean_delta": row["value"].get("mean", 0.0),
                "std": row["value"].get("std", 0.0),
            })

    if not rows:
        return None

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    colors = ["#e74c3c" if d > 0 else "#3498db" if d < 0 else "#95a5a6"
              for d in plot_df["mean_delta"]]
    ax.bar(plot_df["method_name"], plot_df["mean_delta"],
           yerr=plot_df["std"], capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
    ax.set_title(f"{class_name} — Segment Count Delta (pred \u2212 GT)")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean segment count delta")
    fig.tight_layout()

    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig


# ---------------------------------------------------------------------------
# Gap chain metrics grouped bar
# ---------------------------------------------------------------------------


def plot_gap_chain_metrics(
    df_sc: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: Optional[PlotMetadata] = None,
) -> Optional[plt.Figure]:
    """Grouped bar chart of gap chain match rate, gap count match rate, and
    mean gap chain LCS ratio per method.

    Parameters
    ----------
    df_sc : pd.DataFrame
        Long-format DataFrame filtered to STRUCTURAL_COHERENCE rows.
    class_name : str
        Human-readable class name.
    save_path : Path | None
        If provided, the figure is saved to this path.
    metadata : PlotMetadata | None
        If provided, a pictogram panel is added to the figure.

    Returns
    -------
    Figure | None
    """
    rows = []

    for _, row in df_sc.iterrows():
        method = row["method_name"]
        key = row["metric_key"]
        val = row["value"]

        if key == "gap_chain_match_rate" and isinstance(val, (int, float)):
            rows.append({"method_name": method, "metric": "gap_chain_match_rate", "value": float(val)})
        elif key == "gap_count_match_rate" and isinstance(val, (int, float)):
            rows.append({"method_name": method, "metric": "gap_count_match_rate", "value": float(val)})
        elif key == "gap_chain_lcs_ratio" and isinstance(val, dict):
            rows.append({"method_name": method, "metric": "gap_chain_lcs_ratio_mean", "value": val.get("mean", 0.0)})

    if not rows:
        return None

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    sns.barplot(data=plot_df, x="metric", y="value", hue="method_name", ax=ax)
    ax.set_title(f"{class_name} — Gap Chain Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Rate / Ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Method", loc="lower right", fontsize=9)
    fig.tight_layout()

    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig
