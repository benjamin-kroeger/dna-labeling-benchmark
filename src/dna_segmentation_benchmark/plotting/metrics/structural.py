"""Plotting functions for STRUCTURAL_COHERENCE metrics.

Provides visualisations for intron-chain precision/recall, per-transcript
soft exon metrics (recall + hallucinations), transcript match
classification, segment count delta, and boundary shift distributions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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

    Each bar section is annotated with its raw count.

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
                rows.append(
                    {
                        "method_name": row["method_name"],
                        "match_class": match_class,
                        "count": count,
                    }
                )

    if not rows:
        return None

    plot_df = pd.DataFrame(rows)

    raw_pivot = plot_df.pivot_table(
        index="method_name",
        columns="match_class",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    # Normalise to fractions for the stacked bar
    norm_pivot = raw_pivot.div(raw_pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    norm_pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")

    # Annotate each bar section with its raw count
    for container, col_name in zip(ax.containers, norm_pivot.columns):
        for bar_idx, patch in enumerate(container.patches):
            height = patch.get_height()
            if height < 0.02:
                continue
            method = norm_pivot.index[bar_idx]
            raw_count = (
                int(raw_pivot.at[method, col_name])
                if (method in raw_pivot.index and col_name in raw_pivot.columns)
                else 0
            )
            if raw_count == 0:
                continue
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_y() + height / 2
            ax.text(
                x,
                y,
                str(raw_count),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
            )

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
            rows.append(
                {
                    "method_name": row["method_name"],
                    "mean_delta": row["value"].get("mean", 0.0),
                    "std": row["value"].get("std", 0.0),
                }
            )

    if not rows:
        return None

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    colors = ["#e74c3c" if d > 0 else "#3498db" if d < 0 else "#95a5a6" for d in plot_df["mean_delta"]]
    ax.bar(
        plot_df["method_name"],
        plot_df["mean_delta"],
        yerr=plot_df["std"],
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
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
# Boundary shift distribution (histograms + scatter)
# ---------------------------------------------------------------------------


def plot_boundary_shift_distribution(
    df_sc: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: Optional[PlotMetadata] = None,
) -> Optional[plt.Figure]:
    """Three-panel distribution figure for BOUNDARY_SHIFT transcripts.

    Only transcripts classified as BOUNDARY_SHIFT (non-zero count) are
    included.

    Panels
    ------
    Left   : histogram of shifted boundary count per transcript.
    Middle : histogram of total bp offset per transcript.
    Right  : scatter of count vs total bp offset.

    Parameters
    ----------
    df_sc : pd.DataFrame
        Long-format DataFrame filtered to STRUCTURAL_COHERENCE rows.
    class_name : str
        Human-readable class name.
    save_path, metadata : optional
        Forwarded to :func:`_save_figure` and :func:`_add_pictogram_panel`.

    Returns
    -------
    Figure | None
    """
    # Collect per-method lists keyed by boundary_shift_count / boundary_shift_total
    raw: dict[str, dict[str, list]] = {}
    for _, row in df_sc.iterrows():
        key = row["metric_key"]
        val = row["value"]
        method = row["method_name"]
        if key not in ("boundary_shift_count", "boundary_shift_total"):
            continue
        if not isinstance(val, list):
            continue
        raw.setdefault(method, {})[key] = val

    if not raw:
        return None

    # Filter to BOUNDARY_SHIFT transcripts only (count > 0)
    filtered: dict[str, dict[str, list]] = {}
    for method, data in raw.items():
        counts = data.get("boundary_shift_count", [])
        totals = data.get("boundary_shift_total", [])
        if not counts:
            continue
        pairs = [(c, t) for c, t in zip(counts, totals) if c > 0]
        if pairs:
            filtered[method] = {
                "count": [c for c, _ in pairs],
                "total": [t for _, t in pairs],
            }

    if not filtered:
        return None

    # Build long-format rows
    hist_rows: list[dict] = []
    scatter_rows: list[dict] = []
    for method, data in filtered.items():
        for c, t in zip(data["count"], data["total"]):
            hist_rows.append({"method": method, "count": c, "total": t})
            scatter_rows.append({"method": method, "count": c, "total": t})

    df_hist = pd.DataFrame(hist_rows)
    df_scatter = pd.DataFrame(scatter_rows)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

    # Panel 1 — count histogram (discrete bins, overlayed)
    max_count = int(df_hist["count"].max())
    bins_count = np.arange(0.5, max_count + 1.5, 1)
    sns.histplot(
        data=df_hist,
        x="count",
        hue="method",
        bins=bins_count,
        multiple="layer",
        element="step",
        alpha=0.4,
        ax=axes[0],
    )
    axes[0].set_xlabel("Shifted boundary positions per transcript", labelpad=8)
    axes[0].set_ylabel("Transcripts")
    axes[0].set_title("Shifted Boundary Count", pad=10)
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Panel 2 — total bp offset histogram (overlayed)
    max_total = float(df_hist["total"].max())
    bins_total = np.linspace(0, max_total, 31)
    sns.histplot(
        data=df_hist,
        x="total",
        hue="method",
        bins=bins_total,
        multiple="layer",
        element="step",
        alpha=0.4,
        ax=axes[1],
    )
    axes[1].set_xlabel("Total bp offset across shifted boundaries", labelpad=8)
    axes[1].set_ylabel("Transcripts")
    axes[1].set_title("Total Boundary Shift (bp)", pad=10)

    # Panel 3 — scatter: count vs total (log y, integer x ticks at unit interval)
    sns.scatterplot(
        data=df_scatter,
        x="count",
        y="total",
        hue="method",
        alpha=0.65,
        s=40,
        ax=axes[2],
    )
    axes[2].set_xlabel("Shifted boundary count", labelpad=8)
    axes[2].set_ylabel("Total bp offset (log scale)")
    axes[2].set_title("Count vs Total bp Offset", pad=10)
    axes[2].set_yscale("log")
    max_scatter_count = int(df_scatter["count"].max())
    axes[2].set_xticks(np.arange(1, max_scatter_count + 1, 1))

    fig.suptitle(
        f"{class_name} — Boundary Shift Distribution  (BOUNDARY_SHIFT transcripts only)",
        fontsize=13,
    )

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig


# ---------------------------------------------------------------------------
# Per-transcript soft exon metrics (recall distribution + hallucination count)
# ---------------------------------------------------------------------------


def plot_per_transcript_soft_exon_metrics(
    df_sc: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: Optional[PlotMetadata] = None,
) -> Optional[plt.Figure]:
    """Two-panel histogram of per-transcript soft structural metrics.

    Complements the strict all-or-nothing ``intron_chain`` metric with a
    distribution view: "how many transcripts got 90% of their exons
    right" vs "how many got none". The hallucination panel exposes the
    precision side — how many spurious predicted exons the model emits
    per transcript — without conflating it with boundary errors.

    Panels
    ------
    Left  : histogram of per-transcript **exon recall** — the fraction
            of GT exons whose ``(start, end)`` was recovered exactly.
            Continuous in [0, 1]; bins of width 0.05.
    Right : histogram of per-transcript **hallucinated exon count** —
            predicted exons whose ``(start, end)`` is not present in GT.
            Integer ≥ 0; discrete bins of width 1.

    Parameters
    ----------
    df_sc : pd.DataFrame
        Long-format DataFrame filtered to STRUCTURAL_COHERENCE rows.
        The rows with ``metric_key`` in
        ``{"exon_recall_per_transcript",
        "hallucinated_exon_count_per_transcript"}`` must carry raw
        per-sequence value lists.
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
    recall_rows: list[dict] = []
    hallucination_rows: list[dict] = []
    for _, row in df_sc.iterrows():
        key = row["metric_key"]
        val = row["value"]
        method = row["method_name"]
        if not isinstance(val, list):
            continue
        if key == "exon_recall_per_transcript":
            for v in val:
                if v is None:
                    continue
                recall_rows.append({"method": method, "value": float(v)})
        elif key == "hallucinated_exon_count_per_transcript":
            for v in val:
                if v is None:
                    continue
                hallucination_rows.append({"method": method, "value": int(v)})

    if not recall_rows and not hallucination_rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Panel 1 — per-transcript exon recall distribution
    if recall_rows:
        df_recall = pd.DataFrame(recall_rows)
        recall_bins = np.linspace(0.0, 1.0, 21)
        sns.histplot(
            data=df_recall,
            x="value",
            hue="method",
            bins=recall_bins,
            multiple="layer",
            element="step",
            alpha=0.4,
            ax=axes[0],
        )
        axes[0].set_xlim(0.0, 1.0)
    else:
        axes[0].set_visible(False)
    axes[0].set_xlabel("Fraction of GT exons exactly recovered", labelpad=8)
    axes[0].set_ylabel("Transcripts")
    axes[0].set_title("Per-transcript Exon Recall", pad=10)

    # Panel 2 — per-transcript hallucinated exon count distribution
    if hallucination_rows:
        df_hallu = pd.DataFrame(hallucination_rows)
        max_count = int(df_hallu["value"].max())
        # Discrete integer bins [0, 1, 2, ..., max+1]
        hallu_bins = np.arange(-0.5, max_count + 1.5, 1)
        sns.histplot(
            data=df_hallu,
            x="value",
            hue="method",
            bins=hallu_bins,
            multiple="layer",
            element="step",
            alpha=0.4,
            ax=axes[1],
        )
        axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        axes[1].set_visible(False)
    axes[1].set_xlabel("Hallucinated exons per transcript", labelpad=8)
    axes[1].set_ylabel("Transcripts")
    axes[1].set_title("Per-transcript Hallucinated Exon Count", pad=10)

    fig.suptitle(
        f"{class_name} — Per-transcript Soft Exon Metrics",
        fontsize=13,
    )

    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig


# ---------------------------------------------------------------------------
# Gap chain metrics grouped bar (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def plot_intron_chain_metrics(
    df_sc: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: Optional[PlotMetadata] = None,
) -> Optional[plt.Figure]:
    """Grouped bar chart of intron chain precision and recall per method.

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

        if key == "intron_precision" and isinstance(val, dict):
            rows.append({"method_name": method, "metric": "intron_precision", "value": val.get("mean", 0.0)})
        elif key == "intron_recall" and isinstance(val, dict):
            rows.append({"method_name": method, "metric": "intron_recall", "value": val.get("mean", 0.0)})

    if not rows:
        return None

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    sns.barplot(data=plot_df, x="metric", y="value", hue="method_name", ax=ax)
    ax.set_title(f"{class_name} — Intron Chain Metrics")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Rate / Ratio")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Method", loc="lower right", fontsize=9)
    fig.tight_layout()

    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig
