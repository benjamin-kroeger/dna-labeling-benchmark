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

from ...eval.transcript_classification import TranscriptMatchClass
from ..config import DEFAULT_FIG_SIZE, PlotMetadata
from ..utils import (
    _save_figure,
    _add_pictogram_panel,
    severity_palette,
    spezi_palette,
    text_color_for_bg,
)

logger = logging.getLogger(__name__)

# Enum declaration order is the severity order (EXACT → … → MISSED), so a
# green→red ramp makes the stacked bar read as a quality gradient.
_MATCH_CLASS_ORDER = [c.value for c in TranscriptMatchClass]
MATCH_CLASS_COLORS: dict[str, tuple[float, float, float]] = dict(
    zip(_MATCH_CLASS_ORDER, severity_palette(len(_MATCH_CLASS_ORDER)))
)

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
    bar_colors = [MATCH_CLASS_COLORS.get(c, "#888888") for c in norm_pivot.columns]
    norm_pivot.plot(kind="bar", stacked=True, ax=ax, color=bar_colors)

    # Annotate each bar section with its raw count, in a colour that stays
    # legible against the section's (green→red) background.
    for container, col_name in zip(ax.containers, norm_pivot.columns):
        section_color = MATCH_CLASS_COLORS.get(col_name, (0.5, 0.5, 0.5))
        label_color = text_color_for_bg(section_color)
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
                color=label_color,
            )

    # Per-method N above each bar — fractions are uninterpretable without the
    # denominator, which differs across methods.
    method_totals = raw_pivot.sum(axis=1)
    for bar_idx, method in enumerate(norm_pivot.index):
        ax.text(
            bar_idx,
            1.01,
            f"n={int(method_totals.loc[method])}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax.set_ylim(0, 1.08)

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
# Boundary shift distribution (per-boundary offset distributions)
# ---------------------------------------------------------------------------


def plot_boundary_shift_distribution(
    df_sc: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: Optional[PlotMetadata] = None,
) -> Optional[plt.Figure]:
    """Per-boundary offset distributions for topology-correct transcripts.

    Consumes the ``boundary_shift_offsets`` records (one per *shifted*
    boundary, see
    :func:`~dna_segmentation_benchmark.eval.chain_comparison._measure_shifted_boundaries`)
    pooled across every transcript whose predicted exon count matches the
    ground truth.  Each record carries a signed array-coordinate ``offset``
    (positive = predicted edge shifted to the right / array-3') and a
    ``position`` tag (``internal`` splice junction vs ``terminal`` TSS/TES).

    The view is deliberately complementary to the boundary-precision
    *landscape* (``eval/boundary_precision.py``): the landscape pools every
    overlapping section pair globally, whereas this figure is **conditioned on
    correct chain topology** and resolves each individual junction, so it
    answers "given the model got the exon count right, how precisely are the
    junctions placed, and is precision worse at the fuzzy transcript ends?".

    Panels
    ------
    Left   : ECDF of ``|offset|`` per method (log x).  Reads directly as
             "fraction of misplaced junctions within *k* bp"; per-method
             headline stats (median ``|offset|`` and % within ±2 bp) are
             annotated.
    Middle : density histogram of the **signed** offset over a robust window,
             exposing the ±1/±2 bp spike and any directional (5'/3') bias.
    Right  : ``|offset|`` split by internal vs terminal boundary (log y),
             separating precise splice junctions from inherently fuzzy
             transcript ends.

    Parameters
    ----------
    df_sc : pd.DataFrame
        Long-format DataFrame filtered to STRUCTURAL_COHERENCE rows.  Rows with
        ``metric_key == "boundary_shift_offsets"`` carry the pooled record list.
    class_name : str
        Human-readable class name.
    save_path, metadata : optional
        Forwarded to :func:`_save_figure` and :func:`_add_pictogram_panel`.

    Returns
    -------
    Figure | None
        ``None`` when no shifted boundaries were recorded for any method.
    """
    # Pool per-boundary records per method (records already flattened upstream).
    records: list[dict] = []
    for _, row in df_sc.iterrows():
        if row["metric_key"] != "boundary_shift_offsets":
            continue
        value = row["value"]
        if not isinstance(value, list):
            continue
        for rec in value:
            offset = rec["offset"]
            records.append(
                {
                    "method": row["method_name"],
                    "offset": offset,
                    "abs_offset": abs(offset),
                    "position": rec["position"],
                }
            )

    if not records:
        return None

    df = pd.DataFrame(records)

    methods = sorted(df["method"].unique())
    palette = dict(zip(methods, spezi_palette(len(methods))))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)

    # --- Panel 1 — ECDF of |offset| with headline stats ---------------------
    sns.ecdfplot(
        data=df,
        x="abs_offset",
        hue="method",
        hue_order=methods,
        palette=palette,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_xscale("log")
    axes[0].axvline(2, color="grey", linestyle=":", linewidth=1)
    axes[0].set_xlabel("|boundary offset| (bp, log scale)", labelpad=8)
    axes[0].set_ylabel("Cumulative fraction of shifted boundaries")
    axes[0].set_title("Offset ECDF — junction placement precision", pad=10)
    # Per-method headline stats, colour-matched, stacked in the empty corner.
    for idx, method in enumerate(methods):
        abs_off = df.loc[df["method"] == method, "abs_offset"]
        within_2 = (abs_off <= 2).mean() * 100
        axes[0].text(
            0.98,
            0.02 + idx * 0.06,
            f"{method}: median {abs_off.median():.0f} bp · {within_2:.0f}% ≤2 bp",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=palette[method],
        )

    # --- Panel 2 — signed offset density over a robust window ---------------
    window = int(np.ceil(df["abs_offset"].quantile(0.98)))
    window = max(5, min(window, 60))
    bins_signed = np.arange(-window - 0.5, window + 1.5, 1.0)
    sns.histplot(
        data=df,
        x="offset",
        hue="method",
        hue_order=methods,
        palette=palette,
        bins=bins_signed,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.35,
        ax=axes[1],
    )
    axes[1].axvline(0, color="grey", linestyle="--", linewidth=1)
    axes[1].set_xlim(-window - 0.5, window + 0.5)
    axes[1].set_xlabel("Signed boundary offset (bp, pred − GT, array-3' positive)", labelpad=8)
    axes[1].set_ylabel("Density")
    axes[1].set_title("Signed Offset — directional bias & small-shift spike", pad=10)
    frac_clipped = (df["abs_offset"] > window).mean()
    if frac_clipped > 0:
        axes[1].text(
            0.99,
            0.97,
            f"{frac_clipped * 100:.1f}% beyond ±{window} bp (clipped)",
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            style="italic",
            color="#555555",
        )

    # --- Panel 3 — internal vs terminal |offset| ----------------------------
    positions = [p for p in ("internal", "terminal") if p in set(df["position"])]
    sns.boxplot(
        data=df,
        x="position",
        y="abs_offset",
        hue="method",
        order=positions,
        hue_order=methods,
        palette=palette,
        fliersize=2,
        ax=axes[2],
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Boundary type", labelpad=8)
    axes[2].set_ylabel("|boundary offset| (bp, log scale)")
    axes[2].set_title("Internal splice vs terminal (TSS/TES) precision", pad=10)
    axes[2].legend(title="Method", fontsize=8)

    fig.suptitle(
        f"{class_name} — Boundary Shift Distribution  "
        f"(topology-correct transcripts · n={len(df)} shifted boundaries)",
        fontsize=13,
    )

    _add_pictogram_panel(fig, metadata, logger)

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
