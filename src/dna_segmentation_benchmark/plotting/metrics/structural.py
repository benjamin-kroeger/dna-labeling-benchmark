"""Plotting functions for STRUCTURAL_COHERENCE metrics.

Provides visualisations for intron-chain precision/recall, per-transcript
exon recovery (recall + precision + false-exon count), transcript match
classification, segment count delta, and boundary shift distributions.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

# The transcript classes with *correct topology* (equal segment count + every
# positional pair overlapping) — exactly the set gated in
# ``transcript_classification._classify_segment_match``. These are the only
# transcripts that contribute to the boundary-shift distribution, so they form
# its denominator / eligibility population.
TOPOLOGY_CORRECT_CLASSES: frozenset[str] = frozenset(
    {
        TranscriptMatchClass.EXACT.value,
        TranscriptMatchClass.BOUNDARY_SHIFT_INTERNAL.value,
        TranscriptMatchClass.BOUNDARY_SHIFT_TERMINAL.value,
    }
)

# ---------------------------------------------------------------------------
# Transcript match classification stacked bar
# ---------------------------------------------------------------------------


def _transcript_match_pivot(df_sc: pd.DataFrame) -> pd.DataFrame | None:
    """Per-method × match-class raw count pivot from ``df_sc``.

    Shared by the standalone stacked bar and the boundary-shift figure, which
    needs the denominator for :func:`_topology_eligibility` without drawing the
    bar.  Returns ``None`` when *df_sc* carries no
    ``transcript_match_distribution`` rows.
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

    return plot_df.pivot_table(
        index="method_name",
        columns="match_class",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )


def _draw_transcript_match_bar(ax: plt.Axes, df_sc: pd.DataFrame) -> pd.DataFrame | None:
    """Draw the transcript-match severity stacked bar onto *ax*.

    Renders one stacked bar per method (green→red severity gradient, raw count
    annotated per section, per-method ``n=`` above the bar) and returns the
    per-method × class raw count pivot so callers can derive denominators /
    eligibility.  Title, axis labels and legend are left to the caller.

    Returns ``None`` (without drawing) when *df_sc* carries no
    ``transcript_match_distribution`` rows.
    """
    raw_pivot = _transcript_match_pivot(df_sc)
    if raw_pivot is None:
        return None

    # Normalise to fractions for the stacked bar
    norm_pivot = raw_pivot.div(raw_pivot.sum(axis=1), axis=0)

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
    return raw_pivot


def _topology_eligibility(raw_pivot: pd.DataFrame) -> dict[str, dict]:
    """Per-method topology-correct (eligible) share of GT transcripts.

    The boundary-shift panels are conditioned on the topology-correct set
    (:data:`TOPOLOGY_CORRECT_CLASSES`), so a low fraction means the offset
    distribution describes only a small, possibly non-representative slice of
    the transcripts.  Returns ``{method: {"frac", "n_correct", "n_total"}}``.
    """
    totals = raw_pivot.sum(axis=1)
    present = [c for c in TOPOLOGY_CORRECT_CLASSES if c in raw_pivot.columns]
    correct = (
        raw_pivot[present].sum(axis=1)
        if present
        else pd.Series(0, index=raw_pivot.index)
    )
    eligibility: dict[str, dict] = {}
    for method in raw_pivot.index:
        total = int(totals.loc[method])
        n_correct = int(correct.loc[method])
        eligibility[method] = {
            "frac": n_correct / total if total else 0.0,
            "n_correct": n_correct,
            "n_total": total,
        }
    return eligibility


def plot_transcript_match_distribution(
    df_sc: pd.DataFrame,
    class_name: str,
    save_path: Path | None = None,
    metadata: PlotMetadata | None = None,
) -> plt.Figure | None:
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
    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    raw_pivot = _draw_transcript_match_bar(ax, df_sc)
    if raw_pivot is None:
        plt.close(fig)
        return None

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
    save_path: Path | None = None,
    metadata: PlotMetadata | None = None,
) -> plt.Figure | None:
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


def _short_method_labels(methods: list[str]) -> dict[str, str]:
    """Map each method name to a display label with the shared prefix stripped.

    Method names typically share a long ``_``-delimited prefix (e.g. the
    species). Dropping the longest common prefix keeps in-panel legends and
    captions readable. Falls back to the full name if stripping would empty it
    or there is nothing in common.
    """
    if len(methods) < 2:
        return {m: m for m in methods}
    parts = [m.split("_") for m in methods]
    common = 0
    for tokens in zip(*parts):
        if len(set(tokens)) == 1:
            common += 1
        else:
            break
    return {m: ("_".join(p[common:]) or m) for m, p in zip(methods, parts)}


def plot_boundary_shift_distribution(
    df_sc: pd.DataFrame,
    class_name: str,
    save_path: Path | None = None,
    metadata: PlotMetadata | None = None,
) -> plt.Figure | None:
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
    Left      : ECDF of ``|offset|`` per method (log x).  Reads as "fraction of
                *shifted* junctions within *k* bp" (exact/offset-0 junctions are
                excluded upstream); per-method headline stats annotated.
    Centre    : density histogram of the **signed** offset over a robust window,
                exposing the ±1/±2 bp spike and any directional (5'/3') bias.
    Right     : ``|offset|`` split by internal vs terminal boundary (log y),
                separating precise splice junctions from inherently fuzzy
                transcript ends.  Only topology-correct transcripts
                (``exact`` / ``boundary_shift_*``) feed the offsets, so the
                per-method eligibility caption overlaid here keeps that
                conditioning in view; a tight box on a tiny eligible set must not
                be read as "good".  The transcript-match classification itself is
                the standalone ``transcript_match`` figure.

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
    # Method names often share a long prefix (e.g. the species), which makes the
    # in-panel captions and legends overflow and collide once the pictogram
    # panel squeezes the axes. Strip the shared '_'-delimited prefix for display.
    short = _short_method_labels(methods)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 6),
        constrained_layout=True,
    )

    # Eligibility denominator for the conditioning caption on the box panel.
    # The transcript-match classification lives in its standalone figure; here
    # we only need the per-method topology-correct share so a tight offset box
    # on a tiny eligible set cannot be misread as "good".
    raw_pivot = _transcript_match_pivot(df_sc)
    eligibility = _topology_eligibility(raw_pivot) if raw_pivot is not None else {}

    # --- Panel 0 — ECDF of |offset| with headline stats ---------------------
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
    # Exact (offset-0) junctions are excluded upstream, so the % is over the
    # *shifted* boundaries only — not overall junction accuracy.
    for idx, method in enumerate(methods):
        abs_off = df.loc[df["method"] == method, "abs_offset"]
        within_2 = (abs_off <= 2).mean() * 100
        axes[0].text(
            0.98,
            0.02 + idx * 0.06,
            f"{short[method]}: median {abs_off.median():.0f} bp · {within_2:.0f}% of shifted ≤2 bp",
            transform=axes[0].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=palette[method],
        )

    # --- Panel 1 — signed offset density over a robust window ---------------
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
    # Signed-offset histogram peaks at the centre and tapers at the edges, so
    # the upper-left corner is empty — park the (shortened) legend there clear
    # of the bars and the clipped-fraction note pinned at upper-right.
    if axes[1].get_legend() is not None:
        sns.move_legend(axes[1], "upper left", title="Method", fontsize=8)
        for txt in axes[1].get_legend().get_texts():
            txt.set_text(short.get(txt.get_text(), txt.get_text()))
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

    # --- Panel 2 — internal vs terminal |offset| ----------------------------
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
    # Drop the boxplot's auto legend: it collides with the eligibility captions
    # below, which already colour-code each method and carry more information.
    if axes[2].get_legend() is not None:
        axes[2].get_legend().remove()
    # Per-method eligibility caption — the share of GT transcripts these boxes
    # are conditioned on. A low fraction (flagged ⚠) means the boxes summarise
    # only a small slice, so precise-looking boxes may not reflect real skill.
    for idx, method in enumerate(methods):
        elig = eligibility.get(method)
        if elig is None:
            continue
        low = elig["frac"] < 0.25
        axes[2].text(
            0.02,
            0.98 - idx * 0.06,
            f"{'⚠ ' if low else ''}{short[method]}: eligible {elig['frac']:.0%} "
            f"({elig['n_correct']}/{elig['n_total']} tx)",
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=palette[method],
            fontweight="bold" if low else "normal",
        )

    fig.suptitle(
        f"{class_name} — Boundary Shift Distribution "
        f"(topology-correct transcripts · n={len(df)} shifted boundaries)",
        fontsize=13,
    )

    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig


# ---------------------------------------------------------------------------
# Per-transcript exon recovery (recall + precision distributions, false count)
# ---------------------------------------------------------------------------


def plot_per_transcript_exon_recovery(
    df_sc: pd.DataFrame,
    class_name: str,
    save_path: Path | None = None,
    metadata: PlotMetadata | None = None,
) -> plt.Figure | None:
    """Three-panel histogram of per-transcript exon-recovery metrics.

    Complements the strict all-or-nothing ``exon_chain`` tiers with a
    distribution view: "how many transcripts got 90% of their exons
    right" vs "how many got none". Recall and precision are the two
    symmetric fraction axes; the false-exon panel exposes the absolute
    spurious-exon burden the binary tiers collapse into a single FP.

    Panels
    ------
    Left   : per-transcript **exon recall** — fraction of GT exons whose
             ``(start, end)`` was recovered exactly. Continuous in [0, 1].
    Middle : per-transcript **exon precision** — fraction of predicted exons
             whose ``(start, end)`` is an exact GT match. Continuous in [0, 1].
    Right  : per-transcript **false exon count** — predicted exons whose
             ``(start, end)`` is not present in GT. Integer ≥ 0; bins of width 1.

    Parameters
    ----------
    df_sc : pd.DataFrame
        Long-format DataFrame filtered to STRUCTURAL_COHERENCE rows.
        The rows with ``metric_key`` in
        ``{"exon_recall_per_transcript", "exon_precision_per_transcript",
        "false_exon_count_per_transcript"}`` must carry raw
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
    precision_rows: list[dict] = []
    false_exon_rows: list[dict] = []
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
        elif key == "exon_precision_per_transcript":
            for v in val:
                if v is None:
                    continue
                precision_rows.append({"method": method, "value": float(v)})
        elif key == "false_exon_count_per_transcript":
            for v in val:
                if v is None:
                    continue
                false_exon_rows.append({"method": method, "value": int(v)})

    if not recall_rows and not precision_rows and not false_exon_rows:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    # Panels 1 & 2 — per-transcript exon recall / precision distributions
    fraction_bins = np.linspace(0.0, 1.0, 21)
    for ax, rows, xlabel, title in (
        (axes[0], recall_rows, "Fraction of GT exons exactly recovered", "Per-transcript Exon Recall"),
        (axes[1], precision_rows, "Fraction of predicted exons that are exact GT matches", "Per-transcript Exon Precision"),
    ):
        if rows:
            sns.histplot(
                data=pd.DataFrame(rows),
                x="value",
                hue="method",
                bins=fraction_bins,
                multiple="layer",
                element="step",
                alpha=0.4,
                ax=ax,
            )
            ax.set_xlim(0.0, 1.0)
        else:
            ax.set_visible(False)
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel("Transcripts")
        ax.set_title(title, pad=10)

    # Panel 3 — per-transcript false exon count distribution
    if false_exon_rows:
        df_false = pd.DataFrame(false_exon_rows)
        max_count = int(df_false["value"].max())
        # Discrete integer bins [0, 1, 2, ..., max+1]
        false_bins = np.arange(-0.5, max_count + 1.5, 1)
        sns.histplot(
            data=df_false,
            x="value",
            hue="method",
            bins=false_bins,
            multiple="layer",
            element="step",
            alpha=0.4,
            ax=axes[2],
        )
        axes[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        axes[2].set_visible(False)
    axes[2].set_xlabel("False exons per transcript", labelpad=8)
    axes[2].set_ylabel("Transcripts")
    axes[2].set_title("Per-transcript False Exon Count", pad=10)

    fig.suptitle(
        f"{class_name} — Per-transcript Exon Recovery",
        fontsize=13,
    )

    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig
