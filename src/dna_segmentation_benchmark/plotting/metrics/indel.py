"""Plots for the boundary-typed INDEL metric.

Each method's INDEL payload is ``{"by_boundary": {exon_type: {event_type:
[run_length, ...]}}, "exon_opportunities": {exon_type: int},
"n_gt_segments": int, "n_pred_segments": int}`` — run lengths keyed by the GT
exon position they touch, plus the per-exon-type *opportunity* counts needed to
turn counts into rates.  ``exon_type`` is one of ``five_prime_terminal_exon``,
``internal_exon``, ``three_prime_terminal_exon``, ``single_exon_gene``.

Views produced:

* :func:`plot_stacked_indel_counts_bar` — per-method event counts, summed over
  boundaries (the familiar high-level summary).
* :func:`plot_indel_rates_by_boundary` — per-method boundary × event-type
  **rate** heatmap (events ÷ opportunities). The comparable benchmarking view.
* :func:`plot_indel_counts_by_boundary` — per-method boundary × event-type count
  heatmap (log colour scale): the raw-magnitude view.
* :func:`plot_individual_error_lengths_histograms` — a boundary × event-type
  grid of overlaid per-method run-length distributions: *how large* the slips
  are at each junction.

Rows follow the canonical order from :data:`SEMANTIC_BOUNDARY_ORDER`.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

from ..config import PlotMetadata, DEFAULT_FIG_SIZE
from ..utils import _add_icon_to_ax, _save_figure, _add_pictogram_panel
from ...label_definition import SEMANTIC_BOUNDARY_ORDER

logger = logging.getLogger(__name__)

#: Canonical left-to-right ordering of the eight INDEL event types.
_EVENT_ORDER = (
    "5_prime_extensions",
    "3_prime_extensions",
    "joined",
    "whole_insertions",
    "5_prime_deletions",
    "3_prime_deletions",
    "split",
    "whole_deletions",
)

#: Events whose rate denominator is the count of GT exons of that position type
#: (anchored slips sit on / extend one exon; splits and whole deletions consume
#: one exon).  ``joined`` and ``whole_insertions`` normalise differently (intron
#: / gene counts) and are handled explicitly in :func:`_event_denominator`.
_EXON_OPPORTUNITY_EVENTS = frozenset(
    {
        "5_prime_extensions",
        "3_prime_extensions",
        "5_prime_deletions",
        "3_prime_deletions",
        "split",
        "whole_deletions",
    }
)

#: Events excluded from the *rate* heatmap.  A whole insertion is a detached false
#: positive with no GT exon under it, so it has no bounded GT opportunity: its
#: "rate" is unbounded and would break the shared colour scale.  It stays in the
#: count heatmap, where an absolute magnitude is meaningful.
_RATE_EXCLUDED_EVENTS = frozenset({"whole_insertions"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_events(all_indel_data: dict[str, dict]):
    """Yield ``(method, boundary, event_type, lengths)`` over every payload.

    ``all_indel_data`` maps method name → the full INDEL payload dict.
    ``lengths`` is either the raw run-length list (per-species results, still
    in memory before leaning) or an already-collapsed integer count (leaned /
    cross-species-averaged results — see ``_count_indel_events`` in
    ``thesis_code.common.runner``); use :func:`_event_count` to read either
    uniformly.
    """
    for method, payload in all_indel_data.items():
        if not isinstance(payload, dict):
            continue
        for boundary, buckets in payload.get("by_boundary", {}).items():
            for event_type, lengths in buckets.items():
                yield method, boundary, event_type, lengths


def _event_count(value: list | int | float) -> int:
    """Event count for one ``by_boundary`` leaf, whether raw list or pre-collapsed count."""
    return len(value) if isinstance(value, (list, tuple)) else int(value)


def _pretty_boundary(boundary: str) -> str:
    return boundary.replace("_", " ").title()


def _pretty_event(event_type: str) -> str:
    return event_type.replace("_", " ").title()


def _present_events(all_indel_data: dict[str, dict]) -> list[str]:
    """Event types (in canonical order) that have at least one run."""
    seen = {event for _, _, event, lengths in _iter_events(all_indel_data) if lengths}
    return [event for event in _EVENT_ORDER if event in seen]


def _present_boundaries(all_indel_data: dict[str, dict]) -> list[str]:
    """Semantic boundary categories that have at least one run, in canonical order."""
    seen = {boundary for _, boundary, _, lengths in _iter_events(all_indel_data) if lengths}
    return [b for b in SEMANTIC_BOUNDARY_ORDER if b in seen]


def _event_denominator(payload: dict, event_type: str, boundary: str) -> int:
    """Opportunity count for one (event, boundary) under exon-typing.

    All exon-keyed events (5'/3' extensions & deletions, splits, whole deletions)
    divide by the count of GT exons of that position type.  ``joined`` divides by
    the GT intron count (``n_gt_segments`` − ``n_genes``, derived from
    ``exon_opportunities``; each window is one transcript).  Any other event has
    no bounded GT opportunity and returns 0 — notably ``whole_insertions``, which
    the rate plot excludes entirely (:data:`_RATE_EXCLUDED_EVENTS`).
    """
    exon_opp = payload.get("exon_opportunities", {})
    if event_type in _EXON_OPPORTUNITY_EVENTS:
        return int(exon_opp.get(boundary, 0))
    if event_type == "joined":
        n_genes = int(exon_opp.get("five_prime_terminal_exon", 0)) + int(exon_opp.get("single_exon_gene", 0))
        return max(0, int(payload.get("n_gt_segments", 0)) - n_genes)  # n_introns
    return 0


def _count_matrix(payload: dict, boundaries: list[str], events: list[str]) -> np.ndarray:
    """Event-count matrix (len(boundaries) × len(events)) for one method."""
    by_boundary = payload.get("by_boundary", {})
    mat = np.zeros((len(boundaries), len(events)), dtype=float)
    for r, boundary in enumerate(boundaries):
        buckets = by_boundary.get(boundary, {})
        for c, event in enumerate(events):
            mat[r, c] = _event_count(buckets.get(event, 0))
    return mat


def _rate_matrix(payload: dict, boundaries: list[str], events: list[str]) -> np.ndarray:
    """Per-(boundary, event) rate matrix; ``nan`` where the opportunity is zero."""
    counts = _count_matrix(payload, boundaries, events)
    rates = np.full_like(counts, np.nan)
    for r, boundary in enumerate(boundaries):
        for c, event in enumerate(events):
            denom = _event_denominator(payload, event, boundary)
            if denom > 0:
                rates[r, c] = counts[r, c] / denom
    return rates


# ---------------------------------------------------------------------------
# 1. Per-method event-count summary (boundaries aggregated)
# ---------------------------------------------------------------------------


def plot_stacked_indel_counts_bar(
    all_indel_data: dict[str, dict],
    class_name: str,
    save_path: Path | None = None,
    metadata: PlotMetadata | None = None,
) -> plt.Figure | None:
    """Stacked horizontal bar of INDEL event counts per method (summed over boundaries)."""
    methods = list(all_indel_data)
    if not methods:
        logger.info("No INDEL data for class %s.", class_name)
        return None

    data = {method: dict.fromkeys(_EVENT_ORDER, 0) for method in methods}
    for method, _boundary, event_type, lengths in _iter_events(all_indel_data):
        if event_type in data[method]:
            data[method][event_type] += _event_count(lengths)

    counts = pd.DataFrame(data).T.reindex(columns=_EVENT_ORDER)
    counts = counts.loc[:, counts.sum(axis=0) > 0]
    if counts.empty or counts.to_numpy().sum() == 0:
        logger.info("No INDEL count data for class %s.", class_name)
        return None

    totals = counts.sum(axis=1)
    counts = counts.loc[totals.sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    counts.plot(kind="barh", stacked=True, ax=ax, colormap="viridis")

    max_val = totals.max()
    for i, (_idx, total) in enumerate(totals.sort_values(ascending=True).items()):
        ax.text(
            total + 0.01 * max(max_val, 1),
            i,
            str(int(total)),
            va="center",
            ha="left",
            fontweight="bold",
        )

    ax.set_xlim(0, max(max_val * 1.15, 1))
    ax.set_title(f"INDEL Counts by Method — {class_name}", fontsize=16)
    ax.set_xlabel("Total Number of INDELs", fontsize=12)
    ax.set_ylabel("Method Name", fontsize=12)
    ax.legend(title="INDEL Type", loc="lower right", fontsize=9)

    fig.tight_layout()
    _add_pictogram_panel(fig, metadata, logger=logger)

    if save_path is not None:
        _save_figure(fig, save_path, logger=logger)
    return fig


# ---------------------------------------------------------------------------
# 2. Boundary × event-type heatmaps, one panel per method (where errors land)
# ---------------------------------------------------------------------------


def _per_method_boundary_heatmap(
    all_indel_data: dict[str, dict],
    class_name: str,
    *,
    matrix_fn,
    annot_matrices: dict | None = None,
    annot_fmt: str,
    norm=None,
    vmax=None,
    cbar_label: str,
    title: str,
    save_path: Path | None,
    metadata: PlotMetadata | None,
    exclude_events: frozenset[str] = frozenset(),
) -> plt.Figure | None:
    """Shared engine for the count/rate per-method boundary heatmaps.

    ``matrix_fn(payload, boundaries, events) -> ndarray`` builds each method's
    colour matrix on the shared grid; all panels share one colour scale
    (``norm`` or ``vmax``) and one colourbar so methods are comparable.
    ``annot_matrices`` overrides the annotation text (e.g. show raw counts on a
    rate-coloured cell); ``nan`` cells are masked (blank).
    """
    boundaries = _present_boundaries(all_indel_data)
    events = [e for e in _present_events(all_indel_data) if e not in exclude_events]
    methods = [m for m in all_indel_data if isinstance(all_indel_data[m], dict)]
    if not boundaries or not events or not methods:
        logger.info("No boundary-typed INDEL data for class %s.", class_name)
        return None

    matrices = {m: matrix_fn(all_indel_data[m], boundaries, events) for m in methods}
    boundary_labels = [_pretty_boundary(b) for b in boundaries]
    event_labels = [_pretty_event(e) for e in events]

    # Cell-based sizing: size the figure from a fixed square cell plus fixed
    # inch margins. The old width-per-event (1.2 in) made each panel ~9.6 in
    # wide, so cells stretched into wide rectangles and the default wspace left
    # ~2 in gaps between panels — a very flat, white-space-heavy strip.
    n_ev, n_b, n_m = len(events), len(boundaries), len(methods)
    cell_in = 0.7  # side length of one heatmap cell
    left_in = 2.3  # long y-tick labels on the first panel
    right_in = 1.4  # colourbar + its label
    top_in = 1.2  # suptitle + per-panel titles
    bottom_in = 1.5  # rotated x-tick labels + axis label
    wspace_in = 0.4  # gap between method panels

    panel_w_in = cell_in * n_ev
    fig_width = left_in + panel_w_in * n_m + wspace_in * (n_m - 1) + right_in
    height = bottom_in + cell_in * n_b + top_in
    if metadata is not None:
        # The pictogram panel claims ~25% of the final width; widen the figure so
        # the heatmaps and colourbar keep their size after _add_pictogram_panel
        # rescales them into the left content area.
        fig_width /= 0.75
    # Note: no sharey — a shared y-axis lets a later panel's ``yticklabels=False``
    # clear the boundary labels on panel 0.  All panels have identical row counts,
    # so they line up without sharing.
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(fig_width, height),
        squeeze=False,
    )
    axes = axes[0]

    for i, method in enumerate(methods):
        ax = axes[i]
        mat = matrices[method]
        annot = annot_matrices[method] if annot_matrices is not None else True
        sns.heatmap(
            mat,
            ax=ax,
            norm=norm,
            vmin=None if norm is not None else 0,
            vmax=None if norm is not None else vmax,
            annot=annot,
            fmt="" if annot_matrices is not None else annot_fmt,
            cmap="rocket_r",
            cbar=False,
            mask=np.isnan(mat),
            linewidths=0.5,
            linecolor="white",
            xticklabels=event_labels,
            yticklabels=(boundary_labels if i == 0 else False),
        )
        ax.set_facecolor("0.92")  # masked / no-opportunity cells render grey
        ax.set_title(method, fontsize=13)
        ax.set_xlabel("Event type", fontsize=11)
        ax.set_ylabel("Exon position" if i == 0 else "", fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

    # Explicit inch-based margins (converted to fractions) keep the panels tight
    # and the cells square; subplots_adjust because tight_layout would re-expand
    # the axes after the colourbar reserved its margin.
    left = left_in / fig_width
    right = 1 - right_in / fig_width
    bottom = bottom_in / height
    top = 1 - top_in / height
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace_in / panel_w_in)

    # Dedicated colourbar axes in the reserved right margin. fig.colorbar(
    # ax=list(axes)) would instead steal width from the panels and squash the
    # square cells.
    cbar_ax = fig.add_axes([right + 0.012, bottom, 0.16 / fig_width, top - bottom])
    fig.colorbar(axes[-1].collections[0], cax=cbar_ax, label=cbar_label)
    fig.suptitle(f"{title} — {class_name}", fontsize=15)
    _add_pictogram_panel(fig, metadata, logger=logger)

    if save_path is not None:
        _save_figure(fig, save_path, logger=logger)
    return fig


def plot_indel_rates_by_boundary(
    all_indel_data: dict[str, dict],
    class_name: str,
    save_path: Path | None = None,
    metadata: PlotMetadata | None = None,
) -> plt.Figure | None:
    """Per-method exon-position × event-type **rate** heatmap (the comparable view).

    Each cell is ``events ÷ opportunities``: anchored slips, splits and whole
    deletions divide by the count of GT exons of that position type; joins by GT
    intron count.  Whole insertions are excluded (see ``_RATE_EXCLUDED_EVENTS``):
    they are detached false positives with no bounded GT opportunity, so their
    rate is unbounded — they appear only in the count heatmap.
    Colour = rate (shared linear scale); the cell annotation is the rate value.
    Cells with no opportunity or zero events are masked grey.
    """
    boundaries = _present_boundaries(all_indel_data)
    events = [e for e in _present_events(all_indel_data) if e not in _RATE_EXCLUDED_EVENTS]
    methods = [m for m in all_indel_data if isinstance(all_indel_data[m], dict)]
    if not boundaries or not events or not methods:
        logger.info("No boundary-typed INDEL data for class %s.", class_name)
        return None

    def _rate_masked(payload: dict, boundaries: list[str], events: list[str]) -> np.ndarray:
        counts = _count_matrix(payload, boundaries, events)
        rates = _rate_matrix(payload, boundaries, events)
        rates[counts == 0] = np.nan  # gray out zero-event cells, same as count plot
        return rates

    # Annotate with the rate value; blank for no-data or zero-event cells.
    annot_matrices: dict[str, np.ndarray] = {}
    for m in methods:
        rate_mat = _rate_matrix(all_indel_data[m], boundaries, events)
        count_mat = _count_matrix(all_indel_data[m], boundaries, events)
        annot = np.full(rate_mat.shape, "", dtype=object)
        visible = ~np.isnan(rate_mat) & (count_mat > 0)
        annot[visible] = [f"{v:.2f}" for v in rate_mat[visible]]
        annot_matrices[m] = annot

    rate_max = max(
        (np.nanmax(_rate_masked(all_indel_data[m], boundaries, events)) if boundaries else 0.0) for m in methods
    )
    vmax = float(rate_max) if np.isfinite(rate_max) and rate_max > 0 else 1.0

    return _per_method_boundary_heatmap(
        all_indel_data,
        class_name,
        matrix_fn=_rate_masked,
        annot_matrices=annot_matrices,
        annot_fmt="",
        norm=None,
        vmax=vmax,
        cbar_label="Error rate (events ÷ opportunities)",
        title="INDEL Error Rate by GT Boundary",
        save_path=save_path,
        metadata=metadata,
        exclude_events=_RATE_EXCLUDED_EVENTS,
    )


def plot_indel_counts_by_boundary(
    all_indel_data: dict[str, dict],
    class_name: str,
    save_path: Path | None = None,
    metadata: PlotMetadata | None = None,
) -> plt.Figure | None:
    """Per-method GT boundary × event-type **count** heatmap (log colour scale).

    The raw-magnitude companion to :func:`plot_indel_rates_by_boundary`.  A log
    colour norm keeps a single dominant cell (e.g. thousands of spurious whole
    insertions) from washing out the tens-count boundary slips.  Zero cells are
    masked (blank).
    """

    def _count_or_nan(payload: dict, boundaries: list[str], events: list[str]) -> np.ndarray:
        mat = _count_matrix(payload, boundaries, events)
        mat[mat == 0] = np.nan  # LogNorm needs positive values; blank the zeros
        return mat

    boundaries = _present_boundaries(all_indel_data)
    events = _present_events(all_indel_data)
    methods = [m for m in all_indel_data if isinstance(all_indel_data[m], dict)]
    if not boundaries or not events or not methods:
        logger.info("No boundary-typed INDEL data for class %s.", class_name)
        return None

    vmax = max((_count_matrix(all_indel_data[m], boundaries, events).max() for m in methods), default=0.0)
    norm = LogNorm(vmin=1, vmax=max(vmax, 2))

    return _per_method_boundary_heatmap(
        all_indel_data,
        class_name,
        matrix_fn=_count_or_nan,
        annot_fmt=".0f",
        norm=norm,
        cbar_label="Event count (log scale)",
        title="INDEL Events by GT Boundary",
        save_path=save_path,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 3. Boundary × event-type run-length distributions (how large the slips are)
# ---------------------------------------------------------------------------


def plot_individual_error_lengths_histograms(
    all_indel_data: dict[str, dict],
    class_name: str,
    save_dir: Path | None = None,
) -> dict[str, plt.Figure]:
    """One run-length histogram figure per exon-position category.

    Each figure uses a fixed 4-col × 2-row grid: row 0 holds the four insertion
    event types, row 1 the four deletion event types.  Empty cells show ``—``.
    Returns ``{boundary: figure}``; saves to
    ``save_dir/indel_lengths_{boundary}.png`` when *save_dir* is given.
    """
    boundaries = _present_boundaries(all_indel_data)
    if not boundaries:
        logger.info("No boundary-typed INDEL length data for class %s.", class_name)
        return {}

    lengths: dict[tuple[str, str], dict[str, list[int]]] = {}
    for method, boundary, event_type, runs in _iter_events(all_indel_data):
        # Leaned / macro-averaged payloads collapse each bucket to a count (see
        # ``_count_indel_events``) and no longer carry the run lengths this
        # histogram needs — skip those, same as any other non-averageable distribution.
        if not isinstance(runs, list) or not runs:
            continue
        lengths.setdefault((boundary, event_type), {}).setdefault(method, []).extend(runs)

    methods = list(all_indel_data)
    palette = sns.color_palette("tab10", n_colors=max(len(methods), 1))
    method_colors = dict(zip(methods, palette))

    # Fixed grid: insertions on top row, deletions on bottom row
    event_grid = [_EVENT_ORDER[:4], _EVENT_ORDER[4:]]

    figures: dict[str, plt.Figure] = {}

    for boundary in boundaries:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=4,
            figsize=(14, 6.8),
            squeeze=False,
        )

        has_any = False
        for r, event_row in enumerate(event_grid):
            for c, event_type in enumerate(event_row):
                ax = axes[r][c]
                cell = lengths.get((boundary, event_type), {})
                has_data = False
                for method_name, runs in cell.items():
                    positive = [run for run in runs if run > 0]
                    if positive:
                        sns.histplot(
                            np.log10(positive),
                            bins=20,
                            kde=False,
                            ax=ax,
                            color=method_colors.get(method_name),
                            label=method_name,
                            alpha=0.6,
                        )
                        has_data = True
                        has_any = True

                if not has_data:
                    ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes, color="0.6")
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    log_ticks = ax.get_xticks()
                    ax.set_xticks(log_ticks)
                    ax.set_xticklabels([f"{10**x:.0f}" if np.isfinite(x) else "" for x in log_ticks])

                ax.set_title(_pretty_event(event_type), fontsize=11, pad=8)
                _add_icon_to_ax(ax, event_type, y_rel_pos=1.42, logger=logger)
                ax.set_xlabel("Run length (nt)" if r == 1 else "", fontsize=9)
                ax.set_ylabel("Count" if c == 0 else "", fontsize=9)

        if not has_any:
            plt.close(fig)
            continue

        handles, label_texts = [], []
        for ax_row in axes:
            for ax_ in ax_row:
                for handle, label in zip(*ax_.get_legend_handles_labels()):
                    if label not in label_texts:
                        handles.append(handle)
                        label_texts.append(label)
                if ax_.get_legend() is not None:
                    ax_.get_legend().remove()

        fig.suptitle(f"{_pretty_boundary(boundary)} — {class_name}", fontsize=14, y=0.98)
        fig.subplots_adjust(
            left=0.055,
            right=0.985,
            bottom=0.16 if handles else 0.09,
            top=0.78,
            wspace=0.20,
            hspace=1.4,
        )
        if handles:
            fig.legend(
                handles, label_texts,
                loc="lower center",
                ncol=max(len(methods), 1),
                fontsize=11,
                bbox_to_anchor=(0.5, 0.0),
                bbox_transform=fig.transFigure,
            )

        if save_dir is not None:
            _save_figure(fig, save_dir / f"indel_lengths_{boundary}.png", logger=logger)

        figures[boundary] = fig

    return figures
