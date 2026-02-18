"""Summary-statistics plotting for multi-method benchmark comparisons.

Every public plotting function follows the same contract:

* Returns a :class:`matplotlib.figure.Figure` — ready for ``wandb.log()``,
  ``mlflow.log_figure()``, or any other experiment-tracker.
* Accepts an optional ``save_path`` — when provided the figure is written to
  disk (parent directories are created automatically).
* Never calls ``plt.show()`` — the caller decides when/whether to display.

The orchestrator :func:`compare_multiple_predictions` returns a
``dict[str, Figure]`` mapping descriptive keys to every figure it created.
"""

from __future__ import annotations

import dataclasses
import logging
import textwrap
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from importlib import resources
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import MaxNLocator

from .evaluate_predictors import EvalMetrics
from .label_definition import LabelConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PACKAGE_NAME = "dna_segmentation_benchmark"
ICON_PATH = resources.files(PACKAGE_NAME) / "icons"
ICON_MAP = {
    "5_prime_extensions": ICON_PATH / "left_extension.png",
    "3_prime_extensions": ICON_PATH / "right_extension.png",
    "whole_insertions": ICON_PATH / "exon_insertion.png",
    "joined": ICON_PATH / "joined_exons.png",
    "5_prime_deletions": ICON_PATH / "left_deletion.png",
    "3_prime_deletions": ICON_PATH / "right_deletion.png",
    "whole_deletions": ICON_PATH / "exon_deletion.png",
    "split": ICON_PATH / "split_exons.png",
}

DEFAULT_FIG_SIZE = (16, 10)
DEFAULT_MULTI_PLOT_FIG_SIZE = (18, 12)


# ---------------------------------------------------------------------------
# Plot metadata — pictogram panel content
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PlotMetadata:
    """Icon and explanatory text shown in the right-side pictogram panel.

    Attributes
    ----------
    icon_path : Path | Traversable | None
        Path to a PNG icon.  ``None`` means no icon yet.
    description : str
        Short paragraph explaining what the plot shows.  Rendered as
        word-wrapped text below the icon.
    display_name : str
        Human-readable title rendered above the icon.
    show_tp_tn_fp_fn : bool
        If ``True`` a compact TP / TN / FP / FN definitions block is
        rendered at the bottom of the panel.
    """

    icon_path: Path | Traversable | None = None
    description: str = ""
    display_name: str = ""
    show_tp_tn_fp_fn: bool = False


# Placeholder entries — fill in ``icon_path`` and ``description`` as
# pictograms are created.  Keys must match those used in
# :func:`compare_multiple_predictions`.
PLOT_METADATA: dict[str, PlotMetadata] = {
    # INDEL summary
    "indel_counts": PlotMetadata(display_name="INDEL Counts"),
    "indel_lengths": PlotMetadata(display_name="INDEL Length Distribution"),
    # ML precision / recall (one entry per level)
    "ml_nucleotide_level_metrics": PlotMetadata(
        display_name="Nucleotide-Level Metrics",
        show_tp_tn_fp_fn=True,
    ),
    "ml_neighborhood_hit_metrics": PlotMetadata(
        display_name="Neighborhood Hit Metrics",
        icon_path=ICON_PATH / "overlap.png",
        show_tp_tn_fp_fn=True,
    ),
    "ml_internal_hit_metrics": PlotMetadata(
        display_name="Internal Hit Metrics",
        icon_path=ICON_PATH / "internal.png",
        show_tp_tn_fp_fn=True,
    ),
    "ml_full_coverage_hit_metrics": PlotMetadata(
        display_name="Full Coverage Hit Metrics",
        icon_path=ICON_PATH / "full_coverage.png",
        show_tp_tn_fp_fn=True,
    ),
    "ml_perfect_boundary_hit_metrics": PlotMetadata(
        display_name="Perfect Boundary Hit Metrics",
        icon_path=ICON_PATH / "prefect_hit.png",
        show_tp_tn_fp_fn=True,
    ),
    "ml_inner_section_boundaries_metrics": PlotMetadata(
        display_name="Inner Section Boundaries",
        show_tp_tn_fp_fn=True,
    ),
    "ml_all_section_boundaries_metrics": PlotMetadata(
        display_name="All Section Boundaries",
        show_tp_tn_fp_fn=True,
    ),
    # IoU
    "iou_average": PlotMetadata(
        display_name="Average IoU",
        icon_path=ICON_PATH / "iou.png",
        description="Measures the intersection over the union of any 2"
                    " overlapping ground truth and predicted section.",
    ),
    "iou_distribution": PlotMetadata(
        display_name="IoU Distribution",
        icon_path=ICON_PATH / "iou.png",
        description="Measures the intersection over the union of any 2"
                    " overlapping ground truth and predicted section.",
    ),
    # Frameshift
    "frameshift": PlotMetadata(display_name="Frameshift Distribution"),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, save_path: Path) -> None:
    """Save *fig* to *save_path*, creating parent dirs as needed."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    logger.info("Saved figure to %s", save_path)


def _add_icon_to_ax(
        ax: plt.Axes,
        icon_path: str,
        zoom: float = 0.2,
        x_rel_pos: float = 0.5,
        y_rel_pos: float = 1.25,
) -> None:
    """Place an image (icon) above *ax*."""
    try:
        icon_img = plt.imread(icon_path)
        imagebox = OffsetImage(icon_img, zoom=zoom)
        ab = AnnotationBbox(
            imagebox, (x_rel_pos, y_rel_pos), xycoords=ax.transAxes, frameon=False
        )
        ax.add_artist(ab)
    except FileNotFoundError:
        logger.warning("Icon not found: %s", icon_path)
    except Exception:
        logger.warning("Could not load icon: %s", icon_path, exc_info=True)


def _add_pictogram_panel(
        fig: plt.Figure,
        metadata: PlotMetadata | None,
        panel_width_fraction: float = 0.22,
) -> None:
    """Add a right-side pictogram panel to *fig*.

    The panel displays an icon (scaled to fit within the panel) with
    explanatory text below it, and optionally a TP/TN/FP/FN
    definitions block.  All existing axes in *fig* are shrunk to make
    room.

    If *metadata* is ``None`` or contains nothing to render, the
    function is a **no-op**.
    """
    if metadata is None:
        return
    has_content = (
        metadata.icon_path is not None
        or metadata.description
        or metadata.show_tp_tn_fp_fn
        or metadata.display_name
    )
    if not has_content:
        return

    # Shrink every existing axes to make room on the right
    for ax in fig.get_axes():
        box = ax.get_position()
        ax.set_position([
            box.x0,
            box.y0,
            box.width * (1 - panel_width_fraction),
            box.height,
        ])

    # Create the panel axes on the freed right-hand side
    panel_left = 1 - panel_width_fraction + 0.01
    panel_width = panel_width_fraction - 0.02
    panel_ax = fig.add_axes([panel_left, 0.05, panel_width, 0.90])
    panel_ax.set_axis_off()

    # Panel dimensions in inches
    fig_w_in, fig_h_in = fig.get_size_inches()
    panel_width_in = panel_width * fig_w_in
    panel_height_in = 0.90 * fig_h_in  # panel is 90% of fig height

    # Convert panel_ax coordinates to figure coordinates for precise placement
    # of elements relative to the panel.
    # panel_ax.transAxes.transform((x, y)) gives figure coordinates.
    # fig.transFigure.inverted().transform((x, y)) gives figure fraction coordinates.
    # We want to work in figure fraction coordinates for text and icon placement.
    panel_bbox = panel_ax.get_position()
    panel_x0, panel_y0, panel_w, panel_h = panel_bbox.x0, panel_bbox.y0, panel_bbox.width, panel_bbox.height

    # y_cursor is in figure fraction coordinates, relative to the top of the panel
    y_cursor = panel_y0 + panel_h * 0.95  # Start near the top of the panel

    # --- Display name ---
    if metadata.display_name:
        # Text x-position is center of panel, y-position is y_cursor
        text_x = panel_x0 + panel_w / 2
        fig.text(
            text_x, y_cursor, metadata.display_name,
            ha="center", va="top", fontsize=13, fontweight="bold",
            wrap=True,
        )
        y_cursor -= panel_h * 0.08  # Move cursor down

    # --- Icon (scaled to fit panel, constrained by both width and height) ---
    # Reserve vertical budget: title ~8%, icon max 40%, description ~20%, TP/TN ~20%
    max_icon_height_frac = 0.40  # max 40% of panel height for the icon
    if metadata.icon_path is not None:
        try:
            icon_img = plt.imread(str(metadata.icon_path))
            icon_w_px = icon_img.shape[1]
            icon_h_px = icon_img.shape[0]

            # Calculate zoom to fit within 85% of panel width and max_icon_height_frac of panel height
            zoom_w = (panel_width_in * fig.dpi * 0.85) / icon_w_px
            max_icon_h_px = max_icon_height_frac * panel_height_in * fig.dpi
            zoom_h = max_icon_h_px / icon_h_px
            zoom = min(zoom_w, zoom_h)

            # Create an inset axes for the icon to ensure it respects bounds
            icon_rendered_w_in = (icon_w_px * zoom) / fig.dpi
            icon_rendered_h_in = (icon_h_px * zoom) / fig.dpi

            # Convert rendered dimensions to figure fraction
            icon_w_fig_frac = icon_rendered_w_in / fig_w_in
            icon_h_fig_frac = icon_rendered_h_in / fig_h_in

            # Calculate icon axes position: centered horizontally, top aligned with y_cursor
            icon_ax_x0 = panel_x0 + (panel_w - icon_w_fig_frac) / 2
            icon_ax_y0 = y_cursor - icon_h_fig_frac # Top of icon is at y_cursor

            icon_ax = fig.add_axes([icon_ax_x0, icon_ax_y0, icon_w_fig_frac, icon_h_fig_frac])
            icon_ax.imshow(icon_img)
            # Pad limits slightly so edge pixels are never clipped
            icon_ax.set_xlim(-1, icon_w_px)
            icon_ax.set_ylim(icon_h_px, -3)
            icon_ax.set_axis_off()

            y_cursor -= icon_h_fig_frac + panel_h * 0.04 # Move cursor down past icon and add spacing
        except Exception:
            logger.warning(
                "Could not load panel icon: %s", metadata.icon_path,
                exc_info=True,
            )

    # --- Description text ---
    if metadata.description:
        wrapped = textwrap.fill(metadata.description, width=26)
        text_x = panel_x0 + panel_w / 2
        fig.text(
            text_x, y_cursor, wrapped,
            ha="center", va="top", fontsize=9,
            linespacing=1.4,
        )
        n_lines = wrapped.count("\n") + 1
        y_cursor -= n_lines * panel_h * 0.05 + panel_h * 0.04 # Move cursor down past description and add spacing

    # --- TP / TN / FP / FN definitions (placed at bottom of panel) ---
    if metadata.show_tp_tn_fp_fn:
        definitions = (
            "\u2022 TP: Correctly predicted\n"
            "\u2022 TN: Correctly absent\n"
            "\u2022 FP: Falsely predicted\n"
            "\u2022 FN: Falsely missed"
        )
        # Place at fixed position near the bottom to avoid overlap
        # Calculate bottom-aligned y-position for definitions block
        tp_y_bottom = panel_y0 + panel_h * 0.05 # 5% from bottom of panel
        # Estimate height of definitions block (4 lines * line_height_factor)
        # This is a rough estimate, actual height depends on font size and dpi
        estimated_line_height_fig_frac = 0.025 * (fig_h_in / DEFAULT_FIG_SIZE[1]) # Scale by figure height
        estimated_block_height_fig_frac = 4 * estimated_line_height_fig_frac * 1.5 # 4 lines, linespacing 1.5
        tp_y_top = tp_y_bottom + estimated_block_height_fig_frac

        # Ensure it doesn't overlap with content above
        final_tp_y = min(y_cursor - panel_h * 0.02, tp_y_top) # 2% buffer from above content

        fig.text(
            panel_x0 + panel_w * 0.05, final_tp_y, definitions,
            ha="left", va="top", fontsize=9,
            linespacing=1.5,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#f0f0f0",
                edgecolor="#cccccc",
                alpha=0.9,
            ),
        )


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------


def plot_individual_error_lengths_histograms(
        df_indel_lengths: pd.DataFrame,
        class_name: str,
        save_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """Histograms of INDEL error lengths (log-scaled), one subplot per type.

    Parameters
    ----------
    df_indel_lengths : pd.DataFrame
        Long-format frame with columns ``method_name``, ``metric_key``,
        ``value`` (lists of error arrays).
    class_name : str
        Human-readable class name for the title.
    save_path : Path | None
        If given, write the figure to this path.

    Returns
    -------
    Figure | None
        The matplotlib figure, or ``None`` when there is no data.
    """
    if df_indel_lengths.empty:
        logger.info("No INDEL length data for class %s.", class_name)
        return None

    method_error_lengths = (
        df_indel_lengths
        .groupby(["method_name", "metric_key"])["value"]
        .apply(
            lambda x: (
                [len(y) for y in x.iloc[0]]
                if not x.empty and isinstance(x.iloc[0], list)
                else []
            )
        )
        .unstack(fill_value=None)
    )

    unique_methods = method_error_lengths.index.tolist()
    if not unique_methods:
        return None

    palette = sns.color_palette("tab10", n_colors=len(unique_methods))
    method_colors = dict(zip(unique_methods, palette))

    error_types = [c for c in ICON_MAP if c in method_error_lengths.columns]
    if not error_types:
        return None

    n_cols = 4
    n_rows = (len(error_types) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(DEFAULT_MULTI_PLOT_FIG_SIZE[0], n_rows * 5),
        gridspec_kw={
            "hspace": 0.9, "wspace": 0.3,
            "bottom": 0.12, "top": 0.8, "left": 0.05, "right": 0.95,
        },
    )
    axes = axes.flatten()

    for i, error_type in enumerate(error_types):
        ax = axes[i]
        has_data = False

        for method_name in unique_methods:
            lengths = method_error_lengths.loc[method_name, error_type]
            if lengths and isinstance(lengths, list) and any(np.isfinite(lengths)):
                positive = [length for length in lengths if length > 0]
                if positive:
                    sns.histplot(
                        np.log10(positive), bins=30, kde=True, ax=ax,
                        color=method_colors[method_name], label=method_name,
                        alpha=0.7,
                    )
                    has_data = True

        if not has_data:
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center", transform=ax.transAxes,
            )

        ax.set_title(error_type.replace("_", " ").title(), fontsize=12)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        log_ticks = ax.get_xticks()
        ax.set_xticks(log_ticks)
        ax.set_xticklabels(
            [f"{10 ** x:.0f}" if np.isfinite(x) else "" for x in log_ticks]
        )
        ax.set_xlabel("Length (log scaled)")

        if error_type in ICON_MAP:
            _add_icon_to_ax(ax, ICON_MAP[error_type], zoom=0.18, y_rel_pos=1.35)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Length Distribution of INDELs — {class_name}", fontsize=16, y=0.98)
    fig.supylabel("Frequency", fontsize=14, x=0.01)

    # Deduplicated legend
    handles, labels = [], []
    for ax_ in axes:
        for h, l in zip(*ax_.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(
            handles, labels, loc="lower center",
            ncol=len(unique_methods), fontsize=12, bbox_to_anchor=(0.5, 0.01),
        )

    if save_path is not None:
        _save_figure(fig, save_path)
    return fig


def plot_stacked_indel_counts_bar(
        df_indel_counts: pd.DataFrame,
        class_name: str,
        save_path: Optional[Path] = None,
        metadata: PlotMetadata | None = None,
) -> Optional[plt.Figure]:
    """Stacked horizontal bar chart of INDEL counts per method.

    Returns
    -------
    Figure | None
    """
    if df_indel_counts.empty:
        logger.info("No INDEL count data for class %s.", class_name)
        return None

    counts = (
        df_indel_counts
        .groupby(["method_name", "metric_key"])["value"]
        .apply(
            lambda x: (
                len(x.iloc[0]) if not x.empty and isinstance(x.iloc[0], list) else 0
            )
        )
        .unstack(fill_value=0)
    )

    if counts.empty:
        return None

    totals = counts.sum(axis=1)
    counts = counts.loc[totals.sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    counts.plot(kind="barh", stacked=True, ax=ax, colormap="viridis")

    max_val = totals.max()
    for i, (idx, total) in enumerate(totals.sort_values(ascending=True).items()):
        ax.text(
            total + 0.01 * max(max_val, 1), i, str(total),
            va="center", ha="left", fontweight="bold",
        )

    ax.set_xlim(0, max(max_val * 1.15, 1))
    ax.set_title(f"INDEL Counts by Method — {class_name}", fontsize=16)
    ax.set_xlabel("Total Number of INDELs", fontsize=12)
    ax.set_ylabel("Method Name", fontsize=12)
    ax.legend(title="INDEL Type", loc="lower right", fontsize=9)

    fig.tight_layout()
    _add_pictogram_panel(fig, metadata)

    if save_path is not None:
        _save_figure(fig, save_path)
    return fig


def plot_frameshift_percentage_bar(
        df_frameshift_metrics: pd.DataFrame,
        class_name: str,
        save_path: Optional[Path] = None,
        metadata: PlotMetadata | None = None,
) -> Optional[plt.Figure]:
    """Bar chart of codon reading-frame distribution per method.

    Returns
    -------
    Figure | None
    """
    if df_frameshift_metrics.empty:
        logger.info("No frameshift data for class %s.", class_name)
        return None

    only_frames = df_frameshift_metrics[
        df_frameshift_metrics["metric_key"] == "gt_frames"
        ]
    if only_frames.empty:
        return None

    def _frame_pcts(series: pd.Series) -> pd.DataFrame:
        frame_list = series.iloc[0] if not series.empty else []
        if not isinstance(frame_list, list) or not frame_list:
            return pd.DataFrame({
                "Frame": ["Ground Truth (0)", "Shift 1 (1)", "Shift 2 (2)"],
                "Percentage": [0.0, 0.0, 0.0],
            })
        flat = np.concatenate(frame_list)
        flat = flat[np.isfinite(flat)].astype(int)
        counts = np.bincount(flat, minlength=3)[:3] if flat.size else np.zeros(3, dtype=int)
        total = counts.sum()
        pcts = (counts / total * 100) if total > 0 else np.zeros(3)
        return pd.DataFrame({
            "Frame": ["Ground Truth (0)", "Shift 1 (1)", "Shift 2 (2)"],
            "Percentage": pcts,
        })

    frame_df = (
        only_frames
        .groupby("method_name")["value"]
        .apply(_frame_pcts)
        .reset_index(level="method_name")
    )

    if frame_df.empty:
        return None

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    sns.barplot(data=frame_df, y="Percentage", x="Frame", hue="method_name", ax=ax)

    for container in ax.containers:
        ax.bar_label(container, label_type="edge", padding=3, fmt="%.2f%%")

    ax.set_title(
        f"Codon Reading Frame Distribution — {class_name}", fontsize=16,
    )
    ax.set_xlabel("Reading Frame", fontsize=12)
    ax.set_ylabel("Percentage of Codons", fontsize=12)
    ax.legend(title="Method Name", loc="upper right", fontsize=9)

    fig.tight_layout()
    _add_pictogram_panel(fig, metadata)

    if save_path is not None:
        _save_figure(fig, save_path)
    return fig


def plot_ml_metrics_bar(
        df_ml_metrics: pd.DataFrame,
        class_name: str,
        save_path_prefix: Optional[Path] = None,
        metadata_map: dict[str, PlotMetadata] | None = None,
) -> list[plt.Figure]:
    """Grouped bar chart of ML metrics (precision / recall) per level.

    One figure is created per metric level (nucleotide, section, etc.).

    Parameters
    ----------
    save_path_prefix : Path | None
        If given, figures are saved as ``<prefix>_<level>.png``.
    metadata_map : dict[str, PlotMetadata] | None
        Mapping from ``ml_<level>`` key to :class:`PlotMetadata`.
        Looked up per-level to attach the correct pictogram panel.

    Returns
    -------
    list[Figure]
        One figure per metric level.
    """
    if df_ml_metrics.empty:
        logger.info("No ML metrics data for class %s.", class_name)
        return []

    # Filter out iou_stats as it has a different schema (mean/std/count) vs precision/recall
    # and is plotted separately in plot_iou_metrics.
    df_ml_metrics = df_ml_metrics[df_ml_metrics["metric_key"] != "iou_stats"].copy()

    if df_ml_metrics.empty:
        return []

    ml_scores = (
        df_ml_metrics
        .groupby(["method_name", "metric_key"])["value"]
        .apply(lambda x: x.iloc[0] if not x.empty else 0)
        .unstack(fill_value=0)
    )

    if ml_scores.empty:
        return []

    melted = (
        ml_scores
        .reset_index()
        .melt(id_vars=["method_name", "metric_key"], var_name="metric", value_name="Score")
    )

    if metadata_map is None:
        metadata_map = {}

    figures: list[plt.Figure] = []

    for level in melted["metric_key"].unique():
        level_df = melted[melted["metric_key"] == level].copy()

        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        sns.barplot(data=level_df, y="Score", x="metric", hue="method_name", ax=ax)

        for container in ax.containers:
            ax.bar_label(container, label_type="edge", padding=3, fmt="%.3f")

        ax.set_title(f"{level} — {class_name}", fontsize=16)
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(title="Method Name", loc="upper right", fontsize=9)
        fig.tight_layout()

        _add_pictogram_panel(fig, metadata_map.get(f"ml_{level}"))

        if save_path_prefix is not None:
            _save_figure(fig, save_path_prefix.with_name(
                f"{save_path_prefix.stem}_{level}.png"
            ))

        figures.append(fig)

    return figures


def plot_iou_metrics(
        df_iou: pd.DataFrame,
        class_name: str,
        save_path_prefix: Optional[Path] = None,
        metadata_average: PlotMetadata | None = None,
        metadata_distribution: PlotMetadata | None = None,
) -> list[plt.Figure]:
    """Generate IoU analysis plots: Average Bar Chart and Distribution Violin Plot.

    Parameters
    ----------
    df_iou : pd.DataFrame
        Filtered DataFrame containing only 'iou_scores' rows.
    class_name : str
        Human-readable class name.
    save_path_prefix : Path | None
        Prefix for saving figures (e.g. '.../EXON_iou').
    metadata_average, metadata_distribution : PlotMetadata | None
        Pictogram panel content for the average / distribution figure.

    Returns
    -------
    list[Figure]
        [Average IoU Figure, Distribution Figure]
    """
    if df_iou.empty:
        logger.info("No IoU data for class %s.", class_name)
        return []

    # Explode the lists of scores into individual rows
    exploded = (
        df_iou
        .explode("value")
        .astype({"value": float})
        .rename(columns={"value": "IoU"})
    )
    # Drop NaNs if any
    exploded = exploded.dropna(subset=["IoU"])

    if exploded.empty:
        return []

    figures = []

    # --- Plot 1: Average IoU Score per Method (Bar Plot) ---
    avg_scores = exploded.groupby("method_name")["IoU"].mean().reset_index()

    fig1, ax1 = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    sns.barplot(
        data=avg_scores, x="method_name", y="IoU", hue="method_name",
        ax=ax1, palette="viridis", legend=False
    )

    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.3f", padding=3)

    ax1.set_title(f"Average IoU Score per Method — {class_name}", fontsize=16)
    ax1.set_ylabel("Average Intersection over Union", fontsize=12)
    ax1.set_xlabel("Method Name", fontsize=12)
    ax1.set_ylim(0, 1.05)
    fig1.tight_layout()
    _add_pictogram_panel(fig1, metadata_average)

    if save_path_prefix:
        _save_figure(fig1, save_path_prefix.with_name(f"{save_path_prefix.name}_average.png"))
    figures.append(fig1)

    # --- Plot 2: IoU Distribution per Method (Violin Plot) ---
    fig2, ax2 = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    sns.violinplot(
        data=exploded, x="method_name", y="IoU", hue="method_name",
        ax=ax2, palette="viridis", cut=0, inner="quartile"
    )

    ax2.set_title(f"IoU Score Distribution per Method — {class_name}", fontsize=16)
    ax2.set_ylabel("Intersection over Union", fontsize=12)
    ax2.set_xlabel("Method Name", fontsize=12)
    ax2.set_ylim(0, 1.05)

    fig2.tight_layout()
    _add_pictogram_panel(fig2, metadata_distribution)

    if save_path_prefix:
        _save_figure(fig2, save_path_prefix.with_name(f"{save_path_prefix.name}_distribution.png"))
    figures.append(fig2)

    return figures


def plot_boundary_precision_landscapes(
        df_fuzzy_boundaries: pd.DataFrame,
        max_range: int = 10,
        metadata: PlotMetadata | None = None,
) -> list[plt.Figure]:
    """Plot the two diagnostic matrices to visualize model bias and reliability.

    Each method in *df_fuzzy_boundaries* produces one figure with two
    sub-plots:

    1. **Bias Matrix** — 2-D histogram of signed boundary residuals.
    2. **Reliability Matrix** — cumulative recall surface.

    Both matrices are stored as ``pd.DataFrame`` objects whose index
    represents the **5' dimension** (rows) and whose columns represent
    the **3' dimension**.  The y-axis is inverted so that the lowest
    value sits at the bottom (standard mathematical orientation).
    """
    figures: list[plt.Figure] = []

    for method in df_fuzzy_boundaries["method_name"].unique().tolist():
        bias_matrix, reliability_matrix = (
            df_fuzzy_boundaries[
                df_fuzzy_boundaries["method_name"] == method
                ]["value"].iloc[0]
        )

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # --- Plot 1: The Bias Matrix (The "Exon Fingerprint") ---
        sns.heatmap(
            bias_matrix,
            ax=axes[0],
            cmap="YlGnBu",
            cbar_kws={"label": "Frequency (Number of Exons)"},
        )
        axes[0].set_title(
            f"Boundary Bias Landscape (±{max_range}bp)", fontsize=14, pad=15,
        )
        axes[0].set_ylabel(bias_matrix.index.name, fontsize=12)
        axes[0].set_xlabel(bias_matrix.columns.name, fontsize=12)

        # Add crosshairs at the (0,0) perfect-match centre
        axes[0].axvline(max_range + 0.5, color="red", linestyle="--", alpha=0.5)
        axes[0].axhline(max_range + 0.5, color="red", linestyle="--", alpha=0.5)

        # Invert y-axis so the lowest residual is at the bottom
        axes[0].invert_yaxis()

        # --- Plot 2: The Reliability Matrix (The "Tolerance Budget") ---
        sns.heatmap(
            reliability_matrix,
            ax=axes[1],
            cmap="magma",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Recall (Percentage of Exons Found)"},
        )
        axes[1].set_title(
            f"Cumulative Reliability (0 to {max_range}bp Tolerance)",
            fontsize=14,
            pad=15,
        )
        axes[1].set_ylabel(reliability_matrix.index.name, fontsize=12)
        axes[1].set_xlabel(reliability_matrix.columns.name, fontsize=12)

        # Invert y-axis so tolerance 0 is at the bottom
        axes[1].invert_yaxis()

        fig.suptitle(f"{method}", fontsize=14)
        plt.tight_layout()
        _add_pictogram_panel(fig, metadata)
        figures.append(fig)

    return figures


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compare_multiple_predictions(
        per_method_benchmark_res: dict[str, dict],
        label_config: LabelConfig,
        classes: list[int],
        metrics_to_eval: list[EvalMetrics],
        output_dir: Optional[Path] = None,
) -> dict[str, plt.Figure]:
    """Generate all summary plots and return them as a dict.

    Parameters
    ----------
    per_method_benchmark_res : dict[str, dict]
        Outer key = method name, inner dict = benchmark result as returned
        by :func:`benchmark_gt_vs_pred_multiple`.
    label_config : LabelConfig
        Used to resolve class token → human-readable name.
    classes : list[int]
        Token values that were evaluated.
    metrics_to_eval : list[EvalMetrics]
        Which metric groups were computed.
    output_dir : Path | None
        If provided, every figure is saved as a PNG inside this directory.

    Returns
    -------
    dict[str, Figure]
        Keys are descriptive strings such as
        ``"EXON_indel_counts"``, ``"EXON_indel_lengths"``, etc.
        Suitable for direct ``wandb.log()`` usage.
    """
    # ---- Build long-format DataFrame ------------------------------------
    rows: list[list] = []
    for method_name, benchmark_results in per_method_benchmark_res.items():
        for class_name, metric_groupings in benchmark_results.items():
            class_name_str = class_name if isinstance(class_name, str) else str(class_name)
            for metric_group, metric_data in metric_groupings.items():
                metric_group_str = (
                    metric_group if isinstance(metric_group, str) else metric_group.name
                )
                for single_metric_key, value in metric_data.items():
                    rows.append([
                        method_name,
                        class_name_str,
                        metric_group_str,
                        single_metric_key,
                        value,
                    ])

    if not rows:
        logger.warning("No benchmark data collected — nothing to plot.")
        return {}

    df = pd.DataFrame(
        rows, columns=["method_name", "measured_class", "metric_group", "metric_key", "value"],
    )

    figures: dict[str, plt.Figure] = {}

    for class_token in classes:
        class_name = label_config.name_of(class_token)

        # ---- INDEL plots ------------------------------------------------
        if EvalMetrics.INDEL in metrics_to_eval:
            df_indel = df[
                (df["measured_class"] == class_name)
                & (df["metric_group"] == EvalMetrics.INDEL.name)
                ].copy()

            if not df_indel.empty:
                fig = plot_stacked_indel_counts_bar(
                    df_indel, class_name,
                    save_path=(output_dir / f"{class_name}_indel_counts.png") if output_dir else None,
                    metadata=PLOT_METADATA.get("indel_counts"),
                )
                if fig is not None:
                    figures[f"{class_name}_indel_counts"] = fig

                fig = plot_individual_error_lengths_histograms(
                    df_indel, class_name,
                    save_path=(output_dir / f"{class_name}_indel_lengths.png") if output_dir else None,
                )
                if fig is not None:
                    figures[f"{class_name}_indel_lengths"] = fig

        # ---- Fuzzy boundary plots ---------------------------------------
        df_fuzzy = df[
            (df["measured_class"] == class_name)
            & (df["metric_group"] == EvalMetrics.ML.name)
            & (df["metric_key"] == "fuzzy_metrics")
            ].copy()

        df = df[df["metric_key"] != "fuzzy_metrics"]

        fuzzy_metrics_figs = plot_boundary_precision_landscapes(
            df_fuzzy,
            metadata=PLOT_METADATA.get("fuzzy_metrics"),
        )
        for i, fig in enumerate(fuzzy_metrics_figs):
            figures[f"{i}_fuzzy_metrics"] = fig

        # ---- IoU plots --------------------------------------------------
        # IoU scores are stored under the SECTION group
        if EvalMetrics.SECTION in metrics_to_eval:
            df_iou = df[
                (df["measured_class"] == class_name)
                & (df["metric_group"] == EvalMetrics.SECTION.name)
                & (df["metric_key"] == "iou_scores")
                ].copy()

            if not df_iou.empty:
                prefix = (output_dir / f"{class_name}_iou") if output_dir else None
                iou_figs = plot_iou_metrics(
                    df_iou, class_name,
                    save_path_prefix=prefix,
                    metadata_average=PLOT_METADATA.get("iou_average"),
                    metadata_distribution=PLOT_METADATA.get("iou_distribution"),
                )
                for idx, fig in enumerate(iou_figs):
                    suffix = "average" if idx == 0 else "distribution"
                    figures[f"{class_name}_iou_{suffix}"] = fig

        # ---- ML plots ---------------------------------------------------
        if EvalMetrics.ML in metrics_to_eval:
            df_ml = df[
                (df["measured_class"] == class_name)
                & (df["metric_group"] == EvalMetrics.ML.name)
                ].copy()

            if not df_ml.empty:
                prefix = (output_dir / f"{class_name}_ml") if output_dir else None
                ml_figs = plot_ml_metrics_bar(
                    df_ml, class_name,
                    save_path_prefix=prefix,
                    metadata_map=PLOT_METADATA,
                )
                for idx, fig in enumerate(ml_figs):
                    figures[f"{class_name}_ml_{idx}"] = fig

        # ---- Frameshift plots -------------------------------------------
        if EvalMetrics.FRAMESHIFT in metrics_to_eval:
            df_fs = df[
                (df["measured_class"] == class_name)
                & (df["metric_group"] == EvalMetrics.FRAMESHIFT.name)
                ].copy()

            if not df_fs.empty:
                fig = plot_frameshift_percentage_bar(
                    df_fs, class_name,
                    save_path=(output_dir / f"{class_name}_frameshift.png") if output_dir else None,
                    metadata=PLOT_METADATA.get("frameshift"),
                )
                if fig is not None:
                    figures[f"{class_name}_frameshift"] = fig

    return figures
