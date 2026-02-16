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

import logging
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

ICON_MAP = {
    "5_prime_extensions": resources.files(PACKAGE_NAME) / "icons" / "left_extension.png",
    "3_prime_extensions": resources.files(PACKAGE_NAME) / "icons" / "right_extension.png",
    "whole_insertions": resources.files(PACKAGE_NAME) / "icons" / "exon_insertion.png",
    "joined": resources.files(PACKAGE_NAME) / "icons" / "joined_exons.png",
    "5_prime_deletions": resources.files(PACKAGE_NAME) / "icons" / "left_deletion.png",
    "3_prime_deletions": resources.files(PACKAGE_NAME) / "icons" / "right_deletion.png",
    "whole_deletions": resources.files(PACKAGE_NAME) / "icons" / "exon_deletion.png",
    "split": resources.files(PACKAGE_NAME) / "icons" / "split_exons.png",
}

DEFAULT_FIG_SIZE = (16, 10)
DEFAULT_MULTI_PLOT_FIG_SIZE = (18, 12)


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
    ax.legend(title="INDEL Type", bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path)
    return fig


def plot_frameshift_percentage_bar(
        df_frameshift_metrics: pd.DataFrame,
        class_name: str,
        save_path: Optional[Path] = None,
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
    ax.legend(title="Method Name", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()

    if save_path is not None:
        _save_figure(fig, save_path)
    return fig


def plot_ml_metrics_bar(
        df_ml_metrics: pd.DataFrame,
        class_name: str,
        save_path_prefix: Optional[Path] = None,
) -> list[plt.Figure]:
    """Grouped bar chart of ML metrics (precision / recall) per level.

    One figure is created per metric level (nucleotide, section, etc.).

    Parameters
    ----------
    save_path_prefix : Path | None
        If given, figures are saved as ``<prefix>_<level>.png``.

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
        ax.legend(title="Method Name", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()

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
    
    # Add accumulation of points if dataset is not huge, or just stick to violin
    # sns.stripplot(data=exploded, x="method_name", y="IoU", color="black", alpha=0.3, size=2, ax=ax2)
    
    fig2.tight_layout()

    if save_path_prefix:
        _save_figure(fig2, save_path_prefix.with_name(f"{save_path_prefix.name}_distribution.png"))
    figures.append(fig2)

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
                )
                if fig is not None:
                    figures[f"{class_name}_indel_counts"] = fig

                fig = plot_individual_error_lengths_histograms(
                    df_indel, class_name,
                    save_path=(output_dir / f"{class_name}_indel_lengths.png") if output_dir else None,
                )
                if fig is not None:
                    figures[f"{class_name}_indel_lengths"] = fig

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
                iou_figs = plot_iou_metrics(df_iou, class_name, save_path_prefix=prefix)
                for idx, fig in enumerate(iou_figs):
                    # 0 = Average, 1 = Distribution
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
                ml_figs = plot_ml_metrics_bar(df_ml, class_name, save_path_prefix=prefix)
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
                )
                if fig is not None:
                    figures[f"{class_name}_frameshift"] = fig

    return figures
