import logging
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from ..config import ICON_MAP, DEFAULT_MULTI_PLOT_FIG_SIZE, PlotMetadata, DEFAULT_FIG_SIZE
from ..utils import _add_icon_to_ax, _save_figure, _add_pictogram_panel

logger = logging.getLogger(__name__)


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
        df_indel_lengths.groupby(["method_name", "metric_key"])["value"]
        .apply(lambda x: ([len(y) for y in x.iloc[0]] if not x.empty and isinstance(x.iloc[0], list) else []))
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
            "hspace": 0.9,
            "wspace": 0.3,
            "bottom": 0.12,
            "top": 0.8,
            "left": 0.05,
            "right": 0.95,
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
                        np.log10(positive),
                        bins=30,
                        kde=True,
                        ax=ax,
                        color=method_colors[method_name],
                        label=method_name,
                        alpha=0.7,
                    )
                    has_data = True

        if not has_data:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(error_type.replace("_", " ").title(), fontsize=12)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        log_ticks = ax.get_xticks()
        ax.set_xticks(log_ticks)
        ax.set_xticklabels([f"{10**x:.0f}" if np.isfinite(x) else "" for x in log_ticks])
        ax.set_xlabel("Length (log scaled)")

        if error_type in ICON_MAP:
            _add_icon_to_ax(ax, ICON_MAP[error_type], zoom=0.18, y_rel_pos=1.35, logger=logger)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Length Distribution of INDELs — {class_name}", fontsize=16, y=0.98)
    fig.supylabel("Frequency", fontsize=14, x=0.01)

    # Deduplicated legend
    handles, labels = [], []
    for ax_ in axes:
        for handle, label in zip(*ax_.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(unique_methods),
            fontsize=12,
            bbox_to_anchor=(0.5, 0.01),
        )

    if save_path is not None:
        _save_figure(fig, save_path, logger=logger)
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
        df_indel_counts.groupby(["method_name", "metric_key"])["value"]
        .apply(lambda x: (len(x.iloc[0]) if not x.empty and isinstance(x.iloc[0], list) else 0))
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
            total + 0.01 * max(max_val, 1),
            i,
            str(total),
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
