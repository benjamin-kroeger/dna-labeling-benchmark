import logging
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import PlotMetadata, DEFAULT_FIG_SIZE
from ..utils import _save_figure, _add_pictogram_panel

logger = logging.getLogger(__name__)

def plot_iou_metrics(
        df_iou: pd.DataFrame,
        class_name: str,
        save_path_prefix: Optional[Path] = None,
        metadata_average: PlotMetadata | None = None,
        metadata_distribution: PlotMetadata | None = None,
) -> list[plt.Figure]:
    """Generate IoU analysis plots: Average Bar Chart and Distribution Raincloud Plot.

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
    _add_pictogram_panel(fig1, metadata_average,logger=logger)

    if save_path_prefix:
        _save_figure(fig1, save_path_prefix.with_name(f"{save_path_prefix.name}_average.png"),logger=logger)
    figures.append(fig1)

    # --- Plot 2: IoU Distribution per Method (ECDF Plot) -----------
    #
    # When data is highly concentrated at exactly 1.0 (perfect matches), 
    # density plots (violins, rainclouds) break down. An Empirical Cumulative 
    # Distribution Function (ECDF) plotted as a survival curve is ideal here.
    # It shows "Proportion of predictions with an IoU score >= X".
    # ------------------------------------------------------------------

    fig2, ax2 = plt.subplots(figsize=DEFAULT_FIG_SIZE)

    sns.ecdfplot(
        data=exploded, 
        x="IoU", 
        hue="method_name", 
        ax=ax2, 
        palette="viridis", 
        complementary=True, 
        linewidth=2.5
    )

    ax2.set_title(
        f"IoU Score Distribution (Survival Curve) — {class_name}", fontsize=16,
    )
    ax2.set_ylabel("Proportion of Predictions with IoU $\\geq$ X", fontsize=12)
    ax2.set_xlabel("Intersection over Union Score (X)", fontsize=12)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    
    # Add grid lines so it's easy to read across the percentage points
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    # ECDF creates its own legend. If we call ax2.legend() directly without handles, 
    # it can accidentally clear the items. Moving the existing legend is safer.
    if ax2.get_legend() is not None:
        sns.move_legend(ax2, "lower left", title="Method Name", fontsize=10)

    fig2.tight_layout()
    _add_pictogram_panel(fig2, metadata_distribution,logger=logger)

    if save_path_prefix:
        _save_figure(
            fig2,
            save_path_prefix.with_name(
                f"{save_path_prefix.name}_distribution.png",
            ),logger=logger
        )
    figures.append(fig2)

    return figures
