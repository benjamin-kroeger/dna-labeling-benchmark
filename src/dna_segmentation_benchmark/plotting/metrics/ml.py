import logging
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import PlotMetadata, DEFAULT_FIG_SIZE
from ..utils import _save_figure, _add_pictogram_panel

logger = logging.getLogger(__name__)

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

        _add_pictogram_panel(fig, metadata_map.get(level),logger=logger)

        if save_path_prefix is not None:
            _save_figure(fig, save_path_prefix.with_name(
                f"{save_path_prefix.stem}_{level}.png"
            ),logger=logger)

        figures.append(fig)

    return figures
