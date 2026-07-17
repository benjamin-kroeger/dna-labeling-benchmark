import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import PlotMetadata, DEFAULT_FIG_SIZE
from ..utils import _save_figure, _add_pictogram_panel, _annotate_bar_with_errorbar

logger = logging.getLogger(__name__)


def plot_ml_metrics_bar(
    df_ml_metrics: pd.DataFrame,
    class_name: str,
    save_path_prefix: Path | None = None,
    metadata_map: dict[str, PlotMetadata] | None = None,
) -> dict[str, plt.Figure]:
    """Grouped bar chart of ML metrics (precision / recall) per level.

    One figure is created per metric level (nucleotide, section, etc.).

    Parameters
    ----------
    df_ml_metrics : pd.DataFrame
        Long-form ML-metrics table (one row per method / level / metric).
    class_name : str
        Class label used in log messages and figure titles.
    save_path_prefix : Path | None
        If given, figures are saved as ``<prefix>_<level>.png``.
    metadata_map : dict[str, PlotMetadata] | None
        Mapping from the bare metric-level name (e.g. ``"nucleotide"``,
        ``"neighborhood_hit"``) to :class:`PlotMetadata`.
        Looked up per-level to attach the correct pictogram panel.

    Returns
    -------
    dict[str, Figure]
        Mapping of metric-level name (e.g. ``"neighborhood_hit"``,
        ``"ts_level_match_rate"``) to its figure. Iteration order matches
        the order in which levels first appear in ``df_ml_metrics``.
    """
    if df_ml_metrics.empty:
        logger.info("No ML metrics data for class %s.", class_name)
        return {}

    # Filter out iou_stats as it has a different schema (mean/std/count) vs precision/recall
    # and is plotted separately in plot_iou_metrics.
    df_ml_metrics = df_ml_metrics[df_ml_metrics["metric_key"] != "iou_stats"].copy()

    if df_ml_metrics.empty:
        return {}

    ml_scores = (
        df_ml_metrics.groupby(["method_name", "metric_key"], sort=False)["value"]
        .apply(lambda x: x.iloc[0] if not x.empty else 0)
        .unstack(fill_value=0)
    )

    if ml_scores.empty:
        return {}

    melted = ml_scores.reset_index().melt(id_vars=["method_name", "metric_key"], var_name="metric", value_name="Score")

    if metadata_map is None:
        metadata_map = {}

    figures: dict[str, plt.Figure] = {}

    for level in melted["metric_key"].unique():
        full_level_df = melted[melted["metric_key"] == level].copy()

        # Separate bootstrap stderr entries from the bar-plot metrics
        is_stderr = full_level_df["metric"].str.endswith("_stderr")
        stderr_df = full_level_df[is_stderr]
        level_df = full_level_df[~is_stderr].dropna(subset=["Score"])

        # Build lookup: (method_name, base_metric) -> stderr value
        stderr_lookup: dict[tuple, float] = {}
        for _, row in stderr_df.iterrows():
            base = row["metric"].replace("_stderr", "")
            v = row["Score"]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                stderr_lookup[(row["method_name"], base)] = float(v)

        method_order = list(level_df["method_name"].unique())
        metric_order = list(level_df["metric"].unique())

        fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
        sns.barplot(
            data=level_df,
            y="Score",
            x="metric",
            hue="method_name",
            order=metric_order,
            hue_order=method_order,
            errorbar=None,
            ax=ax,
        )

        bar_containers = list(ax.containers)
        for container, method in zip(bar_containers, method_order):
            for patch, metric_name in zip(container.patches, metric_order):
                se = stderr_lookup.get((method, metric_name))
                _annotate_bar_with_errorbar(ax, patch, patch.get_height(), se)

        md = metadata_map.get(level)
        title = md.display_name if (md and md.display_name) else str(level)
        ax.set_title(f"{title} — {class_name}", fontsize=16)
        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(title="Method Name", loc="upper right", fontsize=9)
        fig.tight_layout()

        _add_pictogram_panel(fig, md, logger=logger)

        if save_path_prefix is not None:
            _save_figure(fig, save_path_prefix.with_name(f"{save_path_prefix.stem}_{level}.png"), logger=logger)

        figures[str(level)] = fig

    return figures
