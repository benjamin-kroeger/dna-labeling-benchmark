import logging
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import PlotMetadata, DEFAULT_FIG_SIZE
from ..utils import _save_figure, _add_pictogram_panel

logger = logging.getLogger(__name__)


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

    only_frames = df_frameshift_metrics[df_frameshift_metrics["metric_key"] == "gt_frames"]
    if only_frames.empty:
        return None

    def _frame_pcts(series: pd.Series) -> pd.DataFrame:
        frame_list = series.iloc[0] if not series.empty else []
        if not isinstance(frame_list, list) or not frame_list:
            return pd.DataFrame(
                {
                    "Frame": ["Ground Truth (0)", "Shift 1 (1)", "Shift 2 (2)"],
                    "Percentage": [0.0, 0.0, 0.0],
                }
            )
        flat = np.asarray(frame_list, dtype=float)
        flat = flat[np.isfinite(flat)].astype(int)
        counts = np.bincount(flat, minlength=3)[:3] if flat.size else np.zeros(3, dtype=int)
        total = counts.sum()
        pcts = (counts / total * 100) if total > 0 else np.zeros(3)
        return pd.DataFrame(
            {
                "Frame": ["Ground Truth (0)", "Shift 1 (1)", "Shift 2 (2)"],
                "Percentage": pcts,
            }
        )

    frame_df = only_frames.groupby("method_name")["value"].apply(_frame_pcts).reset_index(level="method_name")

    if frame_df.empty:
        return None

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    sns.barplot(data=frame_df, y="Percentage", x="Frame", hue="method_name", ax=ax)

    for container in ax.containers:
        ax.bar_label(container, label_type="edge", padding=3, fmt="%.2f%%")

    ax.set_title(
        f"Codon Reading Frame Distribution — {class_name}",
        fontsize=16,
    )
    ax.set_xlabel("Reading Frame", fontsize=12)
    ax.set_ylabel("Percentage of Codons", fontsize=12)
    ax.legend(title="Method Name", loc="upper right", fontsize=9)

    # Build per-method boundary-indel mod-3 note, if the counts were computed.
    indel_rows = df_frameshift_metrics[
        df_frameshift_metrics["metric_key"].isin(("boundary_indel_total", "boundary_indel_in_frame"))
    ]
    if not indel_rows.empty:
        pivoted = indel_rows.pivot_table(index="method_name", columns="metric_key", values="value", aggfunc="first")
        parts = []
        for method, row in pivoted.iterrows():
            total = int(row.get("boundary_indel_total", 0))
            in_frame = int(row.get("boundary_indel_in_frame", 0))
            pct = f"{in_frame / total * 100:.0f}%" if total > 0 else "n/a"
            parts.append(f"{method}: {in_frame}/{total} boundary indels in-frame ({pct})")
        ax.annotate(
            "  |  ".join(parts),
            xy=(0.5, -0.15),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=11,
            color="#333333",
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.8),
        )

    fig.tight_layout()
    _add_pictogram_panel(fig, metadata, logger=logger)

    if save_path is not None:
        _save_figure(fig, save_path, logger=logger)
    return fig
