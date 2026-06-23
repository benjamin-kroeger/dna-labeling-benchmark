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


def plot_phase_drift_percentage_bar(
    df_phase_drift_metrics: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: PlotMetadata | None = None,
) -> Optional[plt.Figure]:
    """Bar chart of coding-phase drift distribution per method.

    Returns
    -------
    Figure | None
    """
    if df_phase_drift_metrics.empty:
        logger.info("No coding-phase drift data for class %s.", class_name)
        return None

    only_frames = df_phase_drift_metrics[df_phase_drift_metrics["metric_key"] == "gt_frames"]
    if only_frames.empty:
        return None

    def _frame_pcts(series: pd.Series) -> pd.DataFrame:
        frame_list = series.iloc[0] if not series.empty else []
        if not isinstance(frame_list, list) or not frame_list:
            return pd.DataFrame(
                {
                    "Phase Offset": ["In-phase (0)", "Offset +1", "Offset +2"],
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
                "Phase Offset": ["In-phase (0)", "Offset +1", "Offset +2"],
                "Percentage": pcts,
            }
        )

    frame_df = only_frames.groupby("method_name")["value"].apply(_frame_pcts).reset_index(level="method_name")

    if frame_df.empty:
        return None

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    sns.barplot(data=frame_df, y="Percentage", x="Phase Offset", hue="method_name", ax=ax)

    for container in ax.containers:
        ax.bar_label(container, label_type="edge", padding=2, fmt="%.1f%%", fontsize=6, rotation=90)

    ax.set_ylim(0, 115)  # 15% headroom for rotated bar labels (fontsize=6, rotation=90, padding=2)
    ax.set_title(
        f"Coding-Phase Drift Distribution — {class_name}",
        fontsize=16,
    )
    ax.set_xlabel("Phase Offset (relative coding-base drift mod 3)", fontsize=12)
    ax.set_ylabel("Percentage of Co-CDS Positions", fontsize=12)
    ax.legend(title="Method Name", loc="upper right", fontsize=9)

    # Annotation lines: skip counts + boundary-indel in-frame rate
    annotation_parts = []

    skip_rows = df_phase_drift_metrics[
        df_phase_drift_metrics["metric_key"].isin(("n_skipped_non_divisible", "n_skipped_short"))
    ]
    if not skip_rows.empty:
        skip_pivoted = skip_rows.pivot_table(index="method_name", columns="metric_key", values="value", aggfunc="first")
        for method, row in skip_pivoted.iterrows():
            non_div = int(row.get("n_skipped_non_divisible", 0))
            short = int(row.get("n_skipped_short", 0))
            if non_div > 0 or short > 0:
                annotation_parts.append(
                    f"{method}: {non_div} skipped (non-divisible GT), {short} skipped (pred < 3 CDS bases)"
                )

    indel_rows = df_phase_drift_metrics[
        df_phase_drift_metrics["metric_key"].isin(("boundary_indel_total", "boundary_indel_in_frame"))
    ]
    if not indel_rows.empty:
        indel_pivoted = indel_rows.pivot_table(index="method_name", columns="metric_key", values="value", aggfunc="first")
        for method, row in indel_pivoted.iterrows():
            total = int(row.get("boundary_indel_total", 0))
            in_frame = int(row.get("boundary_indel_in_frame", 0))
            pct = f"{in_frame / total * 100:.0f}%" if total > 0 else "n/a"
            annotation_parts.append(f"{method}: {in_frame}/{total} boundary indels in-frame ({pct})")

    if annotation_parts:
        ax.annotate(
            "\n".join(annotation_parts),
            xy=(0.5, -0.08),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=7,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc", linewidth=0.8),
        )

    # Reserve bottom margin only for the lines actually rendered in the annotation box.
    bottom_margin = min(0.04 + 0.016 * len(annotation_parts), 0.30) if annotation_parts else 0.02
    fig.tight_layout(rect=[0, bottom_margin, 1, 1])
    _add_pictogram_panel(fig, metadata, logger=logger)

    if save_path is not None:
        _save_figure(fig, save_path, logger=logger)
    return fig
