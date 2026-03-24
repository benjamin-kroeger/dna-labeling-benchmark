"""Plotting functions for DIAGNOSTIC_DEPTH metrics.

Provides visualisations for junction error taxonomy, segment length
distributions, and position bias analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..config import DEFAULT_FIG_SIZE, PlotMetadata
from ..utils import _save_figure, _add_pictogram_panel

logger = logging.getLogger(__name__)

# Error types in display order
_ERROR_TYPES = [
    "exon_skip_count",
    "segment_retention_count",
    "novel_insertion_count",
    "cascade_shift_count",
    "compensating_error_count",
]

_ERROR_DISPLAY_NAMES = {
    "exon_skip_count": "Exon Skip",
    "segment_retention_count": "Segment Retention",
    "novel_insertion_count": "Novel Insertion",
    "cascade_shift_count": "Cascade Shift",
    "compensating_error_count": "Compensating Errors",
}


# ---------------------------------------------------------------------------
# Junction error taxonomy stacked bar
# ---------------------------------------------------------------------------


def plot_junction_error_taxonomy(
    df_dd: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: Optional[PlotMetadata] = None,
) -> Optional[plt.Figure]:
    """Stacked bar chart of junction error types per method.

    Parameters
    ----------
    df_dd : pd.DataFrame
        Long-format DataFrame filtered to DIAGNOSTIC_DEPTH rows.
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
    for _, row in df_dd.iterrows():
        if row["metric_key"] in _ERROR_TYPES and isinstance(row["value"], (int, float)):
            rows.append({
                "method_name": row["method_name"],
                "error_type": _ERROR_DISPLAY_NAMES.get(row["metric_key"], row["metric_key"]),
                "count": int(row["value"]),
            })

    if not rows:
        return None

    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot_table(
        index="method_name", columns="error_type",
        values="count", fill_value=0, aggfunc="sum",
    )

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title(f"{class_name} — Junction Error Taxonomy")
    ax.set_xlabel("Method")
    ax.set_ylabel("Count")
    ax.legend(title="Error Type", loc="upper right", fontsize=8)
    fig.tight_layout()

    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig


# ---------------------------------------------------------------------------
# Position bias grouped bar
# ---------------------------------------------------------------------------


def plot_position_bias(
    df_dd: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: Optional[PlotMetadata] = None,
) -> Optional[plt.Figure]:
    """Grouped bar chart of match rate by position zone per method.

    Parameters
    ----------
    df_dd : pd.DataFrame
        Long-format DataFrame filtered to DIAGNOSTIC_DEPTH rows.
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
    zone_keys = {
        "position_bias_5prime_match_rate": "5' (first 25%)",
        "position_bias_interior_match_rate": "Interior (middle 50%)",
        "position_bias_3prime_match_rate": "3' (last 25%)",
    }

    rows = []
    for _, row in df_dd.iterrows():
        if row["metric_key"] in zone_keys and isinstance(row["value"], (int, float)):
            rows.append({
                "method_name": row["method_name"],
                "zone": zone_keys[row["metric_key"]],
                "match_rate": float(row["value"]),
            })

    if not rows:
        return None

    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    sns.barplot(data=plot_df, x="method_name", y="match_rate", hue="zone", ax=ax)
    ax.set_title(f"{class_name} — Position Bias")
    ax.set_xlabel("Method")
    ax.set_ylabel("Match Rate")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Position Zone", loc="lower right", fontsize=9)
    fig.tight_layout()

    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig
