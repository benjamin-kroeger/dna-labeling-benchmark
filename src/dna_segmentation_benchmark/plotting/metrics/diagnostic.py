"""Plotting functions for DIAGNOSTIC_DEPTH metrics.

Provides visualisations for segment length distributions and the
100-bin per-nucleotide mismatch histogram.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import DEFAULT_FIG_SIZE, PlotMetadata
from ..utils import _save_figure, _add_pictogram_panel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position bias histogram (100 bins)
# ---------------------------------------------------------------------------


def plot_position_bias(
    df_dd: pd.DataFrame,
    class_name: str,
    save_path: Optional[Path] = None,
    metadata: Optional[PlotMetadata] = None,
) -> Optional[plt.Figure]:
    """Line chart of per-nucleotide mismatch density across the coding span.

    Each bin represents a 1-percentile slice of the coding region
    (bin 0 = start of first GT coding segment, bin 99 = end of last).
    The y-axis shows the cumulative count of mismatch nucleotides —
    GT positions absent from the prediction (FN) and predicted positions
    absent from GT within the coding span (FP) — summed across all
    evaluated sequences.

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
        if row["metric_key"] == "position_bias_histogram" and isinstance(row["value"], list):
            hist = row["value"]
            if len(hist) == 100:
                rows.append({"method_name": row["method_name"], "histogram": hist})

    if not rows:
        return None

    fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
    x = np.arange(100)

    for entry in rows:
        ax.plot(x, entry["histogram"], label=entry["method_name"], linewidth=1.5)

    ax.set_title(f"{class_name} — Nucleotide Mismatch Location (coding span)")
    ax.set_xlabel("Position in coding span (%)")
    ax.set_ylabel("Mismatch nucleotides (cumulative across sequences)")
    ax.set_xlim(0, 99)
    ax.legend(title="Method", loc="upper right", fontsize=9)
    fig.tight_layout()

    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig
