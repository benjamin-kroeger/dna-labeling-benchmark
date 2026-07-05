"""Plotting functions for DIAGNOSTIC_DEPTH metrics.

Provides visualisations for segment length distributions and the
100-bin per-nucleotide mismatch histogram.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
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
    save_path: Path | None = None,
    metadata: PlotMetadata | None = None,
) -> plt.Figure | None:
    """Per-nucleotide mismatch density across the coding span, split FN / FP.

    Each bin represents a 1-percentile slice of the coding region
    (bin 0 = start of first GT coding segment, bin 99 = end of last).
    The figure has two subplots so that under-prediction and
    over-prediction can be told apart:

    * **Left** — false negatives (GT coding positions absent from the
      prediction).  Spikes here indicate where the model misses GT bases.
    * **Right** — false positives (predicted coding positions inside the
      GT coding span that are not in GT).  Spikes here indicate where the
      model paints extra bases.

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
    fn_rows: list[dict] = []
    fp_rows: list[dict] = []

    for _, row in df_dd.iterrows():
        value = row["value"]
        if not isinstance(value, list) or len(value) != 100:
            continue
        if row["metric_key"] == "position_bias_histogram_fn":
            fn_rows.append({"method_name": row["method_name"], "histogram": value})
        elif row["metric_key"] == "position_bias_histogram_fp":
            fp_rows.append({"method_name": row["method_name"], "histogram": value})

    if not fn_rows or not fp_rows:
        return None

    fig, axes = plt.subplots(1, 2, figsize=DEFAULT_FIG_SIZE, sharey=True)
    for entry in fn_rows:
        axes[0].plot(entry["histogram"], label=entry["method_name"], linewidth=1.5)
    for entry in fp_rows:
        axes[1].plot(entry["histogram"], label=entry["method_name"], linewidth=1.5)

    axes[0].set_title("False negatives (GT coding missed by prediction)")
    axes[1].set_title("False positives (predicted coding absent from GT)")
    for ax in axes:
        ax.set_xlabel("Position in coding span (%)")
        ax.set_xlim(0, 99)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Mismatch nucleotides (cumulative across sequences)")
    axes[1].legend(title="Method", loc="upper right", fontsize=9)
    title = metadata.display_name if (metadata and metadata.display_name) else "Nucleotide Mismatch Location (coding span)"
    fig.suptitle(f"{title} — {class_name}")

    fig.tight_layout()
    _add_pictogram_panel(fig, metadata, logger)

    if save_path:
        _save_figure(fig, save_path, logger)

    return fig
