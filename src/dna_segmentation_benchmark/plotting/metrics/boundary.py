import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm

from ..config import PlotMetadata
from ..utils import _add_pictogram_panel

logger = logging.getLogger(__name__)


def _landscape_frames(landscape: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the bias/reliability DataFrames from the serialisable dict."""
    max_range = landscape["max_range"]
    bias_ticks = np.arange(-max_range, max_range + 1)
    tolerance_ticks = np.arange(max_range + 1)
    bias_matrix = pd.DataFrame(
        np.asarray(landscape["bias_matrix"], dtype=float),
        index=pd.Index(bias_ticks, name="5' Residual (Pred − GT)"),
        columns=pd.Index(bias_ticks, name="3' Residual (Pred − GT)"),
    )
    reliability_matrix = pd.DataFrame(
        np.asarray(landscape["reliability_matrix"], dtype=float),
        index=pd.Index(tolerance_ticks, name="5' Tolerance (bp)"),
        columns=pd.Index(tolerance_ticks, name="3' Tolerance (bp)"),
    )
    return bias_matrix, reliability_matrix


def plot_boundary_precision_landscapes(
    df_fuzzy_boundaries: pd.DataFrame,
    class_name: str,
    max_range: int = 10,
    metadata: PlotMetadata | None = None,
) -> list[plt.Figure]:
    """Plot the two diagnostic matrices to visualize model bias and reliability.

    Each method in *df_fuzzy_boundaries* produces one figure with two
    sub-plots:

    1. **Bias Matrix** — 2-D histogram of signed boundary residuals.
    2. **Reliability Matrix** — cumulative recall surface.

    Each landscape arrives as a JSON-serialisable dict
    (``{max_range, bias_matrix, reliability_matrix}``) and is rebuilt into two
    ``pd.DataFrame`` objects whose index represents the **5' dimension** (rows)
    and whose columns represent the **3' dimension**.  The y-axis is inverted so
    that the lowest value sits at the bottom (standard mathematical orientation).
    """
    figures: list[plt.Figure] = []

    for method in df_fuzzy_boundaries["method_name"].unique().tolist():
        landscape = df_fuzzy_boundaries[df_fuzzy_boundaries["method_name"] == method]["value"].iloc[0]
        max_range = landscape["max_range"]
        bias_matrix, reliability_matrix = _landscape_frames(landscape)

        fig, axes = plt.subplots(1, 2, figsize=(20, 7))

        # --- Plot 1: The Bias Matrix (The "Exon Fingerprint") ---
        sns.heatmap(
            bias_matrix,
            ax=axes[0],
            cmap="YlGnBu",
            norm=LogNorm(vmin=1, vmax=max(bias_matrix.values.max(), 1)),
            cbar_kws={"label": f"Frequency (Number of {class_name} Sections, log scale)"},
        )
        axes[0].set_title(
            f"Boundary Bias Landscape (±{max_range}bp)",
            fontsize=14,
            pad=15,
        )
        # Reconcile the residual sign with the biological edit at each edge:
        # the sign→extension/deletion mapping is opposite between the two edges.
        # 5' edge (rows): residual < 0 → exon starts earlier → extension.
        # 3' edge (cols): residual < 0 → exon ends earlier → deletion.
        axes[0].set_ylabel(f"{bias_matrix.index.name}\n(−) extension     |     deletion (+)", fontsize=12)
        axes[0].set_xlabel(f"{bias_matrix.columns.name}\n(−) deletion     |     extension (+)", fontsize=12)

        # Add crosshairs at the (0,0) perfect-match centre
        axes[0].axvline(max_range + 0.5, color="red", linestyle="--", alpha=0.5)
        axes[0].axhline(max_range + 0.5, color="red", linestyle="--", alpha=0.5)

        # Invert y-axis so the lowest residual is at the bottom
        axes[0].invert_yaxis()

        # --- Plot 2: The Reliability Matrix (The "Tolerance Budget") ---
        sns.heatmap(
            reliability_matrix,
            ax=axes[1],
            cmap="magma",
            annot=True,
            fmt=".2f",
            cbar_kws={"label": f"Recall (Fraction of {class_name} Sections Found)"},
        )
        axes[1].set_title(
            f"Cumulative Recall (0 to {max_range}bp Tolerance)",
            fontsize=14,
            pad=15,
        )
        axes[1].set_ylabel(reliability_matrix.index.name, fontsize=12)
        axes[1].set_xlabel(reliability_matrix.columns.name, fontsize=12)

        # Invert y-axis so tolerance 0 is at the bottom
        axes[1].invert_yaxis()

        fig.suptitle(f"{method}", fontsize=14)
        plt.tight_layout()
        _add_pictogram_panel(fig, metadata, logger=logger)
        figures.append(fig)

    return figures
