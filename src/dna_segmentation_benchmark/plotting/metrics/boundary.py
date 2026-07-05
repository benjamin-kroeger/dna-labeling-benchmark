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
        index=pd.Index(tolerance_ticks, name="5' Tolerance ±(bp)"),
        columns=pd.Index(tolerance_ticks, name="3' Tolerance ±(bp)"),
    )
    return bias_matrix, reliability_matrix


def plot_boundary_precision_landscapes(
    df_fuzzy_boundaries: pd.DataFrame,
    class_name: str,
    max_range: int = 10,
    bias_metadata: PlotMetadata | None = None,
    recall_metadata: PlotMetadata | None = None,
) -> list[plt.Figure]:
    """Plot the two diagnostic matrices to visualize model bias and reliability.

    Returns **two figures grouped by metric** (small multiples), so methods can
    be compared side by side against the same ground truth:

    1. **Bias figure** — one subplot per method, each a 2-D histogram of signed
       boundary residuals; all subplots share one raw-count log color scale.
    2. **Reliability figure** — one subplot per method, each a cumulative recall
       surface on a shared 0–1 color scale.

    Each landscape arrives as a JSON-serialisable dict
    (``{max_range, bias_matrix, reliability_matrix}``) and is rebuilt into two
    ``pd.DataFrame`` objects whose index represents the **5' dimension** (rows)
    and whose columns represent the **3' dimension**.  The y-axis is inverted so
    that the lowest value sits at the bottom (standard mathematical orientation).
    """
    methods = df_fuzzy_boundaries["method_name"].unique().tolist()
    if not methods:
        return []

    # Rebuild every method's landscape once: (method, max_range, bias, reliability).
    landscapes = []
    for method in methods:
        landscape = df_fuzzy_boundaries[df_fuzzy_boundaries["method_name"] == method]["value"].iloc[0]
        bias_matrix, reliability_matrix = _landscape_frames(landscape)
        landscapes.append((method, landscape["max_range"], bias_matrix, reliability_matrix))

    ncols = min(len(methods), 4)
    nrows = int(np.ceil(len(methods) / ncols))
    max_range = landscapes[0][1]

    # --- Figure 1: Bias landscapes, one subplot per method ---
    # Shared raw-count log scale so intensities are comparable across methods
    # (all methods are scored against the same ground truth).
    global_bias_max = max(bias.values.max() for _, _, bias, _ in landscapes)
    bias_norm = LogNorm(vmin=1, vmax=max(global_bias_max, 1))

    fig_bias, axes_bias = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6 * nrows), squeeze=False)
    flat_bias = axes_bias.flatten()
    bias_mappable = None
    for i, (ax, (method, mr, bias_matrix, _)) in enumerate(zip(flat_bias, landscapes)):
        sns.heatmap(bias_matrix, ax=ax, cmap="YlGnBu", norm=bias_norm, cbar=False)
        bias_mappable = ax.collections[0]
        ax.set_title(method, fontsize=12)
        ax.axvline(mr + 0.5, color="red", linestyle="--", alpha=0.5)
        ax.axhline(mr + 0.5, color="red", linestyle="--", alpha=0.5)
        ax.invert_yaxis()
        # Small-multiples: label only the outer edges so inner tick labels
        # don't clip. Reconcile the residual sign with the biological edit at
        # each edge: the sign→extension/deletion mapping is opposite between
        # the two edges. 5' edge (rows): residual < 0 → exon starts earlier →
        # extension. 3' edge (cols): residual < 0 → exon ends earlier → deletion.
        if i % ncols == 0:
            ax.set_ylabel(f"{bias_matrix.index.name}\n(−) extension     |     deletion (+)", fontsize=10)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        if i + ncols >= len(landscapes):
            ax.set_xlabel(f"{bias_matrix.columns.name}\n(−) deletion     |     extension (+)", fontsize=10)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
    for ax in flat_bias[len(landscapes):]:
        ax.axis("off")
    fig_bias.suptitle(f"Boundary Bias — {class_name} (±{max_range} bp)", fontsize=15)
    fig_bias.tight_layout(rect=(0, 0, 1, 0.96))
    if bias_mappable is not None:
        fig_bias.colorbar(
            bias_mappable,
            ax=list(flat_bias[: len(landscapes)]),
            fraction=0.025,
            pad=0.02,
            label=f"Frequency (Number of {class_name} Sections, log scale)",
        )
    _add_pictogram_panel(fig_bias, bias_metadata, logger=logger)

    # --- Figure 2: Cumulative recall, one subplot per method (shared 0–1 scale) ---
    fig_rel, axes_rel = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 6 * nrows), squeeze=False)
    flat_rel = axes_rel.flatten()
    rel_mappable = None
    for i, (ax, (method, _, _, reliability_matrix)) in enumerate(zip(flat_rel, landscapes)):
        sns.heatmap(
            reliability_matrix, ax=ax, cmap="magma", vmin=0, vmax=1,
            annot=True, fmt=".2f", cbar=False,
        )
        rel_mappable = ax.collections[0]
        ax.set_title(method, fontsize=12)
        ax.invert_yaxis()
        # Small-multiples: label only the outer edges to avoid clipping.
        if i % ncols == 0:
            ax.set_ylabel(reliability_matrix.index.name, fontsize=10)
        else:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
        if i + ncols >= len(landscapes):
            ax.set_xlabel(reliability_matrix.columns.name, fontsize=10)
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
    for ax in flat_rel[len(landscapes):]:
        ax.axis("off")
    fig_rel.suptitle(f"Cumulative Recall with Relaxed Boundaries — {class_name} (0–{max_range} bp)", fontsize=15)
    fig_rel.tight_layout(rect=(0, 0, 1, 0.96))
    if rel_mappable is not None:
        fig_rel.colorbar(
            rel_mappable,
            ax=list(flat_rel[: len(landscapes)]),
            fraction=0.025,
            pad=0.02,
            label=f"Recall (Fraction of {class_name} Sections Found)",
        )
    _add_pictogram_panel(fig_rel, recall_metadata, logger=logger)

    return [fig_bias, fig_rel]
