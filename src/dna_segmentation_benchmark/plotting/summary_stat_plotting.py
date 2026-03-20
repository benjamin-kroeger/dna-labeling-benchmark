"""Summary-statistics plotting for multi-method benchmark comparisons.

Every public plotting function follows the same contract:

* Returns a :class:`matplotlib.figure.Figure` — ready for ``wandb.log()``,
  ``mlflow.log_figure()``, or any other experiment-tracker.
* Accepts an optional ``save_path`` — when provided the figure is written to
  disk (parent directories are created automatically).
* Never calls ``plt.show()`` — the caller decides when/whether to display.

The orchestrator :func:`compare_multiple_predictions` returns a
``dict[str, Figure]`` mapping descriptive keys to every figure it created.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from .config import PLOT_METADATA
from ..label_definition import LabelConfig, EvalMetrics

# --- Import from new metrics submodules ---
from .metrics.indel import (
    plot_individual_error_lengths_histograms, 
    plot_stacked_indel_counts_bar
)
from .metrics.frameshift import plot_frameshift_percentage_bar
from .metrics.ml import plot_ml_metrics_bar
from .metrics.iou import plot_iou_metrics
from .metrics.boundary import plot_boundary_precision_landscapes
from .metrics.transitions import plot_transition_matrices, plot_false_transitions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compare_multiple_predictions(
        per_method_benchmark_res: dict[str, dict],
        label_config: LabelConfig,
        classes: list[int],
        metrics_to_eval: list[EvalMetrics],
        output_dir: Optional[Path] = None,
) -> dict[str, plt.Figure]:
    """Generate all summary plots and return them as a dict.

    Parameters
    ----------
    per_method_benchmark_res : dict[str, dict]
        Outer key = method name, inner dict = benchmark result as returned
        by :func:`benchmark_gt_vs_pred_multiple`.
    label_config : LabelConfig
        Used to resolve class token → human-readable name.
    classes : list[int]
        Token values that were evaluated.
    metrics_to_eval : list[EvalMetrics]
        Which metric groups were computed.
    output_dir : Path | None
        If provided, every figure is saved as a PNG inside this directory.

    Returns
    -------
    dict[str, Figure]
        Keys are descriptive strings such as
        ``"EXON_indel_counts"``, ``"EXON_indel_lengths"``, etc.
        Suitable for direct ``wandb.log()`` usage.
    """
    # ---- Build long-format DataFrame ------------------------------------
    rows: list[list] = []
    figures: dict[str, plt.Figure] = {}

    # Collect false transition data across all methods for combined plotting
    all_false_transition_data: dict[str, dict] = {}

    for method_name, benchmark_results in per_method_benchmark_res.items():
        transition_matrices = benchmark_results.pop("transition_failures", {})
        fig_transitions = plot_transition_matrices(transition_matrices, label_config, method_name=method_name)
        if fig_transitions is not None:
             figures[f"{method_name}_transition_matrices"] = fig_transitions

        all_false_transition_data[method_name] = benchmark_results.pop("false_transitions", {})
        
        for class_name, metric_groupings in benchmark_results.items():
            class_name_str = class_name if isinstance(class_name, str) else str(class_name)
            for metric_group, metric_data in metric_groupings.items():
                metric_group_str = (
                    metric_group if isinstance(metric_group, str) else metric_group.name
                )
                for single_metric_key, value in metric_data.items():
                    rows.append([
                        method_name,
                        class_name_str,
                        metric_group_str,
                        single_metric_key,
                        value,
                    ])

    # ---- Combined false-transition plot (all methods) --------------------
    fig_false = plot_false_transitions(all_false_transition_data, label_config)
    if fig_false is not None:
        figures["false_transitions"] = fig_false

    if not rows:
        logger.warning("No benchmark data collected — nothing to plot.")
        return figures

    df = pd.DataFrame(
        rows, columns=["method_name", "measured_class", "metric_group", "metric_key", "value"],
    )

    for class_token in classes:
        class_name = label_config.name_of(class_token)

        # ---- INDEL plots ------------------------------------------------
        if EvalMetrics.INDEL in metrics_to_eval:
            df_indel = df[
                (df["measured_class"] == class_name)
                & (df["metric_group"] == EvalMetrics.INDEL.name)
                ].copy()

            if not df_indel.empty:
                fig = plot_stacked_indel_counts_bar(
                    df_indel, class_name,
                    save_path=(output_dir / f"{class_name}_indel_counts.png") if output_dir else None,
                    metadata=PLOT_METADATA.get("indel_counts"),
                )
                if fig is not None:
                    figures[f"{class_name}_indel_counts"] = fig

                fig = plot_individual_error_lengths_histograms(
                    df_indel, class_name,
                    save_path=(output_dir / f"{class_name}_indel_lengths.png") if output_dir else None,
                )
                if fig is not None:
                    figures[f"{class_name}_indel_lengths"] = fig

        # ---- Fuzzy boundary landscape plots (from BOUNDARY_EXACTNESS) ----
        if EvalMetrics.BOUNDARY_EXACTNESS in metrics_to_eval:
            df_fuzzy = df[
                (df["measured_class"] == class_name)
                & (df["metric_group"] == EvalMetrics.BOUNDARY_EXACTNESS.name)
                & (df["metric_key"] == "fuzzy_metrics")
                ].copy()

            df = df[df["metric_key"] != "fuzzy_metrics"]

            fuzzy_metrics_figs = plot_boundary_precision_landscapes(
                df_fuzzy,
                metadata=PLOT_METADATA.get("fuzzy_metrics"),
            )
            for i, fig in enumerate(fuzzy_metrics_figs):
                figures[f"{i}_fuzzy_metrics"] = fig

        # ---- IoU plots (from BOUNDARY_EXACTNESS) -------------------------
        if EvalMetrics.BOUNDARY_EXACTNESS in metrics_to_eval:
            df_iou = df[
                (df["measured_class"] == class_name)
                & (df["metric_group"] == EvalMetrics.BOUNDARY_EXACTNESS.name)
                & (df["metric_key"] == "iou_scores")
                ].copy()

            if not df_iou.empty:
                prefix = (output_dir / f"{class_name}_iou") if output_dir else None
                iou_figs = plot_iou_metrics(
                    df_iou, class_name,
                    save_path_prefix=prefix,
                    metadata_average=PLOT_METADATA.get("iou_average"),
                    metadata_distribution=PLOT_METADATA.get("iou_distribution"),
                )
                for idx, fig in enumerate(iou_figs):
                    suffix = "average" if idx == 0 else "distribution"
                    figures[f"{class_name}_iou_{suffix}"] = fig

        # ---- Region-discovery bar plots -----------------------------------
        if EvalMetrics.REGION_DISCOVERY in metrics_to_eval:
            df_rd = df[
                (df["measured_class"] == class_name)
                & (df["metric_group"] == EvalMetrics.REGION_DISCOVERY.name)
                ].copy()

            if not df_rd.empty:
                prefix = (output_dir / f"{class_name}_region_discovery") if output_dir else None
                rd_figs = plot_ml_metrics_bar(
                    df_rd, class_name,
                    save_path_prefix=prefix,
                    metadata_map=PLOT_METADATA,
                )
                for idx, fig in enumerate(rd_figs):
                    figures[f"{class_name}_region_discovery_{idx}"] = fig

        # ---- Nucleotide-classification bar plots --------------------------
        if EvalMetrics.NUCLEOTIDE_CLASSIFICATION in metrics_to_eval:
            df_nc = df[
                (df["measured_class"] == class_name)
                & (df["metric_group"] == EvalMetrics.NUCLEOTIDE_CLASSIFICATION.name)
                ].copy()

            if not df_nc.empty:
                prefix = (output_dir / f"{class_name}_nucleotide_classification") if output_dir else None
                nc_figs = plot_ml_metrics_bar(
                    df_nc, class_name,
                    save_path_prefix=prefix,
                    metadata_map=PLOT_METADATA,
                )
                for idx, fig in enumerate(nc_figs):
                    figures[f"{class_name}_nucleotide_classification_{idx}"] = fig

        # ---- Frameshift plots -------------------------------------------
        if EvalMetrics.FRAMESHIFT in metrics_to_eval:
            df_fs = df[
                (df["measured_class"] == class_name)
                & (df["metric_group"] == EvalMetrics.FRAMESHIFT.name)
                ].copy()

            if not df_fs.empty:
                fig = plot_frameshift_percentage_bar(
                    df_fs, class_name,
                    save_path=(output_dir / f"{class_name}_frameshift.png") if output_dir else None,
                    metadata=PLOT_METADATA.get("frameshift"),
                )
                if fig is not None:
                    figures[f"{class_name}_frameshift"] = fig

    return figures
