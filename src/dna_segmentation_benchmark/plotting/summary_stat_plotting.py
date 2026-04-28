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
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from .config import PLOT_METADATA
from ..label_definition import LabelConfig, EvalMetrics

# --- Import from new metrics submodules ---
from .metrics.indel import plot_individual_error_lengths_histograms, plot_stacked_indel_counts_bar
from .metrics.frameshift import plot_frameshift_percentage_bar
from .metrics.ml import plot_ml_metrics_bar
from .metrics.iou import plot_iou_metrics
from .metrics.boundary import plot_boundary_precision_landscapes
from .metrics.diagnostic import plot_position_bias
from .metrics.structural import (
    plot_transcript_match_distribution,
    plot_segment_count_delta,
    plot_boundary_shift_distribution,
    plot_per_transcript_soft_exon_metrics,
)
from .metrics.transitions import plot_transition_matrices, plot_false_transitions
from .utils import _save_figure

logger = logging.getLogger(__name__)


def _slugify_plot_token(value: str) -> str:
    """Convert a method name into a filesystem-safe token."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compare_multiple_predictions(
        per_method_benchmark_res: dict[str, dict],
        label_config: LabelConfig,
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
    metrics_to_eval : list[EvalMetrics]
        Which metric groups were computed.
    output_dir : Path | None
        If provided, every figure is saved as a PNG inside this directory.

    Returns
    -------
    dict[str, Figure]
        Keys are descriptive strings such as ``"indel_counts"``,
        ``"indel_lengths"``, etc.
        Suitable for direct ``wandb.log()`` usage.
    """
    # ---- Build long-format DataFrame ------------------------------------
    rows: list[list] = []
    figures: dict[str, plt.Figure] = {}

    # Collect false transition data across all methods for combined plotting
    all_false_transition_data: dict[str, dict] = {}
    single_method_mode = len(per_method_benchmark_res) == 1

    for method_name, benchmark_results in per_method_benchmark_res.items():
        if set(benchmark_results.keys()) == {"per_transcript", "global"}:
            benchmark_results = benchmark_results["per_transcript"]

        benchmark_results = dict(benchmark_results)

        transition_matrices = benchmark_results.pop("transition_failures", {})
        fig_transitions = plot_transition_matrices(transition_matrices, label_config, method_name=method_name)
        if fig_transitions is not None:
            key = "transition_matrices" if single_method_mode else f"{method_name}_transition_matrices"
            figures[key] = fig_transitions

        all_false_transition_data[method_name] = benchmark_results.pop("false_transitions", {})

        for metric_group, metric_data in benchmark_results.items():
            metric_group_str = metric_group if isinstance(metric_group, str) else metric_group.name
            for single_metric_key, value in metric_data.items():
                rows.append(
                    [
                        method_name,
                        metric_group_str,
                        single_metric_key,
                        value,
                    ]
                )

    # ---- Combined false-transition plot (all methods) --------------------
    fig_false = plot_false_transitions(all_false_transition_data, label_config)
    if fig_false is not None:
        figures["false_transitions"] = fig_false

    if not rows:
        logger.warning("No benchmark data collected — nothing to plot.")
        return figures

    df = pd.DataFrame(
        rows,
        columns=["method_name", "metric_group", "metric_key", "value"],
    )

    class_name = label_config.name_of(label_config.coding_label)

    # ---- INDEL plots ------------------------------------------------
    if EvalMetrics.INDEL in metrics_to_eval:
        df_indel = df[(df["metric_group"] == EvalMetrics.INDEL.name)].copy()

        if not df_indel.empty:
            fig = plot_stacked_indel_counts_bar(
                df_indel,
                class_name,
                save_path=(output_dir / "indel_counts.png") if output_dir else None,
                metadata=PLOT_METADATA.get("indel_counts"),
            )
            if fig is not None:
                figures["indel_counts"] = fig

            fig = plot_individual_error_lengths_histograms(
                df_indel,
                class_name,
                save_path=(output_dir / "indel_lengths.png") if output_dir else None,
            )
            if fig is not None:
                figures["indel_lengths"] = fig

    # ---- Fuzzy boundary landscape plots (from BOUNDARY_EXACTNESS) ----
    if EvalMetrics.BOUNDARY_EXACTNESS in metrics_to_eval:
        df_fuzzy = df[
            (df["metric_group"] == EvalMetrics.BOUNDARY_EXACTNESS.name) & (df["metric_key"] == "fuzzy_metrics")
            ].copy()

        df = df[df["metric_key"] != "fuzzy_metrics"]

        fuzzy_metrics_figs = plot_boundary_precision_landscapes(
            df_fuzzy,
            class_name=class_name,
            metadata=PLOT_METADATA.get("boundary_landscape"),
        )
        for method_name, fig in zip(df_fuzzy["method_name"].unique().tolist(), fuzzy_metrics_figs):
            key = "boundary_landscape" if single_method_mode else f"{method_name}_boundary_landscape"
            if output_dir is not None:
                filename = (
                    "boundary_landscape.png"
                    if single_method_mode
                    else f"boundary_landscape_{_slugify_plot_token(method_name)}.png"
                )
                _save_figure(fig, output_dir / filename, logger=logger)
            figures[key] = fig

    # ---- IoU plots (from BOUNDARY_EXACTNESS) -------------------------
    if EvalMetrics.BOUNDARY_EXACTNESS in metrics_to_eval:
        df_iou = df[
            (df["metric_group"] == EvalMetrics.BOUNDARY_EXACTNESS.name) & (df["metric_key"] == "iou_scores")
            ].copy()

        if not df_iou.empty:
            prefix = (output_dir / "iou") if output_dir else None
            iou_figs = plot_iou_metrics(
                df_iou,
                class_name,
                save_path_prefix=prefix,
                metadata_average=PLOT_METADATA.get("iou_average"),
                metadata_distribution=PLOT_METADATA.get("iou_distribution"),
            )
            for idx, fig in enumerate(iou_figs):
                suffix = "average" if idx == 0 else "distribution"
                figures[f"iou_{suffix}"] = fig

    # ---- Region-discovery bar plots -----------------------------------
    if EvalMetrics.REGION_DISCOVERY in metrics_to_eval:
        df_rd = df[(df["metric_group"] == EvalMetrics.REGION_DISCOVERY.name)].copy()

        if not df_rd.empty:
            prefix = (output_dir / "region_discovery") if output_dir else None
            rd_figs = plot_ml_metrics_bar(
                df_rd,
                class_name,
                save_path_prefix=prefix,
                metadata_map=PLOT_METADATA,
            )
            for idx, fig in enumerate(rd_figs):
                figures[f"region_discovery_{idx}"] = fig

    # ---- Nucleotide-classification bar plots --------------------------
    if EvalMetrics.NUCLEOTIDE_CLASSIFICATION in metrics_to_eval:
        df_nc = df[(df["metric_group"] == EvalMetrics.NUCLEOTIDE_CLASSIFICATION.name)].copy()

        if not df_nc.empty:
            prefix = (output_dir / "nucleotide_classification") if output_dir else None
            nc_figs = plot_ml_metrics_bar(
                df_nc,
                class_name,
                save_path_prefix=prefix,
                metadata_map=PLOT_METADATA,
            )
            for idx, fig in enumerate(nc_figs):
                figures[f"nucleotide_classification_{idx}"] = fig

    # ---- Frameshift plots -------------------------------------------
    if EvalMetrics.FRAMESHIFT in metrics_to_eval:
        df_fs = df[(df["metric_group"] == EvalMetrics.FRAMESHIFT.name)].copy()

        if not df_fs.empty:
            fig = plot_frameshift_percentage_bar(
                df_fs,
                class_name,
                save_path=(output_dir / "frameshift.png") if output_dir else None,
                metadata=PLOT_METADATA.get("frameshift"),
            )
            if fig is not None:
                figures["frameshift"] = fig

    # ---- Structural coherence plots ------------------------------------
    if EvalMetrics.STRUCTURAL_COHERENCE in metrics_to_eval:
        df_sc = df[(df["metric_group"] == EvalMetrics.STRUCTURAL_COHERENCE.name)].copy()

        # Combined precision / recall overview — one figure per measure,
        # reusing plot_ml_metrics_bar (x = metric, hue = method).
        _PR_KEYS = ("intron_chain", "intron_chain_subset", "intron_chain_superset", "exon_chain", "exon_chain_superset", "exon_chain_subset")
        _PR_DISPLAY = {
            "intron_chain": "Exact intron chain",
            "intron_chain_subset": "Intron Subset",
            "intron_chain_superset": "Intron Superset",
            "exon_chain": "Exact exon chain",
            "exon_chain_superset": "Exon Superset",
            "exon_chain_subset": "Exon Subset",
        }
        _method_scores: dict[str, dict] = {}
        for _, _row in df_sc.iterrows():
            if _row["metric_key"] in _PR_KEYS and isinstance(_row["value"], dict):
                _method_scores.setdefault(_row["method_name"], {})[_row["metric_key"]] = _row["value"]

        if _method_scores:
            _pr_rows = []
            for _method, _scores in _method_scores.items():
                for _measure in ("precision", "recall"):
                    _combined = {
                        _PR_DISPLAY[k]: v.get(_measure, 0.0) for k, v in _scores.items() if isinstance(v, dict)
                    }
                    _pr_rows.append(
                        {
                            "method_name": _method,
                            "metric_group": EvalMetrics.STRUCTURAL_COHERENCE.name,
                            "metric_key": "ts_level_" + _measure,
                            "value": _combined,
                        }
                    )
            _df_pr = pd.DataFrame(_pr_rows)
            _prefix = (output_dir / "transcript_pr_overview") if output_dir else None
            for _idx, _fig in enumerate(
                    plot_ml_metrics_bar(_df_pr, class_name, save_path_prefix=_prefix, metadata_map=PLOT_METADATA)
            ):
                figures[f"transcript_pr_overview_{_idx}"] = _fig

        # Transcript match class distribution with count annotations
        fig = plot_transcript_match_distribution(
            df_sc,
            class_name,
            save_path=(output_dir / "transcript_match.png") if output_dir else None,
            metadata=PLOT_METADATA.get("transcript_match"),
        )
        if fig is not None:
            figures["transcript_match"] = fig

        # Segment count delta per model
        fig = plot_segment_count_delta(
            df_sc,
            class_name,
            save_path=(output_dir / "segment_count_delta.png") if output_dir else None,
            metadata=PLOT_METADATA.get("segment_count_delta"),
        )
        if fig is not None:
            figures["segment_count_delta"] = fig

        # Boundary shift distribution (histograms + scatter)
        fig = plot_boundary_shift_distribution(
            df_sc,
            class_name,
            save_path=(output_dir / "boundary_shift_dist.png") if output_dir else None,
            metadata=PLOT_METADATA.get("boundary_shift_distribution"),
        )
        if fig is not None:
            figures["boundary_shift_dist"] = fig

        # Per-transcript continuous exon recovery + hallucinated exon count
        fig = plot_per_transcript_soft_exon_metrics(
            df_sc,
            class_name,
            save_path=(output_dir / "per_transcript_soft_exon.png") if output_dir else None,
            metadata=PLOT_METADATA.get("per_transcript_soft_exon"),
        )
        if fig is not None:
            figures["per_transcript_soft_exon"] = fig

    # ---- Diagnostic depth plots ----------------------------------------
    if EvalMetrics.DIAGNOSTIC_DEPTH in metrics_to_eval:
        df_dd = df[(df["metric_group"] == EvalMetrics.DIAGNOSTIC_DEPTH.name)].copy()

        if not df_dd.empty:
            fig = plot_position_bias(
                df_dd,
                class_name,
                save_path=(output_dir / "position_bias.png") if output_dir else None,
                metadata=PLOT_METADATA.get("position_bias"),
            )
            if fig is not None:
                figures["position_bias"] = fig

    return figures
