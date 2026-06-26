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
from .metrics.indel import (
    plot_individual_error_lengths_histograms,
    plot_indel_counts_by_boundary,
    plot_indel_rates_by_boundary,
    plot_stacked_indel_counts_bar,
)
from .metrics.phase_drift import plot_phase_drift_percentage_bar
from .metrics.ml import plot_ml_metrics_bar
from .metrics.iou import plot_iou_metrics
from .metrics.boundary import plot_boundary_precision_landscapes
from .metrics.diagnostic import plot_position_bias
from .metrics.structural import (
    plot_transcript_match_distribution,
    plot_segment_count_delta,
    plot_boundary_shift_distribution,
    plot_per_transcript_exon_recovery,
)
from .metrics.transitions import plot_transition_matrices, plot_false_transitions
from .metrics.splice_sites import plot_splice_site_confusion_matrices, plot_splice_site_pr_bar
from .utils import _save_figure

logger = logging.getLogger(__name__)


def _slugify_plot_token(value: str) -> str:
    """Convert a method name into a filesystem-safe token."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _scope_display_name(label_config: LabelConfig, scope: str | None) -> str:
    """Return a human-readable target name for one scope."""
    if scope is None:
        scope = label_config.evaluation_scope.value
    if scope == "transcript_exon":
        if label_config.annotation_mode.name == "EXON_INTRON":
            return label_config.name_of(label_config.exon_label)
        return "TRANSCRIPT_EXON"
    if scope == "cds":
        return "CDS"
    if scope == "intron":
        return "INTRON"
    return scope.upper()


def _metric_scopes(df_metric: pd.DataFrame) -> list[str | None]:
    """Return ordered scope values for one metric subset."""
    if "scope" not in df_metric.columns:
        return [None]
    scopes = [scope for scope in df_metric["scope"].dropna().unique().tolist()]
    if not scopes:
        return [None]
    ordered: list[str | None] = []
    if "transcript_exon" in scopes:
        ordered.append("transcript_exon")
        scopes.remove("transcript_exon")
    ordered.extend(sorted(scopes))
    return ordered


def _figure_key(base: str, scope: str | None, default_scope: str | None) -> str:
    """Build a stable figure key, preserving legacy names for the default scope."""
    if scope is None or scope == default_scope:
        return base
    return f"{base}_{scope}"


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
    all_splice_site_data: dict[str, dict] = {}
    # INDEL is boundary-typed (a nested by_boundary dict), so it does not fit the
    # flat long-format rows; collect it per method and plot it directly.
    all_indel_data: dict[str, dict] = {}
    single_method_mode = len(per_method_benchmark_res) == 1

    for method_name, benchmark_results in per_method_benchmark_res.items():
        if set(benchmark_results.keys()) == {"aggregated", "global"}:
            benchmark_results = benchmark_results["aggregated"]

        benchmark_results = dict(benchmark_results)

        # Pop transition keys unconditionally so they never leak into the generic
        # long-format rows loop; only plot them when STATE_TRANSITIONS was run.
        transition_matrices = benchmark_results.pop("transition_failures", {})
        false_transitions = benchmark_results.pop("false_transitions", {})
        if EvalMetrics.STATE_TRANSITIONS in metrics_to_eval:
            fig_transitions = plot_transition_matrices(transition_matrices, label_config, method_name=method_name)
            if fig_transitions is not None:
                key = "transition_matrices" if single_method_mode else f"{method_name}_transition_matrices"
                figures[key] = fig_transitions
            all_false_transition_data[method_name] = false_transitions

        for metric_group, metric_data in benchmark_results.items():
            metric_group_str = metric_group if isinstance(metric_group, str) else metric_group.name
            if metric_group_str == "metadata":
                continue
            # INDEL: nested by_boundary payload + opportunity denominators,
            # plotted directly (not via the flat rows).
            if metric_group_str == EvalMetrics.INDEL.name:
                all_indel_data[method_name] = metric_data
                continue
            # STRUCTURAL_COHERENCE nests chain metrics and (optionally) splice
            # sites under one group. Pull the splice sites out for their own
            # plots and flatten the chain metrics into the long-format rows.
            if metric_group_str == EvalMetrics.STRUCTURAL_COHERENCE.name:
                splice = metric_data.get("splice_site_results")
                if splice:
                    all_splice_site_data[method_name] = splice
                for scope, scope_payload in metric_data.get("scopes", {}).items():
                    for single_metric_key, value in scope_payload.items():
                        rows.append([method_name, metric_group_str, scope, single_metric_key, value])
                for single_metric_key, value in metric_data.items():
                    if single_metric_key in {"scopes", "splice_site_results"}:
                        continue
                    rows.append([method_name, metric_group_str, "intron", single_metric_key, value])
                continue
            if isinstance(metric_data, dict) and "scopes" in metric_data:
                for scope, scope_payload in metric_data["scopes"].items():
                    for single_metric_key, value in scope_payload.items():
                        rows.append([method_name, metric_group_str, scope, single_metric_key, value])
                continue
            for single_metric_key, value in metric_data.items():
                rows.append([method_name, metric_group_str, None, single_metric_key, value])

    # ---- Combined false-transition plot (all methods) --------------------
    if EvalMetrics.STATE_TRANSITIONS in metrics_to_eval:
        fig_false = plot_false_transitions(all_false_transition_data, label_config)
        if fig_false is not None:
            figures["false_transitions"] = fig_false

    # ---- Splice-site plots (all methods) --------------------------------
    if all_splice_site_data:
        fig_ss_cm = plot_splice_site_confusion_matrices(
            all_splice_site_data,
            save_path=(output_dir / "splice_site_confusion.png") if output_dir else None,
        )
        if fig_ss_cm is not None:
            figures["splice_site_confusion"] = fig_ss_cm

        fig_ss_pr = plot_splice_site_pr_bar(
            all_splice_site_data,
            save_path=(output_dir / "splice_site_pr.png") if output_dir else None,
        )
        if fig_ss_pr is not None:
            figures["splice_site_pr"] = fig_ss_pr

    # ---- INDEL plots (boundary-typed; plotted directly from the payloads) ----
    if EvalMetrics.INDEL in metrics_to_eval and any(
        isinstance(p, dict) and p.get("by_boundary") for p in all_indel_data.values()
    ):
        class_name = _scope_display_name(label_config, None)
        fig = plot_stacked_indel_counts_bar(
            all_indel_data,
            class_name,
            save_path=(output_dir / "indel_counts.png") if output_dir else None,
            metadata=PLOT_METADATA.get("indel_counts"),
        )
        if fig is not None:
            figures["indel_counts"] = fig

        fig = plot_indel_rates_by_boundary(
            all_indel_data,
            class_name,
            save_path=(output_dir / "indel_rates_by_boundary.png") if output_dir else None,
            metadata=PLOT_METADATA.get("indel_rates_by_boundary"),
        )
        if fig is not None:
            figures["indel_rates_by_boundary"] = fig

        fig = plot_indel_counts_by_boundary(
            all_indel_data,
            class_name,
            save_path=(output_dir / "indel_counts_by_boundary.png") if output_dir else None,
            metadata=PLOT_METADATA.get("indel_counts_by_boundary"),
        )
        if fig is not None:
            figures["indel_counts_by_boundary"] = fig

        for boundary, fig in plot_individual_error_lengths_histograms(
            all_indel_data,
            class_name,
            save_dir=output_dir,
        ).items():
            figures[f"indel_lengths_{boundary}"] = fig

    if not rows:
        logger.warning("No benchmark data collected — nothing to plot.")
        return figures

    df = pd.DataFrame(
        rows,
        columns=["method_name", "metric_group", "scope", "metric_key", "value"],
    )

    # ---- Fuzzy boundary landscape plots (from BOUNDARY_EXACTNESS) ----
    if EvalMetrics.BOUNDARY_EXACTNESS in metrics_to_eval:
        df_fuzzy = df[
            (df["metric_group"] == EvalMetrics.BOUNDARY_EXACTNESS.name) & (df["metric_key"] == "fuzzy_metrics")
            ].copy()

        df = df[df["metric_key"] != "fuzzy_metrics"]

        default_scope = _metric_scopes(df_fuzzy)[0]
        for scope in _metric_scopes(df_fuzzy):
            df_scope = df_fuzzy[df_fuzzy["scope"] == scope] if scope is not None else df_fuzzy[df_fuzzy["scope"].isna()]
            class_name = _scope_display_name(label_config, scope)
            fuzzy_metrics_figs = plot_boundary_precision_landscapes(
                df_scope,
                class_name=class_name,
                metadata=PLOT_METADATA.get("boundary_landscape"),
            )
            for method_name, fig in zip(df_scope["method_name"].unique().tolist(), fuzzy_metrics_figs):
                base_key = "boundary_landscape" if single_method_mode else f"{method_name}_boundary_landscape"
                key = _figure_key(base_key, scope, default_scope)
                if output_dir is not None:
                    filename = (
                        f"{base_key}{'' if scope == default_scope else f'_{scope}'}.png"
                    )
                    _save_figure(fig, output_dir / filename, logger=logger)
                figures[key] = fig

    # ---- IoU plots (from BOUNDARY_EXACTNESS) -------------------------
    if EvalMetrics.BOUNDARY_EXACTNESS in metrics_to_eval:
        df_iou = df[
            (df["metric_group"] == EvalMetrics.BOUNDARY_EXACTNESS.name) & (df["metric_key"] == "iou_scores")
            ].copy()

        if not df_iou.empty:
            default_scope = _metric_scopes(df_iou)[0]
            for scope in _metric_scopes(df_iou):
                df_scope = df_iou[df_iou["scope"] == scope] if scope is not None else df_iou[df_iou["scope"].isna()]
                class_name = _scope_display_name(label_config, scope)
                prefix = (output_dir / f"iou{'' if scope == default_scope else f'_{scope}'}") if output_dir else None
                iou_figs = plot_iou_metrics(
                    df_scope,
                    class_name,
                    save_path_prefix=prefix,
                    metadata_average=PLOT_METADATA.get("iou_average"),
                    metadata_distribution=PLOT_METADATA.get("iou_distribution"),
                )
                for idx, fig in enumerate(iou_figs):
                    suffix = "average" if idx == 0 else "distribution"
                    figures[_figure_key(f"iou_{suffix}", scope, default_scope)] = fig

    # ---- Region-discovery bar plots -----------------------------------
    if EvalMetrics.REGION_DISCOVERY in metrics_to_eval:
        df_rd = df[(df["metric_group"] == EvalMetrics.REGION_DISCOVERY.name)].copy()

        if not df_rd.empty:
            default_scope = _metric_scopes(df_rd)[0]
            for scope in _metric_scopes(df_rd):
                df_scope = df_rd[df_rd["scope"] == scope] if scope is not None else df_rd[df_rd["scope"].isna()]
                class_name = _scope_display_name(label_config, scope)
                prefix = (output_dir / f"region_discovery{'' if scope == default_scope else f'_{scope}'}") if output_dir else None
                rd_figs = plot_ml_metrics_bar(
                    df_scope,
                    class_name,
                    save_path_prefix=prefix,
                    metadata_map=PLOT_METADATA,
                )
                for level, fig in rd_figs.items():
                    figures[_figure_key(f"region_discovery_{level}", scope, default_scope)] = fig

    # ---- Nucleotide-classification bar plots --------------------------
    if EvalMetrics.NUCLEOTIDE_CLASSIFICATION in metrics_to_eval:
        df_nc = df[(df["metric_group"] == EvalMetrics.NUCLEOTIDE_CLASSIFICATION.name)].copy()

        if not df_nc.empty:
            default_scope = _metric_scopes(df_nc)[0]
            for scope in _metric_scopes(df_nc):
                df_scope = df_nc[df_nc["scope"] == scope] if scope is not None else df_nc[df_nc["scope"].isna()]
                class_name = _scope_display_name(label_config, scope)
                prefix = (output_dir / f"nucleotide_classification{'' if scope == default_scope else f'_{scope}'}") if output_dir else None
                nc_figs = plot_ml_metrics_bar(
                    df_scope,
                    class_name,
                    save_path_prefix=prefix,
                    metadata_map=PLOT_METADATA,
                )
                for level, fig in nc_figs.items():
                    figures[_figure_key(f"nucleotide_classification_{level}", scope, default_scope)] = fig

    # ---- Phase-drift plots -------------------------------------------
    if EvalMetrics.PHASE_DRIFT in metrics_to_eval:
        df_fs = df[(df["metric_group"] == EvalMetrics.PHASE_DRIFT.name)].copy()

        if not df_fs.empty:
            default_scope = _metric_scopes(df_fs)[0]
            for scope in _metric_scopes(df_fs):
                df_scope = df_fs[df_fs["scope"] == scope] if scope is not None else df_fs[df_fs["scope"].isna()]
                class_name = _scope_display_name(label_config, scope)
                suffix = "" if scope == default_scope else f"_{scope}"
                fig = plot_phase_drift_percentage_bar(
                    df_scope,
                    class_name,
                    save_path=(output_dir / f"phase_drift{suffix}.png") if output_dir else None,
                    metadata=PLOT_METADATA.get("phase_drift"),
                )
                if fig is not None:
                    figures[_figure_key("phase_drift", scope, default_scope)] = fig

    # ---- Structural coherence plots ------------------------------------
    if EvalMetrics.STRUCTURAL_COHERENCE in metrics_to_eval:
        df_sc = df[(df["metric_group"] == EvalMetrics.STRUCTURAL_COHERENCE.name)].copy()
        structural_scopes = [scope for scope in _metric_scopes(df_sc) if scope != "intron"]
        default_scope = structural_scopes[0] if structural_scopes else None

        for scope in structural_scopes or [None]:
            df_sc_scope = df_sc[
                (df_sc["scope"] == "intron")
                | ((df_sc["scope"] == scope) if scope is not None else df_sc["scope"].isna())
            ].copy()
            class_name = _scope_display_name(label_config, scope)

            # Combined match-rate overview — one figure (x = tier, hue = method),
            # reusing plot_ml_metrics_bar.
            _PR_KEYS = ("intron_chain", "intron_chain_subset", "intron_chain_superset", "exon_chain", "exon_chain_superset", "exon_chain_subset")
            # These are whole-chain, all-or-nothing rates (fraction of transcripts
            # whose entire chain matches), NOT per-junction Sn/Sp — the "rate"
            # wording keeps that distinction explicit.
            _PR_DISPLAY = {
                "intron_chain": "Exact intron-chain rate",
                "intron_chain_subset": "Intron Subset",
                "intron_chain_superset": "Intron Superset",
                "exon_chain": "Exact exon-chain rate",
                "exon_chain_superset": "Exon Superset",
                "exon_chain_subset": "Exon Subset",
            }
            _method_scores: dict[str, dict] = {}
            for _, _row in df_sc_scope.iterrows():
                if _row["metric_key"] in _PR_KEYS and isinstance(_row["value"], dict):
                    _method_scores.setdefault(_row["method_name"], {})[_row["metric_key"]] = _row["value"]

            if _method_scores:
                # Each tier is all-or-nothing: a chain mismatch is booked as BOTH
                # a false positive and a false negative, so per tier fp == fn for
                # every sequence. Precision, recall and F1 (and their bootstrap
                # standard errors) are therefore identical by construction — a
                # separate precision and recall figure would be the same image
                # twice. Collapse to a single per-tier match rate (precision is
                # the representative value) and let the subset (precision-
                # flavoured) vs superset (recall-flavoured) tiers carry the P/R
                # contrast instead. See docs/metrics/structural_coherence.md.
                _pr_rows = []
                for _method, _scores in _method_scores.items():
                    _combined = {
                        _PR_DISPLAY[k]: v.get("precision", 0.0) for k, v in _scores.items() if isinstance(v, dict)
                    }
                    for k, v in _scores.items():
                        if isinstance(v, dict):
                            se = v.get("precision_stderr")
                            if se is not None:
                                _combined[f"{_PR_DISPLAY[k]}_stderr"] = se
                    _pr_rows.append(
                        {
                            "method_name": _method,
                            "metric_group": EvalMetrics.STRUCTURAL_COHERENCE.name,
                            "metric_key": "ts_level_match_rate",
                            "value": _combined,
                        }
                    )
                _df_pr = pd.DataFrame(_pr_rows)
                _prefix = (output_dir / f"transcript_match_rate{'' if scope == default_scope else f'_{scope}'}") if output_dir else None
                _pr_figs = plot_ml_metrics_bar(
                    _df_pr, class_name, save_path_prefix=_prefix, metadata_map=PLOT_METADATA
                )
                for _fig in _pr_figs.values():
                    figures[_figure_key("transcript_match_rate", scope, default_scope)] = _fig

            fig = plot_transcript_match_distribution(
                df_sc_scope,
                class_name,
                save_path=(output_dir / f"transcript_match{'' if scope == default_scope else f'_{scope}'}.png") if output_dir else None,
                metadata=PLOT_METADATA.get("transcript_match"),
            )
            if fig is not None:
                figures[_figure_key("transcript_match", scope, default_scope)] = fig

            fig = plot_segment_count_delta(
                df_sc_scope,
                class_name,
                save_path=(output_dir / f"segment_count_delta{'' if scope == default_scope else f'_{scope}'}.png") if output_dir else None,
                metadata=PLOT_METADATA.get("segment_count_delta"),
            )
            if fig is not None:
                figures[_figure_key("segment_count_delta", scope, default_scope)] = fig

            fig = plot_boundary_shift_distribution(
                df_sc_scope,
                class_name,
                save_path=(output_dir / f"boundary_shift_dist{'' if scope == default_scope else f'_{scope}'}.png") if output_dir else None,
                metadata=PLOT_METADATA.get("boundary_shift_distribution"),
            )
            if fig is not None:
                figures[_figure_key("boundary_shift_dist", scope, default_scope)] = fig

            fig = plot_per_transcript_exon_recovery(
                df_sc_scope,
                class_name,
                save_path=(output_dir / f"per_transcript_exon_recovery{'' if scope == default_scope else f'_{scope}'}.png") if output_dir else None,
                metadata=PLOT_METADATA.get("per_transcript_exon_recovery"),
            )
            if fig is not None:
                figures[_figure_key("per_transcript_exon_recovery", scope, default_scope)] = fig

    # ---- Diagnostic depth plots ----------------------------------------
    if EvalMetrics.DIAGNOSTIC_DEPTH in metrics_to_eval:
        df_dd = df[(df["metric_group"] == EvalMetrics.DIAGNOSTIC_DEPTH.name)].copy()

        if not df_dd.empty:
            default_scope = _metric_scopes(df_dd)[0]
            for scope in _metric_scopes(df_dd):
                df_scope = df_dd[df_dd["scope"] == scope] if scope is not None else df_dd[df_dd["scope"].isna()]
                class_name = _scope_display_name(label_config, scope)
                suffix = "" if scope == default_scope else f"_{scope}"
                fig = plot_position_bias(
                    df_scope,
                    class_name,
                    save_path=(output_dir / f"position_bias{suffix}.png") if output_dir else None,
                    metadata=PLOT_METADATA.get("position_bias"),
                )
                if fig is not None:
                    figures[_figure_key("position_bias", scope, default_scope)] = fig

    return figures
