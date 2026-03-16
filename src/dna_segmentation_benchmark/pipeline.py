"""High-level convenience pipeline for GFF/GTF-based benchmarking.

Wraps the load → map → build-arrays → benchmark steps into a single call,
so that common workflows can be expressed in a few lines.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .eval.evaluate_predictors import EvalMetrics, benchmark_gt_vs_pred_multiple
from .io_utils import DEFAULT_TRANSCRIPT_TYPES, collect_gff
from .label_definition import LabelConfig
from .transcript_mapping import build_paired_arrays, map_transcripts

logger = logging.getLogger(__name__)


def benchmark_from_gff(
    gt_path: str | Path,
    pred_paths: dict[str, str | Path],
    label_config: LabelConfig,
    classes: list[int],
    metrics: list[EvalMetrics] | None = None,
    *,
    min_overlap: float = 0.2,
    exclude_features: list[str] | None = None,
    transcript_types: list[str] | None = None,
) -> dict[str, dict]:
    """Run the full benchmark pipeline from GFF/GTF files.

    This is a convenience wrapper that performs:

    1. Parse GT and prediction GFF/GTF files
    2. Map GT transcripts to predictions (strand-aware, per-chromosome)
    3. Build paired annotation arrays
    4. Compute all requested metrics

    Parameters
    ----------
    gt_path : str | Path
        Path to the ground-truth GFF/GTF annotation file.
    pred_paths : dict[str, str | Path]
        ``{predictor_name: path}`` for each prediction file.
    label_config : LabelConfig
        Token-to-name mapping and semantic roles.
    classes : list[int]
        Integer tokens to evaluate (e.g., ``[0]`` for CDS only).
    metrics : list[EvalMetrics] | None
        Metric groups to compute.  Defaults to
        ``[REGION_DISCOVERY, NUCLEOTIDE_CLASSIFICATION]``.
    min_overlap : float
        Minimum overlap fraction for transcript matching (default 0.2).
    exclude_features : list[str] | None
        GFF feature types to ignore (e.g., ``["gene"]``).
    transcript_types : list[str] | None
        Feature types that define transcript boundaries.

    Returns
    -------
    dict[str, dict]
        ``{predictor_name: aggregated_results}`` — one entry per predictor,
        each containing the nested result dict as returned by
        :func:`benchmark_gt_vs_pred_multiple`.  Ready to pass directly to
        :func:`compare_multiple_predictions`.
    """
    exclude_features = exclude_features or []
    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    if metrics is None:
        metrics = [EvalMetrics.REGION_DISCOVERY, EvalMetrics.NUCLEOTIDE_CLASSIFICATION]

    # 1. Parse files
    gt_df = collect_gff(str(gt_path), exclude_features=exclude_features)
    pred_dfs = {
        name: collect_gff(str(p), exclude_features=exclude_features)
        for name, p in pred_paths.items()
    }

    # 2. Map transcripts
    mappings = map_transcripts(
        gt_path=gt_path,
        pred_paths={name: str(p) for name, p in pred_paths.items()},
        min_overlap=min_overlap,
        transcript_types=transcript_types,
        exclude_features=exclude_features,
    )

    if not mappings:
        raise ValueError(
            "No transcript mappings found. Check that GT and prediction "
            "files share overlapping genomic regions on the same strand."
        )

    # 3. Build arrays per predictor
    gt_by_pred: dict[str, list[np.ndarray]] = {name: [] for name in pred_paths}
    pred_by_pred: dict[str, list[np.ndarray]] = {name: [] for name in pred_paths}

    for mapping in mappings:
        gt_arr, pred_arrs = build_paired_arrays(
            mapping=mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=label_config,
            transcript_types=transcript_types,
        )

        for pred_name in pred_paths:
            if mapping.is_unmatched_prediction:
                owns = any(
                    m.predictor_name == pred_name
                    for m in mapping.matched_predictions
                )
                if not owns:
                    continue

            gt_by_pred[pred_name].append(gt_arr)
            pred_by_pred[pred_name].append(pred_arrs[pred_name])

    # 4. Benchmark each predictor
    all_results: dict[str, dict] = {}

    for pred_name in pred_paths:
        gt_labels = gt_by_pred[pred_name]
        pred_labels = pred_by_pred[pred_name]

        if not gt_labels:
            logger.warning("No mapped transcripts for '%s', skipping.", pred_name)
            continue

        all_results[pred_name] = benchmark_gt_vs_pred_multiple(
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            label_config=label_config,
            classes=classes,
            metrics=metrics,
        )

    return all_results
