"""High-level convenience pipeline for GFF/GTF-based benchmarking.

Wraps the load → map → build-arrays → benchmark steps into a single call,
so that common workflows can be expressed in a few lines.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .eval.evaluate_predictors import EvalMetrics, benchmark_gt_vs_pred_multiple
from .eval.global_metrics import compute_global_metrics
from .io_utils import DEFAULT_TRANSCRIPT_TYPES, collect_gff
from .label_definition import LabelConfig
from .transcript_mapping import (
    LocusMatchingMode,
    build_paired_arrays,
    export_mapping_table,
    map_transcripts,
)

logger = logging.getLogger(__name__)


def benchmark_from_gff(
    gt_path: str | Path,
    pred_paths: dict[str, str | Path],
    label_config: LabelConfig,
    metrics: list[EvalMetrics] | None = None,
    *,
    exclude_features: list[str] | None = None,
    transcript_types: list[str] | None = None,
    mapping_output_path: str | Path | None = None,
    locus_matching_mode: LocusMatchingMode = LocusMatchingMode.FULL_DISCOVERY,
) -> dict[str, dict]:
    """Run the full benchmark pipeline from GFF/GTF files.

    This is a convenience wrapper that performs:

    1. Parse GT and prediction GFF/GTF files
    2. Map GT transcripts to predictions (strand-aware, locus-based)
    3. Build paired annotation arrays
    4. Compute all requested metrics

    Which labels to evaluate and which GFF feature types to treat as exons
    are derived from ``label_config`` automatically:

    * ``label_config.evaluation_labels`` — determines the per-class metric
      targets (exon, intron, splice sites).
    * ``label_config.exon_feature_type`` — selects the GFF feature type(s)
      to use for intron-chain computation and array painting (e.g.
      ``["exon"]`` for Helixer, ``["CDS"]`` for Augustus).

    Parameters
    ----------
    gt_path : str | Path
        Path to the ground-truth GFF/GTF annotation file.
    pred_paths : dict[str, str | Path]
        ``{predictor_name: path}`` for each prediction file.
    label_config : LabelConfig
        Token-to-name mapping, semantic roles, and GFF feature type.
    metrics : list[EvalMetrics] | None
        Metric groups to compute.  Defaults to
        ``[REGION_DISCOVERY, NUCLEOTIDE_CLASSIFICATION]``.
    exclude_features : list[str] | None
        GFF feature types to ignore (e.g., ``["gene"]``).
    transcript_types : list[str] | None
        Feature types that define transcript boundaries.
    mapping_output_path : str | Path | None
        If given, write the GT-to-prediction mapping table to this path
        (TSV format, similar to gffcompare's ``.loci`` file).
    locus_matching_mode : LocusMatchingMode
        Controls how transcripts are paired within each locus.
        ``FULL_DISCOVERY`` (default) maximises 1:1 matches.
        ``BEST_PER_LOCUS`` keeps only the single best-scoring pair per locus;
        suited for single-transcript predictors (e.g. Augustus).

    Returns
    -------
    dict[str, dict]
        ``{predictor_name: {"per_transcript": ..., "global": ...}}``
    """
    exclude_features = exclude_features or []
    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    exon_types = list(label_config.exon_feature_type)
    if metrics is None:
        metrics = [EvalMetrics.REGION_DISCOVERY, EvalMetrics.NUCLEOTIDE_CLASSIFICATION]

    # 1. Parse files
    gt_df = collect_gff(str(gt_path), exclude_features=exclude_features)
    pred_dfs = {
        name: collect_gff(str(p), exclude_features=exclude_features)
        for name, p in pred_paths.items()
    }

    # 2. Map transcripts
    # GT intron chains use the default "exon" feature type (covers GENCODE,
    # Ensembl, stringtie and other reference annotations).
    # Pred intron chains use exon_types from the label config so that CDS-only
    # predictors (e.g. Augustus) have their introns computed correctly.
    mappings = map_transcripts(
        gt_path=gt_path,
        pred_paths={name: str(p) for name, p in pred_paths.items()},
        transcript_types=transcript_types,
        pred_exon_types=exon_types,
        exclude_features=exclude_features,
        locus_matching_mode=locus_matching_mode,
    )

    if not mappings:
        raise ValueError(
            "No transcript mappings found. Check that GT and prediction "
            "files share overlapping genomic regions on the same strand."
        )

    if mapping_output_path is not None:
        export_mapping_table(mappings, mapping_output_path)

    # 3. Build arrays per predictor
    gt_by_pred: dict[str, list[np.ndarray]] = {name: [] for name in pred_paths}
    pred_by_pred: dict[str, list[np.ndarray]] = {name: [] for name in pred_paths}

    for mapping in mappings:
        gt_arr, pred_arrays = build_paired_arrays(
            mapping=mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=label_config,
            transcript_types=transcript_types,
            exon_types=None,             # GT: paint all non-transcript child features
            pred_exon_types=exon_types,  # Pred: specific feature type from label config
        )

        for pred_name in pred_paths:
            has_match = any(m.predictor_name == pred_name for m in mapping.matched_predictions)
            if mapping.is_unmatched_prediction:
                if not has_match:
                    continue
            elif locus_matching_mode == LocusMatchingMode.BEST_PER_LOCUS and not has_match:
                continue

            gt_by_pred[pred_name].append(gt_arr)
            pred_by_pred[pred_name].append(pred_arrays[pred_name])

    # 4. Benchmark each predictor
    all_results: dict[str, dict] = {}

    for pred_name in pred_paths:
        gt_labels = gt_by_pred[pred_name]
        pred_labels = pred_by_pred[pred_name]

        if not gt_labels:
            logger.warning("No mapped transcripts for '%s', skipping.", pred_name)
            continue

        per_transcript = benchmark_gt_vs_pred_multiple(
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            label_config=label_config,
            metrics=metrics,
        )

        global_result = compute_global_metrics(
            gt_df=gt_df,
            pred_df=pred_dfs[pred_name],
            mappings=mappings,
            predictor_name=pred_name,
            label_config=label_config,
            exon_types=exon_types,
            transcript_types=transcript_types,
        )

        all_results[pred_name] = {
            "per_transcript": per_transcript,
            "global": global_result,
        }

    return all_results
