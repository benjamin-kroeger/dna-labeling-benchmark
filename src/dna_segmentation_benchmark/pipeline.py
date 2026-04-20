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


FeatureTypeInput = str | list[str]
PredFeatureTypeInput = FeatureTypeInput | dict[str, FeatureTypeInput]


def _coerce_feature_types(
    feature_types: FeatureTypeInput,
    *,
    arg_name: str,
) -> list[str]:
    """Normalize one feature-type argument to a non-empty list of strings."""
    if isinstance(feature_types, str):
        feature_types = [feature_types]

    if not isinstance(feature_types, list) or not feature_types:
        raise ValueError(f"{arg_name} must be a string or non-empty list of strings.")

    normalized: list[str] = []
    for feature_type in feature_types:
        if not isinstance(feature_type, str) or not feature_type:
            raise ValueError(f"{arg_name} entries must be non-empty strings.")
        normalized.append(feature_type)
    return normalized


def _normalise_pred_exon_feature_types(
    pred_names: list[str],
    pred_exon_feature_types: PredFeatureTypeInput | None,
    *,
    default: list[str],
) -> dict[str, list[str]]:
    """Return per-predictor exon feature-type lists.

    ``None`` means predictions use the same feature types as GT.  A string or
    list applies to every predictor.  A dict allows predictor-specific parsing,
    e.g. ``{"augustus": "CDS", "helixer": "exon"}``.
    """
    if pred_exon_feature_types is None:
        return {name: list(default) for name in pred_names}

    if isinstance(pred_exon_feature_types, dict):
        return {
            name: _coerce_feature_types(
                pred_exon_feature_types.get(name, default),
                arg_name=f"pred_exon_feature_types[{name!r}]",
            )
            for name in pred_names
        }

    normalized = _coerce_feature_types(
        pred_exon_feature_types,
        arg_name="pred_exon_feature_types",
    )
    return {name: list(normalized) for name in pred_names}


def benchmark_from_gff(
    gt_path: str | Path,
    pred_paths: dict[str, str | Path],
    label_config: LabelConfig,
    metrics: list[EvalMetrics] | None = None,
    *,
    gt_exon_feature_types: FeatureTypeInput = "exon",
    pred_exon_feature_types: PredFeatureTypeInput | None = None,
    exclude_features: list[str] | None = None,
    transcript_types: list[str] | None = None,
    mapping_output_path: str | Path | None = None,
    locus_matching_mode: LocusMatchingMode = LocusMatchingMode.FULL_DISCOVERY,
    infer_introns: bool = False,
) -> dict[str, dict]:
    """Run the full benchmark pipeline from GFF/GTF files.

    This is a convenience wrapper that performs:

    1. Parse GT and prediction GFF/GTF files
    2. Map GT transcripts to predictions (strand-aware, locus-based)
    3. Build paired annotation arrays
    4. Compute all requested metrics

    ``label_config`` defines the integer label semantics used in the produced
    arrays: background, exon/coding, optional intron, and optional splice-site
    tokens.  GFF/GTF feature names are parser configuration and are passed
    explicitly via ``gt_exon_feature_types`` and ``pred_exon_feature_types``.

    Parameters
    ----------
    gt_path : str | Path
        Path to the ground-truth GFF/GTF annotation file.
    pred_paths : dict[str, str | Path]
        ``{predictor_name: path}`` for each prediction file.
    label_config : LabelConfig
        Token-to-name mapping and semantic label roles.  It does not decide
        which GFF/GTF feature rows are read as exons.
    metrics : list[EvalMetrics] | None
        Metric groups to compute.  Defaults to
        ``[REGION_DISCOVERY, NUCLEOTIDE_CLASSIFICATION]``.
    gt_exon_feature_types : str | list[str]
        GFF/GTF feature types to treat as exon/coding intervals in the ground
        truth annotation.  Accepts a single string (``"exon"``) or a list
        (``["exon", "CDS"]``).  Defaults to ``"exon"``.
    pred_exon_feature_types : str | list[str] | dict[str, str | list[str]] | None
        GFF/GTF feature types to treat as exon/coding intervals in prediction
        annotations.  ``None`` means use ``gt_exon_feature_types`` for every
        predictor.  A string/list applies to every predictor.  A dict maps
        predictor name to feature types, e.g. ``{"augustus": "CDS",
        "helixer": "exon"}``.  Use ``"CDS"`` for tools like Augustus that emit
        CDS features instead of exon features.
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
    infer_introns : bool
        If ``True``, fill background gaps between adjacent coding segments
        with ``label_config.intron_label`` before benchmarking.

    Returns
    -------
    dict[str, dict]
        ``{predictor_name: {"per_transcript": ..., "global": ...}}``
    """
    exclude_features = exclude_features or []
    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    gt_exon_types = _coerce_feature_types(
        gt_exon_feature_types,
        arg_name="gt_exon_feature_types",
    )
    pred_exon_types_by_name = _normalise_pred_exon_feature_types(
        list(pred_paths.keys()),
        pred_exon_feature_types,
        default=gt_exon_types,
    )
    if metrics is None:
        metrics = [EvalMetrics.REGION_DISCOVERY, EvalMetrics.NUCLEOTIDE_CLASSIFICATION]

    # 1. Parse files
    gt_df = collect_gff(str(gt_path), exclude_features=exclude_features)
    pred_dfs = {
        name: collect_gff(str(p), exclude_features=exclude_features)
        for name, p in pred_paths.items()
    }

    # 2. Map transcripts
    # Intron-chain mapping is derived from the selected exon-like feature rows.
    # GT and prediction feature names are independent because references usually
    # expose "exon" rows while some predictors, such as Augustus, emit "CDS".
    mappings = map_transcripts(
        gt_path=gt_path,
        pred_paths={name: str(p) for name, p in pred_paths.items()},
        transcript_types=transcript_types,
        exon_types=gt_exon_types,
        pred_exon_types=pred_exon_types_by_name,
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
            exon_types=gt_exon_types,
            pred_exon_types=pred_exon_types_by_name,
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
            infer_introns=infer_introns,
        )

        global_result = compute_global_metrics(
            gt_df=gt_df,
            pred_df=pred_dfs[pred_name],
            mappings=mappings,
            predictor_name=pred_name,
            label_config=label_config,
            gt_exon_types=gt_exon_types,
            pred_exon_types=pred_exon_types_by_name[pred_name],
            transcript_types=transcript_types,
        )

        all_results[pred_name] = {
            "per_transcript": per_transcript,
            "global": global_result,
        }

    return all_results
