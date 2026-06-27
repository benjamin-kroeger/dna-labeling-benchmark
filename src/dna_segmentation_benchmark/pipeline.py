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
from .feature_roles import (
    FeatureRoleMap,
    PredFeatureRoleMapInput,
    normalize_feature_role_map,
    normalize_pred_feature_role_maps,
)
from .io_utils import DEFAULT_TRANSCRIPT_TYPES, collect_gff
from .label_definition import AnnotationMode, LabelConfig, _DEFAULT_METRICS
from .transcript_mapping import (
    LocusMatchingMode,
    TranscriptMapping,
    _build_df_index,
    build_paired_arrays,
    export_mapping_table,
    map_transcripts,
)

logger = logging.getLogger(__name__)


def _include_mapping_for_predictor(
    mapping: TranscriptMapping,
    pred_name: str,
    mode: LocusMatchingMode,
) -> bool:
    """Return True if this mapping entry should contribute arrays for *pred_name*.

    Three cases in BEST_PER_LOCUS mode:
    - ``is_unmatched_prediction``: a real prediction with no GT overlap — include
      only for the predictor that owns it (FP entry).
    - ``fn_for_predictors`` non-empty: a locus-level miss entry — include only
      for predictors listed there (locus FN).
    - Otherwise: a normal matched entry — skip if the predictor didn't match
      (its FN is recorded by the locus-FN entry; don't double-count).
    """
    has_match = any(m.predictor_name == pred_name for m in mapping.matched_predictions)
    if mapping.is_unmatched_prediction:
        return has_match
    if mapping.fn_for_predictors:
        return pred_name in mapping.fn_for_predictors
    if mode == LocusMatchingMode.BEST_PER_LOCUS and not has_match:
        return False
    return True


def benchmark_from_gff(
    gt_path: str | Path,
    pred_paths: dict[str, str | Path],
    label_config: LabelConfig,
    metrics: list[EvalMetrics] | None = None,
    *,
    gt_feature_role_map: FeatureRoleMap | None = None,
    pred_feature_role_maps: PredFeatureRoleMapInput = None,
    transcript_types: list[str] | None = None,
    exclude_features: list[str] | None = None,
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

    ``label_config`` defines the integer label semantics.  Feature-role maps
    control how GFF/GTF feature names are translated to benchmark roles such as
    ``"exon"``, ``"cds"``, ``"five_prime_utr"``, ``"three_prime_utr"``.  When
    omitted, mode-specific defaults are used — ``EXON_INTRON`` maps both
    ``"exon"`` and ``"CDS"`` to the exon role; ``UTR_CDS_INTRON`` maps
    ``"five_prime_UTR"``, ``"CDS"``, and ``"three_prime_UTR"`` to their
    respective roles.

    Parameters
    ----------
    gt_path : str | Path
        Path to the ground-truth GFF/GTF annotation file.
    pred_paths : dict[str, str | Path]
        ``{predictor_name: path}`` for each prediction file.
    label_config : LabelConfig
        Token-to-name mapping and semantic label roles.
    metrics : list[EvalMetrics] | None
        Metric groups to compute.  Defaults to ``_DEFAULT_METRICS``
        (``REGION_DISCOVERY``, ``BOUNDARY_EXACTNESS``,
        ``NUCLEOTIDE_CLASSIFICATION``) — the same default used by
        :func:`benchmark_gt_vs_pred_single` /
        :func:`benchmark_gt_vs_pred_multiple`, so all entry points agree.
    gt_feature_role_map : dict[str, str] | None
        Maps GT GFF/GTF feature types to benchmark roles.  When ``None``,
        the mode-specific default is used.
    pred_feature_role_maps : dict[str, str] | dict[str, dict[str, str]] | None
        Maps prediction GFF/GTF feature types to roles.  A flat
        ``{feature_type: role}`` dict applies to every predictor; a nested
        ``{predictor_name: {feature_type: role}}`` dict allows predictor-specific
        parsing.  ``None`` means every predictor uses the GT role map.
    transcript_types : list[str] | None
        Feature types that define transcript boundaries.
        Defaults to ``["mRNA", "transcript"]``.
    exclude_features : list[str] | None
        GFF feature types to ignore (e.g., ``["gene"]``).
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
        ``{predictor_name: {"aggregated": ..., "global": ...}}`` where
        ``aggregated`` is the corpus-level micro-averaged summary returned by
        :func:`benchmark_gt_vs_pred_multiple` (not a per-transcript list).

    See Also
    --------
    compare_multiple_predictions :
        Renders comparison plots from these results.  Feed it the per-method
        ``aggregated`` payloads, e.g.::

            results = benchmark_from_gff(...)
            figures = compare_multiple_predictions(
                per_method_benchmark_res={
                    name: r["aggregated"] for name, r in results.items()
                },
                label_config=label_config,
                metrics_to_eval=metrics,
            )

        (It also accepts the full ``{"aggregated": ..., "global": ...}``
        wrapper per method and unwraps it automatically.)
    """
    exclude_features = exclude_features or []
    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    if metrics is None:
        metrics = list(_DEFAULT_METRICS)

    if (
        gt_feature_role_map is None
        and label_config.annotation_mode == AnnotationMode.EXON_INTRON
    ):
        logger.warning(
            "EXON_INTRON mode: no feature_role_map supplied — CDS features will be "
            "painted with exon_label=%d. Use UTR_CDS_INTRON mode for a dedicated CDS label.",
            label_config.exon_label,
        )

    # 1. Parse files
    gt_df = collect_gff(str(gt_path), exclude_features=exclude_features)
    pred_dfs = {name: collect_gff(str(p), exclude_features=exclude_features) for name, p in pred_paths.items()}

    # 2. Map transcripts (reuse the DataFrames parsed in step 1)
    mappings = map_transcripts(
        gt_path=gt_path,
        pred_paths={name: str(p) for name, p in pred_paths.items()},
        label_config=label_config,
        gt_df=gt_df,
        pred_dfs=pred_dfs,
        transcript_types=transcript_types,
        exclude_features=exclude_features,
        gt_feature_role_map=gt_feature_role_map,
        pred_feature_role_maps=pred_feature_role_maps,
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

    # Precompute the per-mapping invariants once: normalized role maps and the
    # (seqid, parent) lookup indices are reused for every mapping below.
    resolved_gt_map = normalize_feature_role_map(
        gt_feature_role_map, label_config, arg_name="gt_feature_role_map"
    )
    resolved_pred_maps = normalize_pred_feature_role_maps(
        list(pred_paths),
        pred_feature_role_maps,
        default=resolved_gt_map,
        label_config=label_config,
    )
    gt_index = _build_df_index(gt_df, transcript_types)
    pred_indices = {name: _build_df_index(df, transcript_types) for name, df in pred_dfs.items()}

    for mapping in mappings:
        gt_arr, pred_arrays = build_paired_arrays(
            mapping=mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=label_config,
            transcript_types=transcript_types,
            _gt_index=gt_index,
            _pred_indices=pred_indices,
            _gt_role_map=resolved_gt_map,
            _pred_role_maps=resolved_pred_maps,
        )

        for pred_name in pred_paths:
            if not _include_mapping_for_predictor(mapping, pred_name, locus_matching_mode):
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

        aggregated = benchmark_gt_vs_pred_multiple(
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            label_config=label_config,
            metrics=metrics,
            infer_introns=infer_introns,
        )

        logger.info("Finished per-transcript benchmarking for '%s', now computing global metrics...", pred_name)

        global_result = compute_global_metrics(
            gt_df=gt_df,
            pred_df=pred_dfs[pred_name],
            mappings=mappings,
            predictor_name=pred_name,
            label_config=label_config,
            transcript_types=transcript_types,
            gt_feature_role_map=resolved_gt_map,
            pred_feature_role_map=resolved_pred_maps[pred_name],
        )

        all_results[pred_name] = {
            "aggregated": aggregated,
            "global": global_result,
        }

    return all_results
