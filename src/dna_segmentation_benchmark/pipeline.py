"""High-level convenience pipeline for GFF/GTF-based benchmarking.

Wraps the load → map → build-arrays → benchmark steps into a single call,
so that common workflows can be expressed in a few lines.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from .eval.evaluate_predictors import (
    EvalMetrics,
    StreamingBenchmark,
    _warn_phase_drift,
    benchmark_gt_vs_pred_single,
)
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
    _build_df_index,
    _include_mapping_for_predictor,
    build_paired_arrays,
    export_mapping_table,
    map_transcripts,
)

logger = logging.getLogger(__name__)

# ponytail: module-level state inherited by forked workers; not thread-safe, fine for single-call use
_BENCH_STATE: dict = {}


def _bench_worker_chunk(mappings_chunk: list) -> dict:
    """Worker: process a chunk of mappings and return {pred_name: (merged_data, count)}."""
    s = _BENCH_STATE
    benches = {
        name: StreamingBenchmark(s["label_config"], s["metrics"], infer_introns=s["infer_introns"])
        for name in s["pred_names"]
    }
    for mapping in mappings_chunk:
        gt_arr, pred_arrays = build_paired_arrays(
            mapping=mapping,
            gt_df=s["gt_df"],
            pred_dfs=s["pred_dfs"],
            label_config=s["label_config"],
            transcript_types=s["transcript_types"],
            _gt_index=s["gt_index"],
            _pred_indices=s["pred_indices"],
            _gt_role_map=s["resolved_gt_map"],
            _pred_role_maps=s["resolved_pred_maps"],
        )
        for pred_name in s["pred_names"]:
            if not _include_mapping_for_predictor(mapping, pred_name, s["mode"]):
                continue
            benches[pred_name].add(gt_arr, pred_arrays[pred_name])
    return {name: (benches[name]._accumulator.merged(), benches[name].count) for name in s["pred_names"]}


def _stream_benchmark_over_mappings(
    mappings: list,
    gt_df,
    pred_dfs: dict,
    label_config: LabelConfig,
    transcript_types: list[str],
    resolved_gt_map: FeatureRoleMap,
    resolved_pred_maps: dict[str, FeatureRoleMap],
    mode: LocusMatchingMode,
    metrics: list[EvalMetrics],
    *,
    infer_introns: bool = False,
    return_individual_results: bool = False,
) -> dict[str, dict | list]:
    """Benchmark every predictor in a single streaming pass over the mappings.

    Shared by :func:`benchmark_from_gff` and the ``dna-benchmark run`` CLI so the
    two entry points cannot drift.  Replaces the previous build-all-arrays-then-
    benchmark flow: :func:`build_paired_arrays` is called **once per mapping** and
    the resulting per-predictor arrays are fed straight into per-predictor
    consumers, so only one mapping's arrays are alive at a time (the fix for OOM
    on large GFF files).  Pre-normalized role maps and the per-DataFrame indices
    are computed once and reused for every mapping (the fast path), and
    :func:`_include_mapping_for_predictor` selects which predictors consume each
    mapping.

    Returns ``{predictor_name: result}`` for every predictor that received at
    least one mapping.  In aggregate mode (default) each value is the summary
    returned by :func:`benchmark_gt_vs_pred_multiple`; in individual mode each
    value is the per-sequence list.  Predictors with no mappings are omitted so
    callers can warn and skip them.
    """
    pred_names = list(pred_dfs)
    gt_index = _build_df_index(gt_df, transcript_types)
    pred_indices = {name: _build_df_index(df, transcript_types) for name, df in pred_dfs.items()}

    benches: dict[str, StreamingBenchmark] = {}
    collected: dict[str, list] = {}
    if return_individual_results:
        _warn_phase_drift(metrics)
        collected = {name: [] for name in pred_names}
    else:
        benches = {
            name: StreamingBenchmark(label_config, metrics, infer_introns=infer_introns)
            for name in pred_names
        }

    n_workers = max(1, (os.cpu_count() or 1) - 1)
    use_parallel = not return_individual_results and n_workers > 1 and len(mappings) >= 500

    if use_parallel:
        global _BENCH_STATE
        _BENCH_STATE = {
            "gt_df": gt_df, "pred_dfs": pred_dfs, "label_config": label_config,
            "transcript_types": transcript_types, "gt_index": gt_index, "pred_indices": pred_indices,
            "resolved_gt_map": resolved_gt_map, "resolved_pred_maps": resolved_pred_maps,
            "mode": mode, "metrics": list(metrics), "infer_introns": infer_introns,
            "pred_names": pred_names,
        }
        chunk_size = max(1, len(mappings) // (n_workers * 50))
        chunks = [mappings[i:i + chunk_size] for i in range(0, len(mappings), chunk_size)]
        ctx = multiprocessing.get_context("fork")  # ponytail: Linux fork; for macOS use spawn + initializer
        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                futures = {pool.submit(_bench_worker_chunk, chunk): len(chunk) for chunk in chunks}
                with tqdm(total=len(mappings), desc="Benchmarking", unit="mapping") as pbar:
                    for future in as_completed(futures):
                        for pred_name, (merged_data, count) in future.result().items():
                            benches[pred_name].merge_from_merged(merged_data, count)
                        pbar.update(futures[future])
        finally:
            # Release the run's gt/pred DataFrames + indices; this global is only
            # ever reassigned, so without this it pins one full run's data across
            # calls (e.g. a clade's per-species loop) and inflates the next run's
            # peak — enough to OOM the heavy pooled run that follows.
            _BENCH_STATE = {}
    else:
        for mapping in tqdm(mappings, desc="Benchmarking", unit="mapping"):
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
            for pred_name in pred_names:
                if not _include_mapping_for_predictor(mapping, pred_name, mode):
                    continue
                if return_individual_results:
                    collected[pred_name].append(
                        benchmark_gt_vs_pred_single(
                            gt_labels=gt_arr,
                            pred_labels=pred_arrays[pred_name],
                            label_config=label_config,
                            metrics=metrics,
                            infer_introns=infer_introns,
                        )
                    )
                else:
                    benches[pred_name].add(gt_arr, pred_arrays[pred_name])
            # gt_arr / pred_arrays fall out of scope on the next iteration and are freed.

    if return_individual_results:
        return {name: res for name, res in collected.items() if res}
    results = {}
    for name, bench in tqdm(benches.items(), desc="Summarising", unit="predictor"):
        if bench.count:
            results[name] = bench.result()
    return results


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
    logger.info("Starting Benchmark from GFF/GTF: GT=%s, predictors=%s", gt_path, list(pred_paths))
    exclude_features = exclude_features or []
    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    if metrics is None:
        metrics = list(_DEFAULT_METRICS)
    logger.debug(f"Running with metrics: {metrics}")

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
    logger.debug("Finished Pyranges reading of gff")

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
    logger.debug("Finished ground truth to prediction transcript mappings")

    if not mappings:
        raise ValueError(
            "No transcript mappings found. Check that GT and prediction "
            "files share overlapping genomic regions on the same strand."
        )

    if mapping_output_path is not None:
        export_mapping_table(mappings, mapping_output_path)

    # 3. Build arrays per predictor.  Pre-normalize the role maps once; they are
    # reused both for array building and for the global metrics below.
    resolved_gt_map = normalize_feature_role_map(
        gt_feature_role_map, label_config, arg_name="gt_feature_role_map"
    )
    resolved_pred_maps = normalize_pred_feature_role_maps(
        list(pred_paths),
        pred_feature_role_maps,
        default=resolved_gt_map,
        label_config=label_config,
    )
    aggregated_by_pred = _stream_benchmark_over_mappings(
        mappings,
        gt_df,
        pred_dfs,
        label_config,
        transcript_types,
        resolved_gt_map,
        resolved_pred_maps,
        locus_matching_mode,
        metrics,
        infer_introns=infer_introns,
    )

    # 4. Attach global metrics for each predictor that had mapped transcripts.
    all_results: dict[str, dict] = {}

    for pred_name in tqdm(list(pred_paths), desc="Global metrics", unit="predictor"):
        aggregated = aggregated_by_pred.get(pred_name)
        if aggregated is None:
            logger.warning("No mapped transcripts for '%s', skipping.", pred_name)
            continue

        global_result = compute_global_metrics(
            gt_df=gt_df,
            pred_df=pred_dfs[pred_name],
            mappings=mappings,
            predictor_name=pred_name,
            label_config=label_config,
            transcript_types=transcript_types,
            gt_feature_role_map=resolved_gt_map,
            pred_feature_role_map=resolved_pred_maps[pred_name],
            locus_matching_mode=locus_matching_mode,
        )

        all_results[pred_name] = {
            "aggregated": aggregated,
            "global": global_result,
        }

    return all_results
