"""Cross-sequence aggregation for the benchmark.

Two responsibilities:

* :func:`recursive_merge` — accumulate per-sequence result dicts into a single
  dict (scalars become lists, ndarrays sum element-wise, dicts recurse).
* :func:`_aggregate_summary_metrics` — reduce those raw accumulated counts into
  the user-facing precision/recall/F1 and distribution summaries.
"""

from collections import Counter

import numpy as np

from .boundary_precision import _compute_boundary_precision_landscape
from .statistics import _compute_summary_statistics, _compute_distribution_stats
from .structural_summary import POSITION_BIAS_HISTOGRAM_BINS
from ..label_definition import EvalMetrics


def recursive_merge(target: dict, source: dict) -> dict:
    """Recursively merge *source* into *target*, skipping ``None`` values."""
    for key, source_value in source.items():
        if source_value is None:
            continue

        if key not in target:
            if isinstance(source_value, dict):
                target[key] = {}
                recursive_merge(target[key], source_value)
            elif isinstance(source_value, list):
                target[key] = list(source_value)
            elif isinstance(source_value, np.ndarray):
                target[key] = source_value
            else:
                target[key] = [source_value]
        else:
            target_value = target[key]
            if isinstance(source_value, dict) and isinstance(target_value, dict):
                recursive_merge(target_value, source_value)
            elif isinstance(target_value, list):
                if isinstance(source_value, list):
                    target_value.extend(source_value)
                else:
                    target_value.append(source_value)
            elif isinstance(target_value, np.ndarray):
                target[key] += source_value
            else:
                target[key] = [target_value, source_value]
    return target


def _aggregate_summary_metrics(aggregated: dict, metrics: list[EvalMetrics]) -> dict:
    """Compute user-facing summary statistics from raw accumulated counts.

    After multi-sequence merging, the raw tp/fn/fp lists are converted into
    precision & recall (and F1 for nucleotide level).  Raw counts are
    *replaced* by the computed summaries so they are not exposed to the user.
    """
    if "false_transitions" in aggregated:
        # recursive_merge sums np.ndarray (matrices) element-wise already.
        # It wraps int values (totals) into lists — sum them back.
        aggregated["false_transitions"]["stable_position_counts"] = {
            k: sum(v) if isinstance(v, list) else v
            for k, v in aggregated["false_transitions"]["stable_position_counts"].items()
        }

    # -- REGION_DISCOVERY: precision & recall per strictness level ------
    if EvalMetrics.REGION_DISCOVERY in metrics and EvalMetrics.REGION_DISCOVERY.name in aggregated:
        rd = aggregated[EvalMetrics.REGION_DISCOVERY.name]
        for level_key in ("neighborhood_hit", "internal_hit", "full_coverage_hit", "perfect_boundary_hit"):
            rd[level_key] = _compute_summary_statistics(**rd[level_key])

    # -- BOUNDARY_EXACTNESS: IoU stats + landscape -
    if EvalMetrics.BOUNDARY_EXACTNESS in metrics and EvalMetrics.BOUNDARY_EXACTNESS.name in aggregated:
        be = aggregated[EvalMetrics.BOUNDARY_EXACTNESS.name]

        if "iou_scores" in be:
            be["iou_stats"] = _compute_distribution_stats(be["iou_scores"], is_abs=False)

        if "fuzzy_metrics" in be:
            be["fuzzy_metrics"] = _compute_boundary_precision_landscape(
                residuals=be["fuzzy_metrics"]["boundary_residuals"],
                total_gt_count=sum(be["fuzzy_metrics"]["total_gt"]),
            )

    # -- NUCLEOTIDE_CLASSIFICATION: precision, recall, F1 ---------------
    if EvalMetrics.NUCLEOTIDE_CLASSIFICATION in metrics and EvalMetrics.NUCLEOTIDE_CLASSIFICATION.name in aggregated:
        nc = aggregated[EvalMetrics.NUCLEOTIDE_CLASSIFICATION.name]
        nuc_counts = nc["nucleotide"]
        summary = _compute_summary_statistics(**nuc_counts)
        p, r = summary.get("precision", 0), summary.get("recall", 0)
        summary["f1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        nc["nucleotide"] = summary

    # -- STRUCTURAL_COHERENCE: chain, grammar, transcript classification --
    if EvalMetrics.STRUCTURAL_COHERENCE in metrics:
        sc = aggregated.get(EvalMetrics.STRUCTURAL_COHERENCE.name, {})
        if sc:
            for _key in ("intron_chain", "intron_chain_subset", "intron_chain_superset"):
                if _key in sc:
                    sc[_key] = _compute_summary_statistics(**sc[_key])

            if "segment_count_delta" in sc and isinstance(sc["segment_count_delta"], list):
                sc["segment_count_delta"] = _compute_distribution_stats(
                    sc["segment_count_delta"],
                    is_abs=False,
                )

            for key in ("segment_count_gt", "segment_count_pred", "intron_count_gt", "intron_count_pred"):
                if key in sc and isinstance(sc[key], list):
                    sc[key] = sum(sc[key])

            if "transcript_match_class" in sc and isinstance(sc["transcript_match_class"], list):
                counts = Counter(sc["transcript_match_class"])
                total = sum(counts.values())
                sc["transcript_match_distribution"] = dict(counts)
                sc["exact_match_rate"] = counts.get("exact", 0) / total if total > 0 else 0.0

            for tier_key in ("exon_chain", "exon_chain_subset", "exon_chain_superset"):
                if tier_key in sc:
                    sc[tier_key] = _compute_summary_statistics(**sc[tier_key])

    # -- SPLICE_SITES: sum raw counts, compute precision/recall ----------------
    if "splice_sites" in aggregated:
        ss = aggregated["splice_sites"]
        for key in ("both_correct", "donor_only", "acceptor_only", "neither",
                    "donor_tp", "donor_fp", "donor_fn",
                    "acceptor_tp", "acceptor_fp", "acceptor_fn"):
            if isinstance(ss.get(key), list):
                ss[key] = sum(ss[key])
        d_tp, d_fp, d_fn = ss["donor_tp"], ss["donor_fp"], ss["donor_fn"]
        a_tp, a_fp, a_fn = ss["acceptor_tp"], ss["acceptor_fp"], ss["acceptor_fn"]
        ss["donor_precision"] = d_tp / (d_tp + d_fp) if (d_tp + d_fp) > 0 else 0.0
        ss["donor_recall"] = d_tp / (d_tp + d_fn) if (d_tp + d_fn) > 0 else 0.0
        ss["acceptor_precision"] = a_tp / (a_tp + a_fp) if (a_tp + a_fp) > 0 else 0.0
        ss["acceptor_recall"] = a_tp / (a_tp + a_fn) if (a_tp + a_fn) > 0 else 0.0

    # -- DIAGNOSTIC_DEPTH: segment length distribution + position bias histogram
    if EvalMetrics.DIAGNOSTIC_DEPTH in metrics:
        dd = aggregated.get(EvalMetrics.DIAGNOSTIC_DEPTH.name, {})
        if dd:
            if "length_emd" in dd and isinstance(dd["length_emd"], list):
                dd["length_emd"] = _compute_distribution_stats(
                    dd["length_emd"],
                    is_abs=False,
                )

            for hist_key in (
                "position_bias_histogram_fn",
                "position_bias_histogram_fp",
            ):
                if hist_key in dd and isinstance(dd[hist_key], list):
                    raw = dd[hist_key]
                    n_bins = POSITION_BIAS_HISTOGRAM_BINS
                    if len(raw) > n_bins:
                        if len(raw) % n_bins != 0:
                            raise ValueError(
                                f"position-bias histogram aggregation expected a flat list whose "
                                f"length is a multiple of {n_bins} bins, got {len(raw)}. "
                                "This usually means a per-sequence histogram was emitted with a "
                                "different bin count; the producer in structural_summary.py and "
                                "this aggregator must use the same POSITION_BIAS_HISTOGRAM_BINS."
                            )
                        arr = np.array(raw).reshape(-1, n_bins)
                        dd[hist_key] = arr.sum(axis=0).tolist()

    return aggregated
