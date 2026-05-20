"""Core evaluation logic for the DNA segmentation benchmark.

Compares ground-truth nucleotide-level annotations with predicted annotations
and computes a rich set of metrics:

* **INDEL** – 5'/3' extensions/deletions, whole insertions/deletions,
  joins/splits.
* **REGION_DISCOVERY** – precision & recall at four overlap strictness
  levels (neighbourhood, internal, full-coverage, perfect-boundary).
* **BOUNDARY_EXACTNESS** – IoU statistics, boundary-residual bias /
  reliability landscape, terminal-boundary flags.
* **NUCLEOTIDE_CLASSIFICATION** – per-base precision, recall, and F1.
* **FRAMESHIFT** – per-position reading-frame deviation between GT and
  predicted exon chains.

All functions accept a :class:`LabelConfig` that maps integer tokens to
human-readable names and declares semantic roles (background, coding, …).
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections.abc import Iterator
from copy import deepcopy
from typing import Optional

import numpy as np
from tqdm import tqdm

from .boundary_precision import _compute_boundary_precision_landscape
from .chain_comparison import (
    _compute_intron_chain_metrics,
    _compute_chain_metrics,
    _compute_boundary_shift_metrics,
    _compute_per_transcript_exon_soft_metrics,
)
from .frame_shift import _get_frame_shift_metrics
from .intersection_over_union import _compute_intersection_over_union_score
from .state_transitions import _compute_state_change_errors
from .structure import extract_structure
from .structural_summary import _compute_structural_summary, POSITION_BIAS_HISTOGRAM_BINS
from .transcript_classification import _classify_transcript_match
from .utils import get_contiguous_groups, recursive_merge, _compute_summary_statistics, _compute_distribution_stats
from .splice_sites import eval_splice_site_junctions
from ..label_definition import LabelConfig, EvalMetrics, _DEFAULT_METRICS

# ---------------------------------------------------------------------------
# Helpers — which groups need section overlap to be computed
# ---------------------------------------------------------------------------

_SECTION_DEPENDENT_GROUPS = frozenset(
    {
        EvalMetrics.REGION_DISCOVERY,
        EvalMetrics.BOUNDARY_EXACTNESS,
    }
)

# Arrays at or above this length are treated as possible chromosome-scale input.
# On such arrays, a coding-to-coding gap can be either a true intron or an
# intergenic region between unrelated transcripts, so inference becomes
# conservative and warning-backed.
_INFER_INTRONS_LARGE_ARRAY_WARNING_LENGTH = 1_000_000

# Fallback cutoff for large arrays when the gap-length distribution has no
# clear split: infer gaps up to this multiple of a typical short gap.
_INFER_INTRONS_LARGE_GAP_RATIO = 20

# Minimum multiplicative jump between consecutive sorted gap lengths required
# to treat the distribution as two-mode: short intron-like gaps followed by
# long intergenic-like gaps.
_INFER_INTRONS_BIMODAL_MIN_JUMP_RATIO = 5.0


def _needs_section_analysis(metrics: list[EvalMetrics]) -> bool:
    """Return ``True`` if any requested metric needs section-overlap data."""
    return bool(_SECTION_DEPENDENT_GROUPS & set(metrics))


def _infer_introns_from_coding_gaps(
    labels: np.ndarray,
    label_config: LabelConfig,
) -> np.ndarray:
    """Return labels with inferred introns between adjacent coding segments.

    The function only rewrites positions that currently equal
    ``label_config.background_label``.  Existing coding, intron, splice-site,
    or other non-background labels are preserved.

    For transcript-sized arrays, every background gap between adjacent coding
    runs is interpreted as an intron.  This matches the usual representation
    produced by GFF/GTF exon or CDS painting, where exons are coding-label runs
    and introns are the unlabeled gaps between them.

    For large arrays, defined by
    :data:`_INFER_INTRONS_LARGE_ARRAY_WARNING_LENGTH`, the function emits a
    warning and applies a conservative gap-length cutoff.  Chromosome-sized
    arrays can contain both true introns and intergenic distances between
    separate transcripts; blindly filling every gap would turn intergenic
    regions into introns.  The cutoff is estimated from the coding-gap length
    distribution by :func:`_large_array_inferable_gap_cutoff`.

    Parameters
    ----------
    labels
        One-dimensional label array to transform.
    label_config
        Provides ``coding_label``, ``intron_label``, and
        ``background_label``.

    Returns
    -------
    np.ndarray
        A copy of *labels* with selected background gaps relabelled as introns.
    """
    is_large_input = len(labels) >= _INFER_INTRONS_LARGE_ARRAY_WARNING_LENGTH
    if is_large_input:
        warnings.warn(
            "infer_introns=True was requested for a large input array. "
            "Intron inference will estimate a gap-length cutoff from the "
            "coding-gap distribution: it first looks for a bimodal split "
            "between short intron-like gaps and long intergenic-like gaps, "
            "then falls back to a conservative typical-gap multiplier if no "
            "clear split is found. Prefer transcript/window-level arrays or "
            "explicit intron labels when possible.",
            stacklevel=2,
        )

    coding = label_config.coding_label
    intron = label_config.intron_label
    background = label_config.background_label

    if coding is None or intron is None:
        warnings.warn(
            "infer_introns=True requires both coding_label and intron_label; leaving labels unchanged.",
            stacklevel=2,
        )
        return labels.copy()

    inferred = labels.copy()
    coding_groups = get_contiguous_groups(np.where(inferred == coding)[0])
    gap_lengths = [
        int(right[0]) - int(left[-1]) - 1
        for left, right in zip(coding_groups, coding_groups[1:])
        if int(right[0]) > int(left[-1]) + 1
    ]
    large_gap_cutoff = _large_array_inferable_gap_cutoff(gap_lengths) if is_large_input else None

    for left, right in zip(coding_groups, coding_groups[1:]):
        gap_start = int(left[-1]) + 1
        gap_end = int(right[0])
        if gap_start >= gap_end:
            continue
        if large_gap_cutoff is not None and (gap_end - gap_start) > large_gap_cutoff:
            continue
        gap = inferred[gap_start:gap_end]
        gap[gap == background] = intron

    return inferred


def _large_array_inferable_gap_cutoff(gap_lengths: list[int]) -> int | None:
    """Estimate the largest gap length that should be inferred as intronic.

    This helper is used only for large input arrays, where coding gaps may be a
    mixture of:

    * short, intron-like gaps between exons of the same transcript
    * long, intergenic-like gaps between different transcripts

    The preferred path assumes this mixture is approximately bimodal.  It sorts
    all coding-to-coding gap lengths and finds the largest multiplicative jump
    between adjacent values.  If the jump is at least
    :data:`_INFER_INTRONS_BIMODAL_MIN_JUMP_RATIO`, the split is treated as the
    valley before the second mode.  The cutoff is set to the geometric midpoint
    between the last short gap and first long gap, i.e. a value between the two
    modes rather than inside either mode.

    If no clear jump is found, the function falls back to a conservative rule:
    infer only gaps up to
    ``median(lower_half_of_gap_lengths) * _INFER_INTRONS_LARGE_GAP_RATIO``.
    This preserves the old ``20x typical short gap`` behavior for unimodal or
    noisy distributions.

    Parameters
    ----------
    gap_lengths
        Positive lengths of background gaps between adjacent coding runs.

    Returns
    -------
    int | None
        Maximum gap length to fill as intron.  ``None`` means no cutoff could
        be estimated, so all gaps remain eligible.
    """
    if len(gap_lengths) < 2:
        return None

    sorted_gaps = np.array(sorted(gap_lengths), dtype=float)
    adjacent_ratios = sorted_gaps[1:] / np.maximum(sorted_gaps[:-1], 1.0)
    largest_jump_idx = int(np.argmax(adjacent_ratios))
    largest_jump = float(adjacent_ratios[largest_jump_idx])
    if largest_jump >= _INFER_INTRONS_BIMODAL_MIN_JUMP_RATIO:
        first_long_gap = sorted_gaps[largest_jump_idx + 1]
        last_short_gap = sorted_gaps[largest_jump_idx]
        return int(np.floor(np.sqrt(last_short_gap * first_long_gap)))

    lower_half = sorted_gaps[: max(1, len(sorted_gaps) // 2)]
    typical_gap = float(np.median(lower_half))
    if typical_gap <= 0:
        return None

    return int(typical_gap * _INFER_INTRONS_LARGE_GAP_RATIO)


# ---------------------------------------------------------------------------
# Single-sequence benchmark
# ---------------------------------------------------------------------------


def benchmark_gt_vs_pred_single(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_config: LabelConfig,
    metrics: Optional[list[EvalMetrics]] = None,
    mask_labels: Optional[np.ndarray] = None,
    infer_introns: bool = False,
) -> dict[str, dict]:
    """Compare a single ground-truth sequence against a single prediction.

    Parameters
    ----------
    gt_labels : np.ndarray
        1-D array of ground-truth nucleotide-level integer tokens.
    pred_labels : np.ndarray
        1-D array of predicted integer tokens (same length as *gt_labels*).
    label_config : LabelConfig
        Maps integer tokens to names and declares semantic roles.
    metrics : list[EvalMetrics] | None
        Which metric groups to compute.  Defaults to
        ``[REGION_DISCOVERY, BOUNDARY_EXACTNESS, NUCLEOTIDE_CLASSIFICATION]``.
    mask_labels : np.ndarray | None
        Optional boolean mask (True = exclude). Must match length of GT.
    infer_introns : bool
        If ``True``, background gaps between adjacent coding segments are
        relabelled as introns before any metric is computed.

    Returns
    -------
    dict
        Dict keyed directly by metric group name plus the transition
        analysis keys ``"transition_failures"`` and
        ``"false_transitions"``. When ``STRUCTURAL_COHERENCE`` is
        requested, the ``STRUCTURAL_COHERENCE`` entry contains:

        * ``intron_chain`` / ``intron_chain_subset`` / ``intron_chain_superset``
          — binary TP/FP/FN comparing the intron segment boundary sets,
          aggregated to corpus precision/recall across sequences.
        * ``exon_chain`` / ``exon_chain_subset`` / ``exon_chain_superset``
          — same set semantics applied to coding (exon) segments.
          Subset: pred ⊆ GT (all pred exons are real, may miss some GT).
          Superset: pred ⊇ GT (every GT exon found, may have extras).
        * ``exon_recall_per_transcript`` — float in [0, 1]: fraction of
          GT exons whose ``(start, end)`` was recovered exactly.
        * ``hallucinated_exon_count_per_transcript`` — int ≥ 0: number
          of predicted exons whose ``(start, end)`` is absent from GT.
        * ``segment_count_delta`` — ``pred_count - gt_count``
          (positive = over-segmentation).
        * ``boundary_shift_count`` / ``boundary_shift_total`` — number
          of shifted boundary positions and their summed absolute offset
          in bp across transcripts where GT and pred segment counts match.
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS
    metrics = frozenset(metrics)

    if infer_introns:
        gt_labels = _infer_introns_from_coding_gaps(gt_labels, label_config)
        pred_labels = _infer_introns_from_coding_gaps(pred_labels, label_config)

    if mask_labels is None:
        return _benchmark_chunk(gt_labels, pred_labels, label_config, metrics)

    chunk_results = [
        _benchmark_chunk(gt_labels[s:e], pred_labels[s:e], label_config, metrics)
        for s, e in _iter_unmasked_spans(mask_labels)
    ]
    return functools.reduce(recursive_merge, chunk_results, {}) if chunk_results else {}


def _iter_unmasked_spans(mask_labels: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` half-open spans of kept (unmasked) positions.

    ``mask_labels`` is True where a position must be excluded.  Each yielded
    span is a maximal run of kept positions and is evaluated independently.
    """
    is_valid = ~mask_labels.astype(bool)
    padded = np.pad(is_valid, (1, 1), mode="constant", constant_values=False)
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0]
    for s, e in zip(starts, ends):
        if e > s:
            yield int(s), int(e)


def _benchmark_chunk(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_config: LabelConfig,
    metrics: frozenset[EvalMetrics],
) -> dict[str, dict]:
    """Evaluate one fully-unmasked GT/pred span.

    ``metrics`` must already be normalised to a frozenset; intron inference
    and mask splitting are handled by :func:`benchmark_gt_vs_pred_single`.
    """
    coding = label_config.coding_label

    # Row 0 = GT, Row 1 = prediction  (NO sentinel padding here)
    arr = np.stack((gt_labels, pred_labels), axis=0)

    metric_results: dict[str, dict] = {}
    metric_results.update(_eval_transitions(arr, label_config))

    grouped_insertions = get_contiguous_groups(np.where((arr[0] != coding) & (arr[1] == coding))[0])
    grouped_deletions = get_contiguous_groups(np.where((arr[0] == coding) & (arr[1] != coding))[0])
    grouped_gt_sections = get_contiguous_groups(np.where(arr[0] == coding)[0])
    grouped_pred_sections = get_contiguous_groups(np.where(arr[1] == coding)[0])

    if EvalMetrics.INDEL in metrics:
        metric_results[EvalMetrics.INDEL.name] = _eval_indel(
            grouped_insertions, grouped_deletions, gt_labels, pred_labels, label_config
        )

    if _needs_section_analysis(metrics):
        metric_results.update(_eval_sections(grouped_gt_sections, grouped_pred_sections, metrics))

    if EvalMetrics.NUCLEOTIDE_CLASSIFICATION in metrics:
        metric_results[EvalMetrics.NUCLEOTIDE_CLASSIFICATION.name] = {
            "nucleotide": _compute_nucleotide_level_confusion(gt_labels, pred_labels, coding),
        }

    if EvalMetrics.FRAMESHIFT in metrics:
        metric_results[EvalMetrics.FRAMESHIFT.name] = _eval_frameshift(gt_labels, pred_labels, label_config)

    if EvalMetrics.STRUCTURAL_COHERENCE in metrics:
        metric_results.update(_eval_structural(gt_labels, pred_labels, label_config))

    if EvalMetrics.DIAGNOSTIC_DEPTH in metrics:
        metric_results[EvalMetrics.DIAGNOSTIC_DEPTH.name] = _compute_structural_summary(
            grouped_gt_sections, grouped_pred_sections
        )

    return metric_results


def _eval_transitions(arr: np.ndarray, label_config: LabelConfig) -> dict:
    """Return the always-on state-transition diagnostic fragments."""
    transition_analysis = _compute_state_change_errors(gt_pred_arr=arr, label_config=label_config)
    return {
        "transition_failures": transition_analysis.gt_transition_matrices,
        "false_transitions": {
            "late_catchup": transition_analysis.late_catchup_matrices,
            "premature": transition_analysis.premature_matrices,
            "spurious": transition_analysis.spurious_matrices,
            "stable_position_counts": transition_analysis.stable_position_counts,
        },
    }


def _eval_indel(
    grouped_insertions: list[np.ndarray],
    grouped_deletions: list[np.ndarray],
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_config: LabelConfig,
) -> dict:
    """Sort insertion/deletion runs into 5'/3'/whole/join-split buckets."""
    background_value = label_config.background_label

    # _classify_mismatches looks one position before/after each group, so pad
    # with one background sentinel on each side for safe access.
    padded_gt = np.concatenate(([background_value], gt_labels, [background_value]))
    padded_pred = np.concatenate(([background_value], pred_labels, [background_value]))
    padded_arr = np.stack((padded_gt, padded_pred), axis=0)

    # Shift indices by +1 to match the padded array layout
    padded_insertions = [g + 1 for g in grouped_insertions]
    padded_deletions = [g + 1 for g in grouped_deletions]

    ext5, ext3, joined, whole_ins = _classify_mismatches(
        grouped_indices=padded_insertions,
        gt_pred_arr=padded_arr,
        class_value=label_config.coding_label,
    )
    del5, del3, split, whole_del = _classify_mismatches(
        grouped_indices=padded_deletions,
        gt_pred_arr=padded_arr,
        class_value=label_config.coding_label,
    )

    return {
        "5_prime_extensions": ext5,
        "3_prime_extensions": ext3,
        "whole_insertions": whole_ins,
        "joined": joined,
        "5_prime_deletions": del5,
        "3_prime_deletions": del3,
        "whole_deletions": whole_del,
        "split": split,
    }


def _eval_sections(
    grouped_gt_sections: list[np.ndarray],
    grouped_pred_sections: list[np.ndarray],
    metrics: frozenset[EvalMetrics],
) -> dict:
    """Build REGION_DISCOVERY and/or BOUNDARY_EXACTNESS fragments.

    Only the section-overlap data required by the requested metrics is
    computed: region discovery needs the greedy 1:1 matching, boundary
    exactness needs the IoU/residual sweep.
    """
    need_region_discovery = EvalMetrics.REGION_DISCOVERY in metrics
    need_boundary_stats = EvalMetrics.BOUNDARY_EXACTNESS in metrics

    section_data = _analyze_section_overlap_and_boundaries(
        grouped_gt_section_indices=grouped_gt_sections,
        grouped_pred_section_indices=grouped_pred_sections,
        need_region_discovery=need_region_discovery,
        need_boundary_stats=need_boundary_stats,
    )

    out: dict = {}
    if need_region_discovery:
        out[EvalMetrics.REGION_DISCOVERY.name] = {
            "neighborhood_hit": section_data["neighborhood_hit"],
            "internal_hit": section_data["internal_hit"],
            "full_coverage_hit": section_data["full_coverage_hit"],
            "perfect_boundary_hit": section_data["perfect_boundary_hit"],
        }
    if need_boundary_stats:
        out[EvalMetrics.BOUNDARY_EXACTNESS.name] = {
            "first_sec_correct_3_prime_boundary": section_data["first_sec_correct_3_prime_boundary"],
            "last_sec_correct_5_prime_boundary": section_data["last_sec_correct_5_prime_boundary"],
            "iou_scores": section_data["iou_scores"],
            "fuzzy_metrics": section_data["fuzzy_metrics"],
        }
    return out


def _eval_frameshift(gt_labels: np.ndarray, pred_labels: np.ndarray, label_config: LabelConfig) -> dict:
    """Per-position reading-frame deviation between GT and predicted exons."""
    if label_config.coding_label is None:
        raise ValueError(
            "FRAMESHIFT metric requested but LabelConfig.coding_label is not "
            "set.  Provide a coding_label when constructing your LabelConfig."
        )
    return _get_frame_shift_metrics(
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        coding_value=label_config.coding_label,
    )


def _eval_structural(gt_labels: np.ndarray, pred_labels: np.ndarray, label_config: LabelConfig) -> dict:
    """Build STRUCTURAL_COHERENCE (and, when configured, splice_sites)."""
    gt_struct = extract_structure(gt_labels, label_config)
    pred_struct = extract_structure(pred_labels, label_config)

    sc_result: dict = {}
    sc_result.update(_compute_intron_chain_metrics(gt_struct, pred_struct, label_config))
    sc_result.update(_compute_per_transcript_exon_soft_metrics(gt_struct, pred_struct, label_config))

    gt_coding = gt_struct.filter_by_label(label_config.coding_label)
    pred_coding = pred_struct.filter_by_label(label_config.coding_label)

    if len(gt_coding) > 0:
        sc_result.update(_compute_chain_metrics(gt_struct, pred_struct, label_config.coding_label, "exon_chain"))
        sc_result.update(_compute_boundary_shift_metrics(gt_struct, pred_struct, label_config.coding_label))
        sc_result["segment_count_delta"] = len(pred_coding) - len(gt_coding)

        match_cls = _classify_transcript_match(gt_struct, pred_struct, label_config.coding_label)
        if match_cls is not None:
            sc_result["transcript_match_class"] = match_cls.value

    out: dict = {EvalMetrics.STRUCTURAL_COHERENCE.name: sc_result}

    if (
        label_config.intron_label is not None
        and label_config.splice_donor_label is not None
        and label_config.splice_acceptor_label is not None
    ):
        splice_confusion = eval_splice_site_junctions(gt_struct, pred_struct, label_config)
        out["splice_sites"] = dataclasses.asdict(splice_confusion)

    return out


# ---------------------------------------------------------------------------
# Multi-sequence benchmark
# ---------------------------------------------------------------------------


def benchmark_gt_vs_pred_multiple(
    gt_labels: list[np.ndarray],
    pred_labels: list[np.ndarray],
    label_config: LabelConfig,
    metrics: Optional[list[EvalMetrics]] = None,
    return_individual_results: bool = False,
    mask_labels: Optional[list[np.ndarray]] = None,
    infer_introns: bool = False,
) -> dict | list[dict]:
    """Run :func:`benchmark_gt_vs_pred_single` over paired GT/pred lists.

    Parameters
    ----------
    gt_labels, pred_labels : list[np.ndarray]
        Equally-sized lists of 1-D integer token arrays.
    label_config : LabelConfig
        Token-to-name mapping and semantic roles.
    metrics : list[EvalMetrics] | None
        Metric groups to compute.
    return_individual_results : bool
        If ``True``, return per-sequence results as a list instead of
        aggregating.
    mask_labels : list[np.ndarray] | None
        Optional boolean masks (True = exclude). Must match length of GT.
    infer_introns : bool
        If ``True``, background gaps between adjacent coding segments are
        relabelled as introns before each sequence is evaluated.

    Returns
    -------
    dict | list[dict]
        Aggregated (default) or per-sequence results.
    """
    if len(gt_labels) != len(pred_labels):
        raise ValueError(f"GT and prediction lists must have equal length, got {len(gt_labels)} vs {len(pred_labels)}.")
    if mask_labels is not None and len(mask_labels) != len(gt_labels):
        raise ValueError(f"Mask list length ({len(mask_labels)}) must match GT list length ({len(gt_labels)}).")

    metrics = deepcopy(metrics) if metrics is not None else list(_DEFAULT_METRICS)

    if EvalMetrics.FRAMESHIFT in metrics:
        warnings.warn(
            "The FRAMESHIFT metric should only be used when you are certain "
            "that the transcript contains all annotated exons.  Otherwise "
            "the results will be misleading.",
            stacklevel=2,
        )

    results = [] if return_individual_results else None
    aggregated: dict = {}
    for i in tqdm(range(len(gt_labels)), desc="Running benchmark"):
        seq_result = benchmark_gt_vs_pred_single(
            gt_labels=gt_labels[i],
            pred_labels=pred_labels[i],
            label_config=label_config,
            metrics=metrics,
            mask_labels=mask_labels[i] if mask_labels is not None else None,
            infer_introns=infer_introns,
        )
        if return_individual_results:
            results.append(seq_result)
        elif seq_result:
            recursive_merge(aggregated, seq_result)

    if return_individual_results:
        return results

    aggregated = _aggregate_summary_metrics(aggregated, metrics)

    return aggregated


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
                from collections import Counter

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


def _classify_mismatches(
    grouped_indices: list[np.ndarray],
    gt_pred_arr: np.ndarray,
    class_value: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Sort contiguous mismatch groups into four categories.

    Depending on whether the caller is analyzing insertions or deletions the
    four buckets correspond to:

    * 5'-extensions / 5'-deletions
    * 3'-extensions / 3'-deletions
    * joins / splits
    * whole insertions / whole deletions
    """
    on_5_prime: list[np.ndarray] = []
    on_3_prime: list[np.ndarray] = []
    on_both: list[np.ndarray] = []
    on_neither: list[np.ndarray] = []

    for mismatch in grouped_indices:
        if mismatch.size == 0:
            continue

        first_idx = mismatch[0]
        last_idx = mismatch[-1]

        target_on_3_prime = int(gt_pred_arr[0, last_idx + 1]) == int(gt_pred_arr[1, last_idx + 1]) == class_value

        target_on_5_prime = int(gt_pred_arr[0, first_idx - 1]) == int(gt_pred_arr[1, first_idx - 1]) == class_value

        adjusted = mismatch - 1

        if target_on_3_prime and target_on_5_prime:
            on_both.append(adjusted)
        elif target_on_3_prime:
            on_5_prime.append(adjusted)
        elif target_on_5_prime:
            on_3_prime.append(adjusted)
        else:
            on_neither.append(adjusted)

    return on_5_prime, on_3_prime, on_both, on_neither


# ---------------------------------------------------------------------------
# Section-level metrics
# ---------------------------------------------------------------------------


def _compute_nucleotide_level_confusion(
    gt_labels: np.ndarray, pred_labels: np.ndarray, class_value: int
) -> dict[str, int]:
    """Calculate granular base accuracy as a dict of confusion metrics."""
    gt_pos = gt_labels == class_value
    pred_pos = pred_labels == class_value

    return {
        "tn": int(np.count_nonzero(~gt_pos & ~pred_pos)),
        "fp": int(np.count_nonzero(~gt_pos & pred_pos)),
        "fn": int(np.count_nonzero(gt_pos & ~pred_pos)),
        "tp": int(np.count_nonzero(gt_pos & pred_pos)),
    }


def _analyze_section_overlap_and_boundaries(
    grouped_gt_section_indices: list[np.ndarray],
    grouped_pred_section_indices: list[np.ndarray],
    need_region_discovery: bool = True,
    need_boundary_stats: bool = True,
) -> dict:
    """Analyze overlap and boundary precision between section groups.

    ``need_region_discovery`` controls the greedy 1:1 matching and the
    strict-match sweep (the ``*_hit`` keys); ``need_boundary_stats`` controls
    the IoU/residual collection and terminal-boundary flags.  The returned
    dict only contains the keys for the branches that were requested.  Both
    default to ``True`` so direct callers get the full result.

    Uses ``np.searchsorted`` on sorted section endpoints to restrict the
    inner comparison to only the predicted sections that *can* overlap each
    GT section, bringing typical complexity from O(G × P) down to O(G + P).

    Region-discovery metrics (``neighborhood_hit``, ``internal_hit``,
    ``full_coverage_hit``) use greedy **1:1 matching** based on maximum
    overlap length so that each GT section is claimed by at most one
    prediction.  Predictions that cannot be matched because a better-
    fitting prediction already claimed the GT are counted as false
    positives.

    ``perfect_boundary_hit`` uses a per-prediction sweep: any prediction
    that exactly matches *some* GT section counts as a TP regardless of
    matching assignment.

    Two terminal-boundary flags are also collected:

    * ``first_sec_correct_3_prime_boundary`` — set to 1 iff *some*
      prediction's downstream end (``end`` in array coordinates) matches
      the **first** GT section's downstream end.  For multi-section
      transcripts this is the inner splice site of the leading exon, not
      the outer transcript start.
    * ``last_sec_correct_5_prime_boundary`` — set to 1 iff *some*
      prediction's upstream start matches the **last** GT section's
      upstream start.  For multi-section transcripts this is the inner
      splice site of the trailing exon, not the outer transcript end.

    Both flags use *array-orientation* semantics ("5'" = lower index,
    "3'" = higher index).  They do not account for biological strand —
    on the minus strand the array-5' end corresponds to the biological
    3' end of the transcript.
    """
    total_gt = len(grouped_gt_section_indices)
    total_pred = len(grouped_pred_section_indices)

    # Per-prediction flags for perfect-boundary (sweep-based, no 1:1 matching)
    gt_hit_strict = np.zeros(total_gt, dtype=bool)
    pred_hit_strict = np.zeros(total_pred, dtype=bool)

    boundary_residuals: list[tuple[int, int]] = []
    iou_scores: list[float] = []

    first_sec_correct_3_prime = 0
    last_sec_correct_5_prime = 0

    # Overlap candidates for 1:1 matching
    candidates: list[tuple[int, int, int]] = []  # (overlap_len, g_idx, p_idx)

    # ---- 1. Sweep — collect candidates & populate sweep-based metrics -----
    if total_gt > 0 and total_pred > 0:
        # Pre-compute (min, max) bounds arrays — sections are already sorted
        # by position because they come from contiguous-group splitting.
        pred_mins = np.array([s[0] for s in grouped_pred_section_indices])
        pred_maxs = np.array([s[-1] for s in grouped_pred_section_indices])

        # iterate over each ground truth section
        for g_idx, gt_section in enumerate(grouped_gt_section_indices):
            gt_min = int(gt_section[0])
            gt_max = int(gt_section[-1])

            # Find the range of pred sections that could overlap [gt_min, gt_max]:
            #   pred_min <= gt_max  →  candidates start from index 0 up to right_bound
            #   pred_max >= gt_min  →  candidates start from left_bound onward
            left_bound = np.searchsorted(pred_maxs, gt_min, side="left")
            right_bound = np.searchsorted(pred_mins, gt_max, side="right")

            # iterate over all possible pred sections which overlap the gt section
            for p_idx in range(left_bound, right_bound):
                p_min = int(pred_mins[p_idx])
                p_max = int(pred_maxs[p_idx])

                # --- A. Overlap (Any contact) ---
                if not (p_max < gt_min or p_min > gt_max):
                    if need_boundary_stats:
                        # Boundary residuals + IoU for every overlapping pair
                        boundary_residuals.append((p_min - gt_min, p_max - gt_max))
                        iou_scores.append(
                            _compute_intersection_over_union_score(
                                gt_start=gt_min,
                                gt_end=gt_max,
                                pred_start=p_min,
                                pred_end=p_max,
                            )
                        )

                    if need_region_discovery:
                        # Collect candidate (overlap length + indices) for 1:1 matching
                        overlap_len = min(gt_max, p_max) - max(gt_min, p_min) + 1
                        candidates.append((overlap_len, g_idx, p_idx))

                # --- B. Strict Match (sweep-based, for perfect_boundary_hit) ---
                if need_region_discovery and p_min == gt_min and p_max == gt_max:
                    gt_hit_strict[g_idx] = True
                    pred_hit_strict[p_idx] = True

                if need_boundary_stats:
                    if g_idx == 0 and p_max == gt_max:
                        first_sec_correct_3_prime = 1
                    # For the last exon, the left boundary (min) is the 5' end.
                    if g_idx == total_gt - 1 and p_min == gt_min:
                        last_sec_correct_5_prime = 1

    result: dict = {}

    if need_region_discovery:
        result.update(
            _summarise_region_discovery(
                candidates=candidates,
                grouped_gt_section_indices=grouped_gt_section_indices,
                grouped_pred_section_indices=grouped_pred_section_indices,
                total_gt=total_gt,
                total_pred=total_pred,
                gt_hit_strict=gt_hit_strict,
                pred_hit_strict=pred_hit_strict,
            )
        )

    if need_boundary_stats:
        result["first_sec_correct_3_prime_boundary"] = first_sec_correct_3_prime
        result["last_sec_correct_5_prime_boundary"] = last_sec_correct_5_prime
        result["iou_scores"] = iou_scores
        result["fuzzy_metrics"] = {"boundary_residuals": boundary_residuals, "total_gt": total_gt}

    return result


def _summarise_region_discovery(
    candidates: list[tuple[int, int, int]],
    grouped_gt_section_indices: list[np.ndarray],
    grouped_pred_section_indices: list[np.ndarray],
    total_gt: int,
    total_pred: int,
    gt_hit_strict: np.ndarray,
    pred_hit_strict: np.ndarray,
) -> dict:
    """Greedy 1:1 match the overlap candidates and classify discovery tiers.

    Candidates are claimed largest-overlap-first so each GT/pred section is
    paired with its best fit; unmatched predictions become false positives.
    ``perfect_boundary_hit`` is taken from the strict-match sweep instead and
    does not depend on the 1:1 assignment.
    """
    candidates.sort(key=lambda x: x[0], reverse=True)

    claimed_gt: set[int] = set()
    claimed_pred: set[int] = set()
    matches: list[tuple[int, int]] = []  # (g_idx, p_idx)

    for _overlap_len, g_idx, p_idx in candidates:
        # Best fits come first, so claim the GT/pred pair greedily.
        if g_idx not in claimed_gt and p_idx not in claimed_pred:
            claimed_gt.add(g_idx)
            claimed_pred.add(p_idx)
            matches.append((g_idx, p_idx))

    num_unmatched_pred = total_pred - len(matches)

    matched_neighborhood = 0  # All matches have overlap by definition
    matched_internal = 0  # Pred entirely inside GT
    matched_full_coverage = 0  # Pred fully covers GT

    for g_idx, p_idx in matches:
        gt_section = grouped_gt_section_indices[g_idx]
        pred_section = grouped_pred_section_indices[p_idx]
        gt_min = int(gt_section[0])
        gt_max = int(gt_section[-1])
        p_min = int(pred_section[0])
        p_max = int(pred_section[-1])

        # Every matched pair has overlap → neighborhood TP
        matched_neighborhood += 1

        # Internal / Envelop (prediction entirely INSIDE GT)
        if p_min >= gt_min and p_max <= gt_max:
            matched_internal += 1

        # Full Coverage / Encompass (prediction fully COVERS GT)
        if p_min <= gt_min and p_max >= gt_max:
            matched_full_coverage += 1

    return {
        # 1:1-matched discovery metrics
        "neighborhood_hit": {
            "tp": matched_neighborhood,
            "fn": total_gt - matched_neighborhood,
            "fp": num_unmatched_pred,
        },
        # Forgives under-prediction (matched prediction is inside GT).
        "internal_hit": {
            "tp": matched_internal,
            "fn": total_gt - matched_internal,
            "fp": num_unmatched_pred,
        },
        # Forgives over-prediction (matched prediction covers GT).
        "full_coverage_hit": {
            "tp": matched_full_coverage,
            "fn": total_gt - matched_full_coverage,
            "fp": num_unmatched_pred,
        },
        # Sweep-based (no 1:1 matching) — handles fragmented predictions well
        "perfect_boundary_hit": {
            "tp": int(np.sum(gt_hit_strict)),
            "fn": int(total_gt - np.sum(gt_hit_strict)),
            "fp": int(total_pred - np.sum(pred_hit_strict)),
        },
    }
