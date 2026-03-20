"""Core evaluation logic for the DNA segmentation benchmark.

Compares ground-truth nucleotide-level annotations with predicted annotations
and computes a rich set of metrics:

* **INDEL** – 5'/3' extensions/deletions, whole insertions/deletions,
  joins/splits.
* **REGION_DISCOVERY** – precision & recall at four overlap strictness
  levels (neighbourhood, internal, full-coverage, perfect-boundary).
* **BOUNDARY_EXACTNESS** – IoU statistics, boundary-residual bias /
  reliability landscape, inner / all section-boundary precision & recall,
  terminal-boundary flags.
* **NUCLEOTIDE_CLASSIFICATION** – per-base precision, recall, and F1.
* **FRAMESHIFT** – per-position reading-frame deviation between GT and
  predicted exon chains.

All functions accept a :class:`LabelConfig` that maps integer tokens to
human-readable names and declares semantic roles (background, coding, …).
"""

from __future__ import annotations

import functools
import warnings
from copy import deepcopy
from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from .boundary_precision import _compute_boundary_precision_landscape
from .frame_shift import _get_frame_shift_metrics
from .intersection_over_union import _compute_intersection_over_union_score
from .state_transitions import _compute_state_change_errors
from .utils import get_contiguous_groups, recursive_merge, _compute_summary_statistics, _compute_distribution_stats
from ..label_definition import LabelConfig, EvalMetrics, _DEFAULT_METRICS


# ---------------------------------------------------------------------------
# Helpers — which groups need section overlap to be computed
# ---------------------------------------------------------------------------

_SECTION_DEPENDENT_GROUPS = frozenset({
    EvalMetrics.REGION_DISCOVERY,
    EvalMetrics.BOUNDARY_EXACTNESS,
})


def _needs_section_analysis(metrics: list[EvalMetrics]) -> bool:
    """Return ``True`` if any requested metric needs section-overlap data."""
    return bool(_SECTION_DEPENDENT_GROUPS & set(metrics))


# ---------------------------------------------------------------------------
# Single-sequence benchmark
# ---------------------------------------------------------------------------


def benchmark_gt_vs_pred_single(
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
        label_config: LabelConfig,
        classes: list[int],
        metrics: Optional[list[EvalMetrics]] = None,
        mask_labels: Optional[np.ndarray] = None,
) -> dict[str, dict[str, dict]]:
    """Compare a single ground-truth sequence against a single prediction.

    Parameters
    ----------
    gt_labels : np.ndarray
        1-D array of ground-truth nucleotide-level integer tokens.
    pred_labels : np.ndarray
        1-D array of predicted integer tokens (same length as *gt_labels*).
    label_config : LabelConfig
        Maps integer tokens to names and declares semantic roles.
    classes : list[int]
        Which token values to compute metrics for (e.g. ``[0, 2]`` for
        EXON and INTRON).
    metrics : list[EvalMetrics] | None
        Which metric groups to compute.  Defaults to
        ``[REGION_DISCOVERY, BOUNDARY_EXACTNESS, NUCLEOTIDE_CLASSIFICATION]``.
    mask_labels : np.ndarray | None
        Optional boolean mask (True = exclude). Must match length of GT.

    Returns
    -------
    dict
        Nested dict keyed by human-readable class name → metric group → results.
    """
    if mask_labels is not None:
        is_valid = ~mask_labels.astype(bool)
        padded = np.pad(is_valid, (1, 1), mode='constant', constant_values=False)
        starts = np.where(~padded[:-1] & padded[1:])[0]
        ends = np.where(padded[:-1] & ~padded[1:])[0]
        
        chunk_results = []
        for s, e in zip(starts, ends):
            if e > s:
                # Remove mask_labels to avoid infinite recursion
                chunk_res = benchmark_gt_vs_pred_single(
                    gt_labels=gt_labels[s:e],
                    pred_labels=pred_labels[s:e],
                    label_config=label_config,
                    classes=classes,
                    metrics=metrics,
                    mask_labels=None,
                )
                chunk_results.append(chunk_res)
        
        return functools.reduce(recursive_merge, chunk_results, {}) if chunk_results else {}

    if metrics is None:
        metrics = _DEFAULT_METRICS

    background_value = label_config.background_label

    # Row 0 = GT, Row 1 = prediction  (NO sentinel padding here)
    arr = np.stack((gt_labels, pred_labels), axis=0)

    metric_results: dict[str, dict] = {}

    transition_analysis = _compute_state_change_errors(gt_pred_arr=arr, label_config=label_config)
    metric_results["transition_failures"] = transition_analysis.gt_transition_matrices
    metric_results["false_transitions"] = {
        "matrices": transition_analysis.false_transition_matrices,
        "totals": transition_analysis.false_transition_totals,
    }

    for class_token in classes:
        class_name = label_config.name_of(class_token)
        metric_results[class_name] = {}

        # Boolean masks for insertions / deletions
        insertion_mask = (arr[0, :] != class_token) & (arr[1, :] == class_token)
        insertion_indices = np.where(insertion_mask)[0]

        deletion_mask = (arr[0, :] == class_token) & (arr[1, :] != class_token)
        deletion_indices = np.where(deletion_mask)[0]

        # Contiguous sections in GT and prediction
        gt_section_indices = np.where(arr[0, :] == class_token)[0]
        pred_section_indices = np.where(arr[1, :] == class_token)[0]

        grouped_insertions = get_contiguous_groups(insertion_indices)
        grouped_deletions = get_contiguous_groups(deletion_indices)
        grouped_gt_sections = get_contiguous_groups(gt_section_indices)
        grouped_pred_sections = get_contiguous_groups(pred_section_indices)

        # ---- INDEL metrics ------------------------------------------------
        if EvalMetrics.INDEL in metrics or _needs_section_analysis(metrics):
            # _classify_mismatches looks one position before/after each group,
            # so pad with one background sentinel on each side for safe access.
            padded_gt = np.concatenate(([background_value], gt_labels, [background_value]))
            padded_pred = np.concatenate(([background_value], pred_labels, [background_value]))
            padded_arr = np.stack((padded_gt, padded_pred), axis=0)

            # Shift indices by +1 to match the padded array layout
            padded_insertions = [g + 1 for g in grouped_insertions]
            padded_deletions = [g + 1 for g in grouped_deletions]

            ext5, ext3, joined, whole_ins = _classify_mismatches(
                grouped_indices=padded_insertions,
                gt_pred_arr=padded_arr,
                class_value=class_token,
            )
            del5, del3, split, whole_del = _classify_mismatches(
                grouped_indices=padded_deletions,
                gt_pred_arr=padded_arr,
                class_value=class_token,
            )

            indel_results = {
                "5_prime_extensions": ext5,
                "3_prime_extensions": ext3,
                "whole_insertions": whole_ins,
                "joined": joined,
                "5_prime_deletions": del5,
                "3_prime_deletions": del3,
                "whole_deletions": whole_del,
                "split": split,
            }
            if EvalMetrics.INDEL in metrics:
                metric_results[class_name][EvalMetrics.INDEL.name] = indel_results

        # ---- Section-overlap analysis (shared by REGION_DISCOVERY & BOUNDARY_EXACTNESS)
        if _needs_section_analysis(metrics):
            section_data = _analyze_section_overlap_and_boundaries(
                grouped_gt_section_indices=grouped_gt_sections,
                grouped_pred_section_indices=grouped_pred_sections,
            )

            # -- REGION_DISCOVERY: precision & recall at four strictness levels
            if EvalMetrics.REGION_DISCOVERY in metrics:
                metric_results[class_name][EvalMetrics.REGION_DISCOVERY.name] = {
                    "neighborhood_hit": section_data["neighborhood_hit"],
                    "internal_hit": section_data["internal_hit"],
                    "full_coverage_hit": section_data["full_coverage_hit"],
                    "perfect_boundary_hit": section_data["perfect_boundary_hit"],
                }

            # -- BOUNDARY_EXACTNESS: IoU, boundary residuals, section-boundary flags
            if EvalMetrics.BOUNDARY_EXACTNESS in metrics:
                metric_results[class_name][EvalMetrics.BOUNDARY_EXACTNESS.name] = {
                    "inner_section_boundaries": section_data["inner_section_boundaries"],
                    "all_section_boundaries": section_data["all_section_boundaries"],
                    "first_sec_correct_3_prime_boundary": section_data["first_sec_correct_3_prime_boundary"],
                    "last_sec_correct_5_prime_boundary": section_data["last_sec_correct_5_prime_boundary"],
                    "iou_scores": section_data["iou_scores"],
                    "fuzzy_metrics": section_data["fuzzy_metrics"],
                }

        # ---- NUCLEOTIDE_CLASSIFICATION: per-base precision, recall, F1 ----
        if EvalMetrics.NUCLEOTIDE_CLASSIFICATION in metrics:
            nuc_confusion = _compute_nucleotide_level_confusion(gt_labels, pred_labels, class_token)
            metric_results[class_name][EvalMetrics.NUCLEOTIDE_CLASSIFICATION.name] = {
                "nucleotide": nuc_confusion,
            }

        # ---- Frameshift metrics -------------------------------------------
        if EvalMetrics.FRAMESHIFT in metrics:
            coding_value = label_config.coding_label
            if coding_value is None:
                raise ValueError(
                    "FRAMESHIFT metric requested but LabelConfig.coding_label "
                    "is not set.  Provide a coding_label when constructing "
                    "your LabelConfig."
                )
            if class_token == coding_value:
                metric_results[class_name][EvalMetrics.FRAMESHIFT.name] = (
                    _get_frame_shift_metrics(
                        gt_labels=gt_labels,
                        pred_labels=pred_labels,
                        coding_value=coding_value,
                    )
                )

    return metric_results


# ---------------------------------------------------------------------------
# Multi-sequence benchmark
# ---------------------------------------------------------------------------


def benchmark_gt_vs_pred_multiple(
        gt_labels: list[np.ndarray],
        pred_labels: list[np.ndarray],
        label_config: LabelConfig,
        classes: list[int],
        metrics: Optional[list[EvalMetrics]] = None,
        return_individual_results: bool = False,
        mask_labels: Optional[list[np.ndarray]] = None,
) -> dict | list[dict]:
    """Run :func:`benchmark_gt_vs_pred_single` over paired GT/pred lists.

    Parameters
    ----------
    gt_labels, pred_labels : list[np.ndarray]
        Equally-sized lists of 1-D integer token arrays.
    label_config : LabelConfig
        Token-to-name mapping and semantic roles.
    classes : list[int]
        Token values for which to compute metrics.
    metrics : list[EvalMetrics] | None
        Metric groups to compute.
    return_individual_results : bool
        If ``True``, return per-sequence results as a list instead of
        aggregating.
    mask_labels : list[np.ndarray] | None
        Optional boolean masks (True = exclude). Must match length of GT.

    Returns
    -------
    dict | list[dict]
        Aggregated (default) or per-sequence results.
    """
    if len(gt_labels) != len(pred_labels):
        raise ValueError(
            f"GT and prediction lists must have equal length, "
            f"got {len(gt_labels)} vs {len(pred_labels)}."
        )
    if mask_labels is not None and len(mask_labels) != len(gt_labels):
        raise ValueError(
            f"Mask list length ({len(mask_labels)}) must match "
            f"GT list length ({len(gt_labels)})."
        )

    metrics = deepcopy(metrics) if metrics is not None else list(_DEFAULT_METRICS)

    if EvalMetrics.FRAMESHIFT in metrics:
        warnings.warn(
            "The FRAMESHIFT metric should only be used when you are certain "
            "that the transcript contains all annotated exons.  Otherwise "
            "the results will be misleading.",
            stacklevel=2,
        )

    results = []
    for i in tqdm(range(len(gt_labels)), desc="Running benchmark"):
        seq_result = benchmark_gt_vs_pred_single(
            gt_labels=gt_labels[i],
            pred_labels=pred_labels[i],
            label_config=label_config,
            classes=classes,
            metrics=metrics,
            mask_labels=mask_labels[i] if mask_labels is not None else None,
        )
        results.append(seq_result)

    if return_individual_results:
        return results

    aggregated = functools.reduce(recursive_merge, [res for res in results if res], {})

    aggregated = _aggregate_summary_metrics(aggregated, metrics)

    return aggregated


def _aggregate_summary_metrics(aggregated: dict, metrics: list[EvalMetrics]) -> dict:
    """Compute user-facing summary statistics from raw accumulated counts.

    After multi-sequence merging, the raw tp/fn/fp lists are converted into
    precision & recall (and F1 for nucleotide level).  Raw counts are
    *replaced* by the computed summaries so they are not exposed to the user.
    """
    for _class_name, class_results in aggregated.items():
        if _class_name == "transition_failures":
            continue
        if _class_name == "false_transitions":
            # recursive_merge sums np.ndarray (matrices) element-wise already.
            # It wraps int values (totals) into lists — sum them back.
            class_results["totals"] = {
                k: sum(v) if isinstance(v, list) else v
                for k, v in class_results["totals"].items()
            }
            continue

        # -- REGION_DISCOVERY: precision & recall per strictness level ------
        if EvalMetrics.REGION_DISCOVERY in metrics:
            rd = class_results[EvalMetrics.REGION_DISCOVERY.name]
            for level_key in ("neighborhood_hit", "internal_hit",
                              "full_coverage_hit", "perfect_boundary_hit"):
                rd[level_key] = _compute_summary_statistics(**rd[level_key])

        # -- BOUNDARY_EXACTNESS: IoU stats + landscape + section boundaries -
        if EvalMetrics.BOUNDARY_EXACTNESS in metrics:
            be = class_results[EvalMetrics.BOUNDARY_EXACTNESS.name]

            # Inner / all section boundary precision & recall
            for boundary_key in ("inner_section_boundaries", "all_section_boundaries"):
                be[boundary_key] = _compute_summary_statistics(**be[boundary_key])

            # IoU distribution statistics
            if "iou_scores" in be:
                be["iou_stats"] = _compute_distribution_stats(be["iou_scores"], is_abs=False)

            # Boundary-residual bias / reliability landscape
            if "fuzzy_metrics" in be:
                be["fuzzy_metrics"] = _compute_boundary_precision_landscape(
                    residuals=be["fuzzy_metrics"]["boundary_residuals"],
                    total_gt_count=sum(be["fuzzy_metrics"]["total_gt"]),
                )

        # -- NUCLEOTIDE_CLASSIFICATION: precision, recall, F1 ---------------
        if EvalMetrics.NUCLEOTIDE_CLASSIFICATION in metrics:
            nc = class_results[EvalMetrics.NUCLEOTIDE_CLASSIFICATION.name]
            nuc_counts = nc["nucleotide"]
            summary = _compute_summary_statistics(**nuc_counts)
            # Add F1 (only meaningful at nucleotide level)
            p, r = summary.get("precision", 0), summary.get("recall", 0)
            summary["f1"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
            nc["nucleotide"] = summary

    return aggregated


def _classify_mismatches(
        grouped_indices: list[np.ndarray],
        gt_pred_arr: np.ndarray,
        class_value: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Sort contiguous mismatch groups into four categories.

    Depending on whether the caller is analysing insertions or deletions the
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

        target_on_3_prime = (
                int(gt_pred_arr[0, last_idx + 1])
                == int(gt_pred_arr[1, last_idx + 1])
                == class_value
        )

        target_on_5_prime = (
                int(gt_pred_arr[0, first_idx - 1])
                == int(gt_pred_arr[1, first_idx - 1])
                == class_value
        )

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
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
        class_value: int
) -> dict[str, int]:
    """Calculate granular base accuracy as a dict of confusion metrics."""
    binary_gt = np.where(gt_labels == class_value, 1, 0)
    binary_pred = np.where(pred_labels == class_value, 1, 0)

    # labels=[0, 1] ensures a 2x2 matrix
    cm = confusion_matrix(binary_gt, binary_pred, labels=[0, 1])
    nuc_tn, nuc_fp, nuc_fn, nuc_tp = map(int, cm.ravel())
    
    return {"tn": nuc_tn, "fp": nuc_fp, "fn": nuc_fn, "tp": nuc_tp}


def _analyze_section_overlap_and_boundaries(
        grouped_gt_section_indices: list[np.ndarray],
        grouped_pred_section_indices: list[np.ndarray],
) -> dict:
    """Analyze overlap and boundary precision between section groups.

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
    """
    total_gt = len(grouped_gt_section_indices)
    total_pred = len(grouped_pred_section_indices)

    # Per-prediction flags for perfect-boundary (sweep-based, no 1:1 matching)
    gt_hit_strict = np.zeros(total_gt, dtype=bool)
    pred_hit_strict = np.zeros(total_pred, dtype=bool)

    boundary_residuals: list[tuple[int, int]] = []
    iou_scores: list[float] = []

    fully_matching_sections = 0
    inner_boundary_matching_sections = 0
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
                    # Boundary residuals (computed for every overlapping pair)
                    res_5p = p_min - gt_min
                    res_3p = p_max - gt_max
                    boundary_residuals.append((res_5p, res_3p))

                    # IoU (computed for every overlapping pair)
                    iou_scores.append(
                        _compute_intersection_over_union_score(
                            gt_start=gt_min, gt_end=gt_max,
                            pred_start=p_min, pred_end=p_max,
                        )
                    )

                    # Collect candidate for 1:1 matching
                    overlap_start = max(gt_min, p_min)
                    overlap_end = min(gt_max, p_max)
                    overlap_len = overlap_end - overlap_start + 1
                    # add discovered overlap to list of candidates
                    # store the indices of the gt and pred exon
                    candidates.append((overlap_len, g_idx, p_idx))

                # --- B. Strict Match (sweep-based, for perfect_boundary_hit) ---
                if p_min == gt_min and p_max == gt_max:
                    gt_hit_strict[g_idx] = True
                    pred_hit_strict[p_idx] = True
                    fully_matching_sections += 1

                    # Internal/Boundary Analysis
                    if 0 < g_idx < total_gt - 1:
                        inner_boundary_matching_sections += 1

                if g_idx == 0 and p_max == gt_max:
                    first_sec_correct_3_prime = 1

                # For the last exon, the left boundary (min) is the 5' end.
                if g_idx == total_gt - 1 and p_min == gt_min:
                    last_sec_correct_5_prime = 1

    # ---- 2. Greedy 1:1 matching (largest overlap first) -------------------
    # handle internal_hit neighborhood hit and full coverage hit
    candidates.sort(key=lambda x: x[0], reverse=True)

    claimed_gt: set[int] = set()
    claimed_pred: set[int] = set()
    matches: list[tuple[int, int]] = []  # (g_idx, p_idx)

    # iterate over all candidates
    for _overlap_len, g_idx, p_idx in candidates:
        # since its ordered best fits come first
        # claim set gt section and mark pred as assigend
        if g_idx not in claimed_gt and p_idx not in claimed_pred:
            claimed_gt.add(g_idx)
            claimed_pred.add(p_idx)
            matches.append((g_idx, p_idx))

    num_unmatched_pred = total_pred - len(matches)

    # ---- 3. Classify matched pairs for discovery tiers --------------------
    matched_neighborhood = 0   # All matches have overlap by definition
    matched_internal = 0       # Pred entirely inside GT
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

    # ---- 4. Sequence-Level Aggregates -------------------------------------
    num_inner_expected = total_gt - 2 if total_gt > 2 else (1 if total_gt == 2 else 0)
    if total_gt > 1:
        inner_boundaries = {
            "tp": 1 if inner_boundary_matching_sections == num_inner_expected and total_pred > 0 else 0,
            "fp": 1 if inner_boundary_matching_sections != num_inner_expected and total_pred > 0 else 0,
            "fn": 1 if total_pred == 0 else 0,
            "tn": 0,
        }
    elif total_gt == 0 and total_pred > 0:
        # No GT sections but predictions exist → false positive
        inner_boundaries = {"tp": 0, "fp": 1, "fn": 0, "tn": 0}
        all_section_boundaries = {"tp": 0, "fp": 1, "fn": 0, "tn": 0}
    else:
        # total_gt <= 1 and no spurious predictions → nothing to evaluate
        inner_boundaries = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        all_section_boundaries = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    if total_gt > 0:
        all_section_boundaries = {
            "tp": 1 if fully_matching_sections == total_gt else 0,
            "fp": 1 if (fully_matching_sections != total_gt and total_pred > 0) or (total_gt == 0 and total_pred > 0) else 0,
            "fn": 1 if total_pred == 0 else 0,
            "tn": 0,
        }

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
            "fp": int(total_pred - np.sum(pred_hit_strict))
        },
        "inner_section_boundaries": inner_boundaries,
        "all_section_boundaries": all_section_boundaries,
        "first_sec_correct_3_prime_boundary": first_sec_correct_3_prime,
        "last_sec_correct_5_prime_boundary": last_sec_correct_5_prime,
        "iou_scores": iou_scores,
        "fuzzy_metrics": {
            "boundary_residuals": boundary_residuals,
            "total_gt": total_gt
        }
    }
