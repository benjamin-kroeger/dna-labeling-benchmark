"""Core evaluation logic for the DNA segmentation benchmark.

Compares ground-truth nucleotide-level annotations with predicted annotations
and computes a rich set of metrics:

* **INDEL** – 5'/3' extensions/deletions, whole insertions/deletions,
  joins/splits.
* **SECTION** – nucleotide-level, encompassing-section, strict-section,
  inner-boundary, and total-boundary confusion counts.
* **ML** – aggregated precision & recall across section levels (computed from
  the SECTION counts after multi-sequence merging).
* **FRAMESHIFT** – per-position reading-frame deviation between GT and
  predicted exon chains.

All functions accept a :class:`LabelConfig` that maps integer tokens to
human-readable names and declares semantic roles (background, coding, …).
"""

from __future__ import annotations

import functools
import warnings
from copy import deepcopy
from enum import Enum
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from .label_definition import LabelConfig


# ---------------------------------------------------------------------------
# Public Enums
# ---------------------------------------------------------------------------


class EvalMetrics(Enum):
    """Available evaluation metric groups."""

    INDEL = 0
    SECTION = 1
    ML = 2  # summary statistics from SECTION (single-seq not computed directly)
    _MLMULTIPLE = 3  # reserved for cross-sequence averaging
    FRAMESHIFT = 4


_DEFAULT_METRICS = [EvalMetrics.SECTION, EvalMetrics.ML]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def get_contiguous_groups(indices: np.ndarray) -> list[np.ndarray]:
    """Split *indices* into sub-arrays of contiguous runs."""
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    return np.split(indices, breaks)


# ---------------------------------------------------------------------------
# Single-sequence benchmark
# ---------------------------------------------------------------------------


def benchmark_gt_vs_pred_single(
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
        label_config: LabelConfig,
        classes: list[int],
        metrics: Optional[list[EvalMetrics]] = None,
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
        Which metric groups to compute.  Defaults to ``[SECTION, ML]``.

    Returns
    -------
    dict
        Nested dict keyed by human-readable class name → metric group → results.
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS

    background_value = label_config.background_label

    # Pad with one background sentinel on each side for safe look-ahead/-behind
    gt_labels = np.concatenate(([background_value], gt_labels, [background_value]))
    pred_labels = np.concatenate(([background_value], pred_labels, [background_value]))

    # Row 0 = GT, Row 1 = prediction
    arr = np.stack((gt_labels, pred_labels), axis=0)

    metric_results: dict[str, dict] = {}

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
        if EvalMetrics.INDEL in metrics or EvalMetrics.SECTION in metrics:
            ext5, ext3, joined, whole_ins = _classify_mismatches(
                grouped_indices=grouped_insertions,
                gt_pred_arr=arr,
                class_value=class_token,
            )
            del5, del3, split, whole_del = _classify_mismatches(
                grouped_indices=grouped_deletions,
                gt_pred_arr=arr,
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
            metric_results[class_name][EvalMetrics.INDEL.name] = indel_results

        # ---- Section-level metrics ----------------------------------------
        if EvalMetrics.SECTION in metrics:
            confusion = _get_metrics_across_levels(
                grouped_gt_section_indices=grouped_gt_sections,
                grouped_pred_section_indices=grouped_pred_sections,
                gt_labels=gt_labels,
                pred_labels=pred_labels,
                class_value=class_token,
            )
            metric_results[class_name][EvalMetrics.SECTION.name] = confusion

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

    Returns
    -------
    dict | list[dict]
        Aggregated (default) or per-sequence results.
    """
    assert len(gt_labels) == len(pred_labels), (
        "There must be equally many ground-truth and prediction sequences."
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
        )
        results.append(seq_result)

    if return_individual_results:
        return results

    aggregated = functools.reduce(recursive_merge, results, {})

    if EvalMetrics.ML in metrics:
        for _class_name, class_results in aggregated.items():
            class_results[EvalMetrics.ML.name] = {}
            section = class_results[EvalMetrics.SECTION.name]
            ml = class_results[EvalMetrics.ML.name]
            ml["nucleotide_level_metrics"] = _compute_summary_statistics(**section["nucleotide"])
            ml["encompass_section_match_metrics"] = _compute_summary_statistics(**section["section"])
            ml["strict_section_match_metrics"] = _compute_summary_statistics(**section["strict_section"])
            ml["correct_inner_section_boundaries_metrics"] = _compute_summary_statistics(**section["inner_section_boundaries"])
            ml["correct_overall_section_boundaries_metrics"] = _compute_summary_statistics(**section["all_section_boundaries"])

    return aggregated


# ---------------------------------------------------------------------------
# Dictionary merging
# ---------------------------------------------------------------------------


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
            else:
                target[key] = [target_value, source_value]
    return target


# ---------------------------------------------------------------------------
# INDEL classification
# ---------------------------------------------------------------------------


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


import numpy as np
from sklearn.metrics import confusion_matrix

import numpy as np
from sklearn.metrics import confusion_matrix

import numpy as np
from sklearn.metrics import confusion_matrix

import numpy as np
from sklearn.metrics import confusion_matrix


def _compute_intersection_over_union_score(gt_start: int, gt_end: int, pred_start, pred_end):
    # Intersection
    i_start = max(gt_start, pred_start)
    i_end = min(gt_end, pred_end)
    intersect_len = max(0, i_end - i_start + 1)

    # Union
    u_start = min(gt_start, pred_start)
    u_end = max(gt_end, pred_end)
    union_len = u_end - u_start + 1

    iou = intersect_len / union_len if union_len > 0 else 0.0

    return iou


def _get_metrics_across_levels(
        grouped_gt_section_indices: list[np.ndarray],
        grouped_pred_section_indices: list[np.ndarray],
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
        class_value: int,
) -> dict:
    """
    Compute confusion counts at nucleotide and section levels with split/merge tracking.

    ===========================================================================
    THE METRIC HIERARCHY (From Forgiving to Punishing)
    ===========================================================================
    1. Overlap:   "Neighborhood" Discovery. Any contact between GT and Pred.
    2. Envelop:   Under-prediction support. Pred is a smaller segment INSIDE GT.
    3. Encompass: Over-prediction support. Pred is a larger segment COVERING GT.
    4. Strict:    Perfect Identity. Coordinates match exactly (The Double Penalty).
    ===========================================================================
    """

    # ---- 1. Nucleotide level (Granular Base Accuracy) ---------------------
    binary_gt = np.where(gt_labels == class_value, 1, 0)
    binary_pred = np.where(pred_labels == class_value, 1, 0)

    # labels=[0, 1] ensures a 2x2 matrix; [1:-1] slices off prepended/appended tags.
    cm = confusion_matrix(binary_gt[1:-1], binary_pred[1:-1], labels=[0, 1])
    nuc_tn, nuc_fp, nuc_fn, nuc_tp = map(int, cm.ravel())

    # ---- 2. Initialize Tracking -------------------------------------------
    total_gt = len(grouped_gt_section_indices)
    total_pred = len(grouped_pred_section_indices)

    gt_hit_overlap = np.zeros(total_gt, dtype=bool)
    pred_hit_overlap = np.zeros(total_pred, dtype=bool)
    gt_hit_envelop = np.zeros(total_gt, dtype=bool)  # Forgives Under-pred
    gt_hit_encompass = np.zeros(total_gt, dtype=bool)  # Forgives Over-pred
    gt_hit_strict = np.zeros(total_gt, dtype=bool)
    pred_hit_strict = np.zeros(total_pred, dtype=bool)

    # New Metric Stores
    iou_scores = []

    fully_matching_sections = 0
    inner_boundary_matching_sections = 0
    first_sec_correct_3_prime = 0
    last_sec_correct_5_prime = 0

    # ---- 3. Mapping Logic -------------------------------------------------
    for g_idx, gt_section in enumerate(grouped_gt_section_indices):
        gt_min, gt_max = np.min(gt_section), np.max(gt_section)

        for p_idx, pred_section in enumerate(grouped_pred_section_indices):
            p_min, p_max = np.min(pred_section), np.max(pred_section)

            # --- A. Overlap (Any contact) ---
            if not (p_max < gt_min or p_min > gt_max):
                gt_hit_overlap[g_idx] = True
                pred_hit_overlap[p_idx] = True

                # --- Metrics: IoU ----
                iou_scores.append(_compute_intersection_over_union_score(gt_start=gt_min, gt_end=gt_max, pred_start=p_min, pred_end=p_max))

                # --- B. Envelop (Prediction is entirely INSIDE GT) ---
                if p_min >= gt_min and p_max <= gt_max:
                    gt_hit_envelop[g_idx] = True

                # --- C. Encompass (Prediction fully COVERS GT) ---
                if p_min <= gt_min and p_max >= gt_max:
                    gt_hit_encompass[g_idx] = True

            # --- D. Strict Match (Coordinates match exactly) ---
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

    # ---- 4. Sequence-Level Aggregates -------------------------------------
    num_inner_expected = total_gt - 2 if total_gt > 2 else (1 if total_gt == 2 else 0)
    inner_boundaries = {
        "tp": 1 if total_gt > 1 and inner_boundary_matching_sections == num_inner_expected and total_pred > 0 else 0,
        "fp": 1 if total_gt > 1 and inner_boundary_matching_sections != num_inner_expected and total_pred > 0 else 0,
        "fn": 1 if total_gt > 1 and total_pred == 0 else 0,
        "tn": 0,
    } if total_gt > 1 else {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

    return {
        "nucleotide": {"tn": nuc_tn, "fp": nuc_fp, "fn": nuc_fn, "tp": nuc_tp},
        "neighborhood_hit": {
            "tp": int(np.sum(gt_hit_overlap)),
            "fn": int(total_gt - np.sum(gt_hit_overlap)),
            "fp": int(total_pred - np.sum(pred_hit_overlap))
        },
        # Forgives under-prediction (prediction segment is inside GT).
        "internal_hit": {
            "tp": int(np.sum(gt_hit_envelop)),
            "fn": int(total_gt - np.sum(gt_hit_envelop)),
        },
        # Forgives over-prediction (prediction segment covers GT).
        "full_coverage_hit": {
            "tp": int(np.sum(gt_hit_encompass)),
            "fn": int(total_gt - np.sum(gt_hit_encompass)),
        },
        "perfect_boundary_hit": {
            "tp": int(np.sum(gt_hit_strict)),
            "fn": int(total_gt - np.sum(gt_hit_strict)),
            "fp": int(total_pred - np.sum(pred_hit_strict))
        },
        "inner_section_boundaries": inner_boundaries,
        "all_section_boundaries": {
            "tp": 1 if total_gt > 0 and fully_matching_sections == total_gt else 0,
            "fp": 1 if total_gt > 0 and fully_matching_sections != total_gt and total_pred > 0 else 0,
            "fn": 1 if total_gt > 0 and total_pred == 0 else 0,
            "tn": 0
        },
        "first_sec_correct_3_prime_boundary": first_sec_correct_3_prime,
        "last_sec_correct_5_prime_boundary": last_sec_correct_5_prime,
        "iou_scores": iou_scores

    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _compute_summary_statistics(fn: list, tp: list, fp: list, tn: list) -> dict:
    """Compute precision and recall from aggregated confusion counts."""
    total_tp = sum(tp)
    total_fp = sum(fp)
    total_fn = sum(fn)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    return {"precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Frameshift metrics
# ---------------------------------------------------------------------------


def _get_frame_shift_metrics(
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
        coding_value: int,
) -> dict:
    """Compute per-position reading-frame deviation."""
    gt_exon_indices = np.where(gt_labels == coding_value)[0]
    pred_exon_indices = np.where(pred_labels == coding_value)[0]

    if len(gt_exon_indices) == 0 or len(pred_exon_indices) == 0:
        return {"gt_frames": []}

    assert len(gt_exon_indices) % 3 == 0, "There is no clear codon usage"

    gt_codons = gt_exon_indices.reshape(-1, 3)
    possible_pred_codons = sliding_window_view(pred_exon_indices, 3)

    gt_codon_view = gt_codons.view([("", gt_codons.dtype)] * 3).reshape(-1)
    pred_codon_view = possible_pred_codons.view(
        [("", possible_pred_codons.dtype)] * 3
    ).reshape(-1)
    _common_codons = np.intersect1d(gt_codon_view, pred_codon_view)

    valid_mask = (
            np.isin(np.arange(len(gt_labels)), gt_exon_indices)
            & np.isin(np.arange(len(gt_labels)), pred_exon_indices)
    )

    frame_list = np.full(len(gt_labels), np.inf)

    gt_cumsum = np.searchsorted(gt_exon_indices, np.arange(len(gt_labels)), side="right")
    pred_cumsum = np.searchsorted(pred_exon_indices, np.arange(len(gt_labels)), side="right")

    frame_list[valid_mask] = np.abs(pred_cumsum[valid_mask] - gt_cumsum[valid_mask]) % 3

    return {"gt_frames": frame_list[1:-1]}
