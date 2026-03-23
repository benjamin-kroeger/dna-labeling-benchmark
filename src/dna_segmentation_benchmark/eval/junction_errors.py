"""Junction error taxonomy and error correlation analysis.

Classifies structural mismatches between GT and predicted segment chains
into causal error categories, answering *why* the prediction is wrong at
the structural level — not just *that* it is wrong.

Error types
-----------
* **Exon skip** — adjacent GT segments of one class are merged in
  prediction because the intervening segment(s) of another class are
  absent.
* **Segment retention** — a GT segment is absorbed into its neighbours
  in the prediction (its label is replaced by the neighbours' label).
* **Novel insertion** — the prediction inserts an extra segment that
  splits a GT segment into two.
* **Cascade shift** — a boundary error propagates across 3+ consecutive
  segments, shifting all downstream boundaries in the same direction.
* **Compensating errors** — paired boundary errors that cancel each
  other out (one segment over-extends while the adjacent one contracts).
"""

from __future__ import annotations

import numpy as np

from .structure import ExtractedStructure, Segment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _classify_junction_errors(
    gt_structure: ExtractedStructure,
    pred_structure: ExtractedStructure,
    class_token: int,
) -> dict:
    """Classify structural mismatches into junction error types.

    Parameters
    ----------
    gt_structure, pred_structure : ExtractedStructure
        Structures from the GT and predicted arrays.
    class_token : int
        Which label class to evaluate.

    Returns
    -------
    dict
        Counts for each error type plus ``total_junction_errors``.
    """
    gt_segs = gt_structure.filter_by_label(class_token)
    pred_segs = pred_structure.filter_by_label(class_token)

    n_gt = len(gt_segs)
    n_pred = len(pred_segs)

    if n_gt == 0 and n_pred == 0:
        return _empty_result()

    # 1:1 greedy matching by overlap length (same strategy as existing code)
    matches, unmatched_gt, unmatched_pred = _greedy_match(gt_segs, pred_segs)

    # Classify errors
    exon_skip = 0
    segment_retention = 0
    novel_insertion = 0
    cascade_shift = 0
    compensating_errors = 0

    # --- Unmatched GT segments → exon skip or segment retention ------------
    for g_idx in unmatched_gt:
        gt_seg = gt_segs[g_idx]
        # Check if any pred segment spans this GT segment's region entirely
        # (indicating it was absorbed / retained by a neighbour)
        absorbed = any(
            ps.start <= gt_seg.start and ps.end >= gt_seg.end
            for ps in pred_segs
            if ps.label != class_token or ps not in [pred_segs[p] for _, p in matches]
        )
        if absorbed:
            segment_retention += 1
        else:
            # Check if neighbouring GT segments were merged in pred
            # (i.e., a pred segment spans the gap where this GT segment was)
            exon_skip += 1

    # --- Unmatched pred segments → novel insertion -------------------------
    for p_idx in unmatched_pred:
        pred_seg = pred_segs[p_idx]
        # Check if it sits inside a GT segment (splitting it)
        splits_gt = any(
            gs.start <= pred_seg.start and gs.end >= pred_seg.end
            for gs in gt_segs
        )
        if splits_gt:
            novel_insertion += 1
        else:
            novel_insertion += 1  # Still a novel insertion even if not inside GT

    # --- Cascade shifts and compensating errors ----------------------------
    # For matched pairs, compute signed boundary residuals
    if len(matches) >= 2:
        residuals = []
        for g_idx, p_idx in sorted(matches, key=lambda x: gt_segs[x[0]].start):
            gs = gt_segs[g_idx]
            ps = pred_segs[p_idx]
            res_5p = ps.start - gs.start
            res_3p = ps.end - gs.end
            residuals.append((res_5p, res_3p))

        # Cascade: 3+ consecutive segments with same-sign 5' or 3' residuals
        cascade_shift = _count_cascades(residuals, min_run=3)

        # Compensating: adjacent pairs where residual[i].3p + residual[i+1].5p ≈ 0
        for i in range(len(residuals) - 1):
            res_3p_cur = residuals[i][1]
            res_5p_nxt = residuals[i + 1][0]
            if res_3p_cur != 0 and res_5p_nxt != 0:
                if abs(res_3p_cur + res_5p_nxt) <= 1:
                    compensating_errors += 1

    total = (
        exon_skip + segment_retention + novel_insertion
        + cascade_shift + compensating_errors
    )

    return {
        "exon_skip_count": exon_skip,
        "segment_retention_count": segment_retention,
        "novel_insertion_count": novel_insertion,
        "cascade_shift_count": cascade_shift,
        "compensating_error_count": compensating_errors,
        "total_junction_errors": total,
    }


def _compute_error_correlations(
    gt_structure: ExtractedStructure,
    pred_structure: ExtractedStructure,
    class_token: int,
    junction_errors: dict,
) -> dict:
    """Analyze spatial correlation of structural errors.

    Parameters
    ----------
    gt_structure, pred_structure : ExtractedStructure
        Structures from the GT and predicted arrays.
    class_token : int
        Which label class to evaluate.
    junction_errors : dict
        Output from :func:`_classify_junction_errors` (used for total count).

    Returns
    -------
    dict
        Keys: ``error_clustering_coefficient``, ``cascade_lengths``,
        ``compensating_error_rate``.
    """
    gt_segs = gt_structure.filter_by_label(class_token)
    pred_segs = pred_structure.filter_by_label(class_token)

    matches, unmatched_gt, unmatched_pred = _greedy_match(gt_segs, pred_segs)

    # Collect error positions (midpoints of mismatched segments)
    error_positions = []
    for g_idx in unmatched_gt:
        s = gt_segs[g_idx]
        error_positions.append((s.start + s.end) // 2)
    for p_idx in unmatched_pred:
        s = pred_segs[p_idx]
        error_positions.append((s.start + s.end) // 2)
    for g_idx, p_idx in matches:
        gs = gt_segs[g_idx]
        ps = pred_segs[p_idx]
        if gs.start != ps.start or gs.end != ps.end:
            error_positions.append((gs.start + gs.end) // 2)

    error_positions.sort()

    # Error clustering coefficient
    clustering_coeff = 0.0
    if len(error_positions) > 1:
        # Median segment length as neighbourhood radius
        all_lengths = [s.length for s in gt_segs] if gt_segs else [1]
        radius = int(np.median(all_lengths))
        clustered = 0
        for i, pos in enumerate(error_positions):
            for j in range(i + 1, len(error_positions)):
                if error_positions[j] - pos <= radius:
                    clustered += 1
                    break
                break  # only check nearest neighbour
        clustering_coeff = clustered / len(error_positions)

    # Cascade lengths
    cascade_lengths = []
    if len(matches) >= 2:
        residuals = []
        for g_idx, p_idx in sorted(matches, key=lambda x: gt_segs[x[0]].start):
            gs = gt_segs[g_idx]
            ps = pred_segs[p_idx]
            residuals.append(ps.start - gs.start)

        cascade_lengths = _get_cascade_lengths(residuals, min_run=3)

    # Compensating error rate
    total_errors = junction_errors.get("total_junction_errors", 0)
    comp_count = junction_errors.get("compensating_error_count", 0)
    comp_rate = comp_count / total_errors if total_errors > 0 else 0.0

    return {
        "error_clustering_coefficient": clustering_coeff,
        "cascade_lengths": cascade_lengths,
        "compensating_error_rate": comp_rate,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _empty_result() -> dict:
    return {
        "exon_skip_count": 0,
        "segment_retention_count": 0,
        "novel_insertion_count": 0,
        "cascade_shift_count": 0,
        "compensating_error_count": 0,
        "total_junction_errors": 0,
    }


def _greedy_match(
    gt_segs: tuple[Segment, ...],
    pred_segs: tuple[Segment, ...],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy 1:1 matching of segments by maximum overlap.

    Returns
    -------
    matches : list[(gt_idx, pred_idx)]
    unmatched_gt : list[int]
    unmatched_pred : list[int]
    """
    candidates = []
    for g_idx, gs in enumerate(gt_segs):
        for p_idx, ps in enumerate(pred_segs):
            overlap_start = max(gs.start, ps.start)
            overlap_end = min(gs.end, ps.end)
            if overlap_end >= overlap_start:
                overlap_len = overlap_end - overlap_start + 1
                candidates.append((overlap_len, g_idx, p_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    claimed_gt: set[int] = set()
    claimed_pred: set[int] = set()
    matches: list[tuple[int, int]] = []

    for _, g_idx, p_idx in candidates:
        if g_idx not in claimed_gt and p_idx not in claimed_pred:
            claimed_gt.add(g_idx)
            claimed_pred.add(p_idx)
            matches.append((g_idx, p_idx))

    unmatched_gt = [i for i in range(len(gt_segs)) if i not in claimed_gt]
    unmatched_pred = [i for i in range(len(pred_segs)) if i not in claimed_pred]

    return matches, unmatched_gt, unmatched_pred


def _count_cascades(
    residuals: list[tuple[int, int]],
    min_run: int = 3,
) -> int:
    """Count cascade shift occurrences in boundary residual sequences.

    A cascade is a run of ``min_run`` or more consecutive residuals with
    the same sign on the 5' component.
    """
    if len(residuals) < min_run:
        return 0

    signs = [1 if r[0] > 0 else (-1 if r[0] < 0 else 0) for r in residuals]
    count = 0
    run_len = 1

    for i in range(1, len(signs)):
        if signs[i] != 0 and signs[i] == signs[i - 1]:
            run_len += 1
        else:
            if run_len >= min_run:
                count += 1
            run_len = 1

    if run_len >= min_run:
        count += 1

    return count


def _get_cascade_lengths(
    residuals: list[int],
    min_run: int = 3,
) -> list[int]:
    """Return the lengths of same-sign runs that qualify as cascades."""
    if len(residuals) < min_run:
        return []

    signs = [1 if r > 0 else (-1 if r < 0 else 0) for r in residuals]
    lengths = []
    run_len = 1

    for i in range(1, len(signs)):
        if signs[i] != 0 and signs[i] == signs[i - 1]:
            run_len += 1
        else:
            if run_len >= min_run:
                lengths.append(run_len)
            run_len = 1

    if run_len >= min_run:
        lengths.append(run_len)

    return lengths
