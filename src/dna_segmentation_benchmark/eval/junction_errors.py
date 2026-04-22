"""Helpers for overlap-based segment matching.

Only the greedy matcher is currently used directly by
``structural_summary.py`` to identify unmatched GT and predicted segments for
the position-bias histogram.
"""

from __future__ import annotations

from .structure import Segment


def _greedy_match(
    gt_segments: tuple[Segment, ...],
    pred_segments: tuple[Segment, ...],
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily match overlapping GT and predicted segments by overlap size.

    Parameters
    ----------
    gt_segments, pred_segments
        Ordered segments to compare.

    Returns
    -------
    tuple
        ``(matches, unmatched_gt, unmatched_pred)`` where ``matches`` is a list
        of ``(gt_index, pred_index)`` pairs and the unmatched collections are
        lists of remaining indices.
    """
    candidates: list[tuple[int, int, int]] = []
    for gt_idx, gt_segment in enumerate(gt_segments):
        for pred_idx, pred_segment in enumerate(pred_segments):
            overlap_start = max(gt_segment.start, pred_segment.start)
            overlap_end = min(gt_segment.end, pred_segment.end)
            if overlap_start > overlap_end:
                continue
            overlap_len = overlap_end - overlap_start + 1
            candidates.append((overlap_len, gt_idx, pred_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    claimed_gt: set[int] = set()
    claimed_pred: set[int] = set()
    matches: list[tuple[int, int]] = []

    for _overlap_len, gt_idx, pred_idx in candidates:
        if gt_idx in claimed_gt or pred_idx in claimed_pred:
            continue
        claimed_gt.add(gt_idx)
        claimed_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx))

    unmatched_gt = [idx for idx in range(len(gt_segments)) if idx not in claimed_gt]
    unmatched_pred = [idx for idx in range(len(pred_segments)) if idx not in claimed_pred]
    return matches, unmatched_gt, unmatched_pred
