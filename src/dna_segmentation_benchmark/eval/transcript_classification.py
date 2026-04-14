"""Holistic transcript match classification.

Classifies each ``(gt_array, pred_array)`` pair into one structural
category describing *how* the prediction relates to the ground truth
as a whole — not per-section, but as a complete annotation.

Classification hierarchy (evaluated top-to-bottom):

1. ``MISSED`` — prediction has no segments of this class.
2. ``EXACT`` — identical segment chains.
3. ``BOUNDARY_SHIFT`` — same number of segments in same label order,
   but one or more internal boundary positions differ.
4. ``MISSING_SEGMENTS`` — prediction chain is a strict ordered subset
   of GT (some GT segments are absent from prediction).
5. ``EXTRA_SEGMENTS`` — GT chain is a strict ordered subset of
   prediction (prediction contains additional segments).
6. ``STRUCTURALLY_DIFFERENT`` — none of the above; use gap chain LCS
   ratio for continuous similarity scoring.
"""

from __future__ import annotations

from enum import Enum

from .chain_comparison import _lcs_length, _boundaries
from .structure import ExtractedStructure, Segment


class TranscriptMatchClass(str, Enum):
    """Structural classification of a (GT, pred) pair."""

    EXACT = "exact"
    BOUNDARY_SHIFT = "boundary_shift"
    MISSING_SEGMENTS = "missing_segments"
    EXTRA_SEGMENTS = "extra_segments"
    STRUCTURALLY_DIFFERENT = "structurally_different"
    MISSED = "missed"


def _measure_shifted_internal_boundaries(
    gt_segs: tuple[Segment, ...],
    pred_segs: tuple[Segment, ...],
) -> tuple[int, int]:
    """Count and sum shifted boundary positions, ignoring first start and last end.

    The first segment's start and the last segment's end correspond to UTR
    boundaries and are excluded.  All other boundary positions (splice-site
    donors and acceptors) are compared pairwise.

    Parameters
    ----------
    gt_segs, pred_segs : tuple[Segment, ...]
        Segment chains of equal length.

    Returns
    -------
    (count, total) : tuple[int, int]
        *count* — number of internal boundary positions that differ.
        *total* — sum of absolute position offsets across those boundaries (bp).
    """
    n = len(gt_segs)
    if n == 0:
        return 0, 0
    count = 0
    total = 0
    for i, (g, p) in enumerate(zip(gt_segs, pred_segs)):
        if i > 0 and g.start != p.start:        # not UTR: splice acceptor
            count += 1
            total += abs(g.start - p.start)
        if i < n - 1 and g.end != p.end:        # not UTR: splice donor
            count += 1
            total += abs(g.end - p.end)
    return count, total


def _classify_transcript_match(
    gt_structure: ExtractedStructure,
    pred_structure: ExtractedStructure,
    class_token: int,
) -> tuple[TranscriptMatchClass | None, int, int, int, int]:
    """Holistically classify the structural relationship for *class_token*.

    Parameters
    ----------
    gt_structure, pred_structure : ExtractedStructure
        Structures extracted from the GT and predicted arrays.
    class_token : int
        Which label class to evaluate.

    Returns
    -------
    (TranscriptMatchClass | None, int, int, int, int)
        * match class (``None`` when GT has no segments — not applicable)
        * ``boundary_shift_count`` — number of shifted internal boundary positions
          (0 for all classes except ``BOUNDARY_SHIFT``)
        * ``boundary_shift_total`` — sum of absolute position offsets across
          shifted boundaries in bp (0 for all classes except ``BOUNDARY_SHIFT``)
        * ``n_gt`` — number of GT segments
        * ``n_pred`` — number of predicted segments
    """
    gt_segs = gt_structure.filter_by_label(class_token)
    pred_segs = pred_structure.filter_by_label(class_token)

    n_gt = len(gt_segs)
    n_pred = len(pred_segs)

    # -- GT has no segments of this class → classification not applicable ------
    if n_gt == 0:
        return None, 0, 0, 0, 0

    # -- No prediction at all ------------------------------------------------
    if n_pred == 0:
        return TranscriptMatchClass.MISSED, 0, 0, n_gt, n_pred

    gt_bounds = _boundaries(gt_segs)
    pred_bounds = _boundaries(pred_segs)

    # -- Exact match ----------------------------------------------------------
    if gt_bounds == pred_bounds:
        return TranscriptMatchClass.EXACT, 0, 0, n_gt, n_pred

    # -- Same count: measure internal (splice-site) boundary shifts -----------
    if n_gt == n_pred:
        shift_count, shift_total = _measure_shifted_internal_boundaries(gt_segs, pred_segs)
        return TranscriptMatchClass.BOUNDARY_SHIFT, shift_count, shift_total, n_gt, n_pred

    # -- Subset / Superset via LCS -------------------------------------------
    lcs_len = _lcs_length(gt_bounds, pred_bounds)

    # Pred is a strict ordered subset of GT: segments are missing
    if lcs_len == n_pred and n_pred < n_gt:
        return TranscriptMatchClass.MISSING_SEGMENTS, 0, 0, n_gt, n_pred

    # GT is a strict ordered subset of pred: extra segments inserted
    if lcs_len == n_gt and n_gt < n_pred:
        return TranscriptMatchClass.EXTRA_SEGMENTS, 0, 0, n_gt, n_pred

    return TranscriptMatchClass.STRUCTURALLY_DIFFERENT, 0, 0, n_gt, n_pred


# ---------------------------------------------------------------------------
# Transcript-level precision / recall tiers
# ---------------------------------------------------------------------------

# Which match classes count as TP at each tier
_TIER_TP_CLASSES: dict[str, frozenset[TranscriptMatchClass]] = {
    "transcript_exact": frozenset({
        TranscriptMatchClass.EXACT,
    }),
    "pred_is_superset": frozenset({
        TranscriptMatchClass.EXACT,
        TranscriptMatchClass.BOUNDARY_SHIFT,
        TranscriptMatchClass.EXTRA_SEGMENTS,
    }),
    "pred_is_subset": frozenset({
        TranscriptMatchClass.EXACT,
        TranscriptMatchClass.BOUNDARY_SHIFT,
        TranscriptMatchClass.MISSING_SEGMENTS,
    }),
}


def _compute_transcript_level_pr(
    match_cls: TranscriptMatchClass | None,
    boundary_shift_count: int = 0,
    boundary_shift_total: int = 0,
) -> dict[str, dict[str, int] | int] | None:
    """Compute per-sequence TP/FN/FP at four transcript-level tiers.

    Tiers (cumulative, each more lenient):

    * ``transcript_exact`` — identical segment chains.
    * ``pred_is_superset`` — all GT segments present in pred
      (pred may have extras).  Recall-oriented: forgives over-prediction.
    * ``pred_is_subset`` — all pred segments match a GT
      segment (some GT segments may be missing).  Precision-oriented:
      forgives under-prediction.

    Also returns:

    * ``boundary_shift_count`` — number of internal splice-site boundary
      positions that differ (``BOUNDARY_SHIFT`` only, else 0).
    * ``boundary_shift_total`` — sum of absolute position offsets across
      those shifted boundaries in bp (``BOUNDARY_SHIFT`` only, else 0).

    Parameters
    ----------
    match_cls : TranscriptMatchClass | None
        Classification from :func:`_classify_transcript_match`.
        ``None`` means GT has no segments (not applicable).
    boundary_shift_count : int
        Number of shifted internal boundaries.
    boundary_shift_total : int
        Sum of absolute position offsets of shifted boundaries (bp).

    Returns
    -------
    dict | None
        Per-tier ``{"tp": int, "fn": int, "fp": int}`` dicts plus
        ``"boundary_shift_count": int`` and ``"boundary_shift_total": int``.
        ``None`` when *match_cls* is ``None``.
    """
    if match_cls is None:
        return None

    has_pred = match_cls != TranscriptMatchClass.MISSED

    result: dict[str, dict[str, int] | int] = {}
    for tier_name, tp_classes in _TIER_TP_CLASSES.items():
        is_tp = match_cls in tp_classes
        result[tier_name] = {
            "tp": int(is_tp),
            "fn": int(not is_tp),                       # GT not recovered at this tier
            "fp": int(has_pred and not is_tp),          # pred exists but invalid at this tier
        }

    result["boundary_shift_count"] = boundary_shift_count
    result["boundary_shift_total"] = boundary_shift_total
    return result
