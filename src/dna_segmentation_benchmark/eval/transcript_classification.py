"""Holistic transcript match classification.

Classifies each ``(gt_array, pred_array)`` pair into one structural
category describing *how* the prediction relates to the ground truth
as a whole — not per-section, but as a complete annotation.

Classification hierarchy (evaluated top-to-bottom):

1. ``MISSED`` — prediction has no segments of this class.
2. ``EXACT`` — identical segment chains.
3. ``BOUNDARY_SHIFT`` — same number of segments in same label order,
   but boundary positions differ.
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
from .structure import ExtractedStructure


class TranscriptMatchClass(str, Enum):
    """Structural classification of a (GT, pred) pair."""

    EXACT = "exact"
    BOUNDARY_SHIFT = "boundary_shift"
    MISSING_SEGMENTS = "missing_segments"
    EXTRA_SEGMENTS = "extra_segments"
    STRUCTURALLY_DIFFERENT = "structurally_different"
    MISSED = "missed"


def _classify_transcript_match(
    gt_structure: ExtractedStructure,
    pred_structure: ExtractedStructure,
    class_token: int,
) -> TranscriptMatchClass | None:
    """Holistically classify the structural relationship for *class_token*.

    Parameters
    ----------
    gt_structure, pred_structure : ExtractedStructure
        Structures extracted from the GT and predicted arrays.
    class_token : int
        Which label class to evaluate.

    Returns
    -------
    TranscriptMatchClass | None
        ``None`` when GT has no segments of *class_token* (classification
        not applicable).
    """
    gt_segs = gt_structure.filter_by_label(class_token)
    pred_segs = pred_structure.filter_by_label(class_token)

    n_gt = len(gt_segs)
    n_pred = len(pred_segs)

    # -- GT has no segments of this class → classification not applicable ------
    if n_gt == 0:
        return None

    # -- No prediction at all ------------------------------------------------
    if n_pred == 0:
        return TranscriptMatchClass.MISSED

    gt_bounds = _boundaries(gt_segs)
    pred_bounds = _boundaries(pred_segs)

    # -- Exact match ----------------------------------------------------------
    if gt_bounds == pred_bounds:
        return TranscriptMatchClass.EXACT

    # -- Same count but different boundaries ----------------------------------
    if n_gt == n_pred:
        return TranscriptMatchClass.BOUNDARY_SHIFT

    # -- Subset / Superset via LCS -------------------------------------------
    lcs_len = _lcs_length(gt_bounds, pred_bounds)

    # Pred is a strict ordered subset of GT: segments are missing
    if lcs_len == n_pred and n_pred < n_gt:
        return TranscriptMatchClass.MISSING_SEGMENTS

    # GT is a strict ordered subset of pred: extra segments inserted
    if lcs_len == n_gt and n_gt < n_pred:
        return TranscriptMatchClass.EXTRA_SEGMENTS

    return TranscriptMatchClass.STRUCTURALLY_DIFFERENT
