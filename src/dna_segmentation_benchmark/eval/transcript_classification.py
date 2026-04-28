"""Holistic transcript match classification.

Classifies each (gt_array, pred_array) pair into one structural category
describing *how* the prediction relates to the ground truth as a whole.

Classification hierarchy (evaluated top-to-bottom):

1. MISSED — prediction has no segments of this class.
2. EXACT — identical segment sets.
3. BOUNDARY_SHIFT_INTERNAL — same segment count; outer gene-locus boundaries
   (first segment start and last segment end) match, but one or more internal
   splice-site boundaries differ.
4. BOUNDARY_SHIFT_TERMINAL — same segment count; terminal boundaries also
   differ (the predicted gene-locus start and/or end differs from GT).
5. MISSING_SEGMENTS — pred is a strict set-subset of GT: every predicted
   exon is a real GT exon, but some GT exons are absent.
6. EXTRA_SEGMENTS — GT is a strict set-subset of pred: every GT exon is
   found, but the prediction contains additional novel exons.
7. PARTIAL_OVERLAP — at least one (start, end) pair is shared, but the sets
   are neither equal nor in a subset relationship.
8. NO_OVERLAP — no (start, end) pair is shared between GT and prediction.
"""

from __future__ import annotations

from enum import Enum

from .structure import ExtractedStructure


class TranscriptMatchClass(str, Enum):
    """Structural classification of a (GT, pred) pair."""

    EXACT = "exact"
    BOUNDARY_SHIFT_INTERNAL = "boundary_shift_internal"
    BOUNDARY_SHIFT_TERMINAL = "boundary_shift_terminal"
    MISSING_SEGMENTS = "missing_segments"
    EXTRA_SEGMENTS = "extra_segments"
    PARTIAL_OVERLAP = "partial_overlap"
    NO_OVERLAP = "no_overlap"
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
        Classification (``None`` when GT has no segments — not applicable).
    """
    gt_segs = gt_structure.filter_by_label(class_token)
    pred_segs = pred_structure.filter_by_label(class_token)

    n_gt = len(gt_segs)
    n_pred = len(pred_segs)

    if n_gt == 0:
        return None

    if n_pred == 0:
        return TranscriptMatchClass.MISSED

    gt_set: frozenset[tuple[int, int]] = frozenset((s.start, s.end) for s in gt_segs)
    pred_set: frozenset[tuple[int, int]] = frozenset((s.start, s.end) for s in pred_segs)

    if gt_set == pred_set:
        return TranscriptMatchClass.EXACT

    if n_gt == n_pred:
        # Same segment count — pairwise positional comparison for terminal check
        if gt_segs[0].start == pred_segs[0].start and gt_segs[-1].end == pred_segs[-1].end:
            return TranscriptMatchClass.BOUNDARY_SHIFT_INTERNAL
        return TranscriptMatchClass.BOUNDARY_SHIFT_TERMINAL

    if pred_set < gt_set:
        return TranscriptMatchClass.MISSING_SEGMENTS

    if gt_set < pred_set:
        return TranscriptMatchClass.EXTRA_SEGMENTS

    if gt_set & pred_set:
        return TranscriptMatchClass.PARTIAL_OVERLAP

    return TranscriptMatchClass.NO_OVERLAP