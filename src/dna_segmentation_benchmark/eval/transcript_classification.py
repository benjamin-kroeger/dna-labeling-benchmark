"""Holistic transcript match classification.

Classifies each (gt_array, pred_array) pair into one structural category
describing *how* the prediction relates to the ground truth as a whole.

Classification hierarchy (evaluated top-to-bottom):

1. MISSED — prediction has no segments of this class.
2. EXACT — identical segment sets.
3. BOUNDARY_SHIFT_INTERNAL — same segment count *and* every predicted segment
   overlaps its positional GT counterpart; the outer gene-locus boundaries
   (first segment start and last segment end) match, but one or more internal
   splice-site boundaries differ.
4. BOUNDARY_SHIFT_TERMINAL — same segment count and full pairwise overlap as
   above, but a terminal gene-locus boundary (the predicted start and/or end)
   differs from GT.
5. MISSING_SEGMENTS — pred is a strict set-subset of GT: every predicted
   exon is a real GT exon, but some GT exons are absent.
6. EXTRA_SEGMENTS — GT is a strict set-subset of pred: every GT exon is
   found, but the prediction contains additional novel exons.
7. PARTIAL_OVERLAP — at least one (start, end) pair is shared, but the sets
   are neither equal nor in a subset relationship.
8. SUBSTITUTION — no (start, end) pair is shared, yet at least one predicted
   segment overlaps a GT segment in base coordinates (relocated/substituted
   exons).
9. NO_OVERLAP — no shared (start, end) pair and no base-coordinate overlap at
   all between GT and prediction.

Note
----
``EXACT`` requires *identical* exon coordinates, including the terminal /
TSS-TES boundaries. This is intentionally **stricter** than gffcompare's ``=``
class code, which ignores terminal-exon outer boundaries. For the
junction-tolerant, containment-aware view (the gffcompare analog) see
:class:`dna_segmentation_benchmark.transcript_mapping.MatchClass`. The
``BOUNDARY_SHIFT_INTERNAL`` / ``BOUNDARY_SHIFT_TERMINAL`` distinction concerns
the gene-locus outer boundaries (``segments[0].start`` / ``segments[-1].end``)
and is only evaluated once every predicted segment overlaps its positional GT
counterpart — equal segment count alone is not sufficient.
"""

from __future__ import annotations

from enum import Enum

from .structure import Segment


class TranscriptMatchClass(str, Enum):
    """Structural classification of a (GT, pred) pair.

    Enum declaration order is significant: it doubles as the severity order
    (best → worst) used for plot legends and the green→red colour gradient.
    """

    EXACT = "exact"
    BOUNDARY_SHIFT_INTERNAL = "boundary_shift_internal"
    BOUNDARY_SHIFT_TERMINAL = "boundary_shift_terminal"
    MISSING_SEGMENTS = "missing_segments"
    EXTRA_SEGMENTS = "extra_segments"
    PARTIAL_OVERLAP = "partial_overlap"
    SUBSTITUTION = "substitution"
    NO_OVERLAP = "no_overlap"
    MISSED = "missed"


def _classify_segment_match(
    gt_segments: tuple[Segment, ...],
    pred_segments: tuple[Segment, ...],
) -> TranscriptMatchClass | None:
    """Holistically classify a structural relationship for pre-selected segments.

    Segments are assumed ordered ascending by position. Returns ``None`` when
    GT has no segments (not applicable for this scope).
    """
    n_gt = len(gt_segments)
    n_pred = len(pred_segments)

    if n_gt == 0:
        return None

    if n_pred == 0:
        return TranscriptMatchClass.MISSED

    gt_set: frozenset[tuple[int, int]] = frozenset((s.start, s.end) for s in gt_segments)
    pred_set: frozenset[tuple[int, int]] = frozenset((s.start, s.end) for s in pred_segments)

    if gt_set == pred_set:
        return TranscriptMatchClass.EXACT

    # Boundary shift only when every positional pair actually overlaps. Equal
    # segment count alone is not enough — disjoint or substituted equal-count
    # chains must fall through to the overlap ladder below.
    if n_gt == n_pred and all(g.overlaps(p) for g, p in zip(gt_segments, pred_segments)):
        if gt_segments[0].start == pred_segments[0].start and gt_segments[-1].end == pred_segments[-1].end:
            return TranscriptMatchClass.BOUNDARY_SHIFT_INTERNAL
        return TranscriptMatchClass.BOUNDARY_SHIFT_TERMINAL

    if pred_set < gt_set:
        return TranscriptMatchClass.MISSING_SEGMENTS

    if gt_set < pred_set:
        return TranscriptMatchClass.EXTRA_SEGMENTS

    if gt_set & pred_set:
        return TranscriptMatchClass.PARTIAL_OVERLAP

    # No identical exon. Distinguish positional overlap (substitution) from
    # true disjointness. Segment counts per transcript are small, so the
    # pairwise scan is cheap.
    if any(g.overlaps(p) for g in gt_segments for p in pred_segments):
        return TranscriptMatchClass.SUBSTITUTION

    return TranscriptMatchClass.NO_OVERLAP
