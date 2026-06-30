"""Unit tests for the holistic transcript match classifier.

Exercises :func:`_classify_segment_match` directly with hand-built segment
chains (only segment geometry matters), plus regression coverage for the
equal-count boundary-shift bug and the base-overlap ladder.
"""

from __future__ import annotations

import pytest

from dna_segmentation_benchmark.eval.structure import Segment
from dna_segmentation_benchmark.eval.transcript_classification import (
    TranscriptMatchClass as C,
    _classify_segment_match,
)


def _segs(*pairs: tuple[int, int]) -> tuple[Segment, ...]:
    """Build an ordered segment chain from (start, end) pairs (label irrelevant)."""
    return tuple(Segment(label=0, start=s, end=e) for s, e in pairs)


@pytest.mark.parametrize(
    "gt, pred, expected",
    [
        # exact identity
        (_segs((0, 10), (20, 30)), _segs((0, 10), (20, 30)), C.EXACT),
        # internal splice shift: outer hull intact, every pair overlaps
        (
            _segs((0, 10), (20, 30), (40, 50)),
            _segs((0, 10), (22, 30), (40, 50)),
            C.BOUNDARY_SHIFT_INTERNAL,
        ),
        # terminal shift: outer boundaries move, pairs still overlap
        (_segs((0, 10), (20, 30)), _segs((2, 10), (20, 32)), C.BOUNDARY_SHIFT_TERMINAL),
        # REGRESSION #1: equal count but fully disjoint -> NO_OVERLAP (was BOUNDARY_SHIFT)
        (
            _segs((0, 10), (20, 30), (40, 50)),
            _segs((100, 110), (120, 130), (140, 150)),
            C.NO_OVERLAP,
        ),
        # REGRESSION #1/#4: equal count, middle exon relocated but ends identical
        # -> PARTIAL_OVERLAP (shares identical exons), NOT a boundary shift
        (
            _segs((0, 10), (20, 30), (40, 50)),
            _segs((0, 10), (200, 210), (40, 50)),
            C.PARTIAL_OVERLAP,
        ),
        # strict subset -> missing
        (_segs((0, 10), (20, 30), (40, 50)), _segs((0, 10), (40, 50)), C.MISSING_SEGMENTS),
        # strict superset -> extra
        (_segs((0, 10), (40, 50)), _segs((0, 10), (20, 30), (40, 50)), C.EXTRA_SEGMENTS),
        # shares one identical exon, not subset -> partial
        (_segs((0, 10), (20, 30)), _segs((0, 10), (60, 70)), C.PARTIAL_OVERLAP),
        # single-exon overlap with both ends moved is an equal-count terminal shift
        (_segs((0, 100)), _segs((10, 90)), C.BOUNDARY_SHIFT_TERMINAL),
        # REGRESSION #5: count mismatch, base overlap, no identical exon -> SUBSTITUTION
        # (old code fell through to NO_OVERLAP despite heavy base overlap)
        (_segs((0, 100)), _segs((10, 40), (60, 90)), C.SUBSTITUTION),
        # equal count, no identical exon, all pairs overlap but shifted a lot -> boundary shift terminal
        (_segs((0, 10), (20, 30)), _segs((1, 11), (21, 31)), C.BOUNDARY_SHIFT_TERMINAL),
        # truly disjoint single exons -> no overlap
        (_segs((0, 10)), _segs((100, 110)), C.NO_OVERLAP),
        # touching end-to-end (no shared base) -> no overlap
        (_segs((0, 9)), _segs((10, 19)), C.NO_OVERLAP),
    ],
)
def test_classify_segment_match(gt, pred, expected):
    assert _classify_segment_match(gt, pred) == expected


def test_missed_when_no_prediction():
    assert _classify_segment_match(_segs((0, 10)), ()) == C.MISSED


def test_none_when_no_ground_truth():
    assert _classify_segment_match((), _segs((0, 10))) is None


def test_substitution_requires_overlap_not_identity():
    """Equal count, no identical exon, partial pairwise overlap -> not a boundary shift."""
    gt = _segs((0, 10), (20, 30), (40, 50))
    # pair 2 does not overlap (200..210) -> boundary-shift guard must fail
    pred = _segs((5, 15), (200, 210), (45, 55))
    assert _classify_segment_match(gt, pred) == C.SUBSTITUTION
