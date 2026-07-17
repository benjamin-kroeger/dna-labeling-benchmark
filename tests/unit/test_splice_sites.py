"""Tests for splice-site evaluation robustness (WS3).

Covers:
- Unequal donor/acceptor counts → no ValueError, gt_malformed_junctions reported
- Missing intron between donor and acceptor → no AssertionError, counted malformed
- Clean valid pairs → regression golden values unchanged
- Malformed prediction → pred_malformed_junctions reported, scoring still correct
"""
from __future__ import annotations

import pytest

from gene_calling_benchmark.eval.splice_sites import eval_splice_site_junctions
from gene_calling_benchmark.eval.structure import ExtractedStructure, Segment
from gene_calling_benchmark.label_definition import LabelConfig, AnnotationMode

from support.constants import ACCEPTOR, DONOR, EXON, INTRON, NONCODING

CFG = LabelConfig(
    annotation_mode=AnnotationMode.EXON_INTRON,
    background_label=NONCODING,
    exon_label=EXON,
    intron_label=INTRON,
    splice_donor_label=DONOR,
    splice_acceptor_label=ACCEPTOR,
)


def _struct(*segs: tuple[int, int, int]) -> ExtractedStructure:
    """Build an ExtractedStructure from (label, start, end) tuples."""
    segments = tuple(Segment(label=lbl, start=s, end=e) for lbl, s, e in segs)
    length = segments[-1].end + 1 if segments else 0
    return ExtractedStructure(segments=segments, length=length)


# ── Regression: two clean introns ────────────────────────────────────────────

def test_clean_two_introns_no_malformed():
    # GT and pred both have D-I-I-A and D-I-A patterns (identical).
    gt = _struct(
        (DONOR, 2, 2), (INTRON, 3, 5), (ACCEPTOR, 6, 6),
        (DONOR, 9, 9), (INTRON, 10, 11), (ACCEPTOR, 12, 12),
    )
    result = eval_splice_site_junctions(gt, gt, CFG)

    assert result.both_correct == 2
    assert result.donor_tp == 2
    assert result.acceptor_tp == 2
    assert result.donor_fp == 0
    assert result.donor_fn == 0
    assert result.acceptor_fp == 0
    assert result.acceptor_fn == 0
    assert result.gt_malformed_junctions == 0
    assert result.pred_malformed_junctions == 0


# ── Unequal donor/acceptor counts (previously raised ValueError) ──────────────

def test_unequal_donor_acceptor_count_no_crash():
    # GT: two donors, one acceptor — the second donor has no following acceptor.
    gt = _struct(
        (DONOR, 1, 1), (INTRON, 2, 4), (ACCEPTOR, 5, 5),
        (DONOR, 7, 7), (INTRON, 8, 9),               # no ACCEPTOR follows
    )
    pred = _struct(
        (DONOR, 1, 1), (INTRON, 2, 4), (ACCEPTOR, 5, 5),
    )

    result = eval_splice_site_junctions(gt, pred, CFG)

    assert result.gt_malformed_junctions == 1
    assert result.both_correct == 1
    assert result.donor_tp == 1
    assert result.acceptor_tp == 1


# ── Missing intron between donor and acceptor (previously raised AssertionError) ─

def test_missing_intron_between_donor_and_acceptor_no_crash():
    # GT: DONOR immediately followed by exon (not INTRON), then ACCEPTOR.
    # The structural walk finds DONOR → EXON (stops scan), EXON is not ACCEPTOR →
    # donor counted malformed. Then ACCEPTOR appears without a preceding donor
    # → also counted malformed.
    gt = _struct(
        (DONOR, 2, 2), (EXON, 3, 4), (ACCEPTOR, 5, 5),
    )
    pred = _struct(
        (DONOR, 2, 2), (INTRON, 3, 4), (ACCEPTOR, 5, 5),
    )

    result = eval_splice_site_junctions(gt, pred, CFG)

    assert result.gt_malformed_junctions == 2   # orphan donor + orphan acceptor
    assert result.both_correct == 0
    assert result.donor_tp == 0
    assert result.acceptor_tp == 0
    # Pred's donor and acceptor are not in GT's valid pairs → all are FP
    assert result.donor_fp == 1
    assert result.acceptor_fp == 1


# ── Malformed prediction: lone donor, GT is clean ────────────────────────────

def test_malformed_pred_lone_donor_no_crash():
    # GT: one clean intron.  Pred: same intron + an extra orphan donor at end.
    gt = _struct(
        (DONOR, 1, 1), (INTRON, 2, 4), (ACCEPTOR, 5, 5),
    )
    pred = _struct(
        (DONOR, 1, 1), (INTRON, 2, 4), (ACCEPTOR, 5, 5),
        (DONOR, 7, 7),                                # orphan donor
    )

    result = eval_splice_site_junctions(gt, pred, CFG)

    assert result.pred_malformed_junctions == 1
    assert result.gt_malformed_junctions == 0
    assert result.both_correct == 1
    assert result.donor_tp == 1
    assert result.acceptor_tp == 1
    # The orphan pred donor is not in GT's valid donor set → FP
    assert result.donor_fp == 1
    assert result.acceptor_fp == 0
