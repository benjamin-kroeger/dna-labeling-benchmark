"""Tests for gffcompare -R / -Q incomplete-annotation corrections.

``ignore_novel_predictions`` (-Q) drops predictions overlapping no GT from the
precision side; ``ignore_missed_reference`` (-R) drops GT overlapping no
prediction from the sensitivity side.  Both default off.  The crucial contract
is that an *overlapping-but-unmatched* prediction (or GT) is still counted — the
filter is coordinate-overlap based, not match based.
"""

from __future__ import annotations

import math

import pandas as pd

from dna_segmentation_benchmark.eval.global_metrics import (
    compute_global_metrics,
    compute_overlap_keepsets,
)
from dna_segmentation_benchmark.label_definition import AnnotationMode, LabelConfig
from dna_segmentation_benchmark.transcript_mapping import (
    LocusMatchingMode,
    MatchClass,
    PredictionMatch,
    TranscriptMapping,
    _include_mapping_for_predictor,
)

_CFG = LabelConfig(annotation_mode=AnnotationMode.EXON_INTRON, background_label=8, exon_label=0)
_TT = ["mRNA"]
_PRED = "pred"


def _row(seqid, strand, type_, start, end, gff_id, parent=None):
    return {"seqid": seqid, "strand": strand, "type": type_, "start": start,
            "end": end, "gff_id": gff_id, "parent": parent}


def _tx(gff_id, start, end, seqid="chr1", strand="+"):
    return _row(seqid, strand, "mRNA", start, end, gff_id)


def _ex(gff_id, start, end, parent, seqid="chr1", strand="+"):
    return _row(seqid, strand, "exon", start, end, gff_id, parent)


def _match(transcript_id, start, end):
    return PredictionMatch(predictor_name=_PRED, transcript_id=transcript_id, start=start,
                           end=end, match_class=MatchClass.EXACT, base_overlap=end - start + 1,
                           junction_f1=1.0)


def _mapping(gt_id, start, end, matches=(), seqid="chr1", strand="+", is_unmatched=False):
    return TranscriptMapping(seqid=seqid, strand=strand, gt_id=gt_id, gt_start=start,
                             gt_end=end, matched_predictions=list(matches),
                             is_unmatched_prediction=is_unmatched)


def _run(gt_rows, pred_rows, mappings, **flags):
    return compute_global_metrics(
        gt_df=pd.DataFrame(gt_rows), pred_df=pd.DataFrame(pred_rows), mappings=mappings,
        predictor_name=_PRED, label_config=_CFG, transcript_types=_TT, **flags,
    )


# ---------------------------------------------------------------------------
# compute_overlap_keepsets
# ---------------------------------------------------------------------------


def test_keepsets_novel_and_missed():
    gt = [_tx("tx1", 100, 200), _tx("tx2", 900, 1000)]  # tx2 overlaps nothing
    pred = [_tx("p1", 120, 220), _tx("p2", 500, 600)]   # p2 overlaps nothing
    ref_keep, pred_keep = compute_overlap_keepsets(pd.DataFrame(gt), pd.DataFrame(pred), _TT)
    assert ref_keep == {("chr1", "tx1")}
    assert pred_keep == {("chr1", "p1")}


def test_keepsets_strand_aware():
    gt = [_tx("tx1", 100, 200, strand="+")]
    pred = [_tx("p1", 100, 200, strand="-")]  # same span, opposite strand
    ref_keep, pred_keep = compute_overlap_keepsets(pd.DataFrame(gt), pd.DataFrame(pred), _TT)
    assert ref_keep == set()
    assert pred_keep == set()


def test_keepsets_recycled_transcript_id_is_seqid_qualified():
    """A recycled id (Tiberius restarts ``g1.t1`` per sequence) must not let an
    overlapping copy on chr1 keep its novel namesake on chr2 alive under -Q."""
    gt = [_tx("g1.t1", 100, 200, seqid="chr1")]
    pred = [
        _tx("g1.t1", 120, 220, seqid="chr1"),   # overlaps GT -> keep
        _tx("g1.t1", 500, 600, seqid="chr2"),   # same id, no GT anywhere -> drop
    ]
    _ref_keep, pred_keep = compute_overlap_keepsets(pd.DataFrame(gt), pd.DataFrame(pred), _TT)
    assert pred_keep == {("chr1", "g1.t1")}
    assert ("chr2", "g1.t1") not in pred_keep


def test_recycled_id_exons_not_fused_across_seqids():
    """Bucketing scope intervals by a bare parent pooled every same-named
    transcript's exons into one bucket, where the coordinate-only merge fused
    exons from unrelated scaffolds into one chimeric interval."""
    from dna_segmentation_benchmark.eval.global_metrics import (
        _collect_scoped_transcript_intervals,
    )
    from dna_segmentation_benchmark.feature_roles import normalize_feature_role_map

    rows = [
        _ex("e1", 100, 200, "g1.t1", seqid="chr1"),
        _ex("e2", 400, 500, "g1.t1", seqid="chr1"),
        _ex("e3", 150, 450, "g1.t1", seqid="chr2"),  # spans chr1's intron, other scaffold
    ]
    rmap = normalize_feature_role_map(None, _CFG, arg_name="m")
    ivs, _orphans = _collect_scoped_transcript_intervals(
        pd.DataFrame(rows), rmap, _CFG, _CFG.evaluation_scope
    )
    assert set(ivs) == {("chr1", "+", "g1.t1"), ("chr2", "+", "g1.t1")}
    assert ivs[("chr1", "+", "g1.t1")] == [("chr1", "+", 100, 200), ("chr1", "+", 400, 500)]
    assert ivs[("chr2", "+", "g1.t1")] == [("chr2", "+", 150, 450)]


# ---------------------------------------------------------------------------
# -Q : novel predictions
# ---------------------------------------------------------------------------


def test_Q_drops_novel_prediction():
    gt = [_tx("tx1", 100, 200), _ex("e1", 100, 200, "tx1")]
    pred = [_tx("p1", 100, 200), _ex("pe1", 100, 200, "p1"),
            _tx("p2", 500, 600), _ex("pe2", 500, 600, "p2")]  # novel
    mappings = [
        _mapping("tx1", 100, 200, [_match("p1", 100, 200)]),
        _mapping("__unmatched_pred__p2", 500, 600, [_match("p2", 500, 600)], is_unmatched=True),
    ]

    base = _run(gt, pred, mappings)
    assert math.isclose(base["transcript"]["precision"], 0.5)
    assert base["nucleotide"]["scopes"]["transcript_exon"]["precision"] < 1.0

    corrected = _run(gt, pred, mappings, ignore_novel_predictions=True)
    assert math.isclose(corrected["transcript"]["precision"], 1.0)
    assert math.isclose(corrected["nucleotide"]["scopes"]["transcript_exon"]["precision"], 1.0)
    # Sensitivity untouched by -Q.
    assert math.isclose(corrected["transcript"]["sensitivity"], 1.0)


def test_Q_keeps_overlapping_but_unmatched_prediction():
    """The subtlety: an unmatched pred that OVERLAPS GT is still an FP under -Q."""
    gt = [_tx("tx1", 100, 500), _ex("e1", 100, 500, "tx1")]
    pred = [_tx("p1", 100, 500), _ex("pe1", 100, 500, "p1"),
            _tx("p2", 120, 180), _ex("pe2", 120, 180, "p2"),   # overlaps tx1, unmatched
            _tx("p3", 800, 900), _ex("pe3", 800, 900, "p3")]   # truly novel
    mappings = [
        _mapping("tx1", 100, 500, [_match("p1", 100, 500)]),
        _mapping("__unmatched_pred__p2", 120, 180, [_match("p2", 120, 180)], is_unmatched=True),
        _mapping("__unmatched_pred__p3", 800, 900, [_match("p3", 800, 900)], is_unmatched=True),
    ]
    # Baseline: p2 and p3 both lower precision → 1/3.
    assert math.isclose(_run(gt, pred, mappings)["transcript"]["precision"], 1 / 3)
    # -Q drops only p3 (novel); p2 overlaps tx1 so it stays → 1/2.
    corrected = _run(gt, pred, mappings, ignore_novel_predictions=True)
    assert math.isclose(corrected["transcript"]["precision"], 0.5)


# ---------------------------------------------------------------------------
# -R : missed reference
# ---------------------------------------------------------------------------


def test_R_drops_missed_reference():
    gt = [_tx("tx1", 100, 200), _ex("e1", 100, 200, "tx1"),
          _tx("tx2", 500, 600), _ex("e2", 500, 600, "tx2")]  # never overlapped
    pred = [_tx("p1", 100, 200), _ex("pe1", 100, 200, "p1")]
    mappings = [
        _mapping("tx1", 100, 200, [_match("p1", 100, 200)]),
        _mapping("tx2", 500, 600),
    ]

    assert math.isclose(_run(gt, pred, mappings)["transcript"]["sensitivity"], 0.5)
    corrected = _run(gt, pred, mappings, ignore_missed_reference=True)
    assert math.isclose(corrected["transcript"]["sensitivity"], 1.0)
    assert math.isclose(corrected["transcript"]["precision"], 1.0)


def test_R_keeps_overlapping_but_unmatched_reference():
    gt = [_tx("tx1", 100, 500), _ex("e1", 100, 500, "tx1"),
          _tx("tx2", 120, 180), _ex("e2", 120, 180, "tx2"),   # overlaps p1, unmatched
          _tx("tx3", 800, 900), _ex("e3", 800, 900, "tx3")]   # truly missed
    pred = [_tx("p1", 100, 500), _ex("pe1", 100, 500, "p1")]
    mappings = [
        _mapping("tx1", 100, 500, [_match("p1", 100, 500)]),
        _mapping("tx2", 120, 180),
        _mapping("tx3", 800, 900),
    ]
    # Baseline: tx2, tx3 both missed → 1/3.
    assert math.isclose(_run(gt, pred, mappings)["transcript"]["sensitivity"], 1 / 3)
    # -R drops only tx3; tx2 overlaps p1 so it stays a miss → 1/2.
    corrected = _run(gt, pred, mappings, ignore_missed_reference=True)
    assert math.isclose(corrected["transcript"]["sensitivity"], 0.5)


def test_default_is_unchanged():
    """No flags → identical to a call without the kwargs at all."""
    gt = [_tx("tx1", 100, 200), _ex("e1", 100, 200, "tx1"), _tx("tx2", 500, 600)]
    pred = [_tx("p1", 100, 200), _ex("pe1", 100, 200, "p1"), _tx("p2", 900, 950)]
    mappings = [
        _mapping("tx1", 100, 200, [_match("p1", 100, 200)]),
        _mapping("tx2", 500, 600),
        _mapping("__unmatched_pred__p2", 900, 950, [_match("p2", 900, 950)], is_unmatched=True),
    ]
    r1 = _run(gt, pred, mappings)
    r2 = _run(gt, pred, mappings, ignore_novel_predictions=False, ignore_missed_reference=False)
    assert r1["transcript"] == r2["transcript"]
    assert r1["nucleotide"] == r2["nucleotide"]


# ---------------------------------------------------------------------------
# Gate: _include_mapping_for_predictor — 4 flag/entry combinations
# ---------------------------------------------------------------------------

_MODE = LocusMatchingMode.FULL_DISCOVERY


def _gate(mapping, **kw):
    return _include_mapping_for_predictor(mapping, _PRED, _MODE, **kw)


def test_gate_Q_drops_novel_keeps_overlapping():
    novel = _mapping("__unmatched_pred__pN", 500, 600, [_match("pN", 500, 600)], is_unmatched=True)
    overlap = _mapping("__unmatched_pred__pO", 120, 180, [_match("pO", 120, 180)], is_unmatched=True)
    pred_keep = {("chr1", "pO")}  # pO overlaps GT, pN does not
    assert _gate(novel, pred_keep=pred_keep, ignore_novel=True) is False
    assert _gate(overlap, pred_keep=pred_keep, ignore_novel=True) is True
    # Flag off → both kept.
    assert _gate(novel, pred_keep=pred_keep, ignore_novel=False) is True


def test_gate_Q_recycled_id_dropped_on_other_seqid():
    """Same prediction id on another seqid overlaps no GT — the keep-set entry for
    chr1 must not rescue it."""
    other = _mapping("__unmatched_pred__pO", 500, 600, [_match("pO", 500, 600)],
                     seqid="chr2", is_unmatched=True)
    assert _gate(other, pred_keep={("chr1", "pO")}, ignore_novel=True) is False


def test_gate_R_drops_missed_keeps_overlapping():
    missed = _mapping("tx_missed", 800, 900)
    overlap = _mapping("tx_overlap", 120, 180)
    ref_keep = {("chr1", "tx_overlap")}  # overlaps a pred, tx_missed does not
    assert _gate(missed, ref_keep=ref_keep, ignore_missed=True) is False
    assert _gate(overlap, ref_keep=ref_keep, ignore_missed=True) is True
    assert _gate(missed, ref_keep=ref_keep, ignore_missed=False) is True
