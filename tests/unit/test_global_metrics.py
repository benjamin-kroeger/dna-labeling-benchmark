"""Value-level unit tests for compute_global_metrics (S1 fix).

Previous coverage was limited to key-presence assertions in the integration
tests.  These tests assert concrete numeric values for each sub-metric group
using minimal, hand-constructed DataFrames and TranscriptMapping objects.
No GFF files are required.

Covered sub-functions
---------------------
- _compute_transcript_level_metrics  (via transcript key)
- _compute_locus_isoform_metrics     (via locus_isoform key)
- _compute_global_exon_metrics       (via exon key, including de-dup contract)
- _compute_global_nucleotide_metrics (via nucleotide key)
- _compute_gene_level_metrics        (via gene key)
"""

from __future__ import annotations

import math

import pandas as pd

from gene_calling_benchmark.eval.global_metrics import (
    _compute_locus_isoform_metrics,
    compute_global_metrics,
)
from gene_calling_benchmark.label_definition import AnnotationMode, LabelConfig
from gene_calling_benchmark.transcript_mapping import (
    LocusMatchingMode,
    MatchClass,
    PredictionMatch,
    TranscriptMapping,
)

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

_CFG = LabelConfig(
    annotation_mode=AnnotationMode.EXON_INTRON,
    background_label=8,
    exon_label=0,
)
_TRANSCRIPT_TYPES = ["mRNA"]
_PRED = "pred"


# ---------------------------------------------------------------------------
# DataFrame / TranscriptMapping construction helpers
# ---------------------------------------------------------------------------


def _row(seqid, strand, type_, start, end, gff_id, parent=None):
    return {
        "seqid": seqid,
        "strand": strand,
        "type": type_,
        "start": start,
        "end": end,
        "gff_id": gff_id,
        "parent": parent,
    }


def _tx(gff_id, start, end, seqid="chr1", strand="+"):
    return _row(seqid, strand, "mRNA", start, end, gff_id)


def _ex(gff_id, start, end, parent, seqid="chr1", strand="+"):
    return _row(seqid, strand, "exon", start, end, gff_id, parent)


def _match(transcript_id, start, end):
    return PredictionMatch(
        predictor_name=_PRED,
        transcript_id=transcript_id,
        start=start,
        end=end,
        match_class=MatchClass.EXACT,
        base_overlap=end - start + 1,
        junction_f1=1.0,
    )


def _mapping(gt_id, start, end, matches=(), seqid="chr1", strand="+", is_unmatched=False):
    return TranscriptMapping(
        seqid=seqid,
        strand=strand,
        gt_id=gt_id,
        gt_start=start,
        gt_end=end,
        matched_predictions=list(matches),
        is_unmatched_prediction=is_unmatched,
    )


def _run(gt_rows, pred_rows, mappings):
    return compute_global_metrics(
        gt_df=pd.DataFrame(gt_rows),
        pred_df=pd.DataFrame(pred_rows),
        mappings=mappings,
        predictor_name=_PRED,
        label_config=_CFG,
        transcript_types=_TRANSCRIPT_TYPES,
    )


# ---------------------------------------------------------------------------
# Transcript-level metrics
# ---------------------------------------------------------------------------


def test_transcript_perfect_match():
    """Sensitivity=1.0 and precision=1.0 when every GT transcript is matched."""
    gt = [_tx("tx1", 100, 200), _ex("ex1", 100, 200, "tx1")]
    pred = [_tx("p1", 100, 200), _ex("px1", 100, 200, "p1")]
    mappings = [_mapping("tx1", 100, 200, [_match("p1", 100, 200)])]

    r = _run(gt, pred, mappings)["transcript"]

    assert r["ref_transcript_count"] == 1
    assert r["ref_transcript_matched"] == 1
    assert math.isclose(r["sensitivity"], 1.0)
    assert math.isclose(r["precision"], 1.0)
    assert math.isclose(r["f1"], 1.0)


def test_transcript_missed_reference_lowers_sensitivity():
    """Sensitivity halves when one of two GT transcripts is unmatched."""
    gt = [
        _tx("tx1", 100, 200), _ex("ex1", 100, 200, "tx1"),
        _tx("tx2", 300, 400), _ex("ex2", 300, 400, "tx2"),
    ]
    pred = [_tx("p1", 100, 200), _ex("px1", 100, 200, "p1")]
    mappings = [
        _mapping("tx1", 100, 200, [_match("p1", 100, 200)]),
        _mapping("tx2", 300, 400),
    ]

    r = _run(gt, pred, mappings)["transcript"]

    assert r["ref_transcript_count"] == 2
    assert r["ref_transcript_matched"] == 1
    assert math.isclose(r["sensitivity"], 0.5)
    assert math.isclose(r["precision"], 1.0)


def test_transcript_hallucinated_prediction_lowers_precision():
    """Precision halves when a prediction has no GT assignment."""
    gt = [_tx("tx1", 100, 200), _ex("ex1", 100, 200, "tx1")]
    pred = [
        _tx("p1", 100, 200), _ex("px1", 100, 200, "p1"),
        _tx("p2", 500, 600), _ex("px2", 500, 600, "p2"),
    ]
    mappings = [
        _mapping("tx1", 100, 200, [_match("p1", 100, 200)]),
        # Unmatched prediction: no GT partner, lowers precision.
        _mapping("__unmatched_pred__p2", 500, 600, [_match("p2", 500, 600)], is_unmatched=True),
    ]

    r = _run(gt, pred, mappings)["transcript"]

    assert r["ref_transcript_count"] == 1
    assert r["ref_transcript_matched"] == 1
    assert math.isclose(r["sensitivity"], 1.0)
    assert math.isclose(r["precision"], 0.5)


# ---------------------------------------------------------------------------
# Exon-level metrics — de-duplication contract (S2 regression guard)
# ---------------------------------------------------------------------------


def test_exon_perfect_match():
    """Sensitivity=1.0 and precision=1.0 for an identical single exon."""
    gt = [_tx("tx1", 100, 200), _ex("ex1", 100, 200, "tx1")]
    pred = [_tx("p1", 100, 200), _ex("px1", 100, 200, "p1")]
    mappings = [_mapping("tx1", 100, 200, [_match("p1", 100, 200)])]

    r = _run(gt, pred, mappings)["exon"]["scopes"]["transcript_exon"]

    assert r["ref_exon_count"] == 1
    assert r["ref_exon_matched"] == 1
    assert math.isclose(r["sensitivity"], 1.0)
    assert math.isclose(r["precision"], 1.0)


def test_exon_dedup_shared_interval_counted_once():
    """
    A (seqid, strand, start, end) shared by two GT isoforms is counted once,
    not once per isoform.  This is the de-dup contract asserted by S2.

    GT locus:
        tx1 = [ex_A(100,150), ex_B(200,300)]
        tx2 = [ex_A(100,150), ex_C(250,300)]   ← same ex_A coordinates
    Pred:
        p1  = [ex_A(100,150), ex_B(200,300)]   ← matches tx1 exactly

    De-duplicated GT unique exons: {(100,150), (200,300), (250,300)} = 3
    Pred unique exons:             {(100,150), (200,300)}            = 2
    Matched:                       2
    → sensitivity = 2/3,  precision = 1.0

    If de-dup were absent (per-isoform counting):
        GT exon count would be 4 → sensitivity would be 0.5
    """
    gt = [
        _tx("tx1", 100, 300),
        _ex("ea1", 100, 150, "tx1"),
        _ex("eb1", 200, 300, "tx1"),
        _tx("tx2", 100, 300),
        _ex("ea2", 100, 150, "tx2"),  # same coordinates as ea1
        _ex("ec2", 250, 300, "tx2"),
    ]
    pred = [
        _tx("p1", 100, 300),
        _ex("pea", 100, 150, "p1"),
        _ex("peb", 200, 300, "p1"),
    ]
    mappings = [
        _mapping("tx1", 100, 300, [_match("p1", 100, 300)]),
        _mapping("tx2", 100, 300),
    ]

    r = _run(gt, pred, mappings)["exon"]["scopes"]["transcript_exon"]

    assert r["ref_exon_count"] == 3, (
        f"Expected 3 unique GT exons after de-dup, got {r['ref_exon_count']}. "
        "If 4, de-dup is not working (counting per-isoform)."
    )
    assert r["ref_exon_matched"] == 2
    assert math.isclose(r["sensitivity"], 2 / 3)
    assert math.isclose(r["precision"], 1.0)


# ---------------------------------------------------------------------------
# Intron-chain metrics — coordinate-exact, mapping-independent
# ---------------------------------------------------------------------------


def test_intron_chain_perfect_match():
    """Identical two-exon intron chain → sensitivity = precision = 1.0."""
    gt = [_tx("tx1", 100, 300), _ex("ea", 100, 150, "tx1"), _ex("eb", 200, 300, "tx1")]
    pred = [_tx("p1", 100, 300), _ex("pa", 100, 150, "p1"), _ex("pb", 200, 300, "p1")]

    r = _run(gt, pred, [])["intron_chain"]["scopes"]["transcript_exon"]

    assert r["ref_chain_count"] == 1
    assert r["ref_chain_matched"] == 1
    assert math.isclose(r["sensitivity"], 1.0)
    assert math.isclose(r["precision"], 1.0)


def test_intron_chain_single_exon_excluded():
    """Single-exon transcripts carry no chain → empty denominator, Sn = 0.0."""
    gt = [_tx("tx1", 100, 200), _ex("ea", 100, 200, "tx1")]
    pred = [_tx("p1", 100, 200), _ex("pa", 100, 200, "p1")]

    r = _run(gt, pred, [])["intron_chain"]["scopes"]["transcript_exon"]

    assert r["ref_chain_count"] == 0
    assert math.isclose(r["sensitivity"], 0.0)


def test_intron_chain_splice_shift_breaks_match():
    """A shifted internal splice site (different intron) is not a match."""
    gt = [_tx("tx1", 100, 300), _ex("ea", 100, 150, "tx1"), _ex("eb", 200, 300, "tx1")]
    # First exon ends at 160 → intron 161-199 instead of 151-199.
    pred = [_tx("p1", 100, 300), _ex("pa", 100, 160, "p1"), _ex("pb", 200, 300, "p1")]

    r = _run(gt, pred, [])["intron_chain"]["scopes"]["transcript_exon"]

    assert r["ref_chain_count"] == 1
    assert r["ref_chain_matched"] == 0
    assert math.isclose(r["sensitivity"], 0.0)
    assert math.isclose(r["precision"], 0.0)


# ---------------------------------------------------------------------------
# Whole-transcript exact-structure metrics
# ---------------------------------------------------------------------------


def test_transcript_exact_full_structure_match():
    """Identical multi-exon structure → sensitivity = precision = 1.0."""
    gt = [_tx("tx1", 100, 300), _ex("ea", 100, 150, "tx1"), _ex("eb", 200, 300, "tx1")]
    pred = [_tx("p1", 100, 300), _ex("pa", 100, 150, "p1"), _ex("pb", 200, 300, "p1")]

    r = _run(gt, pred, [])["transcript_exact"]["scopes"]["transcript_exon"]

    assert r["ref_transcript_count"] == 1
    assert r["ref_transcript_matched"] == 1
    assert math.isclose(r["sensitivity"], 1.0)
    assert math.isclose(r["precision"], 1.0)


def test_transcript_terminal_boundary_is_lenient():
    """A terminal exon that merely extends (e.g. into UTR) still matches: only
    the outer boundary of the first/last exon is wildcarded (gffcompare style)."""
    gt = [_tx("tx1", 100, 300), _ex("ea", 100, 150, "tx1"), _ex("eb", 200, 300, "tx1")]
    # Identical intron (151-199); the last exon extends to 320 (UTR-like).
    pred = [_tx("p1", 100, 320), _ex("pa", 100, 150, "p1"), _ex("pb", 200, 320, "p1")]

    tx = _run(gt, pred, [])["transcript_exact"]["scopes"]["transcript_exon"]

    assert tx["ref_transcript_matched"] == 1
    assert math.isclose(tx["sensitivity"], 1.0)


def test_transcript_internal_splice_shift_breaks_match():
    """An internal splice shift changes a non-terminal boundary → no match,
    even though terminal boundaries are lenient."""
    gt = [_tx("tx1", 100, 400), _ex("ea", 100, 150, "tx1"), _ex("eb", 200, 250, "tx1"), _ex("ec", 300, 400, "tx1")]
    # Middle exon shifted (200-260 instead of 200-250) → internal boundary differs.
    pred = [_tx("p1", 100, 400), _ex("pa", 100, 150, "p1"), _ex("pb", 200, 260, "p1"), _ex("pc", 300, 400, "p1")]

    tx = _run(gt, pred, [])["transcript_exact"]["scopes"]["transcript_exon"]

    assert tx["ref_transcript_matched"] == 0
    assert math.isclose(tx["sensitivity"], 0.0)


def test_transcript_exact_includes_single_exon():
    """Single-exon transcripts are part of the transcript denominator."""
    gt = [_tx("tx1", 100, 200), _ex("ea", 100, 200, "tx1")]
    pred = [_tx("p1", 100, 200), _ex("pa", 100, 200, "p1")]

    r = _run(gt, pred, [])["transcript_exact"]["scopes"]["transcript_exon"]

    assert r["ref_transcript_count"] == 1
    assert r["ref_transcript_matched"] == 1
    assert math.isclose(r["sensitivity"], 1.0)


# ---------------------------------------------------------------------------
# Nucleotide metrics
# ---------------------------------------------------------------------------


def test_nucleotide_perfect_match():
    """Precision=1.0 and recall=1.0 for identical exon coverage."""
    gt = [_tx("tx1", 100, 200), _ex("ex1", 100, 200, "tx1")]
    pred = [_tx("p1", 100, 200), _ex("px1", 100, 200, "p1")]
    mappings = [_mapping("tx1", 100, 200, [_match("p1", 100, 200)])]

    r = _run(gt, pred, mappings)["nucleotide"]["scopes"]["transcript_exon"]

    assert r["tp"] > 0
    assert r["fp"] == 0
    assert r["fn"] == 0
    assert math.isclose(r["precision"], 1.0)
    assert math.isclose(r["recall"], 1.0)
    assert math.isclose(r["f1"], 1.0)


def test_nucleotide_partial_coverage():
    """Missed exon bases appear as false negatives (fn > 0, recall < 1)."""
    # GT exon spans 100–200 (101 bases); pred exon only covers 100–150 (51 bases).
    gt = [_tx("tx1", 100, 200), _ex("ex1", 100, 200, "tx1")]
    pred = [_tx("p1", 100, 150), _ex("px1", 100, 150, "p1")]
    mappings = [_mapping("tx1", 100, 200, [_match("p1", 100, 150)])]

    r = _run(gt, pred, mappings)["nucleotide"]["scopes"]["transcript_exon"]

    assert r["tp"] == 51   # bases 100-150 inclusive
    assert r["fn"] == 50   # bases 151-200 inclusive
    assert r["fp"] == 0
    assert math.isclose(r["recall"], 51 / 101)
    assert math.isclose(r["precision"], 1.0)


# ---------------------------------------------------------------------------
# Gene / locus-level metrics
# ---------------------------------------------------------------------------


def test_gene_perfect_match():
    """Single matched locus → sensitivity=1.0, precision=1.0."""
    gt = [_tx("tx1", 100, 200), _ex("ex1", 100, 200, "tx1")]
    pred = [_tx("p1", 100, 200), _ex("px1", 100, 200, "p1")]
    mappings = [_mapping("tx1", 100, 200, [_match("p1", 100, 200)])]

    r = _run(gt, pred, mappings)["gene"]

    assert r["ref_locus_count"] == 1
    assert r["ref_locus_matched"] == 1
    assert math.isclose(r["sensitivity"], 1.0)
    assert math.isclose(r["precision"], 1.0)


def test_gene_missed_locus_lowers_sensitivity():
    """GT locus on chr2 has no prediction → gene sensitivity = 0.5."""
    gt = [
        _tx("tx1", 100, 200),
        _row("chr2", "+", "mRNA", 100, 200, "tx2"),
    ]
    pred = [_tx("p1", 100, 200)]
    mappings = [
        _mapping("tx1", 100, 200, [_match("p1", 100, 200)]),
        _mapping("tx2", 100, 200, seqid="chr2"),
    ]

    r = _run(gt, pred, mappings)["gene"]

    assert r["ref_locus_count"] == 2
    assert r["ref_locus_matched"] == 1
    assert math.isclose(r["sensitivity"], 0.5)
    assert math.isclose(r["precision"], 1.0)


# ---------------------------------------------------------------------------
# Locus isoform metrics
# ---------------------------------------------------------------------------


def test_locus_isoform_partial_recall():
    """
    One locus with two overlapping GT isoforms; only one receives a prediction.
    Expected: recall=0.5, locus_count=1, missed_per_locus=[1].
    """
    mappings = [
        _mapping("tx1", 100, 300, [_match("p1", 100, 300)]),
        _mapping("tx2", 150, 250),  # overlaps tx1 → same locus, no match
    ]
    gt = [_tx("tx1", 100, 300), _tx("tx2", 150, 250)]
    pred = [_tx("p1", 100, 300)]

    r = _run(gt, pred, mappings)["locus_isoform"]

    assert r["locus_count"] == 1
    assert r["ref_isoform_count"] == 2
    assert r["ref_isoform_matched"] == 1
    assert math.isclose(r["recall"], 0.5)
    assert r["missed_per_locus"] == [1]


def test_locus_isoform_full_recall():
    """Both isoforms in one locus are matched → recall=1.0, missed=[0]."""
    mappings = [
        _mapping("tx1", 100, 300, [_match("p1", 100, 300)]),
        _mapping("tx2", 150, 250, [_match("p2", 150, 250)]),
    ]
    gt = [_tx("tx1", 100, 300), _tx("tx2", 150, 250)]
    pred = [_tx("p1", 100, 300), _tx("p2", 150, 250)]

    r = _run(gt, pred, mappings)["locus_isoform"]

    assert r["locus_count"] == 1
    assert r["ref_isoform_count"] == 2
    assert r["ref_isoform_matched"] == 2
    assert math.isclose(r["recall"], 1.0)
    assert r["missed_per_locus"] == [0]


def test_locus_isoform_two_separate_loci():
    """Transcripts on different chromosomes form independent loci."""
    mappings = [
        _mapping("tx1", 100, 200, [_match("p1", 100, 200)]),         # chr1
        _mapping("tx2", 100, 200, seqid="chr2"),                      # chr2, missed
    ]
    gt = [_tx("tx1", 100, 200), _row("chr2", "+", "mRNA", 100, 200, "tx2")]
    pred = [_tx("p1", 100, 200)]

    r = _run(gt, pred, mappings)["locus_isoform"]

    assert r["locus_count"] == 2
    assert r["ref_isoform_count"] == 2
    assert r["ref_isoform_matched"] == 1
    assert math.isclose(r["recall"], 0.5)
    assert sorted(r["missed_per_locus"]) == [0, 1]


# ---------------------------------------------------------------------------
# BEST_PER_LOCUS: locus_isoform must count each locus once per predictor
# ---------------------------------------------------------------------------


def _bpl_match(pred, start, end, junction_f1, match_class=MatchClass.EXACT):
    return PredictionMatch(
        predictor_name=pred,
        transcript_id=f"{pred}_t",
        start=start,
        end=end,
        match_class=match_class,
        base_overlap=end - start + 1,
        junction_f1=junction_f1,
    )


def test_locus_isoform_best_per_locus_no_double_count():
    """A peer's match must not inflate another predictor's isoform denominator.

    Regression for the locus-FN double-count: two single-isoform loci; 'good'
    matches both, 'bad' matches only locus A. Under the bug 'bad' reported
    recall 1/3; correct is 1/2.
    """
    mappings = [
        # Locus A — matched by both (shared Case-A entry).
        TranscriptMapping(
            seqid="chr1", strand="+", gt_id="gtA", gt_start=100, gt_end=400,
            matched_predictions=[_bpl_match("good", 100, 400, 1.0),
                                 _bpl_match("bad", 100, 400, 1.0)],
        ),
        # Locus B — Case A for 'good'.
        TranscriptMapping(
            seqid="chr1", strand="+", gt_id="gtB", gt_start=1000, gt_end=1400,
            matched_predictions=[_bpl_match("good", 1000, 1400, 1.0)],
        ),
        # Locus B — Case B (clean miss) for 'bad'.
        TranscriptMapping(
            seqid="chr1", strand="+", gt_id="gtB", gt_start=1000, gt_end=1400,
            matched_predictions=[], fn_for_predictors=["bad"],
        ),
    ]
    bpl = LocusMatchingMode.BEST_PER_LOCUS
    bad = _compute_locus_isoform_metrics(mappings, "bad", bpl)
    good = _compute_locus_isoform_metrics(mappings, "good", bpl)

    assert bad["ref_isoform_count"] == 2          # not 3
    assert bad["ref_isoform_matched"] == 1
    assert math.isclose(bad["recall"], 0.5)        # not 0.333
    assert good["ref_isoform_count"] == 2
    assert math.isclose(good["recall"], 1.0)


def test_locus_isoform_case_c_overlap_counts_as_miss():
    """A Case-C overlap (junction_f1 == 0) is not a recovered isoform."""
    mappings = [
        TranscriptMapping(
            seqid="chr1", strand="+", gt_id="gtB", gt_start=1000, gt_end=1400,
            matched_predictions=[
                _bpl_match("ugly", 1000, 1400, 0.0, MatchClass.OVERLAPPING)
            ],
        ),
    ]
    r = _compute_locus_isoform_metrics(mappings, "ugly", LocusMatchingMode.BEST_PER_LOCUS)
    assert r["ref_isoform_count"] == 1
    assert r["ref_isoform_matched"] == 0
    assert math.isclose(r["recall"], 0.0)
