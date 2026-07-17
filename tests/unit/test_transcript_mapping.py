"""Tests for the transcript mapping module.

Covers:
- Base overlap computation
- Pair classification (intron chain matching)
- Strand-aware mapping of GT <-> predictions
- Unmatched GT transcripts (no prediction -> empty pred array)
- Unmatched predictions (no GT transcript -> GT from region features)
- Array construction from mappings
- Debug TSV export
"""

import numpy as np
import pandas as pd
import pytest

from gene_calling_benchmark.io_utils import collect_gff
from gene_calling_benchmark.transcript_mapping import (
    LocusMatchingMode,
    MatchClass,
    _TranscriptInfo,
    _base_overlap,
    _build_intron_chain_index,
    _classify_pair,
    build_paired_arrays,
    export_mapping_table,
    map_transcripts,
)

from support.gff import UTR_ROLE_MAP, UTR_ROLE_MAP_NO_CDS, write_gff


# ------------------------------------------------------------------
# Fixtures (``simple_config`` / ``utr_config`` / ``utr_gt_gff`` /
# ``utr_pred_gff`` come from conftest)
# ------------------------------------------------------------------


@pytest.fixture
def gt_gff(tmp_path):
    """Ground-truth GFF with two transcripts on different strands.

    gene1 (chr1, +, 1-100) -> mRNA1 (1-100) -> CDS (10-30)
    gene2 (chr1, -, 200-400) -> mRNA2 (200-400) -> CDS (250-300)
    gene3 (chr2, +, 1-50) -> mRNA3 (1-50) -> CDS (5-20)
    """
    content = """\
##gff-version 3
chr1\tTest\tgene\t1\t100\t.\t+\t.\tID=gene1
chr1\tTest\tmRNA\t1\t100\t.\t+\t.\tID=mRNA1;Parent=gene1
chr1\tTest\tCDS\t10\t30\t.\t+\t0\tID=cds1;Parent=mRNA1
chr1\tTest\tgene\t200\t400\t.\t-\t.\tID=gene2
chr1\tTest\tmRNA\t200\t400\t.\t-\t.\tID=mRNA2;Parent=gene2
chr1\tTest\tCDS\t250\t300\t.\t-\t0\tID=cds2;Parent=mRNA2
chr2\tTest\tgene\t1\t50\t.\t+\t.\tID=gene3
chr2\tTest\tmRNA\t1\t50\t.\t+\t.\tID=mRNA3;Parent=gene3
chr2\tTest\tCDS\t5\t20\t.\t+\t0\tID=cds3;Parent=mRNA3
"""
    f = tmp_path / "gt.gff"
    f.write_text(content)
    return str(f)


@pytest.fixture
def pred_a_gff(tmp_path):
    """Prediction A: overlaps both GT transcripts on chr1.

    predA_t1 (chr1, +, 5-80) -> CDS (15-35), overlaps mRNA1
    predA_t2 (chr1, -, 210-380) -> CDS (260-290), overlaps mRNA2
    """
    content = """\
##gff-version 3
chr1\tPredA\tmRNA\t5\t80\t.\t+\t.\tID=predA_t1
chr1\tPredA\tCDS\t15\t35\t.\t+\t0\tID=predA_cds1;Parent=predA_t1
chr1\tPredA\tmRNA\t210\t380\t.\t-\t.\tID=predA_t2
chr1\tPredA\tCDS\t260\t290\t.\t-\t0\tID=predA_cds2;Parent=predA_t2
"""
    f = tmp_path / "pred_a.gff"
    f.write_text(content)
    return str(f)


@pytest.fixture
def pred_b_gff(tmp_path):
    """Prediction B: one unmatched + one overlapping mRNA1.

    predB_t1 (chr1, +, 500-600) -> CDS (520-560), no GT overlap
    predB_t2 (chr1, +, 20-90) -> CDS (25-40), overlaps mRNA1
    """
    content = """\
##gff-version 3
chr1\tPredB\tmRNA\t500\t600\t.\t+\t.\tID=predB_t1
chr1\tPredB\tCDS\t520\t560\t.\t+\t0\tID=predB_cds1;Parent=predB_t1
chr1\tPredB\tmRNA\t20\t90\t.\t+\t.\tID=predB_t2
chr1\tPredB\tCDS\t25\t40\t.\t+\t0\tID=predB_cds2;Parent=predB_t2
"""
    f = tmp_path / "pred_b.gff"
    f.write_text(content)
    return str(f)


@pytest.fixture
def pred_wrong_strand_gff(tmp_path):
    """Prediction on the wrong strand -- must NOT map to mRNA1 (+).

    pred_ws_t1 (chr1, -, 5-80): same coords as mRNA1 but on '-' strand.
    """
    content = """\
##gff-version 3
chr1\tPredWS\tmRNA\t5\t80\t.\t-\t.\tID=pred_ws_t1
chr1\tPredWS\tCDS\t15\t35\t.\t-\t0\tID=pred_ws_cds1;Parent=pred_ws_t1
"""
    f = tmp_path / "pred_ws.gff"
    f.write_text(content)
    return str(f)


@pytest.fixture
def overlapping_gt_gff(tmp_path):
    """GT with two overlapping transcripts on the same strand."""
    content = """\
##gff-version 3
chr1\tTest\tmRNA\t1\t100\t.\t+\t.\tID=mRNA_A
chr1\tTest\tCDS\t10\t30\t.\t+\t0\tID=cds_A;Parent=mRNA_A
chr1\tTest\tmRNA\t50\t150\t.\t+\t.\tID=mRNA_B
chr1\tTest\tCDS\t60\t90\t.\t+\t0\tID=cds_B;Parent=mRNA_B
"""
    f = tmp_path / "overlapping_gt.gff"
    f.write_text(content)
    return str(f)


@pytest.fixture
def pred_only_chr3_gff(tmp_path):
    """Prediction on chr3, which has no GT data at all."""
    content = """\
##gff-version 3
chr3\tPred\tmRNA\t1\t50\t.\t+\t.\tID=pred_chr3_t1
chr3\tPred\tCDS\t10\t30\t.\t+\t0\tID=pred_chr3_cds1;Parent=pred_chr3_t1
"""
    f = tmp_path / "pred_chr3.gff"
    f.write_text(content)
    return str(f)


@pytest.fixture
def gt_two_loci_gff(tmp_path):
    """Two GT transcripts in separate loci (large intergenic gap)."""
    content = """\
##gff-version 3
chr1\tTest\tmRNA\t100\t200\t.\t+\t.\tID=gt_left
chr1\tTest\tCDS\t120\t180\t.\t+\t0\tID=gt_left_cds;Parent=gt_left
chr1\tTest\tmRNA\t1000\t1100\t.\t+\t.\tID=gt_right
chr1\tTest\tCDS\t1020\t1080\t.\t+\t0\tID=gt_right_cds;Parent=gt_right
"""
    f = tmp_path / "gt_two_loci.gff"
    f.write_text(content)
    return str(f)


@pytest.fixture
def pred_spanning_gff(tmp_path):
    """One prediction spanning the intergenic gap, overlapping both GT loci."""
    content = """\
##gff-version 3
chr1\tPredSpan\tmRNA\t150\t1050\t.\t+\t.\tID=pred_span
chr1\tPredSpan\tCDS\t150\t1050\t.\t+\t0\tID=pred_span_cds;Parent=pred_span
"""
    f = tmp_path / "pred_spanning.gff"
    f.write_text(content)
    return str(f)


@pytest.fixture
def gt_over_gff(tmp_path):
    """GT transcript spanning 1-100 with CDS 10-90."""
    content = """\
##gff-version 3
chr1\tTest\tmRNA\t1\t100\t.\t+\t.\tID=gt_over
chr1\tTest\tCDS\t10\t90\t.\t+\t0\tID=gt_over_cds;Parent=gt_over
"""
    f = tmp_path / "gt_over.gff"
    f.write_text(content)
    return str(f)


@pytest.fixture
def pred_over_gff(tmp_path):
    """Prediction over-extending past the GT: span 1-150, CDS 10-120."""
    content = """\
##gff-version 3
chr1\tPredOver\tmRNA\t1\t150\t.\t+\t.\tID=pred_over
chr1\tPredOver\tCDS\t10\t120\t.\t+\t0\tID=pred_over_cds;Parent=pred_over
"""
    f = tmp_path / "pred_over.gff"
    f.write_text(content)
    return str(f)


# ------------------------------------------------------------------
# Unit tests: _base_overlap
# ------------------------------------------------------------------


def _make_single_exon(gff_id: str, start: int, end: int) -> _TranscriptInfo:
    return _TranscriptInfo(
        gff_id=gff_id, start=start, end=end,
        intron_chain=frozenset(), is_single_exon=True,
    )


def _make_multi_exon(
    gff_id: str,
    start: int,
    end: int,
    introns: list[tuple[int, int]],
) -> _TranscriptInfo:
    return _TranscriptInfo(
        gff_id=gff_id, start=start, end=end,
        intron_chain=frozenset(introns), is_single_exon=False,
    )


class TestBaseOverlap:
    """Tests for the base overlap helper."""

    def test_identical_intervals(self):
        assert _base_overlap(10, 50, 10, 50) == 41

    def test_no_overlap(self):
        assert _base_overlap(10, 20, 30, 40) == 0

    def test_partial_overlap(self):
        # [10, 30] and [20, 40]: overlap = [20, 30] = 11 bases
        assert _base_overlap(10, 30, 20, 40) == 11

    def test_containment(self):
        # [10, 50] contains [20, 30]: overlap = 11
        assert _base_overlap(10, 50, 20, 30) == 11

    def test_adjacent_no_overlap(self):
        assert _base_overlap(10, 20, 21, 30) == 0

    def test_single_base_overlap(self):
        assert _base_overlap(10, 20, 20, 30) == 1


# ------------------------------------------------------------------
# Unit tests: _classify_pair
# ------------------------------------------------------------------


class TestClassifyPair:
    """Tests for the intron-chain pair classifier."""

    def test_single_exon_exact(self):
        gt = _make_single_exon("gt", 1, 100)
        pred = _make_single_exon("pred", 1, 100)
        assert _classify_pair(gt, pred) == MatchClass.EXACT

    def test_single_exon_contained(self):
        gt = _make_single_exon("gt", 1, 100)
        pred = _make_single_exon("pred", 20, 80)
        assert _classify_pair(gt, pred) == MatchClass.CONTAINED

    def test_single_exon_contains(self):
        gt = _make_single_exon("gt", 20, 80)
        pred = _make_single_exon("pred", 1, 100)
        assert _classify_pair(gt, pred) == MatchClass.CONTAINS

    def test_single_exon_overlapping(self):
        gt = _make_single_exon("gt", 1, 60)
        pred = _make_single_exon("pred", 40, 100)
        assert _classify_pair(gt, pred) == MatchClass.OVERLAPPING

    def test_exact_match(self):
        introns = [(30, 50), (80, 100)]
        gt = _make_multi_exon("gt", 1, 150, introns)
        pred = _make_multi_exon("pred", 1, 150, introns)
        assert _classify_pair(gt, pred) == MatchClass.EXACT

    def test_contained(self):
        gt_introns = [(30, 50), (80, 100), (130, 150)]
        pred_introns = [(30, 50), (80, 100)]
        gt = _make_multi_exon("gt", 1, 200, gt_introns)
        pred = _make_multi_exon("pred", 1, 120, pred_introns)
        assert _classify_pair(gt, pred) == MatchClass.CONTAINED

    def test_contains(self):
        gt_introns = [(30, 50)]
        pred_introns = [(30, 50), (80, 100)]
        gt = _make_multi_exon("gt", 10, 60, gt_introns)
        pred = _make_multi_exon("pred", 1, 200, pred_introns)
        assert _classify_pair(gt, pred) == MatchClass.CONTAINS

    def test_shared_junction(self):
        gt_introns = [(30, 50), (80, 100)]
        pred_introns = [(30, 50), (120, 150)]
        gt = _make_multi_exon("gt", 1, 200, gt_introns)
        pred = _make_multi_exon("pred", 1, 200, pred_introns)
        assert _classify_pair(gt, pred) == MatchClass.SHARED_JUNCTION

    def test_overlapping_no_shared_junctions(self):
        gt_introns = [(30, 50)]
        pred_introns = [(70, 90)]
        gt = _make_multi_exon("gt", 1, 100, gt_introns)
        pred = _make_multi_exon("pred", 1, 100, pred_introns)
        assert _classify_pair(gt, pred) == MatchClass.OVERLAPPING


# ------------------------------------------------------------------
# Unit tests: _build_intron_chain_index
# ------------------------------------------------------------------


class TestBuildIntronChainIndex:
    """The intron-chain index must merge interleaved exon/CDS rows."""

    def test_gencode_style_exon_and_cds_rows_yield_clean_chain(self):
        """A transcript with both exon rows and nested CDS rows (GENCODE/RefSeq
        convention) must not produce spurious reverse-coordinate junctions.

        tx1: exon [100,200] + exon [300,400] (one real intron 200->300), with
        nested CDS [150,200] and CDS [300,350].  Sorting by start interleaves
        them; without merging, exon->CDS transitions would emit (200,150) and
        (400,300).
        """
        df = pd.DataFrame(
            [
                {"seqid": "chr1", "type": "exon", "parent": "tx1", "start": 100, "end": 200},
                {"seqid": "chr1", "type": "CDS", "parent": "tx1", "start": 150, "end": 200},
                {"seqid": "chr1", "type": "exon", "parent": "tx1", "start": 300, "end": 400},
                {"seqid": "chr1", "type": "CDS", "parent": "tx1", "start": 300, "end": 350},
            ]
        )
        index = _build_intron_chain_index(df, "chr1", ["exon", "CDS"])
        assert index["tx1"] == frozenset({(200, 300)})

    def test_single_merged_exon_is_single_exon(self):
        """Overlapping exon + CDS for a single-exon transcript -> empty chain."""
        df = pd.DataFrame(
            [
                {"seqid": "chr1", "type": "exon", "parent": "tx1", "start": 100, "end": 200},
                {"seqid": "chr1", "type": "CDS", "parent": "tx1", "start": 120, "end": 180},
            ]
        )
        index = _build_intron_chain_index(df, "chr1", ["exon", "CDS"])
        assert index["tx1"] == frozenset()


# ------------------------------------------------------------------
# Integration tests: map_transcripts
# ------------------------------------------------------------------


class TestMapTranscripts:
    """Integration tests for the main mapping function."""

    def test_basic_mapping_single_predictor(self, gt_gff, pred_a_gff):
        """PredA overlaps both GT transcripts on chr1."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff},

            exclude_features=["gene"],
        )

        gt_ids = {
            m.gt_id for m in mappings
            if not m.is_unmatched_prediction
        }
        assert "mRNA1" in gt_ids
        assert "mRNA2" in gt_ids

        for m in mappings:
            for match in m.matched_predictions:
                assert match.predictor_name == "PredA"

    def test_unmatched_prediction(self, gt_gff, pred_b_gff):
        """predB_t1 doesn't overlap any GT -> unmatched entry."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredB": pred_b_gff},

            exclude_features=["gene"],
        )

        unmatched = [m for m in mappings if m.is_unmatched_prediction]
        assert len(unmatched) >= 1

        unmatched_pred_ids = {
            match.transcript_id
            for m in unmatched
            for match in m.matched_predictions
        }
        assert "predB_t1" in unmatched_pred_ids

    def test_unmatched_gt_transcripts_included(self, gt_gff, pred_a_gff):
        """GT transcripts with no matching prediction are still included."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff},

            exclude_features=["gene"],
        )

        # mRNA3 is on chr2 but PredA has no chr2 predictions.
        # It should still appear in the output with empty matches.
        mRNA3_mappings = [
            m for m in mappings
            if m.gt_id == "mRNA3" and not m.is_unmatched_prediction
        ]
        assert len(mRNA3_mappings) == 1
        assert mRNA3_mappings[0].matched_predictions == []

    def test_strand_isolation(self, gt_gff, pred_wrong_strand_gff):
        """A prediction on the wrong strand must not map to a GT transcript."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredWS": pred_wrong_strand_gff},

            exclude_features=["gene"],
        )

        real_mappings = [
            m for m in mappings if not m.is_unmatched_prediction
        ]
        for m in real_mappings:
            pred_ids = {p.transcript_id for p in m.matched_predictions}
            assert "pred_ws_t1" not in pred_ids

    def test_multiple_predictors(self, gt_gff, pred_a_gff, pred_b_gff):
        """Multiple predictors can be passed at once."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff, "PredB": pred_b_gff},

            exclude_features=["gene"],
        )

        mRNA1_mappings = [
            m for m in mappings
            if m.gt_id == "mRNA1" and not m.is_unmatched_prediction
        ]
        assert len(mRNA1_mappings) == 1

        predictor_names = {
            p.predictor_name
            for p in mRNA1_mappings[0].matched_predictions
        }
        assert "PredA" in predictor_names
        assert "PredB" in predictor_names

    def test_overlapping_gt_accepted(
        self, overlapping_gt_gff, pred_a_gff,
    ):
        """Overlapping GT transcripts on the same strand are supported."""
        mappings = map_transcripts(
            gt_path=overlapping_gt_gff,
            pred_paths={"PredA": pred_a_gff},

        )
        # Both overlapping GT transcripts should appear in the mappings
        gt_ids = {m.gt_id for m in mappings if not m.is_unmatched_prediction}
        assert len(gt_ids) >= 2

    def test_prediction_spanning_two_loci_matched_once(
        self, gt_two_loci_gff, pred_spanning_gff,
    ):
        """A single prediction overlapping two GT loci is assigned to only one,
        not double-counted across loci (regression)."""
        mappings = map_transcripts(
            gt_path=gt_two_loci_gff,
            pred_paths={"PredSpan": pred_spanning_gff},
            exclude_features=["gene"],
        )
        matched_to_span = [
            m
            for m in mappings
            if not m.is_unmatched_prediction
            and any(p.transcript_id == "pred_span" for p in m.matched_predictions)
        ]
        assert len(matched_to_span) == 1

    def test_prediction_on_unknown_chromosome(
        self, gt_gff, pred_only_chr3_gff,
    ):
        """Predictions on seqids not in GT produce unmatched entries."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"Pred": pred_only_chr3_gff},

            exclude_features=["gene"],
        )

        chr3_unmatched = [
            m for m in mappings
            if m.is_unmatched_prediction and m.seqid == "chr3"
        ]
        assert len(chr3_unmatched) == 1
        assert chr3_unmatched[0].matched_predictions[0].transcript_id == (
            "pred_chr3_t1"
        )


# ------------------------------------------------------------------
# Integration tests: map_transcripts (BEST_PER_LOCUS)
# ------------------------------------------------------------------


class TestMapTranscriptsBestPerLocus:
    """BEST_PER_LOCUS scores every GT locus once per predictor: matched loci as
    pairs, missed loci as locus-level FN, intergenic predictions as FP."""

    def _locus_id(self, m):
        return m.gt_id

    def test_missed_locus_becomes_locus_fn(self, gt_gff, pred_a_gff):
        """A GT locus the predictor never matched is a locus-level FN entry."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff},
            exclude_features=["gene"],
            locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
        )
        # PredA covers chr1 (mRNA1, mRNA2) but nothing on chr2 (mRNA3).
        fn = {self._locus_id(m): m for m in mappings if m.fn_for_predictors}
        assert set(fn) == {"mRNA3"}
        assert fn["mRNA3"].fn_for_predictors == ["PredA"]
        assert fn["mRNA3"].matched_predictions == []
        assert not fn["mRNA3"].is_unmatched_prediction

    def test_intergenic_prediction_is_fp_overlapping_is_not(self, gt_gff, pred_b_gff):
        """Only predictions overlapping no GT locus become FP (no double count)."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredB": pred_b_gff},
            exclude_features=["gene"],
            locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
        )
        fp_ids = {
            match.transcript_id
            for m in mappings if m.is_unmatched_prediction
            for match in m.matched_predictions
        }
        assert "predB_t1" in fp_ids       # 500-600: overlaps no GT → FP
        assert "predB_t2" not in fp_ids   # overlaps mRNA1 → never a separate FP

    def test_locus_fn_is_per_predictor(self, gt_gff, pred_a_gff, pred_b_gff):
        """Each predictor is charged a clean FN only for loci it personally missed.

        With per-(locus, predictor) emission, a missed locus yields one Case-B
        entry per missing predictor (same ``gt_id``), so aggregate across entries.
        """
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff, "PredB": pred_b_gff},
            exclude_features=["gene"],
            locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
        )
        fn: dict[str, list[str]] = {}
        for m in mappings:
            if m.fn_for_predictors:
                fn.setdefault(self._locus_id(m), []).extend(m.fn_for_predictors)
        # mRNA1 matched by both → no FN; mRNA2 matched by PredA only (PredB has no
        # overlap there → clean miss); mRNA3 (chr2) missed by both, neither overlaps.
        assert "mRNA1" not in fn
        assert sorted(fn["mRNA2"]) == ["PredB"]
        assert sorted(fn["mRNA3"]) == ["PredA", "PredB"]


# ------------------------------------------------------------------
# BEST_PER_LOCUS Case B (clean miss) vs Case C (overlap, wrong structure)
# ------------------------------------------------------------------


class TestBestPerLocusOverlapVsMiss:
    """An overlapping-but-wrong prediction (Case C) must be scored against the
    real GT, NOT dropped or treated as a clean miss (Case B)."""

    def _scenario(self, tmp_path):
        # One GT locus: 2-exon transcript gtB, intron 1101..1299.
        gt = write_gff(tmp_path / "gt.gff3", [
            "chr1\tT\ttranscript\t1000\t1400\t.\t+\t.\tID=gtB",
            "chr1\tT\texon\t1000\t1100\t.\t+\t.\tID=gtB.e1;Parent=gtB",
            "chr1\tT\texon\t1300\t1400\t.\t+\t.\tID=gtB.e2;Parent=gtB",
        ])
        # 'silent' predicts nothing (Case B).
        silent = write_gff(tmp_path / "silent.gff3", [])
        # 'ugly' overlaps the locus with a DIFFERENT intron (1151..1249) → no
        # shared junction → junction_f1 == 0 (Case C).
        ugly = write_gff(tmp_path / "ugly.gff3", [
            "chr1\tT\ttranscript\t1000\t1400\t.\t+\t.\tID=ugX",
            "chr1\tT\texon\t1000\t1150\t.\t+\t.\tID=ugX.e1;Parent=ugX",
            "chr1\tT\texon\t1250\t1400\t.\t+\t.\tID=ugX.e2;Parent=ugX",
        ])
        return gt, silent, ugly

    def test_case_c_scored_against_real_gt_not_dropped(self, tmp_path, simple_config):
        gt, silent, ugly = self._scenario(tmp_path)
        mappings = map_transcripts(
            gt_path=gt,
            pred_paths={"silent": silent, "ugly": ugly},
            label_config=simple_config,
            locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
        )
        # 'silent' → Case B: a clean-miss FN entry, no prediction.
        silent_entries = [
            m for m in mappings
            if "silent" in m.fn_for_predictors or
            any(pm.predictor_name == "silent" for pm in m.matched_predictions)
        ]
        assert len(silent_entries) == 1
        assert silent_entries[0].fn_for_predictors == ["silent"]
        assert silent_entries[0].matched_predictions == []

        # 'ugly' → Case C: a real pair against gtB, with junction_f1 == 0.
        ugly_entries = [
            m for m in mappings
            if any(pm.predictor_name == "ugly" for pm in m.matched_predictions)
        ]
        assert len(ugly_entries) == 1
        entry = ugly_entries[0]
        assert entry.gt_id == "gtB"
        assert entry.fn_for_predictors == []
        (pm,) = entry.matched_predictions
        assert pm.transcript_id == "ugX"
        assert pm.junction_f1 == 0.0
        assert pm.match_class == MatchClass.OVERLAPPING

    def test_case_c_pred_array_is_non_null(self, tmp_path, simple_config):
        """The Case-C entry yields a real (non-null) prediction array, so the
        wrong prediction is actually scored — unlike the clean miss."""
        gt, silent, ugly = self._scenario(tmp_path)
        gt_df = collect_gff(gt)
        pred_dfs = {"silent": collect_gff(silent), "ugly": collect_gff(ugly)}
        mappings = map_transcripts(
            gt_path=gt,
            pred_paths={"silent": silent, "ugly": ugly},
            label_config=simple_config,
            locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
        )
        ugly_entry = next(
            m for m in mappings
            if any(pm.predictor_name == "ugly" for pm in m.matched_predictions)
        )
        _gt_arr, pred_arrs = build_paired_arrays(
            mapping=ugly_entry,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=simple_config,
        )
        exon = simple_config.exon_label
        # 'ugly' painted exon bases; 'silent' (null) painted none.
        assert (pred_arrs["ugly"] == exon).any()
        assert not (pred_arrs["silent"] == exon).any()


# ------------------------------------------------------------------
# Integration tests: build_paired_arrays
# ------------------------------------------------------------------


class TestBuildPairedArrays:
    """Tests for array construction from mappings."""

    def _get_dfs(self, gt_gff, pred_paths, excl=None):
        """Helper: collect GT and pred DataFrames."""
        excl = excl or ["gene"]
        gt_df = collect_gff(gt_gff, exclude_features=excl)
        pred_dfs = {
            name: collect_gff(path, exclude_features=excl)
            for name, path in pred_paths.items()
        }
        return gt_df, pred_dfs

    def test_unmatched_prediction_gt_from_region(
        self, gt_gff, pred_b_gff, simple_config,
    ):
        """An unmatched prediction's GT array reflects actual GT features."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredB": pred_b_gff},

            exclude_features=["gene"],
        )

        unmatched = [m for m in mappings if m.is_unmatched_prediction]
        assert len(unmatched) >= 1

        gt_df, pred_dfs = self._get_dfs(
            gt_gff, {"PredB": pred_b_gff},
        )

        gt_arr, pred_arrs = build_paired_arrays(
            mapping=unmatched[0],
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=simple_config,
        )

        # predB_t1 is at 500-600, no GT features there -> all background
        np.testing.assert_array_equal(
            gt_arr, np.full(len(gt_arr), 1, dtype=np.int32),
        )

    def test_coding_regions_in_gt_array(
        self, gt_gff, pred_a_gff, simple_config,
    ):
        """A real GT mapping paints CDS regions as coding."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff},

            exclude_features=["gene"],
        )

        mRNA1_mapping = next(m for m in mappings if m.gt_id == "mRNA1")

        gt_df, pred_dfs = self._get_dfs(
            gt_gff, {"PredA": pred_a_gff},
        )

        gt_arr, pred_arrs = build_paired_arrays(
            mapping=mRNA1_mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=simple_config,
        )

        # mRNA1 spans 1-100 (100 bases), CDS at 10-30
        assert len(gt_arr) == 100
        np.testing.assert_array_equal(gt_arr[0:9], np.full(9, 1))
        np.testing.assert_array_equal(gt_arr[9:30], np.full(21, 0))
        np.testing.assert_array_equal(gt_arr[30:], np.full(70, 1))

    def test_prediction_array_content(
        self, gt_gff, pred_a_gff, simple_config,
    ):
        """Prediction arrays have coding regions at expected positions."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff},

            exclude_features=["gene"],
        )

        mRNA1_mapping = next(m for m in mappings if m.gt_id == "mRNA1")

        gt_df, pred_dfs = self._get_dfs(
            gt_gff, {"PredA": pred_a_gff},
        )

        gt_arr, pred_arrs = build_paired_arrays(
            mapping=mRNA1_mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=simple_config,
        )

        assert "PredA" in pred_arrs
        pred_arr = pred_arrs["PredA"]

        # mRNA1 spans 1-100; predA_t1 CDS is at 15-35
        # Local: 15-1=14 to 35-1+1=35 -> indices 14..34
        assert len(pred_arr) == 100
        np.testing.assert_array_equal(pred_arr[0:14], np.full(14, 1))
        np.testing.assert_array_equal(pred_arr[14:35], np.full(21, 0))
        np.testing.assert_array_equal(pred_arr[35:], np.full(65, 1))

    def test_unmatched_predictor_gets_background(
        self, gt_gff, pred_a_gff, pred_b_gff, simple_config,
    ):
        """A predictor with no match at a GT locus gets all-background."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff, "PredB": pred_b_gff},

            exclude_features=["gene"],
        )

        # mRNA2 (chr1, -) is matched by PredA but not PredB
        mRNA2_mapping = next(m for m in mappings if m.gt_id == "mRNA2")
        pred_b_match = [
            p for p in mRNA2_mapping.matched_predictions
            if p.predictor_name == "PredB"
        ]
        assert len(pred_b_match) == 0

        gt_df, pred_dfs = self._get_dfs(
            gt_gff,
            {"PredA": pred_a_gff, "PredB": pred_b_gff},
        )

        gt_arr, pred_arrs = build_paired_arrays(
            mapping=mRNA2_mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=simple_config,
        )

        # PredB has no match -> all background
        assert "PredB" in pred_arrs
        np.testing.assert_array_equal(
            pred_arrs["PredB"],
            np.full(len(gt_arr), 1, dtype=np.int32),
        )

    def test_utr_role_maps_paint_distinct_labels(
        self, utr_gt_gff, utr_pred_gff, utr_config,
    ):
        mappings = map_transcripts(
            gt_path=utr_gt_gff,
            pred_paths={"Pred": utr_pred_gff},
            label_config=utr_config,
            gt_feature_role_map=UTR_ROLE_MAP,
            pred_feature_role_maps={"Pred": UTR_ROLE_MAP_NO_CDS},
        )

        mapping = next(m for m in mappings if m.gt_id == "gt_tx1")
        gt_df, pred_dfs = self._get_dfs(utr_gt_gff, {"Pred": utr_pred_gff}, excl=[])
        gt_arr, pred_arrs = build_paired_arrays(
            mapping=mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=utr_config,
            gt_feature_role_map=UTR_ROLE_MAP,
            pred_feature_role_maps={"Pred": UTR_ROLE_MAP_NO_CDS},
        )

        np.testing.assert_array_equal(gt_arr[0:5], np.full(5, 4))
        np.testing.assert_array_equal(gt_arr[5:20], np.full(15, 0))
        np.testing.assert_array_equal(gt_arr[20:30], np.full(10, 5))

        pred_arr = pred_arrs["Pred"]
        np.testing.assert_array_equal(pred_arr[0:20], np.full(20, 4))
        np.testing.assert_array_equal(pred_arr[20:30], np.full(10, 5))

    def test_matched_prediction_overhang_is_captured(
        self, gt_over_gff, pred_over_gff, simple_config,
    ):
        """A matched prediction over-extending past the GT span must not be
        clipped: the window widens to the union of GT and pred spans so the
        terminal overhang is preserved (regression: GT-span-only window)."""
        mappings = map_transcripts(
            gt_path=gt_over_gff,
            pred_paths={"PredOver": pred_over_gff},
            exclude_features=["gene"],
        )
        mapping = next(m for m in mappings if m.gt_id == "gt_over")
        gt_df, pred_dfs = self._get_dfs(gt_over_gff, {"PredOver": pred_over_gff})

        gt_arr, pred_arrs = build_paired_arrays(
            mapping=mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=simple_config,
        )

        # Window = union of GT (1-100) and pred (1-150) -> length 150.
        assert len(gt_arr) == 150
        pred_arr = pred_arrs["PredOver"]
        assert len(pred_arr) == 150

        # GT CDS 10-90 -> indices 9..89; nothing past genomic 100.
        np.testing.assert_array_equal(gt_arr[9:90], np.full(81, 0))
        np.testing.assert_array_equal(gt_arr[100:150], np.full(50, 1))

        # Pred CDS 10-120 -> indices 9..119; the overhang at 100..119 (past the
        # GT end) is exactly the part the old GT-span-only window clipped.
        np.testing.assert_array_equal(pred_arr[9:120], np.full(111, 0))
        np.testing.assert_array_equal(pred_arr[120:150], np.full(30, 1))

    def test_minus_strand_arrays_are_biologically_oriented(
        self, gt_gff, pred_a_gff, simple_config,
    ):
        """Minus-strand arrays are in biological 5'→3' order.

        mRNA2: chr1, -, 200-400 (length 201).  GT CDS at 250-300 (genomic).
        Biological 5' = genomic 400 → index 0.
        After reversal the CDS lands at biological indices 100-150, not 50-100.
        """
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff},
            exclude_features=["gene"],
        )
        mRNA2_mapping = next(m for m in mappings if m.gt_id == "mRNA2")
        gt_df, pred_dfs = self._get_dfs(gt_gff, {"PredA": pred_a_gff})

        gt_arr, pred_arrs = build_paired_arrays(
            mapping=mRNA2_mapping,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            label_config=simple_config,
        )

        assert len(gt_arr) == 201
        # Biological layout: [bg*100][CDS*51][bg*50]
        np.testing.assert_array_equal(gt_arr[0:100], np.full(100, 1))
        np.testing.assert_array_equal(gt_arr[100:151], np.full(51, 0))
        np.testing.assert_array_equal(gt_arr[151:], np.full(50, 1))

        # predA_t2 CDS at 260-290 (genomic) → biological indices 110-140
        pred_arr = pred_arrs["PredA"]
        assert len(pred_arr) == 201
        np.testing.assert_array_equal(pred_arr[0:110], np.full(110, 1))
        np.testing.assert_array_equal(pred_arr[110:141], np.full(31, 0))
        np.testing.assert_array_equal(pred_arr[141:], np.full(60, 1))


# ------------------------------------------------------------------
# Tests: export_mapping_table
# ------------------------------------------------------------------


class TestExportMappingTable:
    """Tests for the debug TSV export."""

    def test_export_creates_file(self, gt_gff, pred_a_gff, tmp_path):
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff},

            exclude_features=["gene"],
        )

        output_file = tmp_path / "mapping_debug.tsv"
        result_path = export_mapping_table(mappings, output_file)

        assert result_path.exists()
        content = result_path.read_text()
        assert "seqid" in content
        assert "mRNA1" in content or "mRNA2" in content

    def test_unmatched_entries_in_export(
        self, gt_gff, pred_b_gff, tmp_path,
    ):
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredB": pred_b_gff},

            exclude_features=["gene"],
        )

        output_file = tmp_path / "mapping_debug.tsv"
        export_mapping_table(mappings, output_file)

        content = output_file.read_text()
        assert "__unmatched_pred__" in content
