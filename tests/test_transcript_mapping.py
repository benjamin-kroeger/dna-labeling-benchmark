"""Tests for the transcript mapping module.

Covers:
- Overlap fraction computation
- Strand-aware mapping of GT <-> predictions
- Unmatched GT transcripts (no prediction -> empty pred array)
- Unmatched predictions (no GT transcript -> GT from region features)
- Overlapping GT transcript detection (error)
- Array construction from mappings
- Debug TSV export
"""

import numpy as np
import pytest

from dna_segmentation_benchmark.io_utils import collect_gff
from dna_segmentation_benchmark.label_definition import LabelConfig
from dna_segmentation_benchmark.transcript_mapping import (
    PredictionMatch,
    TranscriptMapping,
    _compute_overlap_fraction,
    build_paired_arrays,
    export_mapping_table,
    map_transcripts,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def simple_label_config():
    """A minimal two-token label config."""
    return LabelConfig(
        labels={0: "CODING", 1: "BACKGROUND"},
        background_label=1,
        coding_label=0,
    )


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


# ------------------------------------------------------------------
# Unit tests: _compute_overlap_fraction
# ------------------------------------------------------------------


class TestComputeOverlapFraction:
    """Tests for the pure overlap fraction helper."""

    def test_identical_intervals(self):
        assert _compute_overlap_fraction(10, 50, 10, 50) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert _compute_overlap_fraction(10, 20, 30, 40) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # [10, 30] and [20, 40]: overlap = [20, 30] = 11 bases
        # shorter = 21
        frac = _compute_overlap_fraction(10, 30, 20, 40)
        assert frac == pytest.approx(11 / 21)

    def test_containment(self):
        # [10, 50] contains [20, 30]: overlap = 11, shorter = 11
        assert _compute_overlap_fraction(10, 50, 20, 30) == pytest.approx(1.0)

    def test_adjacent_no_overlap(self):
        # [10, 20] and [21, 30] do not overlap
        assert _compute_overlap_fraction(10, 20, 21, 30) == pytest.approx(0.0)

    def test_single_base_overlap(self):
        # [10, 20] and [20, 30]: overlap = 1, shorter = 11
        frac = _compute_overlap_fraction(10, 20, 20, 30)
        assert frac == pytest.approx(1 / 11)

    def test_single_base_interval(self):
        # [10, 10] and [10, 10]: overlap = 1, shorter = 1
        assert _compute_overlap_fraction(10, 10, 10, 10) == pytest.approx(1.0)


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
            min_overlap=0.2,
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
            min_overlap=0.2,
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
            min_overlap=0.2,
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
            min_overlap=0.2,
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
            min_overlap=0.2,
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
            min_overlap=0.2,
        )
        # Both overlapping GT transcripts should appear in the mappings
        gt_ids = {m.gt_id for m in mappings if not m.is_unmatched_prediction}
        assert len(gt_ids) >= 2

    def test_prediction_on_unknown_chromosome(
        self, gt_gff, pred_only_chr3_gff,
    ):
        """Predictions on seqids not in GT produce unmatched entries."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"Pred": pred_only_chr3_gff},
            min_overlap=0.2,
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
        self, gt_gff, pred_b_gff, simple_label_config,
    ):
        """An unmatched prediction's GT array reflects actual GT features."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredB": pred_b_gff},
            min_overlap=0.2,
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
            label_config=simple_label_config,
        )

        # predB_t1 is at 500-600, no GT features there -> all background
        np.testing.assert_array_equal(
            gt_arr, np.full(len(gt_arr), 1, dtype=np.int32),
        )

    def test_coding_regions_in_gt_array(
        self, gt_gff, pred_a_gff, simple_label_config,
    ):
        """A real GT mapping paints CDS regions as coding."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff},
            min_overlap=0.2,
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
            label_config=simple_label_config,
        )

        # mRNA1 spans 1-100 (100 bases), CDS at 10-30
        assert len(gt_arr) == 100
        np.testing.assert_array_equal(gt_arr[0:9], np.full(9, 1))
        np.testing.assert_array_equal(gt_arr[9:30], np.full(21, 0))
        np.testing.assert_array_equal(gt_arr[30:], np.full(70, 1))

    def test_prediction_array_content(
        self, gt_gff, pred_a_gff, simple_label_config,
    ):
        """Prediction arrays have coding regions at expected positions."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff},
            min_overlap=0.2,
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
            label_config=simple_label_config,
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
        self, gt_gff, pred_a_gff, pred_b_gff, simple_label_config,
    ):
        """A predictor with no match at a GT locus gets all-background."""
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff, "PredB": pred_b_gff},
            min_overlap=0.2,
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
            label_config=simple_label_config,
        )

        # PredB has no match -> all background
        assert "PredB" in pred_arrs
        np.testing.assert_array_equal(
            pred_arrs["PredB"],
            np.full(len(gt_arr), 1, dtype=np.int32),
        )


# ------------------------------------------------------------------
# Tests: export_mapping_table
# ------------------------------------------------------------------


class TestExportMappingTable:
    """Tests for the debug TSV export."""

    def test_export_creates_file(self, gt_gff, pred_a_gff, tmp_path):
        mappings = map_transcripts(
            gt_path=gt_gff,
            pred_paths={"PredA": pred_a_gff},
            min_overlap=0.2,
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
            min_overlap=0.2,
            exclude_features=["gene"],
        )

        output_file = tmp_path / "mapping_debug.tsv"
        export_mapping_table(mappings, output_file)

        content = output_file.read_text()
        assert "__unmatched_pred__" in content