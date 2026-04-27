import numpy as np
import pytest

from dna_segmentation_benchmark.eval.evaluate_predictors import EvalMetrics
from dna_segmentation_benchmark.label_definition import LabelConfig, BEND_LABEL_CONFIG

# ------------------------------------------------------------------
# Convenience token constants
# ------------------------------------------------------------------
EXON, DONOR, INTRON, ACCEPTOR, NONCODING = 0, 1, 2, 3, 8

# A second label set to prove label-agnosticism
CUSTOM_CONFIG = LabelConfig(
    background_label=-1,
    exon_label=5,
)

SINGLE_SEQUENCE_TEST_CASES = [
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
                [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INDEL": {
                "5_prime_extensions": [np.array([0, 1, 2])],
                "3_prime_extensions": [np.array([17, 18])],
                "whole_insertions": [np.array([8, 9, 10, 11])],
                "5_prime_deletions": [np.array([12])],
                "3_prime_deletions": [np.array([5, 6, 7])],
                "whole_deletions": [np.array([19, 20])],
                "split": [],
                "joined": [],
            },
            "REGION_DISCOVERY": {
                "neighborhood_hit": {"tp": 2, "fn": 1, "fp": 1},
                "internal_hit": {"tp": 0, "fn": 3},
                "full_coverage_hit": {"tp": 0, "fn": 3},
                "perfect_boundary_hit": {"tp": 0, "fn": 3, "fp": 3},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": 0,
                "last_sec_correct_5_prime_boundary": 0,
                "iou_scores": [0.25,0.57]
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"tn": 4, "fp": 9, "fn": 6, "tp": 6},
            },
        },
        id="exon_all_insertions_deletions",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
                [8, 8, 8, 0, 2, 0, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2, 0, 2, 2, 0, 0, 8, 8, 0, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INDEL": {
                "5_prime_extensions": [],
                "3_prime_extensions": [],
                "whole_insertions": [np.array([9]), np.array([23])],
                "5_prime_deletions": [],
                "3_prime_deletions": [],
                "whole_deletions": [],
                "split": [np.array([4]), np.array([6]), np.array([13]), np.array(15)],
                "joined": [],
            },
            "REGION_DISCOVERY": {
                "neighborhood_hit": {"tp": 3, "fn": 0, "fp": 6},
                "internal_hit": {"tp": 3, "fn": 0, "fp": 6},
                "full_coverage_hit": {"tp": 1, "fn": 2, "fp": 6},
                "perfect_boundary_hit": {"tp": 1, "fn": 2, "fp": 8},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": 1,
                "last_sec_correct_5_prime_boundary": 1,
                "iou_scores": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1]
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"tn": 11, "fp": 2, "fn": 4, "tp": 8},
            },

        },
        id="uncertain_predictions",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
                [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {

            "INDEL": {
                "5_prime_extensions": [],
                "3_prime_extensions": [],
                "whole_insertions": [],
                "5_prime_deletions": [],
                "3_prime_deletions": [],
                "whole_deletions": [np.array([3, 4, 5, 6, 7]), np.array([12, 13, 14, 15, 16]), np.array([19, 20])],
                "split": [],
                "joined": [],
            },
            "REGION_DISCOVERY": {
                "neighborhood_hit": {"tp": 0, "fn": 3, "fp": 0},
                "internal_hit": {"tp": 0, "fn": 3},
                "full_coverage_hit": {"tp": 0, "fn": 3},
                "perfect_boundary_hit": {"tp": 0, "fn": 3, "fp": 0},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": 0,
                "last_sec_correct_5_prime_boundary": 0,
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"tn": 13, "fp": 0, "fn": 12, "tp": 0},
            },

        },
        id="empty_pred",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {

            "INDEL": {
                "5_prime_extensions": [],
                "3_prime_extensions": [],
                "whole_insertions": [np.array([3, 4, 5, 6, 7]), np.array([12, 13, 14, 15, 16]), np.array([19, 20])],
                "5_prime_deletions": [],
                "3_prime_deletions": [],
                "whole_deletions": [],
                "split": [],
                "joined": [],
            },
            "REGION_DISCOVERY": {
                "neighborhood_hit": {"tp": 0, "fn": 0, "fp": 3},
                "internal_hit": {"tp": 0, "fn": 0},
                "full_coverage_hit": {"tp": 0, "fn": 0},
                "perfect_boundary_hit": {"tp": 0, "fn": 0, "fp": 3},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": 0,
                "last_sec_correct_5_prime_boundary": 0,
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"tn": 13, "fp": 12, "fn": 0, "tp": 0},
            },

        },
        id="empty_gt",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8, 8],
                [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {

            "REGION_DISCOVERY": {
                "neighborhood_hit": {"tp": 4, "fn": 0, "fp": 0},
                "internal_hit": {"tp": 1, "fn": 3},
                "full_coverage_hit": {"tp": 4, "fn": 0},
                "perfect_boundary_hit": {"tp": 1, "fn": 3, "fp": 3},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": 1,
                "last_sec_correct_5_prime_boundary": 1,
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"tn": 11, "fp": 6, "fn": 0, "tp": 12},
            },

        },
        id="in_depth_section_test",
    ),
    pytest.param(
        np.array(
            [
                [0, 0, 0, 0, 2, 2, 2, 0, 0, 0],
                [8, 8, 8, 0, 0, 0, 0, 0, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,

        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {

            "INDEL": {
                "5_prime_extensions": [],
                "3_prime_extensions": [],
                "whole_insertions": [],
                "5_prime_deletions": [np.array([0, 1, 2])],
                "3_prime_deletions": [np.array([8, 9])],
                "whole_deletions": [],
                "split": [],
                "joined": [np.array([4, 5, 6])],
            },
            "REGION_DISCOVERY": {
                "neighborhood_hit": {"tp": 1, "fn": 1, "fp": 0},
                "internal_hit": {"tp": 0, "fn": 2},
                "full_coverage_hit": {"tp": 0, "fn": 2},
                "perfect_boundary_hit": {"tp": 0, "fn": 2, "fp": 1},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": 0,
                "last_sec_correct_5_prime_boundary": 0,
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"tn": 0, "fp": 3, "fn": 5, "tp": 2},
            },

        },
        id="exon_joined_with_deletions",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
                [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {

            "REGION_DISCOVERY": {
                "neighborhood_hit": {"tp": 3, "fn": 0, "fp": 0},
                "internal_hit": {"tp": 3, "fn": 0},
                "full_coverage_hit": {"tp": 3, "fn": 0},
                "perfect_boundary_hit": {"tp": 3, "fn": 0, "fp": 0},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": 1,
                "last_sec_correct_5_prime_boundary": 1,
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"tn": 13, "fp": 0, "fn": 0, "tp": 12},
            },

        },
        id="exon_fully_correct",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2],
                [8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {

            "REGION_DISCOVERY": {
                "neighborhood_hit": {"tp": 1, "fn": 0, "fp": 0},
                "internal_hit": {"tp": 1, "fn": 0},
                "full_coverage_hit": {"tp": 1, "fn": 0},
                "perfect_boundary_hit": {"tp": 1, "fn": 0, "fp": 0},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": 1,
                "last_sec_correct_5_prime_boundary": 1,
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"tn": 7, "fp": 0, "fn": 0, "tp": 5},
            },

        },
        id="exon_fully_correct_2",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 2, 2, 2, 2, 0, 0],
                [8, 8, 8, 0, 2, 2, 2, 8, 8, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {

            "INDEL": {
                "5_prime_extensions": [],
                "3_prime_extensions": [],
                "whole_insertions": [],
                "5_prime_deletions": [],
                "3_prime_deletions": [np.array([4])],
                "whole_deletions": [np.array([9, 10])],
                "split": [],
                "joined": [],
            },
            "REGION_DISCOVERY": {
                "neighborhood_hit": {"tp": 1, "fn": 1, "fp": 0},
                "internal_hit": {"tp": 1, "fn": 1},
                "full_coverage_hit": {"tp": 0, "fn": 2},
                "perfect_boundary_hit": {"tp": 0, "fn": 2, "fp": 1},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": 0,
                "last_sec_correct_5_prime_boundary": 0,
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"tn": 7, "fp": 0, "fn": 3, "tp": 1},
            },

        },
        id="exon_test2",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EvalMetrics.FRAMESHIFT],
        {

            "FRAMESHIFT": {
                "gt_frames": np.array(
                    [np.inf] * 3 + [0, 0] + [np.inf] * 8 + [0, 0, 0, 0] + [np.inf] * 4
                )
            }

        },
        id="Frameshift_test",
    ),
    # ---- Different label set (label-agnosticism) ----------------------
    pytest.param(
        np.array(
            [
                [-1, -1, -1, 5, 5, 5, 5, 5, -1, -1, -1, -1, 5, 5, 5, 5, 5, -1, -1, 5, 5],
                [5, 5, 5, 5, 5, -1, -1, -1, 5, 5, 5, 5, -1, 5, 5, 5, 5, 5, 5, -1, -1],
            ]
        ),
        CUSTOM_CONFIG,
        [EvalMetrics.INDEL],
        {
            "INDEL": {
                "5_prime_extensions": [np.array([0, 1, 2])],
                "3_prime_extensions": [np.array([17, 18])],
                "whole_insertions": [np.array([8, 9, 10, 11])],
                "5_prime_deletions": [np.array([12])],
                "3_prime_deletions": [np.array([5, 6, 7])],
                "whole_deletions": [np.array([19, 20])],
                "split": [],
                "joined": [],
            }

        },
        id="Different_label_test",
    ),
    pytest.param(
        np.array(
            [
                [8, 0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 8, 8],
                [8, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            ]
        ),
        BEND_LABEL_CONFIG,

        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {

            "REGION_DISCOVERY": {
                "neighborhood_hit": {"tp": [1, 1], "fn": [0, 0], "fp": [0, 0]},
                "internal_hit": {"tp": [1, 1], "fn": [0, 0]},
                "full_coverage_hit": {"tp": [1, 1], "fn": [0, 0]},
                "perfect_boundary_hit": {"tp": [1, 1], "fn": [0, 0], "fp": [0, 0]},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": [1, 1],
                "last_sec_correct_5_prime_boundary": [1, 1],
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"tn": [3, 4], "fp": [0, 0], "fn": [0, 0], "tp": [3, 2]},
            },

        },
        id="mask_test",
    ),
]

# ------------------------------------------------------------------
# STRUCTURAL_COHERENCE test cases
# ------------------------------------------------------------------

STRUCTURAL_COHERENCE_TEST_CASES = [
    # -----------------------------------------------------------------------
    # Transcript match classification + tier P/R test cases
    #
    # Each case verifies four structural-coherence sub-metrics for a single
    # transcript pair:
    #   intron_chain   — binary intron-set equality (tp/fp/fn)
    #   transcript_exact    — TP only when chains are identical
    #   pred_is_superset    — TP for EXACT | BOUNDARY_SHIFT | EXTRA_SEGMENTS
    #   pred_is_subset      — TP for EXACT | BOUNDARY_SHIFT | MISSING_SEGMENTS
    #
    # NOTE on superset/subset having identical P/R in production data:
    #   precision_superset == precision_subset iff
    #   count(EXTRA_SEGMENTS) == count(MISSING_SEGMENTS) across all sequences.
    #   This is a dataset property, not a bug.  The single-pair tests below
    #   confirm that exact=1.0 for a perfect match and 0.0 otherwise, which
    #   is correct — the low "exact" value in production simply reflects that
    #   very few predicted transcripts have identical splice-site boundaries.
    # -----------------------------------------------------------------------

    # -- Case 1: EXACT — identical 3-exon chains
    # GT/pred exons: [2,5), [7,10), [12,14)  introns: [5,7), [10,12)
    # intron chain link: (7, 10) — same on both sides → tp=1
    # All tiers: TP
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {

            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 1, "fp": 0, "fn": 0},
                "transcript_match_class": "exact",
                "segment_count_delta": 0,
                "transcript_exact": {"tp": 1, "fn": 0, "fp": 0},
                "pred_is_superset": {"tp": 1, "fn": 0, "fp": 0},
                "pred_is_subset": {"tp": 1, "fn": 0, "fp": 0},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },

        },
        id="sc_exact_match",
    ),
    # -- Case 2: BOUNDARY_SHIFT — same exon count, boundaries off by 1 bp each
    # GT  exons: [2,5), [7,10), [12,14)   pred exons: [1,5), [7,9), [12,15)
    # GT  intron chain link: (7,10)        pred: (7,9) → mismatch → tp=0
    # segment_count_delta = 0 (n=3 both)
    # All boundaries counted: seg0.start(2≠1), seg1.end(10≠9), seg2.end(14≠15) → count=3, total=3
    # BOUNDARY_SHIFT is FP for superset AND subset: a shifted boundary means
    # that GT segment is absent from pred and pred segment is absent from GT,
    # so neither strict containment holds.
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
            [8, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {

            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 0, "fp": 1, "fn": 1},
                "transcript_match_class": "boundary_shift",
                "segment_count_delta": 0,
                "transcript_exact": {"tp": 0, "fn": 1, "fp": 1},
                "pred_is_superset": {"tp": 0, "fn": 1, "fp": 1},
                "pred_is_subset": {"tp": 0, "fn": 1, "fp": 1},
                "boundary_shift_count": 3,
                "boundary_shift_total": 3,
            },

        },
        id="sc_boundary_shift",
    ),
    # -- Case 3: MISSING_SEGMENTS — pred skips the middle GT exon
    # GT  exons: [2,5), [7,10), [12,14)   pred exons: [2,5), [12,14)
    # GT has intron chain link (7,10); pred has single intron [5,12) → no chain
    # LCS=2=n_pred < n_gt=3 → MISSING_SEGMENTS
    # superset: FP (GT not fully in pred)   subset: TP (all pred segs in GT)
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {

            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 0, "fp": 1, "fn": 1},
                "transcript_match_class": "missing_segments",
                "segment_count_delta": -1,
                "transcript_exact": {"tp": 0, "fn": 1, "fp": 1},
                "pred_is_superset": {"tp": 0, "fn": 1, "fp": 1},
                "pred_is_subset": {"tp": 1, "fn": 0, "fp": 0},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },

        },
        id="sc_missing_segments",
    ),
    # -- Case 4: EXTRA_SEGMENTS — pred inserts a middle exon not in GT
    # GT  exons: [2,5), [12,14)   pred exons: [2,5), [7,10), [12,14)
    # GT has single intron [5,12) → no intron chain → tp=fp=fn=0 (vacuous)
    # LCS=2=n_gt < n_pred=3 → EXTRA_SEGMENTS
    # superset: TP (GT fully in pred)   subset: FP (pred has extra)
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {

            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 0, "fp": 0, "fn": 0},  # GT has single intron → no chain
                "transcript_match_class": "extra_segments",
                "segment_count_delta": 1,
                "transcript_exact": {"tp": 0, "fn": 1, "fp": 1},
                "pred_is_superset": {"tp": 1, "fn": 0, "fp": 0},
                "pred_is_subset": {"tp": 0, "fn": 1, "fp": 1},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },

        },
        id="sc_extra_segments",
    ),
    # -- Structurally different: completely rearranged predictions
    # GT exons: (1,3), (6,8), (11,12)  → bounds: [(1,3),(6,8),(11,12)]
    # Pred exons: (4,7), (10,13)        → bounds: [(4,7),(10,13)]
    # No boundary pairs match → LCS=0, not subset/superset
    pytest.param(
        np.array([
            [8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
            [8, 8, 8, 8, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {

            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 0, "fp": 1, "fn": 1},
                "transcript_match_class": "structurally_different",
                "segment_count_delta": -1,
                "transcript_exact": {"tp": 0, "fn": 1, "fp": 1},
                "pred_is_superset": {"tp": 0, "fn": 1, "fp": 1},
                "pred_is_subset": {"tp": 0, "fn": 1, "fp": 1},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },

        },
        id="sc_structurally_different",
    ),
    # -- Missed: GT has 2 exons, pred is all noncoding
    # Single intron in GT → no intron chain → tp=fp=fn=0 (vacuous)
    # fp=0 on all tiers because has_pred=False (MISSED class)
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {

            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 0, "fp": 0, "fn": 0},  # single intron, no chain
                "transcript_match_class": "missed",
                "segment_count_delta": -2,
                "transcript_exact": {"tp": 0, "fn": 1, "fp": 0},
                "pred_is_superset": {"tp": 0, "fn": 1, "fp": 0},
                "pred_is_subset": {"tp": 0, "fn": 1, "fp": 0},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },

        },
        id="sc_missed",
    ),
    # -- No GT segments: all noncoding GT, pred has a spurious exon
    # match_cls=None → transcript_match_class and tiers are absent from output
    pytest.param(
        np.array([
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 0, 0, 0, 8, 8, 8, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {

            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 0, "fp": 0, "fn": 0},
            },

        },
        id="sc_no_gt_segments",
    ),
    # -- Mixed errors: 6 GT exons, pred missing 2nd, first/last boundaries shifted
    # GT exons: (1,2),(5,6),(9,10),(13,14),(17,18),(21,22) → 5 gaps
    # Pred exons: (0,2),(9,10),(13,14),(17,18),(21,23)     → 4 gaps
    # Boundary LCS: (9,10),(13,14),(17,18) = 3 ≠ n_pred(5) ≠ n_gt(6) → structurally_different
    # Gap LCS: (10,13),(14,17),(18,21) = 3 of max(5,4)=5 → ratio 0.6
    pytest.param(
        np.array([
            [8, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 8, 8],
            [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,

        [EvalMetrics.STRUCTURAL_COHERENCE],
        {

            "STRUCTURAL_COHERENCE": {
                # GT intron chain: (5,7),(9,11),(13,15),(17,19) — pred misses (5,7)
                "intron_chain": {"tp": 0, "fp": 1, "fn": 1},
                "transcript_match_class": "structurally_different",
                "segment_count_delta": -1,
                "transcript_exact": {"tp": 0, "fn": 1, "fp": 1},
                "pred_is_superset": {"tp": 0, "fn": 1, "fp": 1},
                "pred_is_subset": {"tp": 0, "fn": 1, "fp": 1},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },

        },
        id="sc_six_exon_mixed_errors",
    ),
    # -- Single segment (no gaps): both have exactly 1 exon, identical
    # Empty gap chains → match=True, lcs_ratio=1.0
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,

        [EvalMetrics.STRUCTURAL_COHERENCE],
        {

            "STRUCTURAL_COHERENCE": {
                # Single exon on both sides → no introns → no chain (vacuous tp=fp=fn=0)
                "intron_chain": {"tp": 0, "fp": 0, "fn": 0},
                "transcript_match_class": "exact",
                "segment_count_delta": 0,
                "transcript_exact": {"tp": 1, "fn": 0, "fp": 0},
                "pred_is_superset": {"tp": 1, "fn": 0, "fp": 0},
                "pred_is_subset": {"tp": 1, "fn": 0, "fp": 0},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },

        },
        id="sc_single_segment",
    )
]

# ------------------------------------------------------------------
# DIAGNOSTIC_DEPTH test cases
# ------------------------------------------------------------------

DIAGNOSTIC_DEPTH_TEST_CASES = [
    # -- No structural summary errors: identical 2-exon predictions
    # Length distributions match and no unmatched segments contribute to position bias.
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,

        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [3, 3],
                "pred_segment_lengths": [3, 3],
                "length_emd": 0.0,
                "position_bias_histogram": [0] * 100,
            }
        },
        id="dd_no_errors",
    ),
    # -- Missing middle exon: middle GT coding segment is absent from pred
    # GT exons: (1,2), (5,6), (9,10) — pred exons: (1,2), (9,10)
    # The unmatched middle GT segment fills the middle position-bias bins.
    pytest.param(
        np.array([
            [8, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 8],
            [8, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,

        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {

            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [2, 2, 2],
                "pred_segment_lengths": [2, 2],
                "length_emd": 0.0,
                "position_bias_histogram": [0] * 40 + [1] * 21 + [0] * 39,
            },

        },
        id="dd_exon_skip",
    ),
    # -- Predicted split: pred splits one GT exon into two shorter segments
    # GT exon: (1,9) — pred exons: (1,3) and (6,9)
    # One pred segment is unmatched and contributes to early position-bias bins.
    pytest.param(
        np.array([
            [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
            [8, 0, 0, 0, 2, 2, 0, 0, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,

        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {

            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [9],
                "pred_segment_lengths": [3, 4],
                "length_emd": 5.5,
                "position_bias_histogram": [1] * 34 + [0] * 66,
            },

        },
        id="dd_novel_insertion",
    ),
    # -- Uniform boundary shift: 4 exons all shifted right by 1 position
    # GT exons: (1,3),(6,8),(11,13),(16,18)
    # Pred exons: (2,4),(7,9),(12,14),(17,19)
    # Segment lengths are unchanged, and all pred segments still match a GT segment.
    pytest.param(
        np.array([
            [8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,

        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {

            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [3, 3, 3, 3],
                "pred_segment_lengths": [3, 3, 3, 3],
                "length_emd": 0.0,
                "position_bias_histogram": [0] * 100,
            },

        },
        id="dd_cascade_shift",
    ),
    # -- Balanced extensions: both predicted exons are one base longer
    # GT exons: (1,3),(7,9) — pred exons: (1,4),(6,9)
    # Both segments still match, but the predicted length distribution shifts by +1.
    pytest.param(
        np.array([
            [8, 0, 0, 0, 2, 2, 2, 0, 0, 0, 8],
            [8, 0, 0, 0, 0, 2, 0, 0, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,

        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {

            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [3, 3],
                "pred_segment_lengths": [4, 4],
                "length_emd": 1.0,
                "position_bias_histogram": [0] * 100,
            },

        },
        id="dd_compensating_errors",
    ),
]

MULTI_SEQUENCE_TEST_CASES = [
    pytest.param(
        [np.array([8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8])],
        [np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])],
        BEND_LABEL_CONFIG,

        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {

            "INDEL": {
                "5_prime_extensions": [],
                "3_prime_extensions": [],
                "whole_insertions": [],
                "5_prime_deletions": [],
                "3_prime_deletions": [],
                "whole_deletions": [
                    np.array([3, 4, 5, 6, 7]),
                    np.array([12, 13, 14, 15, 16]),
                    np.array([19, 20]),
                ],
                "split": [],
                "joined": [],
            },
            "REGION_DISCOVERY": {
                "neighborhood_hit": {"precision": 0, "recall": 0.0},
                "internal_hit": {"precision": 0.0, "recall": 0.0},
                "full_coverage_hit": {"precision": 0.0, "recall": 0.0},
                "perfect_boundary_hit": {"precision": 0, "recall": 0.0},
            },
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": [0],
                "last_sec_correct_5_prime_boundary": [0],
                "iou_stats": {"mean": 0.0, "mae": 0.0, "rmse": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0},
            },
            "NUCLEOTIDE_CLASSIFICATION": {
                "nucleotide": {"precision": 0, "recall": 0.0, "f1": 0.0},
            },

        },
        id="no_nuc_positives",
    ),
]
