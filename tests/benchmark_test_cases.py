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
    # Exon / intron chain + boundary shift test cases
    #
    # Intron chain uses set comparison of intron segment boundaries.
    # Exon chain uses the same set semantics on coding segments:
    #   exon_chain         — exact set match
    #   exon_chain_superset — pred ⊇ GT (all GT exons found, extras ok)
    #   exon_chain_subset   — pred ⊆ GT (all pred exons valid, may miss GT)
    # Boundary shifts are only measured when GT and pred have equal segment counts.
    # -----------------------------------------------------------------------

    # -- Case 1: identical 3-exon chains
    # GT/pred exons: (2,4),(7,9),(12,13)  introns: (5,6),(10,11)
    # All exon chain tiers: TP. Boundary shifts: none.
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
                "segment_count_delta": 0,
                "exon_chain":          {"tp": 1, "fn": 0, "fp": 0},
                "exon_chain_superset": {"tp": 1, "fn": 0, "fp": 0},
                "exon_chain_subset":   {"tp": 1, "fn": 0, "fp": 0},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },
        },
        id="sc_exact_match",
    ),
    # -- Case 2: same exon count, all boundaries shifted by 1 bp
    # GT  exons: (2,4),(7,9),(12,13)  pred exons: (1,4),(7,8),(12,14)
    # GT introns: (5,6),(10,11); pred introns: (5,6),(9,11) → mismatch → intron_chain tp=0
    # All exon chain tiers FP (shifted boundary ≠ exact set element; neither subset nor superset)
    # boundary_shift_count=3 (starts/ends of 3 segments differ), total=3
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
                "segment_count_delta": 0,
                "exon_chain":          {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_superset": {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_subset":   {"tp": 0, "fn": 1, "fp": 1},
                "boundary_shift_count": 3,
                "boundary_shift_total": 3,
            },
        },
        id="sc_boundary_shift",
    ),
    # -- Case 3: pred skips the middle GT exon
    # GT  exons: (2,4),(7,9),(12,13)  pred exons: (2,4),(12,13)
    # GT introns: (5,6),(10,11); pred intron: (5,11) → intron_chain mismatch
    # exon_chain_subset TP (pred ⊆ GT), exon_chain_superset FP (GT not in pred)
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
                "segment_count_delta": -1,
                "exon_chain":          {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_superset": {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_subset":   {"tp": 1, "fn": 0, "fp": 0},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },
        },
        id="sc_missing_segments",
    ),
    # -- Case 4: pred inserts a middle exon not in GT
    # GT  exons: (2,4),(12,13)  pred exons: (2,4),(7,9),(12,13)
    # GT intron: (5,11) one intron; pred introns: (5,6),(10,11) two introns → mismatch
    # exon_chain_superset TP (GT ⊆ pred), exon_chain_subset FP (pred has extra)
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 0, "fp": 1, "fn": 1},
                "segment_count_delta": 1,
                "exon_chain":          {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_superset": {"tp": 1, "fn": 0, "fp": 0},
                "exon_chain_subset":   {"tp": 0, "fn": 1, "fp": 1},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },
        },
        id="sc_extra_segments",
    ),
    # -- Case 5: completely rearranged predictions
    # GT exons: (1,3),(6,8),(11,12)  pred exons: (4,7),(10,13)
    # No exon boundary pairs match — all chain metrics FP.
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
                "segment_count_delta": -1,
                "exon_chain":          {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_superset": {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_subset":   {"tp": 0, "fn": 1, "fp": 1},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },
        },
        id="sc_structurally_different",
    ),
    # -- Case 6: GT has 2 exons, pred is all noncoding
    # Pred empty → no exon chain TP; fp=1 (consistent with intron chain semantics).
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 0, "fp": 1, "fn": 1},
                "segment_count_delta": -2,
                "exon_chain":          {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_superset": {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_subset":   {"tp": 0, "fn": 1, "fp": 1},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },
        },
        id="sc_missed",
    ),
    # -- Case 7: all noncoding GT, pred has a spurious exon
    # GT has no coding segments → exon chain metrics absent from output.
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
    # -- Case 8: 6 GT exons vs 5 pred exons with boundary errors
    # GT exons: (1,2),(5,6),(9,10),(13,14),(17,18),(21,22)
    # Pred exons: (0,2),(9,10),(13,14),(17,18),(21,22)
    # pred_set is not a subset of gt_set ((0,2) not in gt) → all FP.
    pytest.param(
        np.array([
            [8, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 8, 8],
            [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 0, "fp": 1, "fn": 1},
                "segment_count_delta": -1,
                "exon_chain":          {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_superset": {"tp": 0, "fn": 1, "fp": 1},
                "exon_chain_subset":   {"tp": 0, "fn": 1, "fp": 1},
                "boundary_shift_count": 0,
                "boundary_shift_total": 0,
            },
        },
        id="sc_six_exon_mixed_errors",
    ),
    # -- Case 9: single exon, identical on both sides
    # No introns → intron_chain vacuous. Exon chain: exact TP.
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {
                "intron_chain": {"tp": 0, "fp": 0, "fn": 0},
                "segment_count_delta": 0,
                "exon_chain":          {"tp": 1, "fn": 0, "fp": 0},
                "exon_chain_superset": {"tp": 1, "fn": 0, "fp": 0},
                "exon_chain_subset":   {"tp": 1, "fn": 0, "fp": 0},
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
