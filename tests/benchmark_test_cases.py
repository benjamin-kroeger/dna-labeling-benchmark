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
    labels={-1: "NONCODING", 5: "CDS", 3: "PROMOTER"},
    background_label=-1,
    coding_label=5,
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
        [EXON],
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "EXON": {
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
                },
                "NUCLEOTIDE_CLASSIFICATION": {
                    "nucleotide": {"tn": 4, "fp": 9, "fn": 6, "tp": 6},
                },
            }
        },
        id="exon_all_insertions_deletions",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8,   0, 0, 0, 0, 0,   2, 2, 2, 2,    0, 0, 0, 0, 0,   2, 2,  0, 0,    8, 8, 8, 8],
                [8, 8, 8,   0, 2, 0, 2, 0,   2, 0, 2, 2,    0, 2, 0, 2, 0,   2, 2,  0, 0,    8, 8, 0, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "EXON": {
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
                    "internal_hit": {"tp": 3, "fn": 0, "fp":6},
                    "full_coverage_hit": {"tp": 1, "fn": 2, "fp":6},
                    "perfect_boundary_hit": {"tp": 1, "fn": 2, "fp": 8},
                },
                "BOUNDARY_EXACTNESS": {
                    "first_sec_correct_3_prime_boundary": 1,
                    "last_sec_correct_5_prime_boundary": 1,
                },
                "NUCLEOTIDE_CLASSIFICATION": {
                    "nucleotide": {"tn": 11, "fp": 2, "fn": 4, "tp": 8},
                },
            }
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
        [EXON],
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "EXON": {
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
            }
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
        [EXON],
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "EXON": {
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
            }
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
        [EXON],
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "EXON": {
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
            }
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
        [EXON],
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "EXON": {
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
            }
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
        [EXON],
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "EXON": {
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
            }
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
        [EXON],
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "EXON": {
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
            }
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
        [INTRON, EXON],
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INTRON": {
                "INDEL": {
                    "5_prime_extensions": [np.array([4])],
                    "3_prime_extensions": [],
                    "whole_insertions": [],
                    "5_prime_deletions": [],
                    "3_prime_deletions": [np.array([7, 8])],
                    "whole_deletions": [],
                    "split": [],
                    "joined": [],
                },
                "REGION_DISCOVERY": {
                    "neighborhood_hit": {"tp": 1, "fn": 0, "fp": 0},
                    "internal_hit": {"tp": 0, "fn": 1},
                    "full_coverage_hit": {"tp": 0, "fn": 1},
                    "perfect_boundary_hit": {"tp": 0, "fn": 1, "fp": 1},
                },
                "BOUNDARY_EXACTNESS": {
                    "first_sec_correct_3_prime_boundary": 0,
                    "last_sec_correct_5_prime_boundary": 0,
                },
                "NUCLEOTIDE_CLASSIFICATION": {
                    "nucleotide": {"tn": 6, "fp": 1, "fn": 2, "tp": 2},
                },
            },
            "EXON": {
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
        },
        id="exon_intron_combination_test",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 8, 8],
                [8, 8, 8, 0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 0, 0, 0, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EXON, DONOR, ACCEPTOR],
        [EvalMetrics.INDEL],
        {
            "EXON": {
                "INDEL": {
                    "5_prime_extensions": [np.array([13])],
                    "3_prime_extensions": [],
                    "whole_insertions": [],
                    "5_prime_deletions": [],
                    "3_prime_deletions": [],
                    "whole_deletions": [],
                    "split": [],
                    "joined": [],
                },
            },
            "DONOR": {
                "INDEL": {
                    "5_prime_extensions": [],
                    "3_prime_extensions": [],
                    "whole_insertions": [],
                    "5_prime_deletions": [],
                    "3_prime_deletions": [np.array([7])],
                    "whole_deletions": [],
                    "split": [],
                    "joined": [],
                },
            },
            "ACCEPTOR": {
                "INDEL": {
                    "5_prime_extensions": [],
                    "3_prime_extensions": [],
                    "whole_insertions": [],
                    "5_prime_deletions": [],
                    "3_prime_deletions": [np.array([13])],
                    "whole_deletions": [],
                    "split": [],
                    "joined": [],
                },
            },
        },
        id="splice_sites_detection",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 0, 2, 2, 2, 2, 0, 0, 8, 8],
                [8, 8, 8, 0, 0, 0, 2, 2, 2, 2, 0, 0, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [INTRON],
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INTRON": {
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
                    "nucleotide": {"tn": 10, "fp": 0, "fn": 0, "tp": 4},
                },
            }
        },
        id="Intron_section_test",
    ),
    pytest.param(
        np.array(
            [
                [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0],
                [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8],
            ]
        ),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.FRAMESHIFT],
        {
            "EXON": {
                "FRAMESHIFT": {
                    "gt_frames": np.array(
                        [np.inf] * 3 + [0, 0] + [np.inf] * 8 + [0, 0, 0, 0] + [np.inf] * 4
                    )
                }
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
        [5],  # CDS token
        [EvalMetrics.INDEL],
        {
            "CDS": {
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
        [EXON],
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "EXON": {
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
            }
        },
        id="mask_test",
    ),
]

# ------------------------------------------------------------------
# STRUCTURAL_COHERENCE test cases
# ------------------------------------------------------------------

STRUCTURAL_COHERENCE_TEST_CASES = [
    # -- Exact match: identical 3-exon segment chains
    # GT/pred exons: (2,4), (7,9), (12,13) → gaps: [(4,7), (9,12)]
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "EXON": {
                "STRUCTURAL_COHERENCE": {
                    "gap_chain_match": True,
                    "gap_chain_lcs_ratio": 1.0,
                    "gap_count_match": True,
                    "gap_count_gt": 2,
                    "gap_count_pred": 2,
                    "segment_count_gt": 3,
                    "segment_count_pred": 3,
                    "segment_count_delta": 0,
                    "transcript_match_class": "exact",
                },
            }
        },
        id="sc_exact_match",
    ),
    # -- Boundary shift: same 3-exon count, shifted boundaries
    # GT exons: (2,4), (7,9), (12,13)  → gaps: [(4,7), (9,12)]
    # Pred exons: (1,4), (7,8), (12,14) → gaps: [(4,7), (8,12)]
    # Gap LCS: only (4,7) matches → 1/2 = 0.5
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
            [8, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "EXON": {
                "STRUCTURAL_COHERENCE": {
                    "gap_chain_match": False,
                    "gap_chain_lcs_ratio": 0.5,
                    "gap_count_match": True,
                    "gap_count_gt": 2,
                    "gap_count_pred": 2,
                    "segment_count_gt": 3,
                    "segment_count_pred": 3,
                    "segment_count_delta": 0,
                    "transcript_match_class": "boundary_shift",
                },
            }
        },
        id="sc_boundary_shift",
    ),
    # -- Missing segments: pred has 2 of GT's 3 exons (middle exon absent)
    # GT exons: (2,4), (7,9), (12,13)  → bounds: [(2,4),(7,9),(12,13)]
    # Pred exons: (2,4), (12,13)        → bounds: [(2,4),(12,13)]
    # Boundary LCS = 2 = n_pred < n_gt → MISSING_SEGMENTS
    # Gap chain GT: [(4,7),(9,12)], pred: [(4,12)] → LCS=0, ratio=0.0
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "EXON": {
                "STRUCTURAL_COHERENCE": {
                    "gap_chain_match": False,
                    "gap_chain_lcs_ratio": 0.0,
                    "gap_count_match": False,
                    "gap_count_gt": 2,
                    "gap_count_pred": 1,
                    "segment_count_gt": 3,
                    "segment_count_pred": 2,
                    "segment_count_delta": -1,
                    "transcript_match_class": "missing_segments",
                },
            }
        },
        id="sc_missing_segments",
    ),
    # -- Extra segments: pred has 3 exons, GT has 2 (pred inserts middle exon)
    # GT exons: (2,4), (12,13)           → bounds: [(2,4),(12,13)]
    # Pred exons: (2,4), (7,9), (12,13)  → bounds: [(2,4),(7,9),(12,13)]
    # Boundary LCS = 2 = n_gt < n_pred → EXTRA_SEGMENTS
    # Gap chain GT: [(4,12)], pred: [(4,7),(9,12)] → LCS=0, ratio=0.0
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "EXON": {
                "STRUCTURAL_COHERENCE": {
                    "gap_chain_match": False,
                    "gap_chain_lcs_ratio": 0.0,
                    "gap_count_match": False,
                    "gap_count_gt": 1,
                    "gap_count_pred": 2,
                    "segment_count_gt": 2,
                    "segment_count_pred": 3,
                    "segment_count_delta": 1,
                    "transcript_match_class": "extra_segments",
                },
            }
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
        [EXON],
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "EXON": {
                "STRUCTURAL_COHERENCE": {
                    "gap_chain_match": False,
                    "gap_chain_lcs_ratio": 0.0,
                    "gap_count_match": False,
                    "gap_count_gt": 2,
                    "gap_count_pred": 1,
                    "segment_count_gt": 3,
                    "segment_count_pred": 2,
                    "segment_count_delta": -1,
                    "transcript_match_class": "structurally_different",
                },
            }
        },
        id="sc_structurally_different",
    ),
    # -- Missed: GT has 2 exons, pred is all noncoding
    # GT exons: (2,4), (7,9) → 1 gap: [(4,7)]
    # Pred exons: none        → 0 gaps
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8],
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "EXON": {
                "STRUCTURAL_COHERENCE": {
                    "gap_chain_match": False,
                    "gap_chain_lcs_ratio": 0.0,
                    "gap_count_match": False,
                    "gap_count_gt": 1,
                    "gap_count_pred": 0,
                    "segment_count_gt": 2,
                    "segment_count_pred": 0,
                    "segment_count_delta": -2,
                    "transcript_match_class": "missed",
                },
            }
        },
        id="sc_missed",
    ),
    # -- No GT segments: all noncoding GT, pred has a spurious exon
    # transcript_match_class should be absent (None → not stored)
    pytest.param(
        np.array([
            [8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 0, 0, 0, 8, 8, 8, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "EXON": {
                "STRUCTURAL_COHERENCE": {
                    "gap_chain_match": True,
                    "gap_chain_lcs_ratio": 1.0,
                    "gap_count_match": True,
                    "gap_count_gt": 0,
                    "gap_count_pred": 0,
                    "segment_count_gt": 0,
                    "segment_count_pred": 1,
                    "segment_count_delta": 1,
                },
            }
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
        [EXON],
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "EXON": {
                "STRUCTURAL_COHERENCE": {
                    "gap_chain_match": False,
                    "gap_chain_lcs_ratio": 0.6,
                    "gap_count_match": False,
                    "gap_count_gt": 5,
                    "gap_count_pred": 4,
                    "segment_count_gt": 6,
                    "segment_count_pred": 5,
                    "segment_count_delta": -1,
                    "transcript_match_class": "structurally_different",
                },
            }
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
        [EXON],
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "EXON": {
                "STRUCTURAL_COHERENCE": {
                    "gap_chain_match": True,
                    "gap_chain_lcs_ratio": 1.0,
                    "gap_count_match": True,
                    "gap_count_gt": 0,
                    "gap_count_pred": 0,
                    "segment_count_gt": 1,
                    "segment_count_pred": 1,
                    "segment_count_delta": 0,
                    "transcript_match_class": "exact",
                },
            }
        },
        id="sc_single_segment",
    ),
]

# ------------------------------------------------------------------
# DIAGNOSTIC_DEPTH test cases
# ------------------------------------------------------------------

DIAGNOSTIC_DEPTH_TEST_CASES = [
    # -- No errors: identical 2-exon predictions
    # All segments match 1:1 with zero residuals
    pytest.param(
        np.array([
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "EXON": {
                "DIAGNOSTIC_DEPTH": {
                    "exon_skip_count": 0,
                    "segment_retention_count": 0,
                    "novel_insertion_count": 0,
                    "cascade_shift_count": 0,
                    "compensating_error_count": 0,
                    "total_junction_errors": 0,
                },
            }
        },
        id="dd_no_errors",
    ),
    # -- Exon skip: middle exon of 3 is absent from pred
    # GT exons: (1,2), (5,6), (9,10) — pred exons: (1,2), (9,10)
    # GT seg 1 unmatched, not absorbed → exon_skip
    pytest.param(
        np.array([
            [8, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 8],
            [8, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "EXON": {
                "DIAGNOSTIC_DEPTH": {
                    "exon_skip_count": 1,
                    "segment_retention_count": 0,
                    "novel_insertion_count": 0,
                    "cascade_shift_count": 0,
                    "compensating_error_count": 0,
                    "total_junction_errors": 1,
                },
            }
        },
        id="dd_exon_skip",
    ),
    # -- Novel insertion: pred splits one GT exon into two
    # GT exon: (1,9) — pred exons: (1,3) and (6,9)
    # GT matched to pred(6,9) (overlap=4), pred(1,3) unmatched → inside GT → novel insertion
    pytest.param(
        np.array([
            [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],
            [8, 0, 0, 0, 2, 2, 0, 0, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "EXON": {
                "DIAGNOSTIC_DEPTH": {
                    "exon_skip_count": 0,
                    "segment_retention_count": 0,
                    "novel_insertion_count": 1,
                    "cascade_shift_count": 0,
                    "compensating_error_count": 0,
                    "total_junction_errors": 1,
                },
            }
        },
        id="dd_novel_insertion",
    ),
    # -- Cascade shift: 4 exons all shifted right by 1 position
    # GT exons: (1,3),(6,8),(11,13),(16,18)
    # Pred exons: (2,4),(7,9),(12,14),(17,19)
    # All 5' residuals = +1 → run of 4 same-sign → 1 cascade
    pytest.param(
        np.array([
            [8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 8],
            [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "EXON": {
                "DIAGNOSTIC_DEPTH": {
                    "exon_skip_count": 0,
                    "segment_retention_count": 0,
                    "novel_insertion_count": 0,
                    "cascade_shift_count": 1,
                    "compensating_error_count": 0,
                    "total_junction_errors": 1,
                },
            }
        },
        id="dd_cascade_shift",
    ),
    # -- Compensating errors: exon 1 extends right by 1, exon 2 starts left by 1
    # GT exons: (1,3),(7,9) — pred exons: (1,4),(6,9)
    # Residuals: (0,+1), (-1,0) → res_3p[0]+res_5p[1] = 1+(-1) = 0
    pytest.param(
        np.array([
            [8, 0, 0, 0, 2, 2, 2, 0, 0, 0, 8],
            [8, 0, 0, 0, 0, 2, 0, 0, 0, 0, 8],
        ]),
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "EXON": {
                "DIAGNOSTIC_DEPTH": {
                    "exon_skip_count": 0,
                    "segment_retention_count": 0,
                    "novel_insertion_count": 0,
                    "cascade_shift_count": 0,
                    "compensating_error_count": 1,
                    "total_junction_errors": 1,
                },
            }
        },
        id="dd_compensating_errors",
    ),
]

MULTI_SEQUENCE_TEST_CASES = [
    pytest.param(
        [np.array([8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8])],
                [np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])],
        BEND_LABEL_CONFIG,
        [EXON],
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "EXON": {
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
            }
        },
        id="no_nuc_positives",
    ),
]
