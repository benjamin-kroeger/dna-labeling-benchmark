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
                    "inner_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
                    "all_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
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
                    "inner_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
                    "all_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
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
                    "inner_section_boundaries": {"tp": 0, "fp": 0, "fn": 1, "tn": 0},
                    "all_section_boundaries": {"tp": 0, "fp": 0, "fn": 1, "tn": 0},
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
                    "inner_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
                    "all_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
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
                    "inner_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
                    "all_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
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
                    "inner_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
                    "all_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
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
                    "inner_section_boundaries": {"tp": 1, "fp": 0, "fn": 0, "tn": 0},
                    "all_section_boundaries": {"tp": 1, "fp": 0, "fn": 0, "tn": 0},
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
                    "inner_section_boundaries": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                    "all_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
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
                    "inner_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
                    "all_section_boundaries": {"tp": 0, "fp": 1, "fn": 0, "tn": 0},
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
                    "inner_section_boundaries": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
                    "all_section_boundaries": {"tp": 1, "fp": 0, "fn": 0, "tn": 0},
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
                    "inner_section_boundaries": {"tp": [0, 0], "fp": [0, 0], "fn": [0, 0], "tn": [0, 0]},
                    "all_section_boundaries": {"tp": [1, 1], "fp": [0, 0], "fn": [0, 0], "tn": [0, 0]},
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
                    "inner_section_boundaries": {"precision": 0, "recall": 0.0},
                    "all_section_boundaries": {"precision": 0, "recall": 0.0},
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
