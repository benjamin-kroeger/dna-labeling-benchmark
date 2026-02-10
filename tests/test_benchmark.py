import math

import numpy as np
import pytest

from dna_segmentation_benchmark.evaluate_predictors import (
    benchmark_gt_vs_pred_single,
    benchmark_gt_vs_pred_multiple,
    EvalMetrics,
)
from dna_segmentation_benchmark.label_definition import LabelConfig, BEND_LABEL_CONFIG


# ------------------------------------------------------------------
# Convenience token constants (replace enum members)
# ------------------------------------------------------------------
EXON, DONOR, INTRON, ACCEPTOR, NONCODING = 0, 1, 2, 3, 8

# A second label set to prove label-agnosticism
CUSTOM_CONFIG = LabelConfig(
    labels={-1: "NONCODING", 5: "CDS", 3: "PROMOTER"},
    background_label=-1,
    coding_label=5,
)


@pytest.mark.parametrize(
    "gt_pred_array, label_config, classes, metrics, expected_errors",
    [
        pytest.param(
            np.array(
                [
                    [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8],
                    [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8],
                ]
            ),
            BEND_LABEL_CONFIG,
            [EXON],
            [EvalMetrics.INDEL, EvalMetrics.SECTION],
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
                    "SECTION": {
                        "nucleotide": {"tn": 4, "fp": 9, "fn": 6, "tp": 6},
                        "section": {"tn": 0, "fp": 3, "fn": 1, "tp": 0},
                        "strict_section": {"tn": 0, "fp": 3, "fn": 1, "tp": 0},
                        "inner_section_boundaries": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "all_section_boundaries": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "first_sec_correct_3_prime_boundary": 0,
                        "last_sec_correct_5_prime_boundary": 0,
                    },
                }
            },
            id="exon_all_insertions_deletions",
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
            [EvalMetrics.SECTION],
            {
                "EXON": {
                    "SECTION": {
                        "nucleotide": {"tn": 11, "fp": 6, "fn": 0, "tp": 12},
                        "section": {"tn": 0, "fp": 0, "fn": 0, "tp": 4},
                        "strict_section": {"tn": 0, "fp": 3, "fn": 0, "tp": 1},
                        "inner_section_boundaries": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "all_section_boundaries": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "first_sec_correct_3_prime_boundary": 1,
                        "last_sec_correct_5_prime_boundary": 1,
                    }
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
            [EvalMetrics.INDEL, EvalMetrics.SECTION],
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
                    "SECTION": {
                        "nucleotide": {"tn": 0, "fp": 3, "fn": 5, "tp": 2},
                        "section": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "strict_section": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "inner_section_boundaries": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "all_section_boundaries": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "first_sec_correct_3_prime_boundary": 0,
                        "last_sec_correct_5_prime_boundary": 0,
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
            [EvalMetrics.SECTION],
            {
                "EXON": {
                    "SECTION": {
                        "nucleotide": {"tn": 13, "fp": 0, "fn": 0, "tp": 12},
                        "section": {"tn": 0, "fp": 0, "fn": 0, "tp": 3},
                        "strict_section": {"tn": 0, "fp": 0, "fn": 0, "tp": 3},
                        "inner_section_boundaries": {"tn": 0, "fp": 0, "fn": 0, "tp": 1},
                        "all_section_boundaries": {"tn": 0, "fp": 0, "fn": 0, "tp": 1},
                        "first_sec_correct_3_prime_boundary": 1,
                        "last_sec_correct_5_prime_boundary": 1,
                    }
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
            [EvalMetrics.SECTION],
            {
                "EXON": {
                    "SECTION": {
                        "nucleotide": {"tn": 7, "fp": 0, "fn": 0, "tp": 5},
                        "section": {"tn": 0, "fp": 0, "fn": 0, "tp": 1},
                        "strict_section": {"tn": 0, "fp": 0, "fn": 0, "tp": 1},
                        "inner_section_boundaries": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
                        "all_section_boundaries": {"tn": 0, "fp": 0, "fn": 0, "tp": 1},
                        "first_sec_correct_3_prime_boundary": 1,
                        "last_sec_correct_5_prime_boundary": 1,
                    }
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
            [EvalMetrics.INDEL, EvalMetrics.SECTION],
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
                    "SECTION": {
                        "nucleotide": {"tn": 6, "fp": 1, "fn": 2, "tp": 2},
                        "section": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "strict_section": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "inner_section_boundaries": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
                        "all_section_boundaries": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "first_sec_correct_3_prime_boundary": 0,
                        "last_sec_correct_5_prime_boundary": 0,
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
                    "SECTION": {
                        "nucleotide": {"tn": 7, "fp": 0, "fn": 3, "tp": 1},
                        "section": {"tn": 0, "fp": 1, "fn": 1, "tp": 0},
                        "strict_section": {"tn": 0, "fp": 1, "fn": 1, "tp": 0},
                        "inner_section_boundaries": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "all_section_boundaries": {"tn": 0, "fp": 1, "fn": 0, "tp": 0},
                        "first_sec_correct_3_prime_boundary": 0,
                        "last_sec_correct_5_prime_boundary": 0,
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
            [EvalMetrics.SECTION],
            {
                "INTRON": {
                    "SECTION": {
                        "nucleotide": {"tn": 10, "fp": 0, "fn": 0, "tp": 4},
                        "section": {"tn": 0, "fp": 0, "fn": 0, "tp": 1},
                        "strict_section": {"tn": 0, "fp": 0, "fn": 0, "tp": 1},
                        "inner_section_boundaries": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
                        "all_section_boundaries": {"tn": 0, "fp": 0, "fn": 0, "tp": 1},
                        "first_sec_correct_3_prime_boundary": 1,
                        "last_sec_correct_5_prime_boundary": 1,
                    }
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
    ],
)
def test_benchmark_single(gt_pred_array, label_config, classes, metrics, expected_errors):
    """Test single-sequence benchmark with various label configs and metrics."""
    benchmark_results = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0],
        pred_labels=gt_pred_array[1],
        label_config=label_config,
        classes=classes,
        metrics=metrics,
    )

    metric_eval_mapping = {
        EvalMetrics.INDEL: _eval_indel_metrics,
        EvalMetrics.SECTION: _eval_section_metrics,
        EvalMetrics.ML: _eval_ml_metrics,
        EvalMetrics.FRAMESHIFT: _eval_frameshift_metrics,
    }

    assert benchmark_results.keys() == expected_errors.keys(), (
        "The benchmark keys do not match the expected keys"
    )

    for class_key in benchmark_results:
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            metric_eval_mapping[metric](expected_results[metric.name], class_results[metric.name])


@pytest.mark.parametrize(
    "gt_arrays, pred_arrays, label_config, classes, metrics, expected_errors",
    [
        pytest.param(
            [np.array([8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8])],
            [np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])],
            BEND_LABEL_CONFIG,
            [EXON],
            [EvalMetrics.INDEL, EvalMetrics.SECTION, EvalMetrics.ML],
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
                    "SECTION": {
                        "nucleotide": {"tn": [13], "fp": [0], "fn": [12], "tp": [0]},
                        "section": {"tn": [0], "fp": [0], "fn": [3], "tp": [0]},
                        "strict_section": {"tn": [0], "fp": [0], "fn": [3], "tp": [0]},
                        "inner_section_boundaries": {"tn": [0], "fp": [0], "fn": [1], "tp": [0]},
                        "all_section_boundaries": {"tn": [0], "fp": [0], "fn": [1], "tp": [0]},
                        "first_sec_correct_3_prime_boundary": [0],
                        "last_sec_correct_5_prime_boundary": [0],
                    },
                    "ML": {
                        "correct_inner_section_boundaries_metrics": {"precision": 0, "recall": 0.0},
                        "correct_overall_section_boundaries_metrics": {"precision": 0, "recall": 0.0},
                        "encompass_section_match_metrics": {"precision": 0, "recall": 0.0},
                        "nucleotide_level_metrics": {"precision": 0, "recall": 0.0},
                        "strict_section_match_metrics": {"precision": 0, "recall": 0.0},
                    },
                }
            },
            id="no_nuc_positives",
        ),
    ],
)
def test_benchmark_multiple(gt_arrays, pred_arrays, label_config, classes, metrics, expected_errors):
    """Test multi-sequence benchmark with aggregation and ML summary."""
    benchmark_results = benchmark_gt_vs_pred_multiple(
        gt_labels=gt_arrays,
        pred_labels=pred_arrays,
        label_config=label_config,
        classes=classes,
        metrics=metrics,
    )

    metric_eval_mapping = {
        EvalMetrics.INDEL: _eval_indel_metrics,
        EvalMetrics.SECTION: _eval_section_metrics,
        EvalMetrics.ML: _eval_ml_metrics,
        EvalMetrics.FRAMESHIFT: _eval_frameshift_metrics,
    }

    assert benchmark_results.keys() == expected_errors.keys(), (
        "The benchmark keys do not match the expected keys"
    )

    for class_key in benchmark_results:
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            metric_eval_mapping[metric](expected_results[metric.name], class_results[metric.name])


# ------------------------------------------------------------------
# LabelConfig-specific tests
# ------------------------------------------------------------------


class TestLabelConfig:
    """Tests for the LabelConfig pydantic model."""

    def test_basic_construction(self):
        config = LabelConfig(
            labels={0: "EXON", 8: "NONCODING"},
            background_label=8,
        )
        assert config.background_label == 8
        assert config.background_name == "NONCODING"

    def test_coding_name(self):
        config = LabelConfig(
            labels={0: "EXON", 8: "BG"},
            background_label=8,
            coding_label=0,
        )
        assert config.coding_name == "EXON"
        assert config.coding_label == 0

    def test_coding_label_none_by_default(self):
        config = LabelConfig(labels={8: "BG"}, background_label=8)
        assert config.coding_label is None
        assert config.coding_name is None

    def test_special_label_not_in_labels_raises(self):
        with pytest.raises(ValueError, match="background_label=99"):
            LabelConfig(labels={0: "EXON"}, background_label=99)

    def test_coding_label_not_in_labels_raises(self):
        with pytest.raises(ValueError, match="coding_label=42"):
            LabelConfig(labels={0: "EXON", 8: "BG"}, background_label=8, coding_label=42)

    def test_name_of(self):
        config = BEND_LABEL_CONFIG
        assert config.name_of(0) == "EXON"
        assert config.name_of(2) == "INTRON"

    def test_frameshift_without_coding_label_raises(self):
        config = LabelConfig(labels={0: "X", 8: "BG"}, background_label=8)
        with pytest.raises(ValueError, match="coding_label"):
            benchmark_gt_vs_pred_single(
                gt_labels=np.array([8, 8, 0, 0, 0, 8, 8]),
                pred_labels=np.array([8, 8, 0, 0, 0, 8, 8]),
                label_config=config,
                classes=[0],
                metrics=[EvalMetrics.FRAMESHIFT],
            )

    def test_frozen(self):
        config = BEND_LABEL_CONFIG
        with pytest.raises(Exception):
            config.background_label = 99


# ------------------------------------------------------------------
# Metric evaluation helpers
# ------------------------------------------------------------------


def _eval_section_metrics(expected_section_metrics, computed_section_metrics):
    assert set(expected_section_metrics.keys()) == set(computed_section_metrics.keys()), (
        "The keys for the section metrics dont match"
    )
    for section_metric in computed_section_metrics:
        assert computed_section_metrics[section_metric] == expected_section_metrics[section_metric], (
            f"The computed output of {section_metric} does not match the expected output"
        )


def _eval_indel_metrics(expected_indel, computed_indel):
    assert set(expected_indel.keys()) == set(computed_indel.keys()), (
        "The keys for the indel metrics dont match"
    )
    for metric in expected_indel:
        computed = computed_indel[metric]
        expected = expected_indel[metric]
        assert len(computed) == len(expected), (
            "The total number of errors does not match the expected number"
        )
        for comp, exp in zip(computed, expected):
            assert (comp == exp).all(), (
                f"The individual errors do not match: {comp} vs {exp}"
            )


def _eval_ml_metrics(expected_ml, computed_ml):
    for metric_key in expected_ml:
        for eval_met in expected_ml[metric_key]:
            assert math.isclose(
                expected_ml[metric_key][eval_met],
                computed_ml[metric_key][eval_met],
                abs_tol=0.001,
                rel_tol=0.011,
            ), f"The {metric_key} values do not match"


def _eval_frameshift_metrics(expected_frameshift, computed_frameshift):
    assert set(expected_frameshift.keys()) == set(computed_frameshift.keys()), (
        "The keys for the frameshift metrics dont match."
    )
    for metric in expected_frameshift:
        assert (expected_frameshift[metric] == computed_frameshift[metric]).all(), (
            "The computed frame assignment does not match the expected frame assignment."
        )
