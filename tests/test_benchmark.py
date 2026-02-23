import math

import numpy as np
import pytest

from dna_segmentation_benchmark.eval.evaluate_predictors import (
    benchmark_gt_vs_pred_single,
    benchmark_gt_vs_pred_multiple,
    EvalMetrics,
)

from benchmark_test_cases import SINGLE_SEQUENCE_TEST_CASES, MULTI_SEQUENCE_TEST_CASES


@pytest.mark.parametrize(
    "gt_pred_array, label_config, classes, metrics, expected_errors",
    SINGLE_SEQUENCE_TEST_CASES,
)
def test_benchmark_single(gt_pred_array, label_config, classes, metrics, expected_errors):
    """Test single-sequence benchmark with various label configs and metrics."""
    benchmark_results = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0],
        pred_labels=gt_pred_array[1],
        mask_labels=gt_pred_array[2] if gt_pred_array.shape[0] > 2 else None,
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

    filtered_keys = set(benchmark_results.keys()) - {"transition_failures"}
    assert filtered_keys == set(expected_errors.keys()), (
        f"The benchmark keys do not match the expected keys. Expected {expected_errors.keys()}, got {filtered_keys}"
    )

    for class_key in benchmark_results:
        if class_key == "transition_failures":
            continue
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            metric_eval_mapping[metric](expected_results[metric.name], class_results[metric.name])


@pytest.mark.parametrize(
    "gt_arrays, pred_arrays, label_config, classes, metrics, expected_errors",
    MULTI_SEQUENCE_TEST_CASES,
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

    filtered_keys = set(benchmark_results.keys()) - {"transition_failures"}
    assert filtered_keys == set(expected_errors.keys()), (
        f"The benchmark keys do not match the expected keys. Expected {expected_errors.keys()}, got {filtered_keys}"
    )

    for class_key in benchmark_results:
        if class_key == "transition_failures":
            continue
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            metric_eval_mapping[metric](expected_results[metric.name], class_results[metric.name])


# ------------------------------------------------------------------
# LabelConfig-specific tests
# ------------------------------------------------------------------


from dna_segmentation_benchmark.label_definition import LabelConfig, BEND_LABEL_CONFIG

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
    # Filter out 'distributions' from computed if not in expected, to maintain backward compatibility of tests
    computed_keys = set(computed_section_metrics.keys())
    expected_keys = set(expected_section_metrics.keys())

    if "iou_scores" in computed_keys and "iou_scores" not in expected_keys:
        computed_keys.remove("iou_scores")

    # Allow computed to have more keys than expected (new features)
    assert expected_keys.issubset(computed_keys), (
        f"The computed metrics are missing expected keys. Missing: {expected_keys - computed_keys}"
    )
    for section_metric in expected_section_metrics:
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
        # Allow extra metrics in computed that are not in expected
        if metric_key not in computed_ml:
            raise AssertionError(f"Expected metric {metric_key} not found in computed results")

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
