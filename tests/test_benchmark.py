import math

import numpy as np
import pytest

from dna_segmentation_benchmark.eval.evaluate_predictors import (
    benchmark_gt_vs_pred_single,
    benchmark_gt_vs_pred_multiple,
    EvalMetrics,
)

from benchmark_test_cases import (
    SINGLE_SEQUENCE_TEST_CASES,
    MULTI_SEQUENCE_TEST_CASES,
    STRUCTURAL_COHERENCE_TEST_CASES,
    DIAGNOSTIC_DEPTH_TEST_CASES,
)


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

    filtered_keys = set(benchmark_results.keys()) - {"transition_failures", "false_transitions"}
    assert filtered_keys == set(expected_errors.keys()), (
        f"The benchmark keys do not match the expected keys. Expected {expected_errors.keys()}, got {filtered_keys}"
    )

    for class_key in benchmark_results:
        if class_key in ("transition_failures", "false_transitions"):
            continue
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            _METRIC_EVAL_DISPATCH[metric](expected_results[metric.name], class_results[metric.name])


@pytest.mark.parametrize(
    "gt_pred_array, label_config, classes, metrics, expected_errors",
    STRUCTURAL_COHERENCE_TEST_CASES,
)
def test_structural_coherence(gt_pred_array, label_config, classes, metrics, expected_errors):
    """Test structural coherence metrics (gap chain, transcript classification)."""
    benchmark_results = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0],
        pred_labels=gt_pred_array[1],
        label_config=label_config,
        classes=classes,
        metrics=metrics,
    )

    for class_key in expected_errors:
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            _METRIC_EVAL_DISPATCH[metric](expected_results[metric.name], class_results[metric.name])


@pytest.mark.parametrize(
    "gt_pred_array, label_config, classes, metrics, expected_errors",
    DIAGNOSTIC_DEPTH_TEST_CASES,
)
def test_diagnostic_depth(gt_pred_array, label_config, classes, metrics, expected_errors):
    """Test diagnostic depth metrics (junction errors, correlations, structural summary)."""
    benchmark_results = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0],
        pred_labels=gt_pred_array[1],
        label_config=label_config,
        classes=classes,
        metrics=metrics,
    )

    for class_key in expected_errors:
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            _METRIC_EVAL_DISPATCH[metric](expected_results[metric.name], class_results[metric.name])


@pytest.mark.parametrize(
    "gt_arrays, pred_arrays, label_config, classes, metrics, expected_errors",
    MULTI_SEQUENCE_TEST_CASES,
)
def test_benchmark_multiple(gt_arrays, pred_arrays, label_config, classes, metrics, expected_errors):
    """Test multi-sequence benchmark with aggregation and summary metrics."""
    benchmark_results = benchmark_gt_vs_pred_multiple(
        gt_labels=gt_arrays,
        pred_labels=pred_arrays,
        label_config=label_config,
        classes=classes,
        metrics=metrics,
    )

    filtered_keys = set(benchmark_results.keys()) - {"transition_failures", "false_transitions"}
    assert filtered_keys == set(expected_errors.keys()), (
        f"The benchmark keys do not match the expected keys. Expected {expected_errors.keys()}, got {filtered_keys}"
    )

    for class_key in benchmark_results:
        if class_key in ("transition_failures", "false_transitions"):
            continue
        class_results = benchmark_results[class_key]
        expected_results = expected_errors[class_key]
        for metric in metrics:
            _METRIC_EVAL_DISPATCH[metric](expected_results[metric.name], class_results[metric.name])


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


def _eval_region_discovery(expected, computed):
    """Verify region-discovery hit counts (or precision/recall if aggregated)."""
    expected_keys = set(expected.keys())
    computed_keys = set(computed.keys())

    assert expected_keys.issubset(computed_keys), (
        f"Region discovery computed metrics are missing expected keys. Missing: {expected_keys - computed_keys}"
    )
    for key in expected:
        _assert_metric_value_equal(expected[key], computed[key], key)


def _eval_boundary_exactness(expected, computed):
    """Verify boundary-exactness metrics."""
    expected_keys = set(expected.keys())
    computed_keys = set(computed.keys())

    assert expected_keys.issubset(computed_keys), (
        f"Boundary exactness computed metrics are missing expected keys. Missing: {expected_keys - computed_keys}"
    )
    for key in expected:
        _assert_metric_value_equal(expected[key], computed[key], key)


def _eval_nucleotide_classification(expected, computed):
    """Verify nucleotide-classification metrics."""
    expected_keys = set(expected.keys())
    computed_keys = set(computed.keys())

    assert expected_keys.issubset(computed_keys), (
        f"Nucleotide classification computed metrics are missing expected keys. Missing: {expected_keys - computed_keys}"
    )
    for key in expected:
        _assert_metric_value_equal(expected[key], computed[key], key)


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


def _eval_frameshift_metrics(expected_frameshift, computed_frameshift):
    assert set(expected_frameshift.keys()) == set(computed_frameshift.keys()), (
        "The keys for the frameshift metrics dont match."
    )
    for metric in expected_frameshift:
        assert (expected_frameshift[metric] == computed_frameshift[metric]).all(), (
            "The computed frame assignment does not match the expected frame assignment."
        )


def _eval_structural_coherence(expected, computed):
    """Verify structural coherence metrics (gap chain, transcript classification)."""
    expected_keys = set(expected.keys())
    computed_keys = set(computed.keys())

    assert expected_keys.issubset(computed_keys), (
        f"Structural coherence computed metrics are missing expected keys. Missing: {expected_keys - computed_keys}"
    )
    for key in expected:
        _assert_metric_value_equal(expected[key], computed[key], key)


def _eval_diagnostic_depth(expected, computed):
    """Verify diagnostic depth metrics (junction errors, correlations, structural summary)."""
    expected_keys = set(expected.keys())
    computed_keys = set(computed.keys())

    assert expected_keys.issubset(computed_keys), (
        f"Diagnostic depth computed metrics are missing expected keys. Missing: {expected_keys - computed_keys}"
    )
    for key in expected:
        _assert_metric_value_equal(expected[key], computed[key], key)


def _assert_metric_value_equal(expected, computed, key_name: str):
    """Compare a single metric value, handling dicts, lists, scalars, and None."""
    if isinstance(expected, dict):
        assert isinstance(computed, dict), f"Expected dict for {key_name}, got {type(computed)}"
        for sub_key in expected:
            assert sub_key in computed, f"Missing sub-key {sub_key} in {key_name}"
            _assert_metric_value_equal(expected[sub_key], computed[sub_key], f"{key_name}.{sub_key}")
    elif isinstance(expected, list):
        assert isinstance(computed, list), f"Expected list for {key_name}, got {type(computed)}"
        assert len(computed) == len(expected), (
            f"List length mismatch for {key_name}: expected {len(expected)}, got {len(computed)}"
        )
        for i, (exp_item, comp_item) in enumerate(zip(expected, computed)):
            _assert_metric_value_equal(exp_item, comp_item, f"{key_name}[{i}]")
    elif expected is None:
        assert computed is None, f"Expected None for {key_name}, got {computed}"
    elif isinstance(expected, float):
        assert math.isclose(expected, computed, abs_tol=0.001, rel_tol=0.011), (
            f"Float mismatch for {key_name}: expected {expected}, got {computed}"
        )
    else:
        assert expected == computed, (
            f"Value mismatch for {key_name}: expected {expected}, got {computed}"
        )


# ------------------------------------------------------------------
# Dispatch table
# ------------------------------------------------------------------

_METRIC_EVAL_DISPATCH = {
    EvalMetrics.INDEL: _eval_indel_metrics,
    EvalMetrics.REGION_DISCOVERY: _eval_region_discovery,
    EvalMetrics.BOUNDARY_EXACTNESS: _eval_boundary_exactness,
    EvalMetrics.NUCLEOTIDE_CLASSIFICATION: _eval_nucleotide_classification,
    EvalMetrics.FRAMESHIFT: _eval_frameshift_metrics,
    EvalMetrics.STRUCTURAL_COHERENCE: _eval_structural_coherence,
    EvalMetrics.DIAGNOSTIC_DEPTH: _eval_diagnostic_depth,
}
