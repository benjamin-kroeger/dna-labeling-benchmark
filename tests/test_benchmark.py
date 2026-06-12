import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from dna_segmentation_benchmark.eval.evaluate_predictors import (
    benchmark_gt_vs_pred_single,
    benchmark_gt_vs_pred_multiple,
    EvalMetrics,
)
from dna_segmentation_benchmark.eval.accumulators import BenchmarkAccumulator
from dna_segmentation_benchmark.pipeline import (
    benchmark_from_gff,
)

from benchmark_test_cases import (
    SINGLE_SEQUENCE_TEST_CASES,
    MULTI_SEQUENCE_TEST_CASES,
    STRUCTURAL_COHERENCE_TEST_CASES,
    DIAGNOSTIC_DEPTH_TEST_CASES,
    SPLICE_SITE_TEST_CASES,
)
from dna_segmentation_benchmark.label_definition import (
    AnnotationMode,
    BenchmarkScope,
    BEND_LABEL_CONFIG,
    LabelConfig,
)


@pytest.mark.parametrize(
    "gt_pred_array, label_config, metrics, expected_errors",
    SINGLE_SEQUENCE_TEST_CASES,
)
def test_benchmark_single(gt_pred_array, label_config, metrics, expected_errors):
    """Test single-sequence benchmark with various label configs and metrics."""
    benchmark_results = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0],
        pred_labels=gt_pred_array[1],
        mask_labels=gt_pred_array[2] if gt_pred_array.shape[0] > 2 else None,
        label_config=label_config,
        metrics=metrics,
    )

    filtered_keys = set(benchmark_results.keys()) - {"transition_failures", "false_transitions", "metadata"}
    assert filtered_keys == set(expected_errors.keys()), (
        f"The benchmark keys do not match the expected keys. Expected {expected_errors.keys()}, got {filtered_keys}"
    )

    for metric in metrics:
        _METRIC_EVAL_DISPATCH[metric](
            expected_errors[metric.name],
            benchmark_results[metric.name],
        )


@pytest.mark.parametrize(
    "gt_pred_array, label_config, metrics, expected_errors",
    STRUCTURAL_COHERENCE_TEST_CASES,
)
def test_structural_coherence(gt_pred_array, label_config, metrics, expected_errors):
    """Test structural coherence metrics (gap chain, transcript classification)."""
    benchmark_results = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0],
        pred_labels=gt_pred_array[1],
        label_config=label_config,
        metrics=metrics,
    )

    for metric in metrics:
        _METRIC_EVAL_DISPATCH[metric](
            expected_errors[metric.name],
            benchmark_results[metric.name],
        )


@pytest.mark.parametrize("gt_pred_array, expected_splice", SPLICE_SITE_TEST_CASES)
def test_splice_site_evaluation(gt_pred_array, expected_splice):
    """Test splice-site confusion matrix counts and derived FP tracking."""
    result = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0],
        pred_labels=gt_pred_array[1],
        label_config=BEND_LABEL_CONFIG,
        metrics=[EvalMetrics.STRUCTURAL_COHERENCE],
    )
    assert "STRUCTURAL_COHERENCE" in result, "Expected 'STRUCTURAL_COHERENCE' key in benchmark results"
    ss = result["STRUCTURAL_COHERENCE"]["splice_site_results"]
    for key, expected_val in expected_splice.items():
        assert ss[key] == expected_val, (
            f"splice_site_results[{key!r}]: expected {expected_val}, got {ss[key]}"
        )


@pytest.mark.parametrize(
    "gt_pred_array, label_config, metrics, expected_errors",
    DIAGNOSTIC_DEPTH_TEST_CASES,
)
def test_diagnostic_depth(gt_pred_array, label_config, metrics, expected_errors):
    """Test diagnostic depth metrics (junction errors, correlations, structural summary)."""
    benchmark_results = benchmark_gt_vs_pred_single(
        gt_labels=gt_pred_array[0],
        pred_labels=gt_pred_array[1],
        label_config=label_config,
        metrics=metrics,
    )

    for metric in metrics:
        _METRIC_EVAL_DISPATCH[metric](
            expected_errors[metric.name],
            benchmark_results[metric.name],
        )


@pytest.mark.parametrize(
    "gt_arrays, pred_arrays, label_config, metrics, expected_errors",
    MULTI_SEQUENCE_TEST_CASES,
)
def test_benchmark_multiple(gt_arrays, pred_arrays, label_config, metrics, expected_errors):
    """Test multi-sequence benchmark with aggregation and summary metrics."""
    benchmark_results = benchmark_gt_vs_pred_multiple(
        gt_labels=gt_arrays,
        pred_labels=pred_arrays,
        label_config=label_config,
        metrics=metrics,
    )

    filtered_keys = set(benchmark_results.keys()) - {"transition_failures", "false_transitions", "metadata"}
    assert filtered_keys == set(expected_errors.keys()), (
        f"The benchmark keys do not match the expected keys. Expected {expected_errors.keys()}, got {filtered_keys}"
    )

    for metric in metrics:
        _METRIC_EVAL_DISPATCH[metric](
            expected_errors[metric.name],
            benchmark_results[metric.name],
        )


def test_benchmark_multiple_streaming_aggregation_matches_merged_individual_results():
    gt_arrays = [
        np.array([8, 8, 0, 0, 0, 2, 2, 0, 0, 8]),
        np.array([8, 0, 0, 2, 2, 0, 0, 2, 2, 8]),
    ]
    pred_arrays = [
        np.array([8, 8, 0, 0, 0, 2, 2, 0, 0, 8]),
        np.array([8, 0, 0, 2, 2, 2, 2, 0, 0, 8]),
    ]
    metrics = [
        EvalMetrics.REGION_DISCOVERY,
        EvalMetrics.BOUNDARY_EXACTNESS,
        EvalMetrics.STRUCTURAL_COHERENCE,
        EvalMetrics.DIAGNOSTIC_DEPTH,
    ]

    aggregated = benchmark_gt_vs_pred_multiple(
        gt_labels=gt_arrays,
        pred_labels=pred_arrays,
        label_config=BEND_LABEL_CONFIG,
        metrics=metrics,
    )

    individual = benchmark_gt_vs_pred_multiple(
        gt_labels=gt_arrays,
        pred_labels=pred_arrays,
        label_config=BEND_LABEL_CONFIG,
        metrics=metrics,
        return_individual_results=True,
    )

    accumulator = BenchmarkAccumulator()
    for result in individual:
        accumulator.add(result)
    merged = accumulator.summarise()

    # ``aggregated`` adds annotation ``metadata`` at the public-API boundary;
    # the raw accumulator output does not. Compare the metric payloads only.
    aggregated_metrics = {key: value for key, value in aggregated.items() if key != "metadata"}
    assert set(merged) == set(aggregated_metrics), (
        f"Streaming vs merged key mismatch: {set(merged) ^ set(aggregated_metrics)}"
    )
    _assert_metric_value_equal(merged, aggregated_metrics, "aggregated")


def test_benchmark_multiple_return_individual_results_matches_single_sequence_outputs():
    gt_arrays = [
        np.array([8, 8, 0, 0, 0, 2, 2, 0, 0, 8]),
        np.array([8, 0, 0, 2, 2, 0, 0, 2, 2, 8]),
    ]
    pred_arrays = [
        np.array([8, 8, 0, 0, 0, 2, 2, 0, 0, 8]),
        np.array([8, 0, 0, 2, 2, 2, 2, 0, 0, 8]),
    ]
    metrics = [
        EvalMetrics.REGION_DISCOVERY,
        EvalMetrics.BOUNDARY_EXACTNESS,
        EvalMetrics.STRUCTURAL_COHERENCE,
    ]

    individual = benchmark_gt_vs_pred_multiple(
        gt_labels=gt_arrays,
        pred_labels=pred_arrays,
        label_config=BEND_LABEL_CONFIG,
        metrics=metrics,
        return_individual_results=True,
    )

    expected = [
        benchmark_gt_vs_pred_single(
            gt_labels=gt,
            pred_labels=pred,
            label_config=BEND_LABEL_CONFIG,
            metrics=metrics,
        )
        for gt, pred in zip(gt_arrays, pred_arrays)
    ]

    _assert_metric_value_equal(expected, individual, "individual_results")


def test_results_include_annotation_metadata():
    result = benchmark_gt_vs_pred_single(
        gt_labels=np.array([8, 0, 0, 8]),
        pred_labels=np.array([8, 0, 0, 8]),
        label_config=BEND_LABEL_CONFIG,
        metrics=[EvalMetrics.REGION_DISCOVERY, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
    )

    assert result["metadata"]["annotation_mode"] == "EXON_INTRON"
    assert result["metadata"]["evaluation_scope"] == "transcript_exon"
    assert "reported_scopes" not in result["metadata"]


def test_utr_cds_mode_compares_transcript_exon_and_cds_in_separate_runs():
    transcript_exon_config = LabelConfig(
        annotation_mode=AnnotationMode.UTR_CDS_INTRON,
        background_label=8,
        evaluation_scope=BenchmarkScope.TRANSCRIPT_EXON,
        five_prime_utr_label=4,
        cds_label=0,
        three_prime_utr_label=5,
    )
    cds_config = LabelConfig(
        annotation_mode=AnnotationMode.UTR_CDS_INTRON,
        background_label=8,
        evaluation_scope=BenchmarkScope.CDS,
        five_prime_utr_label=4,
        cds_label=0,
        three_prime_utr_label=5,
    )
    gt = np.array([8, 4, 4, 0, 0, 5, 5, 8])
    pred = np.array([8, 4, 4, 4, 4, 5, 5, 8])

    transcript_exon_result = benchmark_gt_vs_pred_single(
        gt_labels=gt,
        pred_labels=pred,
        label_config=transcript_exon_config,
        metrics=[
            EvalMetrics.REGION_DISCOVERY,
            EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
            EvalMetrics.STRUCTURAL_COHERENCE,
        ],
    )
    cds_result = benchmark_gt_vs_pred_single(
        gt_labels=gt,
        pred_labels=pred,
        label_config=cds_config,
        metrics=[
            EvalMetrics.REGION_DISCOVERY,
            EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
            EvalMetrics.STRUCTURAL_COHERENCE,
        ],
    )

    assert transcript_exon_result["metadata"]["annotation_mode"] == "UTR_CDS_INTRON"
    assert transcript_exon_result["metadata"]["evaluation_scope"] == "transcript_exon"
    assert cds_result["metadata"]["evaluation_scope"] == "cds"

    assert transcript_exon_result["REGION_DISCOVERY"]["perfect_boundary_hit"].tp == 1
    assert cds_result["REGION_DISCOVERY"]["perfect_boundary_hit"].tp == 0

    assert transcript_exon_result["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"].tp == 6
    assert cds_result["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"].tp == 0

    assert transcript_exon_result["STRUCTURAL_COHERENCE"]["exon_chain"].tp == 1
    assert cds_result["STRUCTURAL_COHERENCE"]["exon_chain"].fn == 1


def test_benchmark_from_gff_uses_feature_role_maps_for_utr_cds_mode(tmp_path):
    gt_path = tmp_path / "gt.gff"
    gt_path.write_text(
        """##gff-version 3
chr1\tGT\tmRNA\t1\t30\t.\t+\t.\tID=gt_tx1
chr1\tGT\tfive_prime_UTR\t1\t5\t.\t+\t.\tID=gt_u5;Parent=gt_tx1
chr1\tGT\tCDS\t6\t20\t.\t+\t0\tID=gt_cds;Parent=gt_tx1
chr1\tGT\tthree_prime_UTR\t21\t30\t.\t+\t.\tID=gt_u3;Parent=gt_tx1
"""
    )
    pred_path = tmp_path / "pred.gff"
    pred_path.write_text(
        """##gff-version 3
chr1\tPred\tmRNA\t1\t30\t.\t+\t.\tID=pred_tx1
chr1\tPred\tfive_prime_UTR\t1\t20\t.\t+\t.\tID=pred_u5;Parent=pred_tx1
chr1\tPred\tthree_prime_UTR\t21\t30\t.\t+\t.\tID=pred_u3;Parent=pred_tx1
"""
    )

    config = LabelConfig(
        annotation_mode=AnnotationMode.UTR_CDS_INTRON,
        background_label=9,
        evaluation_scope=BenchmarkScope.CDS,
        five_prime_utr_label=4,
        cds_label=0,
        three_prime_utr_label=5,
    )

    results = benchmark_from_gff(
        gt_path=gt_path,
        pred_paths={"pred": pred_path},
        label_config=config,
        metrics=[
            EvalMetrics.REGION_DISCOVERY,
            EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
            EvalMetrics.STRUCTURAL_COHERENCE,
        ],
        gt_feature_role_map={
            "five_prime_UTR": "five_prime_utr",
            "CDS": "cds",
            "three_prime_UTR": "three_prime_utr",
        },
        pred_feature_role_maps={
            "pred": {
                "five_prime_UTR": "five_prime_utr",
                "three_prime_UTR": "three_prime_utr",
            }
        },
    )

    per_transcript = results["pred"]["per_transcript"]
    assert per_transcript["metadata"]["annotation_mode"] == "UTR_CDS_INTRON"
    assert per_transcript["metadata"]["evaluation_scope"] == "cds"
    assert per_transcript["REGION_DISCOVERY"]["perfect_boundary_hit"]["precision"] == 0.0

# ------------------------------------------------------------------
# Metric evaluation helpers
# ------------------------------------------------------------------


def _eval_region_discovery(expected, computed):
    """Strictly verify region-discovery hit counts (exact 1:1)."""
    computed = _default_scope_payload(expected, computed)
    _assert_metric_value_equal(expected, computed, "REGION_DISCOVERY")


def _eval_boundary_exactness(expected, computed):
    """Strictly verify boundary-exactness metrics (exact 1:1)."""
    computed = _default_scope_payload(expected, computed)
    _assert_metric_value_equal(expected, computed, "BOUNDARY_EXACTNESS")


def _eval_nucleotide_classification(expected, computed):
    """Strictly verify nucleotide-classification metrics (exact 1:1)."""
    computed = _default_scope_payload(expected, computed)
    _assert_metric_value_equal(expected, computed, "NUCLEOTIDE_CLASSIFICATION")


def _eval_indel_metrics(expected_indel, computed_indel):
    """Compare boundary-typed INDEL output (``by_boundary`` run-length lists).

    ``by_boundary`` is compared strictly (each bucket holds a list of run
    *lengths*, compared order-insensitively because multiple runs at the same
    boundary/bucket carry no intrinsic ordering).  The opportunity denominators
    (``junction_opportunities``, ``n_gt_segments``, ``n_pred_segments``) are
    compared only when a fixture spells them out, so older fixtures that assert
    only ``by_boundary`` still pass.
    """
    computed_indel = _default_scope_payload(expected_indel, computed_indel)
    assert "by_boundary" in computed_indel, "INDEL output is missing 'by_boundary'"

    for denom_key in ("junction_opportunities", "n_gt_segments", "n_pred_segments"):
        if denom_key in expected_indel:
            assert expected_indel[denom_key] == computed_indel[denom_key], (
                f"INDEL {denom_key} differs: expected {expected_indel[denom_key]}, "
                f"got {computed_indel.get(denom_key)}"
            )

    expected_bb = expected_indel["by_boundary"]
    computed_bb = computed_indel["by_boundary"]
    assert set(expected_bb.keys()) == set(computed_bb.keys()), (
        f"INDEL boundary keys differ: missing {set(expected_bb) - set(computed_bb)}, "
        f"unexpected {set(computed_bb) - set(expected_bb)}"
    )
    for boundary, expected_buckets in expected_bb.items():
        computed_buckets = computed_bb[boundary]
        assert set(expected_buckets.keys()) == set(computed_buckets.keys()), (
            f"INDEL buckets differ for boundary {boundary}: expected "
            f"{set(expected_buckets)}, got {set(computed_buckets)}"
        )
        for bucket, expected_lengths in expected_buckets.items():
            assert sorted(computed_buckets[bucket]) == sorted(expected_lengths), (
                f"INDEL run lengths differ for {boundary}/{bucket}: "
                f"expected {expected_lengths}, got {computed_buckets[bucket]}"
            )


def _eval_frameshift_metrics(expected_frameshift, computed_frameshift):
    if "scopes" in computed_frameshift and "scopes" not in expected_frameshift:
        scope_payload = _single_scope_payload(computed_frameshift)
        computed_frameshift = {"gt_frames": np.asarray(scope_payload["frames"])}
    _OPTIONAL_FRAMESHIFT_KEYS = {"boundary_indel_total", "boundary_indel_in_frame", "n_skipped_non_divisible", "n_skipped_short"}
    unexpected = set(computed_frameshift.keys()) - set(expected_frameshift.keys()) - _OPTIONAL_FRAMESHIFT_KEYS
    assert not unexpected, f"Unexpected keys in computed frameshift: {unexpected}"
    assert set(expected_frameshift.keys()) <= set(computed_frameshift.keys()), (
        "The keys for the frameshift metrics dont match."
    )
    for metric in expected_frameshift:
        assert (expected_frameshift[metric] == computed_frameshift[metric]).all(), (
            "The computed frame assignment does not match the expected frame assignment."
        )


def _flatten_structural_scope(computed):
    """Merge a single-scope STRUCTURAL_COHERENCE payload into a flat dict."""
    if "scopes" in computed:
        scope_payload = _single_scope_payload(computed)
        return {
            **scope_payload,
            **{
                key: value
                for key, value in computed.items()
                if key != "scopes"
            },
        }
    return computed


def _eval_structural_coherence(expected, computed):
    """Strictly verify structural coherence metrics (exact 1:1)."""
    if "scopes" not in expected:
        computed = _flatten_structural_scope(computed)
    _assert_metric_value_equal(expected, computed, "STRUCTURAL_COHERENCE")


def _eval_diagnostic_depth(expected, computed):
    """Strictly verify diagnostic depth metrics (exact 1:1)."""
    computed = _default_scope_payload(expected, computed)
    _assert_metric_value_equal(expected, computed, "DIAGNOSTIC_DEPTH")


_COUNT_KEYS = {"tp", "fp", "fn", "tn"}


def _is_landscape_artifact(value) -> bool:
    """True for aggregate-only derived payloads that can't be hand-authored.

    The aggregated boundary ``fuzzy_metrics`` is a ``(DataFrame, ...)`` tuple
    (the precision landscape).  Such DataFrame-bearing values are checked
    structurally (present, non-None) rather than value-by-value, while the
    single-sequence ``fuzzy_metrics`` dict stays strictly compared.
    """
    if isinstance(value, pd.DataFrame):
        return True
    if isinstance(value, (tuple, list)):
        return any(isinstance(item, pd.DataFrame) for item in value)
    return False


def _as_count_bundle(value):
    """Return ``{tp,fp,fn,tn}`` if *value* is a (possibly partial) count bundle.

    Raw :class:`Counts`, terse fixture dicts like ``{"tp": 1, "fn": 2}``, and
    full ``{tp,fp,fn,tn}`` dicts all normalise to the same four-field form so
    that count *values* are always checked in full, regardless of which subset
    a fixture spelled out.  Returns ``None`` for anything that is not a count
    bundle (e.g. summarised ``{precision, recall, ...}`` dicts).
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict) and value and set(value).issubset(_COUNT_KEYS):
        return {field: int(value.get(field, 0)) for field in ("tp", "fp", "fn", "tn")}
    return None


def _assert_metric_value_equal(expected, computed, key_name: str):
    """Strictly compare a metric value 1:1 (keys *and* values).

    Dicts must have identical key sets (no missing, no extra), with two
    deliberate accommodations:

    * **count bundles** (``tp/fp/fn/tn``) are normalised so a fixture may spell
      out a subset, but every one of the four counts is still compared exactly;
    * **``*_stderr`` keys** carry non-deterministic bootstrap values, so they
      are not required in fixtures and only sanity-checked (float ≥ 0 or None).
    """
    # Count bundles: compare all four base counts exactly.
    expected_counts = _as_count_bundle(expected)
    computed_counts = _as_count_bundle(computed)
    if expected_counts is not None or computed_counts is not None:
        assert expected_counts is not None and computed_counts is not None, (
            f"Count-bundle type mismatch for {key_name}: expected {expected!r}, got {computed!r}"
        )
        assert expected_counts == computed_counts, (
            f"Count mismatch for {key_name}: expected {expected_counts}, got {computed_counts}"
        )
        return

    if dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        expected = dataclasses.asdict(expected)
    if dataclasses.is_dataclass(computed) and not isinstance(computed, type):
        computed = dataclasses.asdict(computed)
    if isinstance(expected, dict):
        assert isinstance(computed, dict), f"Expected dict for {key_name}, got {type(computed)}"
        # *_stderr (random bootstrap) and DataFrame-bearing landscape artifacts
        # are soft: not required in fixtures, only sanity-checked.
        def _soft(key, container):
            return (isinstance(key, str) and key.endswith("_stderr")) or _is_landscape_artifact(container.get(key))

        expected_keys = {k for k in expected if not _soft(k, expected)}
        computed_keys = {k for k in computed if not _soft(k, computed)}
        assert expected_keys == computed_keys, (
            f"Key-set mismatch for {key_name}: "
            f"missing {expected_keys - computed_keys}, unexpected {computed_keys - expected_keys}"
        )
        for stderr_key in (k for k in computed if isinstance(k, str) and k.endswith("_stderr")):
            value = computed[stderr_key]
            assert value is None or (isinstance(value, (int, float)) and value >= 0), (
                f"Bootstrap stderr {key_name}.{stderr_key} should be a non-negative float, got {value!r}"
            )
        for soft_key in (k for k in computed if _is_landscape_artifact(computed.get(k))):
            assert computed[soft_key] is not None, f"Missing landscape artifact {key_name}.{soft_key}"
        for sub_key in expected_keys:
            _assert_metric_value_equal(expected[sub_key], computed[sub_key], f"{key_name}.{sub_key}")
    elif isinstance(expected, np.ndarray):
        assert isinstance(computed, np.ndarray), f"Expected ndarray for {key_name}, got {type(computed)}"
        assert np.array_equal(expected, computed), (
            f"Array mismatch for {key_name}: expected {expected}, got {computed}"
        )
    elif isinstance(expected, pd.DataFrame):
        assert isinstance(computed, pd.DataFrame), f"Expected DataFrame for {key_name}, got {type(computed)}"
        assert expected.equals(computed), f"DataFrame mismatch for {key_name}"
    elif isinstance(expected, pd.Series):
        assert isinstance(computed, pd.Series), f"Expected Series for {key_name}, got {type(computed)}"
        assert expected.equals(computed), f"Series mismatch for {key_name}"
    elif isinstance(expected, tuple):
        assert isinstance(computed, tuple), f"Expected tuple for {key_name}, got {type(computed)}"
        assert len(computed) == len(expected), (
            f"Tuple length mismatch for {key_name}: expected {len(expected)}, got {len(computed)}"
        )
        for i, (exp_item, comp_item) in enumerate(zip(expected, computed)):
            _assert_metric_value_equal(exp_item, comp_item, f"{key_name}[{i}]")
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


def _default_scope_payload(expected: dict, computed: dict) -> dict:
    """Unwrap a single default scope when legacy flat expectations are used."""
    if "scopes" not in computed or "scopes" in expected:
        return computed
    return _single_scope_payload(computed)


def _single_scope_payload(computed: dict) -> dict:
    """Return the transcript-exon scope when present, otherwise the sole scope."""
    scopes = computed["scopes"]
    if "transcript_exon" in scopes:
        return scopes["transcript_exon"]
    if len(scopes) != 1:
        raise AssertionError(f"Expected a single scope payload, got {list(scopes.keys())}")
    return next(iter(scopes.values()))


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

