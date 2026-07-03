import numpy as np
import pytest

from dna_segmentation_benchmark.eval.evaluate_predictors import (
    benchmark_gt_vs_pred_single,
    benchmark_gt_vs_pred_multiple,
    StreamingBenchmark,
    EvalMetrics,
)
from dna_segmentation_benchmark.eval.accumulators import BenchmarkAccumulator
from dna_segmentation_benchmark.pipeline import benchmark_from_gff
from dna_segmentation_benchmark.label_definition import (
    AnnotationMode,
    BenchmarkScope,
    BEND_LABEL_CONFIG,
    LabelConfig,
)

from support.cases import (
    SINGLE_SEQUENCE_TEST_CASES,
    MULTI_SEQUENCE_TEST_CASES,
    STRUCTURAL_COHERENCE_TEST_CASES,
    DIAGNOSTIC_DEPTH_TEST_CASES,
    SPLICE_SITE_TEST_CASES,
)
from support.constants import MEDIA_METRICS
from support.gff import UTR_ROLE_MAP, UTR_ROLE_MAP_NO_CDS
from support.metric_compare import assert_metric_value_equal


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
        assert_metric_value_equal(
            expected_errors[metric.name],
            benchmark_results[metric.name],
            metric.name,
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
        assert_metric_value_equal(
            expected_errors[metric.name],
            benchmark_results[metric.name],
            metric.name,
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
    assert_metric_value_equal(
        expected_splice,
        result["STRUCTURAL_COHERENCE"]["splice_site_results"],
        "splice_site_results",
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
        assert_metric_value_equal(
            expected_errors[metric.name],
            benchmark_results[metric.name],
            metric.name,
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
        assert_metric_value_equal(
            expected_errors[metric.name],
            benchmark_results[metric.name],
            metric.name,
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
    assert_metric_value_equal(merged, aggregated_metrics, "aggregated")


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

    assert_metric_value_equal(expected, individual, "individual_results")


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


def test_benchmark_from_gff_uses_feature_role_maps_for_utr_cds_mode(utr_gt_gff, utr_pred_gff):
    config = LabelConfig(
        annotation_mode=AnnotationMode.UTR_CDS_INTRON,
        background_label=9,
        evaluation_scope=BenchmarkScope.CDS,
        five_prime_utr_label=4,
        cds_label=0,
        three_prime_utr_label=5,
    )

    results = benchmark_from_gff(
        gt_path=utr_gt_gff,
        pred_paths={"pred": utr_pred_gff},
        label_config=config,
        metrics=[
            EvalMetrics.REGION_DISCOVERY,
            EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
            EvalMetrics.STRUCTURAL_COHERENCE,
        ],
        gt_feature_role_map=UTR_ROLE_MAP,
        pred_feature_role_maps={"pred": UTR_ROLE_MAP_NO_CDS},
    )

    aggregated = results["pred"]["aggregated"]
    assert aggregated["metadata"]["annotation_mode"] == "UTR_CDS_INTRON"
    assert aggregated["metadata"]["evaluation_scope"] == "cds"
    assert aggregated["REGION_DISCOVERY"]["perfect_boundary_hit"]["precision"] == 0.0


def test_boundary_offsets_are_matched_pairs_not_overlapping_pairs():
    """One prediction spanning two GT sections yields one residual, not two.

    Regression guard for the boundary-landscape inflation fix: offsets/IoU are
    collected over the greedy 1:1-matched pairs, so a prediction that overlaps
    several GT sections contributes a single residual.
    """
    gt = np.array([0, 0, 0, 2, 2, 0, 0, 0, 8, 8])      # exon (0,2) + exon (5,7)
    pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 8, 8])    # one exon (0,7) spanning both

    result = benchmark_gt_vs_pred_single(
        gt_labels=gt,
        pred_labels=pred,
        label_config=BEND_LABEL_CONFIG,
        metrics=[EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.REGION_DISCOVERY],
    )

    offsets = result["BOUNDARY_EXACTNESS"]["fuzzy_metrics"]["boundary_offsets"]
    ious = result["BOUNDARY_EXACTNESS"]["iou_scores"]
    # Two GT sections overlap the single prediction, but only one pair is matched.
    n_matched = result["REGION_DISCOVERY"]["neighborhood_hit"].tp
    assert n_matched == 1
    assert len(offsets) == n_matched == len(ious)


# ---------------------------------------------------------------------------
# Streaming aggregation parity (OOM fix): feeding arrays one at a time must
# reproduce the list-based / materialized results exactly.
# ---------------------------------------------------------------------------

def test_streaming_benchmark_matches_multiple_list_api():
    """StreamingBenchmark.add(...).result() == benchmark_gt_vs_pred_multiple(lists)."""
    gt_arrays = [
        np.array([8, 8, 0, 0, 0, 2, 2, 0, 0, 8]),
        np.array([8, 0, 0, 2, 2, 0, 0, 2, 2, 8]),
        np.array([0, 0, 2, 2, 2, 0, 0, 0, 2, 2]),
    ]
    pred_arrays = [
        np.array([8, 8, 0, 0, 0, 2, 2, 0, 0, 8]),
        np.array([8, 0, 0, 2, 2, 2, 2, 0, 0, 8]),
        np.array([0, 0, 2, 2, 0, 0, 0, 2, 2, 2]),
    ]

    reference = benchmark_gt_vs_pred_multiple(
        gt_labels=gt_arrays,
        pred_labels=pred_arrays,
        label_config=BEND_LABEL_CONFIG,
        metrics=MEDIA_METRICS,
    )

    bench = StreamingBenchmark(BEND_LABEL_CONFIG, MEDIA_METRICS)
    for gt, pred in zip(gt_arrays, pred_arrays):
        bench.add(gt, pred)
    streamed = bench.result()

    assert bench.count == len(gt_arrays)
    assert_metric_value_equal(reference, streamed, "streaming_benchmark")


def test_streaming_benchmark_matches_multiple_list_api_with_masks():
    """The masked per-span path is also identical under streaming."""
    gt_arrays = [
        np.array([8, 8, 0, 0, 0, 2, 2, 0, 0, 8]),
        np.array([8, 0, 0, 2, 2, 0, 0, 2, 2, 8]),
    ]
    pred_arrays = [
        np.array([8, 8, 0, 0, 0, 2, 2, 0, 0, 8]),
        np.array([8, 0, 0, 2, 2, 2, 2, 0, 0, 8]),
    ]
    masks = [
        np.array([False, False, False, False, False, False, False, True, True, True]),
        np.array([True, False, False, False, False, False, False, False, False, True]),
    ]

    reference = benchmark_gt_vs_pred_multiple(
        gt_labels=gt_arrays,
        pred_labels=pred_arrays,
        label_config=BEND_LABEL_CONFIG,
        metrics=MEDIA_METRICS,
        mask_labels=masks,
    )

    bench = StreamingBenchmark(BEND_LABEL_CONFIG, MEDIA_METRICS)
    for gt, pred, mask in zip(gt_arrays, pred_arrays, masks):
        bench.add(gt, pred, mask)
    streamed = bench.result()

    assert_metric_value_equal(reference, streamed, "streaming_benchmark_masked")


