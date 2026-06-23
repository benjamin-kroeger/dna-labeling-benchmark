"""Tests for the typed accumulators (Counts / Stat) and metric accumulators."""

import numpy as np

from dna_segmentation_benchmark.eval.accumulators import (
    BenchmarkAccumulator,
    BoundaryExactnessAccumulator,
    DiagnosticDepthAccumulator,
    StructuralAccumulator,
    TransitionsAccumulator,
)
from dna_segmentation_benchmark.eval.statistics import (
    Counts,
    summarise_counts,
    _compute_summary_statistics,
)


def test_counts_add_and_sum():
    a = Counts(tp=1, fp=2, fn=3, tn=4)
    b = Counts(tp=10, fp=20, fn=30, tn=40)
    assert a + b == Counts(tp=11, fp=22, fn=33, tn=44)
    # sum() starts from int 0 -> exercises __radd__
    assert sum([a, b]) == Counts(tp=11, fp=22, fn=33, tn=44)


def test_summarise_counts_matches_compute_summary_statistics():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(1, 8))
        tp = rng.integers(0, 50, n).tolist()
        fn = rng.integers(0, 50, n).tolist()
        fp = rng.integers(0, 50, n).tolist()
        expected = _compute_summary_statistics(tp=tp, fn=fn, fp=fp)
        per_sequence = [Counts(tp=tp[i], fn=fn[i], fp=fp[i]) for i in range(n)]
        got = summarise_counts(per_sequence).to_dict()
        assert got == expected


def test_summarise_counts_includes_f1():
    # Regression: chain/region tiers used to drop f1 because Stat.f1 stayed None.
    # summarise_counts must now emit a point f1 = 2pr/(p+r) for any counts with
    # both fp and fn present.
    got = summarise_counts([Counts(tp=3, fp=1, fn=1)]).to_dict()
    assert got["f1"] == 0.75  # p = r = 3/4 -> f1 = 3/4

    # Degenerate (no positives) -> f1 present and 0.0, not absent.
    zero = summarise_counts([Counts(tp=0, fp=0, fn=0)]).to_dict()
    assert zero["f1"] == 0.0


def test_transitions_accumulator_sums_matrices_and_counts():
    def fragment(v):
        return {
            "transition_failures": {0: np.array([[v, 0], [0, v]])},
            "false_transitions": {
                "late_catchup": {0: np.array([[v, 0], [0, 0]])},
                "premature": {0: np.zeros((2, 2), dtype=int)},
                "spurious": {0: np.zeros((2, 2), dtype=int)},
                "stable_position_counts": {0: v, 8: 2 * v},
            },
        }

    acc = TransitionsAccumulator()
    acc.add(fragment(1))
    acc.add(fragment(2))
    out = acc.summarise()
    assert (out["transition_failures"][0] == np.array([[3, 0], [0, 3]])).all()
    assert out["false_transitions"]["stable_position_counts"] == {0: 3, 8: 6}


def test_structural_accumulator_sums_and_computes_splice_precision_recall():
    base = dict(
        both_correct=0, donor_only=0, acceptor_only=0, neither=0,
        donor_tp=0, donor_fp=0, donor_fn=0, acceptor_tp=0, acceptor_fp=0, acceptor_fn=0,
        gt_malformed_junctions=0, pred_malformed_junctions=0,
    )

    def fragment(**overrides):
        return {"STRUCTURAL_COHERENCE": {"splice_site_results": {**base, **overrides}}}

    acc = StructuralAccumulator()
    acc.add(fragment(donor_tp=3, donor_fp=1, donor_fn=1))
    acc.add(fragment(donor_tp=1, acceptor_tp=2, acceptor_fn=2))
    out = acc.summarise()["STRUCTURAL_COHERENCE"]["splice_site_results"]
    assert out["donor_tp"] == 4
    assert out["donor_precision"] == 4 / 5
    assert out["donor_recall"] == 4 / 5
    assert out["acceptor_recall"] == 2 / 4


def test_diagnostic_accumulator_sums_histograms_and_summarises_emd():
    def fragment(h, emd):
        return {
            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [1],
                "pred_segment_lengths": [2],
                "length_emd": emd,
                "position_bias_histogram_fn": [h, 0],
                "position_bias_histogram_fp": [0, h],
            }
        }

    acc = DiagnosticDepthAccumulator()
    acc.add(fragment(1, 0.5))
    acc.add(fragment(3, 1.5))
    out = acc.summarise()["DIAGNOSTIC_DEPTH"]
    assert out["position_bias_histogram_fn"] == [4, 0]
    assert out["position_bias_histogram_fp"] == [0, 4]
    assert out["gt_segment_lengths"] == [1, 1]
    assert isinstance(out["length_emd"], dict) and out["length_emd"]["count"] == 2


def test_boundary_accumulator_merged_keeps_raw_summarise_adds_stats():
    def fragment(iou, residuals, total_gt, first, last):
        return {
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": first,
                "last_sec_correct_5_prime_boundary": last,
                "iou_scores": iou,
                "fuzzy_metrics": {"boundary_offsets": residuals, "total_gt": total_gt},
            }
        }

    acc = BoundaryExactnessAccumulator()
    acc.add(fragment([0.5], [(0, 1)], 2, 1, 0))
    acc.add(fragment([0.8], [(1, 0)], 3, 1, 1))

    merged = acc.merged()["BOUNDARY_EXACTNESS"]
    assert merged["iou_scores"] == [0.5, 0.8]
    assert merged["first_sec_correct_3_prime_boundary"] == [1, 1]
    assert merged["fuzzy_metrics"]["total_gt"] == 5

    summarised = acc.summarise()["BOUNDARY_EXACTNESS"]
    assert summarised["iou_scores"] == [0.5, 0.8]  # raw scores kept alongside stats
    assert "iou_stats" in summarised
    # fuzzy_metrics is replaced by the computed landscape (no longer raw residuals).
    assert not isinstance(summarised["fuzzy_metrics"], dict)


def test_benchmark_accumulator_routes_and_ignores_absent_keys():
    acc = BenchmarkAccumulator()
    acc.add({"PHASE_DRIFT": {"gt_frames": [0.0, 1.0], "n_skipped_non_divisible": 0, "n_skipped_short": 1}})
    acc.add({"PHASE_DRIFT": {"gt_frames": [2.0], "n_skipped_non_divisible": 1, "n_skipped_short": 0}})
    assert acc.summarise() == {
        "PHASE_DRIFT": {
            "gt_frames": [0.0, 1.0, 2.0],
            "n_skipped_non_divisible": 1,
            "n_skipped_short": 1,
        }
    }
