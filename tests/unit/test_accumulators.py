"""Tests for the typed accumulators (Counts / Stat) and metric accumulators."""

import math

import numpy as np
import pytest

from dna_segmentation_benchmark.eval.accumulators import (
    BenchmarkAccumulator,
    BoundaryExactnessAccumulator,
    DiagnosticDepthAccumulator,
    IndelAccumulator,
    StructuralAccumulator,
    TransitionsAccumulator,
)
from dna_segmentation_benchmark.eval.statistics import (
    Counts,
    summarise_counts,
)


def test_counts_add_and_sum():
    a = Counts(tp=1, fp=2, fn=3, tn=4)
    b = Counts(tp=10, fp=20, fn=30, tn=40)
    assert a + b == Counts(tp=11, fp=22, fn=33, tn=44)
    # sum() starts from int 0 -> exercises __radd__
    assert sum([a, b]) == Counts(tp=11, fp=22, fn=33, tn=44)


def test_summarise_counts_matches_hand_computed_values():
    counts = [Counts(tp=10, fp=5, fn=0), Counts(tp=20, fp=0, fn=10)]
    got = summarise_counts(counts).to_dict()

    # Micro: pool tp/fp/fn first, then divide.
    #   precision = 30/(30+5) = 6/7 ; recall = 30/(30+10) = 3/4
    #   f1 = 2pr/(p+r) = 0.8 exactly
    assert got == pytest.approx({
        "precision": 6 / 7,
        "recall": 0.75,
        "f1": 0.8,
        # *_stderr are bootstrap outputs — not hand-derivable, but deterministic
        # under the fixed seed (default_rng(42)); pinned so a regression shows.
        "precision_stderr": 0.11911887391362035,
        "recall_stderr": 0.1255542158237274,
        "f1_stderr": 1.7490858178089023e-16,
    })


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
    # EMD values fed in are [0.5, 1.5]; hand-derived distribution stats (both
    # positive so mae == mean; rmse = sqrt(mean([0.25, 2.25])) = sqrt(1.25)):
    assert out["length_emd"] == pytest.approx({
        "count": 2, "mean": 1.0, "mae": 1.0,
        "rmse": math.sqrt(1.25), "std": 0.5, "min": 0.5, "max": 1.5,
    })


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
    # fuzzy_metrics is replaced by the computed landscape dict (no longer raw
    # residuals) — a JSON-serialisable {max_range, bias_matrix, reliability_matrix}.
    landscape = summarised["fuzzy_metrics"]
    assert "boundary_offsets" not in landscape

    # Residuals merged in: (5', 3') offsets (0, 1) and (1, 0); total_gt = 2+3 = 5,
    # max_range = 10 (accumulator default). Hand-derived from
    # _compute_boundary_precision_landscape:
    #   bias_matrix is (2*max_range+1)²; each residual increments
    #   [5'_offset+10, 3'_offset+10].
    #   reliability_matrix[t5, t3] = (residuals with |5'|<=t5 and |3'|<=t3) / total_gt.
    assert landscape["max_range"] == 10
    expected_bias = np.zeros((21, 21))
    expected_bias[10, 11] = 1.0  # residual (0, 1)
    expected_bias[11, 10] = 1.0  # residual (1, 0)
    np.testing.assert_array_equal(np.array(landscape["bias_matrix"]), expected_bias)

    expected_reliability = np.zeros((11, 11))
    expected_reliability[0, 1:] = 1 / 5   # only (0,1) qualifies once t3>=1
    expected_reliability[1:, 0] = 1 / 5   # only (1,0) qualifies at t3=0
    expected_reliability[1:, 1:] = 2 / 5  # both qualify once t5>=1 and t3>=1
    np.testing.assert_allclose(np.array(landscape["reliability_matrix"]), expected_reliability)


def test_boundary_landscape_clips_out_of_range_and_reports_sidedness():
    """Residuals beyond ±max_range saturate into the edge bins (B) instead of being
    silently dropped by histogram2d, and the one-sidedness scalar (C) is computed from
    the raw residuals, clip-free."""
    acc = BoundaryExactnessAccumulator()
    acc.add(
        {
            "BOUNDARY_EXACTNESS": {
                "first_sec_correct_3_prime_boundary": 0,
                "last_sec_correct_5_prime_boundary": 0,
                "iou_scores": [0.1, 0.2],
                # (0, 500): only the 3' edge moves, far out of window.
                # (300, 200): both edges move, far out of window.
                "fuzzy_metrics": {"boundary_offsets": [(0, 500), (300, 200)], "total_gt": 2},
            }
        }
    )
    landscape = acc.summarise()["BOUNDARY_EXACTNESS"]["fuzzy_metrics"]

    # B: both out-of-range pairs are clipped into the edge bins, none dropped.
    bias = np.array(landscape["bias_matrix"])
    assert bias.sum() == 2  # every matched pair retained, not silently discarded
    assert bias[10, 20] == 1.0  # (0, 500) -> (0, +10): row 5'=0, col 3'=+10 edge
    assert bias[20, 20] == 1.0  # (300, 200) -> (+10, +10) corner

    # C: sidedness counted from RAW residuals, independent of the ±10 clip.
    assert landscape["sidedness"] == {
        "total": 2,
        "exact": 0,
        "one_sided": 1,      # (0, 500) — only one edge off
        "two_sided": 1,      # (300, 200) — both edges off
        "one_sided_fraction": 0.5,
        "clipped_from_bias_matrix": 2,
    }


def test_event_free_chunk_still_contributes_its_denominators():
    """A chunk whose payload fired no *event* must still hand over its counters.

    The parallel path calls ``merged()`` once per chunk and feeds the result to
    ``merge_from_merged()``.  Guarding ``merged()`` on an event field (indel runs,
    GT segments, transition failures) made an event-free chunk return ``{}``, so its
    denominators were silently dropped and the totals depended on how the mappings
    happened to be chunked.  Each accumulator below gets one event-free fragment and
    must still report its counts.
    """
    # INDEL: no by_boundary runs, but real opportunity denominators.
    indel = IndelAccumulator()
    indel.add({"INDEL": {"by_boundary": {}, "exon_opportunities": {"internal_exon": 7},
                         "n_gt_segments": 5, "n_pred_segments": 4}})
    sink = IndelAccumulator()
    sink.merge_from_merged(indel.merged())
    assert sink.merged()["INDEL"]["n_gt_segments"] == 5
    assert sink.merged()["INDEL"]["n_pred_segments"] == 4
    assert sink.merged()["INDEL"]["exon_opportunities"] == {"internal_exon": 7}

    # DIAGNOSTIC_DEPTH: an unmatched prediction has no GT segments, but its predicted
    # lengths and position-bias counts are real.
    diag = DiagnosticDepthAccumulator()
    diag.add({"DIAGNOSTIC_DEPTH": {"gt_segment_lengths": [], "pred_segment_lengths": [9],
                                   "length_emd": 2.0, "position_bias_histogram_fn": [1, 0],
                                   "position_bias_histogram_fp": [0, 1]}})
    dsink = DiagnosticDepthAccumulator()
    dsink.merge_from_merged(diag.merged())
    out = dsink.merged()["DIAGNOSTIC_DEPTH"]
    assert out["pred_segment_lengths"] == [9]
    assert out["length_emd"] == [2.0]
    assert out["position_bias_histogram_fp"] == [0, 1]

    # STATE_TRANSITIONS: a chunk with no failures still has stable positions.
    trans = TransitionsAccumulator()
    trans.add({"transition_failures": {},
               "false_transitions": {"late_catchup": {}, "premature": {}, "spurious": {},
                                     "stable_position_counts": {"exon": 3}}})
    tsink = TransitionsAccumulator()
    tsink.merge_from_merged(trans.merged())
    assert tsink.merged()["false_transitions"]["stable_position_counts"] == {"exon": 3}


def test_benchmark_accumulator_routes_and_ignores_absent_keys():
    acc = BenchmarkAccumulator()
    acc.add({"PHASE_DRIFT": {"gt_frames": [0.0, 1.0], "n_skipped_non_divisible": 0, "n_skipped_short": 1}})
    acc.add({"PHASE_DRIFT": {"gt_frames": [2.0], "n_skipped_non_divisible": 1, "n_skipped_short": 0}})
    acc.add({"PHASE_DRIFT": {"gt_frames": [], "n_skipped_no_overlap": 1}})
    # Per-position frames are binned to a [in-phase, +1, +2] count on the way in.
    assert acc.summarise() == {
        "PHASE_DRIFT": {
            "gt_frame_counts": [1, 1, 1],
            "n_skipped_non_divisible": 1,
            "n_skipped_short": 1,
            "n_skipped_no_overlap": 1,
        }
    }
