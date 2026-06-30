"""Unit tests for per-transcript exon-recovery metrics."""

from __future__ import annotations

import math
import numpy as np
import pytest

from dna_segmentation_benchmark.eval.chain_comparison import (
    _compute_exon_recovery_from_segments,
    _segments_for_scope,
    compute_scoped_chain_metrics,
)
from dna_segmentation_benchmark.eval.evaluate_predictors import (
    EvalMetrics,
    benchmark_gt_vs_pred_single,
)
from dna_segmentation_benchmark.eval import preprocessing
from dna_segmentation_benchmark.eval.statistics import Counts
from dna_segmentation_benchmark.eval.structure import extract_structure
from dna_segmentation_benchmark.label_definition import BEND_LABEL_CONFIG, BenchmarkScope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_array(exons: list[tuple[int, int]], total_length: int) -> np.ndarray:
    """Build a 1-D label array with exons (coding) and introns between them.

    Uses BEND_LABEL_CONFIG: background=8, exon=0, intron=2. Regions outside
    the first/last exon span are painted as background; the gaps between
    consecutive exons are painted as intron so structure extraction yields
    alternating exon/intron runs (matching how the real pipeline operates).
    """
    arr = np.full(total_length, BEND_LABEL_CONFIG.background_label, dtype=np.int64)
    for i, (s, e) in enumerate(exons):
        arr[s:e + 1] = BEND_LABEL_CONFIG.exon_label
        if i + 1 < len(exons):
            next_s = exons[i + 1][0]
            arr[e + 1:next_s] = BEND_LABEL_CONFIG.intron_label
    return arr


def _run(gt_exons: list[tuple[int, int]], pred_exons: list[tuple[int, int]], length: int = 500) -> dict:
    gt_arr = _build_array(gt_exons, length)
    pred_arr = _build_array(pred_exons, length)
    gt_struct = extract_structure(gt_arr, BEND_LABEL_CONFIG)
    pred_struct = extract_structure(pred_arr, BEND_LABEL_CONFIG)
    gt_segs = _segments_for_scope(gt_struct, BenchmarkScope.TRANSCRIPT_EXON, BEND_LABEL_CONFIG)
    pred_segs = _segments_for_scope(pred_struct, BenchmarkScope.TRANSCRIPT_EXON, BEND_LABEL_CONFIG)
    return _compute_exon_recovery_from_segments(gt_segs, pred_segs)


def _chain(gt_exons: list[tuple[int, int]], pred_exons: list[tuple[int, int]], length: int = 500) -> dict:
    """Run the full scoped-chain comparison and return its metric dict."""
    gt_arr = _build_array(gt_exons, length)
    pred_arr = _build_array(pred_exons, length)
    gt_struct = extract_structure(gt_arr, BEND_LABEL_CONFIG)
    pred_struct = extract_structure(pred_arr, BEND_LABEL_CONFIG)
    return compute_scoped_chain_metrics(
        gt_struct, pred_struct, BEND_LABEL_CONFIG, BEND_LABEL_CONFIG.evaluation_scope,
    )


# ---------------------------------------------------------------------------
# End-to-end scenarios
# ---------------------------------------------------------------------------


def test_exact_match_recall_is_one_and_no_false_exons():
    exons = [(10, 19), (30, 49), (70, 89), (110, 129), (150, 179)]
    res = _run(exons, exons)
    assert res["exon_recall_per_transcript"] == 1.0
    assert res["exon_precision_per_transcript"] == 1.0
    assert res["false_exon_count_per_transcript"] == 0


def test_nine_of_ten_exons_right():
    """9 of 10 exons exact — recall and precision 0.9, one false exon."""
    gt = [(10, 19), (30, 49), (70, 89), (110, 129), (150, 179),
          (200, 219), (240, 259), (280, 299), (320, 339), (360, 389)]
    pred = list(gt)
    pred[4] = (150, 181)  # boundary-shift one exon → 1 missed GT, 1 false pred
    res = _run(gt, pred, length=500)

    assert math.isclose(res["exon_recall_per_transcript"], 0.9, rel_tol=1e-9)
    assert math.isclose(res["exon_precision_per_transcript"], 0.9, rel_tol=1e-9)
    assert res["false_exon_count_per_transcript"] == 1


def test_false_exons_without_recall_drop():
    """Model recovers all GT exons but adds two spurious ones."""
    gt = [(10, 19), (30, 49), (70, 89)]
    pred = gt + [(200, 219), (240, 259)]
    res = _run(gt, pred, length=300)

    assert res["exon_recall_per_transcript"] == 1.0
    assert math.isclose(res["exon_precision_per_transcript"], 0.6, rel_tol=1e-9)
    assert res["false_exon_count_per_transcript"] == 2


def test_empty_prediction_scores_zero_recall():
    gt = [(10, 19), (30, 49), (70, 89)]
    pred: list[tuple[int, int]] = []
    res = _run(gt, pred, length=200)
    assert res["exon_recall_per_transcript"] == 0.0
    assert res["exon_precision_per_transcript"] == 0.0
    assert res["false_exon_count_per_transcript"] == 0


def test_empty_ground_truth_returns_empty_dict():
    """Transcripts with no GT exons are not applicable and should be excluded."""
    gt: list[tuple[int, int]] = []
    pred = [(10, 19), (30, 49)]
    res = _run(gt, pred, length=200)
    assert res == {}


def test_single_exon_transcript_is_supported():
    res = _run([(10, 99)], [(10, 99)], length=200)
    assert res["exon_recall_per_transcript"] == 1.0
    assert res["exon_precision_per_transcript"] == 1.0
    assert res["false_exon_count_per_transcript"] == 0


def test_boundary_shift_not_contaminated_by_equal_count_substitution():
    """An equal-count pair with a relocated middle exon is NOT a boundary shift.

    The pairwise-overlap guard must keep the relocated exon out of the
    boundary-shift distribution (otherwise its large offset would pollute the
    plot and the metric).
    """
    gt = [(10, 19), (100, 119), (200, 219)]
    pred = [(10, 19), (150, 169), (200, 219)]  # middle exon relocated, no overlap
    res = _chain(gt, pred)

    assert res["boundary_shift_count"] == 0
    assert res["boundary_shift_total"] == 0
    assert res["transcript_match_class"] == "partial_overlap"


def test_boundary_shift_counted_for_genuine_internal_shift():
    """A true internal splice shift (all pairs overlap) is counted."""
    gt = [(10, 19), (100, 119), (200, 219)]
    pred = [(10, 19), (104, 119), (200, 219)]  # middle exon start shifted, still overlaps
    res = _chain(gt, pred)

    assert res["boundary_shift_count"] == 1
    assert res["boundary_shift_total"] == 4
    assert res["transcript_match_class"] == "boundary_shift_internal"


def test_intron_chain_fails_when_introns_are_inferable_but_missing():
    labels = np.array([8, 0, 0, 8, 8, 0, 0, 8, 8, 0, 0, 8])

    with pytest.raises(ValueError, match="infer_introns=True"):
        benchmark_gt_vs_pred_single(
            gt_labels=labels,
            pred_labels=labels,
            label_config=BEND_LABEL_CONFIG,
            metrics=[EvalMetrics.STRUCTURAL_COHERENCE],
        )


def test_infer_introns_fills_gaps_before_structural_metrics():
    labels = np.array([8, 0, 0, 8, 8, 0, 0, 8, 8, 0, 0, 8])

    result = benchmark_gt_vs_pred_single(
        gt_labels=labels,
        pred_labels=labels,
        label_config=BEND_LABEL_CONFIG,
        metrics=[EvalMetrics.STRUCTURAL_COHERENCE],
        infer_introns=True,
    )

    assert result["STRUCTURAL_COHERENCE"]["intron_chain"] == Counts(tp=1, fp=0, fn=0)


def test_infer_introns_warns_on_large_arrays(monkeypatch):
    monkeypatch.setattr(
        preprocessing,
        "_INFER_INTRONS_LARGE_ARRAY_WARNING_LENGTH",
        10,
    )
    labels = np.array([8, 0, 0, 8, 8, 0, 0, 8, 8, 0, 0, 8])

    with pytest.warns(UserWarning, match="large input array"):
        benchmark_gt_vs_pred_single(
            gt_labels=labels,
            pred_labels=labels,
            label_config=BEND_LABEL_CONFIG,
            metrics=[EvalMetrics.STRUCTURAL_COHERENCE],
            infer_introns=True,
        )


def test_large_array_intron_inference_skips_comparatively_large_gaps(monkeypatch):
    monkeypatch.setattr(
        preprocessing,
        "_INFER_INTRONS_LARGE_ARRAY_WARNING_LENGTH",
        10,
    )
    labels = np.full(104, BEND_LABEL_CONFIG.background_label, dtype=np.int64)
    labels[1:3] = BEND_LABEL_CONFIG.exon_label
    labels[5:7] = BEND_LABEL_CONFIG.exon_label
    labels[100:102] = BEND_LABEL_CONFIG.exon_label

    with pytest.warns(UserWarning, match="large input array"):
        inferred = preprocessing._infer_introns_from_coding_gaps(
            labels,
            BEND_LABEL_CONFIG,
        )

    assert (inferred[3:5] == BEND_LABEL_CONFIG.intron_label).all()
    assert (inferred[7:100] == BEND_LABEL_CONFIG.background_label).all()


def test_large_array_gap_cutoff_uses_bimodal_jump_before_second_mode():
    cutoff = preprocessing._large_array_inferable_gap_cutoff(
        [8, 10, 12, 14, 2_000, 2_300, 2_700],
    )

    assert 14 < cutoff < 2_000


def test_large_array_gap_cutoff_falls_back_without_clear_mode_split():
    cutoff = preprocessing._large_array_inferable_gap_cutoff(
        [8, 10, 12, 14, 16, 18, 20],
    )

    assert cutoff == 200


# ---------------------------------------------------------------------------
# Single-exon vs multi-exon exon-chain split
# ---------------------------------------------------------------------------


def test_single_exon_routes_to_single_bucket_only():
    """A single-exon transcript scores exon_chain_single; multi buckets stay empty."""
    res = _chain([(10, 49)], [(10, 49)])
    assert res["exon_chain_single"] == res["exon_chain"]
    assert res["exon_chain_multi"] == Counts()
    assert res["exon_chain_multi_subset"] == Counts()
    assert res["exon_chain_multi_superset"] == Counts()


def test_multi_exon_routes_to_multi_buckets_only():
    """A multi-exon transcript scores the exon_chain_multi tiers; single stays empty."""
    gt = [(10, 19), (40, 59), (90, 109)]
    pred = [(10, 19), (40, 61), (90, 109)]  # one shifted exon -> chain mismatch
    res = _chain(gt, pred)
    assert res["exon_chain_multi"] == res["exon_chain"]
    assert res["exon_chain_multi_subset"] == res["exon_chain_subset"]
    assert res["exon_chain_multi_superset"] == res["exon_chain_superset"]
    assert res["exon_chain_single"] == Counts()


@pytest.mark.parametrize(
    "gt, pred",
    [
        ([(10, 49)], [(10, 49)]),                              # single, match
        ([(10, 49)], [(10, 51)]),                              # single, mismatch
        ([(10, 19), (40, 59), (90, 109)], [(10, 19), (40, 59), (90, 109)]),  # multi, match
        ([(10, 19), (40, 59), (90, 109)], [(10, 19), (40, 61), (90, 109)]),  # multi, mismatch
    ],
)
def test_single_plus_multi_reconstructs_exon_chain(gt, pred):
    """exon_chain_single + exon_chain_multi == exon_chain (exact tier) per pair."""
    res = _chain(gt, pred)
    assert res["exon_chain_single"] + res["exon_chain_multi"] == res["exon_chain"]


def test_combined_single_and_multi_exon_buckets_do_not_bleed():
    """Accumulated counts: single-exon hit + multi-exon miss stay in separate buckets."""
    # single-exon, exact match → exon_chain_single tp=1, exon_chain_multi empty
    single_hit = _chain([(10, 49)], [(10, 49)])
    # multi-exon, shifted boundary → exon_chain_multi fp=1/fn=1, exon_chain_single empty
    multi_miss = _chain([(10, 19), (40, 59)], [(10, 19), (40, 61)])

    combined_single = single_hit["exon_chain_single"] + multi_miss["exon_chain_single"]
    combined_multi = single_hit["exon_chain_multi"] + multi_miss["exon_chain_multi"]
    combined_total = single_hit["exon_chain"] + multi_miss["exon_chain"]

    # buckets are disjoint: only single contributed a tp, only multi contributed fp/fn
    assert combined_single == Counts(tp=1)
    assert combined_multi == Counts(fp=1, fn=1)
    # additive: the two buckets reconstruct the overall chain count
    assert combined_single + combined_multi == combined_total
