"""Unit tests for per-transcript soft exon metrics."""

from __future__ import annotations

import math

import numpy as np

from dna_segmentation_benchmark.label_definition import BEND_LABEL_CONFIG
from dna_segmentation_benchmark.eval.chain_comparison import (
    _compute_per_transcript_exon_soft_metrics,
)
from dna_segmentation_benchmark.eval.structure import extract_structure


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
    return _compute_per_transcript_exon_soft_metrics(gt_struct, pred_struct, BEND_LABEL_CONFIG)


# ---------------------------------------------------------------------------
# End-to-end scenarios
# ---------------------------------------------------------------------------


def test_exact_match_recall_is_one_and_no_hallucinations():
    exons = [(10, 19), (30, 49), (70, 89), (110, 129), (150, 179)]
    res = _run(exons, exons)
    assert res["exon_recall_per_transcript"] == 1.0
    assert res["hallucinated_exon_count_per_transcript"] == 0


def test_nine_of_ten_exons_right():
    """9 of 10 exons exact — recall should be 0.9 and there should be 1 hallucination."""
    gt = [(10, 19), (30, 49), (70, 89), (110, 129), (150, 179),
          (200, 219), (240, 259), (280, 299), (320, 339), (360, 389)]
    pred = list(gt)
    pred[4] = (150, 181)  # boundary-shift one exon → 1 missed GT, 1 hallucinated pred
    res = _run(gt, pred, length=500)

    assert math.isclose(res["exon_recall_per_transcript"], 0.9, rel_tol=1e-9)
    assert res["hallucinated_exon_count_per_transcript"] == 1


def test_hallucinations_without_recall_drop():
    """Model recovers all GT exons but adds two spurious ones."""
    gt = [(10, 19), (30, 49), (70, 89)]
    pred = gt + [(200, 219), (240, 259)]
    res = _run(gt, pred, length=300)

    assert res["exon_recall_per_transcript"] == 1.0
    assert res["hallucinated_exon_count_per_transcript"] == 2


def test_empty_prediction_scores_zero_recall():
    gt = [(10, 19), (30, 49), (70, 89)]
    pred: list[tuple[int, int]] = []
    res = _run(gt, pred, length=200)
    assert res["exon_recall_per_transcript"] == 0.0
    assert res["hallucinated_exon_count_per_transcript"] == 0


def test_empty_ground_truth_returns_empty_dict():
    """Transcripts with no GT exons are not applicable and should be excluded."""
    gt: list[tuple[int, int]] = []
    pred = [(10, 19), (30, 49)]
    res = _run(gt, pred, length=200)
    assert res == {}


def test_single_exon_transcript_is_supported():
    res = _run([(10, 99)], [(10, 99)], length=200)
    assert res["exon_recall_per_transcript"] == 1.0
    assert res["hallucinated_exon_count_per_transcript"] == 0
