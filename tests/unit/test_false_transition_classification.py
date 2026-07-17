"""Acceptance tests for the run-anchored false-transition classification.

Regression suite for the bug where a fabrication (a round-trip excursion inside a
GT-stable run) was booked as two boundary slips (`premature` + `late_catchup`)
instead of two `spurious` transitions, and where the classification therefore
depended on the *flanking* GT labels rather than on the error itself.

Uses the production label config of the thesis real-data runs
(``AnnotationMode.UTR_CDS_INTRON``, ``BenchmarkScope.CDS``).
"""

import numpy as np
import pytest

from gene_calling_benchmark.eval.state_transitions import compute_state_change_errors
from gene_calling_benchmark.label_definition import (
    AnnotationMode,
    BenchmarkScope,
    LabelConfig,
)

BG, CDS, INT, U5, U3 = 1, 2, 3, 4, 5

LABEL_CONFIG = LabelConfig(
    annotation_mode=AnnotationMode.UTR_CDS_INTRON,
    evaluation_scope=BenchmarkScope.CDS,
    background_label=BG,
    cds_label=CDS,
    intron_label=INT,
    five_prime_utr_label=U5,
    three_prime_utr_label=U3,
)


def classify(gt: list[int], pred: list[int]) -> tuple[int, int, int]:
    """(premature, late_catchup, spurious) totals for one gt/pred pair."""
    result = compute_state_change_errors(np.array([gt, pred]), LABEL_CONFIG)

    def total(matrices: dict[int, np.ndarray]) -> int:
        return int(sum(m.sum() for m in matrices.values()))

    return (
        total(result.premature_matrices),
        total(result.late_catchup_matrices),
        total(result.spurious_matrices),
    )


GT_INTERNAL_EXON = [INT] * 30 + [CDS] * 60 + [INT] * 30


def test_1_invented_intron_in_internal_exon_is_fabrication():
    pred = [INT] * 30 + [CDS] * 20 + [INT] * 20 + [CDS] * 20 + [INT] * 30
    assert classify(GT_INTERNAL_EXON, pred) == (0, 0, 2)


def test_2_fabricated_exon_inside_an_intron():
    gt = [CDS] * 30 + [INT] * 60 + [CDS] * 30
    pred = [CDS] * 30 + [INT] * 20 + [CDS] * 20 + [INT] * 20 + [CDS] * 30
    assert classify(gt, pred) == (0, 0, 2)


@pytest.mark.parametrize(
    "left, right",
    [
        pytest.param(INT, INT, id="internal_exon"),
        pytest.param(U5, INT, id="first_cds_exon"),
        pytest.param(INT, U3, id="last_cds_exon"),
        pytest.param(U5, U3, id="single_exon_gene"),
    ],
)
def test_3_flank_invariance(left, right):
    """REGRESSION TEST THAT PINS THE BUG.

    The same invented intron scored 0, 1 or 2 spurious depending purely on the
    flanking GT labels (internal / terminal / single exon), because the old
    predicates matched `CDS->INT` against `next_GT` and `INT->CDS` against
    `prev_GT`. The classification must depend on the error, not on where in the
    gene it happens to fall.
    """
    gt = [left] * 30 + [CDS] * 60 + [right] * 30
    pred = [left] * 30 + [CDS] * 20 + [INT] * 20 + [CDS] * 20 + [right] * 30
    assert classify(gt, pred) == (0, 0, 2)


def test_4_true_premature_exit_stays_out():
    pred = [INT] * 30 + [CDS] * 50 + [INT] * 40
    assert classify(GT_INTERNAL_EXON, pred) == (1, 0, 0)


def test_5_true_late_catchup():
    pred = [INT] * 40 + [CDS] * 50 + [INT] * 30
    assert classify(GT_INTERNAL_EXON, pred) == (0, 1, 0)


def test_6_both_ends_slipped():
    pred = [INT] * 40 + [CDS] * 40 + [INT] * 40
    assert classify(GT_INTERNAL_EXON, pred) == (1, 1, 0)


def test_7_perfect_prediction():
    assert classify(GT_INTERNAL_EXON, GT_INTERNAL_EXON) == (0, 0, 0)


def test_8_two_invented_introns_in_one_exon():
    pred = (
        [INT] * 30
        + [CDS] * 10 + [INT] * 10 + [CDS] * 10 + [INT] * 10 + [CDS] * 20
        + [INT] * 30
    )
    assert len(pred) == len(GT_INTERNAL_EXON)
    assert classify(GT_INTERNAL_EXON, pred) == (0, 0, 4)


def test_9_missed_exon_is_not_a_transition_error():
    """A missed exon is a detection failure: no predicted transition occurs."""
    pred = [INT] * 120
    assert classify(GT_INTERNAL_EXON, pred) == (0, 0, 0)


def test_10_fabricated_exon_in_intergenic_space():
    gt = [BG] * 120
    pred = [BG] * 40 + [CDS] * 40 + [BG] * 40
    assert classify(gt, pred) == (0, 0, 2)


def test_total_false_transitions_are_conserved():
    """The fix only re-routes between buckets: the total must not change.

    Total = (GT-stable windows where pred changes) + (off-track pred changes at a
    GT boundary).
    """
    rng = np.random.default_rng(11)
    labels = np.array([BG, CDS, INT, U5, U3])
    for _ in range(20):
        gt_pred = rng.choice(labels, size=(2, 400))
        gt, pred = gt_pred[0], gt_pred[1]

        gt_stable = gt[:-1] == gt[1:]
        pred_changes = pred[:-1] != pred[1:]
        off_track = (~gt_stable) & (pred[:-1] != gt[:-1]) & pred_changes
        expected = int((gt_stable & pred_changes).sum() + off_track.sum())

        assert sum(classify(list(gt), list(pred))) == expected
