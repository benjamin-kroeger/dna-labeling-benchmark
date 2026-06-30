import numpy as np
import pytest

from dna_segmentation_benchmark.eval.state_transitions import (
    TransitionAnalysis,
    compute_state_change_errors,
)
from dna_segmentation_benchmark.label_definition import AnnotationMode, BEND_LABEL_CONFIG, LabelConfig

from support.constants import ACCEPTOR, DONOR, EXON, INTRON, NONCODING

_LABEL_IDS = [EXON, DONOR, INTRON, ACCEPTOR, NONCODING]
_L = len(_LABEL_IDS)
_IDX = {lid: i for i, lid in enumerate(_LABEL_IDS)}  # label → matrix row/col index


# ------------------------------------------------------------------
# Helpers for building expected TransitionAnalysis objects
# ------------------------------------------------------------------
def _sparse(entries: dict[tuple[int, int], int]) -> np.ndarray:
    """Sparse (_L, _L) matrix: keys are (label_id_row, label_id_col) → value."""
    m = np.zeros((_L, _L), dtype=np.int64)
    for (row_lid, col_lid), val in entries.items():
        m[_IDX[row_lid], _IDX[col_lid]] = val
    return m


def _zeros() -> dict[int, np.ndarray]:
    return {lid: np.zeros((_L, _L), dtype=np.int64) for lid in _LABEL_IDS}


def _mk(
    gt:     dict[int, np.ndarray] | None = None,
    lc:     dict[int, np.ndarray] | None = None,
    pm:     dict[int, np.ndarray] | None = None,
    sp:     dict[int, np.ndarray] | None = None,
    stable: dict[int, int] | None = None,
) -> TransitionAnalysis:
    """Build a TransitionAnalysis with all-zero matrices, overriding specific labels."""
    def _merge(base: dict, overrides: dict | None) -> dict:
        if overrides:
            base.update(overrides)
        return base

    return TransitionAnalysis(
        gt_transition_matrices=_merge(_zeros(), gt),
        late_catchup_matrices=_merge(_zeros(), lc),
        premature_matrices=_merge(_zeros(), pm),
        spurious_matrices=_merge(_zeros(), sp),
        stable_position_counts={**{lid: 0 for lid in _LABEL_IDS}, **(stable or {})},
    )


# ------------------------------------------------------------------
# Test cases
# ------------------------------------------------------------------
STATE_TRANSITION_TEST_CASES = [
    pytest.param(
        # GT:   [8, 8, 8, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 8, 8]
        # Pred: [8, 8, 8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8]
        #
        # GT transitions (window i): 2 (NC→EX), 6 (EX→IN), 10 (IN→EX), 15 (EX→NC)
        # GT transition matrix events:
        #   i=2:  src=NC, GT→EX, pred_tgt=NC → FAILED  (gt_trans[NC][EX, NC]=1)
        #   i=6:  src=EX, GT→IN, pred_tgt=IN → CORRECT (gt_trans[EX][IN, IN]=1)
        #   i=15: src=EX, GT→NC, pred_tgt=NC → CORRECT (gt_trans[EX][NC, NC]=1)
        # False transitions with context:
        #   i=3:  NC→EX in stable EX,  prev=NC, next=IN → LATE_CATCHUP  (lc[EX][NC,EX]=1)
        #   i=8:  IN→EX in stable IN,  prev=EX, next=EX → PREMATURE     (pm[IN][IN,EX]=1)
        #   i=11: EX→IN in stable EX,  prev=IN, next=NC → SPURIOUS      (sp[EX][EX,IN]=1)
        #   i=13: IN→EX in stable EX,  prev=IN, next=NC → LATE_CATCHUP  (lc[EX][IN,EX]=1)
        np.array(
            [
                [8, 8, 8, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 8, 8],
                [8, 8, 8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8],
            ],
            dtype=np.int64,
        ),
        BEND_LABEL_CONFIG,
        _mk(
            gt={
                EXON:     _sparse({(INTRON, INTRON): 1, (NONCODING, NONCODING): 1}),
                NONCODING: _sparse({(EXON, NONCODING): 1}),
            },
            lc={
                EXON: _sparse({(NONCODING, EXON): 1, (INTRON, EXON): 1}),
            },
            pm={
                INTRON: _sparse({(INTRON, EXON): 1}),
            },
            sp={
                EXON: _sparse({(EXON, INTRON): 1}),
            },
            stable={EXON: 7, INTRON: 3, NONCODING: 3},
        ),
        id="detailed_mixed_sequence",
    ),
    pytest.param(
        # GT:   [8, 0, 0, 0, 0, 0, 0, 0, 8]
        # Pred: [8, 0, 2, 0, 2, 0, 2, 0, 8]  ← pred oscillates EX↔IN inside stable EX
        #
        # Both boundaries are NONCODING (prev_GT=NC, next_GT=NC for every window).
        # Pred only ever transitions between EX and IN — neither endpoint matches NC.
        # → ALL 6 false transitions are SPURIOUS; lc=0, pm=0.
        #
        # GT boundaries: i=0 (NC→EX, CORRECT), i=7 (EX→NC, CORRECT)
        np.array(
            [
                [8, 0, 0, 0, 0, 0, 0, 0, 8],
                [8, 0, 2, 0, 2, 0, 2, 0, 8],
            ],
            dtype=np.int64,
        ),
        BEND_LABEL_CONFIG,
        _mk(
            gt={
                NONCODING: _sparse({(EXON, EXON): 1}),       # i=0: NC→EX, pred→EX CORRECT
                EXON:      _sparse({(NONCODING, NONCODING): 1}),  # i=7: EX→NC, pred→NC CORRECT
            },
            sp={
                EXON: _sparse({(EXON, INTRON): 3, (INTRON, EXON): 3}),  # 3× each direction
            },
            stable={EXON: 6},
        ),
        id="switching_in_out_all_spurious",
    ),
    pytest.param(
        # GT:   [8, 8, 0, 0]
        # Pred: [8, 0, 0, 0]  ← pred NC→EX at i=0 while GT is stable NC
        #                        next_GT=EX → PREMATURE (leaves for the correct next state)
        #   At i=1 (GT boundary NC→EX): pred_src=EX != gt_src=NC → excluded from gt_transition_matrices
        np.array([[8, 8, 0, 0], [8, 0, 0, 0]], dtype=np.int64),
        BEND_LABEL_CONFIG,
        _mk(
            pm={NONCODING: _sparse({(NONCODING, EXON): 1})},
            stable={EXON: 1, NONCODING: 1},
        ),
        id="premature_pred_excluded_from_gt_transitions",
    ),
    pytest.param(
        # GT:   [8, 8, 0, 0]
        # Pred: [8, 8, 2, 0]  ← pred NC→NC at boundary (i=1): pred_src==gt_src=NC, pred_tgt=IN → FAILED GT (not NC→EX)
        #                        pred IN→EX in stable EX (i=2): pred_src=IN != prev_GT=NC → SPURIOUS (not late catch-up)
        np.array([[8, 8, 0, 0], [8, 8, 2, 0]], dtype=np.int64),
        BEND_LABEL_CONFIG,
        _mk(
            gt={NONCODING: _sparse({(EXON, INTRON): 1})},
            sp={EXON:      _sparse({(INTRON, EXON): 1})},
            stable={EXON: 1, NONCODING: 1},
        ),
        id="spurious_not_late_catchup_when_wrong_source",
    ),
    pytest.param(
        # GT:   [0, 0, 8, 8]
        # Pred: [0, 2, 8, 8]  ← pred EX→IN in stable EX (i=0): pred_tgt=IN != next_GT=NC → SPURIOUS (not premature)
        #                        off-track pred IN→NC at boundary (i=1): pred_src=IN != gt_src=EX → SPURIOUS
        np.array([[0, 0, 8, 8], [0, 2, 8, 8]], dtype=np.int64),
        BEND_LABEL_CONFIG,
        _mk(
            sp={EXON: _sparse({(EXON, INTRON): 1, (INTRON, NONCODING): 1})},
            stable={EXON: 1, NONCODING: 1},
        ),
        id="spurious_not_premature_when_wrong_target",
    ),
    pytest.param(
        # GT:   [8, 8, 0, 0]
        # Pred: [8, 2, 0, 0]  ← pred NC→IN at i=0 (stable NC, next_GT=EX): pred_tgt=IN != next_GT → SPURIOUS
        #                        off-track pred IN→EX at boundary (i=1): pred_src=IN != gt_src=NC → SPURIOUS
        np.array([[8, 8, 0, 0], [8, 2, 0, 0]], dtype=np.int64),
        BEND_LABEL_CONFIG,
        _mk(
            sp={NONCODING: _sparse({(NONCODING, INTRON): 1, (INTRON, EXON): 1})},
            stable={EXON: 1, NONCODING: 1},
        ),
        id="off_track_boundary_transition_is_spurious",
    ),
]


# ------------------------------------------------------------------
# Reference implementation (loop-based, easy to audit)
# ------------------------------------------------------------------
def _compute_state_change_errors_reference(
        gt_pred_arr: np.ndarray,
        label_config: LabelConfig,
) -> TransitionAnalysis:
    """Reference: simple Python loop, easy to audit against the spec.

    False transitions classified with lookbehind / lookahead:
    - Late catch-up: pred_src == prev_GT AND pred_tgt == curr_GT
    - Premature:     pred_src == curr_GT AND pred_tgt == next_GT
    - Spurious:      everything else (incl. off-track at GT boundaries)

    prev_GT / next_GT: GT label of the run before / after the current stable
    window.  Sentinel = curr_GT when no such run exists (prevents false match).
    """
    label_ids = sorted(label_config.labels.keys())
    num_labels = len(label_ids)
    gt_vals = gt_pred_arr[0]
    pred_vals = gt_pred_arr[1]
    N = len(gt_vals)

    gt_mats   = {lid: np.zeros((num_labels, num_labels), dtype=np.int64) for lid in label_ids}
    lc_mats   = {lid: np.zeros((num_labels, num_labels), dtype=np.int64) for lid in label_ids}
    pm_mats   = {lid: np.zeros((num_labels, num_labels), dtype=np.int64) for lid in label_ids}
    sp_mats   = {lid: np.zeros((num_labels, num_labels), dtype=np.int64) for lid in label_ids}
    stable    = {lid: 0 for lid in label_ids}

    gt_trans_pos = [i for i in range(N - 1) if gt_vals[i] != gt_vals[i + 1]]

    def prev_gt(i: int) -> int:
        cands = [p for p in gt_trans_pos if p < i]
        return int(gt_vals[cands[-1]]) if cands else int(gt_vals[i])

    def next_gt(i: int) -> int:
        cands = [p for p in gt_trans_pos if p >= i]
        return int(gt_vals[cands[0] + 1]) if cands else int(gt_vals[i])

    for i in range(N - 1):
        gs, gt_, ps, pt = int(gt_vals[i]), int(gt_vals[i + 1]), int(pred_vals[i]), int(pred_vals[i + 1])
        gs_i, gt_i, ps_i, pt_i = (label_ids.index(x) for x in (gs, gt_, ps, pt))

        if gs == gt_:
            stable[gs] += 1
            if ps != pt:
                prv, nxt = prev_gt(i), next_gt(i)
                if ps == prv and pt == gs:
                    lc_mats[gs][ps_i, pt_i] += 1
                elif ps == gs and pt == nxt:
                    pm_mats[gs][ps_i, pt_i] += 1
                else:
                    sp_mats[gs][ps_i, pt_i] += 1
        else:
            if ps == gs:
                gt_mats[gs][gt_i, pt_i] += 1
            elif ps != pt:
                sp_mats[gs][ps_i, pt_i] += 1

    return TransitionAnalysis(
        gt_transition_matrices=gt_mats,
        late_catchup_matrices=lc_mats,
        premature_matrices=pm_mats,
        spurious_matrices=sp_mats,
        stable_position_counts=stable,
    )


def _assert_equal(expected: TransitionAnalysis, actual: TransitionAnalysis) -> None:
    assert expected.stable_position_counts == actual.stable_position_counts
    for field in ("gt_transition_matrices", "late_catchup_matrices", "premature_matrices", "spurious_matrices"):
        exp_d = getattr(expected, field)
        act_d = getattr(actual, field)
        assert exp_d.keys() == act_d.keys(), f"{field}: key mismatch"
        for lid in exp_d:
            np.testing.assert_array_equal(exp_d[lid], act_d[lid], err_msg=f"{field}[{lid}]")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
@pytest.mark.parametrize("gt_pred_arr, label_config, expected", STATE_TRANSITION_TEST_CASES)
def test_compute_state_change_errors(gt_pred_arr, label_config, expected):
    _assert_equal(expected, compute_state_change_errors(gt_pred_arr, label_config))


def test_compute_state_change_errors_matches_reference_implementation():
    label_config = LabelConfig(
        annotation_mode=AnnotationMode.EXON_INTRON,
        background_label=5,
        exon_label=0,
        intron_label=9,
        splice_donor_label=2,
        splice_acceptor_label=7,
    )
    rng = np.random.default_rng(7)
    label_values = np.array(sorted(label_config.labels.keys()), dtype=np.int64)

    gt_pred_arr = np.stack(
        [
            rng.choice(label_values, size=256, replace=True),
            rng.choice(label_values, size=256, replace=True),
        ],
        axis=0,
    )

    _assert_equal(
        _compute_state_change_errors_reference(gt_pred_arr, label_config),
        compute_state_change_errors(gt_pred_arr, label_config),
    )


# ------------------------------------------------------------------
# Opt-in gating: STATE_TRANSITIONS controls whether the transition
# fragments are emitted by the benchmark entry points.
# ------------------------------------------------------------------
def test_state_transitions_emitted_only_when_requested():
    from dna_segmentation_benchmark.eval.evaluate_predictors import (
        EvalMetrics,
        benchmark_gt_vs_pred_single,
    )

    gt = np.array([EXON, EXON, DONOR, INTRON, ACCEPTOR, EXON, NONCODING, NONCODING])
    pred = np.array([EXON, DONOR, INTRON, INTRON, ACCEPTOR, EXON, NONCODING, NONCODING])

    without = benchmark_gt_vs_pred_single(
        gt_labels=gt,
        pred_labels=pred,
        label_config=BEND_LABEL_CONFIG,
        metrics=[EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
    )
    assert "transition_failures" not in without
    assert "false_transitions" not in without

    with_transitions = benchmark_gt_vs_pred_single(
        gt_labels=gt,
        pred_labels=pred,
        label_config=BEND_LABEL_CONFIG,
        metrics=[EvalMetrics.NUCLEOTIDE_CLASSIFICATION, EvalMetrics.STATE_TRANSITIONS],
    )
    assert "transition_failures" in with_transitions
    assert "false_transitions" in with_transitions


def test_state_transitions_in_default_metric_set():
    """Default metrics keep transitions on so framing plots still render."""
    from dna_segmentation_benchmark.eval.evaluate_predictors import (
        benchmark_gt_vs_pred_single,
    )

    gt = np.array([EXON, EXON, DONOR, INTRON, ACCEPTOR, EXON, NONCODING, NONCODING])
    pred = np.array([EXON, DONOR, INTRON, INTRON, ACCEPTOR, EXON, NONCODING, NONCODING])

    default = benchmark_gt_vs_pred_single(
        gt_labels=gt,
        pred_labels=pred,
        label_config=BEND_LABEL_CONFIG,
    )
    assert "transition_failures" in default
    assert "false_transitions" in default
