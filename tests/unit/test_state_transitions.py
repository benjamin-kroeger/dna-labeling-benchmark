import numpy as np
import pytest

from dna_segmentation_benchmark.eval.preprocessing import collapse_out_of_scope_content
from dna_segmentation_benchmark.eval.state_transitions import (
    TransitionAnalysis,
    compute_state_change_errors,
)
from dna_segmentation_benchmark.label_definition import (
    AnnotationMode,
    BEND_LABEL_CONFIG,
    BenchmarkScope,
    LabelConfig,
)

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


# ------------------------------------------------------------------
# Scope-aware collapse: STATE_TRANSITIONS honours evaluation_scope so a
# UTR-aware prediction is not penalised against a CDS-only GT under `cds` scope.
# ------------------------------------------------------------------

# UTR_CDS_INTRON tokens (match LabelConfig.default_utr_cds_intron):
# background=8, cds=0, 5'UTR=4, 3'UTR=5, intron=2, donor=1, acceptor=3.
_BG, _CDS, _5UTR, _3UTR, _INTRON = 8, 0, 4, 5, 2


def _utr_config(scope: BenchmarkScope) -> LabelConfig:
    return LabelConfig(
        annotation_mode=AnnotationMode.UTR_CDS_INTRON,
        evaluation_scope=scope,
        background_label=_BG,
        cds_label=_CDS,
        five_prime_utr_label=_5UTR,
        three_prime_utr_label=_3UTR,
        intron_label=_INTRON,
        splice_donor_label=1,
        splice_acceptor_label=3,
    )


def _transitions_equal(a: dict, b: dict) -> bool:
    """True when two benchmark results carry identical transition matrices."""
    fa, fb = a["transition_failures"], b["transition_failures"]
    if fa.keys() != fb.keys() or not all(np.array_equal(fa[k], fb[k]) for k in fa):
        return False
    for cat in ("late_catchup", "premature", "spurious"):
        da, db = a["false_transitions"][cat], b["false_transitions"][cat]
        if da.keys() != db.keys() or not all(np.array_equal(da[k], db[k]) for k in da):
            return False
    return a["false_transitions"]["stable_position_counts"] == b["false_transitions"]["stable_position_counts"]


def test_collapse_out_of_scope_content_demotes_utr_only_under_cds_scope():
    labels = np.array([_BG, _5UTR, _CDS, _INTRON, _CDS, _3UTR, 1, 3, _BG])

    # cds scope: 5'/3' UTR -> background; CDS / intron / splice untouched.
    out = collapse_out_of_scope_content(labels, _utr_config(BenchmarkScope.CDS))
    expected = np.array([_BG, _BG, _CDS, _INTRON, _CDS, _BG, 1, 3, _BG])
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(labels, np.array([_BG, _5UTR, _CDS, _INTRON, _CDS, _3UTR, 1, 3, _BG]))  # not mutated

    # transcript_exon scope: UTR is in scope -> no-op (same object, no copy).
    assert collapse_out_of_scope_content(labels, _utr_config(BenchmarkScope.TRANSCRIPT_EXON)) is labels
    # EXON_INTRON mode: only transcript_exon scope exists -> no-op.
    exon_labels = np.array([NONCODING, EXON, INTRON, EXON, NONCODING])
    assert collapse_out_of_scope_content(exon_labels, BEND_LABEL_CONFIG) is exon_labels


def test_state_transitions_ignore_utr_under_cds_scope():
    """Under `cds` scope, a UTR-aware pred is indistinguishable from a NONCODING pred."""
    from dna_segmentation_benchmark.eval.evaluate_predictors import (
        EvalMetrics,
        benchmark_gt_vs_pred_single,
    )

    gt = np.array([_BG, _BG, _BG, _CDS, _CDS, _CDS, _BG, _BG, _BG])  # CDS-only GT
    pred_with_utr = np.array([_BG, _5UTR, _5UTR, _CDS, _CDS, _CDS, _3UTR, _3UTR, _BG])
    pred_with_nc = np.array([_BG, _BG, _BG, _CDS, _CDS, _CDS, _BG, _BG, _BG])

    kwargs = dict(
        label_config=_utr_config(BenchmarkScope.CDS),
        metrics=[EvalMetrics.STATE_TRANSITIONS],
    )
    res_utr = benchmark_gt_vs_pred_single(gt_labels=gt, pred_labels=pred_with_utr, **kwargs)
    res_nc = benchmark_gt_vs_pred_single(gt_labels=gt, pred_labels=pred_with_nc, **kwargs)
    assert _transitions_equal(res_utr, res_nc)


def test_state_transitions_keep_utr_under_transcript_exon_scope():
    """Regression guard: default scope keeps UTR distinct (does not over-collapse)."""
    from dna_segmentation_benchmark.eval.evaluate_predictors import (
        EvalMetrics,
        benchmark_gt_vs_pred_single,
    )

    gt = np.array([_BG, _BG, _BG, _CDS, _CDS, _CDS, _BG, _BG, _BG])
    pred_with_utr = np.array([_BG, _5UTR, _5UTR, _CDS, _CDS, _CDS, _3UTR, _3UTR, _BG])
    pred_with_nc = np.array([_BG, _BG, _BG, _CDS, _CDS, _CDS, _BG, _BG, _BG])

    kwargs = dict(
        label_config=_utr_config(BenchmarkScope.TRANSCRIPT_EXON),
        metrics=[EvalMetrics.STATE_TRANSITIONS],
    )
    res_utr = benchmark_gt_vs_pred_single(gt_labels=gt, pred_labels=pred_with_utr, **kwargs)
    res_nc = benchmark_gt_vs_pred_single(gt_labels=gt, pred_labels=pred_with_nc, **kwargs)
    assert not _transitions_equal(res_utr, res_nc)
