import numpy as np
import pytest

from dna_segmentation_benchmark.eval.evaluate_predictors import EvalMetrics
from dna_segmentation_benchmark.label_definition import (
    AnnotationMode,
    BenchmarkScope,
    LabelConfig,
    BEND_LABEL_CONFIG,
)

# ------------------------------------------------------------------
# Convenience token constants
# ------------------------------------------------------------------
EXON, DONOR, INTRON, ACCEPTOR, NONCODING = 0, 1, 2, 3, 8

# A second label set to prove label-agnosticism
CUSTOM_CONFIG = LabelConfig(
    annotation_mode=AnnotationMode.EXON_INTRON,
    background_label=-1,
    exon_label=5,
)

CDS_SCOPE_CONFIG = LabelConfig(
    annotation_mode=AnnotationMode.UTR_CDS_INTRON,
    evaluation_scope=BenchmarkScope.CDS,
    background_label=8,
    cds_label=0,
    five_prime_utr_label=4,
    three_prime_utr_label=5,
    intron_label=2,
    splice_donor_label=1,
    splice_acceptor_label=3,
)


def _h(cells, n=100):
    """Build a length-``n`` int list from a ``{bin: count}`` sparse spec."""
    a = [0] * n
    for k, v in cells.items():
        a[k] = v
    return a

SINGLE_SEQUENCE_TEST_CASES = [
    pytest.param(
        np.array([[8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8], [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INDEL": {"by_boundary": {"five_prime_terminal_exon": {"5_prime_extensions": [3]}, "internal_exon": {"whole_insertions": [4], "3_prime_extensions": [2], "3_prime_deletions": [3], "5_prime_deletions": [1]}, "three_prime_terminal_exon": {"whole_deletions": [2]}}, "junction_opportunities": {"five_prime_terminal_exon": 1, "internal_exon": 4, "three_prime_terminal_exon": 1}, "n_gt_segments": 3, "n_pred_segments": 3},
            "REGION_DISCOVERY": {"neighborhood_hit": {"tp": 2, "fp": 1, "fn": 1, "tn": 0}, "internal_hit": {"tp": 0, "fp": 3, "fn": 3, "tn": 0}, "full_coverage_hit": {"tp": 0, "fp": 3, "fn": 3, "tn": 0}, "perfect_boundary_hit": {"tp": 0, "fp": 3, "fn": 3, "tn": 0}},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": 0, "last_sec_correct_5_prime_boundary": 0, "iou_scores": [0.25, 0.5714285714285714], "fuzzy_metrics": {"boundary_offsets": [(-3, -3), (1, 2)], "total_gt": 3}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"tp": 6, "fp": 9, "fn": 6, "tn": 4}},
        },
        id='exon_all_insertions_deletions',
    ),
    pytest.param(
        np.array([[8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8], [8, 8, 8, 0, 2, 0, 2, 0, 2, 0, 2, 2, 0, 2, 0, 2, 0, 2, 2, 0, 0, 8, 8, 0, 8]]),
        CDS_SCOPE_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INDEL": {"by_boundary": {"internal_exon": {"whole_insertions": [1], "split": [1, 1, 1, 1]}, "single_exon_gene": {"whole_insertions": [1]}}, "junction_opportunities": {"five_prime_terminal_exon": 1, "internal_exon": 4, "three_prime_terminal_exon": 1}, "n_gt_segments": 3, "n_pred_segments": 9},
            "REGION_DISCOVERY": {"neighborhood_hit": {"tp": 3, "fp": 6, "fn": 0, "tn": 0}, "internal_hit": {"tp": 3, "fp": 6, "fn": 0, "tn": 0}, "full_coverage_hit": {"tp": 1, "fp": 8, "fn": 2, "tn": 0}, "perfect_boundary_hit": {"tp": 1, "fp": 8, "fn": 2, "tn": 0}},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": 1, "last_sec_correct_5_prime_boundary": 1, "iou_scores": [0.2, 0.2, 1.0], "fuzzy_metrics": {"boundary_offsets": [(0, -4), (0, -4), (0, 0)], "total_gt": 3}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"tp": 8, "fp": 2, "fn": 4, "tn": 11}},
        },
        id='uncertain_predictions',
    ),
    pytest.param(
        np.array([[8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INDEL": {"by_boundary": {"five_prime_terminal_exon": {"whole_deletions": [5]}, "internal_exon": {"whole_deletions": [5]}, "three_prime_terminal_exon": {"whole_deletions": [2]}}},
            "REGION_DISCOVERY": {"neighborhood_hit": {"tp": 0, "fp": 0, "fn": 3, "tn": 0}, "internal_hit": {"tp": 0, "fp": 0, "fn": 3, "tn": 0}, "full_coverage_hit": {"tp": 0, "fp": 0, "fn": 3, "tn": 0}, "perfect_boundary_hit": {"tp": 0, "fp": 0, "fn": 3, "tn": 0}},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": 0, "last_sec_correct_5_prime_boundary": 0, "iou_scores": [], "fuzzy_metrics": {"boundary_offsets": [], "total_gt": 3}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"tp": 0, "fp": 0, "fn": 12, "tn": 13}},
        },
        id='empty_pred',
    ),
    pytest.param(
        np.array([[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INDEL": {"by_boundary": {"single_exon_gene": {"whole_insertions": [5, 5, 2]}}},
            "REGION_DISCOVERY": {"neighborhood_hit": {"tp": 0, "fp": 3, "fn": 0, "tn": 0}, "internal_hit": {"tp": 0, "fp": 3, "fn": 0, "tn": 0}, "full_coverage_hit": {"tp": 0, "fp": 3, "fn": 0, "tn": 0}, "perfect_boundary_hit": {"tp": 0, "fp": 3, "fn": 0, "tn": 0}},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": 0, "last_sec_correct_5_prime_boundary": 0, "iou_scores": [], "fuzzy_metrics": {"boundary_offsets": [], "total_gt": 0}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"tp": 0, "fp": 12, "fn": 0, "tn": 13}},
        },
        id='empty_gt',
    ),
    pytest.param(
        np.array([[8, 8, 8, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8, 8], [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "REGION_DISCOVERY": {"neighborhood_hit": {"tp": 4, "fp": 0, "fn": 0, "tn": 0}, "internal_hit": {"tp": 1, "fp": 3, "fn": 3, "tn": 0}, "full_coverage_hit": {"tp": 4, "fp": 0, "fn": 0, "tn": 0}, "perfect_boundary_hit": {"tp": 1, "fp": 3, "fn": 3, "tn": 0}},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": 1, "last_sec_correct_5_prime_boundary": 1, "iou_scores": [0.4, 0.7142857142857143, 1.0, 0.6666666666666666], "fuzzy_metrics": {"boundary_offsets": [(-3, 0), (-1, 1), (0, 0), (0, 1)], "total_gt": 4}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"tp": 12, "fp": 6, "fn": 0, "tn": 11}},
        },
        id='in_depth_section_test',
    ),
    pytest.param(
        np.array([[0, 0, 0, 0, 2, 2, 2, 0, 0, 0], [8, 8, 8, 0, 0, 0, 0, 0, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INDEL": {"by_boundary": {"internal_exon": {"joined": [3]}, "five_prime_terminal_exon": {"5_prime_deletions": [3]}, "three_prime_terminal_exon": {"3_prime_deletions": [2]}}},
            "REGION_DISCOVERY": {"neighborhood_hit": {"tp": 1, "fp": 0, "fn": 1, "tn": 0}, "internal_hit": {"tp": 0, "fp": 1, "fn": 2, "tn": 0}, "full_coverage_hit": {"tp": 0, "fp": 1, "fn": 2, "tn": 0}, "perfect_boundary_hit": {"tp": 0, "fp": 1, "fn": 2, "tn": 0}},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": 0, "last_sec_correct_5_prime_boundary": 0, "iou_scores": [0.125], "fuzzy_metrics": {"boundary_offsets": [(3, 4)], "total_gt": 2}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"tp": 2, "fp": 3, "fn": 5, "tn": 0}},
        },
        id='exon_joined_with_deletions',
    ),
    pytest.param(
        np.array([[8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8], [8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "REGION_DISCOVERY": {"neighborhood_hit": {"tp": 3, "fp": 0, "fn": 0, "tn": 0}, "internal_hit": {"tp": 3, "fp": 0, "fn": 0, "tn": 0}, "full_coverage_hit": {"tp": 3, "fp": 0, "fn": 0, "tn": 0}, "perfect_boundary_hit": {"tp": 3, "fp": 0, "fn": 0, "tn": 0}},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": 1, "last_sec_correct_5_prime_boundary": 1, "iou_scores": [1.0, 1.0, 1.0], "fuzzy_metrics": {"boundary_offsets": [(0, 0), (0, 0), (0, 0)], "total_gt": 3}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"tp": 12, "fp": 0, "fn": 0, "tn": 13}},
        },
        id='exon_fully_correct',
    ),
    pytest.param(
        np.array([[8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2], [8, 8, 8, 0, 0, 0, 0, 0, 8, 8, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "REGION_DISCOVERY": {"neighborhood_hit": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "internal_hit": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "full_coverage_hit": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "perfect_boundary_hit": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": 1, "last_sec_correct_5_prime_boundary": 1, "iou_scores": [1.0], "fuzzy_metrics": {"boundary_offsets": [(0, 0)], "total_gt": 1}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"tp": 5, "fp": 0, "fn": 0, "tn": 7}},
        },
        id='exon_fully_correct_2',
    ),
    pytest.param(
        np.array([[8, 8, 8, 0, 0, 2, 2, 2, 2, 0, 0], [8, 8, 8, 0, 2, 2, 2, 8, 8, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INDEL": {"by_boundary": {"internal_exon": {"3_prime_deletions": [1]}, "three_prime_terminal_exon": {"whole_deletions": [2]}}},
            "REGION_DISCOVERY": {"neighborhood_hit": {"tp": 1, "fp": 0, "fn": 1, "tn": 0}, "internal_hit": {"tp": 1, "fp": 0, "fn": 1, "tn": 0}, "full_coverage_hit": {"tp": 0, "fp": 1, "fn": 2, "tn": 0}, "perfect_boundary_hit": {"tp": 0, "fp": 1, "fn": 2, "tn": 0}},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": 0, "last_sec_correct_5_prime_boundary": 0, "iou_scores": [0.5], "fuzzy_metrics": {"boundary_offsets": [(0, -1)], "total_gt": 2}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"tp": 1, "fp": 0, "fn": 3, "tn": 7}},
        },
        id='exon_test2',
    ),
    pytest.param(
        np.array([[8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0], [0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 8, 8]]),
        CDS_SCOPE_CONFIG,
        [EvalMetrics.PHASE_DRIFT],
        {
            "PHASE_DRIFT": {"gt_frames": np.array([np.inf, np.inf, np.inf, 0.0, 0.0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 0.0, 0.0, 0.0, 0.0, np.inf, np.inf, np.inf, np.inf])},
        },
        id='phase_drift_test',
    ),
    pytest.param(
        # Perfect prediction: GT == pred, every overlap position has frame 0.
        np.array([[0, 0, 0, 8, 8, 8, 0, 0, 0], [0, 0, 0, 8, 8, 8, 0, 0, 0]]),
        CDS_SCOPE_CONFIG,
        [EvalMetrics.PHASE_DRIFT],
        {
            "PHASE_DRIFT": {"gt_frames": np.array([0., 0., 0., np.inf, np.inf, np.inf, 0., 0., 0.])},
        },
        id='phase_drift_perfect_prediction',
    ),
    pytest.param(
        # Pred is shifted 1 position to the right: one leading background in pred,
        # one trailing background in GT. The single-base lag creates a persistent
        # frame offset of 1 at every overlap position (mod-3 arithmetic: |pred_cumsum
        # - gt_cumsum| = 1 at all overlap sites).
        np.array([[0, 0, 0, 0, 0, 0, 8], [8, 0, 0, 0, 0, 0, 0]]),
        CDS_SCOPE_CONFIG,
        [EvalMetrics.PHASE_DRIFT],
        {
            "PHASE_DRIFT": {"gt_frames": np.array([np.inf, 1., 1., 1., 1., 1., np.inf])},
        },
        id='phase_drift_persistent_plus1',
    ),
    pytest.param(
        # Pred is shifted 2 positions to the right: cumsum difference is 2 at every
        # overlap position.
        np.array([[0, 0, 0, 0, 0, 0, 8, 8], [8, 8, 0, 0, 0, 0, 0, 0]]),
        CDS_SCOPE_CONFIG,
        [EvalMetrics.PHASE_DRIFT],
        {
            "PHASE_DRIFT": {"gt_frames": np.array([np.inf, np.inf, 2., 2., 2., 2., np.inf, np.inf])},
        },
        id='phase_drift_persistent_plus2',
    ),
    pytest.param(
        # GT and pred CDS regions are completely non-overlapping: valid_mask is all
        # False so every position stays at inf.
        np.array([[0, 0, 0, 8, 8, 8], [8, 8, 8, 0, 0, 0]]),
        CDS_SCOPE_CONFIG,
        [EvalMetrics.PHASE_DRIFT],
        {
            "PHASE_DRIFT": {"gt_frames": np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])},
        },
        id='phase_drift_no_overlap',
    ),
    pytest.param(
        # Frame escalates 0→1→2 inside a single exon. GT has 9 consecutive CDS
        # bases; pred drops every 3rd one (positions 2, 5, 8). At each gap the
        # cumsum difference increases by 1, shifting the frame up one step.
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 8], [0, 0, 8, 0, 0, 8, 0, 0, 8, 8]]),
        CDS_SCOPE_CONFIG,
        [EvalMetrics.PHASE_DRIFT],
        {
            "PHASE_DRIFT": {"gt_frames": np.array([0., 0., np.inf, 1., 1., np.inf, 2., 2., np.inf, np.inf])},
        },
        id='phase_drift_escalating_within_exon',
    ),
    pytest.param(
        # Cyclic 0→1→2→0→1→2 across 6 sparse CDS positions. GT has CDS only at
        # every 3rd position (0,3,6,9,12,15); pred inserts one extra CDS between
        # each GT CDS site. The cumsum difference grows by 1 per site and wraps mod
        # 3 to produce the repeating cycle.
        np.array(
            [
                [0, 8, 8, 0, 8, 8, 0, 8, 8, 0, 8, 8, 0, 8, 8, 0],
                [0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0, 8, 0, 0],
            ]
        ),
        CDS_SCOPE_CONFIG,
        [EvalMetrics.PHASE_DRIFT],
        {
            "PHASE_DRIFT": {"gt_frames": np.array([0., np.inf, np.inf, 1., np.inf, np.inf, 2., np.inf, np.inf, 0., np.inf, np.inf, 1., np.inf, np.inf, 2.])},
        },
        id='phase_drift_cyclic_012',
    ),
    pytest.param(
        # GT CDS count (4) is not divisible by 3: the metric skips the sequence and
        # returns an empty frame list.
        np.array([[0, 0, 0, 0, 8, 8, 8], [0, 0, 0, 0, 0, 0, 0]]),
        CDS_SCOPE_CONFIG,
        [EvalMetrics.PHASE_DRIFT],
        {
            "PHASE_DRIFT": {"gt_frames": np.array([])},
        },
        id='phase_drift_gt_cds_not_mod3',
    ),
    pytest.param(
        # Pred has only 1 CDS base (< 3): the metric returns an empty frame list
        # without attempting computation.
        np.array([[0, 0, 0, 8, 0, 0, 0], [8, 0, 8, 8, 8, 8, 8]]),
        CDS_SCOPE_CONFIG,
        [EvalMetrics.PHASE_DRIFT],
        {
            "PHASE_DRIFT": {"gt_frames": np.array([])},
        },
        id='phase_drift_pred_too_few_cds',
    ),
    pytest.param(
        np.array([[-1, -1, -1, 5, 5, 5, 5, 5, -1, -1, -1, -1, 5, 5, 5, 5, 5, -1, -1, 5, 5], [5, 5, 5, 5, 5, -1, -1, -1, 5, 5, 5, 5, -1, 5, 5, 5, 5, 5, 5, -1, -1]]),
        CUSTOM_CONFIG,
        [EvalMetrics.INDEL],
        {
            "INDEL": {"by_boundary": {"five_prime_terminal_exon": {"5_prime_extensions": [3], "5_prime_deletions": [1]}, "internal_exon": {"whole_insertions": [4]}, "three_prime_terminal_exon": {"3_prime_deletions": [3], "3_prime_extensions": [2]}, "single_exon_gene": {"whole_deletions": [2]}}},
        },
        id='Different_label_test',
    ),
    pytest.param(
        np.array([[8, 0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 0, 0, 8, 8], [8, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "REGION_DISCOVERY": {"neighborhood_hit": [{"tp": 1, "fp": 0, "fn": 0, "tn": 0}, {"tp": 1, "fp": 0, "fn": 0, "tn": 0}], "internal_hit": [{"tp": 1, "fp": 0, "fn": 0, "tn": 0}, {"tp": 1, "fp": 0, "fn": 0, "tn": 0}], "full_coverage_hit": [{"tp": 1, "fp": 0, "fn": 0, "tn": 0}, {"tp": 1, "fp": 0, "fn": 0, "tn": 0}], "perfect_boundary_hit": [{"tp": 1, "fp": 0, "fn": 0, "tn": 0}, {"tp": 1, "fp": 0, "fn": 0, "tn": 0}]},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": [1, 1], "last_sec_correct_5_prime_boundary": [1, 1], "iou_scores": [1.0, 1.0], "fuzzy_metrics": {"boundary_offsets": [(0, 0), (0, 0)], "total_gt": 2}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": [{"tp": 3, "fp": 0, "fn": 0, "tn": 3}, {"tp": 2, "fp": 0, "fn": 0, "tn": 4}]},
        },
        id='mask_test',
    ),
    pytest.param(
        # Coding segments touch BOTH array edges (per-transcript windowing): the
        # 5' edge of seg A and the 3' edge of seg B are gene boundaries that
        # coincide with the array end, so their flank is "none" (terminal).  Both
        # the numerator (5'/3' deletion runs) and the denominator
        # (junction_opportunities) must type these as terminal-exon boundaries.
        # A plain label-transition count would miss the two edge boundaries and
        # report only 1 opportunity each instead of 2.
        np.array([[0, 0, 0, 0, 8, 8, 0, 0, 0, 0], [8, 8, 0, 0, 8, 8, 0, 0, 0, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL],
        {
            "INDEL": {"by_boundary": {"five_prime_terminal_exon": {"5_prime_deletions": [2]}, "three_prime_terminal_exon": {"3_prime_deletions": [1]}}, "junction_opportunities": {"five_prime_terminal_exon": 2, "three_prime_terminal_exon": 2}, "n_gt_segments": 2, "n_pred_segments": 2},
        },
        id='indel_window_edge_terminal_exon',
    ),
]

STRUCTURAL_COHERENCE_TEST_CASES = [
    pytest.param(
        np.array([[8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8], [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {"exon_chain": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "exon_chain_subset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "exon_chain_superset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "boundary_shift_count": 0, "boundary_shift_total": 0, "boundary_shift_offsets": [], "exon_recall_per_transcript": 1.0, "exon_precision_per_transcript": 1.0, "false_exon_count_per_transcript": 0, "transcript_match_class": 'exact', "segment_count_delta": 0, "intron_chain": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "intron_chain_subset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "intron_chain_superset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "splice_site_results": {"both_correct": 0, "donor_only": 0, "acceptor_only": 0, "neither": 0, "donor_tp": 0, "donor_fp": 0, "donor_fn": 0, "acceptor_tp": 0, "acceptor_fp": 0, "acceptor_fn": 0, "gt_malformed_junctions": 0, "pred_malformed_junctions": 0}},
        },
        id='sc_exact_match',
    ),
    pytest.param(
        np.array([[8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8], [8, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {"exon_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "boundary_shift_count": 3, "boundary_shift_total": 3, "boundary_shift_offsets": [{"offset": -1, "position": "terminal"}, {"offset": -1, "position": "internal"}, {"offset": 1, "position": "terminal"}], "exon_recall_per_transcript": 0.0, "exon_precision_per_transcript": 0.0, "false_exon_count_per_transcript": 3, "transcript_match_class": 'boundary_shift_terminal', "segment_count_delta": 0, "intron_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "splice_site_results": {"both_correct": 0, "donor_only": 0, "acceptor_only": 0, "neither": 0, "donor_tp": 0, "donor_fp": 0, "donor_fn": 0, "acceptor_tp": 0, "acceptor_fp": 0, "acceptor_fn": 0, "gt_malformed_junctions": 0, "pred_malformed_junctions": 0}},
        },
        id='sc_boundary_shift',
    ),
    pytest.param(
        np.array([[8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8], [8, 8, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {"exon_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_subset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "exon_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "boundary_shift_count": 0, "boundary_shift_total": 0, "boundary_shift_offsets": [], "exon_recall_per_transcript": 0.6666666666666666, "exon_precision_per_transcript": 1.0, "false_exon_count_per_transcript": 0, "transcript_match_class": 'missing_segments', "segment_count_delta": -1, "intron_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "splice_site_results": {"both_correct": 0, "donor_only": 0, "acceptor_only": 0, "neither": 0, "donor_tp": 0, "donor_fp": 0, "donor_fn": 0, "acceptor_tp": 0, "acceptor_fp": 0, "acceptor_fn": 0, "gt_malformed_junctions": 0, "pred_malformed_junctions": 0}},
        },
        id='sc_missing_segments',
    ),
    pytest.param(
        np.array([[8, 8, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 8, 8], [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {"exon_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_superset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "boundary_shift_count": 0, "boundary_shift_total": 0, "boundary_shift_offsets": [], "exon_recall_per_transcript": 1.0, "exon_precision_per_transcript": 0.6666666666666666, "false_exon_count_per_transcript": 1, "transcript_match_class": 'extra_segments', "segment_count_delta": 1, "intron_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "splice_site_results": {"both_correct": 0, "donor_only": 0, "acceptor_only": 0, "neither": 0, "donor_tp": 0, "donor_fp": 0, "donor_fn": 0, "acceptor_tp": 0, "acceptor_fp": 0, "acceptor_fn": 0, "gt_malformed_junctions": 0, "pred_malformed_junctions": 0}},
        },
        id='sc_extra_segments',
    ),
    pytest.param(
        np.array([[8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 8, 8], [8, 8, 8, 8, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {"exon_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "boundary_shift_count": 0, "boundary_shift_total": 0, "boundary_shift_offsets": [], "exon_recall_per_transcript": 0.0, "exon_precision_per_transcript": 0.0, "false_exon_count_per_transcript": 2, "transcript_match_class": 'substitution', "segment_count_delta": -1, "intron_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "splice_site_results": {"both_correct": 0, "donor_only": 0, "acceptor_only": 0, "neither": 0, "donor_tp": 0, "donor_fp": 0, "donor_fn": 0, "acceptor_tp": 0, "acceptor_fp": 0, "acceptor_fn": 0, "gt_malformed_junctions": 0, "pred_malformed_junctions": 0}},
        },
        id='sc_structurally_different',
    ),
    pytest.param(
        np.array([[8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8], [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {"exon_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "boundary_shift_count": 0, "boundary_shift_total": 0, "boundary_shift_offsets": [], "exon_recall_per_transcript": 0.0, "exon_precision_per_transcript": 0.0, "false_exon_count_per_transcript": 0, "transcript_match_class": 'missed', "segment_count_delta": -2, "intron_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "splice_site_results": {"both_correct": 0, "donor_only": 0, "acceptor_only": 0, "neither": 0, "donor_tp": 0, "donor_fp": 0, "donor_fn": 0, "acceptor_tp": 0, "acceptor_fp": 0, "acceptor_fn": 0, "gt_malformed_junctions": 0, "pred_malformed_junctions": 0}},
        },
        id='sc_missed',
    ),
    pytest.param(
        np.array([[8, 8, 8, 8, 8, 8, 8, 8, 8, 8], [8, 8, 0, 0, 0, 8, 8, 8, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {"exon_chain": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "exon_chain_subset": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "exon_chain_superset": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "intron_chain": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "intron_chain_subset": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "intron_chain_superset": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "splice_site_results": {"both_correct": 0, "donor_only": 0, "acceptor_only": 0, "neither": 0, "donor_tp": 0, "donor_fp": 0, "donor_fn": 0, "acceptor_tp": 0, "acceptor_fp": 0, "acceptor_fn": 0, "gt_malformed_junctions": 0, "pred_malformed_junctions": 0}},
        },
        id='sc_no_gt_segments',
    ),
    pytest.param(
        np.array([[8, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 8, 8], [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {"exon_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "exon_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "boundary_shift_count": 0, "boundary_shift_total": 0, "boundary_shift_offsets": [], "exon_recall_per_transcript": 0.5, "exon_precision_per_transcript": 0.6, "false_exon_count_per_transcript": 2, "transcript_match_class": 'partial_overlap', "segment_count_delta": -1, "intron_chain": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_subset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "intron_chain_superset": {"tp": 0, "fp": 1, "fn": 1, "tn": 0}, "splice_site_results": {"both_correct": 0, "donor_only": 0, "acceptor_only": 0, "neither": 0, "donor_tp": 0, "donor_fp": 0, "donor_fn": 0, "acceptor_tp": 0, "acceptor_fp": 0, "acceptor_fn": 0, "gt_malformed_junctions": 0, "pred_malformed_junctions": 0}},
        },
        id='sc_six_exon_mixed_errors',
    ),
    pytest.param(
        np.array([[8, 8, 0, 0, 0, 0, 8, 8], [8, 8, 0, 0, 0, 0, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {"exon_chain": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "exon_chain_subset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "exon_chain_superset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "boundary_shift_count": 0, "boundary_shift_total": 0, "boundary_shift_offsets": [], "exon_recall_per_transcript": 1.0, "exon_precision_per_transcript": 1.0, "false_exon_count_per_transcript": 0, "transcript_match_class": 'exact', "segment_count_delta": 0, "intron_chain": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "intron_chain_subset": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "intron_chain_superset": {"tp": 0, "fp": 0, "fn": 0, "tn": 0}, "splice_site_results": {"both_correct": 0, "donor_only": 0, "acceptor_only": 0, "neither": 0, "donor_tp": 0, "donor_fp": 0, "donor_fn": 0, "acceptor_tp": 0, "acceptor_fp": 0, "acceptor_fn": 0, "gt_malformed_junctions": 0, "pred_malformed_junctions": 0}},
        },
        id='sc_single_segment',
    ),
    pytest.param(
        np.array([[8, 8, 0, 0, 1, 2, 2, 2, 2, 3, 0, 0, 8, 8], [8, 8, 0, 0, 1, 2, 2, 2, 2, 3, 0, 0, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.STRUCTURAL_COHERENCE],
        {
            "STRUCTURAL_COHERENCE": {"exon_chain": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "exon_chain_subset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "exon_chain_superset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "boundary_shift_count": 0, "boundary_shift_total": 0, "boundary_shift_offsets": [], "exon_recall_per_transcript": 1.0, "exon_precision_per_transcript": 1.0, "false_exon_count_per_transcript": 0, "transcript_match_class": 'exact', "segment_count_delta": 0, "intron_chain": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "intron_chain_subset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "intron_chain_superset": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}, "splice_site_results": {"both_correct": 1, "donor_only": 0, "acceptor_only": 0, "neither": 0, "donor_tp": 1, "donor_fp": 0, "donor_fn": 0, "acceptor_tp": 1, "acceptor_fp": 0, "acceptor_fn": 0, "gt_malformed_junctions": 0, "pred_malformed_junctions": 0}},
        },
        id='splice_site_confusion',
    ),
]

DIAGNOSTIC_DEPTH_TEST_CASES = [
    pytest.param(
        np.array([[8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8], [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [3, 3],
                "pred_segment_lengths": [3, 3],
                "length_emd": 0.0,
                "position_bias_histogram_fn": _h({}),
                "position_bias_histogram_fp": _h({}),
            },
        },
        id='dd_no_errors',
    ),
    pytest.param(
        np.array([[8, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 8], [8, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [2, 2, 2],
                "pred_segment_lengths": [2, 2],
                "length_emd": 0.0,
                "position_bias_histogram_fn": _h({40: 1, 50: 1}),
                "position_bias_histogram_fp": _h({}),
            },
        },
        id='dd_exon_skip',
    ),
    pytest.param(
        np.array([[8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8], [8, 0, 0, 0, 2, 2, 0, 0, 0, 0, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [9],
                "pred_segment_lengths": [3, 4],
                "length_emd": 5.5,
                "position_bias_histogram_fn": _h({33: 1, 44: 1}),
                "position_bias_histogram_fp": _h({}),
            },
        },
        id='dd_novel_insertion',
    ),
    pytest.param(
        np.array([[8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 8], [8, 8, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 8, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [3, 3, 3, 3],
                "pred_segment_lengths": [3, 3, 3, 3],
                "length_emd": 0.0,
                "position_bias_histogram_fn": _h({0: 1, 27: 1, 55: 1, 83: 1}),
                "position_bias_histogram_fp": _h({16: 1, 44: 1, 72: 1}),
            },
        },
        id='dd_cascade_shift',
    ),
    pytest.param(
        np.array([[8, 0, 0, 0, 2, 2, 2, 0, 0, 0, 8], [8, 0, 0, 0, 0, 2, 0, 0, 0, 0, 8]]),
        BEND_LABEL_CONFIG,
        [EvalMetrics.DIAGNOSTIC_DEPTH],
        {
            "DIAGNOSTIC_DEPTH": {
                "gt_segment_lengths": [3, 3],
                "pred_segment_lengths": [4, 4],
                "length_emd": 1.0,
                "position_bias_histogram_fn": _h({}),
                "position_bias_histogram_fp": _h({33: 1, 55: 1}),
            },
        },
        id='dd_compensating_errors',
    ),
]

MULTI_SEQUENCE_TEST_CASES = [
    pytest.param(
        [np.array([8, 8, 8, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 8, 8, 8, 8])],
        [np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])],
        BEND_LABEL_CONFIG,
        [EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.BOUNDARY_EXACTNESS, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        {
            "INDEL": {"by_boundary": {"five_prime_terminal_exon": {"whole_deletions": [5]}, "internal_exon": {"whole_deletions": [5]}, "three_prime_terminal_exon": {"whole_deletions": [2]}}, "junction_opportunities": {"five_prime_terminal_exon": 1, "internal_exon": 4, "three_prime_terminal_exon": 1}, "n_gt_segments": 3, "n_pred_segments": 0},
            "REGION_DISCOVERY": {"neighborhood_hit": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "internal_hit": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "full_coverage_hit": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "perfect_boundary_hit": {"precision": 0.0, "recall": 0.0, "f1": 0.0}},
            "BOUNDARY_EXACTNESS": {"first_sec_correct_3_prime_boundary": [0], "last_sec_correct_5_prime_boundary": [0], "iou_scores": [], "iou_stats": {"count": 0, "mean": 0.0, "mae": 0.0, "rmse": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}},
            "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"precision": 0.0, "recall": 0.0, "f1": 0.0}},
        },
        id='no_nuc_positives',
    ),
]

# ------------------------------------------------------------------
# SPLICE_SITE test cases
# Each entry: (gt_pred_array, expected_splice_sites)
# Using BEND_LABEL_CONFIG: bg=8, exon=0, donor=1, intron=2, acceptor=3
# ------------------------------------------------------------------

SPLICE_SITE_TEST_CASES = [
    # -- Case 1: perfect match, 2 introns —————————————————————————————
    # GT/pred: exon-exon-donor-intron×3-acceptor-exon-exon-donor-intron×2-acceptor-exon-exon
    # Both pairs: donor hit, acceptor hit → both_correct×2, no FP.
    pytest.param(
        np.array([
            [0, 0, 1, 2, 2, 2, 3, 0, 0, 1, 2, 2, 3, 0, 0],
            [0, 0, 1, 2, 2, 2, 3, 0, 0, 1, 2, 2, 3, 0, 0],
        ]),
        {
            "both_correct": 2, "donor_only": 0, "acceptor_only": 0, "neither": 0,
            "donor_tp": 2, "donor_fp": 0, "donor_fn": 0,
            "acceptor_tp": 2, "acceptor_fp": 0, "acceptor_fn": 0,
        },
        id="ss_perfect_two_introns",
    ),
    # -- Case 2: pred replaces all donors with intron, keeps acceptors —
    # GT: exon-exon-donor-intron×3-acceptor-exon-exon-donor-intron×2-acceptor-exon-exon
    # pred: exon-exon-intron×4-acceptor-exon-exon-intron×3-acceptor-exon-exon
    # donor positions become intron → pred has no donor segments.
    # Both acceptors survive exactly → acceptor_only×2.
    pytest.param(
        np.array([
            [0, 0, 1, 2, 2, 2, 3, 0, 0, 1, 2, 2, 3, 0, 0],
            [0, 0, 2, 2, 2, 2, 3, 0, 0, 2, 2, 2, 3, 0, 0],
        ]),
        {
            "both_correct": 0, "donor_only": 0, "acceptor_only": 2, "neither": 0,
            "donor_tp": 0, "donor_fp": 0, "donor_fn": 2,
            "acceptor_tp": 2, "acceptor_fp": 0, "acceptor_fn": 0,
        },
        id="ss_all_donors_wrong",
    ),
    # -- Case 3: first pair correct, second pair both wrong + spurious donor ——
    # GT:   exon-exon-donor(2)-intron(3-5)-acceptor(6)-exon-donor(8)-intron(9-10)-acceptor(11)-exon
    # pred: exon-exon-donor(2)-intron(3-5)-acceptor(6)-donor(7)-intron(8-10)-intron(11)-exon
    #   • Pair 1: both correct → both_correct=1
    #   • Pair 2: donor(8) replaced by intron, acceptor(11) replaced by intron → neither=1
    #   • Spurious donor at pos 7 → donor_fp=1
    pytest.param(
        np.array([
            [0, 0, 1, 2, 2, 2, 3, 0, 1, 2, 2, 3, 0],
            [0, 0, 1, 2, 2, 2, 3, 1, 2, 2, 2, 2, 0],
        ]),
        {
            "both_correct": 1, "donor_only": 0, "acceptor_only": 0, "neither": 1,
            "donor_tp": 1, "donor_fp": 1, "donor_fn": 1,
            "acceptor_tp": 1, "acceptor_fp": 0, "acceptor_fn": 1,
        },
        id="ss_mixed_with_spurious_donor",
    ),
    # -- Case 4: 3 introns — neither / donor_only / both_correct + spurious acceptor ——
    # GT:   exon-donor(1)-intron(2-3)-acceptor(4)-exon-donor(6)-intron(7-8)-acceptor(9)-exon-donor(11)-intron(12-13)-acceptor(14)-exon
    # pred: exon-intron(1-4)-spurious_acc(5)-donor(6)-intron(7-9)-exon-donor(11)-intron(12-13)-acceptor(14)-exon
    #   • Pair 1 (donor1, acc4): donor→intron, acceptor→intron → neither=1
    #   • Pair 2 (donor6, acc9): donor correct, acceptor→intron → donor_only=1
    #   • Pair 3 (donor11, acc14): both correct → both_correct=1
    #   • Spurious acceptor at pos 5 → acceptor_fp=1
    pytest.param(
        np.array([
            [0, 1, 2, 2, 3, 0, 1, 2, 2, 3, 0, 1, 2, 2, 3, 0],
            [0, 2, 2, 2, 2, 3, 1, 2, 2, 2, 0, 1, 2, 2, 3, 0],
        ]),
        {
            "both_correct": 1, "donor_only": 1, "acceptor_only": 0, "neither": 1,
            "donor_tp": 2, "donor_fp": 0, "donor_fn": 1,
            "acceptor_tp": 1, "acceptor_fp": 1, "acceptor_fn": 2,
        },
        id="ss_three_introns_mixed_spurious_acceptor",
    ),
]
