"""State-transition analysis for GT vs prediction arrays.

Provides two complementary, GT-anchored views of transition quality:

1. **GT Transition Confusion Matrices** – At every position where the ground
   truth changes label, record what the predictor did.  One confusion matrix
   per GT source label.  The total count per matrix is fixed by GT, making
   it directly comparable across methods.

2. **False Transition Target Matrices** – At every position where the ground
   truth does *not* change label but the predictor does, record what label
   the predictor falsely transitioned to.  One target-distribution vector
   per GT label, plus the total GT-stable count (denominator for rates).
"""

from dataclasses import dataclass

import numpy as np

from ..label_definition import LabelConfig


@dataclass(frozen=True)
class TransitionAnalysis:
    """Container for the two complementary transition views.

    Attributes
    ----------
    gt_transition_matrices : dict[int, np.ndarray]
        Per source-label confusion matrix (shape ``(L, L)``):
        rows = GT target, cols = pred target.
    false_transition_matrices : dict[int, np.ndarray]
        Per GT-stable label, a 1-D array of length ``L`` counting how many
        times the predictor falsely transitioned *to* each target label.
    stable_position_counts : dict[int, int]
        Per label: total number of GT-stable positions (denominator for
        the false-transition rate).
    """

    gt_transition_matrices: dict[int, np.ndarray]
    false_transition_matrices: dict[int, np.ndarray]
    stable_position_counts: dict[int, int]


def _compute_state_change_errors(
        gt_pred_arr: np.ndarray,
        label_config: LabelConfig,
) -> TransitionAnalysis:
    """Compute GT transition confusion matrices and false transition targets.

    Parameters
    ----------
    gt_pred_arr : np.ndarray
        Shape ``(2, N)`` where row 0 is ground truth and row 1 is prediction.
    label_config : LabelConfig
        Defines the set of valid integer labels.

    Returns
    -------
    TransitionAnalysis
        Frozen dataclass with ``gt_transition_matrices``,
        ``false_transition_matrices``, and ``stable_position_counts``.
    """
    label_ids = sorted(label_config.labels.keys())
    num_labels = len(label_ids)
    # get label ids as array
    label_id_array = np.asarray(label_ids, dtype=gt_pred_arr.dtype)

    # Sliding window: shape (N-1, 2, 2)
    # Each window[i] = [[gt[i], gt[i+1]], [pred[i], pred[i+1]]]
    nuc_transitions = np.lib.stride_tricks.sliding_window_view(
        gt_pred_arr, (2, 2),
    )[0]

    # get all gt source, gt target etc labels
    gt_src = nuc_transitions[:, 0, 0]  # all GT labels at position i
    gt_tgt = nuc_transitions[:, 0, 1]  # all GT labels at position i+1
    pred_src = nuc_transitions[:, 1, 0]  # all Pred labels at position i
    pred_tgt = nuc_transitions[:, 1, 1]  # all Pred labels at position i+1

    # build mask for where a transition occurs in the GT (i.e. where gt_src != gt_tgt)
    gt_transition_mask = gt_src != gt_tgt  # state transition in the gt
    gt_stable_mask = ~gt_transition_mask

    # get the indices of where the label would be inserted into the label list
    gt_src_idx = np.searchsorted(label_id_array, gt_src)
    gt_tgt_idx = np.searchsorted(label_id_array, gt_tgt)
    pred_tgt_idx = np.searchsorted(label_id_array, pred_tgt)

    # ---- 1. GT transition confusion matrices ----------------------------
    # build a n dim confusion tensor
    transition_counts = np.zeros(
        (num_labels, num_labels, num_labels), dtype=np.int64,
    )
    # given the nuc labels aligned by label ids filling out the confusion matrix is just incrementing count
    # at the right postitions
    if np.any(gt_transition_mask):
        np.add.at(
            transition_counts,
            (
                gt_src_idx[gt_transition_mask],
                gt_tgt_idx[gt_transition_mask],
                pred_tgt_idx[gt_transition_mask],
            ),
            1,
        )

    gt_transition_matrices: dict[int, np.ndarray] = {
        int(label_id): transition_counts[idx]
        for idx, label_id in enumerate(label_ids)
    }

    # ---- 2. False transition target matrices ----------------------------
    # Eval transitions which are only present in pred

    # Mask for false transitions: GT is stable AND pred changes
    false_transition_mask = gt_stable_mask & (pred_src != pred_tgt)

    # count the number of stable positions per GT label (denominator for false transition rates)
    stable_counts_array = np.bincount(
        gt_src_idx[gt_stable_mask],
        minlength=num_labels,
    ).astype(np.int64)


    false_transition_counts = np.zeros((num_labels, num_labels), dtype=np.int64)
    if np.any(false_transition_mask):
        np.add.at(
            false_transition_counts,
            (
                gt_src_idx[false_transition_mask],
                pred_tgt_idx[false_transition_mask],
            ),
            1,
        )

    false_transition_matrices: dict[int, np.ndarray] = {
        int(label_id): false_transition_counts[idx]
        for idx, label_id in enumerate(label_ids)
    }
    stable_position_counts: dict[int, int] = {
        int(label_id): int(stable_counts_array[idx])
        for idx, label_id in enumerate(label_ids)
    }

    return TransitionAnalysis(
        gt_transition_matrices=gt_transition_matrices,
        false_transition_matrices=false_transition_matrices,
        stable_position_counts=stable_position_counts,
    )
