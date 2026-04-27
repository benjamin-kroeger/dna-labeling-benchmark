"""State-transition analysis for GT vs prediction arrays.

Provides two complementary, GT-anchored views of transition quality:

1. **GT Transition Confusion Matrices** – At every position where the ground
   truth changes label *and* the predictor is still in the same source label
   (``pred_src == gt_src``), record the GT target vs predicted target.  One
   ``(L, L)`` confusion matrix per GT source label; rows = GT target, cols =
   predicted target.  Transitions where ``pred_src != gt_src`` are excluded
   (the predictor already left the source state prematurely).

2. **Classified False Transition Matrices** – Every pred transition that does
   *not* correspond to a valid GT transition is classified into one of three
   context-aware categories, each stored as a ``(L, L)`` pred_src × pred_tgt
   matrix per GT-stable label:

   * **Late catch-up** (``pred_src == prev_GT`` AND ``pred_tgt == curr_GT``):
     the predictor was stuck in the *previous* GT state and only now transitions
     into the current one.  Requires lookbehind: what was the GT state before
     this stable run?
   * **Premature** (``pred_src == curr_GT`` AND ``pred_tgt == next_GT``): the
     predictor leaves the current GT state too early, jumping to the *next* GT
     state before the GT transition occurs.  Requires lookahead: what is the
     next GT state?
   * **Spurious**: everything else — transitions that cannot be explained by
     the surrounding GT trajectory.

   Off-track pred transitions at GT boundaries (``pred_src != gt_src`` AND
   ``pred_src != pred_tgt``) are always **spurious** (context label = ``gt_src``
   before the boundary, same as before).

   When no previous (or next) GT transition exists, ``prev_GT``/``next_GT``
   is set to ``curr_GT`` as a sentinel so it can never accidentally match.
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
        Per GT source label, a ``(L, L)`` confusion matrix (rows = GT target,
        cols = predicted target) counting only source-matched transitions where
        ``pred_src == gt_src``.
    late_catchup_matrices : dict[int, np.ndarray]
        Per GT-stable label, a ``(L, L)`` pred_src × pred_tgt matrix counting
        true late catch-ups (``pred_src == prev_GT``, ``pred_tgt == curr_GT``).
    premature_matrices : dict[int, np.ndarray]
        Per GT-stable label, a ``(L, L)`` matrix counting true premature
        transitions (``pred_src == curr_GT``, ``pred_tgt == next_GT``).
    spurious_matrices : dict[int, np.ndarray]
        Per GT-stable label, a ``(L, L)`` matrix counting all other false
        transitions (not explainable by the surrounding GT trajectory).
    stable_position_counts : dict[int, int]
        Per label: total number of GT-stable positions (denominator for
        the false-transition rate).
    """

    gt_transition_matrices: dict[int, np.ndarray]
    late_catchup_matrices: dict[int, np.ndarray]
    premature_matrices: dict[int, np.ndarray]
    spurious_matrices: dict[int, np.ndarray]
    stable_position_counts: dict[int, int]


def _compute_state_change_errors(
    gt_pred_arr: np.ndarray,
    label_config: LabelConfig,
) -> TransitionAnalysis:
    """Compute GT transition matrices and classified false transition matrices.

    Parameters
    ----------
    gt_pred_arr : np.ndarray
        Shape ``(2, N)`` where row 0 is ground truth and row 1 is prediction.
    label_config : LabelConfig
        Defines the set of valid integer labels.

    Returns
    -------
    TransitionAnalysis
        Frozen dataclass with GT transition matrices, three classified false
        transition matrices, and stable position counts.
    """
    label_ids = sorted(label_config.labels.keys())
    num_labels = len(label_ids)
    label_id_array = np.asarray(label_ids, dtype=gt_pred_arr.dtype)

    # Sliding window: shape (N-1, 2, 2)
    # Each window[i] = [[gt[i], gt[i+1]], [pred[i], pred[i+1]]]
    nuc_transitions = np.lib.stride_tricks.sliding_window_view(
        gt_pred_arr,
        (2, 2),
    )[0]

    gt_src = nuc_transitions[:, 0, 0]
    gt_tgt = nuc_transitions[:, 0, 1]
    pred_src = nuc_transitions[:, 1, 0]
    pred_tgt = nuc_transitions[:, 1, 1]

    gt_transition_mask = gt_src != gt_tgt
    gt_stable_mask = ~gt_transition_mask

    gt_src_idx = np.searchsorted(label_id_array, gt_src)
    gt_tgt_idx = np.searchsorted(label_id_array, gt_tgt)
    pred_src_idx = np.searchsorted(label_id_array, pred_src)
    pred_tgt_idx = np.searchsorted(label_id_array, pred_tgt)

    # ---- 1. GT transition confusion matrices (one per source label) -----
    # Only count where pred_src == gt_src: predictor was in the correct state.
    valid_transition_mask = gt_transition_mask & (pred_src == gt_src)

    gt_transition_counts = np.zeros((num_labels, num_labels, num_labels), dtype=np.int64)
    if np.any(valid_transition_mask):
        np.add.at(
            gt_transition_counts,
            (
                gt_src_idx[valid_transition_mask],
                gt_tgt_idx[valid_transition_mask],
                pred_tgt_idx[valid_transition_mask],
            ),
            1,
        )

    gt_transition_matrices: dict[int, np.ndarray] = {
        int(label_id): gt_transition_counts[idx] for idx, label_id in enumerate(label_ids)
    }

    # ---- 2. Classified false transition matrices -------------------------
    # Collect all positions where a false pred transition occurs:
    # (a) GT stable, pred changes
    # (b) GT boundary where pred_src != gt_src AND pred changes (off-track)
    premature_at_boundary_mask = gt_transition_mask & (pred_src != gt_src) & (pred_src != pred_tgt)
    false_transition_mask = (gt_stable_mask & (pred_src != pred_tgt)) | premature_at_boundary_mask

    # denominator: GT-stable positions only
    stable_counts_array = np.bincount(
        gt_src_idx[gt_stable_mask],
        minlength=num_labels,
    ).astype(np.int64)

    stable_position_counts: dict[int, int] = {
        int(label_id): int(stable_counts_array[idx]) for idx, label_id in enumerate(label_ids)
    }

    # Classify using lookbehind (prev_GT) and lookahead (next_GT).
    # GT transition window positions: window i where gt[i] != gt[i+1]
    gt_vals = gt_pred_arr[0]
    gt_transition_positions = np.where(np.diff(gt_vals))[0]

    late_catchup_counts = np.zeros((num_labels, num_labels, num_labels), dtype=np.int64)
    premature_counts = np.zeros((num_labels, num_labels, num_labels), dtype=np.int64)
    spurious_counts = np.zeros((num_labels, num_labels, num_labels), dtype=np.int64)

    if np.any(false_transition_mask):
        false_pos = np.where(false_transition_mask)[0]
        curr_GT = gt_src[false_pos]
        p_src = pred_src[false_pos]
        p_tgt = pred_tgt[false_pos]
        curr_GT_idx = gt_src_idx[false_pos]
        p_src_idx = pred_src_idx[false_pos]
        p_tgt_idx = pred_tgt_idx[false_pos]

        if len(gt_transition_positions) > 0:
            # Lookbehind: index of the last GT transition strictly before false_pos
            last_trans_idx = np.searchsorted(gt_transition_positions, false_pos, side='right') - 1
            has_prev = last_trans_idx >= 0
            # gt_transition_positions[k] is window k where gt[k] != gt[k+1], so gt[k] is prev_GT
            prev_GT = np.where(
                has_prev,
                gt_vals[gt_transition_positions[np.maximum(last_trans_idx, 0)]],
                curr_GT,  # sentinel: can never match prev_GT check
            )

            # Lookahead: index of the first GT transition at or after false_pos
            next_trans_idx = np.searchsorted(gt_transition_positions, false_pos, side='right')
            has_next = next_trans_idx < len(gt_transition_positions)
            # gt_transition_positions[k]+1 is the position AFTER the boundary = next stable label
            next_GT = np.where(
                has_next,
                gt_vals[gt_transition_positions[np.minimum(next_trans_idx, len(gt_transition_positions) - 1)] + 1],
                curr_GT,  # sentinel: can never match next_GT check
            )
        else:
            prev_GT = curr_GT
            next_GT = curr_GT

        is_late = (p_src == prev_GT) & (p_tgt == curr_GT)
        is_premature = (p_src == curr_GT) & (p_tgt == next_GT)
        is_spurious = ~is_late & ~is_premature

        if np.any(is_late):
            np.add.at(late_catchup_counts, (curr_GT_idx[is_late], p_src_idx[is_late], p_tgt_idx[is_late]), 1)
        if np.any(is_premature):
            np.add.at(premature_counts, (curr_GT_idx[is_premature], p_src_idx[is_premature], p_tgt_idx[is_premature]), 1)
        if np.any(is_spurious):
            np.add.at(spurious_counts, (curr_GT_idx[is_spurious], p_src_idx[is_spurious], p_tgt_idx[is_spurious]), 1)

    late_catchup_matrices: dict[int, np.ndarray] = {
        int(label_id): late_catchup_counts[idx] for idx, label_id in enumerate(label_ids)
    }
    premature_matrices: dict[int, np.ndarray] = {
        int(label_id): premature_counts[idx] for idx, label_id in enumerate(label_ids)
    }
    spurious_matrices: dict[int, np.ndarray] = {
        int(label_id): spurious_counts[idx] for idx, label_id in enumerate(label_ids)
    }

    return TransitionAnalysis(
        gt_transition_matrices=gt_transition_matrices,
        late_catchup_matrices=late_catchup_matrices,
        premature_matrices=premature_matrices,
        spurious_matrices=spurious_matrices,
        stable_position_counts=stable_position_counts,
    )
