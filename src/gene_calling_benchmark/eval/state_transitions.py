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
   categories, each stored as a ``(L, L)`` pred_src × pred_tgt matrix per
   GT-stable label.

   The classification is **run-anchored**: for a GT-stable run ``R = [a, b]``
   with current label ``L``, preceding label ``P`` and following label ``N``,
   the predictor's whole trajectory across ``R`` decides — not the labels of a
   single transition in isolation.

   * **Late catch-up**: the predictor *entered* the run still in ``P``
     (``pred[a] == P``) and the **first** transition inside ``R`` is ``P → L``.
     Only that one transition is a late catch-up.
   * **Premature**: the predictor *leaves* the run in ``N`` (``pred[b] == N``)
     and the **last** transition inside ``R`` is ``L → N``.  Only that one
     transition is premature.
   * **Spurious**: **every other** transition inside ``R`` — a fabrication the
     local GT trajectory cannot explain (an invented intron carved out of a real
     exon, an exon invented inside a real intron, …).

   The anchors are what make a slip a slip: a genuine early exit *stays* exited.
   A round-trip excursion (leave the state and return within the same run) is a
   fabrication and is **spurious on both** of its transitions.  A transition can
   never be both late and premature — that would require ``P == L`` or
   ``L == N``, impossible at a real GT boundary.

   Off-track pred transitions at GT boundaries (``pred_src != gt_src`` AND
   ``pred_src != pred_tgt``) are always **spurious** (context label = ``gt_src``
   before the boundary).

   When no previous (or next) GT run exists (array edges), ``P``/``N`` is set to
   ``L`` as a sentinel: ``pred[a] == L`` forces the first transition's target
   away from ``L``, so the anchor can never accidentally match.
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
        anchored late catch-ups (predictor entered the run in ``prev_GT`` and
        its first in-run transition is ``prev_GT → curr_GT``).
    premature_matrices : dict[int, np.ndarray]
        Per GT-stable label, a ``(L, L)`` matrix counting anchored premature
        transitions (predictor leaves the run in ``next_GT`` and its last in-run
        transition is ``curr_GT → next_GT``).
    spurious_matrices : dict[int, np.ndarray]
        Per GT-stable label, a ``(L, L)`` matrix counting all other false
        transitions — fabrications the surrounding GT trajectory cannot explain.
    stable_position_counts : dict[int, int]
        Per label: total number of GT-stable positions (denominator for
        the false-transition rate).
    """

    gt_transition_matrices: dict[int, np.ndarray]
    late_catchup_matrices: dict[int, np.ndarray]
    premature_matrices: dict[int, np.ndarray]
    spurious_matrices: dict[int, np.ndarray]
    stable_position_counts: dict[int, int]


def compute_state_change_errors(
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

    # A (2, 2) sliding window needs at least two positions. Shorter sequences
    # (a length-0 or length-1 chunk — e.g. a tiny transcript in an online batch)
    # have no transition to analyse, so return an all-zero result instead of
    # letting sliding_window_view raise. A length-1 chunk still contributes its
    # single position to the GT-stable denominator.
    if gt_pred_arr.shape[1] < 2:
        empty = lambda: {int(lid): np.zeros((num_labels, num_labels), dtype=np.int64) for lid in label_ids}
        stable = {int(lid): 0 for lid in label_ids}
        if gt_pred_arr.shape[1] == 1:
            gt0 = int(gt_pred_arr[0, 0])
            if gt0 in stable:
                stable[gt0] = 1
        return TransitionAnalysis(
            gt_transition_matrices=empty(),
            late_catchup_matrices=empty(),
            premature_matrices=empty(),
            spurious_matrices=empty(),
            stable_position_counts=stable,
        )

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
    # (a) GT stable, pred changes                          -> classified below
    # (b) GT boundary where pred_src != gt_src AND pred changes (off-track)
    #     -> always spurious (the predictor had already left the GT source state)
    in_run_mask = gt_stable_mask & (pred_src != pred_tgt)
    off_track_boundary_mask = gt_transition_mask & (pred_src != gt_src) & (pred_src != pred_tgt)

    # denominator: GT-stable positions only
    stable_counts_array = np.bincount(
        gt_src_idx[gt_stable_mask],
        minlength=num_labels,
    ).astype(np.int64)

    stable_position_counts: dict[int, int] = {
        int(label_id): int(stable_counts_array[idx]) for idx, label_id in enumerate(label_ids)
    }

    # Classify against the GT-stable run each false transition falls in.
    # GT transition window positions: window i where gt[i] != gt[i+1]
    gt_vals = gt_pred_arr[0]
    pred_vals = gt_pred_arr[1]
    num_positions = gt_vals.shape[0]
    gt_transition_positions = np.where(np.diff(gt_vals))[0]

    # GT-stable runs: run k spans positions [run_starts[k], run_ends[k]]
    run_starts = np.concatenate(([0], gt_transition_positions + 1))
    run_ends = np.concatenate((gt_transition_positions, [num_positions - 1]))

    late_catchup_counts = np.zeros((num_labels, num_labels, num_labels), dtype=np.int64)
    premature_counts = np.zeros((num_labels, num_labels, num_labels), dtype=np.int64)
    spurious_counts = np.zeros((num_labels, num_labels, num_labels), dtype=np.int64)

    in_run_pos = np.where(in_run_mask)[0]  # ascending
    is_late = np.zeros(in_run_pos.shape, dtype=bool)
    is_premature = np.zeros(in_run_pos.shape, dtype=bool)

    if in_run_pos.size:
        # in_run_pos is never a GT boundary, so this is the index of its run
        run_id = np.searchsorted(gt_transition_positions, in_run_pos, side='right')
        run_start = run_starts[run_id]
        run_end = run_ends[run_id]

        curr_GT = gt_src[in_run_pos]
        # sentinel = curr_GT at the array edges: can never match (see module docstring)
        prev_GT = np.where(run_id > 0, gt_vals[np.maximum(run_start - 1, 0)], curr_GT)
        next_GT = np.where(
            run_id < len(gt_transition_positions),
            gt_vals[np.minimum(run_end + 1, num_positions - 1)],
            curr_GT,
        )

        # first / last false transition within each run (in_run_pos is sorted)
        first_in_run = np.empty(in_run_pos.shape, dtype=bool)
        first_in_run[0] = True
        first_in_run[1:] = run_id[1:] != run_id[:-1]
        last_in_run = np.empty(in_run_pos.shape, dtype=bool)
        last_in_run[-1] = True
        last_in_run[:-1] = run_id[:-1] != run_id[1:]

        # Anchored: pred entered the run in prev_GT / leaves it in next_GT.
        # (pred is constant from run_start up to the first in-run transition, and
        # from the last one up to run_end, so only the free endpoint needs checking.)
        is_late = first_in_run & (pred_vals[run_start] == prev_GT) & (pred_tgt[in_run_pos] == curr_GT)
        is_premature = last_in_run & (pred_vals[run_end] == next_GT) & (pred_src[in_run_pos] == curr_GT)
        is_premature &= ~is_late  # cannot be both at a real GT boundary; guard conservation anyway

    late_pos = in_run_pos[is_late]
    premature_pos = in_run_pos[is_premature]
    spurious_pos = np.concatenate(
        (in_run_pos[~(is_late | is_premature)], np.where(off_track_boundary_mask)[0])
    )

    for counts, positions in (
        (late_catchup_counts, late_pos),
        (premature_counts, premature_pos),
        (spurious_counts, spurious_pos),
    ):
        if positions.size:
            np.add.at(
                counts,
                (gt_src_idx[positions], pred_src_idx[positions], pred_tgt_idx[positions]),
                1,
            )

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
