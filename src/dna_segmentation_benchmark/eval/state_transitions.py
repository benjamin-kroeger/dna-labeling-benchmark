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
from sklearn.metrics import confusion_matrix

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
    false_transition_totals : dict[int, int]
        Per label: total number of GT-stable positions (denominator for
        the false-transition rate).
    """

    gt_transition_matrices: dict[int, np.ndarray]
    false_transition_matrices: dict[int, np.ndarray]
    false_transition_totals: dict[int, int]


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
        ``false_transition_matrices``, and ``false_transition_totals``.
    """
    label_ids = sorted(label_config.labels.keys())
    num_labels = len(label_ids)

    # Sliding window: shape (N-1, 2, 2)
    # Each window[i] = [[gt[i], gt[i+1]], [pred[i], pred[i+1]]]
    nuc_transitions = np.lib.stride_tricks.sliding_window_view(
        gt_pred_arr, (2, 2),
    )[0]

    gt_src = nuc_transitions[:, 0, 0]   # GT label at position i
    gt_tgt = nuc_transitions[:, 0, 1]   # GT label at position i+1
    pred_src = nuc_transitions[:, 1, 0]  # Pred label at position i
    pred_tgt = nuc_transitions[:, 1, 1]  # Pred label at position i+1

    gt_transition_mask = gt_src != gt_tgt
    gt_stable_mask = ~gt_transition_mask

    # ---- 1. GT transition confusion matrices ----------------------------
    per_label_pairs: dict[int, list[tuple[int, int]]] = {
        lid: [] for lid in label_ids
    }

    for window in nuc_transitions[gt_transition_mask]:
        source = int(window[0, 0])
        gt_target = int(window[0, 1])
        pred_target = int(window[1, 1])
        per_label_pairs[source].append((gt_target, pred_target))

    gt_transition_matrices: dict[int, np.ndarray] = {}
    for label_id, pairs in per_label_pairs.items():
        if len(pairs) == 0:
            gt_transition_matrices[label_id] = np.zeros(
                (num_labels, num_labels), dtype=np.int64,
            )
            continue

        gt_targets, pred_targets = zip(*pairs)
        gt_transition_matrices[label_id] = confusion_matrix(
            gt_targets, pred_targets, labels=label_ids,
        ).astype(np.int64)

    # ---- 2. False transition target matrices ----------------------------
    false_transition_matrices: dict[int, np.ndarray] = {}
    false_transition_totals: dict[int, int] = {}

    # Mask for false transitions: GT is stable AND pred changes
    false_mask = gt_stable_mask & (pred_src != pred_tgt)

    for label_id in label_ids:
        # All GT-stable positions for this label
        label_stable_mask = gt_stable_mask & (gt_src == label_id)
        false_transition_totals[label_id] = int(label_stable_mask.sum())

        # Among those, where pred falsely transitions
        label_false_mask = false_mask & (gt_src == label_id)
        false_targets = pred_tgt[label_false_mask]

        # Build a target-distribution vector
        target_counts = np.zeros(num_labels, dtype=np.int64)
        for idx, lid in enumerate(label_ids):
            target_counts[idx] = int((false_targets == lid).sum())

        false_transition_matrices[label_id] = target_counts

    return TransitionAnalysis(
        gt_transition_matrices=gt_transition_matrices,
        false_transition_matrices=false_transition_matrices,
        false_transition_totals=false_transition_totals,
    )
