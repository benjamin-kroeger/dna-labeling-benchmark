"""INDEL metric: classify GT/prediction coding mismatches.

Insertion and deletion runs (coding present on one row but not the other) are
sorted into 5'/3' extensions-or-deletions, whole insertions/deletions, and
join/split events.
"""

from __future__ import annotations

import numpy as np

from ..label_definition import LabelConfig


def _eval_indel(
    grouped_insertions: list[np.ndarray],
    grouped_deletions: list[np.ndarray],
    gt_positive_mask: np.ndarray,
    pred_positive_mask: np.ndarray,
    _label_config: LabelConfig,
) -> dict:
    """Sort insertion/deletion runs into 5'/3'/whole/join-split buckets."""
    # _classify_mismatches looks one position before/after each group, so pad
    # with one background sentinel on each side for safe access.
    padded_gt = np.concatenate(([False], gt_positive_mask.astype(bool), [False]))
    padded_pred = np.concatenate(([False], pred_positive_mask.astype(bool), [False]))
    padded_arr = np.stack((padded_gt, padded_pred), axis=0)

    # Shift indices by +1 to match the padded array layout
    padded_insertions = [g + 1 for g in grouped_insertions]
    padded_deletions = [g + 1 for g in grouped_deletions]

    ext5, ext3, joined, whole_ins = _classify_mismatches(
        grouped_indices=padded_insertions,
        gt_pred_arr=padded_arr,
    )
    del5, del3, split, whole_del = _classify_mismatches(
        grouped_indices=padded_deletions,
        gt_pred_arr=padded_arr,
    )

    return {
        "5_prime_extensions": ext5,
        "3_prime_extensions": ext3,
        "whole_insertions": whole_ins,
        "joined": joined,
        "5_prime_deletions": del5,
        "3_prime_deletions": del3,
        "whole_deletions": whole_del,
        "split": split,
    }


def _classify_mismatches(
    grouped_indices: list[np.ndarray],
    gt_pred_arr: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Sort contiguous mismatch groups into four categories.

    Depending on whether the caller is analyzing insertions or deletions the
    four buckets correspond to:

    * 5'-extensions / 5'-deletions
    * 3'-extensions / 3'-deletions
    * joins / splits
    * whole insertions / whole deletions
    """
    on_5_prime: list[np.ndarray] = []
    on_3_prime: list[np.ndarray] = []
    on_both: list[np.ndarray] = []
    on_neither: list[np.ndarray] = []

    for mismatch in grouped_indices:
        if mismatch.size == 0:
            continue

        first_idx = mismatch[0]
        last_idx = mismatch[-1]

        target_on_3_prime = bool(gt_pred_arr[0, last_idx + 1]) and bool(gt_pred_arr[1, last_idx + 1])
        target_on_5_prime = bool(gt_pred_arr[0, first_idx - 1]) and bool(gt_pred_arr[1, first_idx - 1])

        adjusted = mismatch - 1

        if target_on_3_prime and target_on_5_prime:
            on_both.append(adjusted)
        elif target_on_3_prime:
            on_5_prime.append(adjusted)
        elif target_on_5_prime:
            on_3_prime.append(adjusted)
        else:
            on_neither.append(adjusted)

    return on_5_prime, on_3_prime, on_both, on_neither
