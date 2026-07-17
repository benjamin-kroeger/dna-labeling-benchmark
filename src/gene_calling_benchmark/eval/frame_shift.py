"""Per-position coding-phase drift between GT and prediction.

This module measures **relative coding-phase drift**, not the biological reading
frame.  At each genomic position covered by *both* GT and predicted CDS masks,
it reports ``|cumcount_pred − cumcount_gt| mod 3``, where cumcount is the number
of CDS bases seen from the leftmost array position up to (and including) that
position.  A value of 0 means the two annotations are in lockstep by CDS-base
count; 1 or 2 means one annotation is ahead by that many bases.

This is a structural comparison signal, not an absolute frame computation.
To compute the biological reading frame you would need the GFF ``phase`` column
per CDS feature, which is not consumed here.

Assumption: GT CDS masks are expected to represent complete, in-frame CDS
sequences (total length divisible by 3).  Partial CDS inputs will be counted
in ``n_skipped_non_divisible`` and excluded from the drift computation rather
than silently producing misleading results.

The returned dict always contains:
- ``frames``                  — list of per-position drift values (inf at non-co-CDS positions)
- ``n_skipped_non_divisible`` — sequences skipped because GT CDS length mod 3 != 0
- ``n_skipped_short``         — sequences skipped because the predicted CDS has fewer
                                than 3 bases (minimum needed for a meaningful codon-count)
- ``n_skipped_no_overlap``    — sequences skipped because GT or prediction has no CDS
                                bases at all (nothing to compare)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_frame_shift_metrics(
    gt_positive_mask: np.ndarray,
    pred_positive_mask: np.ndarray,
) -> dict:
    """Compute per-position coding-phase drift.

    Returns
    -------
    dict with keys:
        ``frames``                  — per-position drift list (inf at uncovered positions)
        ``n_skipped_non_divisible`` — 1 if this sequence was skipped due to non-divisible GT length, else 0
        ``n_skipped_short``         — 1 if this sequence was skipped due to short prediction, else 0
        ``n_skipped_no_overlap``    — 1 if this sequence was skipped because GT or pred has no CDS, else 0
    """
    gt_exon_indices = np.where(gt_positive_mask)[0]
    pred_exon_indices = np.where(pred_positive_mask)[0]

    if len(gt_exon_indices) == 0 or len(pred_exon_indices) == 0:
        return {"frames": [], "n_skipped_non_divisible": 0, "n_skipped_short": 0, "n_skipped_no_overlap": 1}

    # Fewer than 3 predicted CDS bases cannot form a codon — not meaningful to compare.
    if len(pred_exon_indices) < 3:
        return {"frames": [], "n_skipped_non_divisible": 0, "n_skipped_short": 1, "n_skipped_no_overlap": 0}

    if len(gt_exon_indices) % 3 != 0:
        logger.warning(
            "GT CDS-base count (%d) is not divisible by 3 — skipping coding-phase drift for this "
            "sequence. This usually means non-CDS positions (e.g. UTRs) have been included in the "
            "CDS mask, or the sequence contains an incomplete CDS. Provide a complete, in-frame "
            "CDS-only mask for a valid drift signal.",
            len(gt_exon_indices),
        )
        return {"frames": [], "n_skipped_non_divisible": 1, "n_skipped_short": 0, "n_skipped_no_overlap": 0}

    valid_mask = gt_positive_mask & pred_positive_mask

    frame_list = np.full(len(gt_positive_mask), np.inf)

    positions = np.arange(len(gt_positive_mask))
    gt_cumsum = np.searchsorted(gt_exon_indices, positions, side="right")
    pred_cumsum = np.searchsorted(pred_exon_indices, positions, side="right")

    frame_list[valid_mask] = np.abs(pred_cumsum[valid_mask] - gt_cumsum[valid_mask]) % 3

    return {"frames": frame_list.tolist(), "n_skipped_non_divisible": 0, "n_skipped_short": 0, "n_skipped_no_overlap": 0}
