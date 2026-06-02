"""Per-position coding-base phase drift between GT and prediction.

The returned ``gt_frames`` list holds, for each genomic position covered by *both*
GT and predicted coding masks, the absolute difference between the cumulative
counts of coding bases (mod 3). This reflects relative coding-base displacement
between the two annotations; it is **not** the biological reading frame, which
would require the GFF ``phase`` column and is not consumed here.

When the GT coding-base count is not divisible by three (for example because UTR
positions have been painted into the coding mask) the metric is skipped for that
sequence with a warning rather than aborting the run.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _get_frame_shift_metrics(
    gt_positive_mask: np.ndarray,
    pred_positive_mask: np.ndarray,
) -> dict:
    """Compute per-position coding-base phase drift."""
    gt_exon_indices = np.where(gt_positive_mask)[0]
    pred_exon_indices = np.where(pred_positive_mask)[0]

    if len(gt_exon_indices) == 0 or len(pred_exon_indices) == 0:
        return {"frames": []}

    if len(pred_exon_indices) < 3:
        return {"frames": []}

    if len(gt_exon_indices) % 3 != 0:
        logger.warning(
            "GT coding-base count (%d) is not divisible by 3 — skipping FRAMESHIFT for this "
            "sequence. This usually means non-CDS positions (e.g. UTRs) have been painted into "
            "the coding mask. Provide a CDS-only mask for a meaningful frameshift signal.",
            len(gt_exon_indices),
        )
        return {"frames": []}

    valid_mask = gt_positive_mask & pred_positive_mask

    frame_list = np.full(len(gt_positive_mask), np.inf)

    positions = np.arange(len(gt_positive_mask))
    gt_cumsum = np.searchsorted(gt_exon_indices, positions, side="right")
    pred_cumsum = np.searchsorted(pred_exon_indices, positions, side="right")

    frame_list[valid_mask] = np.abs(pred_cumsum[valid_mask] - gt_cumsum[valid_mask]) % 3

    return {"frames": frame_list.tolist()}
