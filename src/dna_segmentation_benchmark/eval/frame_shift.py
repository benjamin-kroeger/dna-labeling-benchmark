import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _get_frame_shift_metrics(
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
        coding_value: int,
) -> dict:
    """Compute per-position reading-frame deviation."""
    gt_exon_indices = np.where(gt_labels == coding_value)[0]
    pred_exon_indices = np.where(pred_labels == coding_value)[0]

    if len(gt_exon_indices) == 0 or len(pred_exon_indices) == 0:
        return {"gt_frames": []}

    assert len(gt_exon_indices) % 3 == 0, "There is no clear codon usage"

    gt_codons = gt_exon_indices.reshape(-1, 3)
    possible_pred_codons = sliding_window_view(pred_exon_indices, 3)

    gt_codon_view = gt_codons.view([("", gt_codons.dtype)] * 3).reshape(-1)
    pred_codon_view = possible_pred_codons.view(
        [("", possible_pred_codons.dtype)] * 3
    ).reshape(-1)
    _common_codons = np.intersect1d(gt_codon_view, pred_codon_view)

    valid_mask = (
            np.isin(np.arange(len(gt_labels)), gt_exon_indices)
            & np.isin(np.arange(len(gt_labels)), pred_exon_indices)
    )

    frame_list = np.full(len(gt_labels), np.inf)

    gt_cumsum = np.searchsorted(gt_exon_indices, np.arange(len(gt_labels)), side="right")
    pred_cumsum = np.searchsorted(pred_exon_indices, np.arange(len(gt_labels)), side="right")

    frame_list[valid_mask] = np.abs(pred_cumsum[valid_mask] - gt_cumsum[valid_mask]) % 3

    return {"gt_frames": frame_list}