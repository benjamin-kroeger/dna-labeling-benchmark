"""Per-label structural summary metrics.

Provides distribution-level and positional diagnostics that complement
the per-section metrics:

* **Length EMD** — Earth Mover's Distance between GT and predicted
  segment length distributions.  Quantifies whether the model produces
  segments of the right length.
* **Position bias histogram** — 100-bin per-nucleotide mismatch histogram
  normalised to the coding span.  Bin 0 corresponds to the start of the
  first GT coding segment; bin 99 to the end of the last.  Each nucleotide
  position that differs between GT and prediction (false negative or false
  positive within the coding span) increments its corresponding bin.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_structural_summary(
    grouped_gt_sections: list[np.ndarray],
    grouped_pred_sections: list[np.ndarray],
) -> dict:
    """Compute structural summary from pre-grouped coding-label index arrays.

    Parameters
    ----------
    grouped_gt_sections, grouped_pred_sections : list[np.ndarray]
        Contiguous-run index groups for the coding label, as returned by
        ``get_contiguous_groups``.

    Returns
    -------
    dict
        Keys: ``gt_segment_lengths``, ``pred_segment_lengths``,
        ``length_emd``, ``position_bias_histogram``.
    """
    gt_lengths = [len(g) for g in grouped_gt_sections]
    pred_lengths = [len(g) for g in grouped_pred_sections]

    emd = _wasserstein_distance(gt_lengths, pred_lengths)
    histogram = _compute_position_bias_histogram(grouped_gt_sections, grouped_pred_sections)

    return {
        "gt_segment_lengths": gt_lengths,
        "pred_segment_lengths": pred_lengths,
        "length_emd": emd,
        "position_bias_histogram": histogram,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _wasserstein_distance(a: list[int], b: list[int]) -> float:
    """Compute 1-D Wasserstein (Earth Mover's) distance.

    Uses scipy if available, falls back to a pure-Python implementation.
    """
    if not a or not b:
        return 0.0

    try:
        from scipy.stats import wasserstein_distance

        return float(wasserstein_distance(a, b))
    except ImportError:
        sa = sorted(a)
        sb = sorted(b)
        n = max(len(sa), len(sb))

        qa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sa)), sa)
        qb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sb)), sb)
        return float(np.mean(np.abs(qa - qb)))


def _compute_position_bias_histogram(
    grouped_gt_sections: list[np.ndarray],
    grouped_pred_sections: list[np.ndarray],
    n_bins: int = 100,
) -> list[int]:
    """100-bin per-nucleotide mismatch histogram over the coding span.

    Bin 0 = start of first GT coding segment, bin 99 = end of last.
    Each nucleotide that differs between GT and prediction — either a GT
    position not covered by pred (FN) or a pred position not covered by GT
    within the coding span (FP) — increments its corresponding bin.

    Parameters
    ----------
    grouped_gt_sections, grouped_pred_sections : list[np.ndarray]
        Contiguous-run index groups for the coding label.
    n_bins : int
        Number of histogram bins (default 100).

    Returns
    -------
    list[int]
        Length-``n_bins`` list; each entry is the count of mismatch
        nucleotides whose position falls in that percentile bin.
    """
    if not grouped_gt_sections:
        return [0] * n_bins

    # build a gt mask array with True for coding positions
    coding_start = int(grouped_gt_sections[0][0])
    coding_end = int(grouped_gt_sections[-1][-1])
    span = coding_end - coding_start + 1

    gt_pos = np.concatenate(grouped_gt_sections)
    gt_mask = np.zeros(span, dtype=bool)
    gt_mask[gt_pos - coding_start] = True

    # build a pred mask array with True for coding positions and clip it if the predictions are outside of the gt array
    pred_mask = np.zeros(span, dtype=bool)
    if grouped_pred_sections:
        pred_pos = np.concatenate(grouped_pred_sections)
        clipped = pred_pos[(pred_pos >= coding_start) & (pred_pos <= coding_end)]
        if clipped.size > 0:
            pred_mask[clipped - coding_start] = True

    # XOR to count mismatching positions
    mismatch_pos = np.where(gt_mask ^ pred_mask)[0]
    if mismatch_pos.size == 0:
        return [0] * n_bins

    counts, _ = np.histogram(mismatch_pos, bins=n_bins, range=(0, span))
    return counts.tolist()