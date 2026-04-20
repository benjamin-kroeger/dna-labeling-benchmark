"""Per-label structural summary metrics.

Provides distribution-level and positional diagnostics that complement
the per-section metrics:

* **Length EMD** — Earth Mover's Distance between GT and predicted
  segment length distributions.  Quantifies whether the model produces
  segments of the right length.
* **Position bias histogram** — 100-bin error-location histogram
  normalised to the coding span.  Bin 0 corresponds to the start of the
  first GT coding segment; bin 99 to the end of the last.  Every
  unmatched GT segment (FN) and every unmatched pred segment within the
  coding span (FP) increments all bins it overlaps with.
"""

from __future__ import annotations

from .junction_errors import _greedy_match
from .structure import ExtractedStructure, Segment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_structural_summary(
    gt_structure: ExtractedStructure,
    pred_structure: ExtractedStructure,
    class_token: int,
) -> dict:
    """Compute structural summary for a single class token.

    Parameters
    ----------
    gt_structure, pred_structure : ExtractedStructure
        Structures from the GT and predicted arrays.
    class_token : int
        Which label class to evaluate.

    Returns
    -------
    dict
        Keys: ``gt_segment_lengths``, ``pred_segment_lengths``,
        ``length_emd``, ``position_bias_histogram``.
    """
    gt_segs = gt_structure.filter_by_label(class_token)
    pred_segs = pred_structure.filter_by_label(class_token)

    gt_lengths = [s.length for s in gt_segs]
    pred_lengths = [s.length for s in pred_segs]

    emd = _wasserstein_distance(gt_lengths, pred_lengths)
    histogram = _compute_position_bias_histogram(gt_segs, pred_segs)

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
        import numpy as np
        qa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sa)), sa)
        qb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sb)), sb)
        return float(np.mean(np.abs(qa - qb)))


def _compute_position_bias_histogram(
    gt_segs: tuple[Segment, ...],
    pred_segs: tuple[Segment, ...],
    n_bins: int = 100,
) -> list[int]:
    """100-bin error-location histogram over the coding span.

    Bin 0 = start of first GT coding segment, bin 99 = end of last.
    Each unmatched GT segment (FN) and each unmatched pred segment within
    the coding span (FP) increments every bin it overlaps with.

    Parameters
    ----------
    gt_segs, pred_segs : tuple[Segment, ...]
        Ordered (by position) segments for the target class.
    n_bins : int
        Number of histogram bins (default 100).

    Returns
    -------
    list[int]
        Length-``n_bins`` list; each entry is the count of error regions
        whose position range overlaps that percentile bin.
    """
    if not gt_segs:
        return [0] * n_bins

    coding_start = gt_segs[0].start   # segments are ordered by position
    coding_end = gt_segs[-1].end
    span = coding_end - coding_start + 1
    if span == 0:
        return [0] * n_bins

    _, unmatched_gt, unmatched_pred = _greedy_match(gt_segs, pred_segs)

    bins: list[int] = [0] * n_bins

    for g_idx in unmatched_gt:
        seg = gt_segs[g_idx]
        _fill_bins(bins, seg.start, seg.end, coding_start, span, n_bins)

    for p_idx in unmatched_pred:
        seg = pred_segs[p_idx]
        start = max(seg.start, coding_start)
        end = min(seg.end, coding_end)
        if start <= end:
            _fill_bins(bins, start, end, coding_start, span, n_bins)

    return bins


def _fill_bins(
    bins: list[int],
    start: int,
    end: int,
    coding_start: int,
    span: int,
    n_bins: int,
) -> None:
    """Increment every bin that the interval [start, end] overlaps."""
    bin_lo = int((start - coding_start) / span * n_bins)
    bin_hi = int((end - coding_start + 1) / span * n_bins)
    bin_lo = max(0, min(n_bins - 1, bin_lo))
    bin_hi = max(0, min(n_bins - 1, bin_hi))
    for b in range(bin_lo, bin_hi + 1):
        bins[b] += 1
