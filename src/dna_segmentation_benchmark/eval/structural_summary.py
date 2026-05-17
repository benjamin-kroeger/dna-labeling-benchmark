"""Per-label structural summary metrics.

Provides distribution-level and positional diagnostics that complement
the per-section metrics:

* **Length EMD** — Earth Mover's Distance between GT and predicted
  segment length distributions.  Quantifies whether the model produces
  segments of the right length.
* **Position bias histograms** — 100-bin per-nucleotide mismatch
  histograms normalised to the coding span.  Bin 0 corresponds to the
  start of the first GT coding segment; bin 99 to the end of the last.
  Two histograms are emitted so that under-prediction and over-prediction
  can be told apart visually:

  * ``position_bias_histogram_fn`` — GT positions not covered by the
    prediction (under-prediction).
  * ``position_bias_histogram_fp`` — predicted positions inside the GT
    coding span that are not in GT (over-prediction).
"""

from __future__ import annotations

import numpy as np


#: Number of bins used by the position-bias histograms. Kept as a module-level
#: constant because the aggregator in ``evaluate_predictors.py`` reshapes the
#: flattened histogram lists by this width — changing it in only one place would
#: silently break aggregation.
POSITION_BIAS_HISTOGRAM_BINS: int = 100


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
        ``length_emd``, ``position_bias_histogram_fn``,
        ``position_bias_histogram_fp``.
    """
    gt_lengths = [len(g) for g in grouped_gt_sections]
    pred_lengths = [len(g) for g in grouped_pred_sections]

    emd = _wasserstein_distance(gt_lengths, pred_lengths)
    fn_hist, fp_hist = _compute_position_bias_histograms(
        grouped_gt_sections, grouped_pred_sections
    )

    return {
        "gt_segment_lengths": gt_lengths,
        "pred_segment_lengths": pred_lengths,
        "length_emd": emd,
        "position_bias_histogram_fn": fn_hist,
        "position_bias_histogram_fp": fp_hist,
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


def _compute_position_bias_histograms(
    grouped_gt_sections: list[np.ndarray],
    grouped_pred_sections: list[np.ndarray],
) -> tuple[list[int], list[int]]:
    """Per-nucleotide mismatch histograms over the coding span, split FN / FP.

    Bin 0 = start of first GT coding segment, bin 99 = end of last.
    Two histograms are returned so that under-prediction (FN) and
    over-prediction (FP) within the GT coding span can be distinguished.

    * **FN bins** — count GT coding positions not covered by the prediction.
    * **FP bins** — count predicted coding positions inside the GT coding
      span that are not in GT.  Predicted positions outside the coding
      span are clipped out, keeping the histogram bounded to the gene locus.

    Parameters
    ----------
    grouped_gt_sections, grouped_pred_sections : list[np.ndarray]
        Contiguous-run index groups for the coding label.

    Returns
    -------
    (fn_hist, fp_hist) : tuple[list[int], list[int]]
        Both lists have length :data:`POSITION_BIAS_HISTOGRAM_BINS`.
        ``fn_hist[i] + fp_hist[i]`` is the total mismatch count in percentile
        bin ``i``.
    """
    n_bins = POSITION_BIAS_HISTOGRAM_BINS
    zeros = [0] * n_bins
    if not grouped_gt_sections:
        return zeros, zeros

    # build a gt mask array with True for coding positions
    coding_start = int(grouped_gt_sections[0][0])
    coding_end = int(grouped_gt_sections[-1][-1])
    span = coding_end - coding_start + 1

    gt_pos = np.concatenate(grouped_gt_sections)
    gt_mask = np.zeros(span, dtype=bool)
    gt_mask[gt_pos - coding_start] = True

    # build a pred mask array with True for coding positions and clip it
    # to the GT coding span (predicted bases outside the locus are ignored).
    pred_mask = np.zeros(span, dtype=bool)
    if grouped_pred_sections:
        pred_pos = np.concatenate(grouped_pred_sections)
        clipped = pred_pos[(pred_pos >= coding_start) & (pred_pos <= coding_end)]
        if clipped.size > 0:
            pred_mask[clipped - coding_start] = True

    fn_pos = np.where(gt_mask & ~pred_mask)[0]
    fp_pos = np.where(~gt_mask & pred_mask)[0]

    if fn_pos.size:
        fn_counts, _ = np.histogram(fn_pos, bins=n_bins, range=(0, span))
        fn_hist = fn_counts.tolist()
    else:
        fn_hist = list(zeros)

    if fp_pos.size:
        fp_counts, _ = np.histogram(fp_pos, bins=n_bins, range=(0, span))
        fp_hist = fp_counts.tolist()
    else:
        fp_hist = list(zeros)

    return fn_hist, fp_hist
