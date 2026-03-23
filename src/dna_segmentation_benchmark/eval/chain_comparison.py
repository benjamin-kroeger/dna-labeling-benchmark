"""Gap-chain comparison metrics.

Compares the ordered sequence of gaps between consecutive segments of a
class between ground-truth and predicted label arrays.  For exon segments
the gaps correspond to introns, making this the label-agnostic equivalent
of an intron chain comparison.

Metrics
-------
* **Gap chain match** — exact equality of gap boundary sequences.
* **Gap chain LCS ratio** — longest common subsequence of gap boundary
  pairs as a fraction of the longer chain.  Provides ordering-aware
  partial credit.
* **Segment count delta** — over-/under-segmentation indicator.
"""

from __future__ import annotations

from .structure import ExtractedStructure, Segment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_gap_chain_metrics(
    gt_structure: ExtractedStructure,
    pred_structure: ExtractedStructure,
    class_token: int,
) -> dict:
    """Compare gap chains (inter-segment boundaries) for a single class.

    The gap chain is the ordered sequence of ``(end_i, start_{i+1})`` pairs
    between consecutive segments of *class_token*.  For exon segments these
    gaps correspond to introns, but the metric is label-agnostic.

    Also reports segment counts and delta for over-/under-segmentation
    diagnosis.

    Parameters
    ----------
    gt_structure, pred_structure : ExtractedStructure
        Structures extracted from the GT and predicted label arrays.
    class_token : int
        Which label class to evaluate.

    Returns
    -------
    dict
        Keys: ``gap_chain_match`` (bool), ``gap_chain_lcs_ratio`` (float),
        ``gap_count_match`` (bool), ``gap_count_gt`` (int),
        ``gap_count_pred`` (int), ``segment_count_gt`` (int),
        ``segment_count_pred`` (int), ``segment_count_delta`` (int).
    """
    gt_segs = gt_structure.filter_by_label(class_token)
    pred_segs = pred_structure.filter_by_label(class_token)

    gt_gaps = _gap_chain(gt_segs)
    pred_gaps = _gap_chain(pred_segs)

    n_gt_gaps = len(gt_gaps)
    n_pred_gaps = len(pred_gaps)
    max_len = max(n_gt_gaps, n_pred_gaps)

    lcs_len = _lcs_length(gt_gaps, pred_gaps) if max_len > 0 else 0

    n_gt_segs = len(gt_segs)
    n_pred_segs = len(pred_segs)

    return {
        "gap_chain_match": gt_gaps == pred_gaps,
        "gap_chain_lcs_ratio": lcs_len / max_len if max_len > 0 else 1.0,
        "gap_count_match": n_gt_gaps == n_pred_gaps,
        "gap_count_gt": n_gt_gaps,
        "gap_count_pred": n_pred_gaps,
        "segment_count_gt": n_gt_segs,
        "segment_count_pred": n_pred_segs,
        "segment_count_delta": n_pred_segs - n_gt_segs,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _gap_chain(segments: tuple[Segment, ...]) -> list[tuple[int, int]]:
    """Return ordered gap boundaries between consecutive segments.

    Each gap is ``(end_of_segment_i, start_of_segment_{i+1})``.
    """
    return [(segments[i].end, segments[i + 1].start) for i in range(len(segments) - 1)]


def _boundaries(segments: tuple[Segment, ...]) -> list[tuple[int, int]]:
    """Return a list of ``(start, end)`` boundary pairs."""
    return [(s.start, s.end) for s in segments]


def _lcs_length(
    seq_a: list[tuple[int, int]],
    seq_b: list[tuple[int, int]],
) -> int:
    """Length of the longest common subsequence of boundary pairs."""
    n = len(seq_a)
    m = len(seq_b)

    # Two-row DP
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (m + 1)

    return max(prev)
