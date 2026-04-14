"""Intron-chain comparison metrics.

Compares the set of intron boundaries between consecutive exon segments of a
class between ground-truth and predicted label arrays.  For exon segments the
gaps between segments correspond to introns, making this the label-array
equivalent of an intron chain comparison.

Metrics
-------
* **Intron precision** — fraction of predicted intron positions present in GT.
* **Intron recall** — fraction of GT intron positions found in the prediction.
* **Segment count delta** — over-/under-segmentation indicator.
"""

from __future__ import annotations

from .structure import ExtractedStructure, Segment
from .. import LabelConfig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_intron_chain_metrics(
        gt_structure: ExtractedStructure,
        pred_structure: ExtractedStructure,
        label_config: LabelConfig,
) -> dict:
    """Compare intron chains (inter-segment boundaries) for a single class.

    Each intron is the ``(end_i, start_{i+1})`` boundary pair between two
    consecutive segments of *class_token*.  For exon segments these gaps
    correspond to introns.

    Precision and recall are computed over the **set** of intron positions:

    * ``intron_precision`` — of the predicted introns, what fraction match a GT intron?
    * ``intron_recall``    — of the GT introns, what fraction appear in the prediction?

    Single-exon edge cases: if neither side has introns both scores are 1.0;
    if only one side has introns the other side scores 1.0 vacuously (nothing
    to miss / nothing to be wrong about).

    Parameters
    ----------
    gt_structure, pred_structure : ExtractedStructure
        Structures extracted from the GT and predicted label arrays.
    class_token : int
        Which label class to evaluate.

    Returns
    -------
    dict
        Keys: ``intron_precision`` (float), ``intron_recall`` (float),
        ``intron_count_gt`` (int), ``intron_count_pred`` (int),
        ``segment_count_gt`` (int), ``segment_count_pred`` (int),
        ``segment_count_delta`` (int).
    """
    if not label_config.intron_label:
        raise ValueError("Intron-chain comparison requires an intron label.")

    gt_segs = gt_structure.filter_by_label(label_config.intron_label)
    pred_segs = pred_structure.filter_by_label(label_config.intron_label)

    gt_introns: set[tuple[int, int]] = set(_intron_chain(gt_segs))
    pred_introns: set[tuple[int, int]] = set(_intron_chain(pred_segs))

    if len(gt_introns) == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }

    return {
        "tp": 1 if gt_introns == pred_introns else 0,
        "fp": 1 if gt_introns != pred_introns else 0,
        "fn": 1 if gt_introns != pred_introns else 0,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _intron_chain(segments: tuple[Segment, ...]) -> list[tuple[int, int]]:
    """Return ordered intron boundaries between consecutive segments.

    Each intron is ``(end_of_segment_i, start_of_segment_{i+1})``.
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
