"""Intron-chain comparison metrics.

Compares the set of intron boundaries between consecutive exon segments of a
class between ground-truth and predicted label arrays.  For exon segments the
gaps between segments correspond to introns, making this the label-array
equivalent of an intron chain comparison.

Metrics
-------
* **Intron chain (strict)** — binary TP/FN/FP: 1 iff the full GT and pred
  intron sets are identical.
* **Per-transcript exon recall** — fraction of GT exons exactly recovered
  by the prediction (``shared / n_gt_exons``).  Per-sequence continuous
  score in [0, 1]; aggregated as a distribution across transcripts so
  cases like "9 of 10 exons right" are visible in the tail.
* **Per-transcript hallucinated exon count** — number of predicted exons
  whose ``(start, end)`` does **not** exactly match any GT exon.  A
  precision-side companion to the recall metric that captures spurious
  extra predictions without conflating them with boundary errors.
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
# Per-transcript structural soft metrics (distribution view)
# ---------------------------------------------------------------------------


def _compute_per_transcript_exon_soft_metrics(
        gt_structure: ExtractedStructure,
        pred_structure: ExtractedStructure,
        label_config: LabelConfig,
) -> dict:
    """Per-transcript continuous exon-recovery metrics.

    Complements the strict ``intron_chain`` (all-or-nothing) and the
    corpus-level ``perfect_boundary_hit`` (sums TP/FN across all
    sequences) by yielding *per-transcript* values whose distribution
    across transcripts can be plotted as a histogram.  This surfaces
    graduated agreement — "9 of 10 exons right" is visible in the tail
    of the distribution instead of collapsing to 0 or 1.

    Values:

    * ``exon_recall_per_transcript`` — fraction of GT exons whose exact
      ``(start, end)`` tuple appears in the prediction, i.e.
      ``shared / n_gt_exons``.  In [0, 1].  Absent when GT has no exons.
    * ``hallucinated_exon_count_per_transcript`` — number of predicted
      exons whose ``(start, end)`` does **not** match any GT exon exactly.
      Precision-side companion to the recall metric.  Integer ≥ 0.
      Absent when GT has no exons (the transcript is not applicable).

    Operates on the coding label so it also works for predictors that
    do not emit explicit intron tokens.
    """
    coding = label_config.coding_label

    gt_exons: set[tuple[int, int]] = {(s.start, s.end) for s in gt_structure.filter_by_label(coding)}
    pred_exons: set[tuple[int, int]] = {(s.start, s.end) for s in pred_structure.filter_by_label(coding)}

    if not gt_exons:
        return {}

    shared = gt_exons & pred_exons
    return {
        "exon_recall_per_transcript": len(shared) / len(gt_exons),
        "hallucinated_exon_count_per_transcript": len(pred_exons - gt_exons),
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
