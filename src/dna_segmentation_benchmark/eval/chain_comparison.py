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
    """Compare explicit intron-label chains for one transcript pair.

    The metric expects introns to be present as segments with
    ``label_config.intron_label``.  Each intron-chain element is the boundary
    pair ``(end_i, start_{i+1})`` between consecutive intron-label segments.
    The resulting chain is compared as a set and scored as an all-or-nothing
    transcript-level TP/FP/FN.

    If the structure contains multiple coding segments but no intron-label
    segments, the metric raises ``ValueError`` instead of silently returning a
    vacuous result.  In that situation the input likely came from exon/CDS-only
    arrays.  Call ``benchmark_gt_vs_pred_single(..., infer_introns=True)`` or
    provide explicit intron labels before requesting
    :class:`~dna_segmentation_benchmark.label_definition.EvalMetrics.STRUCTURAL_COHERENCE`.

    Single-exon edge case: if GT has no intron-chain elements, the result is
    ``{"tp": 0, "fp": 0, "fn": 0}``, because there is no intron chain to
    evaluate.

    Notes
    -----
    The public benchmark functions can infer intron-label segments from
    background gaps between adjacent coding segments before this metric runs.
    That inference happens at the raw-array level, so all metrics see the same
    transformed arrays when ``infer_introns=True``.

    Parameters
    ----------
    gt_structure, pred_structure : ExtractedStructure
        Structures extracted from the GT and predicted label arrays.
    label_config : LabelConfig
        Supplies ``coding_label`` and ``intron_label``.

    Returns
    -------
    dict
        Strict transcript-level counts: ``tp``, ``fp``, and ``fn``.
    """
    if label_config.intron_label is None:
        raise ValueError("Intron-chain comparison requires an intron label.")

    gt_segs = gt_structure.filter_by_label(label_config.intron_label)
    pred_segs = pred_structure.filter_by_label(label_config.intron_label)
    _raise_if_introns_missing_but_inferable(gt_structure, label_config, "GT")
    _raise_if_introns_missing_but_inferable(pred_structure, label_config, "prediction")

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


def _raise_if_introns_missing_but_inferable(
    structure: ExtractedStructure,
    label_config: LabelConfig,
    side_name: str,
) -> None:
    """Reject exon/CDS-only structures before intron-chain scoring.

    Multiple coding segments with no intron-label segment indicate that the
    array has exon-like runs separated by background.  Scoring such an array as
    an explicit intron chain would incorrectly treat the transcript as having
    no introns.  The caller must either enable ``infer_introns`` in the public
    benchmark entry point or provide arrays where introns are already labelled.
    """
    coding = label_config.coding_label
    intron = label_config.intron_label
    if coding is None or intron is None:
        return

    coding_segs = structure.filter_by_label(coding)
    intron_segs = structure.filter_by_label(intron)
    if len(coding_segs) > 1 and not intron_segs:
        raise ValueError(
            f"{side_name} contains multiple coding segments but no intron-label "
            "segments. Pass infer_introns=True to benchmark_gt_vs_pred_single "
            "or benchmark_gt_vs_pred_multiple if introns should be inferred "
            "from coding gaps, or provide explicit intron labels before "
            "requesting STRUCTURAL_COHERENCE.",
        )


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
