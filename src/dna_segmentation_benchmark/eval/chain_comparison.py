"""Segment-chain comparison metrics.

Compares the set of intron or exon boundaries between consecutive coding
segments of a class between ground-truth and predicted label arrays.

Metrics
-------
* **Intron chain (strict / subset / superset)** — binary TP/FN/FP comparing
  the full intron-segment boundary sets.
* **Exon chain (strict / subset / superset)** — same set semantics applied to
  coding segments.  Simpler than the old LCS-based tier classification and
  directly comparable to intron chain.
* **Boundary shift** — per-transcript count and total bp offset of shifted
  segment boundaries (only for equal-count pairs).
* **Per-transcript exon recall** — fraction of GT exons exactly recovered.
* **Per-transcript hallucinated exon count** — predicted exons absent from GT.
"""

from __future__ import annotations

from .structure import ExtractedStructure, Segment
from .. import LabelConfig


# ---------------------------------------------------------------------------
# Generic chain comparison (shared by intron and exon metrics)
# ---------------------------------------------------------------------------


def _compute_chain_metrics(
        gt_structure: ExtractedStructure,
        pred_structure: ExtractedStructure,
        label: int,
        metric_prefix: str,
) -> dict:
    """Compare segment boundary sets for one label class.

    Filters both structures to segments matching *label*, converts each to a
    ``frozenset`` of ``(start, end)`` pairs, and scores three binary metrics
    via set comparison.

    Parameters
    ----------
    gt_structure, pred_structure : ExtractedStructure
        Structures extracted from the GT and predicted arrays.
    label : int
        Which label class to evaluate.
    metric_prefix : str
        Prefix for the three output keys.  For introns use ``"intron_chain"``,
        for exons use ``"exon_chain"``.

    Returns
    -------
    dict
        Three sibling dicts, each with ``tp``, ``fp``, ``fn`` counts:

        * ``{metric_prefix}`` — all-or-nothing exact match.
        * ``{metric_prefix}_subset`` — 1 iff pred ⊆ GT (all predicted segments
          are real; may miss some GT segments).
        * ``{metric_prefix}_superset`` — 1 iff pred ⊇ GT (every GT segment was
          found; may contain extra spurious ones).
    """
    gt_segs: set[tuple[int, int]] = {(s.start, s.end) for s in gt_structure.filter_by_label(label)}
    pred_segs: set[tuple[int, int]] = {(s.start, s.end) for s in pred_structure.filter_by_label(label)}

    if len(gt_segs) == 0:
        return {
            metric_prefix: {"tp": 0, "fp": 0, "fn": 0},
            f"{metric_prefix}_subset": {"tp": 0, "fp": 0, "fn": 0},
            f"{metric_prefix}_superset": {"tp": 0, "fp": 0, "fn": 0},
        }

    exact = gt_segs == pred_segs
    subset = bool(pred_segs) and pred_segs <= gt_segs
    superset = bool(pred_segs) and pred_segs >= gt_segs

    return {
        metric_prefix: {"tp": 1, "fp": 0, "fn": 0} if exact else {"tp": 0, "fp": 1, "fn": 1},
        f"{metric_prefix}_subset": {"tp": 1, "fp": 0, "fn": 0} if subset else {"tp": 0, "fp": 1, "fn": 1},
        f"{metric_prefix}_superset": {"tp": 1, "fp": 0, "fn": 0} if superset else {"tp": 0, "fp": 1, "fn": 1},
    }


# ---------------------------------------------------------------------------
# Intron chain
# ---------------------------------------------------------------------------


def _compute_intron_chain_metrics(
        gt_structure: ExtractedStructure,
        pred_structure: ExtractedStructure,
        label_config: LabelConfig,
) -> dict:
    """Compare explicit intron-label chains for one transcript pair.

    Applies sanity checks specific to intron segments (raises if multiple
    coding segments exist but no intron segments), then delegates to
    :func:`_compute_chain_metrics`.

    Returns
    -------
    dict
        Three sibling dicts with ``tp``, ``fp``, ``fn`` counts:
        ``intron_chain``, ``intron_chain_subset``, ``intron_chain_superset``.
    """
    if label_config.intron_label is None:
        raise ValueError("Intron-chain comparison requires an intron label.")

    _raise_if_introns_missing_but_inferable(gt_structure, label_config, "GT")
    _raise_if_introns_missing_but_inferable(pred_structure, label_config, "prediction")

    return _compute_chain_metrics(gt_structure, pred_structure, label_config.intron_label, "intron_chain")


# ---------------------------------------------------------------------------
# Boundary shift (per-transcript, separate from chain PR)
# ---------------------------------------------------------------------------


def _compute_boundary_shift_metrics(
        gt_structure: ExtractedStructure,
        pred_structure: ExtractedStructure,
        label: int,
) -> dict:
    """Count shifted boundary positions for equal-count segment pairs.

    Only meaningful when GT and pred have the same number of segments.  If the
    counts differ, both values are 0.  Used as a per-transcript diagnostic
    complement to the chain set-comparison metrics.

    Returns
    -------
    dict
        ``{"boundary_shift_count": int, "boundary_shift_total": int}``
    """
    gt_segs = gt_structure.filter_by_label(label)
    pred_segs = pred_structure.filter_by_label(label)

    if len(gt_segs) == 0 or len(gt_segs) != len(pred_segs):
        return {"boundary_shift_count": 0, "boundary_shift_total": 0}

    count, total = _measure_shifted_boundaries(gt_segs, pred_segs)
    return {"boundary_shift_count": count, "boundary_shift_total": total}


def _raise_if_introns_missing_but_inferable(
        structure: ExtractedStructure,
        label_config: LabelConfig,
        side_name: str,
) -> None:
    """Reject exon/CDS-only structures before intron-chain scoring."""
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

    Returns
    -------
    dict
        ``exon_recall_per_transcript`` — fraction of GT exons whose exact
        ``(start, end)`` tuple appears in the prediction.  In [0, 1].
        ``hallucinated_exon_count_per_transcript`` — number of predicted exons
        whose ``(start, end)`` does not match any GT exon.  Integer ≥ 0.
        Empty dict when GT has no exons.
    """

    gt_exons: set[tuple[int, int]] = {(s.start, s.end) for s in gt_structure.filter_by_label(label_config.coding_label)}
    pred_exons: set[tuple[int, int]] = {(s.start, s.end) for s in pred_structure.filter_by_label(label_config.coding_label)}

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


def _measure_shifted_boundaries(
        gt_segs: tuple[Segment, ...],
        pred_segs: tuple[Segment, ...],
) -> tuple[int, int]:
    """Count and sum all shifted boundary positions across every segment pair.

    Parameters
    ----------
    gt_segs, pred_segs : tuple[Segment, ...]
        Segment chains of equal length.

    Returns
    -------
    (count, total) : tuple[int, int]
        *count* — number of boundary positions that differ.
        *total* — sum of absolute position offsets across those boundaries (bp).
    """
    if not gt_segs:
        return 0, 0
    count = 0
    total = 0
    for g, p in zip(gt_segs, pred_segs):
        if g.start != p.start:
            count += 1
            total += abs(g.start - p.start)
        if g.end != p.end:
            count += 1
            total += abs(g.end - p.end)
    return count, total


def _intron_chain(segments: tuple[Segment, ...]) -> list[tuple[int, int]]:
    """Return ordered intron boundaries between consecutive segments."""
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
