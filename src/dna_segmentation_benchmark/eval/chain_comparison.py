"""Segment-chain comparison metrics.

Compares the set of intron or exon boundaries between consecutive coding
segments of a class between ground-truth and predicted label arrays.

Metrics
-------
* **Intron chain (strict / subset / superset)** — binary TP/FN/FP comparing
  the full intron-segment boundary sets.  When ``splice_donor_label`` and
  ``splice_acceptor_label`` are defined in the ``LabelConfig``, each logical
  intron spans from the *start* of its donor segment to the *end* of its
  acceptor segment (``DONOR.start → ACCEPTOR.end``), matching the splice-
  junction positions that gffcompare uses.  Without splice-site labels the raw
  intron-segment boundaries are used instead.
* **Exon chain (strict / subset / superset)** — same set semantics applied to
  coding segments, directly comparable to intron chain.
* **Boundary shift** — per-transcript count and total bp offset of shifted
  segment boundaries (only for equal-count pairs).
* **Per-transcript exon recall** — fraction of GT exons exactly recovered.
* **Per-transcript hallucinated exon count** — predicted exons absent from GT.
"""

from __future__ import annotations

from .statistics import Counts
from .structure import ExtractedStructure, Segment, extract_scoped_segments
from .transcript_classification import _classify_segment_match
from .. import LabelConfig
from ..label_definition import BenchmarkScope


# ---------------------------------------------------------------------------
# Generic chain comparison (shared by intron and exon metrics)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Intron chain
# ---------------------------------------------------------------------------


def _extract_logical_intron_spans(
        structure: ExtractedStructure,
        label_config: LabelConfig,
) -> set[tuple[int, int]]:
    """Return (start, end) spans for each logical intron in *structure*.

    When both ``splice_donor_label`` and ``splice_acceptor_label`` are set,
    each logical intron is formed by a ``DONOR (INTRON*) ACCEPTOR`` run and
    spans ``(DONOR.start, ACCEPTOR.end)`` — the actual splice-junction
    positions.  Malformed runs (donor without a following acceptor) are
    silently skipped.

    When splice-site labels are not configured, falls back to the raw
    intron-label segment boundaries.
    """
    donor_label = label_config.splice_donor_label
    acceptor_label = label_config.splice_acceptor_label
    intron_label = label_config.intron_label

    if donor_label is None or acceptor_label is None:
        return {(s.start, s.end) for s in structure.filter_by_label(intron_label)}

    # If splice-site labels are configured but absent from this structure
    # (e.g. a model that only outputs intron labels), fall back to raw intron
    # segment boundaries so the comparison remains consistent with the pred side.
    if not any(s.label == donor_label for s in structure.segments):
        return {(s.start, s.end) for s in structure.filter_by_label(intron_label)}

    spans: set[tuple[int, int]] = set()
    segments = structure.segments
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.label == donor_label:
            donor_start = seg.start
            j = i + 1
            while j < len(segments) and segments[j].label == intron_label:
                j += 1
            if j < len(segments) and segments[j].label == acceptor_label:
                spans.add((donor_start, segments[j].end))
                i = j + 1
                continue
        i += 1
    return spans


def compute_intron_chain_metrics(
        gt_structure: ExtractedStructure,
        pred_structure: ExtractedStructure,
        label_config: LabelConfig,
) -> dict:
    """Compare logical intron chains for one transcript pair.

    Extracts logical intron spans via :func:`_extract_logical_intron_spans`
    (which groups ``DONOR + INTRON* + ACCEPTOR`` into a single span when
    splice-site labels are defined), then performs the same strict / subset /
    superset set comparison as the exon chain metrics.

    Scoring is **transcript-level (whole-chain)**: this pair contributes
    ``tp=1`` only when the entire GT intron set equals the predicted set,
    otherwise ``fp=1, fn=1``.  When :func:`summarise_counts` aggregates these
    across transcripts, the resulting precision/recall is the *fraction of
    transcripts with an exact intron-chain match* — **not** the fraction of
    individual introns correctly predicted.  For an intron-level gradation of
    "nearly right" chains, see the per-transcript soft-exon scalars in
    :func:`_compute_per_transcript_exon_soft_metrics`.

    Returns
    -------
    dict
        Three sibling dicts with ``tp``, ``fp``, ``fn`` counts:
        ``intron_chain``, ``intron_chain_subset``, ``intron_chain_superset``.
    """
    if label_config.intron_label is None:
        raise ValueError("Intron-chain comparison requires an intron label to be defined in the label configuration.")

    _raise_if_introns_missing_but_inferable(gt_structure, label_config, "GT")
    _raise_if_introns_missing_but_inferable(pred_structure, label_config, "prediction")

    gt_spans = _extract_logical_intron_spans(gt_structure, label_config)
    pred_spans = _extract_logical_intron_spans(pred_structure, label_config)

    prefix = "intron_chain"
    if not gt_spans:
        return {
            prefix: Counts(),
            f"{prefix}_subset": Counts(),
            f"{prefix}_superset": Counts(),
        }

    exact = gt_spans == pred_spans
    subset = bool(pred_spans) and pred_spans <= gt_spans
    superset = bool(pred_spans) and pred_spans >= gt_spans

    return {
        prefix: Counts(tp=1) if exact else Counts(fp=1, fn=1),
        f"{prefix}_subset": Counts(tp=1) if subset else Counts(fp=1, fn=1),
        f"{prefix}_superset": Counts(tp=1) if superset else Counts(fp=1, fn=1),
    }


def _segments_for_scope(
        structure: ExtractedStructure,
        scope: BenchmarkScope | str,
        label_config: LabelConfig,
) -> tuple[Segment, ...]:
    """Return scope-resolved segments, collapsing adjacent compatible labels."""
    return extract_scoped_segments(structure, label_config.scope_tokens(scope))


def compute_scoped_chain_metrics(
        gt_structure: ExtractedStructure,
        pred_structure: ExtractedStructure,
        label_config: LabelConfig,
        scope: BenchmarkScope | str,
) -> dict:
    """Compare exonic segment chains for an explicit benchmark scope."""
    gt_segs = _segments_for_scope(gt_structure, scope, label_config)
    pred_segs = _segments_for_scope(pred_structure, scope, label_config)

    if len(gt_segs) == 0:
        return {
            "exon_chain": Counts(),
            "exon_chain_subset": Counts(),
            "exon_chain_superset": Counts(),
        }

    gt_set: set[tuple[int, int]] = {(s.start, s.end) for s in gt_segs}
    pred_set: set[tuple[int, int]] = {(s.start, s.end) for s in pred_segs}

    exact = gt_set == pred_set
    subset = bool(pred_set) and pred_set <= gt_set
    superset = bool(pred_set) and pred_set >= gt_set

    chain_metrics = {
        "exon_chain": Counts(tp=1) if exact else Counts(fp=1, fn=1),
        "exon_chain_subset": Counts(tp=1) if subset else Counts(fp=1, fn=1),
        "exon_chain_superset": Counts(tp=1) if superset else Counts(fp=1, fn=1),
    }
    chain_metrics.update(_compute_boundary_shift_from_segments(gt_segs, pred_segs))
    chain_metrics.update(_compute_soft_metrics_from_segments(gt_segs, pred_segs))

    match_cls = _classify_segment_match(gt_segs, pred_segs)
    if match_cls is not None:
        chain_metrics["transcript_match_class"] = match_cls.value

    chain_metrics["segment_count_delta"] = len(pred_segs) - len(gt_segs)
    return chain_metrics


# ---------------------------------------------------------------------------
# Boundary shift (per-transcript, separate from chain PR)
# ---------------------------------------------------------------------------


def _compute_boundary_shift_from_segments(
        gt_segs: tuple[Segment, ...],
        pred_segs: tuple[Segment, ...],
) -> dict:
    """Count shifted boundaries for a pre-selected scope segment chain."""
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
    intron = label_config.intron_label
    if intron is None:
        return

    coding_segs = _segments_for_scope(
        structure,
        BenchmarkScope.TRANSCRIPT_EXON,
        label_config,
    )
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

    gt_exons = _segments_for_scope(gt_structure, BenchmarkScope.TRANSCRIPT_EXON, label_config)
    pred_exons = _segments_for_scope(pred_structure, BenchmarkScope.TRANSCRIPT_EXON, label_config)

    return _compute_soft_metrics_from_segments(gt_exons, pred_exons)


def _compute_soft_metrics_from_segments(
        gt_exons: tuple[Segment, ...],
        pred_exons: tuple[Segment, ...],
) -> dict:
    """Per-transcript recall and hallucination counts for a scope segment set."""
    gt_set: set[tuple[int, int]] = {(s.start, s.end) for s in gt_exons}
    pred_set: set[tuple[int, int]] = {(s.start, s.end) for s in pred_exons}

    if not gt_set:
        return {}

    shared = gt_set & pred_set
    return {
        "exon_recall_per_transcript": len(shared) / len(gt_set),
        "hallucinated_exon_count_per_transcript": len(pred_set - gt_set),
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

