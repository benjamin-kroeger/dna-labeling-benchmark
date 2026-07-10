"""Segment-chain comparison metrics.

Compares the set of intron or exon boundaries between consecutive coding
segments of a class between ground-truth and predicted label arrays.

Metrics
-------
* **Intron chain (strict / subset / superset)** — binary TP/FN/FP comparing
  the intron boundary sets.  An intron is the gap *between consecutive in-scope
  segments* (e.g. between CDS exons when the scope is CDS), matching
  gffcompare's intron-chain definition.  Because introns are derived from the
  same scoped segment set as the exon chain, UTR introns are excluded under CDS
  scope — a UTR-aware prediction is not penalised against a CDS-only ground
  truth — and an exact exon-chain match implies an exact intron-chain match.
* **Exon chain (exact / subset / superset)** — the same whole-chain set
  comparison applied to exon spans, reported for three populations: all
  transcripts (``exon_chain*`` — the gffcompare-parity headline, "fraction of
  transcripts whose exon set is recovered exactly"), multi-exon transcripts only
  (``exon_chain_multi*`` — apples-to-apples with the intron chain, which is
  inherently multi-exon) and single-exon transcripts only (``exon_chain_single``,
  exact tier only).  For the exact tier the all-transcript key partitions cleanly
  into ``exon_chain_multi`` + ``exon_chain_single``; the subset/superset tiers
  have no single-exon sibling, so their all-transcript keys still include the
  single-exon pairs that ``exon_chain_multi_*`` drop.
* **Boundary shift** — per-transcript count and total bp offset of shifted
  segment boundaries (only for equal-count pairs), plus the signed,
  position-tagged per-boundary offsets that drive the shift-distribution
  plots.  Reported only when GT and prediction have the same segment count
  (i.e. the chain topology is correct) *and* every predicted segment overlaps
  its positional GT counterpart (a relocated exon is not treated as a shifted
  boundary), so the offsets describe junction placement *conditioned on*
  getting the exon count right.
* **Per-transcript exon recall / precision** — fraction of GT exons exactly
  recovered, and fraction of predicted exons that are exact GT matches.
* **Per-transcript false exon count** — predicted exons absent from GT.
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


def _intron_spans_from_segments(segments: tuple[Segment, ...]) -> set[tuple[int, int]]:
    """Introns as the gaps between consecutive in-scope segments.

    This is gffcompare's intron-chain definition: an intron is the span between
    one exon's end and the next exon's start.  With ``scope=CDS`` the segments
    are CDS exons, so the gaps are CDS introns only — a splice junction inside a
    UTR is never produced, because UTR segments are not in the CDS scope.
    """
    return {
        (segments[i].end + 1, segments[i + 1].start - 1)
        for i in range(len(segments) - 1)
    }


def compute_intron_chain_metrics(
        gt_structure: ExtractedStructure,
        pred_structure: ExtractedStructure,
        label_config: LabelConfig,
        scope: BenchmarkScope | str,
) -> dict:
    """Compare intron chains for one transcript pair, scoped like the exon chain.

    An intron is the gap between consecutive in-scope segments
    (:func:`_intron_spans_from_segments`), matching gffcompare.  With
    ``scope=CDS`` these are CDS introns only: UTR introns are excluded by
    construction, so a UTR-aware predictor is not penalised against a CDS-only
    ground truth.  Both chains derive from the same scoped segment set, so an
    exact exon-chain match implies an exact intron-chain match.

    Scoring is **transcript-level (whole-chain)**: this pair contributes
    ``tp=1`` only when the entire GT intron set equals the predicted set,
    otherwise ``fp=1, fn=1``.  When :func:`summarise_counts` aggregates these
    across transcripts, the resulting precision/recall is the *fraction of
    transcripts with an exact intron-chain match* — **not** the fraction of
    individual introns correctly predicted.  Single-exon transcripts have no
    introns, contribute empty :class:`Counts` and drop out, so the denominator
    matches the ``exon_chain_multi`` population exactly.

    Returns
    -------
    dict
        Three sibling dicts with ``tp``, ``fp``, ``fn`` counts:
        ``intron_chain``, ``intron_chain_subset``, ``intron_chain_superset``.
    """
    if label_config.intron_label is None:
        raise ValueError("Intron-chain comparison requires an intron label to be defined in the label configuration.")

    gt_spans = _intron_spans_from_segments(_segments_for_scope(gt_structure, scope, label_config))
    pred_spans = _intron_spans_from_segments(_segments_for_scope(pred_structure, scope, label_config))

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
    """Compare exon chains for one transcript pair within a benchmark scope.

    The exon "chain" is the set of scoped exon ``(start, end)`` spans, with
    adjacent compatible labels already collapsed by :func:`_segments_for_scope`
    (so a ``5'UTR`` + ``CDS`` exon counts once).  Scoring is **transcript-level
    (whole-chain, all-or-nothing)**, mirroring
    :func:`compute_intron_chain_metrics`: the pair scores ``tp=1`` only when the
    predicted span set exactly equals the GT set, otherwise ``fp=1, fn=1``.
    Because a mismatch books *both* an fp and an fn, per pair ``fp == fn``; once
    :func:`summarise_counts` aggregates across transcripts, precision == recall
    and the headline value is simply the **fraction of transcripts whose entire
    exon chain was recovered exactly** — "how many transcripts did I get right"
    — *not* a per-exon rate.

    Match tiers (each an independent all-or-nothing comparison, encoding the
    *direction* of error, not just pass/fail):

    * ``exact``    — ``pred == gt``.
    * ``subset``   — ``pred ⊆ gt``: every predicted exon is real, but GT exons
      may be missing (a conservative / under-calling predictor).
    * ``superset`` — ``pred ⊇ gt``: every GT exon is present, but the prediction
      adds spurious exons (an over-calling / over-splitting predictor).

    Population variants — the same per-pair result reported over three GT
    populations:

    * ``exon_chain`` / ``_subset`` / ``_superset`` — **all** GT transcripts,
      single- and multi-exon alike.  This is the **gffcompare-parity headline**:
      gffcompare reports transcript-level accuracy over every transcript, so this
      key is the directly comparable "fraction of transcripts fully recovered".
    * ``exon_chain_multi`` / ``_subset`` / ``_superset`` — **multi-exon GT only**
      (a one-span "chain" is not really a chain).  The apples-to-apples partner
      of the intron-chain tiers, whose population is already multi-exon-only
      (single-exon transcripts have no introns).
    * ``exon_chain_single`` — **single-exon GT only**, exact tier only (a lone
      exon has no subset/superset structure to distinguish).

    The inapplicable population bucket is empty :class:`Counts`, so each
    variant's aggregated denominator covers only its own population — the same
    drop-out idiom :func:`compute_intron_chain_metrics` uses for single-exon
    pairs.  Only the exact tier partitions cleanly:
    ``exon_chain == exon_chain_multi ⊕ exon_chain_single``.  The subset/superset
    tiers have no single-exon sibling, so ``exon_chain_subset`` /
    ``exon_chain_superset`` still include single-exon pairs that
    ``exon_chain_multi_*`` exclude.

    Returns
    -------
    dict
        The seven chain-tier :class:`Counts` above, plus the per-transcript
        diagnostics merged in from :func:`_compute_boundary_shift_from_segments`
        and :func:`_compute_exon_recovery_from_segments`, a
        ``transcript_match_class`` label (when classifiable) and the signed
        ``segment_count_delta`` (``len(pred) - len(gt)``).

        When the GT has zero in-scope segments there is nothing to score, so
        only the three bare ``exon_chain`` / ``_subset`` / ``_superset`` keys
        (empty :class:`Counts`) are returned — the multi/single tiers and the
        per-transcript diagnostics are omitted on this path.
    """
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

    is_single_exon = len(gt_segs) == 1
    chain_metrics = {
        "exon_chain": Counts(tp=1) if exact else Counts(fp=1, fn=1),
        "exon_chain_subset": Counts(tp=1) if subset else Counts(fp=1, fn=1),
        "exon_chain_superset": Counts(tp=1) if superset else Counts(fp=1, fn=1),
        # Population-split siblings (see docstring): the exon_chain* keys above
        # stay all-transcript; these re-book the same per-pair result over the
        # single- vs multi-exon GT population.  The inapplicable bucket is empty
        # Counts so each rate's denominator covers only its population — the same
        # idiom compute_intron_chain_metrics uses to drop single-exon pairs.
        # Single-exon gets the exact tier only, so only the exact tier partitions
        # cleanly (exon_chain == exon_chain_multi + exon_chain_single).
        "exon_chain_multi": Counts() if is_single_exon else (Counts(tp=1) if exact else Counts(fp=1, fn=1)),
        "exon_chain_multi_subset": Counts() if is_single_exon else (Counts(tp=1) if subset else Counts(fp=1, fn=1)),
        "exon_chain_multi_superset": Counts() if is_single_exon else (Counts(tp=1) if superset else Counts(fp=1, fn=1)),
        "exon_chain_single": (Counts(tp=1) if exact else Counts(fp=1, fn=1)) if is_single_exon else Counts(),
    }
    chain_metrics.update(_compute_boundary_shift_from_segments(gt_segs, pred_segs))
    chain_metrics.update(_compute_exon_recovery_from_segments(gt_segs, pred_segs))

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
    """Count and characterise shifted boundaries for a scope segment chain.

    Returns the per-transcript scalar summary (``boundary_shift_count``,
    ``boundary_shift_total``) together with ``boundary_shift_offsets`` — the
    list of signed, position-tagged per-boundary records that drives the
    boundary-shift distribution plots.  All three are empty/zero unless the
    two chains have matching, non-zero length (correct exon-count topology)
    *and* every predicted segment overlaps its positional GT counterpart.
    Boundaries can only be paired position-by-position when the counts agree,
    and the offsets are only meaningful when the paired segments genuinely
    correspond — a relocated/substituted exon is not a "shifted boundary" and
    would otherwise inject a spurious large offset into the distribution.
    """
    if (
        len(gt_segs) == 0
        or len(gt_segs) != len(pred_segs)
        or not all(g.overlaps(p) for g, p in zip(gt_segs, pred_segs))
    ):
        return {
            "boundary_shift_count": 0,
            "boundary_shift_total": 0,
            "boundary_shift_offsets": [],
        }

    count, total, offsets = _measure_shifted_boundaries(gt_segs, pred_segs)
    return {
        "boundary_shift_count": count,
        "boundary_shift_total": total,
        "boundary_shift_offsets": offsets,
    }


# ---------------------------------------------------------------------------
# Per-transcript exon recovery (distribution view)
# ---------------------------------------------------------------------------


def _compute_exon_recovery_from_segments(
        gt_exons: tuple[Segment, ...],
        pred_exons: tuple[Segment, ...],
) -> dict:
    """Per-transcript exon recall, precision, and false-exon count for a scope.

    Recall and precision are fractions in [0, 1] over exactly-matched
    ``(start, end)`` exons; ``false_exon_count`` is the raw number of predicted
    exons with no exact GT match.  Precision is ``0.0`` when the prediction has
    no exons in scope (no true positives to credit).  Empty dict when GT has no
    exons in scope.
    """
    gt_set: set[tuple[int, int]] = {(s.start, s.end) for s in gt_exons}
    pred_set: set[tuple[int, int]] = {(s.start, s.end) for s in pred_exons}

    if not gt_set:
        return {}

    shared = gt_set & pred_set
    return {
        "exon_recall_per_transcript": len(shared) / len(gt_set),
        "exon_precision_per_transcript": len(shared) / len(pred_set) if pred_set else 0.0,
        "false_exon_count_per_transcript": len(pred_set - gt_set),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _measure_shifted_boundaries(
        gt_segs: tuple[Segment, ...],
        pred_segs: tuple[Segment, ...],
) -> tuple[int, int, list[dict]]:
    """Measure every shifted boundary position across a segment chain.

    The two chains are compared position-by-position (they must have equal
    length); each boundary whose predicted coordinate differs from the
    ground-truth coordinate is both counted and recorded individually.

    Parameters
    ----------
    gt_segs, pred_segs : tuple[Segment, ...]
        Segment chains of equal length, ordered by array position.

    Returns
    -------
    (count, total, offsets) : tuple[int, int, list[dict]]
        *count* — number of boundary positions that differ.
        *total* — sum of absolute position offsets across those boundaries (bp).
        *offsets* — one record per **shifted** boundary, each a dict with keys

        ``offset``
            Signed offset ``pred_edge - gt_edge`` in array coordinates.  A
            positive value means the predicted edge lies to the right (higher
            index, array-3') of the matching GT edge.  The sign therefore
            follows the same array-orientation convention as the boundary
            precision landscape and is **not** strand-resolved — a biological
            donor/acceptor split is intentionally deferred until minus-strand
            arrays are reverse-complemented upstream.
        ``position``
            ``"terminal"`` for the chain's outer start/end (the transcript
            TSS/TES in array orientation), ``"internal"`` for every interior
            splice-site boundary.  This separates inherently fuzzy transcript
            ends from precisely defined splice junctions.

        Only boundaries that actually differ are recorded, so the distribution
        describes the *conditional* shift magnitude ("given a junction is
        misplaced, by how much") rather than overall recall.
    """
    if not gt_segs:
        return 0, 0, []
    count = 0
    total = 0
    offsets: list[dict] = []
    last_index = len(gt_segs) - 1
    for index, (g, p) in enumerate(zip(gt_segs, pred_segs)):
        if g.start != p.start:
            count += 1
            total += abs(g.start - p.start)
            offsets.append(
                {
                    "offset": p.start - g.start,
                    "position": "terminal" if index == 0 else "internal",
                }
            )
        if g.end != p.end:
            count += 1
            total += abs(g.end - p.end)
            offsets.append(
                {
                    "offset": p.end - g.end,
                    "position": "terminal" if index == last_index else "internal",
                }
            )
    return count, total, offsets

