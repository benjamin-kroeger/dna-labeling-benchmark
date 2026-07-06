"""Strand-aware, locus-based mapping of GT to prediction transcripts.

This module provides the core logic for associating ground-truth (GT) and
predicted transcripts from GFF/GTF files using a gffcompare-style algorithm:

1. **Locus grouping** — GT transcripts are clustered into loci (sets of
   mutually reachable overlapping transcripts) by an O(n log n) coordinate
   sweep.
2. **Junction-F1 scoring** — every predicted transcript in a locus is
   scored against all GT transcripts using a continuous junction-F1 metric:
   ``2 × |shared junctions| / (|pred junctions| + |ref junctions|)``.
   Single-exon fallback: reciprocal overlap fraction.
3. **Optimal 1:1 assignment** — the Hungarian algorithm
   (``scipy.optimize.linear_sum_assignment``) maximises the total junction-F1
   across all (GT, pred) pairs in a locus.  Each GT transcript and each
   prediction is assigned at most once per predictor.

Two locus matching modes are supported (see :class:`LocusMatchingMode`):

* **FULL_DISCOVERY** — maximise the number of 1:1 matches per locus.
  Unmatched GT transcripts yield null-prediction FN pairs; unmatched
  predictions yield null-GT FP pairs.  Best for multi-isoform tools.
* **BEST_PER_LOCUS** — keep only the single highest-scoring (GT, pred) pair
  per locus and drop all others.  Best for single-transcript tools
  (e.g. Augustus).

Public API
----------
- :class:`LocusMatchingMode`
- :class:`MatchClass`
- :class:`PredictionMatch`
- :class:`TranscriptMapping`
- :func:`map_transcripts`
- :func:`build_paired_arrays`
- :func:`export_mapping_table`
"""

from __future__ import annotations

import csv
import dataclasses
import logging
from collections import defaultdict
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .eval.utils import _sweep_cluster
from .feature_roles import (
    FeatureRoleMap,
    PredFeatureRoleMapInput,
    exonic_feature_types,
    feature_role_paint_plan,
    normalize_feature_role_map,
    normalize_pred_feature_role_maps,
)
from .io_utils import DEFAULT_TRANSCRIPT_TYPES, collect_gff
from .label_definition import AnnotationMode, LabelConfig

logger = logging.getLogger(__name__)

# Sentinel prefix for synthetic GT entries created for unmatched predictions.
_UNMATCHED_PRED_PREFIX = "__unmatched_pred__"


# ---------------------------------------------------------------------------
# Locus matching mode
# ---------------------------------------------------------------------------


class LocusMatchingMode(str, Enum):
    """Controls how GT and prediction transcripts are paired within a locus.

    Attributes
    ----------
    FULL_DISCOVERY
        All GT transcripts participate; the optimal set of 1:1 matches
        (maximising total junction-F1) is computed via the Hungarian
        algorithm.  Unmatched GT transcripts contribute null-prediction FN
        pairs; unmatched predictions contribute null-GT FP pairs.

        Use this mode for multi-isoform tools (Helixer, SegmentNT, …).

    BEST_PER_LOCUS
        Every GT locus is scored **exactly once per predictor**, so a predictor
        is never rewarded for emitting more isoforms nor penalised for emitting
        fewer than the reference.  The per-transcript denominator is therefore
        the full set of GT loci and is directly comparable across predictors.
        Each (locus, predictor) pair resolves to one of three entries:

        * **Case A — match.**  The predictor's single highest-scoring (GT, pred)
          pair in the locus is kept; all other isoforms and predictions in that
          locus are dropped.

        * **Case C — overlap, wrong structure.**  The predictor matched nothing
          in the locus but has a transcript overlapping it (no shared junction).
          That prediction is paired against the real GT isoform it best overlaps
          (``junction_f1 = 0``), so a confidently-wrong call is penalised on its
          coinciding bases rather than silently dropped.

        * **Case B — clean miss.**  The predictor has no transcript overlapping
          the locus at all: a locus-level FN against the representative (longest)
          isoform, paired with a null prediction.

        Use this mode for single-transcript tools (Augustus, …).

        Each predictor accumulates independently, and Case C only fires for a
        predictor that matched nothing in the locus, so the GT isoform it scores
        against is never also counted in a Case A entry for that same predictor —
        no GT base is double-counted within a predictor's per-transcript numbers.

        Precision note: predictions that overlap **no** GT locus become
        intergenic FP entries (null GT).  But an *extra* prediction that overlaps
        a locus the predictor **already matched** (Case A) is **dropped** from the
        per-transcript view — counting it would force a second entry for that
        locus and double-count the GT bases the matched pair already covers.
        Those dropped bases are still charged by the **global** ``nucleotide``
        precision (union over all predicted bases), so per-transcript precision
        can read slightly optimistically relative to the global numbers; pair the
        two views.  Scoring every overlapping isoform is what ``FULL_DISCOVERY``
        is for.
    """

    FULL_DISCOVERY = "full_discovery"
    BEST_PER_LOCUS = "best_per_locus"


# ---------------------------------------------------------------------------
# Match classification
# ---------------------------------------------------------------------------


class MatchClass(str, Enum):
    """Structural relationship between a predicted and a GT transcript.

    Listed in priority order (highest quality first).
    """

    EXACT = "exact"
    """Identical intron chains — every splice site matches exactly."""

    CONTAINED = "contained"
    """Pred intron chain ⊆ GT intron chain *and* pred span ⊆ GT span."""

    CONTAINS = "contains"
    """GT intron chain ⊆ pred intron chain *and* GT span ⊆ pred span."""

    SHARED_JUNCTION = "shared_junction"
    """At least one shared splice junction between GT and prediction."""

    OVERLAPPING = "overlapping"
    """Coordinate overlap only — no shared junctions (includes single-exon)."""


# ---------------------------------------------------------------------------
# Public Pydantic models
# ---------------------------------------------------------------------------


class PredictionMatch(BaseModel):
    """A single predicted transcript matched to a GT transcript."""

    model_config = {"frozen": True}

    predictor_name: str
    transcript_id: str
    start: int
    end: int
    match_class: MatchClass
    base_overlap: int
    """Number of overlapping bases between GT and prediction spans."""
    junction_f1: float = 0.0
    """Continuous junction-F1 score used for assignment (0–1)."""


class TranscriptMapping(BaseModel):
    """Association between one GT transcript and its matched predictions.

    Attributes
    ----------
    seqid : str
        Chromosome / scaffold identifier.
    strand : str
        ``'+'`` or ``'-'``.
    gt_id : str
        Ground-truth transcript identifier.  For unmatched predictions
        this starts with :data:`_UNMATCHED_PRED_PREFIX`.
    gt_start, gt_end : int
        Genomic coordinates of the evaluation window (1-based, inclusive).
    is_unmatched_prediction : bool
        ``True`` when no GT transcript was assigned to this prediction.
        The GT array is null (all-background) for unmatched predictions.
    matched_predictions : list[PredictionMatch]
        Per-predictor matches.  Empty when no predictor was matched to
        this GT transcript.
    fn_for_predictors : list[str]
        When set, this is a ``BEST_PER_LOCUS`` Case-B (clean miss) entry that
        counts as a miss **only** for the named predictor(s) — those with no
        prediction overlapping the locus.  Other predictors skip it, so every GT
        locus is counted once per predictor without charging a predictor that
        matched (Case A) or overlapped (Case C) the same locus.
    """

    model_config = {"frozen": True}

    seqid: str
    strand: str
    gt_id: str
    gt_start: int
    gt_end: int
    is_unmatched_prediction: bool = False
    matched_predictions: list[PredictionMatch] = []
    fn_for_predictors: list[str] = []


# ---------------------------------------------------------------------------
# Internal: transcript metadata
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _TranscriptInfo:
    """Lightweight per-transcript summary used only during matching."""

    gff_id: str
    start: int  # 1-based inclusive
    end: int  # 1-based inclusive
    intron_chain: frozenset[tuple[int, int]]
    """Unordered set of ``(exon_end, next_exon_start)`` intron boundaries."""
    is_single_exon: bool


# ---------------------------------------------------------------------------
# Internal: intron-chain index
# ---------------------------------------------------------------------------


def _build_intron_chain_index(
    df: pd.DataFrame,
    seqid: str,
    exon_types: list[str],
) -> dict[str, frozenset[tuple[int, int]]]:
    """Build a ``{transcript_id: intron_chain}`` index for one chromosome.

    Groups all exon-type features on *seqid* by their parent transcript in
    a single pass, making this O(n) in the number of exon rows.
    """
    exon_df = df[
        (df["seqid"] == seqid)
        & df["type"].isin(exon_types)
        & df["parent"].notna()
        & df["start"].notna()
        & df["end"].notna()
    ]
    if exon_df.empty:
        return {}

    # Group exon rows by parent in a single pass over plain arrays.  Going via
    # ``groupby(...).sort_values(...)`` paid pandas' per-call overhead once per
    # transcript (tens of thousands of tiny frames); extracting the columns once
    # and grouping in Python is the same result far cheaper.
    parents = exon_df["parent"].to_numpy()
    starts = exon_df["start"].to_numpy()
    ends = exon_df["end"].to_numpy()
    rows_by_parent: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for parent_id, start, end in zip(parents, starts, ends):
        rows_by_parent[str(parent_id)].append((int(start), int(end)))

    index: dict[str, frozenset[tuple[int, int]]] = {}
    for parent_id, rows in rows_by_parent.items():
        rows.sort()

        # Merge overlapping/adjacent rows before deriving junctions.  The default
        # EXON_INTRON role map sends both ``exon`` and nested ``CDS`` features to
        # the exon role, so a GENCODE/RefSeq transcript carries interleaved
        # exon/CDS rows for the same exon.  Without merging, consecutive
        # exon→CDS transitions would emit spurious reverse-coordinate "introns"
        # (e.g. exon [100,200] + CDS [150,200] → (200, 150)) that no prediction
        # can match, corrupting junction-F1 scoring and the Hungarian assignment.
        merged: list[tuple[int, int]] = []
        for start, end in rows:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        if len(merged) < 2:
            index[parent_id] = frozenset()
        else:
            index[parent_id] = frozenset(
                (merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)
            )

    return index


def _build_transcript_infos(
    df: pd.DataFrame,
    seqid: str,
    strand: str,
    transcript_types: list[str],
    chain_index: dict[str, frozenset[tuple[int, int]]],
) -> list[_TranscriptInfo]:
    """Extract :class:`_TranscriptInfo` for every transcript on seqid+strand."""
    rows = df[
        (df["seqid"] == seqid)
        & (df["strand"] == strand)
        & df["type"].isin(transcript_types)
        & df["gff_id"].notna()
        & df["start"].notna()
        & df["end"].notna()
    ]

    infos: list[_TranscriptInfo] = []
    for gff_id, start, end in zip(
        rows["gff_id"].to_numpy(), rows["start"].to_numpy(), rows["end"].to_numpy()
    ):
        gff_id = str(gff_id)
        chain = chain_index.get(gff_id, frozenset())
        infos.append(
            _TranscriptInfo(
                gff_id=gff_id,
                start=int(start),
                end=int(end),
                intron_chain=chain,
                is_single_exon=len(chain) == 0,
            )
        )
    return infos


# ---------------------------------------------------------------------------
# Internal: locus grouping
# ---------------------------------------------------------------------------


def _build_loci(
    transcripts: list[_TranscriptInfo],
) -> list[list[_TranscriptInfo]]:
    """Cluster transcripts into loci by coordinate overlap."""
    return _sweep_cluster(transcripts, key=lambda t: (t.start, t.end))


def _find_preds_overlapping_locus(
    locus: list[_TranscriptInfo],
    pred_infos: list[_TranscriptInfo],
) -> list[_TranscriptInfo]:
    """Return predictions whose span overlaps the locus span."""
    locus_start = min(t.start for t in locus)
    locus_end = max(t.end for t in locus)
    return [p for p in pred_infos if p.start <= locus_end and p.end >= locus_start]


# ---------------------------------------------------------------------------
# Internal: pair scoring
# ---------------------------------------------------------------------------


def _base_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> int:
    """Number of overlapping bases between two 1-based inclusive intervals."""
    return max(0, min(end_a, end_b) - max(start_a, start_b) + 1)


def _classify_pair(gt: _TranscriptInfo, pred: _TranscriptInfo) -> MatchClass:
    """Assign a :class:`MatchClass` to one (GT, prediction) pair.

    Evaluation order (highest priority first):

    1. :attr:`~MatchClass.EXACT` — identical intron chains.
    2. :attr:`~MatchClass.CONTAINED` — pred's introns ⊆ GT's introns and
       pred's span is contained within GT's span.
    3. :attr:`~MatchClass.CONTAINS` — GT's introns ⊆ pred's introns and
       GT's span is contained within pred's span.
    4. :attr:`~MatchClass.SHARED_JUNCTION` — at least one shared intron.
    5. :attr:`~MatchClass.OVERLAPPING` — coordinate overlap only, or either
       transcript is single-exon.
    """
    if gt.is_single_exon or pred.is_single_exon:
        if gt.start == pred.start and gt.end == pred.end:
            return MatchClass.EXACT
        if pred.start >= gt.start and pred.end <= gt.end:
            return MatchClass.CONTAINED
        if gt.start >= pred.start and gt.end <= pred.end:
            return MatchClass.CONTAINS
        return MatchClass.OVERLAPPING

    if gt.intron_chain == pred.intron_chain:
        return MatchClass.EXACT

    if pred.intron_chain <= gt.intron_chain and pred.start >= gt.start and pred.end <= gt.end:
        return MatchClass.CONTAINED

    if gt.intron_chain <= pred.intron_chain and gt.start >= pred.start and gt.end <= pred.end:
        return MatchClass.CONTAINS

    if gt.intron_chain & pred.intron_chain:
        return MatchClass.SHARED_JUNCTION

    return MatchClass.OVERLAPPING


def _compute_assignment_score(gt: _TranscriptInfo, pred: _TranscriptInfo) -> float:
    """Continuous assignment score in [0, 1].

    * **Multi-exon vs multi-exon**: junction F1 =
      ``2 × |shared| / (|pred junctions| + |ref junctions|)``.
      Returns 0.0 when there are no shared junctions.
    * **Any single-exon**: reciprocal overlap fraction =
      ``overlap / max(gt_span, pred_span)``.
    * **No coordinate overlap**: 0.0.

    Using a continuous score instead of categorical class codes allows the
    Hungarian algorithm to find globally optimal assignments, where even a
    cluster of moderate-F1 pairs beats one perfect pair leaving the rest
    unmatched.

    Hard-zero caveat: a multi-exon prediction that overlaps a GT transcript
    spatially but shares **no** splice junctions scores exactly 0.0 (the
    single-exon overlap fallback is deliberately not applied) and is left
    unmatched — pairing structurally incompatible transcripts would produce
    misleading per-transcript metrics.  Downstream consequence: in
    ``BEST_PER_LOCUS`` mode, if no prediction shares a junction with the best
    GT transcript the locus contributes no pair at all, not even a
    degraded-quality match.
    """
    overlap = _base_overlap(gt.start, gt.end, pred.start, pred.end)
    if overlap == 0:
        return 0.0

    if not gt.is_single_exon and not pred.is_single_exon:
        shared = len(gt.intron_chain & pred.intron_chain)
        if shared == 0:
            return 0.0
        p = shared / len(pred.intron_chain)
        r = shared / len(gt.intron_chain)
        return 2.0 * p * r / (p + r)

    # Single-exon (one or both): fall back to overlap fraction.
    gt_len = gt.end - gt.start + 1
    pred_len = pred.end - pred.start + 1
    return overlap / max(gt_len, pred_len)


# ---------------------------------------------------------------------------
# Internal: optimal per-locus assignment
# ---------------------------------------------------------------------------


def _assign_optimal_locus(
    gt_locus: list[_TranscriptInfo],
    pred_locus: list[_TranscriptInfo],
    predictor_name: str,
    mode: LocusMatchingMode,
) -> dict[str, PredictionMatch]:
    """Optimal 1:1 assignment for one (locus, predictor) pair.

    Builds an n×m score matrix (n = GT count, m = pred count), then:

    * **FULL_DISCOVERY** — runs ``scipy.optimize.linear_sum_assignment``
      (Hungarian algorithm, O(n³)) to maximise total junction-F1.  Pairs
      with score 0.0 are never assigned.
    * **BEST_PER_LOCUS** — picks the single argmax entry; returns at most
      one match.

    Parameters
    ----------
    gt_locus : list[_TranscriptInfo]
        GT transcripts in this locus.
    pred_locus : list[_TranscriptInfo]
        Predicted transcripts overlapping this locus.
    predictor_name : str
        Embedded in the returned :class:`PredictionMatch` objects.
    mode : LocusMatchingMode
        Assignment strategy.

    Returns
    -------
    dict[str, PredictionMatch]
        ``{gt_id: PredictionMatch}`` for every accepted pairing.
    """
    n = len(gt_locus)
    m = len(pred_locus)

    score_matrix = np.zeros((n, m), dtype=np.float64)
    for i, gt in enumerate(gt_locus):
        for j, pred in enumerate(pred_locus):
            score_matrix[i, j] = _compute_assignment_score(gt, pred)

    def _make_match(
        gt_info: _TranscriptInfo,
        pred_info: _TranscriptInfo,
        match_score: float,
    ) -> PredictionMatch:
        return PredictionMatch(
            predictor_name=predictor_name,
            transcript_id=pred_info.gff_id,
            start=pred_info.start,
            end=pred_info.end,
            match_class=_classify_pair(gt_info, pred_info),
            base_overlap=_base_overlap(gt_info.start, gt_info.end, pred_info.start, pred_info.end),
            junction_f1=match_score,
        )

    if mode == LocusMatchingMode.BEST_PER_LOCUS:
        flat_idx = int(np.argmax(score_matrix))
        r, c = divmod(flat_idx, m)
        best_score = float(score_matrix[r, c])
        if best_score == 0.0:
            return {}
        return {gt_locus[r].gff_id: _make_match(gt_locus[r], pred_locus[c], best_score)}

    # FULL_DISCOVERY: globally optimal assignment.
    row_ind, col_ind = linear_sum_assignment(-score_matrix)
    assignments: dict[str, PredictionMatch] = {}
    for r, c in zip(row_ind, col_ind):
        pair_score = float(score_matrix[r, c])
        if pair_score == 0.0:
            continue
        assignments[gt_locus[r].gff_id] = _make_match(gt_locus[r], pred_locus[c], pair_score)
    return assignments


# ---------------------------------------------------------------------------
# Internal: per-chromosome, per-strand mapping
# ---------------------------------------------------------------------------


def _map_strand(
    seqid: str,
    strand: str,
    gt_df: pd.DataFrame,
    pred_dfs: dict[str, pd.DataFrame],
    transcript_types: list[str],
    gt_chain_index: dict[str, frozenset[tuple[int, int]]],
    pred_chain_indices: dict[str, dict[str, frozenset[tuple[int, int]]]],
    mode: LocusMatchingMode,
) -> list[TranscriptMapping]:
    """Map GT to predictions for one seqid + strand combination.

    Builds GT loci, scores each predictor's transcripts against every GT in
    the same locus, and performs optimal 1:1 assignment per locus.

    In **FULL_DISCOVERY** mode every GT transcript appears in the output
    (including those with no matches) plus sentinel entries for unmatched
    predictions.

    In **BEST_PER_LOCUS** mode only the single best-matched GT transcript per
    (locus, predictor) pair appears; unmatched GTs and extra preds are dropped.
    """
    gt_infos = _build_transcript_infos(
        gt_df,
        seqid,
        strand,
        transcript_types,
        gt_chain_index,
    )
    pred_infos_by_name: dict[str, list[_TranscriptInfo]] = {
        name: _build_transcript_infos(
            df,
            seqid,
            strand,
            transcript_types,
            pred_chain_indices[name],
        )
        for name, df in pred_dfs.items()
    }

    gt_loci = _build_loci(gt_infos)
    gt_info_lookup: dict[str, _TranscriptInfo] = {t.gff_id: t for t in gt_infos}

    # Per predictor: collect optimal per-locus assignments.
    gt_to_matches: dict[str, list[PredictionMatch]] = {t.gff_id: [] for t in gt_infos}
    matched_pred_ids: dict[str, set[str]] = {name: set() for name in pred_dfs}
    # Predictions that overlap *some* GT locus span (matched or not).  Used by
    # BEST_PER_LOCUS to emit FP entries only for predictions that overlap no GT
    # locus at all — an overlapping-but-unmatched prediction shares bases with a
    # GT locus, so charging it as a separate FP would double-count those bases.
    overlapped_by_pred: dict[str, set[str]] = {name: set() for name in pred_dfs}

    for pred_name, pred_infos in pred_infos_by_name.items():
        for locus in gt_loci:
            preds_in_locus = _find_preds_overlapping_locus(locus, pred_infos)
            overlapped_by_pred[pred_name].update(p.gff_id for p in preds_in_locus)
            # A prediction overlapping two GT loci (e.g. spanning an intergenic
            # gap) must not be assigned — and double-counted — in both.  Once it
            # is matched in an earlier locus, exclude it from later loci for this
            # predictor.  Loci are visited in coordinate order, so the earliest
            # overlapping locus wins.
            preds_in_locus = [p for p in preds_in_locus if p.gff_id not in matched_pred_ids[pred_name]]
            if not preds_in_locus:
                continue
            for gt_id, match in _assign_optimal_locus(locus, preds_in_locus, pred_name, mode).items():
                gt_to_matches[gt_id].append(match)
                matched_pred_ids[pred_name].add(match.transcript_id)

    # Build output mappings according to mode.
    mappings: list[TranscriptMapping] = []

    if mode == LocusMatchingMode.FULL_DISCOVERY:
        # One TranscriptMapping per GT transcript (including unmatched ones).
        for gt_id, matches in gt_to_matches.items():
            mappings.append(
                TranscriptMapping(
                    seqid=seqid,
                    strand=strand,
                    gt_id=gt_id,
                    gt_start=gt_info_lookup[gt_id].start,
                    gt_end=gt_info_lookup[gt_id].end,
                    matched_predictions=matches,
                )
            )

        # Sentinel entries for predictions that were not assigned to any GT.
        for pred_name, pred_infos in pred_infos_by_name.items():
            pred_lookup = {t.gff_id: t for t in pred_infos}
            unmatched_ids = {t.gff_id for t in pred_infos} - matched_pred_ids[pred_name]

            for pred_id in sorted(unmatched_ids):
                pred = pred_lookup[pred_id]
                mappings.append(
                    TranscriptMapping(
                        seqid=seqid,
                        strand=strand,
                        gt_id=f"{_UNMATCHED_PRED_PREFIX}{pred_id}",
                        gt_start=pred.start,
                        gt_end=pred.end,
                        is_unmatched_prediction=True,
                        matched_predictions=[
                            PredictionMatch(
                                predictor_name=pred_name,
                                transcript_id=pred_id,
                                start=pred.start,
                                end=pred.end,
                                match_class=MatchClass.OVERLAPPING,
                                base_overlap=0,
                                junction_f1=0.0,
                            )
                        ],
                    )
                )

    else:  # BEST_PER_LOCUS
        all_pred_names = list(pred_dfs)
        for locus in gt_loci:
            locus_gt_ids = [t.gff_id for t in locus]
            # One mapping per *matched* isoform, carrying that isoform's matches,
            # so each predictor is paired against the GT it actually matched.
            matched_names: set[str] = set()
            for gt_id in locus_gt_ids:
                matches = gt_to_matches[gt_id]
                if matches:
                    matched_names.update(m.predictor_name for m in matches)
                    mappings.append(
                        TranscriptMapping(
                            seqid=seqid,
                            strand=strand,
                            gt_id=gt_id,
                            gt_start=gt_info_lookup[gt_id].start,
                            gt_end=gt_info_lookup[gt_id].end,
                            matched_predictions=matches,
                        )
                    )
            # Predictors with no match in this locus get exactly one entry each,
            # so every locus is counted once per predictor (no double counting):
            #   Case C — a transcript overlaps the locus but shares no junction:
            #            score it against the real GT isoform it best overlaps, so
            #            a confidently-wrong prediction is penalised, not dropped.
            #   Case B — no overlapping transcript at all: a clean miss against the
            #            representative (longest) isoform, paired with a null pred.
            rep = max(locus, key=lambda t: t.end - t.start)
            for pred_name in all_pred_names:
                if pred_name in matched_names:
                    continue
                overlapping = [
                    p
                    for p in _find_preds_overlapping_locus(locus, pred_infos_by_name[pred_name])
                    if p.gff_id not in matched_pred_ids[pred_name]
                ]
                if overlapping:  # Case C — overlap, wrong structure
                    best = max(
                        overlapping,
                        key=lambda p: _base_overlap(rep.start, rep.end, p.start, p.end),
                    )
                    gt_target = max(
                        locus,
                        key=lambda t: _base_overlap(t.start, t.end, best.start, best.end),
                    )
                    # Earliest-locus-wins: a locus-spanning pred is scored only once.
                    matched_pred_ids[pred_name].add(best.gff_id)
                    mappings.append(
                        TranscriptMapping(
                            seqid=seqid,
                            strand=strand,
                            gt_id=gt_target.gff_id,
                            gt_start=gt_target.start,
                            gt_end=gt_target.end,
                            matched_predictions=[
                                PredictionMatch(
                                    predictor_name=pred_name,
                                    transcript_id=best.gff_id,
                                    start=best.start,
                                    end=best.end,
                                    match_class=_classify_pair(gt_target, best),
                                    base_overlap=_base_overlap(
                                        gt_target.start, gt_target.end, best.start, best.end
                                    ),
                                    junction_f1=0.0,
                                )
                            ],
                        )
                    )
                else:  # Case B — clean miss
                    mappings.append(
                        TranscriptMapping(
                            seqid=seqid,
                            strand=strand,
                            gt_id=rep.gff_id,
                            gt_start=rep.start,
                            gt_end=rep.end,
                            matched_predictions=[],
                            fn_for_predictors=[pred_name],
                        )
                    )

        # FP entries for predictions that overlap no GT locus at all (intergenic
        # false positives).  Overlapping-but-unmatched predictions are left to the
        # global precision metric to avoid double-counting their shared bases.
        for pred_name, pred_infos in pred_infos_by_name.items():
            pred_lookup = {t.gff_id: t for t in pred_infos}
            spurious = {t.gff_id for t in pred_infos} - overlapped_by_pred[pred_name]
            for pred_id in sorted(spurious):
                pred = pred_lookup[pred_id]
                mappings.append(
                    TranscriptMapping(
                        seqid=seqid,
                        strand=strand,
                        gt_id=f"{_UNMATCHED_PRED_PREFIX}{pred_id}",
                        gt_start=pred.start,
                        gt_end=pred.end,
                        is_unmatched_prediction=True,
                        matched_predictions=[
                            PredictionMatch(
                                predictor_name=pred_name,
                                transcript_id=pred_id,
                                start=pred.start,
                                end=pred.end,
                                match_class=MatchClass.OVERLAPPING,
                                base_overlap=0,
                                junction_f1=0.0,
                            )
                        ],
                    )
                )

    return mappings


def _process_single_seqid(
    seqid: str,
    gt_df: pd.DataFrame,
    pred_dfs: dict[str, pd.DataFrame],
    transcript_types: list[str],
    gt_feature_role_map: FeatureRoleMap,
    mode: LocusMatchingMode,
    pred_feature_role_maps: dict[str, FeatureRoleMap],
) -> list[TranscriptMapping]:
    """Build intron-chain indexes for one chromosome, then map per strand."""
    gt_chain_index = _build_intron_chain_index(gt_df, seqid, exonic_feature_types(gt_feature_role_map))
    pred_chain_indices = {
        name: _build_intron_chain_index(df, seqid, exonic_feature_types(pred_feature_role_maps[name]))
        for name, df in pred_dfs.items()
    }

    mappings: list[TranscriptMapping] = []
    for strand in ("+", "-"):
        mappings.extend(
            _map_strand(
                seqid=seqid,
                strand=strand,
                gt_df=gt_df,
                pred_dfs=pred_dfs,
                transcript_types=transcript_types,
                gt_chain_index=gt_chain_index,
                pred_chain_indices=pred_chain_indices,
                mode=mode,
            )
        )
    return mappings


# ---------------------------------------------------------------------------
# Public: mapping
# ---------------------------------------------------------------------------


def map_transcripts(
    gt_path: str | Path,
    pred_paths: dict[str, str | Path],
    label_config: LabelConfig | None = None,
    *,
    gt_df: pd.DataFrame | None = None,
    pred_dfs: dict[str, pd.DataFrame] | None = None,
    transcript_types: list[str] | None = None,
    gt_feature_role_map: FeatureRoleMap | None = None,
    pred_feature_role_maps: PredFeatureRoleMapInput = None,
    exclude_features: list[str] | None = None,
    locus_matching_mode: LocusMatchingMode = LocusMatchingMode.FULL_DISCOVERY,
) -> list[TranscriptMapping]:
    """Map GT transcripts to predicted transcripts across multiple predictors.

    Uses a gffcompare-style locus-based algorithm:

    1. Group GT transcripts into loci by coordinate overlap.
    2. Score each predicted transcript against all GT transcripts in the
       same locus using continuous junction-F1.
    3. Assign 1:1 pairings via the Hungarian algorithm (FULL_DISCOVERY) or
       take the single best pair (BEST_PER_LOCUS).

    Parameters
    ----------
    gt_path : str | Path
        Path to the ground-truth GFF/GTF file.
    pred_paths : dict[str, str | Path]
        ``{predictor_name: path}`` for each prediction file.
    label_config : LabelConfig | None
        Label semantics used for later array construction. Needed here so
        feature-role maps can be validated against the active annotation mode.
        When omitted, mapping falls back to ``EXON_INTRON`` defaults.
    gt_df : pd.DataFrame | None
        Pre-parsed GT GFF DataFrame.  When given, *gt_path* is not re-parsed
        (avoids a redundant parse when the caller already has the DataFrame).
    pred_dfs : dict[str, pd.DataFrame] | None
        Pre-parsed prediction DataFrames keyed by predictor name.  When given,
        *pred_paths* is not re-parsed.
    transcript_types : list[str] | None
        Feature types that define transcript boundaries.
        Defaults to :data:`~dna_segmentation_benchmark.io_utils.DEFAULT_TRANSCRIPT_TYPES`.
    gt_feature_role_map : dict[str, str] | None
        Maps GT GFF/GTF feature types to benchmark roles.  When ``None``,
        the mode-specific default is used.
    pred_feature_role_maps : dict[str, str] | dict[str, dict[str, str]] | None
        Per-predictor feature-role maps.  ``None`` means use the GT role map.
    exclude_features : list[str] | None
        Feature types to ignore entirely (e.g. ``["gene"]``).
    locus_matching_mode : LocusMatchingMode
        Assignment strategy.  ``FULL_DISCOVERY`` (default) maximises the
        number of 1:1 matches per locus.  ``BEST_PER_LOCUS`` keeps only the
        single best-scoring pair per locus and drops the rest.

    Returns
    -------
    list[TranscriptMapping]
        In FULL_DISCOVERY mode: one entry per GT transcript (including
        unmatched) plus sentinel entries for unmatched predictions.
        In BEST_PER_LOCUS mode: only matched pairs are returned.
    """
    if locus_matching_mode == LocusMatchingMode.BEST_PER_LOCUS:
        logger.info(
            "BEST_PER_LOCUS mode: every GT locus is scored once per predictor — a "
            "serious match as a pair (Case A); an overlapping-but-wrong prediction "
            "scored against the real GT isoform (Case C); an unmatched locus as a "
            "clean FN against the representative isoform (Case B). Predictions that "
            "overlap no GT locus become intergenic FP entries."
        )

    if label_config is None:
        label_config = LabelConfig(
            annotation_mode=AnnotationMode.EXON_INTRON,
            background_label=1,
            exon_label=0,
        )

    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    exclude_features = exclude_features or []

    if gt_df is None:
        gt_df = collect_gff(str(gt_path), exclude_features=exclude_features)
    if pred_dfs is None:
        pred_dfs = {
            name: collect_gff(str(path), exclude_features=exclude_features) for name, path in pred_paths.items()
        }
    gt_feature_role_map = normalize_feature_role_map(
        gt_feature_role_map, label_config, arg_name="gt_feature_role_map"
    )
    pred_feature_role_maps_by_name = normalize_pred_feature_role_maps(
        list(pred_paths.keys()),
        pred_feature_role_maps,
        default=gt_feature_role_map,
        label_config=label_config,
    )

    gt_seqids: set[str] = set(gt_df["seqid"].dropna().unique().tolist())
    pred_seqids: set[str] = set()
    for df in pred_dfs.values():
        pred_seqids.update(df["seqid"].dropna().unique().tolist())
    all_seqids = gt_seqids | pred_seqids

    logger.info(
        "Found %d GT seqid(s), %d prediction seqid(s) (%d total).",
        len(gt_seqids),
        len(pred_seqids),
        len(all_seqids),
    )

    # Pre-group by seqid once.  _process_single_seqid and its helpers filter by
    # seqid internally; passing the whole DataFrame re-scans every row on every
    # iteration (these per-seqid object-column masks dominated mapping time on
    # whole-genome inputs).  Slicing here makes each iteration touch only its own
    # chromosome's rows, while the helpers' internal filters stay correct.
    gt_by_seqid = {k: v for k, v in gt_df.groupby("seqid", sort=False, observed=True)}
    pred_by_seqid = {
        name: {k: v for k, v in df.groupby("seqid", sort=False, observed=True)}
        for name, df in pred_dfs.items()
    }
    empty_gt = gt_df.iloc[0:0]
    empty_pred = {name: df.iloc[0:0] for name, df in pred_dfs.items()}

    all_mappings: list[TranscriptMapping] = []
    for seqid in tqdm(sorted(all_seqids), desc="Mapping transcripts", unit="seqid"):
        all_mappings.extend(
            _process_single_seqid(
                seqid,
                gt_by_seqid.get(seqid, empty_gt),
                {name: pred_by_seqid[name].get(seqid, empty_pred[name]) for name in pred_dfs},
                transcript_types,
                gt_feature_role_map,
                locus_matching_mode,
                pred_feature_role_maps=pred_feature_role_maps_by_name,
            )
        )

    n_unmatched = sum(1 for m in all_mappings if m.is_unmatched_prediction)
    n_no_pred = sum(1 for m in all_mappings if not m.is_unmatched_prediction and not m.matched_predictions)
    logger.info(
        "Mapping complete: %d entries (%d unmatched predictions, %d GT transcripts with no match).",
        len(all_mappings),
        n_unmatched,
        n_no_pred,
    )
    return all_mappings


# ---------------------------------------------------------------------------
# Public: array construction
# ---------------------------------------------------------------------------


# (seqid, parent) -> child rows as plain ``(type, start, end)`` tuples.
# Pre-built once and reused across mappings: lookups are O(1) and painting
# touches plain Python tuples instead of re-running pandas ``.isin`` / boolean
# indexing on a tiny DataFrame for every one of tens of thousands of calls.
_ChildRow = tuple[str, int, int]
_DFIndex = dict[tuple[str, str], list[_ChildRow]]

# Ordered ``(label_value, feature_types)`` paint passes, compiled once with the
# feature types as a set for O(1) membership in the hot paint loop.
_PaintPlan = list[tuple[int, frozenset[str]]]


def _build_df_index(df: pd.DataFrame, transcript_types: list[str]) -> _DFIndex:
    """Index a GFF DataFrame by ``(seqid, parent)`` for O(1) transcript lookups.

    Rows are filtered (drop transcript rows, NaN parents and NaN coordinates)
    and converted to plain ``(type, start, end)`` tuples here, **once**, so the
    per-transcript paint path carries no pandas overhead.  Top-level genes (NaN
    parent) are excluded; lookups are always keyed by ``(seqid, transcript_id)``.
    """
    sub = df[
        df["parent"].notna()
        & ~df["type"].isin(transcript_types)
        & df["start"].notna()
        & df["end"].notna()
    ]
    index: _DFIndex = defaultdict(list)
    for seqid, parent, typ, start, end in zip(
        sub["seqid"].to_numpy(),
        sub["parent"].to_numpy(),
        sub["type"].to_numpy(),
        sub["start"].to_numpy(),
        sub["end"].to_numpy(),
    ):
        index[(str(seqid), str(parent))].append((str(typ), int(start), int(end)))
    return dict(index)


def _compile_paint_plan(feature_role_map: FeatureRoleMap, label_config: LabelConfig) -> _PaintPlan:
    """Compile a role map into ordered ``(label, feature-type set)`` paint passes."""
    return [
        (label_value, frozenset(feature_types))
        for label_value, feature_types in feature_role_paint_plan(feature_role_map, label_config)
    ]


def _paint_child_rows(
    arr: np.ndarray,
    rows: list[_ChildRow],
    region_start: int,
    paint_plan: _PaintPlan,
) -> None:
    """Paint ``(type, start, end)`` rows into *arr* following *paint_plan* order.

    Mirrors :func:`feature_roles.paint_feature_rows` exactly (same pass order,
    same per-feature clipping, later passes overwrite earlier ones) but operates
    on plain tuples to stay out of pandas in the hot loop.
    """
    n = len(arr)
    for label_value, feature_types in paint_plan:
        for typ, start, end in rows:
            if typ in feature_types:
                local_start = start - region_start
                if local_start < 0:
                    local_start = 0
                local_end = end - region_start + 1
                if local_end > n:
                    local_end = n
                if local_start < local_end:
                    arr[local_start:local_end] = label_value


def _include_mapping_for_predictor(
    mapping: TranscriptMapping,
    pred_name: str,
    mode: LocusMatchingMode,
    *,
    ref_keep: set[str] | None = None,
    pred_keep: set[str] | None = None,
    ignore_novel: bool = False,
    ignore_missed: bool = False,
) -> bool:
    """Return True if this mapping entry should contribute arrays for *pred_name*.

    Three cases in BEST_PER_LOCUS mode:
    - ``is_unmatched_prediction``: a real prediction with no GT overlap — include
      only for the predictor that owns it (FP entry).
    - ``fn_for_predictors`` non-empty: a locus-level miss entry — include only
      for predictors listed there (locus FN).
    - Otherwise: a normal matched entry — skip if the predictor didn't match
      (its FN is recorded by its own Case B/C entry; don't double-count).

    The ``ignore_novel`` / ``ignore_missed`` flags reproduce gffcompare's
    ``-Q`` / ``-R`` incomplete-annotation corrections and are driven by
    coordinate-overlap keep-sets (``pred_keep`` / ``ref_keep``) computed
    independently of the match result:

    - ``ignore_novel`` (``-Q``): drop an unmatched-prediction entry whose
      prediction overlaps *no* GT transcript (``transcript_id`` ∉ ``pred_keep``).
      An overlapping-but-unmatched prediction stays in ``pred_keep``, so it is
      still counted as a false positive.
    - ``ignore_missed`` (``-R``): drop a real-GT entry whose GT transcript
      overlaps *no* prediction (``gt_id`` ∉ ``ref_keep``).  An
      overlapping-but-unmatched GT stays in ``ref_keep``, so it is still a miss.
    """
    if ignore_novel and mapping.is_unmatched_prediction and pred_keep is not None:
        pred_id = next(
            (m.transcript_id for m in mapping.matched_predictions if m.predictor_name == pred_name),
            None,
        )
        if pred_id is not None and pred_id not in pred_keep:
            return False
    if (
        ignore_missed
        and not mapping.is_unmatched_prediction
        and ref_keep is not None
        and mapping.gt_id not in ref_keep
    ):
        return False

    has_match = any(m.predictor_name == pred_name for m in mapping.matched_predictions)
    if mapping.is_unmatched_prediction:
        return has_match
    if mapping.fn_for_predictors:
        return pred_name in mapping.fn_for_predictors
    if mode == LocusMatchingMode.BEST_PER_LOCUS and not has_match:
        return False
    return True


def build_paired_arrays(
    mapping: TranscriptMapping,
    gt_df: pd.DataFrame,
    pred_dfs: dict[str, pd.DataFrame],
    label_config: LabelConfig,
    transcript_types: list[str] | None = None,
    gt_feature_role_map: FeatureRoleMap | None = None,
    pred_feature_role_maps: PredFeatureRoleMapInput = None,
    *,
    _gt_index: _DFIndex | None = None,
    _pred_indices: dict[str, _DFIndex] | None = None,
    _gt_role_map: FeatureRoleMap | None = None,
    _pred_role_maps: dict[str, FeatureRoleMap] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build the GT and per-predictor annotation arrays for one mapping.

    **GT array**

    * Matched GT transcript → built from that transcript's child features.
    * Unmatched prediction (``is_unmatched_prediction=True``) → null array
      (all-background), representing a pure FP with no reference signal.

    **Prediction arrays**

    * Predictor has a match → built from the *matched prediction's* child
      features (transcript-specific, not a region union).
    * Predictor has no match for this GT → null array (all-background),
      representing a missed transcript.

    Using transcript-specific arrays (rather than region-based union) ensures
    that each matched pair is evaluated in isolation: coverage from other
    transcripts in the same locus does not contaminate either array.

    Parameters
    ----------
    mapping : TranscriptMapping
        A single mapping entry from :func:`map_transcripts`.
    gt_df : pd.DataFrame
        Pre-collected GT GFF DataFrame.
    pred_dfs : dict[str, pd.DataFrame]
        ``{predictor_name: DataFrame}`` for each prediction file.
    label_config : LabelConfig
        Label configuration defining integer tokens.
    transcript_types : list[str] | None
        Feature types that denote transcript boundaries.
    gt_feature_role_map : dict[str, str] | None
        Maps GT GFF/GTF feature types to benchmark roles.  When ``None``,
        the mode-specific default is used.
    pred_feature_role_maps : dict[str, str] | dict[str, dict[str, str]] | None
        Per-predictor feature-role maps.  ``None`` means every predictor
        uses the GT role map.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        ``(gt_array, {predictor_name: pred_array})``.
        The dict contains an entry for every predictor in *pred_dfs*.

    Notes
    -----
    The leading-underscore keyword parameters are a private fast path for
    callers that build arrays in a loop (e.g. the pipeline): pre-built
    ``(seqid, parent)`` indices and pre-normalized role maps are reused across
    mappings instead of being rebuilt on every call.  Stand-alone callers omit
    them and the per-call build/normalize path runs as before.
    """
    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    if _gt_role_map is not None:
        gt_feature_role_map = _gt_role_map
    else:
        gt_feature_role_map = normalize_feature_role_map(
            gt_feature_role_map, label_config, arg_name="gt_feature_role_map"
        )
    if _pred_role_maps is not None:
        pred_feature_role_maps_by_name = _pred_role_maps
    else:
        pred_feature_role_maps_by_name = normalize_pred_feature_role_maps(
            list(pred_dfs.keys()),
            pred_feature_role_maps,
            default=gt_feature_role_map,
            label_config=label_config,
        )
    gt_index = _gt_index if _gt_index is not None else _build_df_index(gt_df, transcript_types)
    pred_indices = (
        _pred_indices
        if _pred_indices is not None
        else {name: _build_df_index(df, transcript_types) for name, df in pred_dfs.items()}
    )
    gt_paint_plan = _compile_paint_plan(gt_feature_role_map, label_config)
    pred_paint_plans = {
        name: _compile_paint_plan(rmap, label_config)
        for name, rmap in pred_feature_role_maps_by_name.items()
    }
    # Evaluation window = union of the GT span and every matched prediction's
    # span.  Sizing to the GT span alone clips a matched prediction whose
    # terminal exon over-extends past the GT boundary (a routine TSS/TES
    # difference), which silently zeroes terminal boundary offsets and inflates
    # boundary exactness / IoU.  All predictors share one gt_array, so the window
    # must cover every matched prediction; the GT side simply gains background
    # padding over the extension (no cross-transcript contamination, since each
    # array is painted from its own transcript's features only).
    region_start = mapping.gt_start
    region_end = mapping.gt_end
    for match in mapping.matched_predictions:
        region_start = min(region_start, match.start)
        region_end = max(region_end, match.end)
    array_length = region_end - region_start + 1
    bg_val = label_config.background_label

    # Null array shared across predictors with no match.  uint8 (labels are small
    # non-negative tokens) keeps these per-transcript arrays — the pipeline's
    # dominant memory cost — 4x smaller than int32.
    null_array = np.full(array_length, bg_val, dtype=np.uint8)

    # --- GT array ---
    if mapping.is_unmatched_prediction:
        # Unmatched prediction: no GT reference → null GT array.
        gt_array = null_array.copy()
    else:
        gt_array = _build_annotation_array_from_df(
            df_index=gt_index,
            transcript_id=mapping.gt_id,
            seqid=mapping.seqid,
            region_start=region_start,
            array_length=array_length,
            label_config=label_config,
            paint_plan=gt_paint_plan,
        )

    # --- Prediction arrays: transcript-specific ---
    pred_arrays: dict[str, np.ndarray] = {}
    for pred_name in pred_dfs:
        pred_match = next(
            (m for m in mapping.matched_predictions if m.predictor_name == pred_name),
            None,
        )
        if pred_match is not None:
            pred_arrays[pred_name] = _build_annotation_array_from_df(
                df_index=pred_indices[pred_name],
                transcript_id=pred_match.transcript_id,
                seqid=mapping.seqid,
                region_start=region_start,
                array_length=array_length,
                label_config=label_config,
                paint_plan=pred_paint_plans[pred_name],
            )
        else:
            # Predictor has no match for this GT → null pred array (FN/FP).
            pred_arrays[pred_name] = null_array.copy()

    # Normalise to biological 5'→3' order.  Painting uses genomic left-to-right
    # coordinates; for minus-strand transcripts that is biologically reversed.
    if mapping.strand == "-":
        gt_array = np.ascontiguousarray(gt_array[::-1])
        pred_arrays = {k: np.ascontiguousarray(v[::-1]) for k, v in pred_arrays.items()}

    return gt_array, pred_arrays


# ---------------------------------------------------------------------------
# Internal: array builders (no file I/O)
# ---------------------------------------------------------------------------


def _build_annotation_array_from_df(
    df_index: _DFIndex,
    transcript_id: str,
    seqid: str,
    region_start: int,
    array_length: int,
    label_config: LabelConfig,
    paint_plan: _PaintPlan,
) -> np.ndarray:
    """Build a 1-D annotation array for one transcript's child features.

    Child features of *transcript_id* are projected into local coordinates
    ``[0, array_length)`` relative to *region_start*.

    Parameters
    ----------
    df_index : _DFIndex
        ``(seqid, parent) -> child rows`` index from :func:`_build_df_index`.
    paint_plan : _PaintPlan
        Ordered paint passes from :func:`_compile_paint_plan`.
    """
    bg_val = label_config.background_label
    arr = np.full(array_length, bg_val, dtype=np.uint8)  # small label tokens; see build_paired_arrays

    rows = df_index.get((seqid, transcript_id))
    if rows:
        _paint_child_rows(arr, rows, region_start, paint_plan)
    return arr


# ---------------------------------------------------------------------------
# Public: debug export
# ---------------------------------------------------------------------------


def export_mapping_table(
    mappings: list[TranscriptMapping],
    output_path: str | Path,
) -> Path:
    """Write the mapping list to a human-readable TSV for debugging.

    Columns: ``seqid``, ``strand``, ``gt_id``, ``gt_start``, ``gt_end``,
    ``is_unmatched_prediction``, ``predictor``, ``pred_id``,
    ``pred_start``, ``pred_end``, ``match_class``, ``base_overlap``,
    ``junction_f1``.

    Each row represents one GT-prediction pair.  GT transcripts with
    multiple matched predictions produce multiple rows.  GT transcripts
    with no matches produce a single row with empty prediction columns.

    Parameters
    ----------
    mappings : list[TranscriptMapping]
        Output of :func:`map_transcripts`.
    output_path : str | Path
        Destination file path.

    Returns
    -------
    Path
        The path the file was written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "seqid",
        "strand",
        "gt_id",
        "gt_start",
        "gt_end",
        "is_unmatched_prediction",
        "predictor",
        "pred_id",
        "pred_start",
        "pred_end",
        "match_class",
        "base_overlap",
        "junction_f1",
    ]

    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for mapping in mappings:
            base_row = {
                "seqid": mapping.seqid,
                "strand": mapping.strand,
                "gt_id": mapping.gt_id,
                "gt_start": mapping.gt_start,
                "gt_end": mapping.gt_end,
                "is_unmatched_prediction": mapping.is_unmatched_prediction,
            }

            if not mapping.matched_predictions:
                writer.writerow(
                    {
                        **base_row,
                        "predictor": "",
                        "pred_id": "",
                        "pred_start": "",
                        "pred_end": "",
                        "match_class": "",
                        "base_overlap": "",
                        "junction_f1": "",
                    }
                )
            else:
                for match in mapping.matched_predictions:
                    writer.writerow(
                        {
                            **base_row,
                            "predictor": match.predictor_name,
                            "pred_id": match.transcript_id,
                            "pred_start": match.start,
                            "pred_end": match.end,
                            "match_class": match.match_class.value,
                            "base_overlap": match.base_overlap,
                            "junction_f1": f"{match.junction_f1:.4f}",
                        }
                    )

    logger.info("Mapping table written to %s", output_path)
    return output_path
