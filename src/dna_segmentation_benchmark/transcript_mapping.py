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
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .io_utils import DEFAULT_TRANSCRIPT_TYPES, collect_gff
from .label_definition import LabelConfig

logger = logging.getLogger(__name__)

# Sentinel prefix for synthetic GT entries created for unmatched predictions.
_UNMATCHED_PRED_PREFIX = "__unmatched_pred__"

#: GFF feature types used to extract exon intervals for intron-chain
#: computation.  Override per-call via the ``exon_types`` parameter.
DEFAULT_EXON_TYPES: list[str] = ["exon"]

FeatureTypeInput = str | list[str]
PredFeatureTypeInput = FeatureTypeInput | dict[str, FeatureTypeInput]


def _coerce_feature_types(
    feature_types: FeatureTypeInput,
    *,
    arg_name: str,
) -> list[str]:
    """Normalize a feature-type argument to a non-empty list of strings."""
    if isinstance(feature_types, str):
        feature_types = [feature_types]

    if not isinstance(feature_types, list) or not feature_types:
        raise ValueError(f"{arg_name} must be a string or non-empty list of strings.")

    normalized: list[str] = []
    for feature_type in feature_types:
        if not isinstance(feature_type, str) or not feature_type:
            raise ValueError(f"{arg_name} entries must be non-empty strings.")
        normalized.append(feature_type)
    return normalized


def _normalise_pred_exon_types(
    pred_names: list[str],
    pred_exon_types: PredFeatureTypeInput | None,
    *,
    default: list[str],
) -> dict[str, list[str]]:
    """Return per-predictor exon-like feature types."""
    if pred_exon_types is None:
        return {name: list(default) for name in pred_names}

    if isinstance(pred_exon_types, dict):
        return {
            name: _coerce_feature_types(
                pred_exon_types.get(name, default),
                arg_name=f"pred_exon_types[{name!r}]",
            )
            for name in pred_names
        }

    normalized = _coerce_feature_types(pred_exon_types, arg_name="pred_exon_types")
    return {name: list(normalized) for name in pred_names}


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
        Only the single highest-scoring (GT, pred) pair per locus is kept;
        all other transcripts in the locus are dropped from per-transcript
        evaluation.

        Use this mode for single-transcript tools (Augustus, …).
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
    """

    model_config = {"frozen": True}

    seqid: str
    strand: str
    gt_id: str
    gt_start: int
    gt_end: int
    is_unmatched_prediction: bool = False
    matched_predictions: list[PredictionMatch] = []


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

    index: dict[str, frozenset[tuple[int, int]]] = {}
    for parent_id, group in exon_df.groupby("parent", sort=False):
        exons = group.sort_values("start")
        starts = exons["start"].astype(int).tolist()
        ends = exons["end"].astype(int).tolist()

        if len(starts) < 2:
            index[str(parent_id)] = frozenset()
        else:
            index[str(parent_id)] = frozenset((ends[i], starts[i + 1]) for i in range(len(starts) - 1))

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
    for row in rows.to_dict(orient="records"):
        gff_id = str(row["gff_id"])
        chain = chain_index.get(gff_id, frozenset())
        infos.append(
            _TranscriptInfo(
                gff_id=gff_id,
                start=int(row["start"]),
                end=int(row["end"]),
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
    """Cluster transcripts into loci by coordinate overlap.

    A locus is a maximal set of transcripts connected via pairwise
    coordinate overlap.  Implemented as an O(n log n) interval sweep.
    """
    if not transcripts:
        return []

    sorted_ts = sorted(transcripts, key=lambda t: (t.start, t.end))
    loci: list[list[_TranscriptInfo]] = []
    current_locus = [sorted_ts[0]]
    current_end = sorted_ts[0].end

    for t in sorted_ts[1:]:
        if t.start <= current_end:
            current_locus.append(t)
            current_end = max(current_end, t.end)
        else:
            loci.append(current_locus)
            current_locus = [t]
            current_end = t.end

    loci.append(current_locus)
    return loci


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
    from scipy.optimize import linear_sum_assignment

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

    for pred_name, pred_infos in pred_infos_by_name.items():
        for locus in gt_loci:
            preds_in_locus = _find_preds_overlapping_locus(locus, pred_infos)
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
        # Only include GT transcripts that have at least one prediction match.
        # Unmatched GTs and extra preds within the locus are dropped.
        matched_gt_ids = {gt_id for gt_id, matches in gt_to_matches.items() if matches}
        for gt_id in matched_gt_ids:
            mappings.append(
                TranscriptMapping(
                    seqid=seqid,
                    strand=strand,
                    gt_id=gt_id,
                    gt_start=gt_info_lookup[gt_id].start,
                    gt_end=gt_info_lookup[gt_id].end,
                    matched_predictions=gt_to_matches[gt_id],
                )
            )

    return mappings


def _process_single_seqid(
    seqid: str,
    gt_df: pd.DataFrame,
    pred_dfs: dict[str, pd.DataFrame],
    transcript_types: list[str],
    exon_types: list[str],
    mode: LocusMatchingMode,
    pred_exon_types: PredFeatureTypeInput | None = None,
) -> list[TranscriptMapping]:
    """Build intron-chain indexes for one chromosome, then map per strand."""
    gt_chain_index = _build_intron_chain_index(gt_df, seqid, exon_types)
    pred_exon_types_by_name = _normalise_pred_exon_types(
        list(pred_dfs.keys()),
        pred_exon_types,
        default=exon_types,
    )
    pred_chain_indices = {
        name: _build_intron_chain_index(df, seqid, pred_exon_types_by_name[name]) for name, df in pred_dfs.items()
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
    *,
    transcript_types: list[str] | None = None,
    exon_types: FeatureTypeInput | None = None,
    pred_exon_types: PredFeatureTypeInput | None = None,
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
    transcript_types : list[str] | None
        Feature types that define transcript boundaries.
        Defaults to :data:`~dna_segmentation_benchmark.io_utils.DEFAULT_TRANSCRIPT_TYPES`.
    exon_types : str | list[str] | None
        Feature types used to extract exon intervals for GT intron-chain
        computation.  Defaults to :data:`DEFAULT_EXON_TYPES` (``["exon"]``).
    pred_exon_types : str | list[str] | dict[str, str | list[str]] | None
        Feature types used for *prediction* intron-chain computation.
        When ``None`` (default), falls back to *exon_types*.  A string/list
        applies to every predictor; a dict maps predictor names to feature
        types.  Pass ``"CDS"`` when a predictor emits CDS features instead of
        exon features.
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
    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    exon_types = _coerce_feature_types(
        exon_types or list(DEFAULT_EXON_TYPES),
        arg_name="exon_types",
    )
    exclude_features = exclude_features or []

    gt_df = collect_gff(str(gt_path), exclude_features=exclude_features)
    pred_dfs: dict[str, pd.DataFrame] = {
        name: collect_gff(str(path), exclude_features=exclude_features) for name, path in pred_paths.items()
    }
    pred_exon_types_by_name = _normalise_pred_exon_types(
        list(pred_paths.keys()),
        pred_exon_types,
        default=exon_types,
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

    all_mappings: list[TranscriptMapping] = []
    for seqid in sorted(all_seqids):
        all_mappings.extend(
            _process_single_seqid(
                seqid,
                gt_df,
                pred_dfs,
                transcript_types,
                exon_types,
                locus_matching_mode,
                pred_exon_types=pred_exon_types_by_name,
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


def build_paired_arrays(
    mapping: TranscriptMapping,
    gt_df: pd.DataFrame,
    pred_dfs: dict[str, pd.DataFrame],
    label_config: LabelConfig,
    transcript_types: list[str] | None = None,
    exon_types: FeatureTypeInput | None = None,
    pred_exon_types: PredFeatureTypeInput | None = None,
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
    exon_types : str | list[str] | None
        Feature types to paint in GT arrays.  If ``None``, paints all child
        features except transcript-type features.
    pred_exon_types : str | list[str] | dict[str, str | list[str]] | None
        Feature types to paint in prediction arrays.  When ``None``
        (default), falls back to *exon_types*.  A string/list applies to every
        predictor; a dict maps predictor names to feature types.  Pass
        ``"CDS"`` when a predictor emits CDS features instead of exon features.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        ``(gt_array, {predictor_name: pred_array})``.
        The dict contains an entry for every predictor in *pred_dfs*.
    """
    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    if exon_types is not None:
        exon_types = _coerce_feature_types(exon_types, arg_name="exon_types")
    array_length = mapping.gt_end - mapping.gt_start + 1
    bg_val = label_config.background_label

    # Null array shared across predictors with no match.
    null_array = np.full(array_length, bg_val, dtype=np.int32)

    # --- GT array ---
    if mapping.is_unmatched_prediction:
        # Unmatched prediction: no GT reference → null GT array.
        gt_array = null_array.copy()
    else:
        gt_array = _build_annotation_array_from_df(
            df=gt_df,
            transcript_id=mapping.gt_id,
            seqid=mapping.seqid,
            region_start=mapping.gt_start,
            array_length=array_length,
            label_config=label_config,
            transcript_types=transcript_types,
            exon_types=exon_types,
        )

    # --- Prediction arrays: transcript-specific ---
    pred_arrays: dict[str, np.ndarray] = {}
    for pred_name, pred_df in pred_dfs.items():
        pred_match = next(
            (m for m in mapping.matched_predictions if m.predictor_name == pred_name),
            None,
        )
        if pred_match is not None:
            resolved_pred_exon_types = _resolve_build_pred_exon_types(
                pred_name,
                pred_exon_types,
                default=exon_types,
            )
            pred_arrays[pred_name] = _build_annotation_array_from_df(
                df=pred_df,
                transcript_id=pred_match.transcript_id,
                seqid=mapping.seqid,
                region_start=mapping.gt_start,
                array_length=array_length,
                label_config=label_config,
                transcript_types=transcript_types,
                exon_types=resolved_pred_exon_types,
            )
        else:
            # Predictor has no match for this GT → null pred array (FN/FP).
            pred_arrays[pred_name] = null_array.copy()

    return gt_array, pred_arrays


def _resolve_build_pred_exon_types(
    pred_name: str,
    pred_exon_types: PredFeatureTypeInput | None,
    *,
    default: list[str] | None,
) -> list[str] | None:
    """Resolve prediction feature types for array construction."""
    if pred_exon_types is None:
        return default
    if isinstance(pred_exon_types, dict):
        value = pred_exon_types.get(pred_name, default)
        return _coerce_feature_types(value, arg_name=f"pred_exon_types[{pred_name!r}]") if value is not None else None
    return _coerce_feature_types(pred_exon_types, arg_name="pred_exon_types")


# ---------------------------------------------------------------------------
# Internal: array builders (no file I/O)
# ---------------------------------------------------------------------------


def _build_annotation_array_from_df(
    df: pd.DataFrame,
    transcript_id: str,
    seqid: str,
    region_start: int,
    array_length: int,
    label_config: LabelConfig,
    transcript_types: list[str],
    exon_types: list[str] | None = None,
) -> np.ndarray:
    """Build a 1-D annotation array for one transcript's child features.

    Child features of *transcript_id* are projected into local coordinates
    ``[0, array_length)`` relative to *region_start*.

    Parameters
    ----------
    exon_types : list[str] | None
        If given, only paint child features whose type is in this list.
        If ``None``, paints all children except transcript-type features.
    """
    bg_val = label_config.background_label
    coding_val = label_config.coding_label
    arr = np.full(array_length, bg_val, dtype=np.int32)

    if coding_val is None:
        return arr

    mask = (df["seqid"] == seqid) & (df["parent"] == transcript_id)
    if exon_types is not None:
        mask &= df["type"].isin(exon_types)
    else:
        mask &= ~df["type"].isin(transcript_types)
    children = df[mask][["start", "end"]]

    _paint_features(arr, children, region_start, coding_val)
    return arr


def _build_region_annotation_array(
    df: pd.DataFrame,
    seqid: str,
    strand: str,
    region_start: int,
    array_length: int,
    label_config: LabelConfig,
    transcript_types: list[str],
    exon_types: list[str] | None = None,
) -> np.ndarray:
    """Build a 1-D annotation array from all features in a genomic region.

    Kept for use by :mod:`~dna_segmentation_benchmark.eval.global_metrics`
    which needs union-of-exons arrays for nucleotide-level global metrics.
    Not used by :func:`build_paired_arrays` (which is transcript-specific).
    """
    bg_val = label_config.background_label
    coding_val = label_config.coding_label
    arr = np.full(array_length, bg_val, dtype=np.int32)

    if coding_val is None:
        return arr

    region_end = region_start + array_length - 1
    mask = (
        (df["seqid"] == seqid)
        & (df["strand"] == strand)
        & df["parent"].notna()
        & (df["start"] <= region_end)
        & (df["end"] >= region_start)
    )
    if exon_types is not None:
        mask &= df["type"].isin(exon_types)
    else:
        mask &= ~df["type"].isin(transcript_types)
    children = df[mask][["start", "end"]]

    _paint_features(arr, children, region_start, coding_val)
    return arr


def _paint_features(
    arr: np.ndarray,
    features_df: pd.DataFrame,
    region_start: int,
    label_value: int,
) -> None:
    """Paint feature intervals into *arr* in-place.

    Converts 1-based inclusive GFF coordinates to 0-based array indices
    relative to *region_start*.
    """
    for feat_start, feat_end in zip(features_df["start"], features_df["end"]):
        if pd.isna(feat_start) or pd.isna(feat_end):
            continue
        local_start = max(0, int(feat_start) - region_start)
        local_end = min(len(arr), int(feat_end) - region_start + 1)
        if local_start < local_end:
            arr[local_start:local_end] = label_value


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
