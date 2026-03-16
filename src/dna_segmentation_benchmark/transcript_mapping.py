"""Strand-aware, memory-efficient mapping of GT to prediction transcripts.

This module provides the core logic for associating ground-truth (GT) and
predicted transcripts from GFF/GTF files.  It is designed with three guiding
principles:

1. **Memory efficiency** -- files are parsed once with :func:`collect_gff`
   and processed one chromosome (``seqid``) at a time.
2. **Strand awareness** -- transcripts are only compared when they share
   the same ``seqid`` *and* ``strand``.
3. **Completeness** -- every GT transcript and every prediction transcript
   appears in the output.  Unmatched GT transcripts carry empty prediction
   lists; unmatched predictions carry an ``is_unmatched_prediction`` flag
   and their GT arrays are built from whatever annotation exists at that
   genomic position.

.. note::

   All arrays are in **genomic (left-to-right) orientation** regardless of
   strand.  Both GT and prediction arrays for a given mapping entry share
   the same coordinate space (anchored at ``gt_start``), so prediction
   features outside the GT span are clipped.

Public API
----------
- :class:`PredictionMatch`
- :class:`TranscriptMapping`
- :func:`map_transcripts`
- :func:`build_paired_arrays`
- :func:`export_mapping_table`
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import polars as pl
from pydantic import BaseModel

from .io_utils import DEFAULT_TRANSCRIPT_TYPES, collect_gff
from .label_definition import LabelConfig

logger = logging.getLogger(__name__)

# Sentinel prefix for synthetic GT entries created for unmatched predictions.
_UNMATCHED_PRED_PREFIX = "__unmatched_pred__"


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------


class PredictionMatch(BaseModel):
    """A single predicted transcript that was matched to a GT transcript."""

    model_config = {"frozen": True}

    predictor_name: str
    transcript_id: str
    start: int
    end: int
    overlap_fraction: float


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
        this starts with ``__unmatched_pred__``.
    gt_start, gt_end : int
        Genomic coordinates of the evaluation window (1-based, inclusive).
    is_unmatched_prediction : bool
        ``True`` when no real GT transcript overlapped the prediction.
        The GT array will be built from whatever features exist at
        that genomic position.
    matched_predictions : list[PredictionMatch]
        Predictions that satisfied the overlap threshold.  Empty when
        no predictor matched this GT transcript.
    """

    model_config = {"frozen": True}

    seqid: str
    strand: str
    gt_id: str
    gt_start: int
    gt_end: int
    is_unmatched_prediction: bool = False
    matched_predictions: list[PredictionMatch] = []


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------


def _compute_overlap_fraction(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> float:
    """Return the overlap between two intervals as a fraction of the shorter.

    Both intervals are assumed to be 1-based, inclusive.  Returns 0.0
    when the intervals do not overlap at all.
    """
    overlap_start = max(start_a, start_b)
    overlap_end = min(end_a, end_b)
    overlap_length = max(0, overlap_end - overlap_start + 1)

    if overlap_length == 0:
        return 0.0

    length_a = end_a - start_a + 1
    length_b = end_b - start_b + 1
    shorter = min(length_a, length_b)

    return overlap_length / shorter


def _extract_transcripts_df(
    collected_df: pl.DataFrame,
    transcript_types: list[str],
) -> pl.DataFrame:
    """Filter *collected_df* to transcript rows and return a clean table.

    Returns
    -------
    pl.DataFrame
        Columns: ``gff_id``, ``start``, ``end``, ``strand``.
        Rows with null essential fields are dropped.
    """
    return (
        collected_df
        .filter(pl.col("type").is_in(transcript_types))
        .select("gff_id", "start", "end", "strand")
        .drop_nulls(subset=["gff_id", "start", "end", "strand"])
    )



def _find_overlapping_pairs(
    gt_transcripts: pl.DataFrame,
    pred_transcripts: pl.DataFrame,
    predictor_name: str,
    min_overlap: float = 0.2,
) -> list[tuple[str, str, int, int, int, int, float]]:
    """Find all (GT, pred) pairs with >= *min_overlap* coordinate overlap.

    Both DataFrames must contain columns ``gff_id``, ``start``, ``end``.
    Only rows that share the same strand should be passed in (caller is
    responsible for pre-filtering by strand).

    Returns
    -------
    list[tuple]
        ``(gt_id, pred_id, gt_start, gt_end, pred_start, pred_end,
        overlap_frac)`` for every qualifying pair.
    """
    pairs: list[tuple[str, str, int, int, int, int, float]] = []

    # Materialise to Python for the cross comparison.  Per-chromosome,
    # per-strand cardinality is typically small (hundreds), so this is
    # acceptable.
    gt_rows = gt_transcripts.select("gff_id", "start", "end").to_dicts()
    pred_rows = pred_transcripts.select("gff_id", "start", "end").to_dicts()

    for gt_row in gt_rows:
        gt_id = gt_row["gff_id"]
        gt_s = gt_row["start"]
        gt_e = gt_row["end"]

        for pred_row in pred_rows:
            frac = _compute_overlap_fraction(
                gt_s, gt_e, pred_row["start"], pred_row["end"],
            )
            if frac >= min_overlap:
                pairs.append((
                    gt_id,
                    pred_row["gff_id"],
                    gt_s,
                    gt_e,
                    pred_row["start"],
                    pred_row["end"],
                    frac,
                ))

    return pairs


# ------------------------------------------------------------------
# Public: mapping
# ------------------------------------------------------------------


def map_transcripts(
    gt_path: str | Path,
    pred_paths: dict[str, str | Path],
    *,
    min_overlap: float = 0.2,
    transcript_types: list[str] | None = None,
    exclude_features: list[str] | None = None,
) -> list[TranscriptMapping]:
    """Map GT transcripts to predicted transcripts across multiple predictors.

    Processing is chunked by ``seqid`` (chromosome) so that only one
    chromosome's worth of data is materialised at any time.

    Parameters
    ----------
    gt_path : str | Path
        Path to the ground-truth GFF/GTF file.
    pred_paths : dict[str, str | Path]
        ``{predictor_name: path}`` for each prediction file.
    min_overlap : float
        Minimum overlap fraction (relative to the shorter transcript) to
        consider two transcripts as matching.  Default ``0.2`` (20 %).
    transcript_types : list[str] | None
        Feature types that define transcript boundaries.
        Defaults to :data:`DEFAULT_TRANSCRIPT_TYPES`.
    exclude_features : list[str] | None
        Feature types to ignore entirely (e.g. ``["gene"]``).

    Returns
    -------
    list[TranscriptMapping]
        One entry per GT transcript (including those with no matches),
        plus entries for unmatched predictions.

    """
    transcript_types = transcript_types or DEFAULT_TRANSCRIPT_TYPES
    exclude_features = exclude_features or []

    # 1. Read all files once into memory ------------------------------------
    gt_df = collect_gff(str(gt_path), exclude_features=exclude_features)
    pred_dfs: dict[str, pl.DataFrame] = {
        name: collect_gff(str(path), exclude_features=exclude_features)
        for name, path in pred_paths.items()
    }

    # 2. Discover unique seqids from both GT and predictions ----------------
    gt_seqids: set[str] = set(
        gt_df
        .get_column("seqid")
        .unique()
        .to_list()
    )

    pred_seqids: set[str] = set()
    for pred_df in pred_dfs.values():
        pred_seqids.update(
            pred_df.get_column("seqid").unique().to_list()
        )

    all_seqids = gt_seqids | pred_seqids
    logger.info(
        "Found %d GT seqid(s), %d prediction seqid(s) (%d total).",
        len(gt_seqids),
        len(pred_seqids),
        len(all_seqids),
    )

    # 3. Process chromosome-by-chromosome -----------------------------------
    all_mappings: list[TranscriptMapping] = []

    for seqid in sorted(all_seqids):
        chunk_mappings = _process_single_seqid(
            seqid=seqid,
            gt_df=gt_df,
            pred_dfs=pred_dfs,
            transcript_types=transcript_types,
            min_overlap=min_overlap,
        )
        all_mappings.extend(chunk_mappings)

    n_unmatched = sum(1 for m in all_mappings if m.is_unmatched_prediction)
    n_no_pred = sum(
        1 for m in all_mappings
        if not m.is_unmatched_prediction and not m.matched_predictions
    )
    logger.info(
        "Mapping complete: %d entries (%d unmatched predictions, "
        "%d GT transcripts with no match).",
        len(all_mappings),
        n_unmatched,
        n_no_pred,
    )

    return all_mappings


# ------------------------------------------------------------------
# Internal: per-chromosome processing
# ------------------------------------------------------------------


def _process_single_seqid(
    seqid: str,
    gt_df: pl.DataFrame,
    pred_dfs: dict[str, pl.DataFrame],
    transcript_types: list[str],
    min_overlap: float,
) -> list[TranscriptMapping]:
    """Process a single chromosome and return its transcript mappings."""
    gt_chunk = gt_df.filter(pl.col("seqid") == seqid)

    pred_chunks: dict[str, pl.DataFrame] = {
        name: df.filter(pl.col("seqid") == seqid)
        for name, df in pred_dfs.items()
    }

    gt_transcripts = _extract_transcripts_df(gt_chunk, transcript_types)

    pred_transcripts_by_name: dict[str, pl.DataFrame] = {
        name: _extract_transcripts_df(chunk, transcript_types)
        for name, chunk in pred_chunks.items()
    }

    # Process per strand
    mappings: list[TranscriptMapping] = []

    for strand in ["+", "-"]:
        gt_strand = gt_transcripts.filter(pl.col("strand") == strand)
        pred_strand = {
            name: df.filter(pl.col("strand") == strand)
            for name, df in pred_transcripts_by_name.items()
        }

        strand_mappings = _map_strand(
            seqid=seqid,
            strand=strand,
            gt_transcripts=gt_strand,
            pred_transcripts_by_name=pred_strand,
            min_overlap=min_overlap,
        )
        mappings.extend(strand_mappings)

    return mappings


def _map_strand(
    seqid: str,
    strand: str,
    gt_transcripts: pl.DataFrame,
    pred_transcripts_by_name: dict[str, pl.DataFrame],
    min_overlap: float,
) -> list[TranscriptMapping]:
    """Map GT to predictions for one seqid + strand combination.

    Returns
    -------
    list[TranscriptMapping]
        Entries for *all* GT transcripts (including those with no
        matching prediction), plus entries for unmatched predictions.
    """
    # Track which prediction transcripts have been matched
    matched_pred_ids: dict[str, set[str]] = {
        name: set() for name in pred_transcripts_by_name
    }

    # Collect matches grouped by GT transcript
    gt_to_matches: dict[str, list[PredictionMatch]] = {}
    gt_info: dict[str, tuple[int, int]] = {}

    for gt_row in gt_transcripts.to_dicts():
        gt_id = gt_row["gff_id"]
        gt_info[gt_id] = (gt_row["start"], gt_row["end"])
        gt_to_matches[gt_id] = []

    # Find overlapping pairs for each predictor
    for pred_name, pred_df in pred_transcripts_by_name.items():
        if pred_df.is_empty() or gt_transcripts.is_empty():
            continue

        pairs = _find_overlapping_pairs(
            gt_transcripts, pred_df, pred_name, min_overlap,
        )

        for gt_id, pred_id, _, _, pred_s, pred_e, frac in pairs:
            gt_to_matches[gt_id].append(PredictionMatch(
                predictor_name=pred_name,
                transcript_id=pred_id,
                start=pred_s,
                end=pred_e,
                overlap_fraction=frac,
            ))
            matched_pred_ids[pred_name].add(pred_id)

    # Build TranscriptMapping for EVERY GT transcript (including
    # those with no matches — the empty list signals "no predictions").
    mappings: list[TranscriptMapping] = []

    for gt_id, matches in gt_to_matches.items():
        gt_s, gt_e = gt_info[gt_id]
        mappings.append(TranscriptMapping(
            seqid=seqid,
            strand=strand,
            gt_id=gt_id,
            gt_start=gt_s,
            gt_end=gt_e,
            matched_predictions=matches,
        ))

    # Create entries for unmatched predictions.
    # Pre-index each pred_df for efficient lookup.
    for pred_name, pred_df in pred_transcripts_by_name.items():
        if pred_df.is_empty():
            continue

        all_pred_ids = set(pred_df.get_column("gff_id").to_list())
        unmatched_ids = all_pred_ids - matched_pred_ids[pred_name]

        if not unmatched_ids:
            continue

        # Build lookup once instead of filtering per-ID
        pred_lookup = {
            row["gff_id"]: row
            for row in pred_df.to_dicts()
        }

        for pred_id in sorted(unmatched_ids):
            pred_row = pred_lookup[pred_id]
            pred_s = pred_row["start"]
            pred_e = pred_row["end"]

            mappings.append(TranscriptMapping(
                seqid=seqid,
                strand=strand,
                gt_id=f"{_UNMATCHED_PRED_PREFIX}{pred_id}",
                gt_start=pred_s,
                gt_end=pred_e,
                is_unmatched_prediction=True,
                matched_predictions=[
                    PredictionMatch(
                        predictor_name=pred_name,
                        transcript_id=pred_id,
                        start=pred_s,
                        end=pred_e,
                        overlap_fraction=0.0,
                    )
                ],
            ))

    return mappings


# ------------------------------------------------------------------
# Public: array construction
# ------------------------------------------------------------------


def build_paired_arrays(
    mapping: TranscriptMapping,
    gt_df: pl.DataFrame,
    pred_dfs: dict[str, pl.DataFrame],
    label_config: LabelConfig,
    transcript_types: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build the GT and per-predictor annotation arrays for one mapping.

    For unmatched predictions (``is_unmatched_prediction=True``), the GT
    array is built from whatever GT features exist in the prediction's
    genomic region — not just filled with background.

    For predictors with no match for this mapping, an all-background
    prediction array is returned.

    Parameters
    ----------
    mapping : TranscriptMapping
        A single mapping entry from :func:`map_transcripts`.
    gt_df : pl.DataFrame
        Pre-collected GT GFF DataFrame (from :func:`collect_gff`).
    pred_dfs : dict[str, pl.DataFrame]
        ``{predictor_name: DataFrame}`` for each prediction file.
    label_config : LabelConfig
        Label configuration defining integer tokens.
    transcript_types : list[str] | None
        Feature types that denote transcript boundaries.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        ``(gt_array, {predictor_name: pred_array})``.
        The dict contains an entry for every predictor in *pred_dfs*.
    """
    transcript_types = transcript_types or DEFAULT_TRANSCRIPT_TYPES
    bg_val = label_config.background_label
    array_length = mapping.gt_end - mapping.gt_start + 1

    # --- GT array ---
    if mapping.is_unmatched_prediction:
        gt_array = _build_region_annotation_array(
            df=gt_df,
            seqid=mapping.seqid,
            region_start=mapping.gt_start,
            array_length=array_length,
            label_config=label_config,
            transcript_types=transcript_types,
        )
    else:
        gt_array = _build_annotation_array_from_df(
            df=gt_df,
            transcript_id=mapping.gt_id,
            seqid=mapping.seqid,
            region_start=mapping.gt_start,
            array_length=array_length,
            label_config=label_config,
            transcript_types=transcript_types,
        )

    # --- Prediction arrays ---
    matched_by_predictor: dict[str, PredictionMatch] = {}
    for match in mapping.matched_predictions:
        matched_by_predictor[match.predictor_name] = match

    pred_arrays: dict[str, np.ndarray] = {}

    for pred_name, pred_df in pred_dfs.items():
        if pred_name in matched_by_predictor:
            match = matched_by_predictor[pred_name]
            pred_arrays[pred_name] = _build_annotation_array_from_df(
                df=pred_df,
                transcript_id=match.transcript_id,
                seqid=mapping.seqid,
                region_start=mapping.gt_start,
                array_length=array_length,
                label_config=label_config,
                transcript_types=transcript_types,
            )
        else:
            pred_arrays[pred_name] = np.full(
                array_length, bg_val, dtype=np.int32,
            )

    return gt_array, pred_arrays


# ------------------------------------------------------------------
# Internal: array builders (no file I/O)
# ------------------------------------------------------------------


def _build_annotation_array_from_df(
    df: pl.DataFrame,
    transcript_id: str,
    seqid: str,
    region_start: int,
    array_length: int,
    label_config: LabelConfig,
    transcript_types: list[str],
) -> np.ndarray:
    """Build a 1-D annotation array for one transcript's children.

    Child features of *transcript_id* are projected into local
    coordinates ``[0, array_length)`` relative to *region_start*.

    Parameters
    ----------
    df : pl.DataFrame
        Pre-collected GFF DataFrame (already filtered for excludes).
    transcript_id : str
        The ``ID`` / ``transcript_id`` of the parent transcript.
    seqid : str
        Chromosome to filter on.
    region_start : int
        Genomic coordinate that becomes local index 0.
    array_length : int
        Size of the output array.
    label_config : LabelConfig
        Provides ``background_label`` and ``coding_label``.
    transcript_types : list[str]
        Feature types excluded from being painted as children.

    Returns
    -------
    np.ndarray
        1-D ``int32`` array of length *array_length*.
    """
    bg_val = label_config.background_label
    coding_val = label_config.coding_label
    arr = np.full(array_length, bg_val, dtype=np.int32)

    if coding_val is None:
        return arr

    children = df.filter(
        (pl.col("seqid") == seqid)
        & (pl.col("parent") == transcript_id)
        & (~pl.col("type").is_in(transcript_types))
    ).select("start", "end")

    _paint_features(arr, children, region_start, coding_val)

    return arr


def _build_region_annotation_array(
    df: pl.DataFrame,
    seqid: str,
    region_start: int,
    array_length: int,
    label_config: LabelConfig,
    transcript_types: list[str],
) -> np.ndarray:
    """Build a 1-D annotation array from all GT features in a region.

    Used for unmatched predictions where no specific GT transcript was
    matched.  Paints **all** non-transcript child features that overlap
    the region, regardless of which parent transcript they belong to.
    This allows the GT array to reflect lncRNA or other annotations
    present in the GT at the prediction's location.

    Parameters
    ----------
    df : pl.DataFrame
        Pre-collected GT GFF DataFrame.
    seqid : str
        Chromosome to filter on.
    region_start : int
        Genomic coordinate that becomes local index 0.
    array_length : int
        Size of the output array.
    label_config : LabelConfig
        Provides ``background_label`` and ``coding_label``.
    transcript_types : list[str]
        Feature types excluded from being painted.

    Returns
    -------
    np.ndarray
        1-D ``int32`` array of length *array_length*.
    """
    bg_val = label_config.background_label
    coding_val = label_config.coding_label
    arr = np.full(array_length, bg_val, dtype=np.int32)

    if coding_val is None:
        return arr

    region_end = region_start + array_length - 1

    children = df.filter(
        (pl.col("seqid") == seqid)
        & (pl.col("parent").is_not_null())
        & (~pl.col("type").is_in(transcript_types))
        & (pl.col("start") <= region_end)
        & (pl.col("end") >= region_start)
    ).select("start", "end")

    _paint_features(arr, children, region_start, coding_val)

    return arr


def _paint_features(
    arr: np.ndarray,
    features_df: pl.DataFrame,
    region_start: int,
    label_value: int,
) -> None:
    """Paint feature intervals into *arr* (in-place).

    Converts 1-based inclusive GFF coordinates to 0-based array indices
    relative to *region_start*.
    """
    for row in features_df.iter_rows(named=True):
        feat_start = row["start"]
        feat_end = row["end"]

        if feat_start is None or feat_end is None:
            continue

        local_start = max(0, feat_start - region_start)
        local_end = min(len(arr), feat_end - region_start + 1)

        if local_start < local_end:
            arr[local_start:local_end] = label_value


# ------------------------------------------------------------------
# Public: debug export
# ------------------------------------------------------------------


def export_mapping_table(
    mappings: list[TranscriptMapping],
    output_path: str | Path,
) -> Path:
    """Write the mapping list to a human-readable TSV for debugging.

    Columns: ``seqid``, ``strand``, ``gt_id``, ``gt_start``, ``gt_end``,
    ``is_unmatched_prediction``, ``predictor``, ``pred_id``,
    ``pred_start``, ``pred_end``, ``overlap_fraction``.

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
        "seqid", "strand", "gt_id", "gt_start", "gt_end",
        "is_unmatched_prediction", "predictor", "pred_id",
        "pred_start", "pred_end", "overlap_fraction",
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
                writer.writerow({
                    **base_row,
                    "predictor": "",
                    "pred_id": "",
                    "pred_start": "",
                    "pred_end": "",
                    "overlap_fraction": "",
                })
            else:
                for match in mapping.matched_predictions:
                    writer.writerow({
                        **base_row,
                        "predictor": match.predictor_name,
                        "pred_id": match.transcript_id,
                        "pred_start": match.start,
                        "pred_end": match.end,
                        "overlap_fraction": (
                            f"{match.overlap_fraction:.4f}"
                        ),
                    })

    logger.info("Mapping table written to %s", output_path)
    return output_path