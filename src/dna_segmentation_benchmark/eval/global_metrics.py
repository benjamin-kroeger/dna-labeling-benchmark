"""Global annotation-level metrics for the DNA segmentation benchmark.

Computes metrics over the full set of reference and predicted transcripts,
comparable to gffcompare's nucleotide and exon sensitivity/precision.

Unlike the per-transcript metrics (which evaluate matched pairs in isolation),
global metrics aggregate over *all* transcripts — including unmatched ones —
so false-positive predictions and missed reference transcripts both contribute
to the final numbers.

Four metric groups are computed, each answering a distinct question:

* ``nucleotide``  — *"At the base level, how accurate is the exon coverage?"*
  Union-based: each genomic base is counted once regardless of isoform count.
  Equivalent to gffcompare's nucleotide sensitivity/precision.

* ``exon``  — *"How many exons are exactly reconstructed?"*
  Per-transcript counting (gffcompare style): a shared exon between two
  isoforms counts once per isoform.  An exon is matched when any transcript
  on the opposite side carries the identical (seqid, strand, start, end).

* ``transcript``  — *"How many transcripts are recovered?"*
  Sensitivity = matched ref transcripts / total ref transcripts.
  Precision   = matched pred transcripts / total pred transcripts.

* ``gene``  — *"How many gene loci are detected?"*
  Transcripts are clustered into loci by coordinate overlap (same algorithm
  as map_transcripts).  A locus is matched if any of its transcripts was
  assigned a counterpart on the other side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..label_definition import LabelConfig
from ..transcript_mapping import TranscriptMapping


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_global_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    mappings: list[TranscriptMapping],
    predictor_name: str,
    label_config: LabelConfig,
    gt_exon_types: list[str],
    pred_exon_types: list[str],
    transcript_types: list[str],
) -> dict:
    """Compute global annotation-level metrics for one predictor.

    Parameters
    ----------
    gt_df : pd.DataFrame
        Pre-collected ground-truth GFF DataFrame (from ``collect_gff``).
    pred_df : pd.DataFrame
        Pre-collected prediction GFF DataFrame for this predictor.
    mappings : list[TranscriptMapping]
        Transcript mapping result from ``map_transcripts``.
    predictor_name : str
        Name of the predictor; must match the name used in ``map_transcripts``.
    label_config : LabelConfig
        Label configuration.  ``coding_label`` must be set for nucleotide
        metrics; if it is ``None`` the nucleotide section is returned empty.
    gt_exon_types : list[str]
        GT GFF/GTF feature types that represent exon/coding intervals.
    pred_exon_types : list[str]
        Prediction GFF/GTF feature types that represent exon/coding intervals.
    transcript_types : list[str]
        GFF feature types that define transcript boundaries
        (e.g. ``["mRNA", "transcript"]``).

    Returns
    -------
    dict
        Five keys: ``"nucleotide"``, ``"exon"``, ``"exon_lenient"``,
        ``"transcript"``, ``"gene"``.
        Each value is a flat dict of counts and derived P/R/F1 scores.
        ``"exon"`` uses exact boundary matching; ``"exon_lenient"`` relaxes the
        outer boundary of terminal exons (gffcompare style).
    """
    return {
        "nucleotide": _compute_global_nucleotide_metrics(
            gt_df, pred_df, label_config, gt_exon_types, pred_exon_types, transcript_types,
        ),
        "exon": _compute_global_exon_metrics(
            gt_df, pred_df, gt_exon_types, pred_exon_types,
        ),
        "exon_lenient": _compute_global_exon_lenient_metrics(
            gt_df, pred_df, gt_exon_types, pred_exon_types,
        ),
        "transcript": _compute_transcript_level_metrics(
            mappings, predictor_name,
        ),
        "gene": _compute_gene_level_metrics(
            gt_df, pred_df, mappings, predictor_name, transcript_types,
        ),
    }


# ---------------------------------------------------------------------------
# Nucleotide metrics — union-based
# ---------------------------------------------------------------------------


def _compute_global_nucleotide_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label_config: LabelConfig,
    gt_exon_types: list[str],
    pred_exon_types: list[str],
    transcript_types: list[str],
) -> dict:
    """Nucleotide precision/recall/F1 using union-of-exons per locus.

    Evaluation space: the union of all ref and pred transcript spans on the
    same seqid+strand, merged into non-overlapping regions.  Within each
    region two binary arrays are built — one marking ref-exonic bases, one
    marking pred-exonic bases — and TP/FP/FN are accumulated.

    This matches gffcompare's methodology: each genomic base is counted once
    regardless of how many isoforms cover it.
    """
    if label_config.coding_label is None:
        return {}

    coding_val = label_config.coding_label
    bg_val = label_config.background_label

    total_tp = total_fp = total_fn = 0

    all_seqids = (
        set(gt_df["seqid"].dropna().unique())
        | set(pred_df["seqid"].dropna().unique())
    )

    for seqid in sorted(all_seqids):
        for strand in ("+", "-"):
            ref_spans = _get_transcript_spans(gt_df, seqid, strand, transcript_types)
            pred_spans = _get_transcript_spans(pred_df, seqid, strand, transcript_types)

            all_spans = ref_spans + pred_spans
            if not all_spans:
                continue

            for region_start, region_end in _merge_intervals(all_spans):
                length = region_end - region_start + 1

                ref_arr = _build_exon_union_array(
                    gt_df, seqid, strand, region_start, length,
                    gt_exon_types, coding_val, bg_val,
                )
                pred_arr = _build_exon_union_array(
                    pred_df, seqid, strand, region_start, length,
                    pred_exon_types, coding_val, bg_val,
                )

                ref_exonic = ref_arr == coding_val
                pred_exonic = pred_arr == coding_val
                total_tp += int(np.sum(ref_exonic & pred_exonic))
                total_fp += int(np.sum(~ref_exonic & pred_exonic))
                total_fn += int(np.sum(ref_exonic & ~pred_exonic))

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


# ---------------------------------------------------------------------------
# Exon metrics — de-duplicated exact boundary matching
# ---------------------------------------------------------------------------


def _compute_global_exon_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    gt_exon_types: list[str],
    pred_exon_types: list[str],
) -> dict:
    """Exon sensitivity/precision using de-duplicated exact boundary matching.

    Each unique exon interval ``(seqid, strand, start, end)`` is counted
    once, regardless of how many transcripts carry it.  Counting shared
    exons multiple times would reward predicting highly expressed constitutive
    exons and distort the metric away from structural reconstruction quality.

    An exon is matched when the **exact** ``(seqid, strand, start, end)``
    appears on both sides.  No terminal-exon leniency is applied: both the
    splice-site boundary and the transcript-start/end boundary must match.
    This is stricter than gffcompare (which relaxes the external boundary of
    first/last exons), but avoids the ambiguity introduced by UTR variation
    and TSS heterogeneity.
    """
    ref_exon_keys  = _collect_exon_keys(gt_df, gt_exon_types)
    pred_exon_keys = _collect_exon_keys(pred_df, pred_exon_types)

    n_matched  = len(ref_exon_keys & pred_exon_keys)
    ref_total  = len(ref_exon_keys)
    pred_total = len(pred_exon_keys)

    sensitivity = n_matched / ref_total  if ref_total  > 0 else 0.0
    precision   = n_matched / pred_total if pred_total > 0 else 0.0

    return {
        "ref_exon_count":   ref_total,
        "ref_exon_matched": n_matched,
        "pred_exon_count":   pred_total,
        "pred_exon_matched": n_matched,
        "sensitivity": sensitivity,
        "precision":   precision,
        "f1": _f1(sensitivity, precision),
    }


def _compute_global_exon_lenient_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    gt_exon_types: list[str],
    pred_exon_types: list[str],
) -> dict:
    """Exon sensitivity/precision with terminal-exon boundary leniency.

    Equivalent to gffcompare's exon-level metric: the outer boundary of the
    first and last exon in each transcript (transcription start/end site) is
    not required to match.  Only internal splice-site boundaries must be exact.

    Concretely:
    * **First exon** per transcript (lowest start): only the 3' boundary
      (splice-donor position, ``end``) is required to match.
    * **Last exon** per transcript (highest end): only the 5' boundary
      (splice-acceptor position, ``start``) is required to match.
    * **Internal exons**: both boundaries must match exactly (same as strict).
    * **Single-exon transcripts**: both boundaries compared strictly.

    Matching is performed on de-duplicated lenient canonical keys, so each
    distinct splice-site combination is counted once regardless of isoform count.
    """
    ref_keys = _collect_exon_keys_lenient(gt_df, gt_exon_types)
    pred_keys = _collect_exon_keys_lenient(pred_df, pred_exon_types)

    n_matched = len(ref_keys & pred_keys)
    ref_total = len(ref_keys)
    pred_total = len(pred_keys)

    sensitivity = n_matched / ref_total if ref_total > 0 else 0.0
    precision = n_matched / pred_total if pred_total > 0 else 0.0

    return {
        "ref_exon_count": ref_total,
        "ref_exon_matched": n_matched,
        "pred_exon_count": pred_total,
        "pred_exon_matched": n_matched,
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": _f1(sensitivity, precision),
    }


def _collect_exon_keys_lenient(
    df: pd.DataFrame,
    exon_types: list[str],
) -> set[tuple]:
    """Return lenient canonical keys for exons.

    For each transcript (grouped by ``parent``):
    * First exon (index 0 when sorted by start): ``(seqid, strand, None, end)``
      — the 5' terminal boundary is relaxed, only the splice-donor end must match.
    * Last exon: ``(seqid, strand, start, None)``
      — the 3' terminal boundary is relaxed, only the splice-acceptor start must match.
    * Internal exons: ``(seqid, strand, start, end)`` — both boundaries are splice sites.
    * Single-exon transcripts: ``(seqid, strand, start, end)`` — strict matching.
    * Exons with no parent annotation: included with strict keys.
    """
    exon_mask = df["type"].isin(exon_types)
    exons = df[exon_mask]
    if exons.empty:
        return set()

    keys: set[tuple] = set()

    has_parent = "parent" in df.columns

    if has_parent:
        with_parent = exons[exons["parent"].notna()]
        if not with_parent.empty:
            for _, group in with_parent.groupby("parent"):
                sorted_g = group.sort_values("start")
                n = len(sorted_g)
                for i, row in enumerate(sorted_g.itertuples(index=False)):
                    seqid = row.seqid
                    strand = row.strand
                    start = int(row.start)
                    end = int(row.end)
                    if n == 1:
                        keys.add((seqid, strand, start, end))
                    elif i == 0:
                        keys.add((seqid, strand, None, end))
                    elif i == n - 1:
                        keys.add((seqid, strand, start, None))
                    else:
                        keys.add((seqid, strand, start, end))

        no_parent = exons[exons["parent"].isna()]
        for row in no_parent.itertuples(index=False):
            keys.add((row.seqid, row.strand, int(row.start), int(row.end)))
    else:
        for row in exons.itertuples(index=False):
            keys.add((row.seqid, row.strand, int(row.start), int(row.end)))

    return keys


def _collect_exon_keys(
    df: pd.DataFrame,
    exon_types: list[str],
) -> set[tuple[str, str, int, int]]:
    """Return a set of unique ``(seqid, strand, start, end)`` for all exon rows."""
    mask = (
        df["type"].isin(exon_types)
        & df["seqid"].notna()
        & df["strand"].notna()
        & df["start"].notna()
        & df["end"].notna()
    )
    exons = df[mask]
    return set(zip(
        exons["seqid"],
        exons["strand"],
        exons["start"].astype(int),
        exons["end"].astype(int),
    ))


# ---------------------------------------------------------------------------
# Transcript-level metrics
# ---------------------------------------------------------------------------


def _compute_transcript_level_metrics(
    mappings: list[TranscriptMapping],
    predictor_name: str,
) -> dict:
    """Transcript sensitivity/precision from the mapping result.

    A reference transcript is "matched" if the predictor assigned a
    prediction to it.  Predicted transcripts that were not assigned to any
    reference transcript (``is_unmatched_prediction``) are counted as
    unmatched predictions, reducing precision.
    """
    ref_total = ref_matched = pred_total = pred_matched = 0

    for mapping in mappings:
        pred_hits = [
            m for m in mapping.matched_predictions
            if m.predictor_name == predictor_name
        ]

        if mapping.is_unmatched_prediction:
            if pred_hits:
                pred_total += 1          # unmatched pred lowers precision
        else:
            ref_total += 1
            if pred_hits:
                ref_matched += 1
                pred_total   += 1
                pred_matched += 1

    sensitivity = ref_matched  / ref_total  if ref_total  > 0 else 0.0
    precision   = pred_matched / pred_total if pred_total > 0 else 0.0

    return {
        "ref_transcript_count":   ref_total,
        "ref_transcript_matched": ref_matched,
        "pred_transcript_count":   pred_total,
        "pred_transcript_matched": pred_matched,
        "sensitivity": sensitivity,
        "precision":   precision,
        "f1": _f1(sensitivity, precision),
    }


# ---------------------------------------------------------------------------
# Gene / locus-level metrics
# ---------------------------------------------------------------------------


def _compute_gene_level_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    mappings: list[TranscriptMapping],
    predictor_name: str,
    transcript_types: list[str],
) -> dict:
    """Gene/locus sensitivity and precision.

    Transcripts are clustered into loci per seqid+strand by coordinate
    overlap (same O(n log n) sweep used in ``map_transcripts``).  A locus
    is "matched" when at least one of its transcripts was assigned a
    counterpart on the opposite side.
    """
    matched_gt_ids = {
        mapping.gt_id
        for mapping in mappings
        if not mapping.is_unmatched_prediction
        and any(m.predictor_name == predictor_name for m in mapping.matched_predictions)
    }
    matched_pred_ids = {
        m.transcript_id
        for mapping in mappings
        if not mapping.is_unmatched_prediction
        for m in mapping.matched_predictions
        if m.predictor_name == predictor_name
    }

    gt_locus_count,   gt_locus_matched   = _count_matched_loci(gt_df,   transcript_types, matched_gt_ids)
    pred_locus_count, pred_locus_matched = _count_matched_loci(pred_df, transcript_types, matched_pred_ids)

    sensitivity = gt_locus_matched   / gt_locus_count   if gt_locus_count   > 0 else 0.0
    precision   = pred_locus_matched / pred_locus_count if pred_locus_count > 0 else 0.0

    return {
        "ref_locus_count":   gt_locus_count,
        "ref_locus_matched": gt_locus_matched,
        "pred_locus_count":   pred_locus_count,
        "pred_locus_matched": pred_locus_matched,
        "sensitivity": sensitivity,
        "precision":   precision,
        "f1": _f1(sensitivity, precision),
    }


def _count_matched_loci(
    df: pd.DataFrame,
    transcript_types: list[str],
    matched_ids: set[str],
) -> tuple[int, int]:
    """Count total loci and matched loci in a GFF DataFrame.

    Returns
    -------
    tuple[int, int]
        ``(total_loci, matched_loci)``
    """
    locus_count = locus_matched = 0

    for seqid in sorted(df["seqid"].dropna().unique()):
        for strand in ("+", "-"):
            spans_with_ids = _get_transcript_spans_with_ids(df, seqid, strand, transcript_types)
            if not spans_with_ids:
                continue
            for locus_ids in _cluster_into_loci(spans_with_ids):
                locus_count += 1
                if any(tid in matched_ids for tid in locus_ids):
                    locus_matched += 1

    return locus_count, locus_matched


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _get_transcript_spans(
    df: pd.DataFrame,
    seqid: str,
    strand: str,
    transcript_types: list[str],
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` for all transcripts on a seqid+strand."""
    mask = (
        (df["seqid"] == seqid)
        & (df["strand"] == strand)
        & df["type"].isin(transcript_types)
        & df["start"].notna()
        & df["end"].notna()
    )
    rows = df[mask]
    return list(zip(rows["start"].astype(int), rows["end"].astype(int)))


def _get_transcript_spans_with_ids(
    df: pd.DataFrame,
    seqid: str,
    strand: str,
    transcript_types: list[str],
) -> list[tuple[int, int, str]]:
    """Return ``(start, end, gff_id)`` for all transcripts on a seqid+strand."""
    mask = (
        (df["seqid"] == seqid)
        & (df["strand"] == strand)
        & df["type"].isin(transcript_types)
        & df["start"].notna()
        & df["end"].notna()
        & df["gff_id"].notna()
    )
    rows = df[mask]
    return list(zip(
        rows["start"].astype(int),
        rows["end"].astype(int),
        rows["gff_id"].astype(str),
    ))


def _merge_intervals(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent intervals into non-overlapping regions.

    Parameters
    ----------
    spans : list[tuple[int, int]]
        Unsorted list of ``(start, end)`` pairs (1-based inclusive).

    Returns
    -------
    list[tuple[int, int]]
        Sorted, non-overlapping list of ``(start, end)`` pairs.
    """
    sorted_spans = sorted(spans)
    merged: list[tuple[int, int]] = [sorted_spans[0]]

    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _cluster_into_loci(
    spans_with_ids: list[tuple[int, int, str]],
) -> list[list[str]]:
    """Group transcript ``(start, end, id)`` triples into overlapping loci.

    Uses the same O(n log n) coordinate sweep as ``_build_loci`` in
    ``transcript_mapping``.  Returns a list of loci, each being a list of
    transcript IDs that mutually overlap.
    """
    sorted_spans = sorted(spans_with_ids, key=lambda x: (x[0], x[1]))
    loci: list[list[str]] = []
    current_ids: list[str] = [sorted_spans[0][2]]
    current_end: int = sorted_spans[0][1]

    for start, end, tid in sorted_spans[1:]:
        if start <= current_end:
            current_ids.append(tid)
            current_end = max(current_end, end)
        else:
            loci.append(current_ids)
            current_ids = [tid]
            current_end = end

    loci.append(current_ids)
    return loci


def _build_exon_union_array(
    df: pd.DataFrame,
    seqid: str,
    strand: str,
    region_start: int,
    array_length: int,
    exon_types: list[str],
    coding_val: int,
    bg_val: int,
) -> np.ndarray:
    """Build a 1-D array marking the union of all exon intervals in a region.

    Positions overlapping any exon feature are set to ``coding_val``; all
    other positions are set to ``bg_val``.  Coordinates are clipped to
    ``[0, array_length)`` relative to ``region_start``.

    Parameters
    ----------
    region_start : int
        1-based inclusive start of the region (used as array offset).
    array_length : int
        Length of the output array.
    """
    arr = np.full(array_length, bg_val, dtype=np.int32)
    region_end = region_start + array_length - 1

    mask = (
        (df["seqid"] == seqid)
        & (df["strand"] == strand)
        & df["type"].isin(exon_types)
        & df["start"].notna()
        & df["end"].notna()
        & (df["start"] <= region_end)
        & (df["end"] >= region_start)
    )
    exons = df[mask]

    for feat_start, feat_end in zip(exons["start"], exons["end"]):
        local_start = max(0, int(feat_start) - region_start)
        local_end   = min(array_length, int(feat_end) - region_start + 1)
        if local_start < local_end:
            arr[local_start:local_end] = coding_val

    return arr


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; 0.0 when both are zero."""
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0
