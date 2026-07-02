"""Global annotation-level metrics for the DNA segmentation benchmark.

Computes metrics over the full set of reference and predicted transcripts,
comparable to gffcompare's nucleotide and exon sensitivity/precision.

Unlike the per-transcript metrics (which evaluate matched pairs in isolation),
global metrics aggregate over *all* transcripts — including unmatched ones —
so false-positive predictions and missed reference transcripts both contribute
to the final numbers.

Eight metric groups are computed, each answering a distinct question:

* ``nucleotide``  — *"At the base level, how accurate is the exon coverage?"*
  Union-based: each genomic base is counted once regardless of isoform count.
  Equivalent to gffcompare's nucleotide sensitivity/precision.

* ``exon``  — *"How many exons are exactly reconstructed?"*
  De-duplicated counting: each unique ``(seqid, strand, start, end)``
  interval is counted **once across all transcripts**, regardless of how
  many isoforms share it.  This diverges from gffcompare's per-isoform
  counting; see :func:`_compute_exon_and_structure_metrics` for the rationale.

* ``exon_lenient``  — *"Exon recovery with TSS/TES boundary leniency."*
  Terminal-exon outer boundaries are not required to match (gffcompare
  default ``=``); only internal splice-site boundaries must be exact.

* ``intron_chain``  — *"How many multi-exon intron chains match exactly?"*
  Reproduces gffcompare's intron-chain Sn/Sp by coordinate-exact structure
  matching over the full reference, independent of the locus-matching mode.

* ``transcript_exact``  — *"How many transcripts match exactly (coordinate-exact)?"*
  Reproduces gffcompare's transcript Sn/Sp by coordinate-exact structure
  matching over the full reference, independent of the locus-matching mode.

* ``transcript``  — *"How many transcripts are recovered?"*
  Sensitivity = matched ref transcripts / total ref transcripts.
  Precision   = matched pred transcripts / total pred transcripts.

* ``gene``  — *"How many gene loci are detected?"*
  Transcripts are clustered into loci by coordinate overlap (same algorithm
  as map_transcripts).  A locus is matched if any of its transcripts was
  assigned a counterpart on the other side.

* ``locus_isoform``  — *"How completely are multi-isoform loci recovered?"*
  For each GT locus, counts the fraction of its isoforms that received a
  prediction match.  Unlike the gene metric (which only checks for ANY match),
  this distinguishes a locus where 1 of 5 isoforms was found from one where
  all 5 were found.  Most meaningful with ``FULL_DISCOVERY`` matching, where
  every GT isoform participates in Hungarian assignment.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import _sweep_cluster
from ..feature_roles import FeatureRoleMap, feature_types_for_scope, normalize_feature_role_map
from ..label_definition import BenchmarkScope
from ..label_definition import LabelConfig
from ..transcript_mapping import (
    LocusMatchingMode,
    TranscriptMapping,
    _include_mapping_for_predictor,
)

# (seqid, strand) -> rows. Pre-built once in compute_global_metrics so the
# per-scope / per-region / per-locus helpers slice in O(1) instead of masking
# the whole DataFrame on every call.
_SSIndex = dict[tuple[str, str], pd.DataFrame]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_global_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    mappings: list[TranscriptMapping],
    predictor_name: str,
    label_config: LabelConfig,
    transcript_types: list[str],
    gt_feature_role_map: FeatureRoleMap | None = None,
    pred_feature_role_map: FeatureRoleMap | None = None,
    locus_matching_mode: LocusMatchingMode = LocusMatchingMode.FULL_DISCOVERY,
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
        Label configuration.
    transcript_types : list[str]
        GFF feature types that define transcript boundaries
        (e.g. ``["mRNA", "transcript"]``).

    Returns
    -------
    dict
        Eight keys: ``"nucleotide"``, ``"exon"``, ``"exon_lenient"``,
        ``"intron_chain"``, ``"transcript_exact"``, ``"transcript"``,
        ``"gene"``, ``"locus_isoform"``.
        Each value is a flat dict of counts and derived P/R/F1 scores.
        ``"exon"`` uses exact boundary matching; ``"exon_lenient"`` relaxes the
        outer boundary of terminal exons (gffcompare style).
        ``"intron_chain"`` and ``"transcript_exact"`` reproduce gffcompare's
        intron-chain and transcript Sn/Sp by coordinate-exact structure matching
        over the full reference, independent of the locus-matching mode (see the
        respective functions).  ``"transcript"`` instead reports assignment-based
        transcript recall from ``mappings`` (mode-dependent) and is retained for
        backward compatibility.
        ``"locus_isoform"`` reports per-locus isoform recall — the fraction of
        GT isoforms per locus that received a match, addressing multi-isoform
        caller fairness.  Most meaningful with ``FULL_DISCOVERY`` matching.
    """
    gt_feature_role_map = normalize_feature_role_map(
        gt_feature_role_map, label_config, arg_name="gt_feature_role_map"
    )
    pred_feature_role_map = normalize_feature_role_map(
        pred_feature_role_map, label_config, arg_name="pred_feature_role_map"
    )

    # Pre-group once for O(1) (seqid, strand) slices, reused across every scope,
    # region and locus below.  ``observed=True`` is required because seqid/strand
    # are categorical (default would emit the empty category Cartesian product).
    gt_by_ss = {k: v for k, v in gt_df.groupby(["seqid", "strand"], sort=False, observed=True)}
    pred_by_ss = {k: v for k, v in pred_df.groupby(["seqid", "strand"], sort=False, observed=True)}
    empty_df = gt_df.iloc[0:0]

    # Single scope pass for exon + structure metrics — _collect_scoped_transcript_intervals
    # called once per (df, scope) instead of three times.
    exon_metrics, exon_lenient_metrics, intron_chain_metrics, transcript_exact_metrics = (
        _compute_exon_and_structure_metrics(
            gt_df, pred_df, label_config, gt_feature_role_map, pred_feature_role_map
        )
    )

    return {
        "nucleotide": _compute_global_nucleotide_metrics(
            gt_by_ss,
            pred_by_ss,
            empty_df,
            label_config,
            gt_feature_role_map,
            pred_feature_role_map,
            transcript_types,
        ),
        "exon": exon_metrics,
        "exon_lenient": exon_lenient_metrics,
        "intron_chain": intron_chain_metrics,
        "transcript_exact": transcript_exact_metrics,
        "transcript": _compute_transcript_level_metrics(
            mappings,
            predictor_name,
        ),
        "gene": _compute_gene_level_metrics(
            gt_by_ss,
            pred_by_ss,
            mappings,
            predictor_name,
            transcript_types,
        ),
        "locus_isoform": _compute_locus_isoform_metrics(
            mappings,
            predictor_name,
            locus_matching_mode,
        ),
    }


# ---------------------------------------------------------------------------
# Nucleotide metrics — union-based, interval arithmetic (no array allocation)
# ---------------------------------------------------------------------------


def _compute_global_nucleotide_metrics(
    gt_by_ss: _SSIndex,
    pred_by_ss: _SSIndex,
    empty_df: pd.DataFrame,
    label_config: LabelConfig,
    gt_feature_role_map: FeatureRoleMap,
    pred_feature_role_map: FeatureRoleMap,
    transcript_types: list[str],
) -> dict:
    """Nucleotide precision/recall/F1 using union-of-exons per locus.

    Evaluation space: the union of all ref and pred transcript spans on the
    same seqid+strand, merged into non-overlapping regions.  Within each
    region TP/FP/FN are accumulated via interval intersection arithmetic —
    no per-base arrays are allocated.

    This matches gffcompare's methodology: each genomic base is counted once
    regardless of how many isoforms cover it.
    """
    results: dict[str, dict] = {}
    all_seqids = sorted({s for (s, _strand) in gt_by_ss} | {s for (s, _strand) in pred_by_ss})

    for scope in tqdm(list(label_config.available_scopes()), desc="nucleotide", unit="scope"):
        ref_scope_types = feature_types_for_scope(gt_feature_role_map, label_config, scope)
        pred_scope_types = feature_types_for_scope(pred_feature_role_map, label_config, scope)
        total_tp = total_fp = total_fn = 0

        for seqid in tqdm(all_seqids, desc=f"  nt/{scope.value}", leave=False, unit="chr"):
            for strand in ("+", "-"):
                gt_sub = gt_by_ss.get((seqid, strand), empty_df)
                pred_sub = pred_by_ss.get((seqid, strand), empty_df)
                ref_spans = _get_transcript_spans(gt_sub, transcript_types)
                pred_spans = _get_transcript_spans(pred_sub, transcript_types)

                all_spans = ref_spans + pred_spans
                if not all_spans:
                    continue

                # Extract scope feature coordinates once for this (seqid, strand).
                ref_starts, ref_ends = _scope_feature_intervals(gt_sub, ref_scope_types)
                pred_starts, pred_ends = _scope_feature_intervals(pred_sub, pred_scope_types)

                for region_start, region_end in _merge_intervals(all_spans):
                    ref_ivs = _clip_and_merge(ref_starts, ref_ends, region_start, region_end)
                    pred_ivs = _clip_and_merge(pred_starts, pred_ends, region_start, region_end)

                    ref_len = sum(e - s + 1 for s, e in ref_ivs)
                    pred_len = sum(e - s + 1 for s, e in pred_ivs)
                    tp = _intersection_length(ref_ivs, pred_ivs)

                    total_tp += tp
                    total_fp += pred_len - tp
                    total_fn += ref_len - tp

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        results[scope.value] = {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }

    return {"scopes": results}


# ---------------------------------------------------------------------------
# Exon + intron-chain + transcript-exact — single scope pass
# ---------------------------------------------------------------------------


def _compute_exon_and_structure_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label_config: LabelConfig,
    gt_feature_role_map: FeatureRoleMap,
    pred_feature_role_map: FeatureRoleMap,
) -> tuple[dict, dict, dict, dict]:
    """Exon (exact + lenient), intron-chain, and transcript-exact metrics in one scope pass.

    Previously three separate functions each called ``_collect_scoped_transcript_intervals``
    per scope per DataFrame; this iterates scopes once and calls it twice (gt, pred).

    Returns ``(exon_metrics, exon_lenient_metrics, intron_chain_metrics, transcript_exact_metrics)``.

    * **exon** — de-duplicated exact ``(seqid, strand, start, end)`` boundary matching.
      Each unique exon interval counted once regardless of isoform count.  Stricter than
      gffcompare (no terminal-exon leniency), avoiding TSS/TES ambiguity.

    * **exon_lenient** — terminal-exon outer boundaries wildcarded (gffcompare style).
      Only internal splice-site boundaries must match exactly.

    * **intron_chain** — multi-exon transcripts matched by complete intron chain.
      Single-exon transcripts have no chain and are excluded.

    * **transcript_exact** — transcripts matched by terminal-lenient exon structure.
      Both single- and multi-exon transcripts count.
    """
    exon_results: dict[str, dict] = {}
    exon_lenient_results: dict[str, dict] = {}
    chain_results: dict[str, dict] = {}
    struct_results: dict[str, dict] = {}

    for scope in tqdm(list(label_config.available_scopes()), desc="exon+structure", unit="scope"):
        gt_ivs, gt_orphans = _collect_scoped_transcript_intervals(
            gt_df, gt_feature_role_map, label_config, scope
        )
        pred_ivs, pred_orphans = _collect_scoped_transcript_intervals(
            pred_df, pred_feature_role_map, label_config, scope
        )

        # Exact exon keys
        ref_exon: set = set(gt_orphans)
        for intervals in gt_ivs.values():
            ref_exon.update(intervals)
        pred_exon: set = set(pred_orphans)
        for intervals in pred_ivs.values():
            pred_exon.update(intervals)

        n_matched = len(ref_exon & pred_exon)
        ref_total, pred_total = len(ref_exon), len(pred_exon)
        sn = n_matched / ref_total if ref_total > 0 else 0.0
        pr = n_matched / pred_total if pred_total > 0 else 0.0
        exon_results[scope.value] = {
            "ref_exon_count": ref_total,
            "ref_exon_matched": n_matched,
            "pred_exon_count": pred_total,
            "pred_exon_matched": n_matched,
            "sensitivity": sn,
            "precision": pr,
            "f1": _f1(sn, pr),
        }

        # Lenient exon keys
        ref_lenient = _lenient_keys_from_intervals(gt_ivs, gt_orphans)
        pred_lenient = _lenient_keys_from_intervals(pred_ivs, pred_orphans)
        n_matched = len(ref_lenient & pred_lenient)
        ref_total, pred_total = len(ref_lenient), len(pred_lenient)
        sn = n_matched / ref_total if ref_total > 0 else 0.0
        pr = n_matched / pred_total if pred_total > 0 else 0.0
        exon_lenient_results[scope.value] = {
            "ref_exon_count": ref_total,
            "ref_exon_matched": n_matched,
            "pred_exon_count": pred_total,
            "pred_exon_matched": n_matched,
            "sensitivity": sn,
            "precision": pr,
            "f1": _f1(sn, pr),
        }

        # Structure keys (intron chain + transcript exact)
        gt_struct = _structure_keys_from_intervals(gt_ivs)
        pred_struct = _structure_keys_from_intervals(pred_ivs)
        chain_results[scope.value] = _set_match_metrics(
            [c for _s, c in gt_struct if c],
            [c for _s, c in pred_struct if c],
            "chain",
        )
        struct_results[scope.value] = _set_match_metrics(
            [s for s, _c in gt_struct],
            [s for s, _c in pred_struct],
            "transcript",
        )

    return (
        {"scopes": exon_results},
        {"scopes": exon_lenient_results},
        {"scopes": chain_results},
        {"scopes": struct_results},
    )


def _lenient_keys_from_intervals(
    intervals_by_parent: dict[str, list[tuple[str, str, int, int]]],
    orphan_intervals: set[tuple[str, str, int, int]],
) -> set[tuple]:
    """Lenient exon key set from already-computed intervals."""
    keys: set[tuple] = set()
    for intervals in intervals_by_parent.values():
        n = len(intervals)
        for i, (seqid, strand, start, end) in enumerate(intervals):
            keys.add(_lenient_exon_key(i, n, seqid, strand, start, end))
    keys.update(orphan_intervals)
    return keys


def _structure_keys_from_intervals(
    intervals_by_parent: dict[str, list[tuple[str, str, int, int]]],
) -> list[tuple[frozenset, frozenset]]:
    """(structure_key, intron_chain_key) pairs from already-computed intervals."""
    keys: list[tuple[frozenset, frozenset]] = []
    for intervals in intervals_by_parent.values():
        n = len(intervals)
        lenient = frozenset(
            _lenient_exon_key(i, n, seqid, strand, start, end)
            for i, (seqid, strand, start, end) in enumerate(intervals)
        )
        chain = frozenset(
            (intervals[i][0], intervals[i][1], intervals[i][3] + 1, intervals[i + 1][2] - 1)
            for i in range(n - 1)
        )
        keys.append((lenient, chain))
    return keys


def _lenient_exon_key(
    i: int, n: int, seqid: str, strand: str, start: int, end: int
) -> tuple:
    """Canonical exon key with the outer boundary of terminal exons wildcarded.

    gffcompare's transcript-level '=' leniency: the start of the first exon and
    the end of the last exon become ``None`` (TSS/TES tolerance), while every
    internal splice boundary stays exact.  A single-exon transcript (``n == 1``)
    keeps both boundaries — it is both first and last, so without this guard it
    would wrongly wildcard its start.
    """
    if n == 1:
        return (seqid, strand, start, end)
    if i == 0:
        return (seqid, strand, None, end)
    if i == n - 1:
        return (seqid, strand, start, None)
    return (seqid, strand, start, end)


# ---------------------------------------------------------------------------
# Intron-chain / transcript-exact set-membership helper
# ---------------------------------------------------------------------------


def _set_match_metrics(gt_keys: list, pred_keys: list, label: str) -> dict:
    """Sensitivity/precision/F1 from set-membership matching of two key lists.

    A ref key is matched when it appears in the prediction set, and vice versa.
    *label* names the count fields (``ref_{label}_count`` etc.).
    """
    gt_total, pred_total = len(gt_keys), len(pred_keys)
    gt_set, pred_set = set(gt_keys), set(pred_keys)
    ref_matched = sum(1 for k in gt_keys if k in pred_set)
    pred_matched = sum(1 for k in pred_keys if k in gt_set)
    sensitivity = ref_matched / gt_total if gt_total > 0 else 0.0
    precision = pred_matched / pred_total if pred_total > 0 else 0.0
    return {
        f"ref_{label}_count": gt_total,
        f"ref_{label}_matched": ref_matched,
        f"pred_{label}_count": pred_total,
        f"pred_{label}_matched": pred_matched,
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": _f1(sensitivity, precision),
    }


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
        pred_hits = [m for m in mapping.matched_predictions if m.predictor_name == predictor_name]

        if mapping.is_unmatched_prediction:
            if pred_hits:
                pred_total += 1  # unmatched pred lowers precision
        else:
            ref_total += 1
            if pred_hits:
                ref_matched += 1
                pred_total += 1
                pred_matched += 1

    sensitivity = ref_matched / ref_total if ref_total > 0 else 0.0
    precision = pred_matched / pred_total if pred_total > 0 else 0.0

    return {
        "ref_transcript_count": ref_total,
        "ref_transcript_matched": ref_matched,
        "pred_transcript_count": pred_total,
        "pred_transcript_matched": pred_matched,
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": _f1(sensitivity, precision),
    }


# ---------------------------------------------------------------------------
# Gene / locus-level metrics
# ---------------------------------------------------------------------------


def _compute_gene_level_metrics(
    gt_by_ss: _SSIndex,
    pred_by_ss: _SSIndex,
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

    gt_locus_count, gt_locus_matched = _count_matched_loci(gt_by_ss, transcript_types, matched_gt_ids)
    pred_locus_count, pred_locus_matched = _count_matched_loci(pred_by_ss, transcript_types, matched_pred_ids)

    sensitivity = gt_locus_matched / gt_locus_count if gt_locus_count > 0 else 0.0
    precision = pred_locus_matched / pred_locus_count if pred_locus_count > 0 else 0.0

    return {
        "ref_locus_count": gt_locus_count,
        "ref_locus_matched": gt_locus_matched,
        "pred_locus_count": pred_locus_count,
        "pred_locus_matched": pred_locus_matched,
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": _f1(sensitivity, precision),
    }


def _compute_locus_isoform_metrics(
    mappings: list[TranscriptMapping],
    predictor_name: str,
    locus_matching_mode: LocusMatchingMode,
) -> dict:
    """Per-locus isoform recall for one predictor.

    Considers only the mapping entries this predictor participates in (via
    :func:`_include_mapping_for_predictor`), so each locus is counted exactly
    once per predictor regardless of how many *other* predictors matched it.  A
    GT isoform counts as *recovered* only on a serious match
    (``junction_f1 > 0``); a Case-C overlap-but-wrong entry (``junction_f1 == 0``)
    is a miss.

    In ``FULL_DISCOVERY`` every GT isoform is its own entry, so this is per-locus
    isoform recall (fraction of isoforms matched).  In ``BEST_PER_LOCUS`` each
    predictor owns one entry per locus, so it degenerates to locus recall.

    Returns
    -------
    dict with keys:
        locus_count          – number of GT loci evaluated
        ref_isoform_count    – total GT isoforms across all loci
        ref_isoform_matched  – GT isoforms recovered by a serious match
        recall               – ref_isoform_matched / ref_isoform_count (micro-avg)
        missed_per_locus     – list[int], missed isoform count per locus
    """
    relevant = [
        m
        for m in mappings
        if not m.is_unmatched_prediction
        and _include_mapping_for_predictor(m, predictor_name, locus_matching_mode)
    ]
    if not relevant:
        return {
            "locus_count": 0,
            "ref_isoform_count": 0,
            "ref_isoform_matched": 0,
            "recall": 0.0,
            "missed_per_locus": [],
        }

    matched_gt_ids = {
        m.gt_id
        for m in relevant
        if any(
            pm.predictor_name == predictor_name and pm.junction_f1 > 0
            for pm in m.matched_predictions
        )
    }

    groups: dict[tuple, list] = defaultdict(list)
    for m in relevant:
        groups[(m.seqid, m.strand)].append((m.gt_start, m.gt_end, m.gt_id))

    locus_count = 0
    total_gt = 0
    total_matched = 0
    missed_per_locus: list[int] = []

    for spans in groups.values():
        for locus_ids in ([tid for _, _, tid in g] for g in _sweep_cluster(spans, key=lambda x: (x[0], x[1]))):
            unique_ids = set(locus_ids)
            n_total = len(unique_ids)
            n_matched = sum(1 for gid in unique_ids if gid in matched_gt_ids)
            locus_count += 1
            total_gt += n_total
            total_matched += n_matched
            missed_per_locus.append(n_total - n_matched)

    recall = total_matched / total_gt if total_gt > 0 else 0.0
    return {
        "locus_count": locus_count,
        "ref_isoform_count": total_gt,
        "ref_isoform_matched": total_matched,
        "recall": recall,
        "missed_per_locus": missed_per_locus,
    }


def _count_matched_loci(
    by_ss: _SSIndex,
    transcript_types: list[str],
    matched_ids: set[str],
) -> tuple[int, int]:
    """Count total loci and matched loci in a pre-grouped GFF DataFrame.

    Returns
    -------
    tuple[int, int]
        ``(total_loci, matched_loci)``
    """
    locus_count = locus_matched = 0

    for (_seqid, strand), sub_df in by_ss.items():
        if strand not in ("+", "-"):
            continue
        spans_with_ids = _get_transcript_spans_with_ids(sub_df, transcript_types)
        if not spans_with_ids:
            continue
        for locus_ids in ([tid for _, _, tid in g] for g in _sweep_cluster(spans_with_ids, key=lambda x: (x[0], x[1]))):
            locus_count += 1
            if any(tid in matched_ids for tid in locus_ids):
                locus_matched += 1

    return locus_count, locus_matched


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _get_transcript_spans(
    sub_df: pd.DataFrame,
    transcript_types: list[str],
) -> list[tuple[int, int]]:
    """Return ``(start, end)`` for all transcripts in a (seqid, strand) slice."""
    return [(s, e) for s, e, _ in _get_transcript_spans_with_ids(sub_df, transcript_types)]


def _get_transcript_spans_with_ids(
    sub_df: pd.DataFrame,
    transcript_types: list[str],
) -> list[tuple[int, int, str]]:
    """Return ``(start, end, gff_id)`` for transcripts in a (seqid, strand) slice."""
    mask = (
        sub_df["type"].isin(transcript_types)
        & sub_df["start"].notna()
        & sub_df["end"].notna()
        & sub_df["gff_id"].notna()
    )
    rows = sub_df[mask]
    return list(
        zip(
            rows["start"].astype(int),
            rows["end"].astype(int),
            rows["gff_id"].astype(str),
        )
    )


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


def _collect_scoped_transcript_intervals(
    df: pd.DataFrame,
    feature_role_map: FeatureRoleMap,
    label_config: LabelConfig,
    scope: BenchmarkScope | str,
) -> tuple[dict[str, list[tuple[str, str, int, int]]], set[tuple[str, str, int, int]]]:
    """Return merged scope intervals grouped by transcript parent."""
    scope_feature_types = feature_types_for_scope(feature_role_map, label_config, scope)
    if not scope_feature_types:
        return {}, set()

    mask = (
        df["type"].isin(scope_feature_types)
        & df["seqid"].notna()
        & df["strand"].notna()
        & df["start"].notna()
        & df["end"].notna()
    )
    scoped_rows = df[mask]
    if scoped_rows.empty:
        return {}, set()

    orphan_intervals: set[tuple[str, str, int, int]] = set()
    no_parent = scoped_rows[scoped_rows["parent"].isna()]
    for row in no_parent.itertuples(index=False):
        orphan_intervals.add((str(row.seqid), str(row.strand), int(row.start), int(row.end)))

    # Group parented rows by parent in one pass over plain arrays, then merge —
    # avoids paying pandas groupby + a per-group ``sort_values`` for every one of
    # tens of thousands of transcripts (the dominant global-metrics cost).
    with_parent = scoped_rows[scoped_rows["parent"].notna()]
    rows_by_parent: dict[str, list[tuple[str, str, int, int]]] = defaultdict(list)
    for seqid, strand, start, end, parent in zip(
        with_parent["seqid"].to_numpy(),
        with_parent["strand"].to_numpy(),
        with_parent["start"].to_numpy(),
        with_parent["end"].to_numpy(),
        with_parent["parent"].to_numpy(),
    ):
        rows_by_parent[str(parent)].append((str(seqid), str(strand), int(start), int(end)))

    intervals_by_parent: dict[str, list[tuple[str, str, int, int]]] = {}
    for parent_id, recs in rows_by_parent.items():
        recs.sort(key=lambda r: r[2])
        intervals_by_parent[parent_id] = _merge_sorted_intervals(recs)

    return intervals_by_parent, orphan_intervals


def _merge_sorted_intervals(
    recs: list[tuple[str, str, int, int]],
) -> list[tuple[str, str, int, int]]:
    """Merge start-sorted ``(seqid, strand, start, end)`` rows into intervals.

    Adjacent or overlapping rows (gap <= 1) are fused, taking the maximum end.
    *recs* must already be sorted by start.
    """
    merged: list[tuple[str, str, int, int]] = []
    current_seqid: str | None = None
    current_strand: str | None = None
    current_start: int | None = None
    current_end: int | None = None

    for seqid, strand, start, end in recs:
        if current_start is None:
            current_seqid = seqid
            current_strand = strand
            current_start = start
            current_end = end
            continue
        if start <= current_end + 1:
            current_end = max(current_end, end)
            continue
        merged.append((current_seqid, current_strand, current_start, current_end))
        current_seqid = seqid
        current_strand = strand
        current_start = start
        current_end = end

    if current_start is not None:
        merged.append((current_seqid, current_strand, current_start, current_end))
    return merged


def _scope_feature_intervals(
    sub_df: pd.DataFrame,
    scope_feature_types: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(starts, ends)`` int arrays for scope features in a (seqid,strand) slice.

    Extracted **once** per (seqid, strand, scope) so the per-region union build
    works on plain numpy arrays instead of re-masking the DataFrame each time.
    """
    rows = sub_df[
        sub_df["type"].isin(scope_feature_types)
        & sub_df["start"].notna()
        & sub_df["end"].notna()
    ]
    return rows["start"].to_numpy(dtype=np.int64), rows["end"].to_numpy(dtype=np.int64)


def _clip_and_merge(
    starts: np.ndarray,
    ends: np.ndarray,
    region_start: int,
    region_end: int,
) -> list[tuple[int, int]]:
    """Filter features overlapping [region_start, region_end], clip, and merge.

    Returns a sorted non-overlapping list of ``(start, end)`` pairs clipped to
    the region bounds.  Empty list when no features overlap.
    """
    overlapping = (starts <= region_end) & (ends >= region_start)
    if not overlapping.any():
        return []
    clipped = sorted(
        (max(int(s), region_start), min(int(e), region_end))
        for s, e in zip(starts[overlapping], ends[overlapping])
    )
    merged: list[tuple[int, int]] = [clipped[0]]
    for s, e in clipped[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _intersection_length(
    a: list[tuple[int, int]],
    b: list[tuple[int, int]],
) -> int:
    """Total length of intersection of two sorted non-overlapping interval lists."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo <= hi:
            total += hi - lo + 1
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; 0.0 when both are zero."""
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0
