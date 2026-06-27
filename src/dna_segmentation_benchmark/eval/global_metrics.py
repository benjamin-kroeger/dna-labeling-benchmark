"""Global annotation-level metrics for the DNA segmentation benchmark.

Computes metrics over the full set of reference and predicted transcripts,
comparable to gffcompare's nucleotide and exon sensitivity/precision.

Unlike the per-transcript metrics (which evaluate matched pairs in isolation),
global metrics aggregate over *all* transcripts — including unmatched ones —
so false-positive predictions and missed reference transcripts both contribute
to the final numbers.

Six metric groups are computed, each answering a distinct question:

* ``nucleotide``  — *"At the base level, how accurate is the exon coverage?"*
  Union-based: each genomic base is counted once regardless of isoform count.
  Equivalent to gffcompare's nucleotide sensitivity/precision.

* ``exon``  — *"How many exons are exactly reconstructed?"*
  De-duplicated counting: each unique ``(seqid, strand, start, end)``
  interval is counted **once across all transcripts**, regardless of how
  many isoforms share it.  This diverges from gffcompare's per-isoform
  counting; see :func:`_compute_global_exon_metrics` for the rationale.

* ``exon_lenient``  — *"Exon recovery with TSS/TES boundary leniency."*
  Terminal-exon outer boundaries are not required to match (gffcompare
  default ``=``); only internal splice-site boundaries must be exact.
  See :func:`_compute_global_exon_lenient_metrics`.

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
        Label configuration.  ``coding_label`` must be set for nucleotide
        metrics; if it is ``None`` the nucleotide section is returned empty.
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

    # Intron-chain and whole-transcript metrics share the same per-transcript
    # structure keys, so compute them in one pass over each DataFrame.
    intron_chain_metrics, transcript_exact_metrics = _compute_global_structure_metrics(
        gt_df, pred_df, label_config, gt_feature_role_map, pred_feature_role_map
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
        "exon": _compute_global_exon_metrics(
            gt_df,
            pred_df,
            label_config,
            gt_feature_role_map,
            pred_feature_role_map,
        ),
        "exon_lenient": _compute_global_exon_lenient_metrics(
            gt_df,
            pred_df,
            label_config,
            gt_feature_role_map,
            pred_feature_role_map,
        ),
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
# Nucleotide metrics — union-based
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
    region two binary arrays are built — one marking ref-exonic bases, one
    marking pred-exonic bases — and TP/FP/FN are accumulated.

    This matches gffcompare's methodology: each genomic base is counted once
    regardless of how many isoforms cover it.
    """
    results: dict[str, dict] = {}
    bg_val = label_config.background_label
    all_seqids = {s for (s, _strand) in gt_by_ss} | {s for (s, _strand) in pred_by_ss}

    for scope in label_config.available_scopes():
        scope_label = min(label_config.scope_tokens(scope))
        ref_scope_types = feature_types_for_scope(gt_feature_role_map, label_config, scope)
        pred_scope_types = feature_types_for_scope(pred_feature_role_map, label_config, scope)
        total_tp = total_fp = total_fn = 0

        for seqid in sorted(all_seqids):
            for strand in ("+", "-"):
                gt_sub = gt_by_ss.get((seqid, strand), empty_df)
                pred_sub = pred_by_ss.get((seqid, strand), empty_df)
                ref_spans = _get_transcript_spans(gt_sub, transcript_types)
                pred_spans = _get_transcript_spans(pred_sub, transcript_types)

                all_spans = ref_spans + pred_spans
                if not all_spans:
                    continue

                # Extract scope feature coordinates once for this (seqid, strand);
                # the per-region union build then only does numpy selection.
                ref_starts, ref_ends = _scope_feature_intervals(gt_sub, ref_scope_types)
                pred_starts, pred_ends = _scope_feature_intervals(pred_sub, pred_scope_types)

                for region_start, region_end in _merge_intervals(all_spans):
                    length = region_end - region_start + 1

                    ref_arr = _build_scope_union_array(
                        ref_starts, ref_ends, region_start, length, scope_label, bg_val
                    )
                    pred_arr = _build_scope_union_array(
                        pred_starts, pred_ends, region_start, length, scope_label, bg_val
                    )

                    ref_exonic = ref_arr == scope_label
                    pred_exonic = pred_arr == scope_label
                    total_tp += int(np.sum(ref_exonic & pred_exonic))
                    total_fp += int(np.sum(~ref_exonic & pred_exonic))
                    total_fn += int(np.sum(ref_exonic & ~pred_exonic))

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
# Exon metrics — de-duplicated exact boundary matching
# ---------------------------------------------------------------------------


def _compute_global_exon_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label_config: LabelConfig,
    gt_feature_role_map: FeatureRoleMap,
    pred_feature_role_map: FeatureRoleMap,
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
    results: dict[str, dict] = {}
    for scope in label_config.available_scopes():
        ref_exon_keys = _collect_scoped_exon_keys(gt_df, gt_feature_role_map, label_config, scope)
        pred_exon_keys = _collect_scoped_exon_keys(pred_df, pred_feature_role_map, label_config, scope)

        n_matched = len(ref_exon_keys & pred_exon_keys)
        ref_total = len(ref_exon_keys)
        pred_total = len(pred_exon_keys)

        sensitivity = n_matched / ref_total if ref_total > 0 else 0.0
        precision = n_matched / pred_total if pred_total > 0 else 0.0
        results[scope.value] = {
            "ref_exon_count": ref_total,
            "ref_exon_matched": n_matched,
            "pred_exon_count": pred_total,
            "pred_exon_matched": n_matched,
            "sensitivity": sensitivity,
            "precision": precision,
            "f1": _f1(sensitivity, precision),
        }

    return {"scopes": results}


def _compute_global_exon_lenient_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label_config: LabelConfig,
    gt_feature_role_map: FeatureRoleMap,
    pred_feature_role_map: FeatureRoleMap,
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
    results: dict[str, dict] = {}
    for scope in label_config.available_scopes():
        ref_keys = _collect_scoped_exon_keys_lenient(gt_df, gt_feature_role_map, label_config, scope)
        pred_keys = _collect_scoped_exon_keys_lenient(pred_df, pred_feature_role_map, label_config, scope)

        n_matched = len(ref_keys & pred_keys)
        ref_total = len(ref_keys)
        pred_total = len(pred_keys)

        sensitivity = n_matched / ref_total if ref_total > 0 else 0.0
        precision = n_matched / pred_total if pred_total > 0 else 0.0
        results[scope.value] = {
            "ref_exon_count": ref_total,
            "ref_exon_matched": n_matched,
            "pred_exon_count": pred_total,
            "pred_exon_matched": n_matched,
            "sensitivity": sensitivity,
            "precision": precision,
            "f1": _f1(sensitivity, precision),
        }

    return {"scopes": results}


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


def _collect_scoped_exon_keys_lenient(
    df: pd.DataFrame,
    feature_role_map: FeatureRoleMap,
    label_config: LabelConfig,
    scope: BenchmarkScope | str,
) -> set[tuple]:
    """Return lenient canonical exon keys for one benchmark scope."""
    intervals_by_parent, orphan_intervals = _collect_scoped_transcript_intervals(
        df,
        feature_role_map,
        label_config,
        scope,
    )
    keys: set[tuple] = set()

    for intervals in intervals_by_parent.values():
        n = len(intervals)
        for i, (seqid, strand, start, end) in enumerate(intervals):
            keys.add(_lenient_exon_key(i, n, seqid, strand, start, end))

    keys.update(orphan_intervals)
    return keys


def _collect_scoped_exon_keys(
    df: pd.DataFrame,
    feature_role_map: FeatureRoleMap,
    label_config: LabelConfig,
    scope: BenchmarkScope | str,
) -> set[tuple[str, str, int, int]]:
    """Return unique merged exon intervals for one benchmark scope."""
    intervals_by_parent, orphan_intervals = _collect_scoped_transcript_intervals(
        df,
        feature_role_map,
        label_config,
        scope,
    )
    keys: set[tuple[str, str, int, int]] = set(orphan_intervals)
    for intervals in intervals_by_parent.values():
        keys.update(intervals)
    return keys


# ---------------------------------------------------------------------------
# Intron-chain and whole-transcript metrics — gffcompare structure parity
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


def _compute_global_structure_metrics(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    label_config: LabelConfig,
    gt_feature_role_map: FeatureRoleMap,
    pred_feature_role_map: FeatureRoleMap,
) -> tuple[dict, dict]:
    """Intron-chain and whole-transcript Sn/Sp — gffcompare structure parity.

    Returns ``(intron_chain, transcript_exact)``, each ``{"scopes": {...}}``.
    Both are derived from the per-transcript ``(structure_key, intron_chain_key)``
    pairs of :func:`_transcript_structure_keys`, computed once per DataFrame per
    scope and shared between the two metrics.

    * **intron_chain** — a multi-exon transcript is matched when its complete
      intron chain equals some opposite-side chain (single-exon transcripts have
      no chain and are excluded), exactly as gffcompare's intron-chain row.
    * **transcript_exact** — a transcript is matched when its terminal-lenient
      exon structure (internal splice boundaries exact, outer boundary of the
      first/last exon wildcarded) is reproduced; single-exon transcripts must
      match both boundaries.  Both single- and multi-exon transcripts count.

    Matching is coordinate-keyed and mapping-free, so both results are
    independent of the locus-matching mode.
    """
    chain_results: dict[str, dict] = {}
    struct_results: dict[str, dict] = {}
    for scope in label_config.available_scopes():
        gt_keys = _transcript_structure_keys(gt_df, gt_feature_role_map, label_config, scope)
        pred_keys = _transcript_structure_keys(pred_df, pred_feature_role_map, label_config, scope)
        chain_results[scope.value] = _set_match_metrics(
            [c for _s, c in gt_keys if c],
            [c for _s, c in pred_keys if c],
            "chain",
        )
        struct_results[scope.value] = _set_match_metrics(
            [s for s, _c in gt_keys],
            [s for s, _c in pred_keys],
            "transcript",
        )
    return {"scopes": chain_results}, {"scopes": struct_results}


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
        for locus_ids in _cluster_into_loci(spans):
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
        for locus_ids in _cluster_into_loci(spans_with_ids):
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
    mask = (
        sub_df["type"].isin(transcript_types)
        & sub_df["start"].notna()
        & sub_df["end"].notna()
    )
    rows = sub_df[mask]
    return list(zip(rows["start"].astype(int), rows["end"].astype(int)))


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


def _transcript_structure_keys(
    df: pd.DataFrame,
    feature_role_map: FeatureRoleMap,
    label_config: LabelConfig,
    scope: BenchmarkScope | str,
) -> list[tuple[frozenset, frozenset]]:
    """Per-transcript ``(structure_key, intron_chain_key)`` for one scope.

    ``structure_key`` is the frozenset of terminal-lenient exon keys: the outer
    boundary of the first exon (its ``start``) and the last exon (its ``end``)
    is wildcarded to ``None`` while every internal splice boundary is kept exact
    (single-exon transcripts keep both boundaries).  This is the same leniency
    gffcompare applies at the transcript level, and it is necessary here because
    the reference is frequently CDS-only while predictions carry UTR — without
    it, a correct prediction whose terminal exons merely extend into UTR would
    never match the reference's coding terminus.

    ``intron_chain_key`` is the frozenset of introns, each the gap
    ``(seqid, strand, prev_end + 1, next_start - 1)`` between consecutive merged
    intervals; it is empty for single-exon transcripts.

    Parent-less scope rows are not transcripts (no chain can be formed) and are
    skipped.  Reuses :func:`_collect_scoped_transcript_intervals`, so the scope
    filtering and exon merging match the exon metrics exactly.
    """
    intervals_by_parent, _orphans = _collect_scoped_transcript_intervals(
        df, feature_role_map, label_config, scope
    )
    keys: list[tuple[frozenset, frozenset]] = []
    for intervals in intervals_by_parent.values():
        n = len(intervals)
        lenient: set[tuple] = {
            _lenient_exon_key(i, n, seqid, strand, start, end)
            for i, (seqid, strand, start, end) in enumerate(intervals)
        }
        chain = frozenset(
            (intervals[i][0], intervals[i][1], intervals[i][3] + 1, intervals[i + 1][2] - 1)
            for i in range(n - 1)
        )
        keys.append((frozenset(lenient), chain))
    return keys


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


def _build_scope_union_array(
    feature_starts: np.ndarray,
    feature_ends: np.ndarray,
    region_start: int,
    array_length: int,
    scope_label: int,
    bg_val: int,
) -> np.ndarray:
    """Build a union array for one scope in one genomic region.

    *feature_starts*/*feature_ends* are the scope feature coordinates for the
    whole (seqid, strand) slice (from :func:`_scope_feature_intervals`); only
    those overlapping the region are painted.
    """
    arr = np.full(array_length, bg_val, dtype=np.int32)
    if feature_starts.size == 0:
        return arr

    region_end = region_start + array_length - 1
    overlapping = (feature_starts <= region_end) & (feature_ends >= region_start)
    for feat_start, feat_end in zip(feature_starts[overlapping], feature_ends[overlapping]):
        local_start = max(0, int(feat_start) - region_start)
        local_end = min(array_length, int(feat_end) - region_start + 1)
        if local_start < local_end:
            arr[local_start:local_end] = scope_label

    return arr


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _f1(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; 0.0 when both are zero."""
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0
