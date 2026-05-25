"""Core evaluation orchestration for the DNA segmentation benchmark.

Public entry points (:func:`benchmark_gt_vs_pred_single` and
:func:`benchmark_gt_vs_pred_multiple`) compare ground-truth nucleotide-level
annotations with predictions and dispatch to the per-metric modules:

* **INDEL** — :mod:`indel_metrics`
* **REGION_DISCOVERY / BOUNDARY_EXACTNESS** — :mod:`section_metrics`
* **NUCLEOTIDE_CLASSIFICATION** — local confusion counts
* **FRAMESHIFT** — :mod:`frame_shift`
* **STRUCTURAL_COHERENCE** — :mod:`chain_comparison`, :mod:`structure`,
  :mod:`splice_sites`
* **DIAGNOSTIC_DEPTH** — :mod:`structural_summary`

Input preprocessing (intron inference, mask splitting) lives in
:mod:`preprocessing`; cross-sequence accumulation and reduction live in
:mod:`accumulators`.  All functions accept a :class:`LabelConfig` that maps
integer tokens to names and declares semantic roles (background, coding, …).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator
from copy import deepcopy
from typing import Optional
from typing import Iterable
import numpy as np
from tqdm import tqdm

from .accumulators import BenchmarkAccumulator
from .chain_comparison import (
    _compute_intron_chain_metrics,
    _compute_chain_metrics,
    _compute_boundary_shift_metrics,
    _compute_per_transcript_exon_soft_metrics,
)
from .frame_shift import _get_frame_shift_metrics
from .indel_metrics import _eval_indel
from .preprocessing import _infer_introns_from_coding_gaps, _iter_unmasked_spans
from .section_metrics import _eval_sections
from .state_transitions import _compute_state_change_errors
from .statistics import Counts
from .structure import extract_structure
from .structural_summary import _compute_structural_summary
from .transcript_classification import _classify_transcript_match
from .utils import get_contiguous_groups
from .splice_sites import eval_splice_site_junctions
from ..label_definition import LabelConfig, EvalMetrics, _DEFAULT_METRICS

# ---------------------------------------------------------------------------
# Helpers — which groups need section overlap to be computed
# ---------------------------------------------------------------------------

_SECTION_DEPENDENT_GROUPS = frozenset(
    {
        EvalMetrics.REGION_DISCOVERY,
        EvalMetrics.BOUNDARY_EXACTNESS,
    }
)


def _needs_section_analysis(metrics: Iterable[EvalMetrics]) -> bool:
    """Return ``True`` if any requested metric needs section-overlap data."""
    return bool(_SECTION_DEPENDENT_GROUPS & set(metrics))


# ---------------------------------------------------------------------------
# Single-sequence benchmark
# ---------------------------------------------------------------------------


def benchmark_gt_vs_pred_single(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_config: LabelConfig,
    metrics: Optional[list[EvalMetrics]] = None,
    mask_labels: Optional[np.ndarray] = None,
    infer_introns: bool = False,
) -> dict[str, dict]:
    """Compare a single ground-truth sequence against a single prediction.

    Parameters
    ----------
    gt_labels : np.ndarray
        1-D array of ground-truth nucleotide-level integer tokens.
    pred_labels : np.ndarray
        1-D array of predicted integer tokens (same length as *gt_labels*).
    label_config : LabelConfig
        Maps integer tokens to names and declares semantic roles.
    metrics : list[EvalMetrics] | None
        Which metric groups to compute.  Defaults to
        ``[REGION_DISCOVERY, BOUNDARY_EXACTNESS, NUCLEOTIDE_CLASSIFICATION]``.
    mask_labels : np.ndarray | None
        Optional boolean mask (True = exclude). Must match length of GT.
    infer_introns : bool
        If ``True``, background gaps between adjacent coding segments are
        relabelled as introns before any metric is computed.

    Returns
    -------
    dict
        Dict keyed directly by metric group name plus the transition
        analysis keys ``"transition_failures"`` and
        ``"false_transitions"``. When ``STRUCTURAL_COHERENCE`` is
        requested, the ``STRUCTURAL_COHERENCE`` entry contains:

        * ``intron_chain`` / ``intron_chain_subset`` / ``intron_chain_superset``
          — binary TP/FP/FN comparing the intron segment boundary sets,
          aggregated to corpus precision/recall across sequences.
        * ``exon_chain`` / ``exon_chain_subset`` / ``exon_chain_superset``
          — same set semantics applied to coding (exon) segments.
          Subset: pred ⊆ GT (all pred exons are real, may miss some GT).
          Superset: pred ⊇ GT (every GT exon found, may have extras).
        * ``exon_recall_per_transcript`` — float in [0, 1]: fraction of
          GT exons whose ``(start, end)`` was recovered exactly.
        * ``hallucinated_exon_count_per_transcript`` — int ≥ 0: number
          of predicted exons whose ``(start, end)`` is absent from GT.
        * ``segment_count_delta`` — ``pred_count - gt_count``
          (positive = over-segmentation).
        * ``boundary_shift_count`` / ``boundary_shift_total`` — number
          of shifted boundary positions and their summed absolute offset
          in bp across transcripts where GT and pred segment counts match.
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS
    metrics = frozenset(metrics)

    if infer_introns:
        gt_labels = _infer_introns_from_coding_gaps(gt_labels, label_config)
        pred_labels = _infer_introns_from_coding_gaps(pred_labels, label_config)

    if mask_labels is None:
        # A single unmasked span — return the raw per-sequence fragment.
        return _benchmark_chunk(gt_labels, pred_labels, label_config, metrics)

    # Masked: evaluate each kept span and combine into the un-summarised form.
    accumulator = BenchmarkAccumulator()
    for s, e in _iter_unmasked_spans(mask_labels):
        accumulator.add(_benchmark_chunk(gt_labels[s:e], pred_labels[s:e], label_config, metrics))
    return accumulator.merged()


def _iter_sequence_fragments(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_config: LabelConfig,
    metrics: frozenset,
    mask_labels: Optional[np.ndarray],
    infer_introns: bool,
) -> Iterator[dict]:
    """Yield the raw per-chunk fragment(s) for one sequence.

    One fragment for an unmasked sequence; one per kept span when masked.
    Intron inference is applied to the whole sequence before chunking.
    """
    if infer_introns:
        gt_labels = _infer_introns_from_coding_gaps(gt_labels, label_config)
        pred_labels = _infer_introns_from_coding_gaps(pred_labels, label_config)

    if mask_labels is None:
        yield _benchmark_chunk(gt_labels, pred_labels, label_config, metrics)
    else:
        for s, e in _iter_unmasked_spans(mask_labels):
            yield _benchmark_chunk(gt_labels[s:e], pred_labels[s:e], label_config, metrics)


def _benchmark_chunk(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_config: LabelConfig,
    metrics: frozenset[EvalMetrics],
) -> dict[str, dict]:
    """Evaluate one fully-unmasked GT/pred span.

    ``metrics`` must already be normalised to a frozenset; intron inference
    and mask splitting are handled by :func:`benchmark_gt_vs_pred_single`.
    """
    coding = label_config.coding_label

    # Row 0 = GT, Row 1 = prediction  (NO sentinel padding here)
    arr = np.stack((gt_labels, pred_labels), axis=0)

    metric_results: dict[str, dict] = {}
    metric_results.update(_eval_transitions(arr, label_config))

    grouped_insertions = get_contiguous_groups(np.where((arr[0] != coding) & (arr[1] == coding))[0])
    grouped_deletions = get_contiguous_groups(np.where((arr[0] == coding) & (arr[1] != coding))[0])
    grouped_gt_sections = get_contiguous_groups(np.where(arr[0] == coding)[0])
    grouped_pred_sections = get_contiguous_groups(np.where(arr[1] == coding)[0])

    if EvalMetrics.INDEL in metrics:
        metric_results[EvalMetrics.INDEL.name] = _eval_indel(
            grouped_insertions, grouped_deletions, gt_labels, pred_labels, label_config
        )

    if _needs_section_analysis(metrics):
        section_data, boundary_data = _eval_sections(grouped_gt_sections, grouped_pred_sections, metrics)
        if section_data:
            metric_results[EvalMetrics.REGION_DISCOVERY.name]=section_data
        if boundary_data:
            metric_results[EvalMetrics.BOUNDARY_EXACTNESS.name]=boundary_data

    if EvalMetrics.NUCLEOTIDE_CLASSIFICATION in metrics:
        metric_results[EvalMetrics.NUCLEOTIDE_CLASSIFICATION.name] = {
            "nucleotide": _compute_nucleotide_level_confusion(gt_labels, pred_labels, coding),
        }

    if EvalMetrics.FRAMESHIFT in metrics:
        metric_results[EvalMetrics.FRAMESHIFT.name] = _eval_frameshift(gt_labels, pred_labels, label_config)

    if EvalMetrics.STRUCTURAL_COHERENCE in metrics:
        metric_results[EvalMetrics.STRUCTURAL_COHERENCE.name] = _eval_structural(gt_labels, pred_labels, label_config)

    if EvalMetrics.DIAGNOSTIC_DEPTH in metrics:
        metric_results[EvalMetrics.DIAGNOSTIC_DEPTH.name] = _compute_structural_summary(
            grouped_gt_sections, grouped_pred_sections
        )

    return metric_results


def _eval_transitions(arr: np.ndarray, label_config: LabelConfig) -> dict:
    """Return the always-on state-transition diagnostic fragments."""
    transition_analysis = _compute_state_change_errors(gt_pred_arr=arr, label_config=label_config)
    return {
        "transition_failures": transition_analysis.gt_transition_matrices,
        "false_transitions": {
            "late_catchup": transition_analysis.late_catchup_matrices,
            "premature": transition_analysis.premature_matrices,
            "spurious": transition_analysis.spurious_matrices,
            "stable_position_counts": transition_analysis.stable_position_counts,
        },
    }


def _eval_frameshift(gt_labels: np.ndarray, pred_labels: np.ndarray, label_config: LabelConfig) -> dict:
    """Per-position reading-frame deviation between GT and predicted exons."""
    if label_config.coding_label is None:
        raise ValueError(
            "FRAMESHIFT metric requested but LabelConfig.coding_label is not "
            "set.  Provide a coding_label when constructing your LabelConfig."
        )
    return _get_frame_shift_metrics(
        gt_labels=gt_labels,
        pred_labels=pred_labels,
        coding_value=label_config.coding_label,
    )


def _eval_structural(gt_labels: np.ndarray, pred_labels: np.ndarray, label_config: LabelConfig) -> dict:
    """Build STRUCTURAL_COHERENCE (and, when configured, splice_sites)."""
    gt_struct = extract_structure(gt_labels, label_config)
    pred_struct = extract_structure(pred_labels, label_config)

    chain_metric_results: dict = {}
    chain_metric_results.update(_compute_intron_chain_metrics(gt_struct, pred_struct, label_config))
    chain_metric_results.update(_compute_per_transcript_exon_soft_metrics(gt_struct, pred_struct, label_config))

    gt_coding = gt_struct.filter_by_label(label_config.coding_label)
    pred_coding = pred_struct.filter_by_label(label_config.coding_label)

    if len(gt_coding) > 0:
        chain_metric_results.update(_compute_chain_metrics(gt_struct, pred_struct, label_config.coding_label, "exon_chain"))
        chain_metric_results.update(_compute_boundary_shift_metrics(gt_struct, pred_struct, label_config.coding_label))
        chain_metric_results["segment_count_delta"] = len(pred_coding) - len(gt_coding)

        match_cls = _classify_transcript_match(gt_struct, pred_struct, label_config.coding_label)
        if match_cls is not None:
            chain_metric_results["transcript_match_class"] = match_cls.value

    structural_coherance_results = {
        "chain_metric_results":chain_metric_results,
    }
    if (
        label_config.intron_label is not None
        and label_config.splice_donor_label is not None
        and label_config.splice_acceptor_label is not None
    ):
        splice_confusion = eval_splice_site_junctions(gt_struct, pred_struct, label_config)
        splice_site_result = dataclasses.asdict(splice_confusion)
        structural_coherance_results["splice_site_results"] = splice_site_result

    return structural_coherance_results


def _compute_nucleotide_level_confusion(
    gt_labels: np.ndarray, pred_labels: np.ndarray, class_value: int
) -> Counts:
    """Calculate granular base accuracy as a confusion bundle."""
    gt_pos = gt_labels == class_value
    pred_pos = pred_labels == class_value

    return Counts(
        tn=int(np.count_nonzero(~gt_pos & ~pred_pos)),
        fp=int(np.count_nonzero(~gt_pos & pred_pos)),
        fn=int(np.count_nonzero(gt_pos & ~pred_pos)),
        tp=int(np.count_nonzero(gt_pos & pred_pos)),
    )


# ---------------------------------------------------------------------------
# Multi-sequence benchmark
# ---------------------------------------------------------------------------


def benchmark_gt_vs_pred_multiple(
    gt_labels: list[np.ndarray],
    pred_labels: list[np.ndarray],
    label_config: LabelConfig,
    metrics: Optional[list[EvalMetrics]] = None,
    return_individual_results: bool = False,
    mask_labels: Optional[list[np.ndarray]] = None,
    infer_introns: bool = False,
) -> dict | list[dict]:
    """Run :func:`benchmark_gt_vs_pred_single` over paired GT/pred lists.

    Parameters
    ----------
    gt_labels, pred_labels : list[np.ndarray]
        Equally-sized lists of 1-D integer token arrays.
    label_config : LabelConfig
        Token-to-name mapping and semantic roles.
    metrics : list[EvalMetrics] | None
        Metric groups to compute.
    return_individual_results : bool
        If ``True``, return per-sequence results as a list instead of
        aggregating.
    mask_labels : list[np.ndarray] | None
        Optional boolean masks (True = exclude). Must match length of GT.
    infer_introns : bool
        If ``True``, background gaps between adjacent coding segments are
        relabelled as introns before each sequence is evaluated.

    Returns
    -------
    dict | list[dict]
        Aggregated (default) or per-sequence results.
    """
    if len(gt_labels) != len(pred_labels):
        raise ValueError(f"GT and prediction lists must have equal length, got {len(gt_labels)} vs {len(pred_labels)}.")
    if mask_labels is not None and len(mask_labels) != len(gt_labels):
        raise ValueError(f"Mask list length ({len(mask_labels)}) must match GT list length ({len(gt_labels)}).")

    metrics = deepcopy(metrics) if metrics is not None else list(_DEFAULT_METRICS)

    if EvalMetrics.FRAMESHIFT in metrics:
        warnings.warn(
            "The FRAMESHIFT metric should only be used when you are certain "
            "that the transcript contains all annotated exons.  Otherwise "
            "the results will be misleading.",
            stacklevel=2,
        )

    if return_individual_results:
        return [
            benchmark_gt_vs_pred_single(
                gt_labels=gt_labels[i],
                pred_labels=pred_labels[i],
                label_config=label_config,
                metrics=metrics,
                mask_labels=mask_labels[i] if mask_labels is not None else None,
                infer_introns=infer_introns,
            )
            for i in tqdm(range(len(gt_labels)), desc="Running benchmark")
        ]

    metrics_set = frozenset(metrics)
    accumulator = BenchmarkAccumulator()
    for i in tqdm(range(len(gt_labels)), desc="Running benchmark"):
        for fragment in _iter_sequence_fragments(
            gt_labels[i],
            pred_labels[i],
            label_config,
            metrics_set,
            mask_labels[i] if mask_labels is not None else None,
            infer_introns,
        ):
            accumulator.add(fragment)

    return accumulator.summarise()
