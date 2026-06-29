"""Core evaluation orchestration for the DNA segmentation benchmark.

Public entry points (:func:`benchmark_gt_vs_pred_single` and
:func:`benchmark_gt_vs_pred_multiple`) compare ground-truth nucleotide-level
annotations with predictions and dispatch to the per-metric modules:

* **INDEL** — :mod:`indel_metrics`
* **REGION_DISCOVERY / BOUNDARY_EXACTNESS** — :mod:`section_metrics`
* **NUCLEOTIDE_CLASSIFICATION** — local confusion counts
* **PHASE_DRIFT** — :mod:`frame_shift`
* **STRUCTURAL_COHERENCE** — :mod:`chain_comparison`, :mod:`structure`,
  :mod:`splice_sites`
* **DIAGNOSTIC_DEPTH** — :mod:`structural_summary`

The state-transition diagnostics (``transition_failures`` /
``false_transitions``) are opt-in via ``EvalMetrics.STATE_TRANSITIONS``.  They
remain in the default metric set (``_DEFAULT_METRICS``) so the transition-matrix
plots that frame every other metric still render by default; pass an explicit
``metrics`` list without ``STATE_TRANSITIONS`` to skip the pass.  Every metric
group is opt-in via ``metrics``.

Input preprocessing (intron inference, mask splitting) lives in
:mod:`preprocessing`; cross-sequence accumulation and reduction live in
:mod:`accumulators`.  All functions accept a :class:`LabelConfig` that maps
integer tokens to names and declares semantic roles (background, coding, …).
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator
from typing import Iterable
import numpy as np
from tqdm import tqdm

from .accumulators import BenchmarkAccumulator
from .chain_comparison import (
    compute_intron_chain_metrics,
    compute_scoped_chain_metrics,
)
from .frame_shift import get_frame_shift_metrics
from .indel_metrics import BOUNDARY_ANCHORED_BUCKETS, eval_indel
from .preprocessing import (
    _infer_introns_from_coding_gaps,
    _iter_unmasked_spans,
    resolve_scope_mask,
    resolve_scope_sections,
)
from .section_metrics import eval_sections
from .state_transitions import compute_state_change_errors
from .statistics import Counts
from .structure import extract_structure
from .structural_summary import compute_structural_summary
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


def _validate_metric_config(metrics: Iterable[EvalMetrics], label_config: LabelConfig) -> None:
    """Reject metric/config combinations that are invalid before any work begins.

    Called once at each public entry point so misconfigurations surface
    immediately instead of after some sequences have already been processed.
    """
    if EvalMetrics.PHASE_DRIFT in metrics and not label_config.supports_phase_drift:
        raise ValueError(
            "PHASE_DRIFT is only valid in UTR_CDS_INTRON mode with "
            "evaluation_scope='cds'."
        )


# ---------------------------------------------------------------------------
# Single-sequence benchmark
# ---------------------------------------------------------------------------


def benchmark_gt_vs_pred_single(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_config: LabelConfig,
    metrics: list[EvalMetrics] | None = None,
    mask_labels: np.ndarray | None = None,
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
        Result keyed by metric-group name — the ``.name`` string of each
        requested :class:`EvalMetrics` member (e.g.
        ``EvalMetrics.PHASE_DRIFT`` → ``"PHASE_DRIFT"``).  When
        ``EvalMetrics.STATE_TRANSITIONS`` is requested (it is in the default
        set), the transition keys ``"transition_failures"`` and
        ``"false_transitions"`` are added (see :func:`_eval_transitions`).  A
        ``"metadata"`` entry is always present.

        This single-sequence entry point returns the **raw per-sequence
        fragment**; the aggregating :func:`benchmark_gt_vs_pred_multiple`
        path reduces these to precision/recall and distribution statistics.
        Per group (raw fragment → aggregated form):

        * ``REGION_DISCOVERY`` — the nested discovery tiers
          ``neighborhood_hit`` ⊇ {``internal_hit``, ``full_coverage_hit``} ⊇
          ``perfect_boundary_hit``, each a ``tp/fp/fn`` bundle →
          ``precision``/``recall``/``*_stderr``.
        * ``BOUNDARY_EXACTNESS`` — ``first_sec_correct_3_prime_boundary``,
          ``last_sec_correct_5_prime_boundary``, ``iou_scores``
          (→ ``iou_stats``) and ``fuzzy_metrics`` (signed
          ``boundary_offsets`` → a boundary-precision landscape).
        * ``NUCLEOTIDE_CLASSIFICATION`` — ``nucleotide`` ``tp/fp/fn/tn``
          (→ ``precision``/``recall``/``f1``).
        * ``PHASE_DRIFT`` — ``gt_frames`` per-position phase array plus
          ``n_skipped_*`` (and optional ``boundary_indel_*``) counts.
        * ``DIAGNOSTIC_DEPTH`` — segment-length lists, ``length_emd`` and
          position-bias histograms.

        When ``STRUCTURAL_COHERENCE`` is requested, its entry contains:

        * ``intron_chain`` / ``intron_chain_subset`` / ``intron_chain_superset``
          — **whole-chain** (transcript-level) TP/FP/FN comparing the intron
          segment boundary *sets*: a transcript scores ``tp=1`` only if its
          entire intron set matches exactly, else ``fp=1, fn=1``.  Aggregated
          precision/recall is therefore the *fraction of transcripts with an
          exact chain match*, **not** the fraction of individual introns
          recovered.  A tool that recovers 9 of 10 introns in every transcript
          scores 0 here.  (For intron-level gradation use the per-transcript
          exon-recovery scalars below.)
        * ``exon_chain`` / ``exon_chain_subset`` / ``exon_chain_superset``
          — same whole-chain set semantics applied to coding (exon) segments.
          Subset: pred ⊆ GT (all pred exons are real, may miss some GT).
          Superset: pred ⊇ GT (every GT exon found, may have extras).
        * ``exon_recall_per_transcript`` — float in [0, 1]: fraction of
          GT exons whose ``(start, end)`` was recovered exactly.
        * ``exon_precision_per_transcript`` — float in [0, 1]: fraction of
          predicted exons whose ``(start, end)`` is an exact GT match
          (``0.0`` when the prediction has no exons in scope).
        * ``false_exon_count_per_transcript`` — int ≥ 0: number
          of predicted exons whose ``(start, end)`` is absent from GT.
        * ``segment_count_delta`` — ``pred_count - gt_count``
          (positive = over-segmentation).
        * ``boundary_shift_count`` / ``boundary_shift_total`` — number
          of shifted boundary positions and their summed absolute offset
          in bp across transcripts where GT and pred segment counts match.
        * ``boundary_shift_offsets`` — one record per shifted boundary
          (``{offset, position}``): the signed array-coordinate offset and an
          ``internal`` (splice junction) / ``terminal`` (TSS/TES) tag, used by
          the boundary-shift distribution plots.
    """
    if metrics is None:
        metrics = _DEFAULT_METRICS
    metrics = frozenset(metrics)
    _validate_metric_config(metrics, label_config)

    if infer_introns:
        gt_labels = _infer_introns_from_coding_gaps(gt_labels, label_config)
        pred_labels = _infer_introns_from_coding_gaps(pred_labels, label_config)

    if mask_labels is None:
        # A single unmasked span — return the raw per-sequence fragment.
        return _attach_metadata(_benchmark_chunk(gt_labels, pred_labels, label_config, metrics), label_config)

    # Masked: evaluate each kept span and combine into the un-summarised form.
    accumulator = BenchmarkAccumulator()
    for s, e in _iter_unmasked_spans(mask_labels):
        accumulator.add(_benchmark_chunk(gt_labels[s:e], pred_labels[s:e], label_config, metrics))
    return _attach_metadata(accumulator.merged(), label_config)


def _iter_sequence_fragments(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    label_config: LabelConfig,
    metrics: frozenset,
    mask_labels: np.ndarray | None,
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
    # Row 0 = GT, Row 1 = prediction  (NO sentinel padding here)
    arr = np.stack((gt_labels, pred_labels), axis=0)

    metric_results: dict[str, dict] = {}
    # Transition diagnostics are opt-in via EvalMetrics.STATE_TRANSITIONS (kept in
    # the default metric set so plots still render unless explicitly excluded).
    if EvalMetrics.STATE_TRANSITIONS in metrics:
        transition_analysis = compute_state_change_errors(gt_pred_arr=arr, label_config=label_config)
        metric_results.update({
            "transition_failures": transition_analysis.gt_transition_matrices,
            "false_transitions": {
                "late_catchup": transition_analysis.late_catchup_matrices,
                "premature": transition_analysis.premature_matrices,
                "spurious": transition_analysis.spurious_matrices,
                "stable_position_counts": transition_analysis.stable_position_counts,
            },
        })

    scope = label_config.evaluation_scope
    gt_mask = resolve_scope_mask(gt_labels, scope, label_config)
    pred_mask = resolve_scope_mask(pred_labels, scope, label_config)
    grouped_insertions = get_contiguous_groups(np.where(~gt_mask & pred_mask)[0])
    grouped_deletions = get_contiguous_groups(np.where(gt_mask & ~pred_mask)[0])
    grouped_gt_sections = resolve_scope_sections(gt_labels, scope, label_config)
    grouped_pred_sections = resolve_scope_sections(pred_labels, scope, label_config)


    _indel_result: dict | None = None
    if EvalMetrics.INDEL in metrics or EvalMetrics.PHASE_DRIFT in metrics:
        _indel_result = eval_indel(
            grouped_insertions,
            grouped_deletions,
            gt_mask,
            pred_mask,
            label_config,
            gt_labels,
            len(grouped_gt_sections),
            len(grouped_pred_sections),
        )
        # only output indel if requested
        if EvalMetrics.INDEL in metrics:
            metric_results[EvalMetrics.INDEL.name] = _indel_result

    if _SECTION_DEPENDENT_GROUPS & metrics:
        section_data, boundary_data = eval_sections(
            grouped_gt_sections,
            grouped_pred_sections,
            metrics,
        )
        if section_data:
            metric_results[EvalMetrics.REGION_DISCOVERY.name] = section_data
        if boundary_data:
            metric_results[EvalMetrics.BOUNDARY_EXACTNESS.name] = boundary_data

    if EvalMetrics.NUCLEOTIDE_CLASSIFICATION in metrics:
        metric_results[EvalMetrics.NUCLEOTIDE_CLASSIFICATION.name] = {
            "nucleotide": _compute_nucleotide_level_confusion(
                gt_mask,
                pred_mask,
            )
        }

    if EvalMetrics.PHASE_DRIFT in metrics:
        # Validity is checked up-front at the public entry points.
        metric_results[EvalMetrics.PHASE_DRIFT.name] = _eval_phase_drift(
            gt_mask,
            pred_mask,
            indel_result=_indel_result,
        )

    if EvalMetrics.STRUCTURAL_COHERENCE in metrics:
        metric_results[EvalMetrics.STRUCTURAL_COHERENCE.name] = _eval_structural(gt_labels, pred_labels, label_config)

    if EvalMetrics.DIAGNOSTIC_DEPTH in metrics:
        metric_results[EvalMetrics.DIAGNOSTIC_DEPTH.name] = compute_structural_summary(
            grouped_gt_sections,
            grouped_pred_sections,
        )

    return metric_results




def _eval_phase_drift(
    gt_positive_mask: np.ndarray,
    pred_positive_mask: np.ndarray,
    indel_result: dict | None = None,
) -> dict:
    """Per-position reading-frame deviation for one explicit CDS-like scope.

    When *indel_result* is supplied (the INDEL metric dict for the same chunk),
    the returned dict also carries ``boundary_indel_total`` and
    ``boundary_indel_in_frame`` — counts of boundary-anchored indel events
    (5'/3' extensions and deletions only) and how many of those have lengths
    divisible by 3 (i.e. in-frame).
    """
    raw = get_frame_shift_metrics(
        gt_positive_mask=gt_positive_mask,
        pred_positive_mask=pred_positive_mask,
    )
    result: dict = {
        "gt_frames": np.asarray(raw["frames"]),
        "n_skipped_non_divisible": raw["n_skipped_non_divisible"],
        "n_skipped_short": raw["n_skipped_short"],
    }

    if indel_result is not None:
        lengths = [
            length
            for buckets in indel_result.get("by_boundary", {}).values()
            for bucket, runs in buckets.items()
            if bucket in BOUNDARY_ANCHORED_BUCKETS
            for length in runs
        ]
        result["boundary_indel_total"] = len(lengths)
        result["boundary_indel_in_frame"] = sum(1 for length in lengths if length % 3 == 0)

    return result


def _eval_structural(gt_labels: np.ndarray, pred_labels: np.ndarray, label_config: LabelConfig) -> dict:
    """Build STRUCTURAL_COHERENCE (and, when configured, splice_sites)."""
    gt_struct = extract_structure(gt_labels, label_config)
    pred_struct = extract_structure(pred_labels, label_config)
    scope = label_config.evaluation_scope

    structural_coherence_results = compute_scoped_chain_metrics(
        gt_struct,
        pred_struct,
        label_config,
        scope,
    )
    if label_config.intron_label is not None:
        structural_coherence_results.update(compute_intron_chain_metrics(gt_struct, pred_struct, label_config))
    if (
        label_config.intron_label is not None
        and label_config.splice_donor_label is not None
        and label_config.splice_acceptor_label is not None
    ):
        splice_confusion = eval_splice_site_junctions(gt_struct, pred_struct, label_config)
        splice_site_result = dataclasses.asdict(splice_confusion)
        structural_coherence_results["splice_site_results"] = splice_site_result

    return structural_coherence_results


def _compute_nucleotide_level_confusion(
    gt_positive_mask: np.ndarray, pred_positive_mask: np.ndarray
) -> Counts:
    """Calculate granular base accuracy as a confusion bundle."""
    return Counts(
        tn=int(np.count_nonzero(~gt_positive_mask & ~pred_positive_mask)),
        fp=int(np.count_nonzero(~gt_positive_mask & pred_positive_mask)),
        fn=int(np.count_nonzero(gt_positive_mask & ~pred_positive_mask)),
        tp=int(np.count_nonzero(gt_positive_mask & pred_positive_mask)),
    )


# ---------------------------------------------------------------------------
# Multi-sequence benchmark
# ---------------------------------------------------------------------------


def _warn_phase_drift(metrics: list | frozenset) -> None:
    if EvalMetrics.PHASE_DRIFT in metrics:
        warnings.warn(
            "The PHASE_DRIFT metric should only be used when you are certain "
            "that the transcript contains all annotated exons.  Otherwise "
            "the results will be misleading.",
            stacklevel=3,
        )


class StreamingBenchmark:
    """Incrementally aggregate per-sequence benchmarks without materialising them.

    Wraps a single :class:`BenchmarkAccumulator`.  Feed paired GT/pred arrays one
    at a time with :meth:`add` — each array is reduced to a per-sequence fragment,
    accumulated, and may then be discarded by the caller — and call :meth:`result`
    once to obtain the aggregated summary.  This lets callers build label arrays
    lazily (one transcript pair at a time) instead of holding every array in
    memory at once.

    :func:`benchmark_gt_vs_pred_multiple` (aggregate mode) is implemented on top
    of this class, so streaming and list-based aggregation return identical
    numbers for the same sequence of pairs.
    """

    def __init__(
        self,
        label_config: LabelConfig,
        metrics: list[EvalMetrics] | None = None,
        infer_introns: bool = False,
    ) -> None:
        metrics = list(metrics) if metrics is not None else list(_DEFAULT_METRICS)
        _validate_metric_config(metrics, label_config)
        _warn_phase_drift(metrics)
        self._label_config = label_config
        self._metrics = frozenset(metrics)
        self._infer_introns = infer_introns
        self._accumulator = BenchmarkAccumulator()
        self._count = 0

    @property
    def count(self) -> int:
        """Number of sequences added so far."""
        return self._count

    def add(
        self,
        gt_labels: np.ndarray,
        pred_labels: np.ndarray,
        mask_labels: np.ndarray | None = None,
    ) -> None:
        """Accumulate one GT/pred pair (same per-sequence path as the list API)."""
        for fragment in _iter_sequence_fragments(
            gt_labels,
            pred_labels,
            self._label_config,
            self._metrics,
            mask_labels,
            self._infer_introns,
        ):
            self._accumulator.add(fragment)
        self._count += 1

    def result(self) -> dict:
        """Return the aggregated, metadata-annotated summary of all added pairs."""
        return _attach_metadata(self._accumulator.summarise(), self._label_config)


def benchmark_gt_vs_pred_multiple(
    gt_labels: list[np.ndarray],
    pred_labels: list[np.ndarray],
    label_config: LabelConfig,
    metrics: list[EvalMetrics] | None = None,
    return_individual_results: bool = False,
    mask_labels: list[np.ndarray] | None = None,
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

    metrics = list(metrics) if metrics is not None else list(_DEFAULT_METRICS)

    if return_individual_results:
        _validate_metric_config(metrics, label_config)
        _warn_phase_drift(metrics)
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

    # Aggregate path: stream each pair through a single accumulator.  Identical
    # to the list-based loop it replaces — the validation and PHASE_DRIFT warning
    # are emitted by StreamingBenchmark.
    bench = StreamingBenchmark(label_config, metrics, infer_introns=infer_introns)
    for i in tqdm(range(len(gt_labels)), desc="Running benchmark"):
        bench.add(
            gt_labels[i],
            pred_labels[i],
            mask_labels[i] if mask_labels is not None else None,
        )

    return bench.result()


def _attach_metadata(result: dict, label_config: LabelConfig) -> dict:
    """Attach annotation-mode metadata to a benchmark result payload."""
    payload = dict(result)
    payload["metadata"] = {
        "annotation_mode": label_config.annotation_mode.value,
        "evaluation_scope": label_config.evaluation_scope.value,
    }
    return payload
