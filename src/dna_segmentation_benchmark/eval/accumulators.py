"""Typed cross-sequence accumulators.

Each metric family owns an accumulator that knows the flat fragment shape
emitted by the array benchmark core. Transition matrices remain unscoped
because they already operate on the full label vocabulary.
"""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .boundary_precision import _compute_boundary_precision_landscape
from .statistics import Counts, _compute_distribution_stats, summarise_counts


def _add_matrix_dict(target: dict, source: dict) -> None:
    """Element-wise add per-label matrices from *source* into *target*."""
    for key, matrix in source.items():
        if key in target:
            target[key] = target[key] + matrix
        else:
            target[key] = np.array(matrix, copy=True)


def _coerce_counts(value) -> Counts:
    """Normalise count bundles to :class:`Counts`."""
    if isinstance(value, Counts):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return Counts(
            tp=int(value.get("tp", 0)),
            fp=int(value.get("fp", 0)),
            fn=int(value.get("fn", 0)),
            tn=int(value.get("tn", 0)),
        )
    raise TypeError(f"Unsupported Counts payload type: {type(value)!r}")


@dataclass
class TransitionsAccumulator:
    """Sums the always-on state-transition matrices and stable-position counts."""

    failures: dict = field(default_factory=dict)
    late: dict = field(default_factory=dict)
    premature: dict = field(default_factory=dict)
    spurious: dict = field(default_factory=dict)
    stable: dict = field(default_factory=dict)
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        if "transition_failures" not in fragment:
            return
        self._seen = True
        _add_matrix_dict(self.failures, fragment["transition_failures"])
        false_transitions = fragment["false_transitions"]
        _add_matrix_dict(self.late, false_transitions["late_catchup"])
        _add_matrix_dict(self.premature, false_transitions["premature"])
        _add_matrix_dict(self.spurious, false_transitions["spurious"])
        for label, count in false_transitions["stable_position_counts"].items():
            self.stable[label] = self.stable.get(label, 0) + count

    def _to_dict(self) -> dict:
        if not self._seen:
            return {}
        return {
            "transition_failures": self.failures,
            "false_transitions": {
                "late_catchup": self.late,
                "premature": self.premature,
                "spurious": self.spurious,
                "stable_position_counts": dict(self.stable),
            },
        }

    def merged(self) -> dict:
        return self._to_dict()

    def summarise(self) -> dict:
        return self._to_dict()


@dataclass
class IndelAccumulator:
    """Concatenates boundary-typed mismatch run lengths and their denominators.

    Fragments carry ``{"INDEL": {"by_boundary": {"LEFT:RIGHT": {bucket:
    [length, ...]}}, "junction_opportunities": {"LEFT:RIGHT": int},
    "n_gt_segments": int, "n_pred_segments": int}}``.  Run lengths are merged per
    ``"LEFT:RIGHT"`` boundary name key and per bucket; the *opportunity*
    denominators (junction transition counts and segment counts) are summed so
    downstream consumers can turn pooled counts into per-junction / per-segment
    rates.  Keys are already canonical label-name strings, so the merge is
    purely key-agnostic.
    """

    KEY: ClassVar[str] = "INDEL"

    by_boundary: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
    junction_opportunities: dict = field(default_factory=lambda: defaultdict(int))
    n_gt_segments: int = 0
    n_pred_segments: int = 0
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        payload = fragment.get(self.KEY)
        if not isinstance(payload, dict):
            return
        self._seen = True
        for boundary, bucket_map in payload.get("by_boundary", {}).items():
            for bucket, lengths in bucket_map.items():
                self.by_boundary[boundary][bucket].extend(lengths)
        for boundary, count in payload.get("junction_opportunities", {}).items():
            self.junction_opportunities[boundary] += count
        self.n_gt_segments += int(payload.get("n_gt_segments", 0))
        self.n_pred_segments += int(payload.get("n_pred_segments", 0))

    def _to_dict(self) -> dict:
        if not self._seen:
            return {}
        return {
            self.KEY: {
                "by_boundary": {
                    boundary: {bucket: list(lengths) for bucket, lengths in bucket_map.items()}
                    for boundary, bucket_map in self.by_boundary.items()
                },
                "junction_opportunities": dict(self.junction_opportunities),
                "n_gt_segments": self.n_gt_segments,
                "n_pred_segments": self.n_pred_segments,
            }
        }

    def merged(self) -> dict:
        return self._to_dict()

    def summarise(self) -> dict:
        return self._to_dict()


@dataclass
class RegionDiscoveryAccumulator:
    """Collects region-discovery Counts and containment integers; summarises to P/R and conditional rates.

    ``neighborhood_hit`` and ``perfect_boundary_hit`` are coherent detection
    contingency tables reported as precision/recall.  ``containment`` tracks
    how often matched predictions are spatially contained in (or fully covering)
    the GT, expressed as conditional rates over matched pairs — not P/R.
    """

    KEY: ClassVar[str] = "REGION_DISCOVERY"
    LEVELS: ClassVar[tuple] = (
        "neighborhood_hit",
        "perfect_boundary_hit",
    )

    levels: dict = field(default_factory=lambda: defaultdict(list))
    _containment_matched: list = field(default_factory=list)
    _containment_internal: list = field(default_factory=list)
    _containment_full_coverage: list = field(default_factory=list)
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        payload = fragment.get(self.KEY)
        if not isinstance(payload, dict):
            return
        self._seen = True
        for level in self.LEVELS:
            self.levels[level].append(_coerce_counts(payload[level]))
        containment = payload.get("containment")
        if containment is not None:
            if isinstance(containment, list):
                for c in containment:
                    self._containment_matched.append(int(c.get("matched", 0)))
                    self._containment_internal.append(int(c.get("internal", 0)))
                    self._containment_full_coverage.append(int(c.get("full_coverage", 0)))
            else:
                self._containment_matched.append(int(containment.get("matched", 0)))
                self._containment_internal.append(int(containment.get("internal", 0)))
                self._containment_full_coverage.append(int(containment.get("full_coverage", 0)))

    def merged(self) -> dict:
        if not self._seen:
            return {}
        containment_list = [
            {"matched": m, "internal": i, "full_coverage": f}
            for m, i, f in zip(
                self._containment_matched,
                self._containment_internal,
                self._containment_full_coverage,
            )
        ]
        return {
            self.KEY: {
                **{level: list(self.levels[level]) for level in self.LEVELS},
                "containment": containment_list,
            }
        }

    def summarise(self) -> dict:
        if not self._seen:
            return {}
        total_matched = sum(self._containment_matched)
        total_internal = sum(self._containment_internal)
        total_full_coverage = sum(self._containment_full_coverage)
        containment = {
            "internal_rate": total_internal / total_matched if total_matched > 0 else None,
            "full_coverage_rate": total_full_coverage / total_matched if total_matched > 0 else None,
        }
        return {
            self.KEY: {
                **{level: summarise_counts(self.levels[level]).to_dict() for level in self.LEVELS},
                "containment": containment,
            }
        }


@dataclass
class BoundaryExactnessAccumulator:
    """Collects terminal-boundary flags, IoU scores and residuals."""

    KEY: ClassVar[str] = "BOUNDARY_EXACTNESS"

    first: list = field(default_factory=list)
    last: list = field(default_factory=list)
    iou: list = field(default_factory=list)
    residuals: list = field(default_factory=list)
    total_gt: int = 0
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        payload = fragment.get(self.KEY)
        if not isinstance(payload, dict):
            return
        self._seen = True
        fuzzy_metrics = payload.get("fuzzy_metrics") or {}
        self.first.append(payload.get("first_sec_correct_3_prime_boundary", 0))
        self.last.append(payload.get("last_sec_correct_5_prime_boundary", 0))
        self.iou.extend(payload.get("iou_scores", []))
        self.residuals.extend(fuzzy_metrics.get("boundary_residuals", []))
        self.total_gt += fuzzy_metrics.get("total_gt", 0)

    def merged(self) -> dict:
        if not self._seen:
            return {}
        return {
            self.KEY: {
                "first_sec_correct_3_prime_boundary": list(self.first),
                "last_sec_correct_5_prime_boundary": list(self.last),
                "iou_scores": list(self.iou),
                "fuzzy_metrics": {
                    "boundary_residuals": list(self.residuals),
                    "total_gt": self.total_gt,
                },
            }
        }

    def summarise(self) -> dict:
        if not self._seen:
            return {}
        return {
            self.KEY: {
                "first_sec_correct_3_prime_boundary": list(self.first),
                "last_sec_correct_5_prime_boundary": list(self.last),
                "iou_scores": list(self.iou),
                "iou_stats": _compute_distribution_stats(self.iou, is_abs=False),
                "fuzzy_metrics": _compute_boundary_precision_landscape(
                    residuals=self.residuals,
                    total_gt_count=self.total_gt,
                ),
            }
        }


@dataclass
class NucleotideAccumulator:
    """Collects nucleotide Counts; summarises to P/R/F1."""

    KEY: ClassVar[str] = "NUCLEOTIDE_CLASSIFICATION"

    counts: list = field(default_factory=list)
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        payload = fragment.get(self.KEY)
        if not isinstance(payload, dict):
            return
        self._seen = True
        self.counts.append(_coerce_counts(payload["nucleotide"]))

    def merged(self) -> dict:
        if not self._seen:
            return {}
        return {self.KEY: {"nucleotide": list(self.counts)}}

    def summarise(self) -> dict:
        if not self._seen:
            return {}
        stat = summarise_counts(self.counts)
        p, r = stat.precision or 0.0, stat.recall or 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        return {self.KEY: {"nucleotide": dataclasses.replace(stat, f1=f1).to_dict()}}


@dataclass
class FrameshiftAccumulator:
    """Concatenates coding-phase drift values across sequences."""

    KEY: ClassVar[str] = "FRAMESHIFT"

    frames: list = field(default_factory=list)
    boundary_indel_total: int = 0
    boundary_indel_in_frame: int = 0
    n_skipped_non_divisible: int = 0
    n_skipped_short: int = 0
    _seen: bool = False
    _has_indel_counts: bool = False

    def add(self, fragment: dict) -> None:
        payload = fragment.get(self.KEY)
        if not isinstance(payload, dict):
            return
        self._seen = True
        self.frames.extend(list(payload.get("gt_frames", payload.get("frames", []))))
        if "boundary_indel_total" in payload and "boundary_indel_in_frame" in payload:
            self._has_indel_counts = True
            self.boundary_indel_total += payload["boundary_indel_total"]
            self.boundary_indel_in_frame += payload["boundary_indel_in_frame"]
        self.n_skipped_non_divisible += payload.get("n_skipped_non_divisible", 0)
        self.n_skipped_short += payload.get("n_skipped_short", 0)

    def _to_dict(self) -> dict:
        if not self._seen:
            return {}
        result: dict = {
            "gt_frames": list(self.frames),
            "n_skipped_non_divisible": self.n_skipped_non_divisible,
            "n_skipped_short": self.n_skipped_short,
        }
        if self._has_indel_counts:
            result["boundary_indel_total"] = self.boundary_indel_total
            result["boundary_indel_in_frame"] = self.boundary_indel_in_frame
        return {self.KEY: result}

    def merged(self) -> dict:
        return self._to_dict()

    def summarise(self) -> dict:
        return self._to_dict()


@dataclass
class StructuralAccumulator:
    """Aggregates STRUCTURAL_COHERENCE diagnostics."""

    KEY: ClassVar[str] = "STRUCTURAL_COHERENCE"
    SPLICE_KEY: ClassVar[str] = "splice_site_results"
    EXON_CHAIN_KEYS: ClassVar[tuple] = (
        "exon_chain",
        "exon_chain_subset",
        "exon_chain_superset",
    )
    INTRON_CHAIN_KEYS: ClassVar[tuple] = (
        "intron_chain",
        "intron_chain_subset",
        "intron_chain_superset",
    )
    SPLICE_FIELDS: ClassVar[tuple] = (
        "both_correct",
        "donor_only",
        "acceptor_only",
        "neither",
        "donor_tp",
        "donor_fp",
        "donor_fn",
        "acceptor_tp",
        "acceptor_fp",
        "acceptor_fn",
        "gt_malformed_junctions",
        "pred_malformed_junctions",
    )

    exon_chains: dict = field(default_factory=lambda: defaultdict(list))
    intron_chains: dict = field(default_factory=lambda: defaultdict(list))
    exon_recall: list = field(default_factory=list)
    hallucinated: list = field(default_factory=list)
    segment_count_delta: list = field(default_factory=list)
    transcript_match_class: list = field(default_factory=list)
    boundary_shift_count: list = field(default_factory=list)
    boundary_shift_total: list = field(default_factory=list)
    splice_sums: dict = field(default_factory=dict)
    _seen: bool = False
    _splice_seen: bool = False

    def add(self, fragment: dict) -> None:
        if self.KEY not in fragment:
            return
        self._seen = True
        group = fragment[self.KEY]
        for key in self.EXON_CHAIN_KEYS:
            if key in group:
                self.exon_chains[key].append(_coerce_counts(group[key]))
        if "exon_recall_per_transcript" in group:
            self.exon_recall.append(group["exon_recall_per_transcript"])
        if "hallucinated_exon_count_per_transcript" in group:
            self.hallucinated.append(group["hallucinated_exon_count_per_transcript"])
        if "segment_count_delta" in group:
            self.segment_count_delta.append(group["segment_count_delta"])
        if "transcript_match_class" in group:
            self.transcript_match_class.append(group["transcript_match_class"])
        if "boundary_shift_count" in group:
            self.boundary_shift_count.append(group["boundary_shift_count"])
        if "boundary_shift_total" in group:
            self.boundary_shift_total.append(group["boundary_shift_total"])

        for key in self.INTRON_CHAIN_KEYS:
            if key in group:
                self.intron_chains[key].append(_coerce_counts(group[key]))

        ss = group.get(self.SPLICE_KEY)
        if ss is not None:
            self._splice_seen = True
            for key in self.SPLICE_FIELDS:
                self.splice_sums[key] = self.splice_sums.get(key, 0) + ss[key]

    def _soft_metrics(self, payload: dict) -> None:
        if self.exon_recall:
            payload["exon_recall_per_transcript"] = list(self.exon_recall)
        if self.hallucinated:
            payload["hallucinated_exon_count_per_transcript"] = list(self.hallucinated)
        if self.transcript_match_class:
            payload["transcript_match_class"] = list(self.transcript_match_class)
        if self.boundary_shift_count:
            payload["boundary_shift_count"] = list(self.boundary_shift_count)
        if self.boundary_shift_total:
            payload["boundary_shift_total"] = list(self.boundary_shift_total)

    def _splice_summary(self) -> dict:
        ss = dict(self.splice_sums)
        d_tp, d_fp, d_fn = ss["donor_tp"], ss["donor_fp"], ss["donor_fn"]
        a_tp, a_fp, a_fn = ss["acceptor_tp"], ss["acceptor_fp"], ss["acceptor_fn"]
        ss["donor_precision"] = d_tp / (d_tp + d_fp) if (d_tp + d_fp) > 0 else 0.0
        ss["donor_recall"] = d_tp / (d_tp + d_fn) if (d_tp + d_fn) > 0 else 0.0
        ss["acceptor_precision"] = a_tp / (a_tp + a_fp) if (a_tp + a_fp) > 0 else 0.0
        ss["acceptor_recall"] = a_tp / (a_tp + a_fn) if (a_tp + a_fn) > 0 else 0.0
        return ss

    def merged(self) -> dict:
        if not self._seen:
            return {}
        group: dict = {
            key: list(counts)
            for key, counts in self.exon_chains.items()
        }
        if self.segment_count_delta:
            group["segment_count_delta"] = list(self.segment_count_delta)
        self._soft_metrics(group)
        for key, counts in self.intron_chains.items():
            group[key] = list(counts)
        if self._splice_seen:
            group[self.SPLICE_KEY] = dict(self.splice_sums)
        return {self.KEY: group} if group else {}

    def summarise(self) -> dict:
        if not self._seen:
            return {}
        group: dict = {
            key: summarise_counts(counts).to_dict()
            for key, counts in self.exon_chains.items()
        }
        if self.segment_count_delta:
            group["segment_count_delta"] = _compute_distribution_stats(
                self.segment_count_delta,
                is_abs=False,
            )
        if self.transcript_match_class:
            counts = Counter(self.transcript_match_class)
            total = sum(counts.values())
            group["transcript_match_distribution"] = dict(counts)
            group["exact_match_rate"] = counts.get("exact", 0) / total if total > 0 else 0.0
        self._soft_metrics(group)
        for key, counts in self.intron_chains.items():
            group[key] = summarise_counts(counts).to_dict()
        if self._splice_seen:
            group[self.SPLICE_KEY] = self._splice_summary()
        return {self.KEY: group} if group else {}


@dataclass
class SpliceSiteAccumulator:
    """Public legacy splice-site accumulator used by direct unit tests."""

    KEY: ClassVar[str] = "splice_sites"
    FIELDS: ClassVar[tuple] = StructuralAccumulator.SPLICE_FIELDS

    sums: dict = field(default_factory=dict)
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        payload = fragment.get(self.KEY)
        if not isinstance(payload, dict):
            return
        self._seen = True
        for key in self.FIELDS:
            self.sums[key] = self.sums.get(key, 0) + payload.get(key, 0)

    def _summary(self) -> dict:
        ss = {key: self.sums.get(key, 0) for key in self.FIELDS}
        d_tp, d_fp, d_fn = ss["donor_tp"], ss["donor_fp"], ss["donor_fn"]
        a_tp, a_fp, a_fn = ss["acceptor_tp"], ss["acceptor_fp"], ss["acceptor_fn"]
        ss["donor_precision"] = d_tp / (d_tp + d_fp) if (d_tp + d_fp) > 0 else 0.0
        ss["donor_recall"] = d_tp / (d_tp + d_fn) if (d_tp + d_fn) > 0 else 0.0
        ss["acceptor_precision"] = a_tp / (a_tp + a_fp) if (a_tp + a_fp) > 0 else 0.0
        ss["acceptor_recall"] = a_tp / (a_tp + a_fn) if (a_tp + a_fn) > 0 else 0.0
        return ss

    def merged(self) -> dict:
        if not self._seen:
            return {}
        return {self.KEY: {key: self.sums.get(key, 0) for key in self.FIELDS}}

    def summarise(self) -> dict:
        if not self._seen:
            return {}
        return {self.KEY: self._summary()}


@dataclass
class DiagnosticDepthAccumulator:
    """Concatenates segment lengths, EMDs and bias histograms."""

    KEY: ClassVar[str] = "DIAGNOSTIC_DEPTH"

    gt_lengths: list = field(default_factory=list)
    pred_lengths: list = field(default_factory=list)
    length_emd: list = field(default_factory=list)
    hist_fn: np.ndarray | None = None
    hist_fp: np.ndarray | None = None
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        payload = fragment.get(self.KEY)
        if not isinstance(payload, dict):
            return
        self._seen = True
        self.gt_lengths.extend(payload["gt_segment_lengths"])
        self.pred_lengths.extend(payload["pred_segment_lengths"])
        self.length_emd.append(payload["length_emd"])
        fn = np.asarray(payload["position_bias_histogram_fn"], dtype=np.int64)
        fp = np.asarray(payload["position_bias_histogram_fp"], dtype=np.int64)
        self.hist_fn = fn.copy() if self.hist_fn is None else self.hist_fn + fn
        self.hist_fp = fp.copy() if self.hist_fp is None else self.hist_fp + fp

    def _common(self) -> dict:
        return {
            "gt_segment_lengths": list(self.gt_lengths),
            "pred_segment_lengths": list(self.pred_lengths),
            "position_bias_histogram_fn": self.hist_fn.tolist(),
            "position_bias_histogram_fp": self.hist_fp.tolist(),
        }

    def merged(self) -> dict:
        if not self._seen:
            return {}
        return {self.KEY: {**self._common(), "length_emd": list(self.length_emd)}}

    def summarise(self) -> dict:
        if not self._seen:
            return {}
        return {
            self.KEY: {
                **self._common(),
                "length_emd": _compute_distribution_stats(self.length_emd, is_abs=False),
            }
        }


class BenchmarkAccumulator:
    """Routes each per-sequence fragment to every metric accumulator."""

    def __init__(self) -> None:
        self._accumulators = [
            TransitionsAccumulator(),
            IndelAccumulator(),
            RegionDiscoveryAccumulator(),
            BoundaryExactnessAccumulator(),
            NucleotideAccumulator(),
            FrameshiftAccumulator(),
            SpliceSiteAccumulator(),
            StructuralAccumulator(),
            DiagnosticDepthAccumulator(),
        ]

    def add(self, fragment: dict) -> None:
        for accumulator in self._accumulators:
            accumulator.add(fragment)

    def merged(self) -> dict:
        out: dict = {}
        for accumulator in self._accumulators:
            out.update(accumulator.merged())
        return out

    def summarise(self) -> dict:
        out: dict = {}
        for accumulator in self._accumulators:
            out.update(accumulator.summarise())
        return out
