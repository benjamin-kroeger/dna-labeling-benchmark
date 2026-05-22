"""Typed cross-sequence accumulators.

Replaces the old polymorphic ``recursive_merge`` + ``_aggregate_summary_metrics``
pair.  Each metric group owns an accumulator that knows the exact shape of its
per-sequence fragment and how to combine it:

* ``add(fragment)`` — fold in one sequence's (or masked chunk's) raw fragment.
  Each accumulator no-ops when its key is absent, so the same fragment can be
  offered to every accumulator.
* ``merged()`` — the un-summarised combined form (used for a masked single
  sequence, which is still "one sequence" and so is not reduced to
  precision/recall).
* ``summarise()`` — the final user-facing aggregate.

Because every combine operation here is associative (sum, concat, list-collect),
streaming all per-chunk fragments flat into one accumulator produces the same
result as merging per sequence and then across sequences.
"""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .boundary_precision import _compute_boundary_precision_landscape
from .statistics import _compute_distribution_stats, summarise_counts


def _add_matrix_dict(target: dict, source: dict) -> None:
    """Element-wise add per-label matrices from *source* into *target*."""
    for key, matrix in source.items():
        if key in target:
            target[key] = target[key] + matrix
        else:
            target[key] = np.array(matrix, copy=True)


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
    """Concatenates the per-bucket lists of mismatch index arrays."""

    KEY: ClassVar[str] = "INDEL"

    buckets: dict = field(default_factory=lambda: defaultdict(list))
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        if self.KEY not in fragment:
            return
        self._seen = True
        for bucket, arrays in fragment[self.KEY].items():
            self.buckets[bucket].extend(arrays)

    def _to_dict(self) -> dict:
        if not self._seen:
            return {}
        return {self.KEY: {bucket: list(arrays) for bucket, arrays in self.buckets.items()}}

    def merged(self) -> dict:
        return self._to_dict()

    def summarise(self) -> dict:
        return self._to_dict()


@dataclass
class RegionDiscoveryAccumulator:
    """Collects per-sequence Counts per strictness level; summarises to P/R."""

    KEY: ClassVar[str] = "REGION_DISCOVERY"
    LEVELS: ClassVar[tuple] = (
        "neighborhood_hit",
        "internal_hit",
        "full_coverage_hit",
        "perfect_boundary_hit",
    )

    levels: dict = field(default_factory=lambda: defaultdict(list))
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        if self.KEY not in fragment:
            return
        self._seen = True
        rd = fragment[self.KEY]
        for level in self.LEVELS:
            self.levels[level].append(rd[level])

    def merged(self) -> dict:
        if not self._seen:
            return {}
        return {self.KEY: {level: list(self.levels[level]) for level in self.LEVELS}}

    def summarise(self) -> dict:
        if not self._seen:
            return {}
        return {self.KEY: {level: summarise_counts(self.levels[level]).to_dict() for level in self.LEVELS}}


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
        if self.KEY not in fragment:
            return
        self._seen = True
        be = fragment[self.KEY]
        self.first.append(be["first_sec_correct_3_prime_boundary"])
        self.last.append(be["last_sec_correct_5_prime_boundary"])
        self.iou.extend(be["iou_scores"])
        self.residuals.extend(be["fuzzy_metrics"]["boundary_residuals"])
        self.total_gt += be["fuzzy_metrics"]["total_gt"]

    def merged(self) -> dict:
        if not self._seen:
            return {}
        return {
            self.KEY: {
                "first_sec_correct_3_prime_boundary": list(self.first),
                "last_sec_correct_5_prime_boundary": list(self.last),
                "iou_scores": list(self.iou),
                "fuzzy_metrics": {"boundary_residuals": list(self.residuals), "total_gt": self.total_gt},
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
    """Collects per-sequence nucleotide Counts; summarises to P/R/F1."""

    KEY: ClassVar[str] = "NUCLEOTIDE_CLASSIFICATION"

    counts: list = field(default_factory=list)
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        if self.KEY not in fragment:
            return
        self._seen = True
        self.counts.append(fragment[self.KEY]["nucleotide"])

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
    """Concatenates per-position frame-drift values."""

    KEY: ClassVar[str] = "FRAMESHIFT"

    frames: list = field(default_factory=list)
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        if self.KEY not in fragment:
            return
        self._seen = True
        self.frames.extend(fragment[self.KEY]["gt_frames"])

    def _to_dict(self) -> dict:
        if not self._seen:
            return {}
        return {self.KEY: {"gt_frames": list(self.frames)}}

    def merged(self) -> dict:
        return self._to_dict()

    def summarise(self) -> dict:
        return self._to_dict()


@dataclass
class StructuralAccumulator:
    """Aggregates the whole STRUCTURAL_COHERENCE group.

    The per-sequence fragment nests two sub-results under the group key:
    ``chain_metric_results`` (intron/exon chain Counts plus per-transcript
    soft metrics) and an optional ``splice_site_results`` (donor/acceptor
    confusion counts).  Both are folded in here and re-emitted under the same
    two sub-keys, so splice sites stay part of one eval group rather than a
    separate top-level entry.
    """

    KEY: ClassVar[str] = "STRUCTURAL_COHERENCE"
    CHAIN_KEY: ClassVar[str] = "chain_metric_results"
    SPLICE_KEY: ClassVar[str] = "splice_site_results"
    CHAIN_KEYS: ClassVar[tuple] = (
        "intron_chain",
        "intron_chain_subset",
        "intron_chain_superset",
        "exon_chain",
        "exon_chain_subset",
        "exon_chain_superset",
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
    )

    chains: dict = field(default_factory=lambda: defaultdict(list))
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

        sc = group.get(self.CHAIN_KEY, {})
        for key in self.CHAIN_KEYS:
            if key in sc:
                self.chains[key].append(sc[key])
        if "exon_recall_per_transcript" in sc:
            self.exon_recall.append(sc["exon_recall_per_transcript"])
        if "hallucinated_exon_count_per_transcript" in sc:
            self.hallucinated.append(sc["hallucinated_exon_count_per_transcript"])
        if "segment_count_delta" in sc:
            self.segment_count_delta.append(sc["segment_count_delta"])
        if "transcript_match_class" in sc:
            self.transcript_match_class.append(sc["transcript_match_class"])
        if "boundary_shift_count" in sc:
            self.boundary_shift_count.append(sc["boundary_shift_count"])
        if "boundary_shift_total" in sc:
            self.boundary_shift_total.append(sc["boundary_shift_total"])

        ss = group.get(self.SPLICE_KEY)
        if ss is not None:
            self._splice_seen = True
            for key in self.SPLICE_FIELDS:
                self.splice_sums[key] = self.splice_sums.get(key, 0) + ss[key]

    def _soft_metrics(self, sc: dict) -> None:
        if self.exon_recall:
            sc["exon_recall_per_transcript"] = list(self.exon_recall)
        if self.hallucinated:
            sc["hallucinated_exon_count_per_transcript"] = list(self.hallucinated)
        if self.transcript_match_class:
            sc["transcript_match_class"] = list(self.transcript_match_class)
        if self.boundary_shift_count:
            sc["boundary_shift_count"] = list(self.boundary_shift_count)
        if self.boundary_shift_total:
            sc["boundary_shift_total"] = list(self.boundary_shift_total)

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
        sc = {key: list(counts) for key, counts in self.chains.items()}
        if self.segment_count_delta:
            sc["segment_count_delta"] = list(self.segment_count_delta)
        self._soft_metrics(sc)
        group: dict = {self.CHAIN_KEY: sc}
        if self._splice_seen:
            group[self.SPLICE_KEY] = dict(self.splice_sums)
        return {self.KEY: group}

    def summarise(self) -> dict:
        if not self._seen:
            return {}
        sc = {key: summarise_counts(counts).to_dict() for key, counts in self.chains.items()}
        if self.segment_count_delta:
            sc["segment_count_delta"] = _compute_distribution_stats(self.segment_count_delta, is_abs=False)
        if self.transcript_match_class:
            counts = Counter(self.transcript_match_class)
            total = sum(counts.values())
            sc["transcript_match_distribution"] = dict(counts)
            sc["exact_match_rate"] = counts.get("exact", 0) / total if total > 0 else 0.0
        self._soft_metrics(sc)
        group: dict = {self.CHAIN_KEY: sc}
        if self._splice_seen:
            group[self.SPLICE_KEY] = self._splice_summary()
        return {self.KEY: group}


@dataclass
class DiagnosticDepthAccumulator:
    """Concatenates segment lengths, collects EMD, sums position-bias histograms."""

    KEY: ClassVar[str] = "DIAGNOSTIC_DEPTH"

    gt_lengths: list = field(default_factory=list)
    pred_lengths: list = field(default_factory=list)
    length_emd: list = field(default_factory=list)
    hist_fn: object = None
    hist_fp: object = None
    _seen: bool = False

    def add(self, fragment: dict) -> None:
        if self.KEY not in fragment:
            return
        self._seen = True
        dd = fragment[self.KEY]
        self.gt_lengths.extend(dd["gt_segment_lengths"])
        self.pred_lengths.extend(dd["pred_segment_lengths"])
        self.length_emd.append(dd["length_emd"])
        fn = np.asarray(dd["position_bias_histogram_fn"], dtype=np.int64)
        fp = np.asarray(dd["position_bias_histogram_fp"], dtype=np.int64)
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
        out = self._common()
        out["length_emd"] = list(self.length_emd)
        return {self.KEY: out}

    def summarise(self) -> dict:
        if not self._seen:
            return {}
        out = self._common()
        out["length_emd"] = _compute_distribution_stats(self.length_emd, is_abs=False)
        return {self.KEY: out}


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
