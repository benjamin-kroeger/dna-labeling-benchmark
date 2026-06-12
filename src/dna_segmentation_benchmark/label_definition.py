"""Label semantics for the DNA segmentation benchmark.

This module defines the public :class:`LabelConfig` model and the annotation
mode/scope enums used by the array-based benchmark core.

Stage 1 introduces two explicit annotation modes:

* ``EXON_INTRON`` — one exonic label, optional intron and splice-site labels
* ``UTR_CDS_INTRON`` — distinct ``5' UTR``, ``CDS``, and ``3' UTR`` labels,
  plus optional intron and splice-site labels
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator


class AnnotationMode(str, Enum):
    """Supported annotation semantics for array-based benchmarking."""

    EXON_INTRON = "EXON_INTRON"
    UTR_CDS_INTRON = "UTR_CDS_INTRON"


class BenchmarkScope(str, Enum):
    """Metric scopes available in Stage 1 array benchmarking."""

    TRANSCRIPT_EXON = "transcript_exon"
    CDS = "cds"


class LabelConfig(BaseModel):
    """Complete label definition for one benchmarking run.

    Required fields
    ---------------
    annotation_mode : AnnotationMode
        Declares whether arrays represent a single exon-like class or a
        transcript-anatomy label set with explicit UTR/CDS semantics.
    background_label : int
        Token for non-coding / intergenic / background regions.

    Mode-specific required fields
    -----------------------------
    EXON_INTRON
        ``exon_label``
    UTR_CDS_INTRON
        ``cds_label``, ``five_prime_utr_label``, ``three_prime_utr_label``

    Optional semantic roles
    -----------------------
    intron_label : int | None
        Token for intronic nucleotides.
    splice_donor_label : int | None
        Token for the 5' donor splice site.
    splice_acceptor_label : int | None
        Token for the 3' acceptor splice site.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    annotation_mode: AnnotationMode
    evaluation_scope: BenchmarkScope = BenchmarkScope.TRANSCRIPT_EXON
    background_label: int

    exon_label: Optional[int] = None
    cds_label: Optional[int] = None
    five_prime_utr_label: Optional[int] = None
    three_prime_utr_label: Optional[int] = None

    intron_label: Optional[int] = None
    splice_donor_label: Optional[int] = None
    splice_acceptor_label: Optional[int] = None

    @model_validator(mode="after")
    def _validate_mode_shape(self) -> "LabelConfig":
        """Reject invalid field combinations for the selected mode."""
        if self.annotation_mode == AnnotationMode.EXON_INTRON:
            if self.exon_label is None:
                raise ValueError("EXON_INTRON mode requires exon_label to be set.")
            forbidden = {
                "cds_label": self.cds_label,
                "five_prime_utr_label": self.five_prime_utr_label,
                "three_prime_utr_label": self.three_prime_utr_label,
            }
            present = [name for name, value in forbidden.items() if value is not None]
            if present:
                raise ValueError(
                    "EXON_INTRON mode does not allow "
                    + ", ".join(present)
                    + ". Use UTR_CDS_INTRON mode for explicit UTR/CDS labels."
                )
        else:
            if self.exon_label is not None:
                raise ValueError("UTR_CDS_INTRON mode does not allow exon_label.")
            missing = [
                name
                for name, value in (
                    ("cds_label", self.cds_label),
                    ("five_prime_utr_label", self.five_prime_utr_label),
                    ("three_prime_utr_label", self.three_prime_utr_label),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "UTR_CDS_INTRON mode requires "
                    + ", ".join(missing)
                    + " to be set."
                )
        valid_scopes = self.available_scopes()
        if self.evaluation_scope not in valid_scopes:
            supported = ", ".join(scope.value for scope in valid_scopes)
            raise ValueError(
                "evaluation_scope="
                f"{self.evaluation_scope.value!r} is not valid for annotation_mode="
                f"{self.annotation_mode.value}. Supported scopes: {supported}."
            )

        if (self.splice_donor_label is None) ^ (self.splice_acceptor_label is None):
            raise ValueError(
                "splice_donor_label and splice_acceptor_label must either both be set or both be omitted."
            )
        return self

    @model_validator(mode="after")
    def _validate_unique_labels(self) -> "LabelConfig":
        """All non-None label integers must be distinct."""
        seen: dict[int, str] = {}
        for field_name, value in self._configured_label_fields():
            if value is None:
                continue
            if value in seen:
                raise ValueError(
                    f"{field_name}={value} duplicates {seen[value]}={value}. All label integers must be unique."
                )
            seen[value] = field_name
        return self

    @property
    def supports_frameshift(self) -> bool:
        """Whether CDS-only frameshift evaluation is well-defined."""
        return (
            self.annotation_mode == AnnotationMode.UTR_CDS_INTRON
            and self.evaluation_scope == BenchmarkScope.CDS
            and self.cds_label is not None
        )

    def scope_tokens(self, scope: BenchmarkScope | str) -> frozenset[int]:
        """Return the positive-token set for one benchmark scope."""
        try:
            scope = BenchmarkScope(scope)
        except ValueError as exc:
            raise ValueError(f"Unknown benchmark scope: {scope}") from exc
        if scope == BenchmarkScope.TRANSCRIPT_EXON:
            if self.annotation_mode == AnnotationMode.EXON_INTRON:
                return frozenset((self.exon_label,))
            return frozenset(
                (
                    self.five_prime_utr_label,
                    self.cds_label,
                    self.three_prime_utr_label,
                )
            )
        if scope == BenchmarkScope.CDS:
            if self.annotation_mode != AnnotationMode.UTR_CDS_INTRON:
                raise ValueError(
                    f"Scope {scope.value!r} is not available for annotation_mode={self.annotation_mode.value}."
                )
            return frozenset((self.cds_label,))
        raise ValueError(f"Unknown benchmark scope: {scope}")

    def available_scopes(self) -> tuple[BenchmarkScope, ...]:
        """Return the metric scopes available for this label configuration."""
        if self.annotation_mode == AnnotationMode.UTR_CDS_INTRON:
            return (BenchmarkScope.TRANSCRIPT_EXON, BenchmarkScope.CDS)
        return (BenchmarkScope.TRANSCRIPT_EXON,)

    @property
    def labels(self) -> dict[int, str]:
        """Full ``{token: name}`` mapping for all defined labels."""
        return dict(self._configured_labels())

    @property
    def evaluation_labels(self) -> dict[int, str]:
        """Labels to compute per-class metrics for, excluding background."""
        return {
            token: name
            for token, name in self._configured_labels()
            if token != self.background_label
        }

    def name_of(self, token: int) -> str:
        """Return the human-readable name for *token*."""
        return self.labels.get(token, str(token))

    def _configured_label_fields(self) -> tuple[tuple[str, Optional[int]], ...]:
        """Return the configured label fields as ``(field_name, value)`` pairs."""
        return (
            ("background_label", self.background_label),
            ("exon_label", self.exon_label),
            ("cds_label", self.cds_label),
            ("five_prime_utr_label", self.five_prime_utr_label),
            ("three_prime_utr_label", self.three_prime_utr_label),
            ("intron_label", self.intron_label),
            ("splice_donor_label", self.splice_donor_label),
            ("splice_acceptor_label", self.splice_acceptor_label),
        )

    def _configured_labels(self) -> tuple[tuple[int, str], ...]:
        """Return the concrete ``(token, name)`` pairs for this config."""
        labels: list[tuple[int, str]] = [(self.background_label, "NONCODING")]
        if self.annotation_mode == AnnotationMode.EXON_INTRON:
            labels.append((self.exon_label, "EXON"))
        else:
            labels.extend(
                [
                    (self.five_prime_utr_label, "FIVE_PRIME_UTR"),
                    (self.cds_label, "CDS"),
                    (self.three_prime_utr_label, "THREE_PRIME_UTR"),
                ]
            )
        if self.intron_label is not None:
            labels.append((self.intron_label, "INTRON"))
        if self.splice_donor_label is not None:
            labels.append((self.splice_donor_label, "SPLICE_DONOR"))
        if self.splice_acceptor_label is not None:
            labels.append((self.splice_acceptor_label, "SPLICE_ACCEPTOR"))
        return tuple(labels)


BEND_LABEL_CONFIG = LabelConfig(
    annotation_mode=AnnotationMode.EXON_INTRON,
    background_label=8,
    exon_label=0,
    intron_label=2,
    splice_donor_label=1,
    splice_acceptor_label=3,
)


class EvalMetrics(Enum):
    """Available evaluation metric groups.

    Each value answers one clear question about prediction quality:

    * ``INDEL`` – *"What structural errors exist?"*
      5'/3' extensions/deletions, whole insertions/deletions, splits/joins.
    * ``REGION_DISCOVERY`` – *"Did we find the right regions?"*
      Two coherent detection P/R tiers (``neighborhood_hit``, ``perfect_boundary_hit``)
      plus conditional containment rates among matched pairs
      (``containment.internal_rate``, ``containment.full_coverage_rate``).
      Containment rates answer "when we detected a region, how often was the
      prediction spatially contained in / covering the GT?" — they are rates
      with a stated denominator, not precision/recall.
    * ``BOUNDARY_EXACTNESS`` – *"How precise are the boundaries?"*
      Scope-aware IoU stats, bias/reliability landscape, and boundary flags.
    * ``NUCLEOTIDE_CLASSIFICATION`` – *"Per-base, how accurate is it?"*
      Binary or multiclass outputs depending on annotation mode and scope.
    * ``PHASE_DRIFT`` – *"Is the CDS reading phase preserved?"*
      CDS-only per-position reading-phase deviation.  Measures relative
      CDS-base count drift between GT and prediction; it is a structural
      comparison signal, not a biological frameshift-mutation detector.
    * ``STRUCTURAL_COHERENCE`` – *"Is the segment chain correct as a whole?"*
      Scope-aware chain comparison, transcript classification, and segment
      count diagnostics.
    * ``DIAGNOSTIC_DEPTH`` – *"Why is the prediction structurally wrong?"*
      Scope-aware segment-length EMD and position-bias summaries.
    """

    INDEL = 0
    REGION_DISCOVERY = 1
    BOUNDARY_EXACTNESS = 2
    NUCLEOTIDE_CLASSIFICATION = 3
    PHASE_DRIFT = 4
    STRUCTURAL_COHERENCE = 5
    DIAGNOSTIC_DEPTH = 6


_DEFAULT_METRICS = [
    EvalMetrics.REGION_DISCOVERY,
    EvalMetrics.BOUNDARY_EXACTNESS,
    EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
]

_FULL_SWEEP_METRICS = [
    EvalMetrics.INDEL,
    EvalMetrics.REGION_DISCOVERY,
    EvalMetrics.BOUNDARY_EXACTNESS,
    EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
    EvalMetrics.PHASE_DRIFT,
    EvalMetrics.STRUCTURAL_COHERENCE,
    EvalMetrics.DIAGNOSTIC_DEPTH,
]


# ---------------------------------------------------------------------------
# Semantic boundary labels (INDEL metric)
# ---------------------------------------------------------------------------

#: Flank names that indicate a gene/sequence boundary (not an internal junction).
_TERMINAL_FLANKS: frozenset[str] = frozenset({"NONCODING", "none"})

#: Canonical display order for the four semantic exon-position categories.
SEMANTIC_BOUNDARY_ORDER: tuple[str, ...] = (
    "five_prime_terminal_exon",
    "internal_exon",
    "three_prime_terminal_exon",
    "single_exon_gene",
)


def semantic_boundary_label(left: str, right: str) -> str:
    """Map a GT flank pair to a biological exon-position category.

    A flank is *terminal* when it is ``"NONCODING"`` (gene/intergenic boundary)
    or ``"none"`` (sequence edge).  All other labels (``"INTRON"``, ``"EXON"``,
    ``"CDS"``, UTR labels, splice-site labels) are *internal*.

    Returns one of the four values in :data:`SEMANTIC_BOUNDARY_ORDER`.
    """
    left_terminal = left in _TERMINAL_FLANKS
    right_terminal = right in _TERMINAL_FLANKS
    if left_terminal and right_terminal:
        return "single_exon_gene"
    if left_terminal:
        return "five_prime_terminal_exon"
    if right_terminal:
        return "three_prime_terminal_exon"
    return "internal_exon"
