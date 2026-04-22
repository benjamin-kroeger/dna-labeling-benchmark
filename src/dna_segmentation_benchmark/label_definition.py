"""Label configuration for the DNA segmentation benchmark.

This module provides :class:`LabelConfig`, a Pydantic model that is the single
source of truth for what integer tokens mean in the benchmark.

Each semantic role is a named field rather than an entry in a generic dict.
This makes the YAML config self-documenting and keeps label semantics separate
from parser choices such as which GFF/GTF feature types should be read as exon
intervals.  Those parser choices belong to the GFF pipeline API.

Backward-compatible properties (``coding_label``, ``labels``) are provided so
that internal modules (``state_transitions``, plotting, ``io_utils``, …) require
no changes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator


class LabelConfig(BaseModel):
    """Complete label definition for a benchmarking run.

    Required fields
    ---------------
    background_label : int
        Token for non-coding / intergenic / background regions.
        Used as the sentinel value for padding and null arrays.
    exon_label : int
        Token for exon / CDS nucleotides.

    Optional semantic roles
    -----------------------
    intron_label : int | None
        Token for intronic nucleotides.
    splice_donor_label : int | None
        Token for the 5' donor splice site.
    splice_acceptor_label : int | None
        Token for the 3' acceptor splice site.

    Examples
    --------
    >>> # Helixer / SegmentNT — uses "exon" features
    >>> config = LabelConfig(
    ...     background_label=8,
    ...     exon_label=0,
    ...     intron_label=2,
    ...     splice_donor_label=1,
    ...     splice_acceptor_label=3,
    ... )

    GFF/GTF feature names such as ``"exon"`` or ``"CDS"`` are passed to
    ``benchmark_from_gff`` / ``map_transcripts`` separately.
    """

    model_config = {"frozen": True}

    background_label: int
    exon_label: int
    intron_label: Optional[int] = None
    splice_donor_label: Optional[int] = None
    splice_acceptor_label: Optional[int] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_unique_labels(self) -> "LabelConfig":
        """All non-None label integers must be distinct."""
        candidates = [
            ("background_label", self.background_label),
            ("exon_label", self.exon_label),
            ("intron_label", self.intron_label),
            ("splice_donor_label", self.splice_donor_label),
            ("splice_acceptor_label", self.splice_acceptor_label),
        ]
        seen: dict[int, str] = {}
        for field_name, value in candidates:
            if value is None:
                continue
            if value in seen:
                raise ValueError(
                    f"{field_name}={value} duplicates {seen[value]}={value}. All label integers must be unique."
                )
            seen[value] = field_name
        return self

    # ------------------------------------------------------------------
    # Derived properties — allow callers that use the old API to work unchanged
    # ------------------------------------------------------------------

    @property
    def coding_label(self) -> int:
        """Alias for ``exon_label``.

        Used by ``transcript_mapping``, ``global_metrics``, and ``io_utils``
        which were written against the old ``coding_label`` field name.
        """
        return self.exon_label

    @property
    def labels(self) -> dict[int, str]:
        """Full ``{token: name}`` dict for all defined labels.

        Provides backward compatibility for ``state_transitions.py`` and
        the transition plotting code which iterate over all label IDs.
        """
        result: dict[int, str] = {
            self.background_label: "NONCODING",
            self.exon_label: "EXON",
        }
        if self.intron_label is not None:
            result[self.intron_label] = "INTRON"
        if self.splice_donor_label is not None:
            result[self.splice_donor_label] = "SPLICE_DONOR"
        if self.splice_acceptor_label is not None:
            result[self.splice_acceptor_label] = "SPLICE_ACCEPTOR"
        return result

    @property
    def evaluation_labels(self) -> dict[int, str]:
        """Labels to compute per-class metrics for (excludes background).

        Used by :func:`~dna_segmentation_benchmark.eval.evaluate_predictors.\
benchmark_gt_vs_pred_single` when ``classes`` is not specified.
        """
        result: dict[int, str] = {self.exon_label: "EXON"}
        if self.intron_label is not None:
            result[self.intron_label] = "INTRON"
        if self.splice_donor_label is not None:
            result[self.splice_donor_label] = "SPLICE_DONOR"
        if self.splice_acceptor_label is not None:
            result[self.splice_acceptor_label] = "SPLICE_ACCEPTOR"
        return result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def name_of(self, token: int) -> str:
        """Return the human-readable name for *token*.

        Falls back to ``str(token)`` for unknown tokens rather than raising.
        """
        return self.labels.get(token, str(token))

    @property
    def background_name(self) -> str:
        """Human-readable name of the background label."""
        return "NONCODING"

    @property
    def coding_name(self) -> str:
        """Human-readable name of the exon/coding label."""
        return "EXON"

    @property
    def intron_name(self) -> Optional[str]:
        """Human-readable name of the intron label, or ``None``."""
        return "INTRON" if self.intron_label is not None else None


# ------------------------------------------------------------------
# Pre-built configs for common label sets
# ------------------------------------------------------------------

BEND_LABEL_CONFIG = LabelConfig(
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
      Precision & recall at four overlap strictness levels
      (neighborhood, internal, full-coverage, perfect-boundary).
    * ``BOUNDARY_EXACTNESS`` – *"How precise are the boundaries?"*
      IoU stats, bias/reliability landscape, inner/all section boundary
      precision & recall, terminal-boundary flags.
    * ``NUCLEOTIDE_CLASSIFICATION`` – *"Per-base, how accurate is it?"*
      Precision, recall, and F1 from the nucleotide confusion matrix.
    * ``FRAMESHIFT`` – *"Is the reading frame preserved?"*
      Per-position reading-frame deviation.
    * ``STRUCTURAL_COHERENCE`` – *"Is the segment chain correct as a whole?"*
      Strict binary ``intron_chain`` precision/recall (gffcompare-style),
      per-transcript soft exon recall and hallucinated-exon count
      (continuous distribution view), holistic transcript match
      classification, transcript-level P/R tiers, and segment count
      delta.
    * ``DIAGNOSTIC_DEPTH`` – *"Why is the prediction structurally wrong?"*
      Segment-length EMD and position-bias histogram localising the
      errors along the coding span.
    """

    INDEL = 0
    REGION_DISCOVERY = 1
    BOUNDARY_EXACTNESS = 2
    NUCLEOTIDE_CLASSIFICATION = 3
    FRAMESHIFT = 4
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
    EvalMetrics.FRAMESHIFT,
    EvalMetrics.STRUCTURAL_COHERENCE,
    EvalMetrics.DIAGNOSTIC_DEPTH,
]
