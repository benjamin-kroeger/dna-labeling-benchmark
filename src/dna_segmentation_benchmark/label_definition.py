"""Label configuration for the DNA segmentation benchmark.

This module provides :class:`LabelConfig`, a Pydantic model that serves as the
single source of truth for what integer tokens mean in the benchmark.  No enums
are used – labels are defined as a plain ``dict[int, str]`` mapping token
values to human-readable names.

Specialised metrics declare their data requirements as optional fields on
``LabelConfig``.  Validators ensure the required tokens are present before a
confusing runtime error can occur.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator


class LabelConfig(BaseModel):
    """Complete label definition for a benchmarking run.

    Attributes
    ----------
    labels : dict[int, str]
        Mapping of every token integer to a human-readable name.
        Example: ``{0: "EXON", 1: "DONOR", 2: "INTRON", 3: "ACCEPTOR", 8: "NONCODING"}``
    background_label : int
        Token used for non-coding / intergenic / background regions.
        Used as the sentinel value for boundary padding.
    coding_label : int | None
        Token that represents coding (exon / CDS) nucleotides.
        Required only when the ``FRAMESHIFT`` metric is requested.
    splice_donor_label : int | None
        Token for the donor splice-site.
        Reserved for future splice-junction metrics.
    splice_acceptor_label : int | None
        Token for the acceptor splice-site.
        Reserved for future splice-junction metrics.

    Examples
    --------
    >>> config = LabelConfig(
    ...     labels={0: "EXON", 2: "INTRON", 8: "NONCODING"},
    ...     background_label=8,
    ...     coding_label=0,
    ... )
    >>> config.background_name
    'NONCODING'
    """

    model_config = {"frozen": True}

    labels: dict[int, str]
    background_label: int
    coding_label: Optional[int] = None
    splice_donor_label: Optional[int] = None
    splice_acceptor_label: Optional[int] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _validate_special_labels_exist_in_labels(self) -> "LabelConfig":
        """Ensure every declared role token actually exists in ``labels``."""
        for field_name in ("background_label", "coding_label",
                           "splice_donor_label", "splice_acceptor_label"):
            value = getattr(self, field_name)
            if value is not None and value not in self.labels:
                raise ValueError(
                    f"{field_name}={value} is not present in `labels`. "
                    f"Available tokens: {list(self.labels.keys())}"
                )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def name_of(self, token: int) -> str:
        """Return the human-readable name for *token*.

        Raises ``KeyError`` if *token* is not in ``labels``.
        """
        return self.labels[token]

    @property
    def background_name(self) -> str:
        """Human-readable name of the background label."""
        return self.labels[self.background_label]

    @property
    def coding_name(self) -> Optional[str]:
        """Human-readable name of the coding label, or ``None``."""
        if self.coding_label is None:
            return None
        return self.labels[self.coding_label]


# ------------------------------------------------------------------
# Pre-built configs for common label sets
# ------------------------------------------------------------------

BEND_LABEL_CONFIG = LabelConfig(
    labels={0: "EXON", 1: "DONOR", 2: "INTRON", 3: "ACCEPTOR", 8: "NONCODING"},
    background_label=8,
    coding_label=0,
    splice_donor_label=1,
    splice_acceptor_label=3,
)


class EvalMetrics(Enum):
    """Available evaluation metric groups.

    Each value answers one clear question about prediction quality:

    * ``INDEL`` – *"What structural errors exist?"*
      5'/3' extensions/deletions, whole insertions/deletions, splits/joins.
    * ``REGION_DISCOVERY`` – *"Did we find the right regions?"*
      Precision & recall at four overlap strictness levels.
    * ``BOUNDARY_EXACTNESS`` – *"How precise are the boundaries?"*
      IoU stats, bias/reliability landscape, inner/all section boundary
      precision & recall, terminal-boundary flags.
    * ``NUCLEOTIDE_CLASSIFICATION`` – *"Per-base, how accurate is it?"*
      Precision, recall, and F1 from the nucleotide confusion matrix.
    * ``FRAMESHIFT`` – *"Is the reading frame preserved?"*
      Per-position reading-frame deviation.
    """

    INDEL = 0
    REGION_DISCOVERY = 1
    BOUNDARY_EXACTNESS = 2
    NUCLEOTIDE_CLASSIFICATION = 3
    FRAMESHIFT = 4


_DEFAULT_METRICS = [
    EvalMetrics.REGION_DISCOVERY,
    EvalMetrics.BOUNDARY_EXACTNESS,
    EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
]