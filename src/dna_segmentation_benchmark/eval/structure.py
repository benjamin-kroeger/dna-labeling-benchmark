"""Structure extraction from 1-D label arrays.

Converts a flat nucleotide-level label array into an ordered sequence of
:class:`Segment` objects, providing the foundation for transcript-level
structural metrics.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from ..label_definition import LabelConfig


@dataclasses.dataclass(frozen=True)
class Segment:
    """A contiguous run of a single label in a 1-D array.

    Attributes
    ----------
    label : int
        The integer token for this segment.
    start : int
        Start index (inclusive, 0-based).
    end : int
        End index (inclusive, 0-based).
    """

    label: int
    start: int
    end: int

    @property
    def length(self) -> int:
        """Number of positions spanned by this segment."""
        return self.end - self.start + 1


@dataclasses.dataclass(frozen=True)
class ExtractedStructure:
    """Ordered sequence of segments extracted from a label array.

    Attributes
    ----------
    segments : tuple[Segment, ...]
        Ordered, non-overlapping segments covering the array.
    length : int
        Total length of the source array.
    """

    segments: tuple[Segment, ...]
    length: int

    def filter_by_label(self, label: int) -> tuple[Segment, ...]:
        """Return only segments matching *label*, preserving order."""
        return tuple(s for s in self.segments if s.label == label)

    @property
    def label_sequence(self) -> tuple[int, ...]:
        """Ordered tuple of labels (one per segment)."""
        return tuple(s.label for s in self.segments)

    @property
    def segment_count(self) -> int:
        return len(self.segments)


def extract_structure(
    labels: np.ndarray,
    label_config: Optional[LabelConfig] = None,
    exclude_background: bool = True,
) -> ExtractedStructure:
    """Extract an ordered segment chain from a 1-D integer label array.

    Parameters
    ----------
    labels : np.ndarray
        1-D array of integer tokens.
    label_config : LabelConfig, optional
        If provided and *exclude_background* is ``True``, segments whose
        label equals ``label_config.background_label`` are dropped.
    exclude_background : bool
        Whether to filter out background segments (default ``True``).

    Returns
    -------
    ExtractedStructure
        Immutable structure containing the ordered segment chain.
    """
    if labels.ndim != 1:
        raise ValueError(f"Expected 1-D array, got shape {labels.shape}")

    n = len(labels)
    if n == 0:
        return ExtractedStructure(segments=(), length=0)

    # Find positions where the label changes
    change_points = np.where(np.diff(labels) != 0)[0] + 1
    # Build start/end pairs: starts are [0, cp1, cp2, ...], ends are [cp1-1, cp2-1, ..., n-1]
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points - 1, [n - 1]))

    background = label_config.background_label if label_config is not None else None

    segments = []
    for s, e in zip(starts, ends):
        lbl = int(labels[s])
        if exclude_background and lbl == background:
            continue
        segments.append(Segment(label=lbl, start=int(s), end=int(e)))

    return ExtractedStructure(segments=tuple(segments), length=n)
