"""Input preprocessing performed before any metric is computed.

Two concerns:

* **Scope resolution** — derive scope-specific positive masks from a full label
  array.
* **Intron inference** — relabel background gaps between adjacent transcribed-
  exon runs as introns (:func:`_infer_introns_from_coding_gaps`).
* **Mask splitting** — turn a boolean exclusion mask into the kept spans that
  are evaluated independently (:func:`_iter_unmasked_spans`).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import numpy as np

from .utils import get_contiguous_groups
from ..label_definition import BenchmarkScope, LabelConfig

# Arrays at or above this length are treated as possible chromosome-scale input.
# On such arrays, a coding-to-coding gap can be either a true intron or an
# intergenic region between unrelated transcripts, so inference becomes
# conservative and warning-backed.
_INFER_INTRONS_LARGE_ARRAY_WARNING_LENGTH = 1_000_000

# Fallback cutoff for large arrays when the gap-length distribution has no
# clear split: infer gaps up to this multiple of a typical short gap.
_INFER_INTRONS_LARGE_GAP_RATIO = 20

# Minimum multiplicative jump between consecutive sorted gap lengths required
# to treat the distribution as two-mode: short intron-like gaps followed by
# long intergenic-like gaps.
_INFER_INTRONS_BIMODAL_MIN_JUMP_RATIO = 5.0


def _infer_introns_from_coding_gaps(
    labels: np.ndarray,
    label_config: LabelConfig,
) -> np.ndarray:
    """Return labels with inferred introns between adjacent exonic segments.

    The function only rewrites positions that currently equal
    ``label_config.background_label``.  Existing coding, intron, splice-site,
    or other non-background labels are preserved.

    For transcript-sized arrays, every background gap between adjacent
    transcript-exon runs is interpreted as an intron.  This matches the usual
    representation produced by exon/CDS painting, where exons are explicit
    positive runs and introns are the unlabeled gaps between them.

    For large arrays (length ≥
    :data:`_INFER_INTRONS_LARGE_ARRAY_WARNING_LENGTH`, i.e. ``1_000_000`` bp),
    the function emits a warning and applies a conservative gap-length cutoff.
    Chromosome-sized arrays can contain both true introns and intergenic
    distances between separate transcripts; blindly filling every gap would turn
    intergenic regions into introns.  The cutoff is estimated from the
    coding-gap length distribution by :func:`_large_array_inferable_gap_cutoff`
    using two heuristic thresholds:

    * :data:`_INFER_INTRONS_BIMODAL_MIN_JUMP_RATIO` (``5.0``) — the minimum
      multiplicative jump between consecutive sorted gap lengths needed to treat
      the distribution as bimodal (short intron-like vs long intergenic-like).
      When such a split exists, gaps below it are inferred as introns.
    * :data:`_INFER_INTRONS_LARGE_GAP_RATIO` (``20``) — fallback when no clear
      bimodal split is found: infer gaps up to this multiple of a typical short
      gap.

    These thresholds are heuristics, not tuned constants.  On chromosome-scale
    input, inspect the returned array (or prefer transcript/window-level arrays
    or explicit intron labels) rather than trusting the inference blindly.

    Parameters
    ----------
    labels
        One-dimensional label array to transform.
    label_config
        Provides transcript-exon scope tokens, ``intron_label``, and
        ``background_label``.

    Returns
    -------
    np.ndarray
        A copy of *labels* with selected background gaps relabelled as introns.
    """
    is_large_input = len(labels) >= _INFER_INTRONS_LARGE_ARRAY_WARNING_LENGTH
    if is_large_input:
        warnings.warn(
            "infer_introns=True was requested for a large input array. "
            "Intron inference will estimate a gap-length cutoff from the "
            "coding-gap distribution: it first looks for a bimodal split "
            "between short intron-like gaps and long intergenic-like gaps, "
            "then falls back to a conservative typical-gap multiplier if no "
            "clear split is found. Prefer transcript/window-level arrays or "
            "explicit intron labels when possible.",
            stacklevel=2,
        )

    intron = label_config.intron_label
    if intron is None:
        warnings.warn(
            "infer_introns=True requires intron_label to be set; leaving labels unchanged.",
            stacklevel=2,
        )
        return labels.copy()

    background = label_config.background_label
    exon_indices = np.where(resolve_scope_mask(labels, BenchmarkScope.TRANSCRIPT_EXON, label_config))[0]
    if exon_indices.size < 2:
        return labels.copy()

    # Only need group boundaries, not full index arrays.
    breaks = np.flatnonzero(np.diff(exon_indices) != 1)
    gap_starts = exon_indices[breaks] + 1       # first bg position after each group
    gap_ends = exon_indices[breaks + 1]         # first exon position of next group (exclusive)
    gap_lengths = gap_ends - gap_starts

    positive = gap_lengths > 0
    gap_starts, gap_ends = gap_starts[positive], gap_ends[positive]

    if is_large_input:
        cutoff = _large_array_inferable_gap_cutoff((gap_ends - gap_starts).tolist())
        if cutoff is not None:
            keep = (gap_ends - gap_starts) <= cutoff
            gap_starts, gap_ends = gap_starts[keep], gap_ends[keep]

    if gap_starts.size == 0:
        return labels.copy()

    # Vectorized fill: mark every gap position, then relabel background in one shot.
    delta = np.zeros(len(labels) + 1, dtype=np.int8)
    delta[gap_starts] += 1
    delta[gap_ends] -= 1
    fill_mask = np.cumsum(delta[:-1]).astype(bool)

    inferred = labels.copy()
    inferred[fill_mask & (labels == background)] = intron
    return inferred


def resolve_scope_mask(
    labels: np.ndarray,
    scope: BenchmarkScope | str,
    label_config: LabelConfig,
) -> np.ndarray:
    """Return a boolean mask for all positions belonging to *scope*."""
    # Direct equality is 10-20× faster than np.isin for the 1-3 token sets used here.
    it = iter(label_config.scope_tokens(scope))
    mask = labels == next(it)
    for t in it:
        mask |= labels == t
    return mask


def resolve_scope_sections(
    labels: np.ndarray,
    scope: BenchmarkScope | str,
    label_config: LabelConfig,
) -> list[np.ndarray]:
    """Return contiguous index groups for the requested benchmark scope."""
    return get_contiguous_groups(np.where(resolve_scope_mask(labels, scope, label_config))[0])


def _large_array_inferable_gap_cutoff(gap_lengths: list[int]) -> int | None:
    """Estimate the largest gap length that should be inferred as intronic.

    This helper is used only for large input arrays, where coding gaps may be a
    mixture of:

    * short, intron-like gaps between exons of the same transcript
    * long, intergenic-like gaps between different transcripts

    The preferred path assumes this mixture is approximately bimodal.  It sorts
    all coding-to-coding gap lengths and finds the largest multiplicative jump
    between adjacent values.  If the jump is at least
    :data:`_INFER_INTRONS_BIMODAL_MIN_JUMP_RATIO`, the split is treated as the
    valley before the second mode.  The cutoff is set to the geometric midpoint
    between the last short gap and first long gap, i.e. a value between the two
    modes rather than inside either mode.

    If no clear jump is found, the function falls back to a conservative rule:
    infer only gaps up to
    ``median(lower_half_of_gap_lengths) * _INFER_INTRONS_LARGE_GAP_RATIO``.
    This preserves the old ``20x typical short gap`` behavior for unimodal or
    noisy distributions.

    Parameters
    ----------
    gap_lengths
        Positive lengths of background gaps between adjacent coding runs.

    Returns
    -------
    int | None
        Maximum gap length to fill as intron.  ``None`` means no cutoff could
        be estimated, so all gaps remain eligible.
    """
    if len(gap_lengths) < 2:
        return None

    sorted_gaps = np.array(sorted(gap_lengths), dtype=float)
    adjacent_ratios = sorted_gaps[1:] / np.maximum(sorted_gaps[:-1], 1.0)
    largest_jump_idx = int(np.argmax(adjacent_ratios))
    largest_jump = float(adjacent_ratios[largest_jump_idx])
    if largest_jump >= _INFER_INTRONS_BIMODAL_MIN_JUMP_RATIO:
        first_long_gap = sorted_gaps[largest_jump_idx + 1]
        last_short_gap = sorted_gaps[largest_jump_idx]
        return int(np.floor(np.sqrt(last_short_gap * first_long_gap)))

    lower_half = sorted_gaps[: max(1, len(sorted_gaps) // 2)]
    typical_gap = float(np.median(lower_half))
    if typical_gap <= 0:
        return None

    return int(typical_gap * _INFER_INTRONS_LARGE_GAP_RATIO)


def _iter_unmasked_spans(mask_labels: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` half-open spans of kept (unmasked) positions.

    ``mask_labels`` is True where a position must be excluded.  Each yielded
    span is a maximal run of kept positions and is evaluated independently.
    """
    is_valid = ~mask_labels.astype(bool)
    padded = np.pad(is_valid, (1, 1), mode="constant", constant_values=False)
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0]
    for s, e in zip(starts, ends):
        if e > s:
            yield int(s), int(e)
