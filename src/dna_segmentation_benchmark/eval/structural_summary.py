"""Per-label structural summary metrics.

Provides distribution-level and positional diagnostics that complement
the per-section metrics:

* **Length EMD** — Earth Mover's Distance between GT and predicted
  segment length distributions.  Quantifies whether the model produces
  segments of the right length.
* **Position bias** — match rates stratified by position within the
  array (5' / interior / 3'), revealing whether the model is weaker at
  sequence boundaries.
"""

from __future__ import annotations

from .structure import ExtractedStructure, Segment


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compute_structural_summary(
    gt_structure: ExtractedStructure,
    pred_structure: ExtractedStructure,
    class_token: int,
) -> dict:
    """Compute structural summary for a single class token.

    Parameters
    ----------
    gt_structure, pred_structure : ExtractedStructure
        Structures from the GT and predicted arrays.
    class_token : int
        Which label class to evaluate.

    Returns
    -------
    dict
        Keys: ``gt_segment_lengths``, ``pred_segment_lengths``,
        ``length_emd``, ``position_bias_5prime_match_rate``,
        ``position_bias_interior_match_rate``,
        ``position_bias_3prime_match_rate``.
    """
    gt_segs = gt_structure.filter_by_label(class_token)
    pred_segs = pred_structure.filter_by_label(class_token)

    gt_lengths = [s.length for s in gt_segs]
    pred_lengths = [s.length for s in pred_segs]

    # Length EMD
    emd = _wasserstein_distance(gt_lengths, pred_lengths)

    # Position bias
    bias = _compute_position_bias(
        gt_segs, pred_segs, gt_structure.length,
    )

    return {
        "gt_segment_lengths": gt_lengths,
        "pred_segment_lengths": pred_lengths,
        "length_emd": emd,
        **bias,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _wasserstein_distance(a: list[int], b: list[int]) -> float:
    """Compute 1-D Wasserstein (Earth Mover's) distance.

    Uses scipy if available, falls back to a pure-Python implementation.
    """
    if not a or not b:
        return 0.0

    try:
        from scipy.stats import wasserstein_distance
        return float(wasserstein_distance(a, b))
    except ImportError:
        # Pure-python fallback: sort both distributions and compute
        # the mean absolute difference of quantiles.
        sa = sorted(a)
        sb = sorted(b)
        # Resample to same length via linear interpolation
        n = max(len(sa), len(sb))
        import numpy as np
        qa = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sa)), sa)
        qb = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sb)), sb)
        return float(np.mean(np.abs(qa - qb)))


def _compute_position_bias(
    gt_segs: tuple[Segment, ...],
    pred_segs: tuple[Segment, ...],
    array_length: int,
) -> dict[str, float]:
    """Compute match rates by position zone (5' / interior / 3').

    A GT segment is considered "matched" if any pred segment overlaps it
    by at least 50%.

    Position zones:
    - 5': segment midpoint in first 25% of the array
    - Interior: segment midpoint in middle 50%
    - 3': segment midpoint in last 25%
    """
    if not gt_segs or array_length == 0:
        return {
            "position_bias_5prime_match_rate": 0.0,
            "position_bias_interior_match_rate": 0.0,
            "position_bias_3prime_match_rate": 0.0,
        }

    boundary_25 = array_length * 0.25
    boundary_75 = array_length * 0.75

    zone_counts = {"5prime": 0, "interior": 0, "3prime": 0}
    zone_matches = {"5prime": 0, "interior": 0, "3prime": 0}

    for gs in gt_segs:
        midpoint = (gs.start + gs.end) / 2.0

        if midpoint < boundary_25:
            zone = "5prime"
        elif midpoint > boundary_75:
            zone = "3prime"
        else:
            zone = "interior"

        zone_counts[zone] += 1

        # Check if any pred segment overlaps >= 50% of this GT segment
        for ps in pred_segs:
            overlap_start = max(gs.start, ps.start)
            overlap_end = min(gs.end, ps.end)
            if overlap_end >= overlap_start:
                overlap_len = overlap_end - overlap_start + 1
                if overlap_len >= gs.length * 0.5:
                    zone_matches[zone] += 1
                    break

    return {
        "position_bias_5prime_match_rate": (
            zone_matches["5prime"] / zone_counts["5prime"]
            if zone_counts["5prime"] > 0 else 0.0
        ),
        "position_bias_interior_match_rate": (
            zone_matches["interior"] / zone_counts["interior"]
            if zone_counts["interior"] > 0 else 0.0
        ),
        "position_bias_3prime_match_rate": (
            zone_matches["3prime"] / zone_counts["3prime"]
            if zone_counts["3prime"] > 0 else 0.0
        ),
    }
