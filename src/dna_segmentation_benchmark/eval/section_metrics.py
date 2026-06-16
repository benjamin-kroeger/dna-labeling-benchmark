"""Section-overlap metrics: REGION_DISCOVERY and BOUNDARY_EXACTNESS.

Both metrics share a single sweep over contiguous coding sections
(:func:`_analyze_section_overlap_and_boundaries`).  Region discovery then runs
greedy 1:1 matching (:func:`_summarise_region_discovery`); boundary exactness
collects IoU/residual data and terminal-boundary flags from the same sweep.
"""

from __future__ import annotations

import numpy as np

from .intersection_over_union import _compute_intersection_over_union_score
from .statistics import Counts
from ..label_definition import EvalMetrics


def eval_sections(
        grouped_gt_sections: list[np.ndarray],
        grouped_pred_sections: list[np.ndarray],
        metrics: frozenset[EvalMetrics],
) -> tuple[dict | None, dict | None]:
    """Build REGION_DISCOVERY and/or BOUNDARY_EXACTNESS fragments.

    Only the section-overlap data required by the requested metrics is
    computed: region discovery needs the greedy 1:1 matching, boundary
    exactness needs the IoU/residual sweep.
    """
    need_region_discovery = EvalMetrics.REGION_DISCOVERY in metrics
    need_boundary_stats = EvalMetrics.BOUNDARY_EXACTNESS in metrics

    return _analyze_section_overlap_and_boundaries(
        grouped_gt_section_indices=grouped_gt_sections,
        grouped_pred_section_indices=grouped_pred_sections,
        need_region_discovery=need_region_discovery,
        need_boundary_stats=need_boundary_stats,
    )



def _analyze_section_overlap_and_boundaries(
        grouped_gt_section_indices: list[np.ndarray],
        grouped_pred_section_indices: list[np.ndarray],
        need_region_discovery: bool = True,
        need_boundary_stats: bool = True,
) -> tuple[dict | None, dict | None]:
    """Analyze overlap and boundary precision between section groups.

    ``need_region_discovery`` controls the greedy 1:1 matching and the
    strict-match sweep (the ``*_hit`` keys); ``need_boundary_stats`` controls
    the IoU/residual collection and terminal-boundary flags.  The returned
    dict only contains the keys for the branches that were requested.  Both
    default to ``True`` so direct callers get the full result.

    Uses ``np.searchsorted`` on sorted section endpoints to restrict the
    inner comparison to only the predicted sections that *can* overlap each
    GT section, bringing typical complexity from O(G × P) down to O(G + P).

    Region-discovery metrics (``neighborhood_hit``, ``internal_hit``,
    ``full_coverage_hit``) use greedy **1:1 matching** based on maximum
    overlap length so that each GT section is claimed by at most one
    prediction.  Predictions that cannot be matched because a better-
    fitting prediction already claimed the GT are counted as false
    positives.

    ``perfect_boundary_hit`` uses a per-prediction sweep: any prediction
    that exactly matches *some* GT section counts as a TP regardless of
    matching assignment.

    Two terminal-boundary flags are also collected:

    * ``first_sec_correct_3_prime_boundary`` — set to 1 iff *some*
      prediction's downstream end (``end`` in array coordinates) matches
      the **first** GT section's downstream end.  For multi-section
      transcripts this is the inner splice site of the leading exon, not
      the outer transcript start.
    * ``last_sec_correct_5_prime_boundary`` — set to 1 iff *some*
      prediction's upstream start matches the **last** GT section's
      upstream start.  For multi-section transcripts this is the inner
      splice site of the trailing exon, not the outer transcript end.

    Both flags use *array-orientation* semantics ("5'" = lower index,
    "3'" = higher index).  They do not account for biological strand —
    on the minus strand the array-5' end corresponds to the biological
    3' end of the transcript.

    The collected ``boundary_offsets`` are signed ``(left, right)`` tuples
    of ``pred_edge - gt_edge``: a positive offset means the prediction's
    edge lies to the right (3', higher index) of the matching GT edge, a
    negative offset to the left (5').  A systematically positive mean
    left-offset therefore indicates a consistent 3'-shift of predicted
    starts.  The IoU scores stored alongside are unsigned.
    """
    total_gt = len(grouped_gt_section_indices)
    total_pred = len(grouped_pred_section_indices)

    # Per-prediction flags for perfect-boundary (sweep-based, no 1:1 matching)
    gt_hit_strict = np.zeros(total_gt, dtype=bool)
    pred_hit_strict = np.zeros(total_pred, dtype=bool)

    boundary_offsets: list[tuple[int, int]] = []
    iou_scores: list[float] = []

    first_sec_correct_3_prime = 0
    last_sec_correct_5_prime = 0

    # Overlap candidates for 1:1 matching
    candidates: list[tuple[int, int, int]] = []  # (overlap_len, g_idx, p_idx)

    # ---- 1. Sweep — collect candidates & populate sweep-based metrics -----
    if total_gt > 0 and total_pred > 0:
        # Pre-compute (min, max) bounds arrays — sections are already sorted
        # by position because they come from contiguous-group splitting.
        pred_mins = np.array([s[0] for s in grouped_pred_section_indices])
        pred_maxs = np.array([s[-1] for s in grouped_pred_section_indices])

        # iterate over each ground truth section
        for g_idx, gt_section in enumerate(grouped_gt_section_indices):
            gt_min = int(gt_section[0])
            gt_max = int(gt_section[-1])

            # Find the range of pred sections that could overlap [gt_min, gt_max]:
            #   pred_min <= gt_max  →  candidates start from index 0 up to right_bound
            #   pred_max >= gt_min  →  candidates start from left_bound onward
            left_bound = np.searchsorted(pred_maxs, gt_min, side="left")
            right_bound = np.searchsorted(pred_mins, gt_max, side="right")

            # iterate over all possible pred sections which overlap the gt section
            for p_idx in range(left_bound, right_bound):
                p_min = int(pred_mins[p_idx])
                p_max = int(pred_maxs[p_idx])

                # --- A. Overlap (Any contact) ---
                if not (p_max < gt_min or p_min > gt_max):
                    if need_boundary_stats:
                        # Signed boundary offsets + IoU for every overlapping pair.
                        # Offset = pred_edge - gt_edge, so a positive value means the
                        # prediction's edge sits to the right (3') of the GT edge.
                        boundary_offsets.append((p_min - gt_min, p_max - gt_max))
                        iou_scores.append(
                            _compute_intersection_over_union_score(
                                gt_start=gt_min,
                                gt_end=gt_max,
                                pred_start=p_min,
                                pred_end=p_max,
                            )
                        )

                    if need_region_discovery:
                        # Collect candidate (overlap length + indices) for 1:1 matching
                        overlap_len = min(gt_max, p_max) - max(gt_min, p_min) + 1
                        candidates.append((overlap_len, g_idx, p_idx))

                # --- B. Strict Match (sweep-based, for perfect_boundary_hit) ---
                if need_region_discovery and p_min == gt_min and p_max == gt_max:
                    gt_hit_strict[g_idx] = True
                    pred_hit_strict[p_idx] = True

                if need_boundary_stats:
                    if g_idx == 0 and p_max == gt_max:
                        first_sec_correct_3_prime = 1
                    # For the last exon, the left boundary (min) is the 5' end.
                    if g_idx == total_gt - 1 and p_min == gt_min:
                        last_sec_correct_5_prime = 1

    region_data: dict | None = None
    if need_region_discovery:
        region_data = _summarise_region_discovery(
            candidates=candidates,
            grouped_gt_section_indices=grouped_gt_section_indices,
            grouped_pred_section_indices=grouped_pred_section_indices,
            total_gt=total_gt,
            total_pred=total_pred,
            gt_hit_strict=gt_hit_strict,
            pred_hit_strict=pred_hit_strict,
        )

    boundary_data: dict | None = None
    if need_boundary_stats:
        boundary_data = {
            "first_sec_correct_3_prime_boundary": first_sec_correct_3_prime,
            "last_sec_correct_5_prime_boundary": last_sec_correct_5_prime,
            "iou_scores": iou_scores,
            "fuzzy_metrics": {
                "boundary_offsets": boundary_offsets,
                "total_gt": total_gt,
            },
        }

    if region_data is None and boundary_data is None:
        raise ValueError("At least one of need_region_discovery or need_boundary_stats must be True.")

    return region_data, boundary_data


def _summarise_region_discovery(
        candidates: list[tuple[int, int, int]],
        grouped_gt_section_indices: list[np.ndarray],
        grouped_pred_section_indices: list[np.ndarray],
        total_gt: int,
        total_pred: int,
        gt_hit_strict: np.ndarray,
        pred_hit_strict: np.ndarray,
) -> dict:
    """Greedy 1:1 match the overlap candidates and classify discovery tiers.

    Candidates are claimed largest-overlap-first so each GT/pred section is
    paired with its best fit; unmatched predictions become false positives.
    ``perfect_boundary_hit`` is taken from the strict-match sweep instead and
    does not depend on the 1:1 assignment.

    Note on matching guarantees: this section-level matching is **greedy**,
    whereas the transcript-level assignment in
    :func:`~dna_segmentation_benchmark.transcript_mapping._assign_optimal_locus`
    is **optimal** (Hungarian).  The two layers therefore use different
    guarantees by design.  Within a single transcript window the candidate
    sections are few enough that greedy and optimal almost always coincide, so
    the simpler greedy pass is used here; the optimal solver is reserved for
    the multi-isoform transcript layer where it matters.
    """
    candidates.sort(key=lambda x: x[0], reverse=True)

    claimed_gt: set[int] = set()
    claimed_pred: set[int] = set()
    matches: list[tuple[int, int]] = []  # (g_idx, p_idx)

    for _overlap_len, g_idx, p_idx in candidates:
        # Best fits come first, so claim the GT/pred pair greedily.
        if g_idx not in claimed_gt and p_idx not in claimed_pred:
            claimed_gt.add(g_idx)
            claimed_pred.add(p_idx)
            matches.append((g_idx, p_idx))

    num_unmatched_pred = total_pred - len(matches)

    matched_neighborhood = 0  # All matches have overlap by definition
    matched_internal = 0  # Pred entirely inside GT
    matched_full_coverage = 0  # Pred fully covers GT

    for g_idx, p_idx in matches:
        gt_section = grouped_gt_section_indices[g_idx]
        pred_section = grouped_pred_section_indices[p_idx]
        gt_min = int(gt_section[0])
        gt_max = int(gt_section[-1])
        p_min = int(pred_section[0])
        p_max = int(pred_section[-1])

        # Every matched pair has overlap → neighborhood TP
        matched_neighborhood += 1

        # Internal / Envelop (prediction entirely INSIDE GT)
        if p_min > gt_min and p_max < gt_max:
            matched_internal += 1

        # Full Coverage / Encompass (prediction fully COVERS GT)
        if p_min < gt_min and p_max > gt_max:
            matched_full_coverage += 1

    return {
        # 1:1-matched detection contingency table
        "neighborhood_hit": Counts(
            tp=matched_neighborhood,
            fn=total_gt - matched_neighborhood,
            fp=num_unmatched_pred,
        ),
        # Sweep-based (no 1:1 matching) — handles fragmented predictions well
        "perfect_boundary_hit": Counts(
            tp=int(np.sum(gt_hit_strict)),
            fn=int(total_gt - np.sum(gt_hit_strict)),
            fp=int(total_pred - np.sum(pred_hit_strict)),
        ),
        # Conditional containment counts among matched pairs (not P/R).
        # internal: prediction is entirely inside GT boundary.
        # full_coverage: prediction fully covers GT boundary.
        # Denominatr is matched (= neighborhood TP); rates are computed at summarise time.
        "containment": {
            "matched": matched_neighborhood,
            "internal": matched_internal,
            "full_coverage": matched_full_coverage,
        },
    }
