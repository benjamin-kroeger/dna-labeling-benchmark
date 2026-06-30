"""INDEL metric: classify GT/prediction coding mismatches, typed by boundary.

Insertion and deletion runs (coding present on one row but not the other) are
sorted into 5'/3' extensions-or-deletions, whole insertions/deletions, and
join/split events.  Each run additionally keeps the **GT label pair flanking the
run** so that, e.g., a 5'-extension at a UTR→CDS boundary is distinguished from
one at an intron→CDS boundary.

Orientation assumption
----------------------
All input is assumed to be presented **5'→3'**: lower array index = 5', higher
index = 3'.  Boundary tuples are therefore in array order, so ``(UTR, CDS)``
(5'UTR / start-codon boundary) and ``(CDS, UTR)`` (3'UTR / stop-codon boundary)
are kept distinct.  Re-orienting minus-strand input is the caller's
responsibility.

Output shape
------------
``_eval_indel`` returns ``{"by_boundary": {"LEFT:RIGHT": {bucket: [lengths]}}}``
where ``"LEFT:RIGHT"`` is the ``"<5'-flank>:<3'-flank>"`` GT label-name pair
(e.g. ``"FIVE_PRIME_UTR:CDS"``), :data:`_NO_NEIGHBOUR` (``"none"``) standing in
for a missing neighbour at a sequence end, and each ``lengths`` entry is a run
length in nucleotides.  Only run *lengths* are stored — never index arrays —
because that is all downstream plotting needs and it keeps the per-boundary
fan-out small.

The label-name keys are produced here, at the one layer where ``label_config``
is available, so they are the single canonical form for tests, plotting, the
accumulator merge, and JSON serialisation alike (``json.dumps`` cannot key a
dict by a tuple).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..label_definition import LabelConfig, semantic_boundary_label
from .utils import get_contiguous_groups

#: Bucket names for the four classification slots, for insertion runs.
#: Order: (5'-anchored, 3'-anchored, anchored-on-both, anchored-on-neither).
_INSERTION_BUCKETS = ("5_prime_extensions", "3_prime_extensions", "joined", "whole_insertions")
#: Same four slots, for deletion runs.
_DELETION_BUCKETS = ("5_prime_deletions", "3_prime_deletions", "split", "whole_deletions")
#: The four bucket names that are anchored at a coding-segment boundary (5′/3′
#: extensions and deletions).  Derived from the first two slots of each tuple.
BOUNDARY_ANCHORED_BUCKETS: frozenset[str] = frozenset(_INSERTION_BUCKETS[:2] + _DELETION_BUCKETS[:2])

#: Boundary-key token for "no neighbour" (run touches a sequence end).  A name,
#: not an integer id, so it can never collide with a real label.
_NO_NEIGHBOUR = "none"


def _build_segment_type_array(
    gt_positive_mask: np.ndarray, gt_labels: np.ndarray, label_config: LabelConfig
) -> np.ndarray:
    """Return an object array mapping each coding position to its segment's semantic type.

    Non-coding positions are left as ``None``.  Used so boundary-anchored event
    classification can look up the adjacent segment type in O(1) without rescanning.
    """
    out: np.ndarray = np.empty(len(gt_positive_mask), dtype=object)
    for segment in get_contiguous_groups(np.where(gt_positive_mask)[0]):
        if segment.size == 0:
            continue
        start, end = int(segment[0]), int(segment[-1])
        left_outer = _flank_name(gt_labels, start - 1, label_config)
        right_outer = _flank_name(gt_labels, end + 1, label_config)
        out[start : end + 1] = semantic_boundary_label(left_outer, right_outer)
    return out


def eval_indel(
    grouped_insertions: list[np.ndarray],
    grouped_deletions: list[np.ndarray],
    gt_positive_mask: np.ndarray,
    pred_positive_mask: np.ndarray,
    label_config: LabelConfig,
    gt_labels: np.ndarray,
    n_gt_segments: int,
    n_pred_segments: int,
) -> dict:
    """Sort insertion/deletion runs into per-boundary 5'/3'/whole/join-split buckets.

    Parameters
    ----------
    grouped_insertions, grouped_deletions : list[np.ndarray]
        Contiguous index runs (unpadded coordinates) where exactly one of GT /
        pred is coding.
    gt_positive_mask, pred_positive_mask : np.ndarray
        Boolean coding masks for the active evaluation scope.
    label_config : LabelConfig
        Maps integer flank labels to names (boundary keys, junction keys).
    gt_labels : np.ndarray
        The *unpadded* full GT label row.  Used to read the GT label on each
        side of a mismatch run, which types the boundary the run straddles.
    n_gt_segments, n_pred_segments : int
        Number of GT / predicted coding segments in the active scope.  Serve as
        the *opportunity* denominators for segment-level events (split / whole
        deletions use ``n_gt_segments``; whole insertions use
        ``n_pred_segments``) so downstream consumers can turn counts into rates.

    Returns
    -------
    dict
        ``{"by_boundary": {"LEFT:RIGHT": {bucket: [length, ...]}},
        "junction_opportunities": {"LEFT:RIGHT": int},
        "n_gt_segments": int, "n_pred_segments": int}``.

        ``junction_opportunities`` counts the 5'/3' edges of GT coding segments,
        typed by their outer GT flank (array edge = ``"none"`` = terminal), and is
        the denominator for boundary-anchored events (extensions / deletions) of
        the matching key, so a count becomes "fraction of that boundary type that
        suffered this error".
    """
    # _classify_mismatches looks one position before/after each group in the
    # *mask*, so pad the masks with one background sentinel on each side for safe
    # access.  The GT-label flank lookup uses the unpadded ``gt_labels`` instead.
    padded_gt = np.concatenate(([False], gt_positive_mask.astype(bool), [False]))
    padded_pred = np.concatenate(([False], pred_positive_mask.astype(bool), [False]))
    padded_arr = np.stack((padded_gt, padded_pred), axis=0)

    # Shift indices by +1 to match the padded array layout
    padded_insertions = [g + 1 for g in grouped_insertions]
    padded_deletions = [g + 1 for g in grouped_deletions]

    # Pre-compute segment type per coding position so boundary-anchored event
    # classification can look up the adjacent segment in O(1) — avoids keying
    # events by the immediate run flank (which always has the coding label on one
    # side and can never produce ``single_exon_gene`` for anchored events).
    seg_type_arr = _build_segment_type_array(gt_positive_mask, gt_labels, label_config)

    by_boundary: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    _classify_mismatches(
        padded_insertions, padded_arr, gt_labels, label_config, _INSERTION_BUCKETS, by_boundary, is_insertion=True,
        seg_type_arr=seg_type_arr,
    )
    _classify_mismatches(
        padded_deletions, padded_arr, gt_labels, label_config, _DELETION_BUCKETS, by_boundary, is_insertion=False,
        seg_type_arr=seg_type_arr,
    )

    return {
        "by_boundary": {boundary: dict(buckets) for boundary, buckets in by_boundary.items()},
        "junction_opportunities": _junction_opportunities(gt_positive_mask, gt_labels, label_config, seg_type_arr),
        "n_gt_segments": int(n_gt_segments),
        "n_pred_segments": int(n_pred_segments),
    }


def _junction_opportunities(
    gt_positive_mask: np.ndarray,
    gt_labels: np.ndarray,
    label_config: LabelConfig,
    seg_type_arr: np.ndarray,
) -> dict[str, int]:
    """Count per-boundary opportunities for boundary-anchored slips.

    Single-exon genes (both outer flanks terminal) contribute 2 opportunities to
    the ``single_exon_gene`` bucket (one per junction edge), consistent with the
    numerator events that also key to ``single_exon_gene`` via ``seg_type_arr``.

    All other segments use the original per-junction key: each junction is typed
    by the GT flank just *outside* the segment on that side.  This preserves
    existing per-junction rates for multi-exon genes while routing single-exon
    gene junctions to their own denominator bucket.
    """
    opportunities: dict[str, int] = {}
    for segment in get_contiguous_groups(np.where(gt_positive_mask)[0]):
        if segment.size == 0:
            continue
        start = int(segment[0])
        end = int(segment[-1])
        if str(seg_type_arr[start]) == "single_exon_gene":
            opportunities["single_exon_gene"] = opportunities.get("single_exon_gene", 0) + 2
        else:
            start_name = label_config.name_of(int(gt_labels[start]))
            end_name = label_config.name_of(int(gt_labels[end]))
            key_5 = semantic_boundary_label(_flank_name(gt_labels, start - 1, label_config), start_name)
            key_3 = semantic_boundary_label(end_name, _flank_name(gt_labels, end + 1, label_config))
            opportunities[key_5] = opportunities.get(key_5, 0) + 1
            opportunities[key_3] = opportunities.get(key_3, 0) + 1
    return opportunities


def _flank_name(gt_labels: np.ndarray, idx: int, label_config: LabelConfig) -> str:
    """Return the GT label *name* at original-coordinate *idx*.

    Returns :data:`_NO_NEIGHBOUR` (``"none"``) when *idx* is out of bounds, i.e.
    the run touches a sequence end.  ``"none"`` is a name (not an integer id) so
    it can never collide with a real background/feature label.
    """
    if idx < 0 or idx >= len(gt_labels):
        return _NO_NEIGHBOUR
    return label_config.name_of(int(gt_labels[idx]))


def _outer_flank_name(
    gt_labels: np.ndarray, idx: int, label_config: LabelConfig, seg_type_arr: np.ndarray
) -> str:
    """Like _flank_name but treats a single-exon-gene coding position as terminal.

    When the outer flank of a whole/join/split run lands inside a single-exon-gene
    segment, the coding label (EXON/CDS) would otherwise give ``internal_exon``
    via ``semantic_boundary_label``.  That is wrong — the run sits in the
    inter-gene gap, not inside an intron — so we return ``_NO_NEIGHBOUR`` instead.
    """
    if idx < 0 or idx >= len(gt_labels):
        return _NO_NEIGHBOUR
    if seg_type_arr[idx] is not None and str(seg_type_arr[idx]) == "single_exon_gene":
        return _NO_NEIGHBOUR
    return label_config.name_of(int(gt_labels[idx]))


def _classify_mismatches(
    grouped_indices: list[np.ndarray],
    gt_pred_arr: np.ndarray,
    gt_labels: np.ndarray,
    label_config: LabelConfig,
    bucket_names: tuple[str, str, str, str],
    out: dict[str, dict[str, list[int]]],
    is_insertion: bool,
    seg_type_arr: np.ndarray,
) -> None:
    """Sort contiguous mismatch groups into four buckets, keyed by GT segment type.

    The four ``bucket_names`` slots correspond, in order, to:

    * 5'-extensions / 5'-deletions   (run anchored only on its 3' side)
    * 3'-extensions / 3'-deletions   (run anchored only on its 5' side)
    * joins / splits                 (run anchored on both sides)
    * whole insertions / whole deletions (run anchored on neither side)

    **Boundary key** for boundary-anchored events (5'/3' buckets): the semantic
    type of the adjacent GT coding segment, read from ``seg_type_arr`` at the
    coding position immediately adjacent to the run.  This ensures single-exon
    genes key as ``single_exon_gene`` even though one immediate flank of the run
    is always the coding label — previously the immediate-flank approach could
    never produce ``single_exon_gene`` for anchored events.

    For joined / split / whole events the boundary key uses the outer flanks of
    the mismatch run (both sides non-coding), which cannot produce this ambiguity.
    """
    name_5_prime, name_3_prime, name_both, name_neither = bucket_names

    for mismatch in grouped_indices:
        if mismatch.size == 0:
            continue

        first_idx = mismatch[0]
        last_idx = mismatch[-1]

        # Anchor test uses the *padded mask*: a neighbour anchors the run only if
        # both GT and pred are coding there.
        target_on_3_prime = bool(gt_pred_arr[0, last_idx + 1]) and bool(gt_pred_arr[1, last_idx + 1])
        target_on_5_prime = bool(gt_pred_arr[0, first_idx - 1]) and bool(gt_pred_arr[1, first_idx - 1])

        # Back to unpadded coordinates for the GT-label flank lookup.
        adjusted = mismatch - 1

        if target_on_3_prime and target_on_5_prime:
            bucket = name_both
        elif target_on_3_prime:
            bucket = name_5_prime
        elif target_on_5_prime:
            bucket = name_3_prime
        else:
            bucket = name_neither

        if bucket == name_5_prime:
            # Adjacent segment starts/continues at adjusted[-1]+1 (unpadded).
            probe = int(adjusted[-1]) + 1
            if str(seg_type_arr[probe]) == "single_exon_gene":
                boundary_key = "single_exon_gene"
            elif is_insertion:
                # Inner-edge key: label at last run position, then first coding position.
                # Avoids mis-keying multi-base insertions that bridge to a far segment.
                boundary_key = semantic_boundary_label(
                    _flank_name(gt_labels, probe - 1, label_config),
                    _flank_name(gt_labels, probe, label_config),
                )
            else:
                boundary_key = semantic_boundary_label(
                    _flank_name(gt_labels, int(adjusted[0]) - 1, label_config),
                    _flank_name(gt_labels, probe, label_config),
                )
        elif bucket == name_3_prime:
            # Adjacent segment ends/continues at adjusted[0]-1 (unpadded).
            probe = int(adjusted[0]) - 1
            if str(seg_type_arr[probe]) == "single_exon_gene":
                boundary_key = "single_exon_gene"
            elif is_insertion:
                boundary_key = semantic_boundary_label(
                    _flank_name(gt_labels, probe, label_config),
                    _flank_name(gt_labels, probe + 1, label_config),
                )
            else:
                boundary_key = semantic_boundary_label(
                    _flank_name(gt_labels, probe, label_config),
                    _flank_name(gt_labels, int(adjusted[-1]) + 1, label_config),
                )
        else:
            # Joins, splits, whole events: probe both outer positions.  Use
            # _outer_flank_name so that a coding position in a single-exon-gene
            # segment is treated as terminal (inter-gene gap, not an intron).
            boundary_key = semantic_boundary_label(
                _outer_flank_name(gt_labels, int(adjusted[0]) - 1, label_config, seg_type_arr),
                _outer_flank_name(gt_labels, int(adjusted[-1]) + 1, label_config, seg_type_arr),
            )

        out[boundary_key][bucket].append(int(adjusted.size))
