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


def _eval_indel(
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

    by_boundary: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    _classify_mismatches(padded_insertions, padded_arr, gt_labels, label_config, _INSERTION_BUCKETS, by_boundary)
    _classify_mismatches(padded_deletions, padded_arr, gt_labels, label_config, _DELETION_BUCKETS, by_boundary)

    return {
        "by_boundary": {boundary: dict(buckets) for boundary, buckets in by_boundary.items()},
        "junction_opportunities": _junction_opportunities(gt_positive_mask, gt_labels, label_config),
        "n_gt_segments": int(n_gt_segments),
        "n_pred_segments": int(n_pred_segments),
    }


def _junction_opportunities(
    gt_positive_mask: np.ndarray, gt_labels: np.ndarray, label_config: LabelConfig
) -> dict[str, int]:
    """Count per-boundary opportunities for boundary-anchored slips.

    Each GT coding segment contributes two opportunities — its 5' edge and its
    3' edge — typed by the GT flank just *outside* the segment, read with the
    same :func:`_flank_name` rule the mismatch classifier uses.  A flank off the
    array end is therefore ``"none"`` (terminal), so a coding segment touching a
    sequence/window edge is counted as a terminal-exon boundary, not an absent
    one.  This keeps the denominator consistent with the numerator: under
    per-transcript windowing the gene boundary coincides with the array edge,
    which a plain label-transition count would miss entirely.

    Keys match those produced for boundary-anchored runs, so a run count divided
    by its opportunity is the per-junction error rate.
    """
    opportunities: dict[str, int] = {}
    for segment in get_contiguous_groups(np.where(gt_positive_mask)[0]):
        if segment.size == 0:
            continue
        start = int(segment[0])
        end = int(segment[-1])
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


def _classify_mismatches(
    grouped_indices: list[np.ndarray],
    gt_pred_arr: np.ndarray,
    gt_labels: np.ndarray,
    label_config: LabelConfig,
    bucket_names: tuple[str, str, str, str],
    out: dict[str, dict[str, list[int]]],
) -> None:
    """Sort contiguous mismatch groups into four buckets, keyed by GT boundary.

    The four ``bucket_names`` slots correspond, in order, to:

    * 5'-extensions / 5'-deletions   (run anchored only on its 3' side)
    * 3'-extensions / 3'-deletions   (run anchored only on its 5' side)
    * joins / splits                 (run anchored on both sides)
    * whole insertions / whole deletions (run anchored on neither side)

    Each qualifying run appends its *length* (in nucleotides) to
    ``out["<5'-flank>:<3'-flank>"][bucket]``, where the flank names are read from
    the unpadded ``gt_labels`` immediately outside the run.
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

        # Back to unpadded coordinates for the GT-label flank lookup.  Keys are in
        # array (5'→3') order, so e.g. "FIVE_PRIME_UTR:CDS" and "CDS:FIVE_PRIME_UTR"
        # stay distinct (5'UTR vs 3'UTR boundary).
        adjusted = mismatch - 1
        left_name = _flank_name(gt_labels, int(adjusted[0]) - 1, label_config)
        right_name = _flank_name(gt_labels, int(adjusted[-1]) + 1, label_config)
        boundary_key = semantic_boundary_label(left_name, right_name)
        length = int(adjusted.size)

        if target_on_3_prime and target_on_5_prime:
            bucket = name_both
        elif target_on_3_prime:
            bucket = name_5_prime
        elif target_on_5_prime:
            bucket = name_3_prime
        else:
            bucket = name_neither

        out[boundary_key][bucket].append(length)
