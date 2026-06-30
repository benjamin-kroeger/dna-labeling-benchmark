"""INDEL metric: classify GT/prediction coding mismatches by the GT exon they touch.

Insertion and deletion runs (coding present on one row but not the other) are
sorted into 5'/3' extensions-or-deletions, whole insertions/deletions, and
join/split events.  Each run is keyed by the **semantic position of the GT exon
it affects** — ``five_prime_terminal_exon``, ``internal_exon``,
``three_prime_terminal_exon`` or ``single_exon_gene`` — read from the precomputed
per-position ``seg_type_arr``.  Deletions sit on a GT exon; anchored insertions
extend an adjacent exon; joins bridge an intron (always ``internal_exon``); whole
insertions are free-floating predicted exons keyed by a nearest-GT-exon scan.

Orientation assumption
----------------------
All input is assumed to be presented **5'→3'**: lower array index = 5', higher
index = 3'.  Re-orienting minus-strand input is the caller's responsibility.

Output shape
------------
``eval_indel`` returns ``{"by_boundary": {exon_type: {bucket: [lengths]}},
"exon_opportunities": {exon_type: count}, "n_gt_segments": int,
"n_pred_segments": int}`` where ``exon_type`` is one of the four semantic
positions above and each ``lengths`` entry is a run length in nucleotides.
``exon_opportunities`` counts GT exons per type — the denominator from which the
plotter derives per-exon, per-gene and per-intron rates.  Only run *lengths* are
stored (never index arrays); that is all downstream plotting needs.

The semantic-type keys are produced here, the one layer where ``label_config`` is
available, so they are the single canonical form for tests, plotting, the
accumulator merge, and JSON serialisation alike.
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
    """Sort insertion/deletion runs into per-exon 5'/3'/whole/join-split buckets.

    Parameters
    ----------
    grouped_insertions, grouped_deletions : list[np.ndarray]
        Contiguous index runs (unpadded coordinates) where exactly one of GT /
        pred is coding.
    gt_positive_mask, pred_positive_mask : np.ndarray
        Boolean coding masks for the active evaluation scope.
    label_config : LabelConfig
        Used (with ``gt_labels``) to build the per-position GT exon-type array.
    gt_labels : np.ndarray
        The *unpadded* full GT label row.  Used to type each GT coding segment
        by its outer flanks when building ``seg_type_arr``.
    n_gt_segments, n_pred_segments : int
        Number of GT / predicted coding segments in the active scope.
        ``n_gt_segments`` lets the plotter derive intron counts;
        ``n_pred_segments`` is retained for diagnostics (no longer a denominator).

    Returns
    -------
    dict
        ``{"by_boundary": {exon_type: {bucket: [length, ...]}},
        "exon_opportunities": {exon_type: int},
        "n_gt_segments": int, "n_pred_segments": int}``.

        ``exon_opportunities`` counts GT exons per semantic type and is the
        denominator from which the plotter derives per-exon, per-gene and
        per-intron rates, so a count becomes "fraction of that exon type that
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
        padded_insertions, padded_arr, _INSERTION_BUCKETS, by_boundary, is_insertion=True,
        seg_type_arr=seg_type_arr,
    )
    _classify_mismatches(
        padded_deletions, padded_arr, _DELETION_BUCKETS, by_boundary, is_insertion=False,
        seg_type_arr=seg_type_arr,
    )

    return {
        "by_boundary": {boundary: dict(buckets) for boundary, buckets in by_boundary.items()},
        "exon_opportunities": _exon_opportunities(gt_positive_mask, seg_type_arr),
        "n_gt_segments": int(n_gt_segments),
        "n_pred_segments": int(n_pred_segments),
    }


def _exon_opportunities(gt_positive_mask: np.ndarray, seg_type_arr: np.ndarray) -> dict[str, int]:
    """Count GT exons per semantic position type (+1 per coding segment).

    The denominator for exon-keyed events: each GT exon contributes one
    opportunity to its own type.  An exon has one 5' and one 3' boundary, so a
    single count serves both the 5'- and 3'-side event columns.  Downstream the
    plotter derives gene and intron counts from this dict —
    ``n_genes = #five_prime_terminal_exon + #single_exon_gene`` and
    ``n_introns = n_gt_segments - n_genes`` — both of which assume each
    evaluation window is a complete transcript (true under per-transcript scoping).
    """
    opportunities: dict[str, int] = {}
    for segment in get_contiguous_groups(np.where(gt_positive_mask)[0]):
        if segment.size == 0:
            continue
        key = str(seg_type_arr[int(segment[0])])
        opportunities[key] = opportunities.get(key, 0) + 1
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


def _whole_insertion_key(seg_type_arr: np.ndarray, lo: int, hi: int) -> str:
    """Classify a free-floating predicted exon (whole insertion) by GT context.

    Scans outward for the nearest GT exon on each side (``seg_type_arr`` is
    non-``None`` only at GT-coding positions).  A whole insertion that fills an
    entire GT intron has *coding* immediate flanks, so the classification cannot
    rely on the immediate flank label — only on the nearest exons' types.

    * inside one gene's intron (left exon continues 3', right exon continues 5')
      → ``internal_exon``
    * a gene only to one side → that terminal exon (gene to the right → 5'
      terminal / hallucinated upstream; to the left → 3' terminal / downstream)
    * genes on both sides (between two genes) or no genes → ``single_exon_gene``
    """
    left = next((str(x) for x in seg_type_arr[:lo][::-1] if x is not None), None)
    right = next((str(x) for x in seg_type_arr[hi + 1 :] if x is not None), None)
    is_intron = left in {"five_prime_terminal_exon", "internal_exon"} and right in {
        "internal_exon",
        "three_prime_terminal_exon",
    }
    if left is None and right is None:
        return "single_exon_gene"
    if is_intron:
        return "internal_exon"
    if right is not None and left is None:
        return "five_prime_terminal_exon"
    if left is not None and right is None:
        return "three_prime_terminal_exon"
    return "single_exon_gene"


def _classify_mismatches(
    grouped_indices: list[np.ndarray],
    gt_pred_arr: np.ndarray,
    bucket_names: tuple[str, str, str, str],
    out: dict[str, dict[str, list[int]]],
    is_insertion: bool,
    seg_type_arr: np.ndarray,
) -> None:
    """Sort contiguous mismatch groups into four buckets, keyed by the GT exon they touch.

    The four ``bucket_names`` slots correspond, in order, to:

    * 5'-extensions / 5'-deletions   (run anchored only on its 3' side)
    * 3'-extensions / 3'-deletions   (run anchored only on its 5' side)
    * joins / splits                 (run anchored on both sides)
    * whole insertions / whole deletions (run anchored on neither side)

    **Boundary key** = the semantic position of the GT exon the event affects,
    read from ``seg_type_arr`` (which already encodes ``single_exon_gene``):

    * Deletions are GT-coding throughout, so every deletion bucket keys to the
      exon the run sits on (``seg_type_arr`` at a run position).
    * Anchored insertions (5'/3' extensions) key to the adjacent GT exon they extend.
    * Joins bridge two exons across an intron → always ``internal_exon``.
    * Whole insertions are free-floating predicted exons → keyed by the
      nearest-GT-exon scan in :func:`_whole_insertion_key`.
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

        # Back to unpadded coordinates for the seg_type_arr lookup.
        adjusted = mismatch - 1

        if target_on_3_prime and target_on_5_prime:
            bucket = name_both
        elif target_on_3_prime:
            bucket = name_5_prime
        elif target_on_5_prime:
            bucket = name_3_prime
        else:
            bucket = name_neither

        if not is_insertion:
            # Deletion run is GT-coding throughout → it sits on one GT exon.
            boundary_key = str(seg_type_arr[int(adjusted[0])])
        elif bucket == name_5_prime:
            # 5'-extension: anchored on its 3' side by the exon it extends.
            boundary_key = str(seg_type_arr[int(adjusted[-1]) + 1])
        elif bucket == name_3_prime:
            # 3'-extension: anchored on its 5' side by the exon it extends.
            boundary_key = str(seg_type_arr[int(adjusted[0]) - 1])
        elif bucket == name_both:
            # Join: fills the intron between two exons.
            boundary_key = "internal_exon"
        else:
            # Whole insertion: free-floating predicted exon in GT non-coding.
            boundary_key = _whole_insertion_key(seg_type_arr, int(adjusted[0]), int(adjusted[-1]))

        out[boundary_key][bucket].append(int(adjusted.size))
