"""Unit tests for INDEL boundary-key assignment (``eval_indel``)."""

import numpy as np

from dna_segmentation_benchmark.eval.indel_metrics import eval_indel
from dna_segmentation_benchmark.label_definition import AnnotationMode, LabelConfig


def test_multibase_insertion_keyed_to_segment_edge_not_outer_flank():
    """A boundary-anchored insertion must be keyed by the GT coding-segment type,
    matching the ``junction_opportunities`` denominator.

    Regression (M-2): a multi-base 3'-extension that bridges the gap toward the
    next coding segment used to be keyed by the run's *outer* flank — which lands
    on the next segment's coding base across the gap — mislabelling a terminal
    edge as ``internal_exon``.

    With the segment-type fix, single-exon segments (NONCODING on *both* outer
    flanks) key as ``single_exon_gene`` regardless of which junction the run
    abuts.  The gap here is NONCODING (background, label 8), so both segments
    are single-exon genes — and the 3'-extension keys accordingly.
    """
    cfg = LabelConfig(
        annotation_mode=AnnotationMode.EXON_INTRON,
        background_label=8,
        exon_label=0,
        intron_label=2,
    )
    # Two GT exon segments [0..1] and [4..5] separated by a 2 bp NONCODING gap.
    # Both segments are isolated (no intron on either side) → single-exon genes.
    gt_labels = np.array([0, 0, 8, 8, 0, 0])
    gt_mask = gt_labels == 0
    # Pred over-extends the first segment across the gap (coding [0..3]) and
    # misses the second -> insertion run [2,3] anchored only on its 5' side, and
    # deletion run [4,5] (the missed segment).
    pred_labels = np.array([0, 0, 0, 0, 8, 8])
    pred_mask = pred_labels == 0

    result = eval_indel(
        grouped_insertions=[np.array([2, 3])],
        grouped_deletions=[np.array([4, 5])],
        gt_positive_mask=gt_mask,
        pred_positive_mask=pred_mask,
        label_config=cfg,
        gt_labels=gt_labels,
        n_gt_segments=2,
        n_pred_segments=1,
    )
    bb = result["by_boundary"]
    # The 3'-extension abuts a single-exon gene (NONCODING on both outer flanks)
    # → single_exon_gene, NOT internal_exon (old outer-flank bug) or
    # three_prime_terminal_exon (old inner-edge approach that couldn't see the
    # far side of the segment).
    assert bb["single_exon_gene"]["3_prime_extensions"] == [2]
    assert "3_prime_extensions" not in bb.get("internal_exon", {})
    assert "3_prime_extensions" not in bb.get("three_prime_terminal_exon", {})
    assert "single_exon_gene" in result["junction_opportunities"]


def test_boundary_anchored_events_keyed_by_segment_type_multi_exon():
    """For multi-exon genes (INTRON gap), boundary-anchored events preserve the
    old per-junction key (inner-edge label pair), unchanged by the single-exon fix.
    """
    cfg = LabelConfig(
        annotation_mode=AnnotationMode.EXON_INTRON,
        background_label=8,
        exon_label=0,
        intron_label=2,
    )
    # Two GT exon segments separated by an INTRON → first exon is 5'-terminal.
    gt_labels = np.array([0, 0, 2, 2, 0, 0])
    gt_mask = gt_labels == 0
    # Pred over-extends the first exon into the intron → 3'-extension at [2,3].
    pred_labels = np.array([0, 0, 0, 0, 8, 8])
    pred_mask = pred_labels == 0

    result = eval_indel(
        grouped_insertions=[np.array([2, 3])],
        grouped_deletions=[np.array([4, 5])],
        gt_positive_mask=gt_mask,
        pred_positive_mask=pred_mask,
        label_config=cfg,
        gt_labels=gt_labels,
        n_gt_segments=2,
        n_pred_segments=1,
    )
    bb = result["by_boundary"]
    # Segment [0,1] is 5'-terminal (left outer = "none", right outer = INTRON).
    # The 3'-extension runs into the intron: inner-edge key (EXON, INTRON)
    # → internal_exon (old behaviour preserved for multi-exon genes).
    assert bb["internal_exon"]["3_prime_extensions"] == [2]
    assert "3_prime_extensions" not in bb.get("single_exon_gene", {})
    # The missed segment [4,5] has INTRON on left, "none" on right → 3'-terminal.
    assert bb["three_prime_terminal_exon"]["whole_deletions"] == [2]
