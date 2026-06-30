"""Unit tests for INDEL boundary-key assignment (``eval_indel``)."""

import numpy as np

from dna_segmentation_benchmark.eval.indel_metrics import eval_indel
from dna_segmentation_benchmark.label_definition import AnnotationMode, LabelConfig


def test_multibase_insertion_keyed_to_segment_edge_not_outer_flank():
    """A boundary-anchored insertion is keyed by the GT exon it extends,
    matching the ``exon_opportunities`` denominator.

    A multi-base 3'-extension is keyed by the adjacent GT exon's semantic type
    (via ``seg_type_arr``), not by the run's outer flank.  The gap here is
    NONCODING (background, label 8), so both segments are single-exon genes — and
    the 3'-extension keys to ``single_exon_gene``.
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
    assert "single_exon_gene" in result["exon_opportunities"]


def test_boundary_anchored_events_keyed_by_segment_type_multi_exon():
    """For multi-exon genes, a boundary-anchored event keys to the exon it
    extends, not to the junction the run slips into.
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
    # The 3'-extension extends that exon, so it keys to the exon it extends:
    # five_prime_terminal_exon (not the internal junction it runs into).
    assert bb["five_prime_terminal_exon"]["3_prime_extensions"] == [2]
    assert "3_prime_extensions" not in bb.get("single_exon_gene", {})
    # The missed segment [4,5] has INTRON on left, "none" on right → 3'-terminal.
    assert bb["three_prime_terminal_exon"]["whole_deletions"] == [2]


def test_event_denominator_intron_and_single_exon_gene():
    """``_event_denominator``: joins divide by intron count; single-exon-gene
    whole insertions have no bounded opportunity (0 → masked in the rate plot).
    """
    from dna_segmentation_benchmark.plotting.metrics.indel import _event_denominator

    # 3-exon transcript: one exon of each multi-exon type.
    # n_genes = 1 (one 5'-terminal, no single-exon); n_introns = 3 - 1 = 2.
    payload = {
        "exon_opportunities": {
            "five_prime_terminal_exon": 1,
            "internal_exon": 1,
            "three_prime_terminal_exon": 1,
        },
        "n_gt_segments": 3,
    }
    assert _event_denominator(payload, "joined", "internal_exon") == 2
    assert _event_denominator(payload, "whole_insertions", "internal_exon") == 2
    assert _event_denominator(payload, "whole_insertions", "five_prime_terminal_exon") == 1
    assert _event_denominator(payload, "whole_insertions", "single_exon_gene") == 0
    # Anchored slips, splits and whole deletions → per-exon-type count.
    assert _event_denominator(payload, "3_prime_deletions", "five_prime_terminal_exon") == 1
    assert _event_denominator(payload, "whole_deletions", "three_prime_terminal_exon") == 1
