"""Unit tests for INDEL boundary-key assignment (``eval_indel``)."""

import numpy as np

from dna_segmentation_benchmark.eval.indel_metrics import eval_indel
from dna_segmentation_benchmark.label_definition import AnnotationMode, LabelConfig


def test_multibase_insertion_keyed_to_segment_edge_not_outer_flank():
    """A boundary-anchored insertion must be keyed by the GT coding-segment edge
    it abuts, matching the ``junction_opportunities`` denominator.

    Regression (M-2): a multi-base 3'-extension that bridges the gap toward the
    next coding segment used to be keyed by the run's *outer* flank — which lands
    on the next segment's coding base across the gap — mislabelling a terminal
    edge as ``internal_exon``.  It must instead read the GT label immediately
    outside the segment (here background -> terminal).
    """
    cfg = LabelConfig(
        annotation_mode=AnnotationMode.EXON_INTRON,
        background_label=8,
        exon_label=0,
        intron_label=2,
    )
    # Two GT exon segments [0..1] and [4..5] separated by a 2 bp background gap.
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
    # The 3' extension is filed under the terminal-exon boundary (segment edge),
    # consistent with the denominator -- NOT internal_exon (the across-gap flank).
    assert bb["three_prime_terminal_exon"]["3_prime_extensions"] == [2]
    assert "3_prime_extensions" not in bb.get("internal_exon", {})
    assert "three_prime_terminal_exon" in result["junction_opportunities"]
