from __future__ import annotations

import dataclasses
import logging

from .structure import ExtractedStructure, Segment
from ..label_definition import LabelConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SpliceSiteConfusion:
    both_correct: int = 0
    donor_only: int = 0
    acceptor_only: int = 0
    neither: int = 0

    donor_tp: int = 0
    donor_fp: int = 0
    donor_fn: int = 0

    acceptor_tp: int = 0
    acceptor_fp: int = 0
    acceptor_fn: int = 0

    gt_malformed_junctions: int = 0
    pred_malformed_junctions: int = 0


def _extract_junction_pairs(
    structure: ExtractedStructure,
    donor_label: int,
    acceptor_label: int,
    intron_label: int,
) -> tuple[list[tuple[Segment, Segment]], int]:
    """Walk *structure* and return (valid_pairs, malformed_count).

    A valid pair is a DONOR segment followed by zero or more INTRON segments
    followed by an ACCEPTOR segment.  Donors or acceptors that don't form a
    valid pair are counted as malformed.
    """
    segments = structure.segments
    pairs: list[tuple[Segment, Segment]] = []
    malformed = 0
    i = 0
    while i < len(segments):
        seg = segments[i]
        if seg.label == donor_label:
            j = i + 1
            while j < len(segments) and segments[j].label == intron_label:
                j += 1
            if j < len(segments) and segments[j].label == acceptor_label:
                pairs.append((seg, segments[j]))
                i = j + 1
                continue
            else:
                malformed += 1
        elif seg.label == acceptor_label:
            # Acceptor not consumed by a valid donor → malformed
            malformed += 1
        i += 1
    return pairs, malformed


def eval_splice_site_junctions(
    gt_structure: ExtractedStructure,
    pred_structure: ExtractedStructure,
    label_config: LabelConfig,
) -> SpliceSiteConfusion:
    if label_config.intron_label is None:
        raise ValueError("label_config must define intron_label for splice-site evaluation")
    if label_config.splice_donor_label is None or label_config.splice_acceptor_label is None:
        raise ValueError("label_config must define splice_donor_label and splice_acceptor_label")

    intron_label: int = label_config.intron_label
    donor_label: int = label_config.splice_donor_label
    acceptor_label: int = label_config.splice_acceptor_label

    gt_pairs, gt_malformed = _extract_junction_pairs(
        gt_structure, donor_label, acceptor_label, intron_label
    )
    _, pred_malformed = _extract_junction_pairs(
        pred_structure, donor_label, acceptor_label, intron_label
    )

    if gt_malformed > 0:
        logger.warning(
            "GT has %d malformed splice junction(s) (donor or acceptor without a valid pair); "
            "excluded from TP/FP/FN scoring.",
            gt_malformed,
        )
    if pred_malformed > 0:
        logger.warning(
            "Prediction has %d malformed splice junction(s) (donor or acceptor without a valid pair).",
            pred_malformed,
        )

    confusion = SpliceSiteConfusion(
        gt_malformed_junctions=gt_malformed,
        pred_malformed_junctions=pred_malformed,
    )

    pred_donors = set(pred_structure.filter_by_label(donor_label))
    pred_acceptors = set(pred_structure.filter_by_label(acceptor_label))
    gt_donors_set = {donor for donor, _ in gt_pairs}
    gt_acceptors_set = {acceptor for _, acceptor in gt_pairs}

    for donor, acceptor in gt_pairs:
        donor_hit = donor in pred_donors
        acceptor_hit = acceptor in pred_acceptors

        match (donor_hit, acceptor_hit):
            case (True, True):
                confusion.both_correct += 1
            case (True, False):
                confusion.donor_only += 1
            case (False, True):
                confusion.acceptor_only += 1
            case (False, False):
                confusion.neither += 1

        if donor_hit:
            confusion.donor_tp += 1
        else:
            confusion.donor_fn += 1

        if acceptor_hit:
            confusion.acceptor_tp += 1
        else:
            confusion.acceptor_fn += 1

    confusion.donor_fp = sum(1 for d in pred_donors if d not in gt_donors_set)
    confusion.acceptor_fp = sum(1 for a in pred_acceptors if a not in gt_acceptors_set)

    return confusion
