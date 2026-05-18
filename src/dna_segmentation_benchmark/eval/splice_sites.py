from __future__ import annotations

import dataclasses

from .structure import ExtractedStructure, Segment
from ..label_definition import LabelConfig


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

    gt_donors = gt_structure.filter_by_label(donor_label)
    gt_acceptors = gt_structure.filter_by_label(acceptor_label)

    pred_donors = pred_structure.filter_by_label(donor_label)
    pred_acceptors = pred_structure.filter_by_label(acceptor_label)

    if len(gt_donors) != len(gt_acceptors):
        raise ValueError("There is an uneven amount of donor and acceptor segments in the gt")

    gt_introns = gt_structure.filter_by_label(intron_label)

    confusion = SpliceSiteConfusion()
    for donor, acceptor in zip(gt_donors, gt_acceptors):
        expected_intron = Segment(label=intron_label, start=donor.end + 1, end=acceptor.start - 1)
        if expected_intron not in gt_introns:
            raise AssertionError(
                f"Expected an intron between donor {donor} and acceptor {acceptor}"
            )

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

    gt_donors_set = set(gt_donors)
    gt_acceptors_set = set(gt_acceptors)
    confusion.donor_fp = sum(1 for d in pred_donors if d not in gt_donors_set)
    confusion.acceptor_fp = sum(1 for a in pred_acceptors if a not in gt_acceptors_set)

    return confusion