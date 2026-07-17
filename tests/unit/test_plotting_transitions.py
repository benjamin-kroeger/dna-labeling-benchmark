"""Tests for state-transition plotting under a restricted scope.

Focus: labels demoted out of scope (5'/3' UTR under ``cds`` scope) must not
appear in either transition figure — no source panel, no row/col, and no
"→ UTR" category — even if the (possibly stale) matrices still carry mass in
those cells.  INTRON/splice, which are never demoted, must stay.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering (conftest also sets this)

import matplotlib.pyplot as plt
import numpy as np

from gene_calling_benchmark.label_definition import (
    AnnotationMode,
    BenchmarkScope,
    LabelConfig,
)
from gene_calling_benchmark.plotting.metrics.transitions import (
    plot_false_transitions,
    plot_transition_matrices,
)

# background=8, cds=0, 5'UTR=4, 3'UTR=5, intron=2, donor=1, acceptor=3.
_CDS_SCOPE_CFG = LabelConfig(
    annotation_mode=AnnotationMode.UTR_CDS_INTRON,
    evaluation_scope=BenchmarkScope.CDS,
    background_label=8,
    cds_label=0,
    five_prime_utr_label=4,
    three_prime_utr_label=5,
    intron_label=2,
    splice_donor_label=1,
    splice_acceptor_label=3,
)

_IDS = sorted(_CDS_SCOPE_CFG.labels.keys())  # [0,1,2,3,4,5,8]
_L = len(_IDS)
_POS = {lid: i for i, lid in enumerate(_IDS)}
_CDS, _INTRON, _5UTR = 0, 2, 4


def _zero_mats() -> dict[int, np.ndarray]:
    return {lid: np.zeros((_L, _L), dtype=np.int64) for lid in _IDS}


def _all_texts(fig: plt.Figure) -> list[str]:
    texts: list[str] = []
    if fig._suptitle is not None:
        texts.append(fig._suptitle.get_text())
    for ax in fig.axes:
        texts += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        texts += [t.get_text() for t in ax.get_xticklabels()]
        texts += [t.get_text() for t in ax.get_yticklabels()]
        legend = ax.get_legend()
        if legend is not None:
            texts += [t.get_text() for t in legend.get_texts()]
    return texts


def test_transition_matrices_drop_demoted_utr():
    mats = _zero_mats()
    mats[_CDS][_POS[_INTRON], _POS[_INTRON]] = 5   # CDS→INTRON (kept)
    mats[_CDS][_POS[8], _POS[_5UTR]] = 3           # stale CDS→UTR (must be dropped)

    fig = plot_transition_matrices(mats, _CDS_SCOPE_CFG, method_name="m")
    assert fig is not None
    texts = _all_texts(fig)
    assert not any("UTR" in t for t in texts)
    assert any("INTRON" in t for t in texts) and any("CDS" in t for t in texts)
    # One source panel per active label (7 labels − 2 UTR = 5), UTR panels gone.
    assert sum(t.startswith("Source:") for t in texts) == _L - 2
    plt.close(fig)


def test_false_transitions_drop_demoted_utr():
    premature = _zero_mats()
    premature[_CDS][_POS[_CDS], _POS[_INTRON]] = 4   # Premature → INTRON (kept)
    spurious = _zero_mats()
    spurious[_CDS][_POS[_CDS], _POS[_5UTR]] = 9       # stale Spurious → UTR (must be dropped)

    per_method = {
        "m": {
            "late_catchup": _zero_mats(),
            "premature": premature,
            "spurious": spurious,
            "stable_position_counts": {lid: 10 for lid in _IDS},
        }
    }

    fig = plot_false_transitions(per_method, _CDS_SCOPE_CFG)
    assert fig is not None
    texts = _all_texts(fig)
    assert not any("UTR" in t for t in texts)
    assert any("INTRON" in t for t in texts)
    plt.close(fig)
