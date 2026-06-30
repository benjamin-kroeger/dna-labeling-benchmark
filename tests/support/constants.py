"""Shared label tokens and metric sets for the test suite.

The token values match ``BEND_LABEL_CONFIG`` (background=8, exon=0, donor=1,
intron=2, acceptor=3).
"""

from __future__ import annotations

from dna_segmentation_benchmark.eval.evaluate_predictors import EvalMetrics

EXON, DONOR, INTRON, ACCEPTOR, NONCODING = 0, 1, 2, 3, 8

# The four metrics every EXON_INTRON end-to-end test exercises.
CORE_METRICS = [
    EvalMetrics.REGION_DISCOVERY,
    EvalMetrics.BOUNDARY_EXACTNESS,
    EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
    EvalMetrics.STRUCTURAL_COHERENCE,
]

# CORE_METRICS plus diagnostic depth, used by the media / streaming paths.
MEDIA_METRICS = CORE_METRICS + [EvalMetrics.DIAGNOSTIC_DEPTH]
