"""Shared fixtures for the test suite.

Importable shared *code* (the W&B stub, GFF helpers, parametrized case data and
the metric-comparison machinery) lives in the ``support`` package; this module
holds only the pytest *fixtures* that wrap it.

Provides:

* Paths to the committed dummy GENCODE / Augustus sample files under ``data/``
  (swap in real trimmed data later — see ``tests/data/_make_dummy_fixtures.py``).
* Label-config fixtures for the annotation modes the suite exercises.
* GFF3 (de)serialisation + perturbation helpers (re-exported from
  ``support.gff``) so known-answer predictions can be derived by perturbing the
  ground truth (controlled-mutant approach).
* A ``wandb_stub`` fixture backed by ``support.wandb_stub.FakeWandb`` so the W&B
  logger can be driven by real pipeline output without a live W&B run.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless figure rendering for the media tests

from gene_calling_benchmark.io_utils import collect_gff
from gene_calling_benchmark.label_definition import (
    AnnotationMode,
    BEND_LABEL_CONFIG,
    BenchmarkScope,
    LabelConfig,
)

from support import gff
from support.wandb_stub import FakeWandb

DATA_DIR = Path(__file__).parent / "data"
GENCODE_GTF = DATA_DIR / "gencode_chr2_snippet.gtf"
AUGUSTUS_GFF = DATA_DIR / "augustus_chr2_snippet.gff"

# Transcript on Gene A — the multi-exon protein-coding gene used as the base
# for the controlled-mutant prediction tests.
TARGET_TRANSCRIPT = "ENSTA"


# ---------------------------------------------------------------------------
# Sample-file paths and label configs
# ---------------------------------------------------------------------------


@pytest.fixture
def gencode_gtf() -> str:
    """Path to the (dummy) trimmed GENCODE ground-truth GTF."""
    return str(GENCODE_GTF)


@pytest.fixture
def augustus_gff() -> str:
    """Path to the (dummy) Augustus prediction GFF."""
    return str(AUGUSTUS_GFF)


@pytest.fixture
def gencode_df() -> pd.DataFrame:
    """Normalised DataFrame of the GENCODE ground truth."""
    return collect_gff(str(GENCODE_GTF))


@pytest.fixture
def exon_intron_config() -> LabelConfig:
    """EXON_INTRON config (background/exon/intron/splice) — matches Augustus CDS→exon."""
    return BEND_LABEL_CONFIG


@pytest.fixture
def cds_config() -> LabelConfig:
    """UTR_CDS_INTRON config evaluated at CDS scope (Augustus is CDS-only)."""
    return LabelConfig(
        annotation_mode=AnnotationMode.UTR_CDS_INTRON,
        evaluation_scope=BenchmarkScope.CDS,
        background_label=1,
        cds_label=0,
        five_prime_utr_label=2,
        three_prime_utr_label=3,
    )


@pytest.fixture
def simple_config() -> LabelConfig:
    """Minimal two-token EXON_INTRON config (background=1, exon=0)."""
    return LabelConfig(
        annotation_mode=AnnotationMode.EXON_INTRON,
        background_label=1,
        exon_label=0,
    )


@pytest.fixture
def utr_config() -> LabelConfig:
    """UTR_CDS_INTRON config at the default transcript-exon scope (background=9)."""
    return LabelConfig(
        annotation_mode=AnnotationMode.UTR_CDS_INTRON,
        background_label=9,
        five_prime_utr_label=4,
        cds_label=0,
        three_prime_utr_label=5,
    )


# ---------------------------------------------------------------------------
# Shared UTR/CDS GFF sample files
# ---------------------------------------------------------------------------


@pytest.fixture
def utr_gt_gff(tmp_path) -> str:
    """Ground truth with explicit 5'UTR / CDS / 3'UTR roles for one transcript."""
    return gff.write_gff(
        tmp_path / "utr_gt.gff",
        [
            "chr1\tTest\tmRNA\t1\t30\t.\t+\t.\tID=gt_tx1",
            "chr1\tTest\tfive_prime_UTR\t1\t5\t.\t+\t.\tID=gt_u5;Parent=gt_tx1",
            "chr1\tTest\tCDS\t6\t20\t.\t+\t0\tID=gt_cds;Parent=gt_tx1",
            "chr1\tTest\tthree_prime_UTR\t21\t30\t.\t+\t.\tID=gt_u3;Parent=gt_tx1",
        ],
    )


@pytest.fixture
def utr_pred_gff(tmp_path) -> str:
    """Prediction with a 5'UTR over-extended into the CDS region (no CDS feature)."""
    return gff.write_gff(
        tmp_path / "utr_pred.gff",
        [
            "chr1\tPred\tmRNA\t1\t30\t.\t+\t.\tID=pred_tx1",
            "chr1\tPred\tfive_prime_UTR\t1\t20\t.\t+\t.\tID=pred_u5;Parent=pred_tx1",
            "chr1\tPred\tthree_prime_UTR\t21\t30\t.\t+\t.\tID=pred_u3;Parent=pred_tx1",
        ],
    )


# ---------------------------------------------------------------------------
# GFF3 perturbation helpers (controlled mutants)
# ---------------------------------------------------------------------------


@pytest.fixture
def gff3_tools():
    """Bundle of GFF3 (de)serialisation + perturbation helpers (from support.gff)."""
    return SimpleNamespace(
        write_gff3=gff.write_gff3,
        transcript_subset=gff.transcript_subset,
        drop_internal_exon=gff.drop_internal_exon,
        extend_first_cds=gff.extend_first_cds,
        merge_first_two_exons=gff.merge_first_two_exons,
        target_transcript=TARGET_TRANSCRIPT,
    )


# ---------------------------------------------------------------------------
# W&B stub
# ---------------------------------------------------------------------------


@pytest.fixture
def wandb_stub(monkeypatch) -> FakeWandb:
    """Patch the logger's wandb accessor with a fresh ``FakeWandb`` and return it."""
    fake = FakeWandb()
    monkeypatch.setattr(
        "gene_calling_benchmark.wandb_logger._require_wandb",
        lambda: fake,
    )
    return fake
