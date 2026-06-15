"""Shared fixtures and helpers for the integration tests.

Provides:

* Paths to the committed dummy GENCODE / Augustus fixtures (swap in real
  trimmed data later — see ``tests/fixtures/_make_dummy_fixtures.py``).
* Label-config fixtures for the two annotation modes the suite exercises.
* GFF3 (de)serialisation helpers that turn a normalised ``collect_gff``
  DataFrame back into a file, so known-answer *predictions* can be derived by
  perturbing the ground truth (controlled-mutant approach).
* A ``_FakeWandb`` stub + ``wandb_stub`` fixture so the W&B logger can be
  driven by real pipeline output without a live W&B run.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless figure rendering for the media tests

from dna_segmentation_benchmark.io_utils import DEFAULT_TRANSCRIPT_TYPES, collect_gff
from dna_segmentation_benchmark.label_definition import (
    AnnotationMode,
    BEND_LABEL_CONFIG,
    BenchmarkScope,
    LabelConfig,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GENCODE_GTF = FIXTURES_DIR / "gencode_chr2_snippet.gtf"
AUGUSTUS_GFF = FIXTURES_DIR / "augustus_chr2_snippet.gff"

# Transcript on Gene A — the multi-exon protein-coding gene used as the base
# for the controlled-mutant prediction tests.
TARGET_TRANSCRIPT = "ENSTA"


# ---------------------------------------------------------------------------
# Fixture paths and label configs
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


# ---------------------------------------------------------------------------
# GFF3 (de)serialisation + perturbation helpers
# ---------------------------------------------------------------------------


def write_gff3(df: pd.DataFrame, path, transcript_types: list[str] | None = None) -> str:
    """Serialise a normalised ``collect_gff`` DataFrame back to a GFF3 file.

    Reconstructs transcript rows (``ID=<gff_id>``) and their child features
    (``Parent=<transcript gff_id>``).  Gene rows (no parent) are dropped; the
    pipeline only needs transcripts and their children.
    """
    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    lines = ["##gff-version 3"]

    for row in df.itertuples(index=False):
        if row.type in transcript_types:
            lines.append(
                "\t".join(
                    [str(row.seqid), "test", str(row.type), str(int(row.start)),
                     str(int(row.end)), ".", str(row.strand), ".", f"ID={row.gff_id}"]
                )
            )

    counters: dict[tuple, int] = {}
    for row in df.itertuples(index=False):
        if row.type in transcript_types or pd.isna(row.parent):
            continue
        idx = counters.get((row.parent, row.type), 0)
        counters[(row.parent, row.type)] = idx + 1
        lines.append(
            "\t".join(
                [str(row.seqid), "test", str(row.type), str(int(row.start)),
                 str(int(row.end)), ".", str(row.strand), ".",
                 f"ID={row.parent}.{row.type}.{idx};Parent={row.parent}"]
            )
        )

    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


def transcript_subset(df: pd.DataFrame, transcript_id: str) -> pd.DataFrame:
    """Return the transcript row + child rows for one transcript."""
    is_tx = df["type"].isin(DEFAULT_TRANSCRIPT_TYPES) & (df["gff_id"] == transcript_id)
    is_child = ~df["type"].isin(DEFAULT_TRANSCRIPT_TYPES) & (df["parent"] == transcript_id)
    return df[is_tx | is_child].copy()


def drop_internal_exon(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the middle exon (and any CDS inside it) — simulates exon skipping."""
    exons = df[df["type"] == "exon"].sort_values("start")
    if len(exons) < 3:
        raise ValueError("drop_internal_exon needs a transcript with >= 3 exons.")
    middle = exons.iloc[len(exons) // 2]
    inside_cds = (
        (df["type"] == "CDS")
        & (df["start"] >= middle.start)
        & (df["end"] <= middle.end)
    )
    return df[~((df.index == middle.name) | inside_cds)].copy()


def extend_first_cds(df: pd.DataFrame, delta: int = 1) -> pd.DataFrame:
    """Extend the first CDS by ``delta`` bp — simulates a small CDS indel."""
    df = df.copy()
    first_cds = df[df["type"] == "CDS"].sort_values("start").index[0]
    df.loc[first_cds, "end"] = int(df.loc[first_cds, "end"]) + delta
    return df


def merge_first_two_exons(df: pd.DataFrame) -> pd.DataFrame:
    """Fuse the first two exons into one — simulates intron retention."""
    df = df.copy()
    exons = df[df["type"] == "exon"].sort_values("start")
    if len(exons) < 2:
        raise ValueError("merge_first_two_exons needs a transcript with >= 2 exons.")
    first, second = exons.iloc[0], exons.iloc[1]
    df.loc[first.name, "end"] = int(second.end)
    return df.drop(index=second.name)


@pytest.fixture
def gff3_tools():
    """Bundle of GFF3 (de)serialisation + perturbation helpers."""
    return SimpleNamespace(
        write_gff3=write_gff3,
        transcript_subset=transcript_subset,
        drop_internal_exon=drop_internal_exon,
        extend_first_cds=extend_first_cds,
        merge_first_two_exons=merge_first_two_exons,
        target_transcript=TARGET_TRANSCRIPT,
    )


# ---------------------------------------------------------------------------
# W&B stub
# ---------------------------------------------------------------------------


class _FakeWandb:
    """Minimal in-memory stand-in for the ``wandb`` module."""

    class Image:
        def __init__(self, fig):
            self.fig = fig

    class Video:
        def __init__(self, data, fps, format):
            self.data = data
            self.fps = fps
            self.format = format

    class Table:
        def __init__(self, columns, data):
            self.columns = columns
            self.data = data

    def __init__(self):
        self.logged: list[dict] = []
        self.defined_metrics: list[str] = []
        self.inits: list[dict] = []

    def log(self, data, step=None):
        self.logged.append({"data": data, "step": step})

    def define_metric(self, pattern):
        self.defined_metrics.append(pattern)

    def init(self, **kwargs):
        self.inits.append(kwargs)
        return SimpleNamespace(**kwargs)


@pytest.fixture
def wandb_stub(monkeypatch) -> _FakeWandb:
    """Patch the logger's wandb accessor with a fresh ``_FakeWandb`` and return it."""
    fake = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake,
    )
    return fake
