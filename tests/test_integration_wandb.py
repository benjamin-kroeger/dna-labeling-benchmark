"""Integration tests for the W&B logger driven by real pipeline output.

The unit tests in ``test_wandb_logger.py`` feed the logger hand-built result
dicts.  These tests instead run the actual ``benchmark_from_gff`` pipeline and
push its output through the logger (via the ``_FakeWandb`` stub), so the logger
and the pipeline result schema are verified together — a change to either that
desyncs them is caught here.
"""

from __future__ import annotations

import pytest

from dna_segmentation_benchmark.eval.evaluate_predictors import EvalMetrics
from dna_segmentation_benchmark.pipeline import benchmark_from_gff
from dna_segmentation_benchmark.transcript_mapping import LocusMatchingMode
from dna_segmentation_benchmark.wandb_logger import (
    clear_benchmark_media_video_buffer,
    log_benchmark_all_scalars,
    log_benchmark_media,
    log_benchmark_media_videos,
    log_benchmark_scalars,
)

from conftest import _FakeWandb

METRICS = [
    EvalMetrics.REGION_DISCOVERY,
    EvalMetrics.BOUNDARY_EXACTNESS,
    EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
    EvalMetrics.STRUCTURAL_COHERENCE,
    EvalMetrics.DIAGNOSTIC_DEPTH,
]


@pytest.fixture
def pipeline_result(gencode_gtf, augustus_gff, exon_intron_config):
    """Real benchmark output: ``{"aggregated": ..., "global": ...}`` for Augustus."""
    results = benchmark_from_gff(
        gt_path=gencode_gtf,
        pred_paths={"augustus": augustus_gff},
        label_config=exon_intron_config,
        metrics=METRICS,
        locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
        infer_introns=True,
    )
    return results["augustus"]


def test_log_scalars_from_real_output(pipeline_result, exon_intron_config, wandb_stub):
    """Curated scalars from real output are flattened, prefixed and logged."""
    aggregated = pipeline_result["aggregated"]

    logged = log_benchmark_scalars(aggregated, exon_intron_config, step=7, method_prefix="val")

    assert wandb_stub.logged[-1]["step"] == 7
    data = wandb_stub.logged[-1]["data"]
    assert data == logged
    assert data  # non-empty
    assert all(key.startswith("val/") for key in data)

    # The real nucleotide precision actually reaches the dashboard key.
    key = "val/nucleotide_classification/nucleotide/precision"
    assert key in data
    assert data[key] == pytest.approx(
        aggregated["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]["precision"]
    )
    assert "val/region_discovery/neighborhood_hit/recall" in data


def test_log_all_scalars_unwraps_pipeline_wrapper(pipeline_result, exon_intron_config, wandb_stub):
    """The ``{"aggregated", "global"}`` wrapper is unwrapped; noise keys excluded."""
    logged = log_benchmark_all_scalars(pipeline_result, exon_intron_config, step=2)

    assert wandb_stub.logged[-1]["step"] == 2
    assert logged
    # Every metric group present in the real result surfaces at least one scalar.
    assert any(k.startswith("nucleotide_classification/") for k in logged)
    assert any(k.startswith("region_discovery/") for k in logged)
    # State-transition matrices and metadata must never be logged as scalars.
    assert not any("transition" in k or "metadata" in k for k in logged)


def test_log_media_and_videos_from_real_output(pipeline_result, exon_intron_config, wandb_stub):
    """Plots render to W&B Image panels and buffered frames flush to Videos."""
    clear_benchmark_media_video_buffer()
    aggregated = pipeline_result["aggregated"]

    media = log_benchmark_media(aggregated, exon_intron_config, step=1, method_prefix="val")

    assert media  # at least one figure produced from real results
    assert all(isinstance(v, _FakeWandb.Image) for v in media.values())
    assert all(key.startswith("val/plots/") for key in media)

    videos = log_benchmark_media_videos()
    assert videos  # buffered figures (e.g. transition matrices) became videos
    assert all(isinstance(v, _FakeWandb.Video) for v in videos.values())
    assert all(key.endswith("_video") for key in videos)


def test_log_media_videos_buffer_is_cleared_after_flush(pipeline_result, exon_intron_config, wandb_stub):
    """A second flush with no new frames logs nothing (buffer was cleared)."""
    clear_benchmark_media_video_buffer()
    log_benchmark_media(pipeline_result["aggregated"], exon_intron_config, step=1)
    assert log_benchmark_media_videos()  # first flush has frames
    assert log_benchmark_media_videos() == {}  # buffer now empty
