from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from dna_segmentation_benchmark.label_definition import BEND_LABEL_CONFIG
from dna_segmentation_benchmark.plotting.summary_stat_plotting import compare_multiple_predictions
from dna_segmentation_benchmark.eval.evaluate_predictors import EvalMetrics
from dna_segmentation_benchmark.wandb_logger import (
    clear_benchmark_media_video_buffer,
    init_wandb_with_presets,
    log_benchmark_all_scalars,
    log_benchmark_media,
    log_benchmark_media_videos,
    log_benchmark_scalars,
)


class _FakeWandb:
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


def _boundary_landscape_fixture():
    bias = pd.DataFrame(
        np.array([[0.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 0.0]]),
        index=pd.Index([-1, 0, 1], name="5' Residual (Pred − GT)"),
        columns=pd.Index([-1, 0, 1], name="3' Residual (Pred − GT)"),
    )
    reliability = pd.DataFrame(
        np.array([[0.25, 0.50], [0.75, 1.00]]),
        index=pd.Index([0, 1], name="5' Tolerance (bp)"),
        columns=pd.Index([0, 1], name="3' Tolerance (bp)"),
    )
    return bias, reliability


def _false_transition_fixture():
    mat_0 = np.zeros((3, 3), dtype=np.int64)
    mat_0[0, 1] = 2
    mat_2 = np.zeros((3, 3), dtype=np.int64)
    mat_2[1, 0] = 1
    mat_8 = np.zeros((3, 3), dtype=np.int64)
    mat_8[2, 0] = 4
    zeros = np.zeros((3, 3), dtype=np.int64)
    return {
        "late_catchup": {0: mat_0, 2: zeros, 8: zeros},
        "premature":    {0: zeros, 2: mat_2, 8: zeros},
        "spurious":     {0: zeros, 2: zeros, 8: mat_8},
        "stable_position_counts": {0: 10, 2: 10, 8: 10},
    }


def test_log_benchmark_scalars_logs_only_selected_online_metrics(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )

    results = {
        "BOUNDARY_EXACTNESS": {
            "iou_stats": {"mean": 0.6, "std": 0.1},
        },
        "REGION_DISCOVERY": {
            "neighborhood_hit": {"precision": 0.5, "recall": 0.4},
            "internal_hit": {"precision": 0.45, "recall": 0.35},
            "full_coverage_hit": {"precision": 0.4, "recall": 0.3},
            "perfect_boundary_hit": {"precision": 0.3, "recall": 0.2},
        },
        "STRUCTURAL_COHERENCE": {
            "intron_chain": {"precision": 0.55, "recall": 0.45},
            "exon_chain": {"precision": 0.35, "recall": 0.25},
            "segment_count_delta": {"mean": 1.0, "mae": 1.5},
            "exon_recall_per_transcript": [0.5, 1.0, 0.0, 0.75],
            "hallucinated_exon_count_per_transcript": [0, 1, 2, 1],
            "exact_match_rate": 0.25,
        },
        "NUCLEOTIDE_CLASSIFICATION": {
            "nucleotide": {"precision": 0.99, "recall": 0.98, "f1": 0.97},
        },
    }

    logged = log_benchmark_scalars(
        results,
        BEND_LABEL_CONFIG,
        step=7,
        method_prefix="val",
    )

    assert fake_wandb.logged[0]["step"] == 7
    assert fake_wandb.logged[0]["data"] == logged
    assert set(logged.keys()) == {
        "val/boundary_exactness/iou_mean",
        "val/region_discovery/neighborhood_hit/precision",
        "val/region_discovery/neighborhood_hit/recall",
        "val/region_discovery/internal_hit/precision",
        "val/region_discovery/internal_hit/recall",
        "val/region_discovery/full_coverage_hit/precision",
        "val/region_discovery/full_coverage_hit/recall",
        "val/region_discovery/perfect_boundary_hit/precision",
        "val/region_discovery/perfect_boundary_hit/recall",
        "val/struct_coherence/intron_chain/match_rate",
        "val/struct_coherence/exon_chain/match_rate",
        "val/struct_coherence/segment_count_delta/mean",
        "val/struct_coherence/segment_count_delta/mae",
        "val/struct_coherence/exon_recall_per_transcript/mean",
        "val/struct_coherence/hallucinated_exon_count_per_transcript/mean",
        "val/struct_coherence/exact_match_rate",
        "val/nucleotide_classification/nucleotide/precision",
        "val/nucleotide_classification/nucleotide/recall",
        "val/nucleotide_classification/nucleotide/f1",
    }
    assert logged["val/struct_coherence/exon_recall_per_transcript/mean"] == 0.5625
    assert logged["val/struct_coherence/hallucinated_exon_count_per_transcript/mean"] == 1.0
    assert logged["val/nucleotide_classification/nucleotide/f1"] == 0.97


def test_log_benchmark_scalars_includes_diagnostic_depth_metrics(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )

    results = {
        "DIAGNOSTIC_DEPTH": {
            "length_emd": {"mean": 12.5, "mae": 15.0, "rmse": 18.0, "std": 8.0, "min": 0.0, "max": 100.0},
        },
    }

    logged = log_benchmark_scalars(results, BEND_LABEL_CONFIG, step=1)

    assert "diagnostic_depth/length_emd/mean" in logged
    assert "diagnostic_depth/length_emd/mae" in logged
    assert logged["diagnostic_depth/length_emd/mean"] == 12.5
    assert logged["diagnostic_depth/length_emd/mae"] == 15.0


def test_log_benchmark_scalars_includes_splice_site_metrics(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )

    results = {
        "STRUCTURAL_COHERENCE": {
            "intron_chain": {"precision": 0.8, "recall": 0.7},
            "exon_chain": {"precision": 0.9, "recall": 0.85},
            "splice_site_results": {
                "donor_precision": 0.95,
                "donor_recall": 0.90,
                "acceptor_precision": 0.93,
                "acceptor_recall": 0.88,
            },
        },
    }

    logged = log_benchmark_scalars(results, BEND_LABEL_CONFIG, step=2)

    assert "struct_coherence/splice_site_results/donor_precision" in logged
    assert "struct_coherence/splice_site_results/donor_recall" in logged
    assert "struct_coherence/splice_site_results/acceptor_precision" in logged
    assert "struct_coherence/splice_site_results/acceptor_recall" in logged
    assert logged["struct_coherence/splice_site_results/donor_precision"] == 0.95
    assert logged["struct_coherence/splice_site_results/acceptor_recall"] == 0.88


def test_log_benchmark_scalars_skips_missing_splice_site_results(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )

    results = {
        "STRUCTURAL_COHERENCE": {
            "intron_chain": {"precision": 0.8, "recall": 0.7},
            "exon_chain": {"precision": 0.9, "recall": 0.85},
            # no splice_site_results key
        },
    }

    logged = log_benchmark_scalars(results, BEND_LABEL_CONFIG, step=1)

    assert not any("splice_site" in k for k in logged)
    assert "struct_coherence/intron_chain/match_rate" in logged


def test_log_benchmark_all_scalars_logs_everything(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )

    results = {
        "BOUNDARY_EXACTNESS": {
            "iou_stats": {"mean": 0.6, "std": 0.1},
        },
        "NUCLEOTIDE_CLASSIFICATION": {
            "nucleotide": {"precision": 0.99, "recall": 0.98, "f1": 0.97},
        },
        "transition_failures": {0: np.zeros((3, 3))},
        "false_transitions": {"late_catchup": {}},
        "metadata": {"annotation_mode": "EXON_INTRON"},
    }

    logged = log_benchmark_all_scalars(results, BEND_LABEL_CONFIG, step=5, method_prefix="final")

    assert fake_wandb.logged[0]["step"] == 5
    assert "final/boundary_exactness/iou_stats/mean" in logged
    assert "final/boundary_exactness/iou_stats/std" in logged
    assert "final/nucleotide_classification/nucleotide/precision" in logged
    assert "final/nucleotide_classification/nucleotide/f1" in logged
    # transition_failures, false_transitions and metadata are excluded
    assert not any(
        k.split("/")[1] in {"transition_failures", "false_transitions", "metadata"}
        for k in logged
    )


def test_log_benchmark_all_scalars_unwraps_pipeline_wrapper(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )

    results = {
        "aggregated": {
            "BOUNDARY_EXACTNESS": {"iou_stats": {"mean": 0.7}},
        },
        "global": {},
    }

    logged = log_benchmark_all_scalars(results, BEND_LABEL_CONFIG)

    assert "boundary_exactness/iou_stats/mean" in logged


def test_log_benchmark_media_logs_boundary_position_and_transition_plots(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )
    clear_benchmark_media_video_buffer()

    results = {
        "BOUNDARY_EXACTNESS": {
            "fuzzy_metrics": _boundary_landscape_fixture(),
        },
        "DIAGNOSTIC_DEPTH": {
            "position_bias_histogram_fn": [1] * 100,
            "position_bias_histogram_fp": [1] * 100,
        },
        "transition_failures": {
            0: np.array([
                [2, 1, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 0, 5, 0],
                [0, 0, 0, 0, 6],
            ]),
        },
        "false_transitions": _false_transition_fixture(),
    }

    logged = log_benchmark_media(
        results,
        BEND_LABEL_CONFIG,
        step=3,
        method_prefix="val",
    )

    assert fake_wandb.logged[0]["step"] == 3
    assert fake_wandb.logged[0]["data"] == logged
    assert set(logged.keys()) == {
        "val/plots/boundary_landscape",
        "val/plots/position_bias",
        "val/plots/transition_matrices",
        "val/plots/false_transitions",
    }
    assert all(isinstance(value, _FakeWandb.Image) for value in logged.values())


def test_log_benchmark_media_logs_all_figures_as_images(monkeypatch):
    """Every figure produced by the plotting layer is logged, not just a hardcoded subset."""
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )
    clear_benchmark_media_video_buffer()

    results = {
        "BOUNDARY_EXACTNESS": {
            "fuzzy_metrics": _boundary_landscape_fixture(),
        },
        "DIAGNOSTIC_DEPTH": {
            "position_bias_histogram_fn": [1] * 100,
            "position_bias_histogram_fp": [1] * 100,
        },
        "transition_failures": {
            0: np.array([
                [2, 1, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 0, 5, 0],
                [0, 0, 0, 0, 6],
            ]),
        },
        "false_transitions": _false_transition_fixture(),
    }

    logged = log_benchmark_media(results, BEND_LABEL_CONFIG, step=1)

    # All logged values are Image objects — no figure is silently dropped
    assert all(isinstance(v, _FakeWandb.Image) for v in logged.values())
    # All keys are under plots/
    assert all(k.startswith("plots/") for k in logged.keys())


def test_log_benchmark_media_buffers_frames_for_video_generation(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )
    clear_benchmark_media_video_buffer()

    results = {
        "BOUNDARY_EXACTNESS": {
            "fuzzy_metrics": _boundary_landscape_fixture(),
        },
        "DIAGNOSTIC_DEPTH": {
            "position_bias_histogram_fn": [1] * 100,
            "position_bias_histogram_fp": [1] * 100,
        },
        "transition_failures": {
            0: np.array([
                [2, 1, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 0, 5, 0],
                [0, 0, 0, 0, 6],
            ]),
        },
        "false_transitions": _false_transition_fixture(),
    }

    log_benchmark_media(
        results,
        BEND_LABEL_CONFIG,
        step=1,
    )

    logged = log_benchmark_media_videos()

    assert set(logged.keys()) == {
        "plots/boundary_landscape_video",
        "plots/position_bias_video",
        "plots/transition_matrices_video",
        "plots/false_transitions_video",
    }
    for video in logged.values():
        assert isinstance(video, _FakeWandb.Video)
        assert video.data.ndim == 4
        assert video.data.shape[1] == 3


def test_log_benchmark_media_only_buffers_video_buffer_figures(monkeypatch):
    """Figures not in _VIDEO_BUFFER_FIGURE_KEYS are logged as images but not buffered."""
    from dna_segmentation_benchmark.wandb_logger import _BUFFERED_MEDIA_FRAMES, _VIDEO_BUFFER_FIGURE_KEYS

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )
    clear_benchmark_media_video_buffer()

    results = {
        "BOUNDARY_EXACTNESS": {
            "fuzzy_metrics": _boundary_landscape_fixture(),
        },
        "DIAGNOSTIC_DEPTH": {
            "position_bias_histogram_fn": [1] * 100,
            "position_bias_histogram_fp": [1] * 100,
        },
        "transition_failures": {
            0: np.array([
                [2, 1, 0, 0, 0],
                [0, 3, 0, 0, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 0, 5, 0],
                [0, 0, 0, 0, 6],
            ]),
        },
        "false_transitions": _false_transition_fixture(),
    }

    logged = log_benchmark_media(results, BEND_LABEL_CONFIG, step=1, method_prefix="val")

    # Extract the figure names that were buffered
    buffered_fig_names = {
        key.removeprefix("val/plots/") for key in _BUFFERED_MEDIA_FRAMES
    }
    # All buffered figures must be in the video-buffer allowlist
    assert buffered_fig_names <= _VIDEO_BUFFER_FIGURE_KEYS
    # All logged image keys must be under val/plots/
    assert all(k.startswith("val/plots/") for k in logged)


def test_log_benchmark_media_videos_normalizes_frame_shapes(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )
    clear_benchmark_media_video_buffer()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._BUFFERED_MEDIA_FRAMES",
        {
            "val/plots/position_bias": [
                np.zeros((20, 30, 3), dtype=np.uint8),
                np.zeros((10, 15, 3), dtype=np.uint8),
            ],
        },
    )

    logged = log_benchmark_media_videos()

    assert fake_wandb.logged[0]["step"] is None
    assert set(logged.keys()) == {"val/plots/position_bias_video"}
    video = logged["val/plots/position_bias_video"]
    assert isinstance(video, _FakeWandb.Video)
    assert video.fps == 2
    assert video.format == "gif"
    assert video.data.shape == (2, 3, 20, 30)


def test_init_wandb_with_presets_uses_metric_family_grouping(monkeypatch):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(
        "dna_segmentation_benchmark.wandb_logger._require_wandb",
        lambda: fake_wandb,
    )

    init_wandb_with_presets(
        project="demo-project",
        run_name="demo-run",
        label_config=BEND_LABEL_CONFIG,
        classes=[0, 2],
    )

    assert "region_discovery/*" in fake_wandb.defined_metrics
    assert "boundary_exactness/*" in fake_wandb.defined_metrics
    assert "struct_coherence/*" in fake_wandb.defined_metrics
    assert "diagnostic_depth/*" in fake_wandb.defined_metrics
    assert "nucleotide_classification/*" in fake_wandb.defined_metrics
    assert "val/region_discovery/*" in fake_wandb.defined_metrics
    assert "val/boundary_exactness/*" in fake_wandb.defined_metrics
    assert "val/struct_coherence/*" in fake_wandb.defined_metrics
    assert "plots/*" in fake_wandb.defined_metrics
    assert "val/*" in fake_wandb.defined_metrics
    assert "val/plots/*" in fake_wandb.defined_metrics


def test_compare_multiple_predictions_uses_generic_transition_key_for_single_method():
    results = {
        "single-model": {
            "transition_failures": {
                0: np.array([
                    [2, 1, 0, 0, 0],
                    [0, 3, 0, 0, 0],
                    [0, 0, 4, 0, 0],
                    [0, 0, 0, 5, 0],
                    [0, 0, 0, 0, 6],
                ]),
            },
            "false_transitions": {
                "late_catchup": {0: np.zeros((5, 5), dtype=np.int64)},
                "premature": {0: np.zeros((5, 5), dtype=np.int64)},
                "spurious": {0: np.zeros((5, 5), dtype=np.int64)},
                "stable_position_counts": {0: 10},
            },
        }
    }

    figures = compare_multiple_predictions(
        per_method_benchmark_res=results,
        label_config=BEND_LABEL_CONFIG,
        metrics_to_eval=[EvalMetrics.STATE_TRANSITIONS],
    )

    assert "transition_matrices" in figures
    assert "single-model_transition_matrices" not in figures


def test_compare_multiple_predictions_saves_boundary_landscape_for_single_method(tmp_path):
    results = {
        "single-model": {
            "BOUNDARY_EXACTNESS": {
                "fuzzy_metrics": _boundary_landscape_fixture(),
            },
        }
    }

    figures = compare_multiple_predictions(
        per_method_benchmark_res=results,
        label_config=BEND_LABEL_CONFIG,
        metrics_to_eval=[EvalMetrics.BOUNDARY_EXACTNESS],
        output_dir=tmp_path,
    )

    assert "boundary_landscape" in figures
    assert (tmp_path / "boundary_landscape.png").exists()
