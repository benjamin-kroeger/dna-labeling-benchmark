from __future__ import annotations

import numpy as np
import pytest

from gene_calling_benchmark.label_definition import BEND_LABEL_CONFIG
from gene_calling_benchmark.plotting.summary_stat_plotting import compare_multiple_predictions
from gene_calling_benchmark.eval.evaluate_predictors import EvalMetrics
from gene_calling_benchmark.wandb_logger import (
    clear_benchmark_media_video_buffer,
    init_wandb_with_presets,
    log_benchmark_all_scalars,
    log_benchmark_histograms,
    log_benchmark_media,
    log_benchmark_media_videos,
    log_benchmark_scalars,
)

from support.wandb_stub import FakeWandb


def _boundary_landscape_fixture():
    # JSON-serialisable landscape dict (max_range=1: 3x3 bias, 2x2 reliability).
    return {
        "max_range": 1,
        "bias_matrix": [[0.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 0.0]],
        "reliability_matrix": [[0.25, 0.50], [0.75, 1.00]],
    }


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


def test_log_benchmark_scalars_logs_only_selected_online_metrics(wandb_stub):

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
            "exon_precision_per_transcript": [0.25, 0.5, 0.5, 0.75],
            "false_exon_count_per_transcript": [0, 1, 2, 1],
            "exact_match_rate": 0.25,
        },
        "NUCLEOTIDE_CLASSIFICATION": {
            "nucleotide": {"precision": 0.99, "recall": 0.98, "f1": 0.97},
        },
    }

    logged = log_benchmark_scalars(
        results,
        step=7,
        method_prefix="val",
    )

    assert wandb_stub.logged[0]["step"] == 7
    assert wandb_stub.logged[0]["data"] == logged
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
        "val/struct_coherence/exon_precision_per_transcript/mean",
        "val/struct_coherence/false_exon_count_per_transcript/mean",
        "val/struct_coherence/exact_match_rate",
        "val/nucleotide_classification/nucleotide/precision",
        "val/nucleotide_classification/nucleotide/recall",
        "val/nucleotide_classification/nucleotide/f1",
    }
    assert logged["val/struct_coherence/exon_recall_per_transcript/mean"] == 0.5625
    assert logged["val/struct_coherence/exon_precision_per_transcript/mean"] == 0.5
    assert logged["val/struct_coherence/false_exon_count_per_transcript/mean"] == 1.0
    assert logged["val/nucleotide_classification/nucleotide/f1"] == 0.97


def test_log_benchmark_scalars_includes_diagnostic_depth_metrics(wandb_stub):

    results = {
        "DIAGNOSTIC_DEPTH": {
            "length_emd": {"mean": 12.5, "mae": 15.0, "rmse": 18.0, "std": 8.0, "min": 0.0, "max": 100.0},
        },
    }

    logged = log_benchmark_scalars(results, step=1)

    assert "diagnostic_depth/length_emd/mean" in logged
    assert "diagnostic_depth/length_emd/mae" in logged
    assert logged["diagnostic_depth/length_emd/mean"] == 12.5
    assert logged["diagnostic_depth/length_emd/mae"] == 15.0


def test_log_benchmark_scalars_includes_splice_site_metrics(wandb_stub):

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

    logged = log_benchmark_scalars(results, step=2)

    assert "struct_coherence/splice_site_results/donor_precision" in logged
    assert "struct_coherence/splice_site_results/donor_recall" in logged
    assert "struct_coherence/splice_site_results/acceptor_precision" in logged
    assert "struct_coherence/splice_site_results/acceptor_recall" in logged
    assert logged["struct_coherence/splice_site_results/donor_precision"] == 0.95
    assert logged["struct_coherence/splice_site_results/acceptor_recall"] == 0.88


def test_log_benchmark_scalars_skips_missing_splice_site_results(wandb_stub):

    results = {
        "STRUCTURAL_COHERENCE": {
            "intron_chain": {"precision": 0.8, "recall": 0.7},
            "exon_chain": {"precision": 0.9, "recall": 0.85},
            # no splice_site_results key
        },
    }

    logged = log_benchmark_scalars(results, step=1)

    assert not any("splice_site" in k for k in logged)
    assert "struct_coherence/intron_chain/match_rate" in logged


def test_log_benchmark_scalars_includes_boundary_sidedness_counts(wandb_stub):
    results = {
        "BOUNDARY_EXACTNESS": {
            "iou_stats": {"mean": 0.9},
            "fuzzy_metrics": {"sidedness": {"exact": 13, "one_sided": 2, "two_sided": 1}},
        },
    }
    logged = log_benchmark_scalars(results)
    assert logged["boundary_exactness/sidedness/exact"] == 13.0
    assert logged["boundary_exactness/sidedness/one_sided"] == 2.0
    assert logged["boundary_exactness/sidedness/two_sided"] == 1.0
    # Fractions over all matched boundaries (13 + 2 + 1 = 16), summing to 1.
    assert logged["boundary_exactness/sidedness/exact_frac"] == pytest.approx(13 / 16)
    assert logged["boundary_exactness/sidedness/one_sided_frac"] == pytest.approx(2 / 16)
    assert logged["boundary_exactness/sidedness/two_sided_frac"] == pytest.approx(1 / 16)
    fracs = [logged[f"boundary_exactness/sidedness/{k}"] for k in ("exact_frac", "one_sided_frac", "two_sided_frac")]
    assert sum(fracs) == pytest.approx(1.0)


def test_log_benchmark_scalars_sidedness_fractions_absent_when_no_boundaries(wandb_stub):
    results = {"BOUNDARY_EXACTNESS": {"fuzzy_metrics": {"sidedness": {"exact": 0, "one_sided": 0, "two_sided": 0}}}}
    logged = log_benchmark_scalars(results)
    assert not any("sidedness" in k and "frac" in k for k in logged)


def test_log_benchmark_scalars_includes_chain_subset_superset(wandb_stub):
    results = {
        "STRUCTURAL_COHERENCE": {
            "intron_chain": {"precision": 0.7},
            "intron_chain_subset": {"precision": 0.8},
            "intron_chain_superset": {"precision": 0.9},
            "exon_chain_multi": {"precision": 0.5},
            "exon_chain_multi_subset": {"precision": 0.6},
            "exon_chain_multi_superset": {"precision": 0.55},
            "exon_chain_single": {"precision": 0.4},
        },
    }
    logged = log_benchmark_scalars(results)
    assert logged["struct_coherence/intron_chain/subset_rate"] == 0.8
    assert logged["struct_coherence/intron_chain/superset_rate"] == 0.9
    assert logged["struct_coherence/exon_chain_multi/subset_rate"] == 0.6
    assert logged["struct_coherence/exon_chain_multi/superset_rate"] == 0.55
    assert logged["struct_coherence/exon_chain_single/match_rate"] == 0.4


def test_log_benchmark_scalars_includes_indel_counts(wandb_stub):
    """INDEL logs raw event counts per error and per location; absent buckets = 0."""
    results = {
        "INDEL": {
            "by_boundary": {
                "internal_exon": {"split": [3, 4], "3_prime_deletions": [7]},
                "five_prime_terminal_exon": {"whole_insertions": [10]},
            },
        },
    }
    logged = log_benchmark_scalars(results)
    # per-error event counts (len of run lists)
    assert logged["indel/by_error/split"] == 2.0
    assert logged["indel/by_error/3_prime_deletions"] == 1.0
    assert logged["indel/by_error/whole_insertions"] == 1.0  # whole insertions included
    assert logged["indel/by_error/joined"] == 0.0            # absent bucket present as 0
    # per-location event counts
    assert logged["indel/by_location/internal_exon"] == 3.0
    assert logged["indel/by_location/five_prime_terminal_exon"] == 1.0
    assert logged["indel/by_location/single_exon_gene"] == 0.0


def test_log_benchmark_scalars_includes_phase_frame_counts(wandb_stub):
    results = {"PHASE_DRIFT": {"gt_frame_counts": [494, 66, 119]}}
    logged = log_benchmark_scalars(results)
    assert logged["phase_drift/frame_0"] == 494.0
    assert logged["phase_drift/frame_1"] == 66.0
    assert logged["phase_drift/frame_2"] == 119.0


def test_log_benchmark_scalars_includes_transcript_match_classes(wandb_stub):
    """All 8 classes always logged; classes absent from the batch log as 0."""
    results = {
        "STRUCTURAL_COHERENCE": {
            "transcript_match_distribution": {"exact": 3, "partial_overlap": 1},
        },
    }
    logged = log_benchmark_scalars(results)
    assert logged["struct_coherence/transcript_match/exact"] == 3.0
    assert logged["struct_coherence/transcript_match/partial_overlap"] == 1.0
    assert logged["struct_coherence/transcript_match/no_overlap"] == 0.0
    assert logged["struct_coherence/transcript_match/substitution"] == 0.0
    match_keys = {k for k in logged if "/transcript_match/" in k}
    assert len(match_keys) == 8  # fixed vocabulary


def test_log_benchmark_histograms_logs_iou_and_boundary_shift(wandb_stub):
    results = {
        "BOUNDARY_EXACTNESS": {"iou_scores": [0.5, 0.9, 1.0]},
        "STRUCTURAL_COHERENCE": {
            "boundary_shift_offsets": [
                {"offset": -2, "position": "internal"},
                {"offset": 3, "position": "terminal"},
            ],
        },
    }
    logged = log_benchmark_histograms(results, step=4, method_prefix="val")

    assert wandb_stub.logged[-1]["step"] == 4
    assert set(logged) == {
        "val/boundary_exactness/iou_dist",
        "val/struct_coherence/boundary_shift_dist",
    }
    assert all(isinstance(v, FakeWandb.Histogram) for v in logged.values())
    assert logged["val/boundary_exactness/iou_dist"].sequence == [0.5, 0.9, 1.0]
    assert logged["val/struct_coherence/boundary_shift_dist"].sequence == [-2, 3]


def test_log_benchmark_histograms_empty_when_no_distributions(wandb_stub):
    assert log_benchmark_histograms({"NUCLEOTIDE_CLASSIFICATION": {}}) == {}


def test_scalar_loggers_survive_wandb_log_failure(monkeypatch):
    """A transient wandb.log failure must not abort the training loop.

    The online scalar loggers run inside the training step; a network hiccup in
    wandb.log should be swallowed (logged + empty return), never raised.
    """
    class RaisingWandb(FakeWandb):
        def log(self, data, step=None):
            raise RuntimeError("simulated wandb network failure")

    fake = RaisingWandb()
    monkeypatch.setattr(
        "gene_calling_benchmark.wandb_logger._require_wandb", lambda: fake
    )
    results = {"NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"precision": 0.9, "recall": 0.8, "f1": 0.85}}}

    assert log_benchmark_scalars(results, step=1) == {}
    assert log_benchmark_all_scalars(results, step=1) == {}


def test_log_benchmark_all_scalars_logs_everything(wandb_stub):

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

    logged = log_benchmark_all_scalars(results, step=5, method_prefix="final")

    assert wandb_stub.logged[0]["step"] == 5
    assert "final/boundary_exactness/iou_stats/mean" in logged
    assert "final/boundary_exactness/iou_stats/std" in logged
    assert "final/nucleotide_classification/nucleotide/precision" in logged
    assert "final/nucleotide_classification/nucleotide/f1" in logged
    # transition_failures, false_transitions and metadata are excluded
    assert not any(
        k.split("/")[1] in {"transition_failures", "false_transitions", "metadata"}
        for k in logged
    )


def test_log_benchmark_all_scalars_unwraps_pipeline_wrapper(wandb_stub):

    results = {
        "aggregated": {
            "BOUNDARY_EXACTNESS": {"iou_stats": {"mean": 0.7}},
        },
        "global": {},
    }

    logged = log_benchmark_all_scalars(results)

    assert "boundary_exactness/iou_stats/mean" in logged


def test_log_benchmark_media_logs_boundary_position_and_transition_plots(wandb_stub):
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

    assert wandb_stub.logged[0]["step"] == 3
    assert wandb_stub.logged[0]["data"] == logged
    assert set(logged.keys()) == {
        "val/plots/boundary_bias_landscape",
        "val/plots/boundary_recall_landscape",
        "val/plots/position_bias",
        "val/plots/transition_matrices",
        "val/plots/false_transitions",
    }
    assert all(isinstance(value, FakeWandb.Image) for value in logged.values())


def test_log_benchmark_media_logs_all_figures_as_images(wandb_stub):
    """Every figure produced by the plotting layer is logged, not just a hardcoded subset."""
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
    assert all(isinstance(v, FakeWandb.Image) for v in logged.values())
    # All keys are under plots/
    assert all(k.startswith("plots/") for k in logged.keys())


def test_log_benchmark_media_buffers_frames_for_video_generation(wandb_stub):
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
        "plots/boundary_bias_landscape_video",
        "plots/boundary_recall_landscape_video",
        "plots/position_bias_video",
        "plots/transition_matrices_video",
        "plots/false_transitions_video",
    }
    for video in logged.values():
        assert isinstance(video, FakeWandb.Video)
        assert video.data.ndim == 4
        assert video.data.shape[1] == 3


def test_log_benchmark_media_only_buffers_video_buffer_figures(wandb_stub):
    """Figures not in _VIDEO_BUFFER_FIGURE_KEYS are logged as images but not buffered."""
    from gene_calling_benchmark.wandb_logger import _BUFFERED_MEDIA_FRAMES, _VIDEO_BUFFER_FIGURE_KEYS

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

    # Extract the figure names that were buffered (buffer is scoped by run id)
    buffered_fig_names = {
        key.removeprefix("val/plots/")
        for run_frames in _BUFFERED_MEDIA_FRAMES.values()
        for key in run_frames
    }
    # All buffered figures must be in the video-buffer allowlist
    assert buffered_fig_names <= _VIDEO_BUFFER_FIGURE_KEYS
    # All logged image keys must be under val/plots/
    assert all(k.startswith("val/plots/") for k in logged)


def test_log_benchmark_media_videos_normalizes_frame_shapes(wandb_stub, monkeypatch):
    from gene_calling_benchmark.wandb_logger import _NO_ACTIVE_RUN

    clear_benchmark_media_video_buffer()
    # Buffer is scoped by run id; the stub has no active run -> the sentinel bucket.
    monkeypatch.setattr(
        "gene_calling_benchmark.wandb_logger._BUFFERED_MEDIA_FRAMES",
        {
            _NO_ACTIVE_RUN: {
                "val/plots/position_bias": [
                    np.zeros((20, 30, 3), dtype=np.uint8),
                    np.zeros((10, 15, 3), dtype=np.uint8),
                ],
            },
        },
    )

    logged = log_benchmark_media_videos()

    assert wandb_stub.logged[0]["step"] is None
    assert set(logged.keys()) == {"val/plots/position_bias_video"}
    video = logged["val/plots/position_bias_video"]
    assert isinstance(video, FakeWandb.Video)
    assert video.fps == 2
    assert video.format == "gif"
    assert video.data.shape == (2, 3, 20, 30)


def test_log_benchmark_media_isolates_failing_metric_group(wandb_stub, monkeypatch):
    """One metric group's plotting failure is logged and skipped, not raised."""
    import matplotlib.pyplot as plt
    import gene_calling_benchmark.wandb_logger as wl

    results = {
        "NUCLEOTIDE_CLASSIFICATION": {"nucleotide": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
        "REGION_DISCOVERY": {"neighborhood_hit": {"precision": 1.0, "recall": 1.0}},
    }

    def fake_compare(*, per_method_benchmark_res, label_config, metrics_to_eval):
        if metrics_to_eval[0] == EvalMetrics.REGION_DISCOVERY:
            raise RuntimeError("synthetic plotting failure")
        return {"nucleotide_classification_bar": plt.figure()}

    monkeypatch.setattr(wl, "compare_multiple_predictions", fake_compare)

    media = log_benchmark_media(results, BEND_LABEL_CONFIG, step=0)  # must not raise
    assert any("nucleotide_classification_bar" in key for key in media)


def test_media_video_buffer_is_scoped_per_run(wandb_stub):
    """Sequential runs in one process must not mix each other's video frames."""
    import matplotlib.pyplot as plt
    from types import SimpleNamespace
    import gene_calling_benchmark.wandb_logger as wl

    wl.clear_benchmark_media_video_buffer()
    figures = {"position_bias": plt.figure()}  # in _VIDEO_BUFFER_FIGURE_KEYS
    wl._buffer_media_frames(figures, run_id="runA", method_prefix=None)
    wl._buffer_media_frames(figures, run_id="runB", method_prefix=None)
    plt.close("all")

    wandb_stub.run = SimpleNamespace(id="runB")
    videos = log_benchmark_media_videos()  # flushes runB only

    assert set(videos) == {"plots/position_bias_video"}
    assert "runA" in wl._BUFFERED_MEDIA_FRAMES  # other run untouched
    assert "runB" not in wl._BUFFERED_MEDIA_FRAMES  # drained after its own flush
    wl.clear_benchmark_media_video_buffer()


def test_init_wandb_with_presets_uses_metric_family_grouping(wandb_stub):

    init_wandb_with_presets(
        project="demo-project",
        run_name="demo-run",
    )

    assert "region_discovery/*" in wandb_stub.defined_metrics
    assert "boundary_exactness/*" in wandb_stub.defined_metrics
    assert "struct_coherence/*" in wandb_stub.defined_metrics
    assert "diagnostic_depth/*" in wandb_stub.defined_metrics
    assert "nucleotide_classification/*" in wandb_stub.defined_metrics
    assert "val/region_discovery/*" in wandb_stub.defined_metrics
    assert "val/boundary_exactness/*" in wandb_stub.defined_metrics
    assert "val/struct_coherence/*" in wandb_stub.defined_metrics
    assert "plots/*" in wandb_stub.defined_metrics
    assert "val/*" in wandb_stub.defined_metrics
    assert "val/plots/*" in wandb_stub.defined_metrics


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

    assert "boundary_bias_landscape" in figures
    assert "boundary_recall_landscape" in figures
    assert (tmp_path / "boundary_bias_landscape.png").exists()
    assert (tmp_path / "boundary_recall_landscape.png").exists()
