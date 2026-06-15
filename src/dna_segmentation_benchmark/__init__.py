"""DNA Segmentation Benchmark – evaluate nucleotide-level predictions.

Public API
----------
.. autoclass:: AnnotationMode
.. autoclass:: BenchmarkScope
.. autoclass:: LabelConfig
.. autoclass:: EvalMetrics
.. autofunction:: benchmark_gt_vs_pred_single
.. autofunction:: benchmark_gt_vs_pred_multiple
.. autofunction:: benchmark_from_gff
.. autofunction:: load_dataset
.. autofunction:: list_datasets
.. autofunction:: get_dataset_info
.. autofunction:: compare_multiple_predictions
.. autofunction:: log_benchmark_scalars
.. autofunction:: log_benchmark_all_scalars
.. autofunction:: log_benchmark_media
.. autofunction:: log_benchmark_media_videos
.. autofunction:: clear_benchmark_media_video_buffer
.. autofunction:: init_wandb_with_presets
"""

from .label_definition import (
    AnnotationMode,
    BenchmarkScope,
    LabelConfig,
    BEND_LABEL_CONFIG,
)
from .eval.evaluate_predictors import (
    EvalMetrics,
    benchmark_gt_vs_pred_single,
    benchmark_gt_vs_pred_multiple,
)
from .plotting.summary_stat_plotting import compare_multiple_predictions
from .wandb_logger import (
    clear_benchmark_media_video_buffer,
    log_benchmark_scalars,
    log_benchmark_all_scalars,
    log_benchmark_media,
    log_benchmark_media_videos,
    init_wandb_with_presets,
)
from .pipeline import benchmark_from_gff
from .transcript_mapping import LocusMatchingMode
from .datasets import (
    LoadedDataset,
    get_dataset_info,
    list_datasets,
    load_dataset,
)

__all__ = [
    "LabelConfig",
    "AnnotationMode",
    "BenchmarkScope",
    "BEND_LABEL_CONFIG",
    "EvalMetrics",
    "LocusMatchingMode",
    "benchmark_gt_vs_pred_single",
    "benchmark_gt_vs_pred_multiple",
    "benchmark_from_gff",
    "load_dataset",
    "list_datasets",
    "get_dataset_info",
    "LoadedDataset",
    "compare_multiple_predictions",
    "clear_benchmark_media_video_buffer",
    "log_benchmark_scalars",
    "log_benchmark_all_scalars",
    "log_benchmark_media",
    "log_benchmark_media_videos",
    "init_wandb_with_presets",
]
