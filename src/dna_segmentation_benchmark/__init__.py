"""DNA Segmentation Benchmark – evaluate nucleotide-level predictions.

Public API
----------
.. autoclass:: LabelConfig
.. autoclass:: EvalMetrics
.. autofunction:: benchmark_gt_vs_pred_single
.. autofunction:: benchmark_gt_vs_pred_multiple
.. autofunction:: compare_multiple_predictions
.. autofunction:: log_benchmark_scalars
.. autofunction:: log_benchmark_full
.. autofunction:: init_wandb_with_presets
"""

from .label_definition import LabelConfig, BEND_LABEL_CONFIG
from .eval.evaluate_predictors import (
    EvalMetrics,
    benchmark_gt_vs_pred_single,
    benchmark_gt_vs_pred_multiple,
)
from .plotting.summary_stat_plotting import compare_multiple_predictions
from .wandb_logger import (
    log_benchmark_scalars,
    log_benchmark_full,
    init_wandb_with_presets,
)

__all__ = [
    "LabelConfig",
    "BEND_LABEL_CONFIG",
    "EvalMetrics",
    "benchmark_gt_vs_pred_single",
    "benchmark_gt_vs_pred_multiple",
    "compare_multiple_predictions",
    "log_benchmark_scalars",
    "log_benchmark_full",
    "init_wandb_with_presets",
]
