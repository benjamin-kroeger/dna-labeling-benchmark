"""DNA Segmentation Benchmark – evaluate nucleotide-level predictions.

Public API
----------
.. autoclass:: LabelConfig
.. autoclass:: EvalMetrics
.. autofunction:: benchmark_gt_vs_pred_single
.. autofunction:: benchmark_gt_vs_pred_multiple
.. autofunction:: compare_multiple_predictions
"""

from .label_definition import LabelConfig, BEND_LABEL_CONFIG
from .eval.evaluate_predictors import (
    EvalMetrics,
    benchmark_gt_vs_pred_single,
    benchmark_gt_vs_pred_multiple,
)
from .plotting.summary_stat_plotting import compare_multiple_predictions

__all__ = [
    "LabelConfig",
    "BEND_LABEL_CONFIG",
    "EvalMetrics",
    "benchmark_gt_vs_pred_single",
    "benchmark_gt_vs_pred_multiple",
    "compare_multiple_predictions",
]
