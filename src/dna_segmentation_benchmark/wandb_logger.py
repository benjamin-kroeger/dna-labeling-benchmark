"""W&B logging adapter for the DNA Segmentation Benchmark.

Provides plug-and-play Weights & Biases integration for two use cases:

1. **Online batch logging** – call :func:`log_benchmark_scalars` inside a
   training loop to track lightweight scalar metrics per step/epoch.
2. **Post-hoc comparison** – call :func:`log_benchmark_full` after running
   :func:`compare_multiple_predictions` to log figures, tables, and all
   scalar metrics at once.

Use :func:`init_wandb_with_presets` to initialise a W&B run with
pre-configured metric groupings so the dashboard is organised from the start.

.. note::

   ``wandb`` is an **optional** dependency.  Install it with::

       pip install dna-segmentation-benchmark[wandb]
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .label_definition import LabelConfig

logger = logging.getLogger(__name__)


def _require_wandb():
    """Lazily import wandb or raise a clear error."""
    try:
        import wandb
        return wandb
    except ImportError as exc:
        raise ImportError(
            "The 'wandb' package is required for W&B logging but is not installed.\n"
            "Install it with:  pip install dna-segmentation-benchmark[wandb]"
        ) from exc


# ---------------------------------------------------------------------------
# Scalar flattening
# ---------------------------------------------------------------------------

_SCALAR_TYPES = (int, float, np.integer, np.floating)

# Human-readable section names for W&B dashboard grouping
_GROUP_DISPLAY_NAMES = {
    "REGION_DISCOVERY": "Region Discovery",
    "BOUNDARY_EXACTNESS": "Boundary Exactness",
    "NUCLEOTIDE_CLASSIFICATION": "Nucleotide Classification",
    "INDEL": "INDELs",
    "FRAMESHIFT": "Frameshift",
    "STRUCTURAL_COHERENCE": "Structural Coherence",
    "DIAGNOSTIC_DEPTH": "Diagnostic Depth",
}


def _flatten_leaf(
        data: dict,
        prefix: str,
) -> dict[str, float]:
    """Recursively flatten a dict to scalar key-value pairs."""
    flat: dict[str, float] = {}
    for key, value in data.items():
        path = f"{prefix}/{key}"
        if isinstance(value, _SCALAR_TYPES):
            flat[path] = float(value)
        elif isinstance(value, dict):
            flat.update(_flatten_leaf(value, prefix=path))
    return flat


def _flatten_scalars(
        results: dict,
        label_config: LabelConfig,
        prefix: str = "",
) -> dict[str, float]:
    """Flatten benchmark results into W&B-friendly scalar key-value pairs.

    The first ``/``-segment is the metric group display name
    (e.g. ``"Region Discovery"``), giving each metric group its own
    expandable section in the W&B dashboard.

    Parameters
    ----------
    results : dict
        Aggregated benchmark result dict as returned by
        :func:`benchmark_gt_vs_pred_multiple`.
    label_config : LabelConfig
        Currently unused; kept for API stability.
    prefix : str
        Optional prefix (e.g. ``"val"``) prepended to all keys.

    Returns
    -------
    dict[str, float]
        Flat mapping ready for ``wandb.log()``.
    """
    flat: dict[str, float] = {}

    for group_key, group_data in results.items():
        if group_key in ("transition_failures", "false_transitions"):
            continue
        if not isinstance(group_data, dict):
            continue

        group_display = _GROUP_DISPLAY_NAMES.get(group_key, group_key)
        section = f"{prefix}/{group_display}" if prefix else group_display
        flat.update(_flatten_leaf(group_data, prefix=section))

    return flat


# ---------------------------------------------------------------------------
# Public API: online logging
# ---------------------------------------------------------------------------


def log_benchmark_scalars(
        results: dict,
        label_config: LabelConfig,
        step: Optional[int] = None,
        method_prefix: Optional[str] = None,
) -> dict[str, float]:
    """Log scalar benchmark metrics to an active W&B run.

    Designed for **online evaluation during training**: call this after each
    evaluation batch with a lightweight subset of metrics.

    Parameters
    ----------
    results : dict
        Aggregated result dict from :func:`benchmark_gt_vs_pred_multiple`.
        Can contain any subset of metrics.
    label_config : LabelConfig
        Resolves label IDs to human-readable names.
    step : int, optional
        Training step or epoch number.  If ``None``, W&B uses its internal
        step counter.
    method_prefix : str, optional
        Optional prefix to namespace the metrics (e.g., ``"val"``).

    Returns
    -------
    dict[str, float]
        The flat dict that was logged, for inspection or further use.
    """
    wandb = _require_wandb()

    flat = _flatten_scalars(results, label_config)

    if method_prefix:
        flat = {f"{method_prefix}/{k}": v for k, v in flat.items()}

    log_kwargs: dict[str, Any] = {"data": flat}
    if step is not None:
        log_kwargs["step"] = step

    wandb.log(**log_kwargs)

    logger.info("Logged %d scalar metrics to W&B (step=%s).", len(flat), step)
    return flat


# ---------------------------------------------------------------------------
# Public API: post-hoc full logging
# ---------------------------------------------------------------------------


def log_benchmark_full(
        per_method_results: dict[str, dict],
        figures: dict[str, Any],
        label_config: LabelConfig,
) -> None:
    """Log a complete benchmark comparison to an active W&B run.

    Designed for **post-hoc model comparison**: call this after
    :func:`compare_multiple_predictions` with both the aggregated results
    and the generated figures.

    This logs:

    * All scalar metrics (per method, per metric group)
    * IoU score distributions as interactive ``wandb.Table`` objects
    * All matplotlib figures as ``wandb.Image`` panels

    Parameters
    ----------
    per_method_results : dict[str, dict]
        Outer key = method name, inner dict = aggregated benchmark result.
    figures : dict[str, Figure]
        Figures returned by :func:`compare_multiple_predictions`.
    label_config : LabelConfig
        Resolves label IDs to human-readable names.
    """
    wandb = _require_wandb()

    log_dict: dict[str, Any] = {}

    # ---- Scalar metrics per method --------------------------------------
    for method_name, results in per_method_results.items():
        flat = _flatten_scalars(results, label_config, prefix=method_name)
        log_dict.update(flat)

    # ---- IoU distributions as W&B Tables --------------------------------
    for method_name, results in per_method_results.items():
        be = results.get("BOUNDARY_EXACTNESS", {})
        iou_scores = be.get("iou_scores")
        if iou_scores is not None and len(iou_scores) > 0:
            table = wandb.Table(
                columns=["method", "iou"],
                data=[[method_name, float(s)] for s in iou_scores],
            )
            table_key = f"{method_name}/iou_distribution"
            log_dict[table_key] = table

    # ---- Figures as wandb.Image -----------------------------------------
    for fig_name, fig in figures.items():
        log_dict[f"plots/{fig_name}"] = wandb.Image(fig)

    wandb.log(log_dict)

    num_scalars = sum(1 for v in log_dict.values() if isinstance(v, _SCALAR_TYPES))
    num_tables = sum(1 for v in log_dict.values() if isinstance(v, wandb.Table))
    num_images = sum(1 for v in log_dict.values() if isinstance(v, wandb.Image))
    logger.info(
        "Logged to W&B: %d scalars, %d tables, %d images.",
        num_scalars, num_tables, num_images,
    )


# ---------------------------------------------------------------------------
# Public API: W&B initialisation with metric presets
# ---------------------------------------------------------------------------


def init_wandb_with_presets(
        project: str,
        run_name: str,
        label_config: LabelConfig,
        classes: list[int],
        config: Optional[dict] = None,
        **wandb_init_kwargs,
) -> Any:
    """Initialise a W&B run with pre-configured metric groupings.

    Calls ``wandb.init()`` and then ``wandb.define_metric()`` to set up
    dashboard sections grouped by class and metric type.  This ensures
    that every new run gets an organised dashboard out of the box.

    Parameters
    ----------
    project : str
        W&B project name.
    run_name : str
        Display name for this run.
    label_config : LabelConfig
        Used to derive class names for metric grouping.
    classes : list[int]
        Token values being evaluated, used to set up per-class metric groups.
    config : dict, optional
        Extra configuration to attach to the W&B run (e.g., hyperparameters).
    **wandb_init_kwargs
        Additional keyword arguments forwarded to ``wandb.init()``.

    Returns
    -------
    wandb.Run
        The initialised run object.
    """
    wandb = _require_wandb()

    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        **wandb_init_kwargs,
    )

    # ---- Define metric groupings for auto-dashboard layout ---------------
    # Section names use "CLASS · Group" format so each metric group
    # gets its own expandable section in the W&B panel list.
    for class_token in classes:
        class_name = label_config.name_of(class_token)
        for group_display in _GROUP_DISPLAY_NAMES.values():
            wandb.define_metric(f"{class_name} · {group_display}/*")

    # Validation-prefixed metrics (for online logging with method_prefix="val")
    wandb.define_metric("val/*")

    logger.info(
        "Initialised W&B run '%s' in project '%s' with metric presets for %d classes.",
        run_name, project, len(classes),
    )

    return run
