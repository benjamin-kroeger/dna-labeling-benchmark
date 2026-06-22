"""W&B logging adapter for the DNA Segmentation Benchmark.

Provides plug-and-play Weights & Biases integration for two use cases:

1. **Online batch logging** – call :func:`log_benchmark_scalars` inside a
   training loop to track lightweight scalar metrics per step/epoch.
2. **Online visual tracking** – call :func:`log_benchmark_media` on the same
   aggregated benchmark results to attach stepwise figures, and optionally
   call :func:`log_benchmark_media_videos` later to flush the buffered figure
   history as GIF videos.

Use :func:`init_wandb_with_presets` to initialise a W&B run with
pre-configured metric groupings so the dashboard is organised from the start.

.. note::

   ``wandb`` is an **optional** dependency.  Install it with::

       pip install dna-segmentation-benchmark[wandb]
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from .label_definition import LabelConfig
from .plotting.summary_stat_plotting import compare_multiple_predictions
from .eval.evaluate_predictors import EvalMetrics

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

# Snake-case section names used as the W&B metric-path group segment.
# Keeps full keys URL/glob-friendly, e.g. ``val/struct_coherence/intron_chain/match_rate``.
_GROUP_DISPLAY_NAMES = {
    "REGION_DISCOVERY": "region_discovery",
    "BOUNDARY_EXACTNESS": "boundary_exactness",
    "NUCLEOTIDE_CLASSIFICATION": "nucleotide_classification",
    "INDEL": "indel",
    "PHASE_DRIFT": "phase_drift",
    "STRUCTURAL_COHERENCE": "struct_coherence",
    "DIAGNOSTIC_DEPTH": "diagnostic_depth",
}

# Training-time scalar subset: stable, high-signal metrics only.
# Path leaves may be scalar dict entries or numeric lists (auto-mean reduced).
_ONLINE_SCALAR_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "BOUNDARY_EXACTNESS": {
        "iou_mean": ("iou_stats", "mean"),
    },
    "REGION_DISCOVERY": {
        "neighborhood_hit/precision": ("neighborhood_hit", "precision"),
        "neighborhood_hit/recall": ("neighborhood_hit", "recall"),
        "internal_hit/precision": ("internal_hit", "precision"),
        "internal_hit/recall": ("internal_hit", "recall"),
        "full_coverage_hit/precision": ("full_coverage_hit", "precision"),
        "full_coverage_hit/recall": ("full_coverage_hit", "recall"),
        "perfect_boundary_hit/precision": ("perfect_boundary_hit", "precision"),
        "perfect_boundary_hit/recall": ("perfect_boundary_hit", "recall"),
    },
    "NUCLEOTIDE_CLASSIFICATION": {
        "nucleotide/precision": ("nucleotide", "precision"),
        "nucleotide/recall": ("nucleotide", "recall"),
        "nucleotide/f1": ("nucleotide", "f1"),
    },
    "STRUCTURAL_COHERENCE": {
        # Whole-chain tiers are all-or-nothing, so precision == recall == F1
        # (a chain mismatch is booked as both an FP and an FN). Log the single
        # match rate instead of duplicating it as precision and recall.
        "intron_chain/match_rate": ("intron_chain", "precision"),
        "exon_chain/match_rate": ("exon_chain", "precision"),
        "segment_count_delta/mean": ("segment_count_delta", "mean"),
        "segment_count_delta/mae": ("segment_count_delta", "mae"),
        "exon_recall_per_transcript/mean": ("exon_recall_per_transcript",),
        "hallucinated_exon_count_per_transcript/mean": ("hallucinated_exon_count_per_transcript",),
        "exact_match_rate": ("exact_match_rate",),
        "splice_site_results/donor_precision": ("splice_site_results", "donor_precision"),
        "splice_site_results/donor_recall": ("splice_site_results", "donor_recall"),
        "splice_site_results/acceptor_precision": ("splice_site_results", "acceptor_precision"),
        "splice_site_results/acceptor_recall": ("splice_site_results", "acceptor_recall"),
    },
    "DIAGNOSTIC_DEPTH": {
        "length_emd/mean": ("length_emd", "mean"),
        "length_emd/mae": ("length_emd", "mae"),
    },
}

# Figures to buffer for per-epoch GIF videos. Only short, informative plots
# are included — heavy multi-panel plots (indel histograms, splice-site CMs)
# grow too large to animate usefully.
_VIDEO_BUFFER_FIGURE_KEYS = frozenset({
    "boundary_landscape",
    "position_bias",
    "transition_matrices",
    "false_transitions",
    "transcript_match",
    "segment_count_delta",
    "phase_drift",
})

# Internal label used as the single-method key when feeding the multi-method
# plotting orchestrator. Surfaces in plot legends only.
_INTERNAL_METHOD_LABEL = "model"

_BUFFERED_MEDIA_FRAMES: dict[str, list[np.ndarray]] = {}
_DEFAULT_VIDEO_FPS = 2
_DEFAULT_VIDEO_FORMAT = "gif"


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


def _get_nested_scalar(data: dict, path: tuple[str, ...]) -> float | None:
    """Return a nested scalar value or ``None`` when the path is unavailable.

    If the leaf is a non-empty list of numeric values, its arithmetic mean is
    returned — this lets the spec format point to per-transcript distributions
    (e.g. ``exon_recall_per_transcript``) without an extra aggregation step.
    """
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, _SCALAR_TYPES):
        return float(current)
    if isinstance(current, list) and current and all(
        isinstance(v, _SCALAR_TYPES) for v in current
    ):
        return float(np.mean(current))
    return None


def _default_scope_name(group_data: dict) -> str | None:
    """Return the preferred default scope for one scoped metric group."""
    scopes = group_data.get("scopes")
    if not isinstance(scopes, dict) or not scopes:
        return None
    if "transcript_exon" in scopes:
        return "transcript_exon"
    return next(iter(scopes.keys()))


def _default_scope_payload(group_key: str, group_data: dict) -> dict:
    """Return a legacy-style default payload for one metric group."""
    scope_name = _default_scope_name(group_data)
    if scope_name is None:
        return group_data

    payload = dict(group_data["scopes"][scope_name])
    if group_key == "STRUCTURAL_COHERENCE":
        for key, value in group_data.items():
            if key not in {"scopes", "splice_site_results"}:
                payload[key] = value
    return payload


def _flatten_selected_scalars(
    results: dict,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten the online scalar allowlist into W&B-friendly key-value pairs."""
    flat: dict[str, float] = {}
    for group_key, metrics in _ONLINE_SCALAR_SPECS.items():
        group_data = results.get(group_key)
        if not isinstance(group_data, dict):
            continue
        group_payload = _default_scope_payload(group_key, group_data)
        group_display = _GROUP_DISPLAY_NAMES.get(group_key, group_key)
        section = f"{prefix}/{group_display}" if prefix else group_display
        for leaf_name, source_path in metrics.items():
            value = _get_nested_scalar(group_payload, source_path)
            if value is not None:
                flat[f"{section}/{leaf_name}"] = value
    return flat


def _unwrap_aggregated_results(results: dict) -> dict:
    """Accept either a raw benchmark result or a pipeline wrapper dict."""
    if set(results.keys()) == {"aggregated", "global"}:
        return results["aggregated"]
    return results


def _build_buffered_video_key(
    *,
    plot_name: str,
    method_prefix: str | None,
) -> str:
    """Build the final W&B key stem used for buffered plot videos."""
    key = f"plots/{plot_name}"
    if method_prefix:
        key = f"{method_prefix}/{key}"
    return key


def _render_benchmark_media_figures(
    results: dict,
    label_config: LabelConfig,
) -> dict[str, Any]:
    """Render the benchmark media figures for one aggregated result dict."""
    inner_results = _unwrap_aggregated_results(results)
    metrics_to_eval = _infer_metrics_from_results(inner_results)
    if not metrics_to_eval and "transition_failures" not in inner_results:
        return {}

    return compare_multiple_predictions(
        per_method_benchmark_res={_INTERNAL_METHOD_LABEL: inner_results},
        label_config=label_config,
        metrics_to_eval=metrics_to_eval,
    )


def _close_figures(figures: dict[str, Any]) -> None:
    """Close a figure dict returned by the plotting layer."""
    for fig in figures.values():
        plt.close(fig)


def _figure_to_rgb_frame(fig: Any) -> np.ndarray:
    """Render one matplotlib figure to a uint8 RGB image array."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def _pad_frames_to_common_shape(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Pad RGB frames to a common height/width with a white background."""
    max_height = max(frame.shape[0] for frame in frames)
    max_width = max(frame.shape[1] for frame in frames)
    padded: list[np.ndarray] = []
    for frame in frames:
        height, width, channels = frame.shape
        canvas = np.full((max_height, max_width, channels), 255, dtype=np.uint8)
        canvas[:height, :width, :] = frame
        padded.append(canvas)
    return padded


def _buffer_media_frames(
    figures: dict[str, Any],
    *,
    method_prefix: str | None,
) -> None:
    """Append figures in _VIDEO_BUFFER_FIGURE_KEYS to the internal video-frame buffer."""
    for fig_name, fig in figures.items():
        if fig_name not in _VIDEO_BUFFER_FIGURE_KEYS:
            continue
        buffer_key = _build_buffered_video_key(
            plot_name=fig_name,
            method_prefix=method_prefix,
        )
        _BUFFERED_MEDIA_FRAMES.setdefault(buffer_key, []).append(_figure_to_rgb_frame(fig))


def _infer_metrics_from_results(results: dict) -> list[EvalMetrics]:
    """Infer which metric groups are present in a flattened benchmark result."""
    metrics: list[EvalMetrics] = []
    metric_names = set(results.keys())
    for metric in EvalMetrics:
        if metric.name in metric_names:
            metrics.append(metric)
    return metrics


def _flatten_all_scalars(
    results: dict,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Flatten all scalar-like benchmark results for post-hoc logging."""
    flat: dict[str, float] = {}

    for group_key, group_data in results.items():
        if group_key in ("transition_failures", "false_transitions", "metadata"):
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
        Currently unused; kept for API compatibility.
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

    flat = _flatten_selected_scalars(results, prefix="")
    if method_prefix:
        flat = {f"{method_prefix}/{k}": v for k, v in flat.items()}

    log_kwargs: dict[str, Any] = {"data": flat}
    if step is not None:
        log_kwargs["step"] = step

    wandb.log(**log_kwargs)

    logger.info("Logged %d scalar metrics to W&B (step=%s).", len(flat), step)
    return flat


def log_benchmark_all_scalars(
    results: dict,
    label_config: LabelConfig,
    step: Optional[int] = None,
    method_prefix: Optional[str] = None,
) -> dict[str, float]:
    """Log every numeric scalar in the benchmark result to an active W&B run.

    Unlike :func:`log_benchmark_scalars` (which logs a curated high-signal
    subset), this function flattens the entire result dict recursively.
    Suitable for final evaluation runs or post-training analysis where
    logging cost is not a concern.

    Parameters
    ----------
    results : dict
        Aggregated result dict from :func:`benchmark_gt_vs_pred_multiple`.
        Pipeline wrapper ``{"aggregated": ..., "global": ...}`` is also
        accepted.
    label_config : LabelConfig
        Currently unused; kept for API compatibility.
    step : int, optional
        Training step or epoch number.
    method_prefix : str, optional
        Optional prefix to namespace the metrics (e.g., ``"final"``).

    Returns
    -------
    dict[str, float]
        The flat dict that was logged.
    """
    wandb = _require_wandb()

    inner = _unwrap_aggregated_results(results)
    flat = _flatten_all_scalars(inner, prefix="")
    if method_prefix:
        flat = {f"{method_prefix}/{k}": v for k, v in flat.items()}

    log_kwargs: dict[str, Any] = {"data": flat}
    if step is not None:
        log_kwargs["step"] = step

    wandb.log(**log_kwargs)

    logger.info("Logged %d scalar metrics to W&B (step=%s).", len(flat), step)
    return flat


def log_benchmark_media(
    results: dict,
    label_config: LabelConfig,
    step: Optional[int] = None,
    method_prefix: Optional[str] = None,
) -> dict[str, Any]:
    """Log all diagnostic plots to W&B as stepwise media history.

    Renders every figure produced by the benchmark plotting layer and logs
    them all as W&B Image panels.  A focused subset of high-value figures
    (listed in :data:`_VIDEO_BUFFER_FIGURE_KEYS`) is also buffered for later
    animation via :func:`log_benchmark_media_videos`.

    Parameters
    ----------
    results : dict
        Aggregated benchmark result dict from
        :func:`benchmark_gt_vs_pred_multiple`. A pipeline-style wrapper
        ``{"aggregated": ..., "global": ...}`` is also accepted.
    label_config : LabelConfig
        Label semantics for plot labelling.
    step : int, optional
        Training step or epoch number for W&B media history.
    method_prefix : str, optional
        Optional namespace prefix such as ``"val"``.

    Returns
    -------
    dict[str, Any]
        The logged W&B media payload (keys are W&B panel paths).
    """
    wandb = _require_wandb()

    figures = _render_benchmark_media_figures(results, label_config)
    try:
        media_payload: dict[str, Any] = {}
        for fig_name, fig in figures.items():
            key = f"plots/{fig_name}"
            if method_prefix:
                key = f"{method_prefix}/{key}"
            media_payload[key] = wandb.Image(fig)

        _buffer_media_frames(figures, method_prefix=method_prefix)

        if not media_payload:
            logger.info("No W&B media plots were generated from the benchmark results.")
            return {}

        log_kwargs: dict[str, Any] = {"data": media_payload}
        if step is not None:
            log_kwargs["step"] = step
        wandb.log(**log_kwargs)
        logger.info("Logged %d benchmark media panels to W&B (step=%s).", len(media_payload), step)
        return media_payload
    finally:
        _close_figures(figures)


def clear_benchmark_media_video_buffer() -> None:
    """Drop any buffered media frames without logging them."""
    _BUFFERED_MEDIA_FRAMES.clear()


def log_benchmark_media_videos() -> dict[str, Any]:
    """Log buffered benchmark media histories as W&B videos and clear the buffer."""
    wandb = _require_wandb()

    video_payload: dict[str, Any] = {}
    for plot_key, frames in _BUFFERED_MEDIA_FRAMES.items():
        if not frames:
            continue
        normalized_frames = _pad_frames_to_common_shape(frames)
        video_array = np.stack(normalized_frames, axis=0).transpose(0, 3, 1, 2)
        video_payload[f"{plot_key}_video"] = wandb.Video(
            video_array,
            fps=_DEFAULT_VIDEO_FPS,
            format=_DEFAULT_VIDEO_FORMAT,
        )

    if not video_payload:
        logger.info("No benchmark media videos were generated from the frame history.")
        return {}

    wandb.log(video_payload)
    logger.info("Logged %d benchmark media videos to W&B.", len(video_payload))
    _BUFFERED_MEDIA_FRAMES.clear()
    return video_payload


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
    dashboard sections grouped by metric family. This ensures that every
    new run gets an organised dashboard out of the box.

    Parameters
    ----------
    project : str
        W&B project name.
    run_name : str
        Display name for this run.
    label_config : LabelConfig
        Currently unused; kept for API compatibility.
    classes : list[int]
        Currently unused; kept for API compatibility.
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
    for group_display in _GROUP_DISPLAY_NAMES.values():
        wandb.define_metric(f"{group_display}/*")
        wandb.define_metric(f"val/{group_display}/*")

    wandb.define_metric("plots/*")
    wandb.define_metric("val/*")
    wandb.define_metric("val/plots/*")

    logger.info(
        "Initialised W&B run '%s' in project '%s' with metric-family W&B presets.",
        run_name,
        project,
    )

    return run
