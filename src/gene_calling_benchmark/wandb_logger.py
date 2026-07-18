"""W&B logging adapter for the Gene Calling Benchmark.

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

       pip install gene-calling-benchmark[wandb]
"""

from __future__ import annotations

import logging
from typing import Any

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
            "Install it with:  pip install gene-calling-benchmark[wandb]"
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
        # Boundary-error sidedness (raw counts): how many matched boundaries were
        # exact, off on one end, or off on both. One line each in the workspace.
        "sidedness/exact": ("fuzzy_metrics", "sidedness", "exact"),
        "sidedness/one_sided": ("fuzzy_metrics", "sidedness", "one_sided"),
        "sidedness/two_sided": ("fuzzy_metrics", "sidedness", "two_sided"),
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
        "intron_chain/subset_rate": ("intron_chain_subset", "precision"),
        "intron_chain/superset_rate": ("intron_chain_superset", "precision"),
        "exon_chain/match_rate": ("exon_chain", "precision"),
        "exon_chain_multi/match_rate": ("exon_chain_multi", "precision"),
        "exon_chain_multi/subset_rate": ("exon_chain_multi_subset", "precision"),
        "exon_chain_multi/superset_rate": ("exon_chain_multi_superset", "precision"),
        "exon_chain_single/match_rate": ("exon_chain_single", "precision"),
        "segment_count_delta/mean": ("segment_count_delta", "mean"),
        "segment_count_delta/mae": ("segment_count_delta", "mae"),
        "exon_recall_per_transcript/mean": ("exon_recall_per_transcript",),
        "exon_precision_per_transcript/mean": ("exon_precision_per_transcript",),
        "false_exon_count_per_transcript/mean": ("false_exon_count_per_transcript",),
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

# Count-based online metrics the path-tuple spec format cannot express (they need
# summation over nested lists or a fixed key vocabulary). Logged as RAW COUNTS per
# eval batch: hold the eval set constant across steps for the lines to be
# comparable (the usual ML val-set caveat).
_INDEL_ERROR_BUCKETS = (
    "5_prime_extensions", "3_prime_extensions", "joined", "whole_insertions",
    "5_prime_deletions", "3_prime_deletions", "split", "whole_deletions",
)
_INDEL_LOCATIONS = (
    "five_prime_terminal_exon", "internal_exon",
    "three_prime_terminal_exon", "single_exon_gene",
)
_TRANSCRIPT_MATCH_CLASSES = (
    "exact", "boundary_shift_internal", "boundary_shift_terminal",
    "missing_segments", "extra_segments", "partial_overlap",
    "substitution", "no_overlap",
)

# Figures to buffer for per-epoch GIF videos. Only short, informative plots
# are included — heavy multi-panel plots (indel histograms, splice-site CMs)
# grow too large to animate usefully.
_VIDEO_BUFFER_FIGURE_KEYS = frozenset({
    "boundary_bias_landscape",
    "boundary_recall_landscape",
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

# Buffered video frames, scoped by active W&B run id: ``{run_id: {plot_key: [frames]}}``.
# Scoping by run keeps a sweep's sequential runs (same process) from concatenating
# each other's frames into one video — each run flushes only its own bucket.
_BUFFERED_MEDIA_FRAMES: dict[str, dict[str, list[np.ndarray]]] = {}
_NO_ACTIVE_RUN = "__no_active_run__"
_DEFAULT_VIDEO_FPS = 2
_DEFAULT_VIDEO_FORMAT = "gif"


def _active_run_id(wandb) -> str:
    """Return the current W&B run id, or a sentinel when no run is active."""
    run = getattr(wandb, "run", None)
    return run.id if run is not None else _NO_ACTIVE_RUN


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


def _flatten_indel_counts(indel: dict, section: str) -> dict[str, float]:
    """Raw INDEL event counts per error type and per exon location.

    ``by_boundary`` is ``{location: {error: [run_lengths]}}``; the count is the
    number of runs (events), summed across the other axis. Absent buckets log 0
    so every workspace line exists from step 0.
    """
    by_boundary = indel.get("by_boundary")
    if not isinstance(by_boundary, dict):
        return {}
    by_error = {e: 0 for e in _INDEL_ERROR_BUCKETS}
    by_location = {loc: 0 for loc in _INDEL_LOCATIONS}
    for location, errors in by_boundary.items():
        if not isinstance(errors, dict):
            continue
        for error, runs in errors.items():
            n = len(runs) if isinstance(runs, list) else 0
            if error in by_error:
                by_error[error] += n
            if location in by_location:
                by_location[location] += n
    flat = {f"{section}/by_error/{e}": float(c) for e, c in by_error.items()}
    flat.update({f"{section}/by_location/{loc}": float(c) for loc, c in by_location.items()})
    return flat


def _flatten_phase_frame_counts(phase_drift: dict, section: str) -> dict[str, float]:
    """Raw per-phase coding-base counts (``frame_0``/``1``/``2``)."""
    counts = phase_drift.get("gt_frame_counts")
    if not isinstance(counts, list):
        return {}
    return {f"{section}/frame_{i}": float(c) for i, c in enumerate(counts)}


def _flatten_transcript_match_counts(struct: dict, section: str) -> dict[str, float]:
    """Raw per-class transcript-match counts over the fixed class vocabulary."""
    payload = _default_scope_payload("STRUCTURAL_COHERENCE", struct)
    dist = payload.get("transcript_match_distribution")
    if not isinstance(dist, dict):
        return {}
    return {
        f"{section}/transcript_match/{cls}": float(dist.get(cls, 0))
        for cls in _TRANSCRIPT_MATCH_CLASSES
    }


def _flatten_sidedness_fractions(boundary_exactness: dict, section: str) -> dict[str, float]:
    """Boundary-error sidedness as fractions of all matched boundaries (sum to 1).

    ``exact`` / ``one_sided`` / ``two_sided`` counts (from the boundary-precision
    sidedness block) divided by their sum, so one plot shows the composition of
    matched boundaries. Returns nothing when no boundary matched.
    """
    payload = _default_scope_payload("BOUNDARY_EXACTNESS", boundary_exactness)
    fuzzy = payload.get("fuzzy_metrics")
    sided = fuzzy.get("sidedness") if isinstance(fuzzy, dict) else None
    if not isinstance(sided, dict):
        return {}
    exact = sided.get("exact", 0) or 0
    one_sided = sided.get("one_sided", 0) or 0
    two_sided = sided.get("two_sided", 0) or 0
    total = exact + one_sided + two_sided
    if total <= 0:
        return {}
    return {
        f"{section}/sidedness/exact_frac": exact / total,
        f"{section}/sidedness/one_sided_frac": one_sided / total,
        f"{section}/sidedness/two_sided_frac": two_sided / total,
    }


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

    # Count-based groups: raw counts the path-tuple format can't express.
    def _section(group_key: str) -> str:
        group_display = _GROUP_DISPLAY_NAMES.get(group_key, group_key)
        return f"{prefix}/{group_display}" if prefix else group_display

    indel = results.get("INDEL")
    if isinstance(indel, dict):
        flat.update(_flatten_indel_counts(indel, _section("INDEL")))
    phase_drift = results.get("PHASE_DRIFT")
    if isinstance(phase_drift, dict):
        flat.update(_flatten_phase_frame_counts(phase_drift, _section("PHASE_DRIFT")))
    struct = results.get("STRUCTURAL_COHERENCE")
    if isinstance(struct, dict):
        flat.update(_flatten_transcript_match_counts(struct, _section("STRUCTURAL_COHERENCE")))
    boundary = results.get("BOUNDARY_EXACTNESS")
    if isinstance(boundary, dict):
        flat.update(_flatten_sidedness_fractions(boundary, _section("BOUNDARY_EXACTNESS")))
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
    """Render the benchmark media figures for one aggregated result dict.

    Each metric group is rendered in its own protected call, so one group's
    plotting failure is logged and skipped rather than aborting the whole media
    log — and, in an online setting, the caller's training loop. Isolation is
    per metric group (not per individual plot) to avoid scattering error
    handling through the shared plotting orchestrator.
    """
    inner_results = _unwrap_aggregated_results(results)
    metrics_to_eval = _infer_metrics_from_results(inner_results)
    if not metrics_to_eval:
        return {}

    figures: dict[str, Any] = {}
    for metric in metrics_to_eval:
        try:
            figures.update(
                compare_multiple_predictions(
                    per_method_benchmark_res={_INTERNAL_METHOD_LABEL: inner_results},
                    label_config=label_config,
                    metrics_to_eval=[metric],
                )
            )
        except Exception:
            logger.exception("Rendering media for %s failed; skipping its plots.", metric.name)
    return figures


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
    run_id: str,
    method_prefix: str | None,
) -> None:
    """Append figures in _VIDEO_BUFFER_FIGURE_KEYS to the active run's frame buffer."""
    run_buffer = _BUFFERED_MEDIA_FRAMES.setdefault(run_id, {})
    for fig_name, fig in figures.items():
        if fig_name not in _VIDEO_BUFFER_FIGURE_KEYS:
            continue
        buffer_key = _build_buffered_video_key(
            plot_name=fig_name,
            method_prefix=method_prefix,
        )
        try:
            frame = _figure_to_rgb_frame(fig)
        except Exception:
            logger.exception("Failed to rasterise figure %r for the video buffer; skipping it.", fig_name)
            continue
        run_buffer.setdefault(buffer_key, []).append(frame)


def _infer_metrics_from_results(results: dict) -> list[EvalMetrics]:
    """Infer which metric groups are present in a flattened benchmark result."""
    metrics: list[EvalMetrics] = []
    metric_names = set(results.keys())
    for metric in EvalMetrics:
        if metric.name in metric_names:
            metrics.append(metric)
    # State transitions are keyed by their fragment names, not the enum name, so
    # detect them directly to keep the transition plots rendering when present.
    if ("transition_failures" in metric_names or "false_transitions" in metric_names) and (
        EvalMetrics.STATE_TRANSITIONS not in metrics
    ):
        metrics.append(EvalMetrics.STATE_TRANSITIONS)
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
    step: int | None = None,
    method_prefix: str | None = None,
) -> dict[str, float]:
    """Log scalar benchmark metrics to an active W&B run.

    Designed for **online evaluation during training**: call this after each
    evaluation batch with a lightweight subset of metrics.

    Parameters
    ----------
    results : dict
        Aggregated result dict from :func:`benchmark_from_arrays`.
        Can contain any subset of metrics.
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

    # Online path: a transient wandb.log failure must never abort the training
    # loop it runs inside (same guard as the histogram/media loggers).
    try:
        wandb.log(**log_kwargs)
    except Exception:
        logger.exception("wandb.log failed for %d scalar metrics; continuing.", len(flat))
        return {}

    logger.info("Logged %d scalar metrics to W&B (step=%s).", len(flat), step)
    return flat


def log_benchmark_all_scalars(
    results: dict,
    step: int | None = None,
    method_prefix: str | None = None,
) -> dict[str, float]:
    """Log every numeric scalar in the benchmark result to an active W&B run.

    Unlike :func:`log_benchmark_scalars` (which logs a curated high-signal
    subset), this function flattens the entire result dict recursively.
    Suitable for final evaluation runs or post-training analysis where
    logging cost is not a concern.

    Parameters
    ----------
    results : dict
        Aggregated result dict from :func:`benchmark_from_arrays`.
        Pipeline wrapper ``{"aggregated": ..., "global": ...}`` is also
        accepted.
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

    # Online path: a transient wandb.log failure must never abort the training
    # loop it runs inside (same guard as the histogram/media loggers).
    try:
        wandb.log(**log_kwargs)
    except Exception:
        logger.exception("wandb.log failed for %d scalar metrics; continuing.", len(flat))
        return {}

    logger.info("Logged %d scalar metrics to W&B (step=%s).", len(flat), step)
    return flat


def _collect_distributions(inner: dict) -> dict[str, list]:
    """Extract the numeric-distribution lists to log as native histograms."""
    dists: dict[str, list] = {}
    boundary = inner.get("BOUNDARY_EXACTNESS")
    if isinstance(boundary, dict):
        iou = boundary.get("iou_scores")
        if isinstance(iou, list) and iou:
            dists["boundary_exactness/iou_dist"] = iou
    struct = inner.get("STRUCTURAL_COHERENCE")
    if isinstance(struct, dict):
        offsets = [
            record["offset"]
            for record in struct.get("boundary_shift_offsets", [])
            if isinstance(record, dict) and "offset" in record
        ]
        if offsets:
            dists["struct_coherence/boundary_shift_dist"] = offsets
    return dists


def log_benchmark_histograms(
    results: dict,
    step: int | None = None,
    method_prefix: str | None = None,
) -> dict[str, Any]:
    """Log per-transcript distributions as native ``wandb.Histogram`` panels.

    Renders the IoU-score and boundary-shift-offset distributions as W&B
    histograms so their shape can be tracked over training without the
    matplotlib/video render path.  Call it alongside
    :func:`log_benchmark_scalars` each eval step.

    Parameters
    ----------
    results : dict
        Aggregated result dict from :func:`benchmark_from_arrays`.  A pipeline
        wrapper ``{"aggregated": ..., "global": ...}`` is also accepted.
    step : int, optional
        Training step or epoch number.
    method_prefix : str, optional
        Optional prefix to namespace the metrics (e.g., ``"val"``).

    Returns
    -------
    dict[str, Any]
        The histogram payload that was logged (keys are W&B panel paths).
    """
    wandb = _require_wandb()

    inner = _unwrap_aggregated_results(results)
    payload: dict[str, Any] = {
        key: wandb.Histogram(values) for key, values in _collect_distributions(inner).items()
    }
    if method_prefix:
        payload = {f"{method_prefix}/{k}": v for k, v in payload.items()}

    if not payload:
        logger.info("No benchmark distributions available to log as histograms.")
        return {}

    log_kwargs: dict[str, Any] = {"data": payload}
    if step is not None:
        log_kwargs["step"] = step
    try:
        wandb.log(**log_kwargs)
    except Exception:
        logger.exception("wandb.log failed for %d histograms; continuing.", len(payload))
        return {}

    logger.info("Logged %d benchmark histograms to W&B (step=%s).", len(payload), step)
    return payload


def log_benchmark_media(
    results: dict,
    label_config: LabelConfig,
    step: int | None = None,
    method_prefix: str | None = None,
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
        :func:`benchmark_from_arrays`. A pipeline-style wrapper
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
            try:
                media_payload[key] = wandb.Image(fig)
            except Exception:
                logger.exception("Failed to prepare figure %r for W&B; skipping it.", fig_name)

        _buffer_media_frames(figures, run_id=_active_run_id(wandb), method_prefix=method_prefix)

        if not media_payload:
            logger.info("No W&B media plots were generated from the benchmark results.")
            return {}

        log_kwargs: dict[str, Any] = {"data": media_payload}
        if step is not None:
            log_kwargs["step"] = step
        try:
            wandb.log(**log_kwargs)
        except Exception:
            logger.exception("wandb.log failed for %d media panels; continuing.", len(media_payload))
            return {}
        logger.info("Logged %d benchmark media panels to W&B (step=%s).", len(media_payload), step)
        return media_payload
    finally:
        _close_figures(figures)


def clear_benchmark_media_video_buffer() -> None:
    """Drop all buffered media frames (every run) without logging them."""
    _BUFFERED_MEDIA_FRAMES.clear()


def log_benchmark_media_videos() -> dict[str, Any]:
    """Log the active run's buffered media histories as W&B videos and drop them.

    Only the current run's frames are flushed, so in a sweep each run animates
    its own history; other runs' buffers (if any share the process) are left
    untouched for those runs to flush.
    """
    wandb = _require_wandb()
    run_buffer = _BUFFERED_MEDIA_FRAMES.pop(_active_run_id(wandb), {})

    video_payload: dict[str, Any] = {}
    for plot_key, frames in run_buffer.items():
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

    try:
        wandb.log(video_payload)
    except Exception:
        logger.exception("wandb.log failed for %d media videos; continuing.", len(video_payload))
        return {}
    logger.info("Logged %d benchmark media videos to W&B.", len(video_payload))
    return video_payload


# ---------------------------------------------------------------------------
# Public API: W&B initialisation with metric presets
# ---------------------------------------------------------------------------


def init_wandb_with_presets(
    project: str,
    run_name: str,
    config: dict | None = None,
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
