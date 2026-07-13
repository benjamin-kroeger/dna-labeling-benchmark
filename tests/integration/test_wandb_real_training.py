"""Opt-in integration test: mock gene-calling training loop → a REAL W&B project.

Unlike ``test_wandb_pipeline.py`` (which drives the logger through the
``FakeWandb`` stub), this test writes to a live Weights & Biases project so you
can open the run in the dashboard and confirm scalars, plots and videos land.

It needs network access and a logged-in wandb (``wandb login``), so it is
**skipped unless ``RUN_WANDB_INTEGRATION`` is set**. Target and mode are picked
up from wandb's own env vars — ``WANDB_PROJECT`` / ``WANDB_ENTITY``, and
``WANDB_MODE=offline`` for a dry run that needs no login or network::

    RUN_WANDB_INTEGRATION=1 WANDB_PROJECT=my-proj pytest \
        tests/integration/test_wandb_real_training.py -s
"""

from __future__ import annotations

import math
import os
import numpy as np
import pytest

wandb = pytest.importorskip("wandb")

from dna_segmentation_benchmark import (
    BEND_LABEL_CONFIG,
    benchmark_gt_vs_pred_multiple,
    clear_benchmark_media_video_buffer,
    init_wandb_with_presets,
    log_benchmark_media,
    log_benchmark_media_videos,
    log_benchmark_scalars,
)

from support.constants import MEDIA_METRICS

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_WANDB_INTEGRATION"),
    reason="set RUN_WANDB_INTEGRATION=1 to run the live W&B integration test",
)

EPOCHS = 3
NUM_TRANSCRIPTS = 8


def _make_ground_truth(rng: np.random.Generator) -> list[np.ndarray]:
    """A handful of simple multi-exon transcripts as clean label arrays."""
    cfg = BEND_LABEL_CONFIG
    gt: list[np.ndarray] = []
    for _ in range(NUM_TRANSCRIPTS):
        length = int(rng.integers(180, 260))
        arr = np.full(length, cfg.background_label, dtype=np.int32)
        cursor = int(rng.integers(10, 25))
        while cursor + 40 < length:
            exon_len = int(rng.integers(15, 30))
            arr[cursor : cursor + exon_len] = cfg.exon_label
            cursor += exon_len + int(rng.integers(12, 30))
        gt.append(arr)
    return gt


def _predict(gt: list[np.ndarray], *, shift: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Copy each GT array and jitter every exon boundary by up to ``shift`` — a
    smaller ``shift`` (later epoch) means predictions that hug the truth more."""
    cfg = BEND_LABEL_CONFIG
    preds: list[np.ndarray] = []
    for gt_arr in gt:
        pred = np.full_like(gt_arr, cfg.background_label)
        exon = np.flatnonzero(gt_arr == cfg.exon_label)
        for seg in np.split(exon, np.where(np.diff(exon) > 1)[0] + 1) if exon.size else []:
            start = max(0, seg[0] + int(rng.integers(-shift, shift + 1)))
            end = min(len(pred), seg[-1] + 1 + int(rng.integers(-shift, shift + 1)))
            if end > start:
                pred[start:end] = cfg.exon_label
        preds.append(pred)
    return preds


def _purge_previous_runs(project: str, entity: str | None) -> None:
    """Delete every existing run in the target project so only this run remains.

    No-op in offline/disabled mode (no server to reach) and on the very first
    run when the project does not exist yet.
    """
    if os.getenv("WANDB_MODE", "online") in {"offline", "disabled", "dryrun"}:
        return
    import wandb

    api = wandb.Api()
    entity = entity or api.default_entity
    try:
        runs = list(api.runs(f"{entity}/{project}"))
    except ValueError:
        return  # project does not exist yet — nothing to purge
    for run in runs:
        run.delete()


def _train_scalars(step: int, total: int) -> dict[str, float]:
    progress = step / max(1, total - 1)
    return {
        "train/loss": 1.8 * math.exp(-2.8 * progress) + 0.1,
        "train/token_accuracy": min(0.98, 0.6 + 0.38 * progress),
    }


def test_mock_training_writes_to_real_wandb():
    import wandb

    rng = np.random.default_rng(7)
    gt = _make_ground_truth(rng)

    project = os.getenv("WANDB_PROJECT", "dna-benchmark-integration-test")
    entity = os.getenv("WANDB_ENTITY")
    _purge_previous_runs(project, entity)  # keep only this run visible in the project

    run = init_wandb_with_presets(
        project=project,
        run_name="mock-gene-calling-training",
        config={
            "epochs": EPOCHS,
            "num_transcripts": NUM_TRANSCRIPTS,
            "metrics": [m.name for m in MEDIA_METRICS],
        },
        entity=entity,
    )

    clear_benchmark_media_video_buffer()
    logged: dict[str, float] = {}
    try:
        for epoch in range(EPOCHS):
            wandb.log(_train_scalars(epoch, EPOCHS), step=epoch)
            preds = _predict(gt, shift=max(1, 4 - epoch), rng=rng)
            results = benchmark_gt_vs_pred_multiple(
                gt_labels=gt,
                pred_labels=preds,
                label_config=BEND_LABEL_CONFIG,
                metrics=MEDIA_METRICS,
                infer_introns=True,
            )
            logged = log_benchmark_scalars(
                results, label_config=BEND_LABEL_CONFIG, step=epoch, method_prefix="val"
            )
            log_benchmark_media(
                results, label_config=BEND_LABEL_CONFIG, step=epoch, method_prefix="val"
            )
        videos = log_benchmark_media_videos()
    finally:
        run.finish()

    assert run.id  # a real run object was created
    assert logged and all(k.startswith("val/") for k in logged)  # validation scalars logged
    assert videos  # buffered plots flushed to at least one video
    print(f"\nW&B run: {getattr(run, 'url', None)}")