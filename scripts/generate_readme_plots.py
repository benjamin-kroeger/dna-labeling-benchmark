#!/usr/bin/env python
"""Generate the plots used in README.md.

Run this before publishing to ensure the README images are up to date:

    python scripts/generate_readme_plots.py

Outputs PNGs to ``docs/images/`` which are referenced by the README via
raw GitHub URLs (so they render on both GitHub and PyPI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dna_segmentation_benchmark import (
    EvalMetrics,
    LabelConfig,
    benchmark_gt_vs_pred_multiple,
    compare_multiple_predictions,
)
from dna_segmentation_benchmark.label_definition import _FULL_SWEEP_METRICS

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"

LABEL_CONFIG = LabelConfig(
    labels={0: "EXON", 2: "INTRON", 8: "NONCODING"},
    background_label=8,
    coding_label=0,
    intron_label=2,
)

CLASSES = [0]
# FRAMESHIFT requires exon lengths divisible by 3, so we generate plots
# for it separately with constrained sequences.
METRICS = [m for m in _FULL_SWEEP_METRICS if m != EvalMetrics.FRAMESHIFT]


def _make_gene_sequence(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic gene sequence with exons and introns."""
    arr = np.full(length, 8, dtype=np.int32)
    pos = rng.integers(50, 200)
    while pos < length - 100:
        exon_len = rng.integers(50, 300)
        arr[pos : pos + exon_len] = 0
        pos += exon_len
        intron_len = rng.integers(200, 2000)
        arr[pos : pos + intron_len] = 2
        pos += intron_len
    return arr


def _perturb(gt: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    """Create a noisy prediction from a GT array with realistic errors."""
    pred = gt.copy()

    # Random per-base noise
    flip = rng.random(len(gt)) < noise
    pred[flip] = rng.choice([0, 2, 8], size=flip.sum())

    # Shift some boundaries to create realistic INDEL errors
    for _ in range(rng.integers(1, 6)):
        pos = rng.integers(0, len(gt))
        shift = rng.integers(-30, 30)
        s, e = max(0, pos), min(len(gt), pos + abs(shift))
        src_s = max(0, s + shift)
        src_e = src_s + (e - s)
        if src_e <= len(gt):
            pred[s:e] = gt[src_s:src_e]

    # Occasionally delete an entire exon (for structural coherence metrics)
    if noise > 0.05 and rng.random() < 0.3:
        exon_starts = np.where(np.diff(np.concatenate(([0], (gt == 0).astype(int)))) == 1)[0]
        if len(exon_starts) > 2:
            skip_idx = rng.integers(1, len(exon_starts) - 1)
            start = exon_starts[skip_idx]
            exon_end = start
            while exon_end < len(gt) and gt[exon_end] == 0:
                exon_end += 1
            pred[start:exon_end] = 2  # Replace exon with intron

    return pred


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2024)
    n_sequences = 300

    gt_labels = [
        _make_gene_sequence(rng.integers(5000, 50000), rng)
        for _ in range(n_sequences)
    ]

    methods = {
        "Model A (good)": 0.03,
        "Model B (medium)": 0.07,
        "Model C (weak)": 0.12,
    }

    all_results = {}
    for name, noise in methods.items():
        preds = [_perturb(gt, rng, noise) for gt in gt_labels]
        all_results[name] = benchmark_gt_vs_pred_multiple(
            gt_labels=gt_labels,
            pred_labels=preds,
            label_config=LABEL_CONFIG,
            classes=CLASSES,
            metrics=METRICS,
        )

    figures = compare_multiple_predictions(
        per_method_benchmark_res=all_results,
        label_config=LABEL_CONFIG,
        classes=CLASSES,
        metrics_to_eval=METRICS,
        output_dir=OUTPUT_DIR,
    )

    print(f"Generated {len(figures)} figures in {OUTPUT_DIR}/")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
