# Online W&B Logging

The W&B integration is built around a simple contract: run the benchmark
yourself, then hand the aggregated result dict to the logging helpers.

Relevant public functions:

- {py:func}`dna_segmentation_benchmark.init_wandb_with_presets`
- {py:func}`dna_segmentation_benchmark.log_benchmark_scalars`
- {py:func}`dna_segmentation_benchmark.log_benchmark_media`
- {py:func}`dna_segmentation_benchmark.log_benchmark_media_videos`
- {py:func}`dna_segmentation_benchmark.clear_benchmark_media_video_buffer`

## Minimal Loop

```python
from dna_segmentation_benchmark import (
    BEND_LABEL_CONFIG,
    EvalMetrics,
    benchmark_gt_vs_pred_multiple,
    init_wandb_with_presets,
    log_benchmark_scalars,
    log_benchmark_media,
    log_benchmark_media_videos,
)

run = init_wandb_with_presets(
    project="dna-benchmark",
    run_name="validation",
)

results = benchmark_gt_vs_pred_multiple(
    gt_labels=gt_arrays,
    pred_labels=pred_arrays,
    label_config=BEND_LABEL_CONFIG,
    metrics=[
        EvalMetrics.REGION_DISCOVERY,
        EvalMetrics.BOUNDARY_EXACTNESS,
        EvalMetrics.STRUCTURAL_COHERENCE,
    ],
    infer_introns=True,
)

log_benchmark_scalars(
    results,
    BEND_LABEL_CONFIG,
    step=12,
    method_prefix="val",
)
log_benchmark_media(
    results,
    BEND_LABEL_CONFIG,
    step=12,
    method_prefix="val"
)

# Optional, usually after several calls to log_benchmark_media(...)
log_benchmark_media_videos()

run.finish()
```

```{note}
`benchmark_gt_vs_pred_multiple(...)` tops out at roughly 1000 transcript pairs
per second, so on a large validation set it can take a while. Run it on a
separate thread (or process) off the critical GPU training path so the
benchmark doesn't stall training — hand it the arrays and let it log
asynchronously.
```

## What Gets Logged

`log_benchmark_scalars(...)` logs a curated, high-signal online subset. Keys
follow `[<method_prefix>/]<group>/<leaf>`, and a group only appears if its
metric was in the result:

- `boundary_exactness/iou_mean`
- `region_discovery/{neighborhood_hit,internal_hit,full_coverage_hit,perfect_boundary_hit}/{precision,recall}`
- `nucleotide_classification/nucleotide/{precision,recall,f1}`
- `struct_coherence/intron_chain/match_rate`,
  `struct_coherence/exon_chain/match_rate` (all transcripts),
  `struct_coherence/exon_chain_multi/match_rate` (multi-exon only),
  `struct_coherence/exon_chain_single/match_rate` (single-exon only)
  (whole-chain tiers are all-or-nothing, so precision = recall = F1 — a single
  rate is logged),
  `struct_coherence/segment_count_delta/{mean,mae}`,
  `struct_coherence/exon_recall_per_transcript/mean`,
  `struct_coherence/exon_precision_per_transcript/mean`,
  `struct_coherence/false_exon_count_per_transcript/mean`,
  `struct_coherence/exact_match_rate`,
  `struct_coherence/splice_site_results/{donor,acceptor}_{precision,recall}`
- `diagnostic_depth/length_emd/{mean,mae}`

To log *every* numeric scalar instead of this subset, use
{py:func}`dna_segmentation_benchmark.log_benchmark_all_scalars`.

`log_benchmark_media(...)` renders and logs:

- `position_bias`
- `transition_matrices`
- `false_transitions`

## Media History vs GIF Videos

Each call to {py:func}`dna_segmentation_benchmark.log_benchmark_media` logs
regular W&B images and also stores the rendered RGB frames in an internal
buffer. {py:func}`dna_segmentation_benchmark.log_benchmark_media_videos`
converts the buffered history into GIF videos and clears the buffer.

This design keeps the user-facing API simple:

- call `log_benchmark_media(...)` whenever you want
- call `log_benchmark_media_videos()` later if you want videos
- call `clear_benchmark_media_video_buffer()` if you want to discard buffered
  frames instead

## Dependency Note

Raw-array video logging through `wandb.Video(...)` requires the W&B media
extras. In `pyproject.toml` that should be declared as:

```toml
[project.optional-dependencies]
wandb = ["wandb[media]>=0.26"]
```

Then install with:

```bash
uv sync --extra wandb
```

## Caveats

- The logging helpers expect one aggregated benchmark result dict, not raw
  per-sequence outputs.
- `method_prefix="val"` is only a naming convention. The helper does not decide
  when validation happens.
- Video generation happens only when
  {py:func}`dna_segmentation_benchmark.log_benchmark_media_videos` is called.
