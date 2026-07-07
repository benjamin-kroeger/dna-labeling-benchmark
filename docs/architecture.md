# Architecture

A bird's-eye view of how the benchmark turns a ground-truth/prediction pair into
metrics. Two entry paths converge on a single evaluator, and an explicit
configuration selects which of the eight metric groups run.

```{figure} images/architecture.drawio.svg
:alt: DNA Segmentation Benchmark architecture
:width: 100%

High-level architecture: two entry paths (in-memory arrays or GFF/GTF files)
converge on one evaluator; the configuration selects which of the eight metric
groups run.
```

## How to read it

- **Configuration** (`LabelConfig`, `AnnotationMode`, `BenchmarkScope`,
  `EvalMetrics`) sets the label vocabulary and scope and selects which metric
  groups run. It is referenced by every layer below. See
  {doc}`getting_started/annotation_modes` and {doc}`metrics/conventions`.
- **Two entry paths** converge on the same evaluator:
  - **Path A** — in-memory integer label arrays via
    {py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_multiple`.
  - **Path B** — GFF/GTF files via
    {py:func}`dna_segmentation_benchmark.benchmark_from_gff`, which parses the
    annotations, pairs transcripts, and builds the paired arrays before handing
    off to the evaluator. See {doc}`getting_started/index`.
- **Evaluator** computes the requested **metric groups**. Each group answers one
  question about prediction quality; see {doc}`metrics/overview` for the full
  list and {doc}`metrics/index` for per-metric detail.
- **Outputs** — figures (`plotting/`), corpus-level statistics
  (`compute_global_metrics`, GFF path only), and optional Weights & Biases
  logging (see {doc}`getting_started/wandb_logging`).

