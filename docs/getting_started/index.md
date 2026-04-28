# Getting Started

The benchmark has two main entry points:

- array-based benchmarking via {py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_multiple`
- annotation-file benchmarking via {py:func}`dna_segmentation_benchmark.benchmark_from_gff`

Use the array API when your training or inference pipeline already produces
integer label arrays. Use the GFF/GTF pipeline when you want the benchmark to
handle parsing, transcript pairing, array construction, and aggregation.

```{toctree}
:titlesonly:

array_benchmark
wandb_logging
method_comparison
gff_benchmark
```
