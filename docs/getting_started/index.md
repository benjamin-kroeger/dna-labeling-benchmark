# Getting Started

The benchmark has two main entry points:

- array-based benchmarking via {py:func}`gene_calling_benchmark.benchmark_from_arrays`
- annotation-file benchmarking via {py:func}`gene_calling_benchmark.benchmark_from_gff`

Use the array API when your training or inference pipeline already produces
integer label arrays. Use the GFF/GTF pipeline when you want the benchmark to
handle parsing, transcript pairing, array construction, and aggregation.

Both entry points are anchored to an explicit **annotation mode**
(`EXON_INTRON` or `UTR_CDS_INTRON`). Read {doc}`annotation_modes` first — it
explains what the positive labels mean, how evaluation scopes work, and which
metrics each mode unlocks.

Ready-made reference datasets can be downloaded on demand from the registry —
see {doc}`datasets`.

```{toctree}
:titlesonly:

annotation_modes
datasets
array_benchmark
wandb_logging
method_comparison
gff_benchmark
```
