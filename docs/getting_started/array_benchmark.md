# Array Benchmark

Use {py:func}`gene_calling_benchmark.benchmark_from_arrays` when
you already have paired ground-truth and prediction arrays in memory.

## Minimal Example

```python
import numpy as np

from gene_calling_benchmark import (
    AnnotationMode,
    EvalMetrics,
    LabelConfig,
    benchmark_from_arrays,
)

label_config = LabelConfig(
    annotation_mode=AnnotationMode.EXON_INTRON,
    background_label=8,
    exon_label=0,
    intron_label=2,
    splice_donor_label=1,
    splice_acceptor_label=3,
)

gt_arrays = [
    np.array([8, 8, 0, 0, 2, 2, 0, 0, 8]),
    np.array([8, 0, 0, 2, 2, 0, 0, 8]),
]
pred_arrays = [
    np.array([8, 8, 0, 0, 2, 2, 0, 0, 8]),
    np.array([8, 0, 0, 0, 2, 0, 0, 8]),
]

results = benchmark_from_arrays(
    gt_labels=gt_arrays,
    pred_labels=pred_arrays,
    label_config=label_config,
    metrics=[
        EvalMetrics.REGION_DISCOVERY,
        EvalMetrics.BOUNDARY_EXACTNESS,
        EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
        EvalMetrics.STRUCTURAL_COHERENCE,
    ],
    infer_introns=True,
)
```

## Label Config

`LabelConfig` defines what the integer tokens in your arrays mean. It always
declares an explicit `annotation_mode` — see {doc}`annotation_modes` for the
full discussion of modes and scopes.

Minimal `EXON_INTRON` setup:

```python
from gene_calling_benchmark import AnnotationMode, LabelConfig

label_config = LabelConfig(
    annotation_mode=AnnotationMode.EXON_INTRON,
    background_label=8,
    exon_label=0,
)
```

`EXON_INTRON` with explicit introns and splice-site labels:

```python
label_config = LabelConfig(
    annotation_mode=AnnotationMode.EXON_INTRON,
    background_label=8,
    exon_label=0,
    intron_label=2,
    splice_donor_label=1,
    splice_acceptor_label=3,
)
```

`UTR_CDS_INTRON` setup when your arrays carry distinct UTR and CDS tokens:

```python
from gene_calling_benchmark import AnnotationMode, BenchmarkScope, LabelConfig

label_config = LabelConfig(
    annotation_mode=AnnotationMode.UTR_CDS_INTRON,
    background_label=8,
    cds_label=0,
    five_prime_utr_label=4,
    three_prime_utr_label=5,
    intron_label=2,
    # evaluation_scope=BenchmarkScope.CDS  # to score CDS only per transcript
)
```

In `UTR_CDS_INTRON` the per-transcript metrics use `evaluation_scope`
(`transcript_exon` by default, where `5' UTR + CDS + 3' UTR` count as exonic;
`cds` to score the coding span only).

Use an `EXON_INTRON` config when your arrays only distinguish exonic from
background. Add `intron_label` when you want strict intron-chain evaluation on
arrays that already carry explicit intron tokens.

`LabelConfig` is only about array semantics. Parser choices such as `"exon"` vs
`"CDS"` belong to the GFF/GTF pipeline arguments, not to the label config.

## Optional: Position Masking

You can optionally exclude certain positions from evaluation using a boolean mask:

```python
mask_labels = [
    np.array([False, False, True, True, False, ...]),  # exclude positions 2–3
    np.array([False, False, False, ...]),
]

results = benchmark_from_arrays(
    gt_labels=gt_arrays,
    pred_labels=pred_arrays,
    label_config=label_config,
    mask_labels=mask_labels,  # True = exclude
    metrics=[...],
)
```

Masked positions are not counted in any metric. Use this when you want to
exclude low-confidence regions (e.g. N-stretches in the reference) from
scoring.

## Result Structure

The aggregated result is a flat dictionary keyed by metric family, plus a
`metadata` block. Only the families you request appear. There is no top-level
`EXON` wrapper anymore.

For the metric list in the example above:

```python
{
    "metadata": {...},                # always present (mode, scope, ...)
    "REGION_DISCOVERY": {...},
    "BOUNDARY_EXACTNESS": {...},
    "NUCLEOTIDE_CLASSIFICATION": {...},
    "STRUCTURAL_COHERENCE": {...},
}
```

Requesting `EvalMetrics.STATE_TRANSITIONS` adds `transition_failures` and
`false_transitions` keys.

## When To Use `infer_introns`

{py:func}`gene_calling_benchmark.benchmark_from_arrays` applies
`infer_introns` to the raw GT and prediction arrays before any metric is
computed. That keeps all metric families consistent on the same transformed
input.

Use it when:

- your arrays contain exon or CDS labels but no explicit intron labels
- you still want to evaluate {py:attr}`~gene_calling_benchmark.EvalMetrics.STRUCTURAL_COHERENCE`
  with strict intron-chain scoring

Be careful on very large arrays. In that case the benchmark switches to a
conservative gap-length cutoff and emits a warning, because a chromosome-scale
coding gap can be an intergenic distance rather than a true intron.

### How the cutoff is chosen

For arrays shorter than ~1 Mbp, every background gap between coding segments is
relabelled as intron — the assumption is that the array represents a single
transcript locus.

For arrays at or above that length the benchmark is more conservative: it looks
for a bimodal split in the gap-length distribution (a cluster of intron-scale
gaps versus a cluster of intergenic-scale gaps) and only relabels gaps below the
resulting cutoff, so a chromosome-scale coding gap is not mistaken for an intron.
If your arrays already carry explicit intron labels, leave `infer_introns` off.

## Large Inputs

`benchmark_from_arrays` needs every GT/prediction array in memory at once. When
the validation set is too large to hold all arrays simultaneously, use
{py:class}`gene_calling_benchmark.StreamingBenchmark` instead — feed pairs one
at a time and build each array lazily:

```python
from gene_calling_benchmark import StreamingBenchmark

bench = StreamingBenchmark(label_config, metrics=[...], infer_introns=True)
for gt, pred in build_pairs_lazily():   # one pair at a time; nothing else retained
    bench.add(gt, pred)
results = bench.result()                # identical numbers to benchmark_from_arrays
```

Reach for it only when memory is the constraint; for an in-memory set,
`benchmark_from_arrays` is the one-liner equivalent.

## Choosing Metric Families

Omit `metrics` to get the default four (`REGION_DISCOVERY`,
`BOUNDARY_EXACTNESS`, `NUCLEOTIDE_CLASSIFICATION`, `STATE_TRANSITIONS`). Pass an
explicit list to narrow or extend that set.

Common combinations:

- fast training/validation: `REGION_DISCOVERY`, `BOUNDARY_EXACTNESS`,
  `STRUCTURAL_COHERENCE`
- full structural analysis: add `INDEL`, `NUCLEOTIDE_CLASSIFICATION`,
  `PHASE_DRIFT`

`PHASE_DRIFT` is only valid in `UTR_CDS_INTRON` mode with
`evaluation_scope=BenchmarkScope.CDS`; requesting it in any other configuration
raises an error. Even then it should only be used on transcript-level inputs
where the full CDS is present, because GT coding positions must form complete
codons.

## Next Steps

- {doc}`wandb_logging` for direct W&B logging from one aggregated result
- {doc}`method_comparison` for plotting several methods side by side
