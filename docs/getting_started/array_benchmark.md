# Array Benchmark

Use {py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_multiple` when
you already have paired ground-truth and prediction arrays in memory.

## Minimal Example

```python
import numpy as np

from dna_segmentation_benchmark import (
    AnnotationMode,
    EvalMetrics,
    LabelConfig,
    benchmark_gt_vs_pred_multiple,
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

results = benchmark_gt_vs_pred_multiple(
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
from dna_segmentation_benchmark import AnnotationMode, LabelConfig

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
from dna_segmentation_benchmark import AnnotationMode, BenchmarkScope, LabelConfig

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

## Result Structure

The aggregated result is a flat dictionary keyed by metric family. There is no
top-level `EXON` wrapper anymore.

```python
{
    "REGION_DISCOVERY": {...},
    "BOUNDARY_EXACTNESS": {...},
    "NUCLEOTIDE_CLASSIFICATION": {...},
    "STRUCTURAL_COHERENCE": {...},
    "transition_failures": {...},
    "false_transitions": {...},
}
```

## When To Use `infer_introns`

{py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_multiple` applies
`infer_introns` to the raw GT and prediction arrays before any metric is
computed. That keeps all metric families consistent on the same transformed
input.

Use it when:

- your arrays contain exon or CDS labels but no explicit intron labels
- you still want to evaluate {py:attr}`~dna_segmentation_benchmark.EvalMetrics.STRUCTURAL_COHERENCE`
  with strict intron-chain scoring

Be careful on very large arrays. In that case the benchmark switches to a
conservative gap-length cutoff and emits a warning, because a chromosome-scale
coding gap can be an intergenic distance rather than a true intron.

### How the cutoff is chosen

For arrays shorter than `_INFER_INTRONS_LARGE_ARRAY_WARNING_LENGTH`
(`1_000_000` bp), every background gap between coding segments is
relabelled as intron — the assumption is that the array represents a
single transcript locus.

For arrays at or above that length the benchmark is more conservative
and chooses a gap-length cutoff before relabelling. The cutoff is
selected as follows:

1. Sorted distinct gap lengths are scanned for the **largest
   consecutive jump ratio** (each gap divided by the previous one).
2. If the largest ratio exceeds
   `_INFER_INTRONS_BIMODAL_MIN_JUMP_RATIO` (`5.0`), the gaps look
   bimodal — one cluster of intron-scale gaps and one cluster of
   intergenic-scale gaps. The cutoff is set to the geometric midpoint
   between the two cluster boundaries.
3. Otherwise the distribution is treated as unimodal and the cutoff
   defaults to `_INFER_INTRONS_LARGE_GAP_RATIO` × the median of the
   lower half of gap lengths (`20×` by default).

These constants live at the top of `eval/evaluate_predictors.py` and
are not currently part of the public API. If you need different
behaviour, edit them in source or filter intron labels in advance.

## Choosing Metric Families

Common combinations:

- fast training/validation: `REGION_DISCOVERY`, `BOUNDARY_EXACTNESS`,
  `STRUCTURAL_COHERENCE`
- full structural analysis: add `INDEL`, `NUCLEOTIDE_CLASSIFICATION`,
  `FRAMESHIFT`

`FRAMESHIFT` is only valid in `UTR_CDS_INTRON` mode with
`evaluation_scope=BenchmarkScope.CDS`; requesting it in any other configuration
raises an error. Even then it should only be used on transcript-level inputs
where the full CDS is present, because GT coding positions must form complete
codons.

## Next Steps

- {doc}`wandb_logging` for direct W&B logging from one aggregated result
- {doc}`method_comparison` for plotting several methods side by side
