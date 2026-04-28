# Array Benchmark

Use {py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_multiple` when
you already have paired ground-truth and prediction arrays in memory.

## Minimal Example

```python
import numpy as np

from dna_segmentation_benchmark import (
    EvalMetrics,
    LabelConfig,
    benchmark_gt_vs_pred_multiple,
)

label_config = LabelConfig(
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

`LabelConfig` defines what the integer tokens in your arrays mean.

Required fields:

- `background_label`
- `exon_label`

Optional fields:

- `intron_label`
- `splice_donor_label`
- `splice_acceptor_label`

Minimal exon-only setup:

```python
from dna_segmentation_benchmark import LabelConfig

label_config = LabelConfig(
    background_label=8,
    exon_label=0,
)
```

Setup with explicit introns and splice-site labels:

```python
label_config = LabelConfig(
    background_label=8,
    exon_label=0,
    intron_label=2,
    splice_donor_label=1,
    splice_acceptor_label=3,
)
```

Use an exon-only config when your arrays only distinguish coding from
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

## Choosing Metric Families

Common combinations:

- fast training/validation: `REGION_DISCOVERY`, `BOUNDARY_EXACTNESS`,
  `STRUCTURAL_COHERENCE`
- full structural analysis: add `INDEL`, `NUCLEOTIDE_CLASSIFICATION`,
  `FRAMESHIFT`

`FRAMESHIFT` should only be used on transcript-level inputs where the full CDS
is present. GT coding positions must form complete codons.

## Next Steps

- {doc}`wandb_logging` for direct W&B logging from one aggregated result
- {doc}`method_comparison` for plotting several methods side by side
