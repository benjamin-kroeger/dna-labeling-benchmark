# DNA Segmentation Benchmark

Diagnostic evaluation toolkit for nucleotide-level DNA segmentation models (gene finders like Augustus, Helixer, Tiberius, SegmentNT) against reference annotations (e.g., GENCODE).

Goes beyond standard precision/recall with an **8-type INDEL error taxonomy**, **boundary bias analysis**, **IoU distributions**, and **state transition diagnostics** -- metrics not available in gffcompare, Mikado, or EGASP.

```
pip install dna-segmentation-benchmark
```

## Quick Start

### From GFF/GTF files

```python
from dna_segmentation_benchmark import (
    LabelConfig, EvalMetrics, benchmark_from_gff, compare_multiple_predictions,
)

label_config = LabelConfig(
    labels={0: "CDS", 8: "NONCODING"},
    background_label=8,
    coding_label=0,
)

results = benchmark_from_gff(
    gt_path="ground_truth.gtf",
    pred_paths={"augustus": "predictions.gff"},
    label_config=label_config,
    classes=[0],
    metrics=[EvalMetrics.REGION_DISCOVERY, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
    exclude_features=["gene"],
)

figures = compare_multiple_predictions(
    per_method_benchmark_res=results,
    label_config=label_config,
    classes=[0],
    metrics_to_eval=[EvalMetrics.REGION_DISCOVERY, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
)
```

### From label arrays

```python
from dna_segmentation_benchmark import (
    LabelConfig, EvalMetrics, benchmark_gt_vs_pred_multiple, compare_multiple_predictions,
)

label_config = LabelConfig(
    labels={0: "EXON", 2: "INTRON", 8: "NONCODING"},
    background_label=8,
    coding_label=0,
)

results = benchmark_gt_vs_pred_multiple(
    gt_labels=gt_arrays,       # list[np.ndarray]
    pred_labels=pred_arrays,   # list[np.ndarray]
    label_config=label_config,
    classes=[0],
    metrics=[EvalMetrics.INDEL, EvalMetrics.REGION_DISCOVERY, EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
)
```

### CLI

```bash
dna-benchmark run \
    --gt ground_truth.gtf \
    --pred augustus:predictions.gff \
    --config label_config.yaml \
    --classes 0 \
    --exclude-features gene \
    --output results.json
```

## Metrics

### INDEL Error Taxonomy

Classifies mismatch regions into 8 structural error types: 5'/3' extensions, whole insertions, joins, 5'/3' deletions, whole deletions, and splits. Reveals systematic biases in model predictions.

![INDEL error counts](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_indel_counts.png)

![INDEL error lengths](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_indel_lengths.png)

### Region Discovery (4-level Precision/Recall)

Evaluates section matching at increasing strictness: neighborhood overlap, internal containment, full coverage, and perfect boundary match.

![Region discovery - neighborhood](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_region_discovery_neighborhood_hit.png)

![Region discovery - perfect boundary](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_region_discovery_perfect_boundary_hit.png)

### Nucleotide-Level Classification

Per-base TP/TN/FP/FN with precision, recall, and F1.

![Nucleotide classification](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_nucleotide_classification_nucleotide.png)

### Boundary Exactness (IoU)

Per-section IoU scores with average comparison and survival-curve distributions.

![IoU average](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_iou_average.png)

![IoU distribution](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_iou_distribution.png)

## W&B Integration

Log metrics during training and full diagnostic reports after:

```python
from dna_segmentation_benchmark import init_wandb_with_presets, log_benchmark_scalars, log_benchmark_full

run = init_wandb_with_presets("my-project", "run-name", label_config, classes=[0])

# During training -- lightweight scalar logging per epoch
log_benchmark_scalars(val_results, label_config, step=epoch, method_prefix="val")

# After training -- full report with figures
log_benchmark_full({"my_model": final_results}, figures, label_config)
```

Install with: `pip install dna-segmentation-benchmark[wandb]`

## Examples

See the [`examples/`](examples/) folder:

- **[GTF programmatic example](examples/gtf_programmatic_example.ipynb)** -- end-to-end GFF/GTF evaluation
- **[Array benchmark example](examples/array_benchmark_example.ipynb)** -- starting from numpy label arrays
- **[W&B training loop](examples/wandb_training_loop.ipynb)** -- integration with Weights & Biases

## Updating README Plots

The plots in this README are auto-generated. To refresh them:

```bash
python scripts/generate_readme_plots.py
```

This writes PNGs to `docs/images/` which are referenced by the README.
