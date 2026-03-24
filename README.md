# DNA Segmentation Benchmark

Diagnostic evaluation toolkit for nucleotide-level DNA segmentation models (gene finders like Augustus, Helixer, Tiberius, SegmentNT) against reference annotations (e.g., GENCODE).

Goes beyond standard precision/recall with an **8-type INDEL error taxonomy**, **boundary bias/reliability landscapes**, **gap chain analysis** (label-agnostic intron chain comparison), **transcript match classification**, **junction error diagnosis**, and **state transition analysis** -- metrics not available in gffcompare, Mikado, or EGASP.

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
    labels={0: "CDS", 2: "INTRON", 8: "NONCODING"},
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
import numpy as np
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
    metrics=[
        EvalMetrics.INDEL,
        EvalMetrics.REGION_DISCOVERY,
        EvalMetrics.BOUNDARY_EXACTNESS,
        EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
        EvalMetrics.STRUCTURAL_COHERENCE,
        EvalMetrics.DIAGNOSTIC_DEPTH,
    ],
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

---

## Metrics

Seven metric groups, each answering a distinct question about prediction quality:

| Group | Question |
|-------|----------|
| `NUCLEOTIDE_CLASSIFICATION` | Per-base, how accurate is it? |
| `REGION_DISCOVERY` | Did we find the right regions? |
| `BOUNDARY_EXACTNESS` | How precise are the boundaries? |
| `INDEL` | What structural errors exist? |
| `FRAMESHIFT` | Is the reading frame preserved? |
| `STRUCTURAL_COHERENCE` | Is the overall segment arrangement correct? |
| `DIAGNOSTIC_DEPTH` | Why is the prediction structurally wrong? |

---

### Nucleotide Classification

Per-base TP/TN/FP/FN with precision, recall, and F1. The most basic metric -- treats each position independently.

![Nucleotide classification](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_nucleotide_classification_nucleotide.png)

---

### Region Discovery (4-level Precision/Recall)

Evaluates section matching at increasing strictness using 1:1 greedy matching by overlap length:

| Level | TP condition | What it forgives |
|-------|-------------|-----------------|
| `neighborhood_hit` | Any overlap | Over- and under-prediction |
| `internal_hit` | Prediction inside GT | Over-prediction |
| `full_coverage_hit` | Prediction covers GT | Under-prediction |
| `perfect_boundary_hit` | Exact match (sweep-based) | Nothing |

![Region discovery - neighborhood](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_region_discovery_neighborhood_hit.png)

![Region discovery - perfect boundary](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_region_discovery_perfect_boundary_hit.png)

---

### Boundary Exactness

How precise are predicted boundaries? Includes IoU distributions and two diagnostic matrices:

- **Bias matrix** (21x21): Signed boundary residuals revealing systematic directional errors (e.g., "predictions consistently start 2bp early")
- **Reliability matrix** (11x11): Cumulative recall at tolerances 0--10 bp, showing how quickly recall degrades as boundary tolerance tightens

![IoU average](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_iou_average.png)

![IoU distribution](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_iou_distribution.png)

---

### INDEL Error Taxonomy

Classifies every contiguous mismatch region into one of 8 structural error types:

| Insertions (pred has class, GT does not) | Deletions (GT has class, pred does not) |
|---|---|
| 5' extension | 5' deletion |
| 3' extension | 3' deletion |
| Joined (merges two GT sections) | Split (splits one GT section) |
| Whole insertion (new section) | Whole deletion (missing section) |

![INDEL error counts](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_indel_counts.png)

![INDEL error lengths](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_indel_lengths.png)

---

### Structural Coherence

Evaluates the predicted segment chain **as a whole** -- not per-section, but as a complete ordered arrangement.

#### Gap Chain Comparison

Compares ordered gaps between consecutive segments. For exons, gaps = introns -- making this the **label-agnostic equivalent of intron chain comparison** (gffcompare's key metric).

- `gap_chain_match_rate`: Fraction of sequences with identical gap chains
- `gap_count_match_rate`: Fraction with the same number of gaps
- `gap_chain_lcs_ratio`: LCS-based partial credit (0--1), ordering-aware

![Gap chain metrics](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_gap_chain.png)

#### Transcript Match Classification

Holistic structural classification of each (GT, prediction) pair into one of 6 categories:

| Class | Condition |
|-------|-----------|
| `exact` | Identical segment chains |
| `boundary_shift` | Same segment count, shifted boundaries |
| `missing_segments` | Prediction is ordered subset of GT (segments skipped) |
| `extra_segments` | GT is ordered subset of prediction (segments inserted) |
| `structurally_different` | None of the above |
| `missed` | No prediction for this class |

![Transcript match classification](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_transcript_match.png)

#### Segment Count Delta

Over-segmentation (positive) vs under-segmentation (negative).

![Segment count delta](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_segment_count_delta.png)

---

### Diagnostic Depth

Causal diagnosis of structural errors -- answering *why* the prediction is wrong, not just *that* it is wrong.

#### Junction Error Taxonomy

| Error type | Description |
|-----------|-------------|
| Exon skip | GT segments merged (intervening segment absent) |
| Segment retention | GT segment absorbed by neighbours |
| Novel insertion | Extra segment splits a GT segment |
| Cascade shift | Boundary error propagates across 3+ segments |
| Compensating errors | Paired errors that cancel out |

![Junction error taxonomy](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_junction_errors.png)

#### Position Bias

Match rate stratified by genomic position (5' / interior / 3'), revealing whether errors concentrate at sequence ends.

![Position bias](https://raw.githubusercontent.com/PredictProtein/benchmark/main/docs/images/EXON_position_bias.png)

---

### Frameshift

Reading frame deviation (mod-3) between GT and predicted coding exons. Only valid on single-transcript sequences with `coding_label` configured.

---

### State Transitions (always computed)

Two analyses run on every benchmark call:

- **GT Transition Confusion Matrices**: At every position where GT changes label, what did the predictor do? One heatmap per source label.
- **False Transition Analysis**: At positions where GT is stable (no label change), did the predictor introduce a spurious transition?

---

## Label Configuration

All metrics are label-agnostic. Define your own token mapping:

```python
from dna_segmentation_benchmark import LabelConfig

config = LabelConfig(
    labels={0: "EXON", 1: "DONOR", 2: "INTRON", 3: "ACCEPTOR", 8: "NONCODING"},
    background_label=8,
    coding_label=0,           # Required for FRAMESHIFT
    splice_donor_label=1,     # Reserved for future splice metrics
    splice_acceptor_label=3,  # Reserved for future splice metrics
    intron_label=2,
)
```

A pre-built config for the BEND benchmark is available as `BEND_LABEL_CONFIG`.

---

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

---

## Examples

See the [`examples/`](examples/) folder:

- **[GTF programmatic example](examples/gtf_programmatic_example.ipynb)** -- end-to-end GFF/GTF evaluation
- **[Array benchmark example](examples/array_benchmark_example.ipynb)** -- starting from numpy label arrays
- **[W&B training loop](examples/wandb_training_loop.ipynb)** -- integration with Weights & Biases

---

## Documentation

- **[Metrics Reference](docs/metrics.md)** -- complete documentation of every metric with formulas and aggregation details
- **[Design Rationale](docs/design_rationale.md)** -- architectural decisions and comparison with gffcompare, Mikado, EGASP

---

## Updating README Plots

The plots in this README are auto-generated from synthetic data. To refresh them:

```bash
python scripts/generate_readme_plots.py
```

This writes PNGs to `docs/images/` which are referenced by the README.
