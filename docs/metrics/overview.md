# Metrics Overview

All metric families operate on paired one-dimensional integer label arrays.
The public entry points are:

- {py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_single`
- {py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_multiple`
- {py:func}`dna_segmentation_benchmark.benchmark_from_gff`

## Available Metric Families

| Family | Enum | Main question | Key outputs |
|---|---|---|---|
| Region Discovery | `REGION_DISCOVERY` | Did the prediction find the right sections? | Precision / recall at four overlap strictness levels (neighborhood, internal, full-coverage, perfect-boundary) |
| Boundary Exactness | `BOUNDARY_EXACTNESS` | How accurate are the matched boundaries? | IoU distribution, boundary-residual bias/reliability landscape, terminal-boundary flags |
| Nucleotide Classification | `NUCLEOTIDE_CLASSIFICATION` | How well does coding vs non-coding separate per base? | Precision / recall / F1 from the nucleotide confusion matrix |
| Structural Coherence | `STRUCTURAL_COHERENCE` | Is the transcript chain correct as a whole? | Intron/exon chain P/R (strict, subset, superset), transcript match classes, boundary shift distribution, segment count delta, soft exon recall and hallucinated-exon count |
| Diagnostic Depth | `DIAGNOSTIC_DEPTH` | Where and how severely does the prediction fail structurally? | Segment-length EMD, 100-bin position bias histogram over the coding span |
| Transition Analysis | always available | Where do label changes fail or appear spuriously? | GT transition confusion matrices, false-transition counts (premature, late, spurious) |
| INDEL | `INDEL` | What structural mismatch types occur? | Categorised mismatch groups (5′/3′ extensions, whole insertions/deletions, splits, joins) |
| Frameshift | `FRAMESHIFT` | Is coding frame preserved where GT and prediction overlap? | Per-position reading-frame deviation |

## Recommended Online Subset

For repeated validation during training, the current W&B integration keeps the
online scalar set deliberately small:

- Region Discovery precision/recall
- Boundary Exactness IoU mean
- Structural Coherence intron-chain precision/recall
- Structural Coherence transcript-exact precision/recall