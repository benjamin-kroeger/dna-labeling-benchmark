# Metrics Overview

All metric families operate on paired one-dimensional integer label arrays.
The public entry points are:

- {py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_single`
- {py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_multiple`
- {py:func}`dna_segmentation_benchmark.benchmark_from_gff`

## Available Metric Families

| Family | Enum | Main question | Key outputs |
|---|---|---|---|
| Region Discovery | `REGION_DISCOVERY` | Did the prediction find the right sections? | Precision / recall at four nested detection tiers (neighborhood ⊇ internal / full-coverage ⊇ perfect-boundary) |
| Boundary Exactness | `BOUNDARY_EXACTNESS` | How accurate are the matched boundaries? | IoU distribution, boundary-residual bias/reliability landscape, terminal-boundary flags |
| Nucleotide Classification | `NUCLEOTIDE_CLASSIFICATION` | How well does coding vs non-coding separate per base? | Precision / recall / F1 from the nucleotide confusion matrix |
| Structural Coherence | `STRUCTURAL_COHERENCE` | Is the transcript chain correct as a whole? | Intron/exon chain P/R (strict, subset, superset), transcript match classes, boundary shift distribution, segment count delta, soft exon recall and hallucinated-exon count, donor/acceptor splice-site confusion and P/R (when splice labels are configured) |
| Diagnostic Depth | `DIAGNOSTIC_DEPTH` | Where and how severely does the prediction fail structurally? | Segment-length EMD, 100-bin position bias histogram over the coding span |
| Transition Analysis | *always on* — no enum | Where do label changes fail or appear spuriously? | GT transition confusion matrices, false-transition counts (premature, late, spurious) |
| INDEL | `INDEL` | What structural mismatch types occur? | Categorised mismatch groups (5′/3′ extensions, whole insertions/deletions, splits, joins) |
| Phase Drift | `PHASE_DRIFT` | Does the prediction stay in step by coding-base count where GT and prediction overlap? | Per-position coding-phase drift (modulo 3) |

### A note on "always available"

Transition Analysis has no entry in
{py:class}`~dna_segmentation_benchmark.EvalMetrics`. Its outputs
(`transition_failures` and `false_transitions`) are computed and
emitted unconditionally on every benchmarking call so that the
plotting layer can always show a confusion-matrix view, even when no
explicit enum was passed. To opt out, drop the corresponding figures
on the consumer side rather than the eval side.

## Cross-cutting conventions

Coordinate semantics, strand handling, aggregation strategy, and
sentinel-value behaviour are shared across every metric family. See
{doc}`conventions` for the full list before interpreting any metric in
detail.

## Recommended Online Subset

For repeated validation during training, the current W&B integration logs
the following online scalar set per metric family:

**Boundary Exactness**
- IoU mean

**Region Discovery**
- `neighborhood_hit` precision / recall
- `internal_hit` precision / recall
- `full_coverage_hit` precision / recall
- `perfect_boundary_hit` precision / recall

**Nucleotide Classification**
- Precision / recall / F1

**Structural Coherence**
- Intron / exon chain match rates
- Segment count delta (mean, MAE)
- Exon recall per transcript (mean)
- Hallucinated exon count per transcript (mean)
- Exact match rate
- Splice-site donor / acceptor precision & recall (when configured)

**Diagnostic Depth**
- Length EMD (mean, MAE)

Full details in {py:func}`dna_segmentation_benchmark.log_benchmark_scalars` and
{py:func}`dna_segmentation_benchmark.log_benchmark_all_scalars`.