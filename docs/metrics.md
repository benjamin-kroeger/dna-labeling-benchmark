# Metrics Reference

Complete documentation of every metric computed by the DNA Segmentation Benchmark. All metrics operate on paired 1-D integer label arrays `(gt_labels, pred_labels)` and are label-agnostic — they work for any class token defined in the `LabelConfig`.

Metrics are organised into groups selectable via `EvalMetrics`:

| Group | Enum value | Question it answers |
|-------|-----------|-------------------|
| [Nucleotide Classification](#nucleotide-classification) | `NUCLEOTIDE_CLASSIFICATION` | Per-base, how accurate is it? |
| [Region Discovery](#region-discovery) | `REGION_DISCOVERY` | Did we find the right regions? |
| [Boundary Exactness](#boundary-exactness) | `BOUNDARY_EXACTNESS` | How precise are the boundaries? |
| [INDEL Error Taxonomy](#indel-error-taxonomy) | `INDEL` | What structural errors exist at the section level? |
| [Frameshift](#frameshift) | `FRAMESHIFT` | Is the reading frame preserved? |
| [Structural Coherence](#structural-coherence) | `STRUCTURAL_COHERENCE` | Is the overall segment arrangement correct? |
| [Diagnostic Depth](#diagnostic-depth) | `DIAGNOSTIC_DEPTH` | Why is the prediction structurally wrong? |

Additionally, three **cross-cutting analyses** are available as standalone functions (not per-sequence metrics):
- [Complexity-Stratified Evaluation](#complexity-stratified-evaluation)
- [Cross-Model Agreement](#cross-model-agreement)
- [Statistical Significance](#statistical-significance)

---

## Nucleotide Classification

**Enum**: `EvalMetrics.NUCLEOTIDE_CLASSIFICATION` (default metric)

**What it measures**: Per-base accuracy by treating each label class as a binary classification problem (token vs not-token).

**Metrics produced**:

| Metric | Type | Description |
|--------|------|-------------|
| `precision` | float | TP / (TP + FP) — of all positions predicted as this class, how many are correct? |
| `recall` | float | TP / (TP + FN) — of all GT positions of this class, how many were predicted? |
| `f1` | float | Harmonic mean of precision and recall. Only computed at this level. |

**How it works**: Binarises both arrays (target class = 1, everything else = 0), computes a 2x2 confusion matrix via `sklearn.metrics.confusion_matrix`, extracts TP/TN/FP/FN counts.

**Use as training metric**: Yes. All three values are lightweight scalars suitable for per-epoch W&B logging.

---

## Region Discovery

**Enum**: `EvalMetrics.REGION_DISCOVERY` (default metric)

**What it measures**: Whether predicted sections (contiguous runs of a label) correspond to ground-truth sections. Uses 1:1 greedy matching by maximum overlap length.

**Metrics produced** (precision and recall at four strictness levels):

| Level | TP condition | What it forgives |
|-------|-------------|-----------------|
| `neighborhood_hit` | Any overlap between matched GT and pred sections | Both over- and under-prediction |
| `internal_hit` | Matched prediction is entirely inside its GT section | Over-prediction (pred boundaries exceed GT) |
| `full_coverage_hit` | Matched prediction fully covers its GT section | Under-prediction (pred boundaries fall short) |
| `perfect_boundary_hit` | Exact boundary match (sweep-based, no 1:1 constraint) | Nothing — strictest level |

**Matching**: Greedy 1:1 — candidates sorted by overlap length descending; each GT and pred section claimed at most once. Unmatched predictions count as FP, unmatched GT sections as FN. Exception: `perfect_boundary_hit` uses a sweep (any prediction matching any GT section counts) to handle fragmented predictions fairly.

**Use as training metric**: Yes. P/R at each level are scalars.

---

## Boundary Exactness

**Enum**: `EvalMetrics.BOUNDARY_EXACTNESS` (default metric)

**What it measures**: How precise are the boundaries of predicted sections, and what is the systematic boundary error pattern?

**Metrics produced**:

| Metric | Type | Description |
|--------|------|-------------|
| `first_sec_correct_3_prime_boundary` | count | Does the first GT section's 3' end match any prediction? |
| `last_sec_correct_5_prime_boundary` | count | Does the last GT section's 5' end match any prediction? |
| `iou_scores` | list[float] | Intersection-over-Union for every overlapping (GT, pred) pair. |
| `iou_stats` | dict | Distribution statistics (mean, std, min, max, RMSE) of IoU scores. |
| `fuzzy_metrics` | dict | **Boundary bias matrix** (21x21) and **reliability matrix** (11x11). |

**Bias matrix**: A 21x21 heatmap of signed boundary residuals `(pred_start - gt_start, pred_end - gt_end)` clamped to [-10, +10]. Reveals systematic directional boundary errors (e.g., "predictions consistently start 2bp early").

**Reliability matrix**: An 11x11 cumulative recall matrix showing the fraction of GT sections recovered at each boundary tolerance from 0bp to 10bp. Reveals how quickly recall degrades as boundary tolerance tightens.

**Use as training metric**: IoU mean is a training scalar. Bias/reliability matrices are final validation.

---

## INDEL Error Taxonomy

**Enum**: `EvalMetrics.INDEL`

**What it measures**: Classifies every contiguous mismatch region between GT and prediction into one of 8 structural error types.

**Insertion errors** (pred has the class, GT does not):

| Type | Description |
|------|-------------|
| `5_prime_extensions` | Mismatch touches a correct region on its 3' side — prediction extends a section leftward |
| `3_prime_extensions` | Mismatch touches a correct region on its 5' side — prediction extends a section rightward |
| `joined` | Mismatch touches correct regions on both sides — prediction merges two GT sections |
| `whole_insertions` | Mismatch touches no correct region — prediction invents an entirely new section |

**Deletion errors** (GT has the class, pred does not):

| Type | Description |
|------|-------------|
| `5_prime_deletions` | GT section's 5' boundary is eroded |
| `3_prime_deletions` | GT section's 3' boundary is eroded |
| `split` | GT section is split into two by a deletion in the middle |
| `whole_deletions` | Entire GT section is absent from prediction |

**How it works**: Extracts contiguous mismatch groups, then checks one position before and after each group (using background-padded arrays) to determine which boundaries are touched.

**Use as training metric**: No. Produces lists of error arrays (lengths). Final validation metric with stacked bar charts and length histograms.

---

## Frameshift

**Enum**: `EvalMetrics.FRAMESHIFT`

**What it measures**: Whether the prediction preserves the reading frame of the coding sequence. Computes per-position reading frame deviation (mod 3) using a sliding-window codon intersection approach.

**Requirements**: `LabelConfig.coding_label` must be set. Only computed for the coding class. Only valid on single-transcript sequences.

**Use as training metric**: No. Final validation with percentage bar charts.

---

## Structural Coherence

**Enum**: `EvalMetrics.STRUCTURAL_COHERENCE`

**What it measures**: Whether the predicted segment chain is correct **as a whole** — not per-section, but as a complete ordered arrangement. This is the key difference from Region Discovery, which evaluates sections independently.

Contains two sub-metrics:

### Gap Chain Comparison

Compares the ordered sequence of gaps between consecutive segments of a class. For exon segments, gaps correspond to introns — making this the label-agnostic equivalent of intron chain comparison.

| Metric | Type | Description |
|--------|------|-------------|
| `gap_chain_match` | bool | Whether the GT and pred gap chains are identical (all gap boundaries match exactly). |
| `gap_chain_match_rate` | float | Fraction of sequences with an exact gap chain match (aggregated). |
| `gap_chain_lcs_ratio` | float | LCS of gap boundary pairs / `max(len_gt, len_pred)`. Partial credit for partially correct gap chains. Range [0, 1], higher is better. |
| `gap_count_match` | bool | Whether GT and pred have the same number of gaps. |
| `gap_count_match_rate` | float | Fraction of sequences with matching gap counts (aggregated). |
| `gap_count_gt` | int | Number of gaps (= segments − 1) in GT. |
| `gap_count_pred` | int | Number of gaps (= segments − 1) in prediction. |
| `segment_count_gt` | int | Number of segments of this class in GT. |
| `segment_count_pred` | int | Number of segments of this class in prediction. |
| `segment_count_delta` | int | `pred_count - gt_count`. Positive = over-segmentation, negative = under-segmentation. |

**How it works**: For each class, extracts the ordered segment list and derives the gap chain as `[(end_i, start_{i+1})]` pairs. Compares GT and pred gap chains using exact equality and LCS.

**Use as training metric**: Yes. `gap_chain_match_rate` and `gap_chain_lcs_ratio` mean are lightweight scalars.

### Transcript Match Classification

Classifies each `(gt_array, pred_array)` pair into one holistic structural category.

| Class | Condition |
|-------|-----------|
| `exact` | Identical segment chains (same count, same boundaries). |
| `boundary_shift` | Same number of segments but boundary positions differ. |
| `missing_segments` | Prediction chain is a strict ordered subset of GT (some GT segments absent). |
| `extra_segments` | GT chain is a strict ordered subset of prediction (prediction has additional segments). |
| `structurally_different` | None of the above. |
| `missed` | Prediction has no segments of this class. |

**Aggregation**: After processing multiple sequences, a distribution of match classes is reported plus an `exact_match_rate` scalar.

**Use as training metric**: `exact_match_rate` is a training scalar. The full distribution is a final validation metric.

---

## Diagnostic Depth

**Enum**: `EvalMetrics.DIAGNOSTIC_DEPTH`

**What it measures**: Provides causal diagnosis of structural errors — answering *why* the prediction is wrong, not just *that* it is wrong. Contains three sub-metrics.

### Junction Error Taxonomy

Classifies each structural mismatch into a causal error type. This is the junction-level analog of the INDEL taxonomy.

| Error type | Description | How detected |
|-----------|-------------|-------------|
| `exon_skip` | Adjacent GT segments of one class are merged because the intervening segment is absent in prediction. | Unmatched GT segment whose neighbours are both matched. |
| `segment_retention` | A GT segment is absorbed by its neighbours (its label replaced). | Unmatched GT segment whose region is entirely covered by a pred segment of a different label. |
| `novel_insertion` | An extra predicted segment splits a GT segment. | Unmatched pred segment that falls inside a GT segment's span. |
| `cascade_shift` | A boundary error propagates across 3+ consecutive segments. | 3+ consecutive matched pairs with same-sign 5' boundary residuals. |
| `compensating_errors` | Paired boundary errors that cancel each other out. | Adjacent matched pairs where `residual_3p[i] + residual_5p[i+1] ≈ 0`. |

**Matching**: Uses greedy 1:1 matching by maximum overlap (same as Region Discovery).

**Use as training metric**: No. Final validation metric with stacked bar charts.

### Error Correlation Analysis

Analyses the spatial distribution of errors.

| Metric | Type | Description |
|--------|------|-------------|
| `error_clustering_coefficient` | float | Fraction of errors that occur within one median-segment-length of another error. High values indicate errors cluster in bursts. |
| `cascade_lengths` | list[int] | Lengths of cascade shift chains. Aggregated to distribution stats. |
| `compensating_error_rate` | float | Fraction of total errors that are compensating. |

**Use as training metric**: No. Final validation only.

### Structural Summary

Distribution-level and positional diagnostics per label class.

| Metric | Type | Description |
|--------|------|-------------|
| `gt_segment_lengths` | list[int] | Lengths of all GT segments of this class. |
| `pred_segment_lengths` | list[int] | Lengths of all pred segments of this class. |
| `length_emd` | float | Earth Mover's Distance (Wasserstein-1) between GT and pred segment length distributions. Lower is better. Uses `scipy.stats.wasserstein_distance` if available, falls back to quantile-based approximation. |
| `position_bias_5prime_match_rate` | float | Match rate for GT segments in the first 25% of the array. |
| `position_bias_interior_match_rate` | float | Match rate for GT segments in the middle 50% of the array. |
| `position_bias_3prime_match_rate` | float | Match rate for GT segments in the last 25% of the array. |

**Position bias matching**: A GT segment is considered "matched" if any pred segment overlaps it by at least 50% of the GT segment's length.

**Use as training metric**: `length_emd` mean is a training scalar. Position bias rates and segment length distributions are final validation.

---

## Summary: Training vs Final Validation

| Metric | Logged during training | Final validation |
|--------|----------------------|-----------------|
| Nucleotide P/R/F1 | Yes | Yes |
| Region Discovery P/R (4 levels) | Yes | Yes |
| IoU mean | Yes | Full distribution + bias/reliability matrices |
| INDEL counts | No | Stacked bar + length histograms |
| Frameshift | No | Percentage bar |
| Gap chain match rate | Yes | Per-method rate + LCS ratio distribution |
| Gap chain LCS ratio mean | Yes | Full distribution stats |
| Segment count delta mean | Yes | Full distribution stats |
| Exact match rate | Yes | Full match class distribution |
| Length EMD mean | Yes | Full distribution + histograms |
| Junction error taxonomy | No | Counts + rates + stacked bar |
| Error correlations | No | Clustering coefficient, cascade lengths |
| Position bias | No | Per-zone match rates |
