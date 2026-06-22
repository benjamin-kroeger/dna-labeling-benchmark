# Region Discovery

Region Discovery answers a section-level question: did the predictor recover
the right coding runs, independent of fine-grained boundary residuals?

## How Sections Are Matched

The benchmark first extracts contiguous GT and predicted coding sections. It
then finds all overlapping `(GT, pred)` pairs and sorts them by overlap
length. A greedy 1:1 assignment claims the largest overlaps first.

`perfect_boundary_hit` is different: it uses a sweep over all sections and
counts any exact-boundary match, without the 1:1 assignment.

## Output Structure

Region Discovery produces four precision/recall tiers that **nest by
strictness**:

```
neighborhood_hit  ⊇  { internal_hit , full_coverage_hit }  ⊇  perfect_boundary_hit
```

| Tier | TP criterion | Question answered |
|---|---|---|
| `neighborhood_hit` | matched pair overlaps at all | Did we detect the region? |
| `internal_hit` | prediction lies within GT (`pred ⊆ GT`) | Detected, but under-extended? |
| `full_coverage_hit` | prediction spans GT (`pred ⊇ GT`) | Detected, but over-extended? |
| `perfect_boundary_hit` | both boundaries exact | Did we reproduce exact boundaries? |

Every tier is a coherent contingency table: `TP + FN = total GT` and
`TP + FP = total pred`. `internal_hit` and `full_coverage_hit` are **inclusive
of an exact match**, so a perfect pair counts toward both — that is what makes
the nesting hold.

## Hardened FP — the same rule for every tier

Each tier hardens its FP to `total_pred − that-tier's hits` (the rule
`perfect_boundary_hit` already used). A matched pair that fails a tier's
spatial test is booked **as both an FP** (the prediction is not a hit for this
tier) **and an FN** (the GT is not hit for this tier). This is the standard
Burset/Eval exon-level behavior, and it is what makes precision and recall
well-defined at every level of strictness.

This replaces an earlier `containment` block that reported `internal`/
`full_coverage` as *conditional rates over matched pairs* with a borrowed FP.
Hardening the FP turns them into ordinary detection tiers, so the whole family
reads as one precision/recall ladder.

## The Four P/R Tiers

### `neighborhood_hit`

![region_discovery_neighborhood_hit.png](../images/region_discovery_neighborhood_hit.png)

TP if the matched prediction overlaps the GT section at all. The most forgiving
detection tier — any contact counts.

### `internal_hit`

TP if the matched prediction lies within its GT section (`pred ⊆ GT`, inclusive
of an exact match) — i.e. the prediction did not over-run the GT boundaries. A
matched pair that over-extends past the GT is an FP (and its GT an FN).

### `full_coverage_hit`

TP if the matched prediction fully spans its GT section (`pred ⊇ GT`, inclusive
of an exact match) — i.e. the prediction did not fall short of the GT
boundaries. A matched pair that falls short is an FP (and its GT an FN).

`internal_hit` and `full_coverage_hit` give the **direction** of the boundary
error: contrast their recalls to read under- vs over-extension.

### `perfect_boundary_hit`

![region_discovery_perfect_boundary_hit.png](../images/region_discovery_perfect_boundary_hit.png)

TP only when both boundaries match exactly. Unlike the other tiers this is
sweep-based rather than 1:1 matched, which prevents fragmented predictions from
being miscounted purely because one fragment already claimed a GT section in
the greedy assignment.

## Double-Penalty Behavior

All tiers intentionally treat GT sections and prediction sections as separate
objects. When one GT section is split into two predictions, or two GT sections
are merged into one prediction, you often get both a false negative on the GT
side and a false positive on the prediction side. That is the right behavior if
you want structural section recovery rather than base-level overlap alone.

## Interpretation

- high `neighborhood_hit` recall, low `perfect_boundary_hit` recall: the model
  usually finds the right locus but misses exact boundaries
- high `internal_hit` recall, low `full_coverage_hit` recall: predictions tend
  to be too short (under-extended)
- low `internal_hit` recall, high `full_coverage_hit` recall: predictions tend
  to be too long (over-extended)
- both `internal_hit` and `full_coverage_hit` recall near `neighborhood_hit`:
  matched predictions are close to exact even when `perfect_boundary_hit` is
  lower (a fragmentation effect)

## Caveats

- These metrics are coding-section metrics. They do not use intron labels.
- They are not transcript-chain metrics. Two transcripts can have good section
  discovery while still failing strict structural coherence.
- `perfect_boundary_hit` TP/FP/FN counts come from the sweep, so absolute
  counts are not directly comparable to the 1:1-matched tiers.
- Aggregation is micro-averaged across sequences (see {doc}`conventions`):
  per-sequence integer counts are summed before ratios are computed, so long
  sequences dominate the corpus score.
