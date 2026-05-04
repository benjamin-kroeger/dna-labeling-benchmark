# Boundary Exactness

Boundary Exactness measures how far predicted coding sections deviate from GT
boundaries once there is some overlap.


## What Is Computed

For every overlapping `(GT, pred)` coding section pair, the benchmark records:

- signed 5' residual: `pred_start - gt_start`
- signed 3' residual: `pred_end - gt_end`
- IoU score for the overlapping pair

It also records two terminal-boundary flags:

- `first_sec_correct_3_prime_boundary`
- `last_sec_correct_5_prime_boundary`

These are stored per sequence in
`BOUNDARY_EXACTNESS` and then summarized across many sequences.

## IoU

The Intersection-over-Union for a single overlapping `(GT, pred)` coding
section pair is computed with **inclusive 0-based endpoints**:

```
intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start) + 1)
union        = max(gt_end, pred_end) - min(gt_start, pred_start) + 1
IoU          = intersection / union   (or 0.0 when union == 0)
```

The `+1` terms reflect inclusive bounds — a section with `start == end`
has length 1, not 0. Scores produced by external tools that use
half-open intervals will differ slightly for very short sections; see
`conventions.md`.

The raw `iou_scores` list contains one IoU value per overlapping
`(GT, pred)` pair. After aggregation, `iou_stats["mean"]` is the scalar
used by the W&B online logger.

![Average IoU](../images/iou_average.png)

![IoU cumulative distribution](../images/iou_distribution.png)

## Boundary Residual Landscape

The raw `fuzzy_metrics["boundary_residuals"]` list contains all signed
residual pairs from overlapping sections. Aggregation turns that into the
boundary bias landscape which can show if certain numbers or nucleotides are consistently over or under predicted.
The cumulative reliability highlights how recall improves if each boundary is counted as an exact match given x 
nucleotides of error in both 5' and 3' direction.

![Boundary residuals](../images/example_boundary_landscape.png)


Interpretation:

- values centered near `(0, 0)`: boundaries are usually exact
- mass shifted to negative 5' residuals: predictions start too early
- mass shifted to positive 3' residuals: predictions end too late

## Terminal Boundary Flags

Two binary flags isolate the splice sites adjacent to the **terminal**
coding sections:

- `first_sec_correct_3_prime_boundary` — 1 iff some prediction's
  downstream end matches the **first** GT coding section's downstream
  end. For a multi-exon transcript this is the inner (3'-side) splice
  site of the leading exon — i.e. the boundary between the first exon
  and the first intron.
- `last_sec_correct_5_prime_boundary` — 1 iff some prediction's upstream
  start matches the **last** GT coding section's upstream start. This
  is the inner (5'-side) splice site of the trailing exon.

These are not the outer transcript termini; they target the splice
junctions that are most often hardest to nail when a model otherwise
finds the right gene locus.

The flag names use **array-orientation** (5' = lower array index, 3' =
higher array index). They do not encode biological strand — on the
minus strand the array-5' end corresponds to the biological 3' end of
the transcript. See `conventions.md` for the full convention.

## Caveats

- Only overlapping section pairs contribute. Completely missed GT sections do
  not add IoU or residuals directly; they matter through Region Discovery.
- Multiple predicted sections can contribute multiple residual pairs against
  the same GT section.
- IoU mean is informative, but it hides whether errors are a few large misses
  or many small offsets. Use it together with the distribution plot.
- The bias-landscape histogram is bounded to `±max_range` (default 10 bp).
  Residuals beyond that range silently fall outside the bias matrix while
  still contributing to the cumulative-recall surface.
- See {doc}`conventions` for the inclusive-endpoint rule that governs IoU
  and the array-orientation rule that governs the 5'/3' field names.
