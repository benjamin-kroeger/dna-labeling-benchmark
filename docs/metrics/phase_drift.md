# Phase Drift

Phase Drift measures whether the prediction stays in step with the ground truth
by coding-base count where the two overlap. It is a **structural comparison
signal**, not a biological reading-frame or frameshift-mutation detector — it
never reads the GFF `phase` column.

The metric enum is `EvalMetrics.PHASE_DRIFT`. It is only valid in
`UTR_CDS_INTRON` mode with `evaluation_scope=BenchmarkScope.CDS`; in any other
configuration the benchmark raises a `ValueError` rather than scoring a
UTR-contaminated mask. See {doc}`../getting_started/annotation_modes`.

## What Is Computed

At each position covered by **both** the GT and predicted CDS masks, the
benchmark stores

```
abs(cumcount_pred - cumcount_gt) % 3
```

where `cumcount` is the number of CDS bases seen from the start of the array up
to and including that position. A value of `0` means the two annotations are in
lockstep by CDS-base count; `1` or `2` means one is ahead by that many bases.

Positions where GT or prediction is non-coding are filled with `np.inf` and
skipped by the plotting layer; only finite values are bucketed into
*shift 0* / *shift 1* / *shift 2*. The plotting layer reports those three
buckets as percentages.

## Skipping (not raising) on degenerate inputs

Unlike the mode/scope check above, malformed *content* does not raise — the
sequence is skipped and counted so aggregation stays honest:

- **GT CDS length not divisible by 3** → the sequence is skipped, a warning is
  logged, and `n_skipped_non_divisible` is incremented. This usually means
  non-CDS positions (e.g. UTRs) leaked into the CDS mask, or the CDS is
  incomplete.
- **Fewer than 3 predicted CDS bases** → too short to form a codon-count
  window, so the sequence is skipped and `n_skipped_short` is incremented.

In both cases the per-sequence `frames` list is empty.

The raw single-sequence output is:

```python
{
    "gt_frames": [...],            # per-position drift, inf at non-co-CDS positions
    "n_skipped_non_divisible": int,
    "n_skipped_short": int,
}
```

When INDEL is requested alongside Phase Drift, two extra counts are added —
`boundary_indel_total` and `boundary_indel_in_frame` — the number of
boundary-anchored indel events (5'/3' extensions and deletions) and how many of
those have lengths divisible by 3.

## Interpretation

- mostly shift 0: the prediction stays in step where coding overlaps
- large shift-1 or shift-2 mass: coding alignment is structurally wrong even if
  some boundaries look reasonable

## Caveats

- Only meaningful for transcript-level coding inputs — ideally one transcript
  per array, with the full CDS present so GT coding positions form complete
  codons.
- It is not a substitute for Structural Coherence. It only checks modulo-3
  drift on positions where GT and prediction *both* mark coding.
