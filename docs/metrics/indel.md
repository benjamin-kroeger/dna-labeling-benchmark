# INDEL

The INDEL family classifies contiguous coding mismatches into a structural
error taxonomy instead of reducing them to a single overlap score. Each
mismatch run is additionally **typed by the GT boundary it straddles**, so a
boundary slip at a `5'UTR → CDS` junction is kept separate from one at a
`intron → CDS` junction even though both look identical to the binary coding
mask.

## Example Plots

![INDEL count example](../images/indel_counts.png)

![INDEL error rate by boundary](../images/indel_rates_by_boundary.png)

![INDEL events by boundary](../images/indel_counts_by_boundary.png)

Four complementary views are produced (all boundary-resolved views use one panel
**per method** over a shared, biologically-ordered grid, so methods compare
directly):

- **`indel_counts`** — per method, the total number of each structural error
  type (boundaries aggregated). Answers *which* error types dominate.
- **`indel_rates_by_boundary`** — the headline benchmarking view: a GT-boundary ×
  event-type **rate** heatmap (events ÷ opportunities; see *Rates* below).
  Colour = rate, cell annotation = raw count, grey = no opportunity. Answers
  *how reliably* each method handles each junction, comparably across methods and
  junction types.
- **`indel_counts_by_boundary`** — the raw-magnitude companion: the same grid
  coloured by event count on a **log** scale (so one huge cell does not wash out
  the rest). Answers *where* the errors concentrate in absolute terms.
- **`indel_lengths_<boundary>`** — one figure per exon-position category: a
  4 × 2 grid (insertions top row, deletions bottom row) of overlaid per-method
  run-length histograms. Answers *how large* the slips are at each junction,
  separating small boundary slips from long insertions/deletions or broad
  merge/split regions.

### Run-length distributions by exon position

**Internal exon**

![INDEL length distributions — internal exon](../images/indel_lengths_internal_exon.png)

**5′ terminal exon**

![INDEL length distributions — five prime terminal exon](../images/indel_lengths_five_prime_terminal_exon.png)

**3′ terminal exon**

![INDEL length distributions — three prime terminal exon](../images/indel_lengths_three_prime_terminal_exon.png)

**Single-exon gene**

![INDEL length distributions — single exon gene](../images/indel_lengths_single_exon_gene.png)

## Categories

Insertions are runs where the prediction has coding and GT does not:

- `5_prime_extensions`
- `3_prime_extensions`
- `joined`
- `whole_insertions`

Deletions are runs where GT has coding and the prediction does not:

- `5_prime_deletions`
- `3_prime_deletions`
- `split`
- `whole_deletions`

## How Classification Works

The benchmark pads the GT and prediction coding masks with one background
position on both sides. For each contiguous mismatch run it checks whether the
run touches a *correct* coding region (coding in both GT and prediction)
immediately on the 5' side, the 3' side, both, or neither.

That local neighborhood test drives the event type:

- one touched side: extension or deletion at that end
- both touched sides: `joined` or `split`
- neither side: whole insertion or whole deletion

## Boundary Typing

On top of the event type, each run keeps the **GT label names immediately
flanking it**, read from the unpadded GT label array:

```
left_name  = name(gt_labels[run_start - 1])   # 5' flank
right_name = name(gt_labels[run_end   + 1])   # 3' flank
```

The two are joined into a single boundary key `"LEFT:RIGHT"` (rendered
`LEFT → RIGHT` in plots), for example `FIVE_PRIME_UTR:CDS` or `CDS:INTRON`.
A run that touches a sequence end uses `none` for the missing flank
(e.g. `none:CDS`).

Because keys are in **array (5'→3') order**, `FIVE_PRIME_UTR:CDS` (a
5'UTR / start-codon boundary) and `CDS:THREE_PRIME_UTR` (a 3'UTR / stop-codon
boundary) stay distinct — a distinction a class-blind metric cannot make, since
UTR appears at both ends of a gene. This is only biologically meaningful under
the standard 5'→3' input orientation (see {doc}`conventions`).

## Output Structure

`eval_indel` returns

```python
{
    "by_boundary": {"LEFT:RIGHT": {event_type: [run_length, ...]}},
    "junction_opportunities": {"LEFT:RIGHT": int},  # GT L→R transition counts
    "n_gt_segments": int,                           # GT coding segments
    "n_pred_segments": int,                         # predicted coding segments
}
```

Only run **lengths** (in nucleotides) are stored, not index arrays: that is all
the plots need, and it keeps the per-boundary fan-out small. The label-name
keys are produced once, where `label_config` is available, so the same string
keys are used by the tests, the accumulator merge, the plots, and JSON output
(a tuple key would not be JSON-serialisable). The three denominators are summed
across sequences by the accumulator.

## Rates and opportunities

Raw counts are not comparable across methods, test sets, or junction types
because they have no denominator. Each event is therefore divided by its
**opportunity** count to give a rate, with a denominator chosen per event family:

| event family | events | denominator (opportunity) | reads as |
|---|---|---|---|
| boundary-anchored | 5'/3' extensions & deletions | GT junctions of the matching `LEFT:RIGHT` type (`junction_opportunities`) | fraction of those junctions slipped |
| GT-segment | `split`, `joined`, `whole_deletions` | number of GT coding segments | per-segment fragmentation / loss |
| pred-segment | `whole_insertions` | number of predicted coding segments | fraction of predictions fully spurious |

This split is deliberate: boundary-anchored events measure *localization* of a
junction the method got roughly right, whereas `whole_*` events are *detection*
(presence/absence) — closer to recall (`whole_deletions`) and precision
(`whole_insertions`). The `indel_rates_by_boundary` plot colours by these rates;
cells with zero opportunity are masked.

## Interpretation

- many `3_prime_extensions` / `3_prime_deletions`: systematic end-boundary
  problems
- many `joined`: the model tends to merge adjacent GT coding sections
- many `split`: the model fragments single GT coding sections
- many `whole_insertions`: strong hallucination behavior
- a hotspot at one boundary in `indel_counts_by_boundary` (e.g. `CDS:INTRON`
  3'-deletions): the model consistently fails one specific junction type. Read
  it together with the state-transition *direction* (late vs premature) for the
  same boundary — INDEL gives the slip magnitude, transitions give the timing.

## Caveats

- This taxonomy is scope-specific. "Insertion" / "deletion" are defined
  relative to the configured `evaluation_scope` positive-token set (see
  {doc}`../getting_started/annotation_modes`). The event type comes from the
  binary coding mask; the **boundary type** then restores the GT label context
  (intron vs UTR vs background) that the mask discards.
- The boundary key describes the run's **edges**, not its interior. A run whose
  interior spans several GT labels (e.g. a hallucinated coding stretch over
  UTR–intron–UTR) collapses to the single pair of its flanks, e.g.
  `none:none` or `NONCODING:NONCODING`.
- It is local. It classifies contiguous mismatch runs, not full transcript
  structure.
- 5'/3' and the boundary key order refer to **array orientation**, not
  biological strand. Minus-strand input in genomic coordinates would invert
  the labels. See {doc}`conventions`.
- The run-length distribution plots use a `log10` x-axis with tick labels
  re-projected to linear bp. A peak labelled "100" represents bins *near*
  100 bp, not exactly 100.
