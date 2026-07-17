# Conventions

A handful of conventions are shared across every metric family. Reading
this page once removes a lot of small surprises later.

## Coordinate semantics

* The per-transcript metric families operate on **paired 1-D integer
  label arrays**, not on GFF/GTF records directly. The pipeline converts
  annotation files to arrays before scoring. The genome-wide
  `compute_global_metrics` path is the exception — it scores GFF/GTF
  records (intervals keyed by `seqid`, strand, `start`, `end`) directly;
  see [Strand awareness](#strand-awareness).
* Segment endpoints are treated as **inclusive** on both ends and
  **0-based**. A segment `(start=5, end=5)` has length `1`, not `0`,
  and the IoU formula uses `end - start + 1` for both intersection and
  union. Tools that report scores with half-open intervals will produce
  slightly different numbers for very short segments — IoU on a 1 bp
  segment is the most pronounced case.

## Strand awareness

Array-level metrics are **strand-blind**. They operate on whatever
1-D arrays the caller provides; they never check whether a sequence
came from the plus or minus strand of a reference genome.

In the docs:

* `5'` always means *lower array index* (the start of the array).
* `3'` always means *higher array index* (the end of the array).

For sequences derived from minus-strand transcripts the array is
already in transcript orientation if you used the GTF/GFF pipeline, so
"5' end" and "biological 5' end" agree. If you build arrays by hand
without re-orienting minus-strand transcripts, a metric named
`5_prime_extensions` will refer to the array start, which is the
biological 3' end — the doc's interpretive language is then inverted.

The genome-wide `compute_global_metrics` path (`global_metrics.py`)
*does* keep strand and uses it as part of the matching key.

## Aggregation across sequences

When `benchmark_from_arrays` is given many sequences, the
per-sequence outputs are merged element-wise before any ratio is
computed:

* Confusion counts (`tp`, `fp`, `fn`, `tn`) are **summed** across
  sequences. Precision, recall, and F1 are then derived from the
  totals — i.e. they are **micro-averaged** rather than macro-averaged.
  Long sequences contribute proportionally more to the score than
  short ones.
* Distribution-style outputs (IoU values, frame deviations, segment
  lengths, position-bias histograms) are **concatenated** or summed
  bin-wise. Summary statistics (`mean`, `mae`, `rmse`, `std`) are
  computed once on the corpus-level pool, not averaged-of-averages.
* Boundary-residual landscapes pool (concatenate) the raw residual values
  across sequences; binning into the bias/reliability matrices happens once,
  on the pooled values.
* Transition matrices are summed cell-wise.

* `NUCLEOTIDE_CLASSIFICATION` and `REGION_DISCOVERY` outputs include
  automatic macro-averaged precision/recall/F1 (equal weight per
  sequence, regardless of length) alongside the micro-averaged
  totals. These are emitted as `precision_macro`, `recall_macro`,
  `f1_macro` and their standard-error estimates. Other metric families
  report micro-averaged scores only.

If you need macro-averaged scores for other families (one rate per sequence,
then mean) you must build them yourself from the per-sequence outputs returned
by `return_individual_results=True`.

## Sentinel values

* `state_transitions.py` uses the current GT label as a sentinel for the
  `prev_GT` / `next_GT` lookups when no earlier or later GT transition exists —
  it can never match, so edge positions are never classified as `late_catchup`
  or `premature`.

## Annotation modes and scopes

Every `LabelConfig` declares an `annotation_mode` (`EXON_INTRON` or
`UTR_CDS_INTRON`) that fixes what the positive labels mean, plus an
`evaluation_scope` that fixes the positive-token set the per-transcript metrics
operate on:

* `transcript_exon` — all transcribed exonic sequence. In `EXON_INTRON` that is
  the single exon label; in `UTR_CDS_INTRON` it is `5' UTR + CDS + 3' UTR`.
* `cds` — the coding span only, available in `UTR_CDS_INTRON`.

Per-transcript results run on the configured `evaluation_scope` and record it
under `metadata`. The genome-wide `compute_global_metrics` path ignores
`evaluation_scope` and instead nests *every* available scope under a `scopes`
key. `PHASE_DRIFT` is only defined in `UTR_CDS_INTRON` with the `cds` scope. See
the {doc}`../getting_started/annotation_modes` guide for the full model.

## Background and intron labels

* `LabelConfig.background_label` is filtered out by
  `extract_structure(..., exclude_background=True)` (the default).
  Only labels you care about contribute segments.
* `LabelConfig.intron_label` is required for any intron-chain
  metric. If your annotation has only coding labels, set
  `infer_introns=True` on the `benchmark_gt_vs_pred_*` entry points
  to fill the gaps automatically — the heuristic is documented in the
  array-benchmark getting-started page.

## Plot conventions

* No plotting function ever calls `plt.show()` — the caller decides.
* Every plotting function accepts an optional `save_path` and
  returns the `matplotlib.figure.Figure`, ready for `wandb.log()` or
  similar.
* Pictogram panels (right-side icon + description) are added when a
  matching entry exists in `PLOT_METADATA`. Missing icons are skipped
  silently rather than raising.
