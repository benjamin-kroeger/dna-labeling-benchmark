# Diagnostic Depth

Diagnostic Depth complements the per-section and structural metrics with
distribution-level diagnostics that expose *where* and *how severely* a
predictor fails, rather than just whether it does.

Three diagnostics are emitted and aggregated across the corpus:

- **Length EMD** and the **binary position-bias histograms** are computed for
  the active evaluation scope (one positive class), kept for continuity.
- **Per-class position bias** generalises the position histogram to *every*
  evaluated label, normalised to each label's own span and resolved by the
  confused partner — see [Per-Class Position Bias](#per-class-position-bias).

---

## Length EMD

`length_emd` measures how different the predicted segment length distribution
is from the GT distribution using the 1-D Wasserstein distance (Earth Mover's
Distance).

The Wasserstein distance is the minimum "work" needed to transform one
distribution into the other, where work is mass × distance moved. In this
context:

- Low EMD — the predictor produces segments with lengths that closely match
  GT, even if individual segment boundaries are wrong.
- High EMD — the predictor systematically produces segments that are too short
  or too long relative to the GT distribution.

EMD is insensitive to segment identity: it does not penalise a predictor for
getting individual segments wrong as long as the overall length distribution is
correct. This makes it a useful complement to chain-based metrics — a good EMD
with a poor chain score indicates that boundaries are placed but at the wrong
locations; a poor EMD with a good chain score would indicate something unusual
about boundary placement without length distortion.

**Implementation note:** `scipy.stats.wasserstein_distance` is used when
available; otherwise a quantile-interpolation fallback is applied. Both return
0.0 if either the GT or predicted segment list is empty.

---

## Error Location Bias (Position Bias Histograms)

The position-bias diagnostic answers: *where in a transcript do nucleotide-level
errors concentrate, and are they under-predictions or over-predictions?*

Two histograms are emitted, both with 100 bins normalised to the coding
span (bin 0 = start of the first GT coding segment, bin 99 = end of the
last):

- `position_bias_histogram_fn` — counts GT coding positions that the
  prediction did not cover (under-prediction / false negatives).
- `position_bias_histogram_fp` — counts predicted coding positions that
  fall inside the GT coding span but are not in GT (over-prediction /
  false positives).

Predicted nucleotides outside the GT coding span are clipped before
binning, keeping every histogram bounded to the gene locus. Splitting
FN and FP makes it possible to tell a model that systematically
*deletes* coding bases apart from one that systematically *inserts*
them.

### Reading the plot

![Error Location Bias](../images/position_bias.png)

The figure shows two side-by-side panels:

- **Left** — false-negative density. A spike here means the predictor is
  *missing* GT coding bases at that relative position.
- **Right** — false-positive density. A spike here means the predictor
  is *adding* coding bases at that relative position that GT does not
  agree with.

Both x-axes show position in the coding span as a percentage (0 % =
transcript start, 100 % = transcript end). The y-axis is the cumulative
count of mismatch nucleotides across all evaluated sequences.

Common patterns and their interpretations:

| Shape | Interpretation |
|---|---|
| Flat / near-zero in both panels | Predictions closely match GT at the nucleotide level across the whole span. |
| Elevated at 0 % and/or 100 % in both panels | Terminal boundary errors — the predictor struggles with gene-locus start or end. Common in models that lack UTR context. |
| Elevated in the middle | Internal exon errors — splice-site accuracy degrades for exons far from the transcript termini. |
| FN-heavy, FP-flat | Predictor consistently truncates / under-calls coding bases. |
| FP-heavy, FN-flat | Predictor consistently over-extends / hallucinates coding bases. |
| Uniform low-level signal | Systematic boundary wobble (e.g. a consistent 1-nt shift) affecting all segments equally. |
| Single sharp spike | A positional bias concentrated at one relative location; suggests a systematic offset tied to the model's context window. |

A method with accurate boundaries will show a low, flat curve in both
panels. Comparing curves across methods quickly reveals whether a method
degrades more at transcript boundaries vs. internal exons, and whether
its dominant failure mode is missing or adding coding bases.

### Caveats

- The histogram is aggregated across the corpus by element-wise sum
  (long sequences contribute more counts). This is intentional — the
  metric measures absolute error mass, not per-sequence rates.

---

## Per-Class Position Bias

The binary histograms above answer "where does the *scope* class go wrong."
The per-class diagnostic answers the same question **for every label at once**,
and adds *what* each error was confused with.

For each evaluated (non-background) label that appears in the GT, its own GT
span `[min, max]` is split into 100 bins and three arrays are accumulated:

- `fp` — shape `(100, P)`: predicted-as-this-label positions absent from GT
  there (over-prediction), split by the **GT** partner label it replaced.
- `fn` — shape `(100, P)`: GT positions of this label the prediction got wrong
  (under-prediction), split by the **predicted** partner label it became.
- `total` — shape `(100,)`: bin occupancy, the denominator for an error *rate*.

`P` is the full partner vocabulary (`position_bias_partners`), including
background. Predicted positions outside the label's span are clipped to the
nearest edge bin (terminal over-extension).

**Own-span normalization** is the key design choice: because each label is
plotted on *its own* 0–100 %, boundaries align across heterogeneous transcripts
even when introns sit at different absolute positions. Terminal errors
(start/stop codon, splice slips, TSS/polyA trimming) concentrate into sharp
peaks instead of smearing.

### Reading the plot

The plot renders one track per label, 0-centered (**FP above, FN below**), as
an **error rate** (`fp / total`, `fn / total`) so abundant classes such as CDS
don't dominate. All tracks share one y-scale, autoscaled to the largest rate.
Each bar is colour-stacked by the confused partner, so a track shows *where*
along the class, *how bad*, and *what* it was confused with at once.

Typical UTR_CDS_INTRON readings:

| Track / location / colour | Interpretation |
|---|---|
| 5′UTR FN at 100 %, coloured CDS | start codon called early (UTR tail became CDS) |
| CDS FP at 0 %, coloured 5′UTR | CDS over-extended into the 5′UTR |
| CDS FN/FP at 100 %, coloured 3′UTR | stop codon misplacement |
| 3′UTR FN at 100 %, coloured background | the 3′ tail was trimmed / missed |
| intron FP/FN at 0 % and 100 %, coloured CDS | splice-boundary slips at intron termini |

In `EXON_INTRON` mode the same plot degrades gracefully to an `EXON` track
(plus `INTRON` when an intron label is configured).

### Caveats

- Rates are micro-averaged: `fp`, `fn`, and `total` are summed across sequences
  and divided once, so the rate reflects corpus-level error density.
- A label that never appears in the GT produces no track.
