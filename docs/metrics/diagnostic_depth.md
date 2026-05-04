# Diagnostic Depth

Diagnostic Depth complements the per-section and structural metrics with two
distribution-level diagnostics that expose *where* and *how severely* a
predictor fails, rather than just whether it does.

Both metrics are computed for a single label class (typically `EXON`/CDS) and
aggregated across the corpus.

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

Three histograms are emitted, all with 100 bins normalised to the coding
span (bin 0 = start of the first GT coding segment, bin 99 = end of the
last):

- `position_bias_histogram_fn` — counts GT coding positions that the
  prediction did not cover (under-prediction / false negatives).
- `position_bias_histogram_fp` — counts predicted coding positions that
  fall inside the GT coding span but are not in GT (over-prediction /
  false positives).
- `position_bias_histogram` — element-wise sum of the two above, retained
  for backwards compatibility with older logs.

Predicted nucleotides outside the GT coding span are clipped before
binning, keeping every histogram bounded to the gene locus.

Earlier versions of the benchmark collapsed FN and FP into a single XOR
count, which made it impossible to tell a model that systematically
*deletes* coding bases apart from one that systematically *inserts*
them. The split version preserves that distinction.

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
