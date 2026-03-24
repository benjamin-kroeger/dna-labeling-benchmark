# Design Rationale: Strengths Over Existing Benchmarks

This document explains the design decisions behind `dna-segmentation-benchmark` and highlights where it offers capabilities that existing annotation comparison tools (gffcompare, Mikado compare, EGASP, SQANTI3, SpliceAI evaluation) do not.

---

## 1. Label-Agnostic Gap Chain — Not TP/FP/FN

### What we do

The gap chain metric compares the **ordered sequence of gaps** between consecutive segments of a given class.  For exon segments, gaps correspond to introns — making this the label-agnostic equivalent of an intron chain comparison.

Three levels are reported:

| Metric | Description |
|--------|-------------|
| `gap_chain_match_rate` | Fraction of sequences where the entire gap chain is identical (binary, like gffcompare's intron chain match). |
| `gap_chain_lcs_ratio` | Longest common subsequence of gap boundary pairs as a fraction of the longer chain. Provides partial credit. |
| `gap_count_match_rate` | Fraction of sequences with the correct number of gaps (necessary but not sufficient for exact match). |

### Why edit/LCS-based instead of TP/FP/FN

A TP/FP/FN formulation requires **1:1 matching** — each ground-truth gap must be paired with a predicted gap, and unmatched items become FN or FP.  This is problematic for ordered chains:

1. **Matching ambiguity.**  When gaps shift, split, or merge, the assignment is not obvious.  Which GT gap does a shifted pred gap "belong" to?  Different matching heuristics (greedy overlap, Hungarian, etc.) give different TP/FP/FN counts for the same prediction.

2. **Ordering information is lost.**  Consider:

   ```
   GT gaps:   [(20,30), (40,50), (60,70)]
   Pred A:    [(20,30), (60,70)]           # middle gap missing, flanking gaps correct
   Pred B:    [(20,30), (40,50)]           # last gap missing, first two correct
   ```

   Under TP/FP/FN with any reasonable matching, both predictions score TP=2, FN=1, FP=0.  But Pred B preserves the first two gaps *in sequence* while Pred A skips a gap in the middle — a different and arguably worse structural error.  LCS captures this: Pred B has LCS=2 from a contiguous subsequence, while Pred A also has LCS=2 but from a non-contiguous one.  (In this simple case the LCS value is the same, but in longer chains the distinction becomes significant.)

3. **No partial credit at the chain level.**  gffcompare reports intron chain match as a binary per-transcript verdict (match or no match).  TP/FP/FN at the individual gap level gives per-gap counts but not a sense of "how close was the overall chain?"  The LCS ratio fills this gap: a transcript with 9 of 10 introns correct in order scores 0.9, not just "miss".

### How this compares to other tools

| Tool | Intron chain metric | Partial credit | Ordering-aware |
|------|-------------------|----------------|----------------|
| gffcompare | Binary match (exact intron chain or not) | No | Implicit (binary) |
| Mikado compare | Intron F1 (TP/FP/FN on individual junctions) | Per-junction | No (bag of junctions) |
| EGASP | Intron-level sensitivity/specificity | Per-intron | No |
| **This benchmark** | **Gap chain match + LCS ratio** | **Per-chain (continuous 0–1)** | **Yes (LCS preserves order)** |

---

## 2. INDEL Error Taxonomy — Causal Diagnosis, Not Just Counts

### What we do

Every mismatch region between GT and prediction is classified into one of 8 structural error types:

| Error type | Description |
|-----------|-------------|
| 5' extension | Prediction extends past the GT start |
| 3' extension | Prediction extends past the GT end |
| Whole insertion | Entirely spurious predicted segment |
| Joined | Two or more GT segments merged into one prediction |
| 5' deletion | Prediction starts after the GT start |
| 3' deletion | Prediction ends before the GT end |
| Whole deletion | GT segment entirely absent from prediction |
| Split | One GT segment broken into multiple predictions |

### Why this matters

Existing tools report *that* a prediction is wrong, but not *how*.  gffcompare class codes (=, c, k, j, e, ...) classify transcript-level relationships but do not diagnose section-level structural errors.  Knowing that a model systematically produces 3' extensions (but not 5' extensions) is directly actionable — it points to a specific failure mode in the model's boundary prediction, not just a generic accuracy gap.

No other annotation comparison tool provides this level of causal diagnosis at the section level.

---

## 3. Boundary Bias and Reliability Matrices — Systematic Error Detection

### Boundary bias matrix (21x21)

For every overlapping (GT, pred) section pair, we compute signed boundary residuals: `(pred_start - gt_start, pred_end - gt_end)`, clamped to [-10, +10].  These are accumulated into a 2-D histogram.

This reveals **systematic directional boundary errors** — e.g., "predictions consistently start 2bp early and end 1bp late".  No other tool provides this.  The closest analogue is examining individual boundary residuals manually, which does not scale.

### Reliability matrix (11x11)

A cumulative recall matrix showing what fraction of GT sections are recovered at each boundary tolerance (0–10bp for start and end independently).  This answers: "If I allow 3bp tolerance on the start boundary but require exact end boundaries, what is my recall?"

This is critical for evaluating whether a model's boundary errors are uniformly small or have a long tail of catastrophic misplacements.

---

## 4. Frameshift Analysis — Reading Frame Preservation

For coding sequences, we measure whether the predicted exon structure preserves the reading frame via sliding-window codon intersection (mod-3 deviation).

This is absent from gffcompare, Mikado, EGASP, and SQANTI3.  It is particularly important for evaluating gene finders that will be used for protein prediction: a prediction that finds the right exons but shifts the reading frame by 1bp is functionally useless for downstream protein analysis despite potentially scoring well on boundary metrics.

---

## 5. State Transition Analysis — Error Localisation

We compute a confusion matrix at GT class-change positions (where the ground truth transitions between labels) and separately analyse false transitions at stable positions (where the GT label is constant but the prediction transitions).

This separates two fundamentally different error modes:
- **Boundary confusion**: the model detects a transition but assigns the wrong class pair
- **False transitions**: the model hallucinates boundaries where the GT is stable (fragmentation)

No other annotation comparison tool provides this decomposition.

---

## 6. ML Training-Loop Integration

### The gap

Every existing annotation comparison tool (gffcompare, Mikado, EGASP) is a **batch CLI tool**: you run it after inference, parse its text output, and manually extract numbers.  This workflow does not integrate into a training loop.

### What we provide

- **numpy array API**: input is `list[np.ndarray]`, not GFF files — zero I/O overhead during training
- **W&B adapter**: `log_benchmark_scalars()` logs lightweight scalar metrics per step; `log_benchmark_full()` logs figures and distributions post-hoc
- **matplotlib figures**: `compare_multiple_predictions()` returns a `dict[str, Figure]` ready for any logging backend
- **Configurable metric subsets**: `EvalMetrics` enum lets you compute only the metrics you need (e.g., just `NUCLEOTIDE_CLASSIFICATION` during training, full suite at validation)

This makes the benchmark usable both as a training metric (like accuracy or loss) and as a final evaluation tool — a dual-use design that no competing tool offers.

---

## 7. Transcript Match Classification — Structural Taxonomy

Each (GT, prediction) pair is classified into one holistic structural category: `exact`, `boundary_shift`, `missing_segments`, `extra_segments`, `structurally_different`, or `missed`.

This provides a **distribution over failure modes** across sequences, answering "What fraction of my predictions are structurally correct vs. just boundary-shifted vs. missing segments?"  gffcompare's class codes serve a similar purpose at the transcript level but are tied to GFF input and isoform-aware matching.  Our classification operates directly on label arrays and is label-agnostic.

---

## Summary: Unique Capabilities

| Capability | Available in any competing tool? |
|-----------|--------------------------------|
| 8-type INDEL error taxonomy | No |
| Boundary bias matrix (21x21 signed residuals) | No |
| Reliability matrix (11x11 cumulative recall at tolerances) | No |
| Frameshift analysis | No |
| State transition confusion analysis | No |
| Ordering-aware gap chain with partial credit (LCS ratio) | No (gffcompare is binary; Mikado is bag-of-junctions) |
| Transcript match structural taxonomy on label arrays | No (gffcompare class codes require GFF + isoform matching) |
| W&B / ML training-loop integration | No |
| Label-agnostic design (arbitrary token schemes) | No (all tools assume GFF with gene/mRNA/exon/CDS types) |
