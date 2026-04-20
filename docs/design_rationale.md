# Design Rationale: Strengths Over Existing Benchmarks

This document explains the design decisions behind `dna-segmentation-benchmark` and highlights where it offers capabilities that existing annotation comparison tools (gffcompare, Mikado compare, EGASP, SQANTI3, SpliceAI evaluation) do not.

---

## 1. Strict Intron Chain + Per-transcript Soft Exon Distribution

### What we do

Two complementary structural-chain views are reported side by side:

1. **Strict binary `intron_chain`** — per-sequence `tp/fp/fn ∈ {0, 1}`, where a sequence is a TP **only if** the entire set of GT introns equals the set of predicted introns. Aggregated to corpus precision/recall. This is the gffcompare-compatible reference signal.
2. **Per-transcript soft exon metrics** — two raw per-sequence lists that expose the *distribution* of structural quality across transcripts:
   - `exon_recall_per_transcript` ∈ `[0, 1]`: fraction of GT exons whose `(start, end)` was recovered exactly. A transcript with 9/10 exons right scores 0.9.
   - `hallucinated_exon_count_per_transcript` ≥ 0: number of predicted exons whose `(start, end)` is absent from GT.

### Why both

The strict intron-chain metric hides everything short of a perfect match: a transcript with 9 of 10 exons right is indistinguishable from one with 0 correct. At the corpus level `perfect_boundary_hit` already captures exact exon P/R, but the corpus average collapses the *per-transcript* distribution that matters for diagnosing whether a model "gets most transcripts nearly right" or "gets a few transcripts perfect and the rest completely wrong".

Keeping the per-transcript metrics as raw lists (not reduced to means) lets downstream plotting draw overlayed histograms:

- A thick mass near `recall=1.0` with an empty right tail on hallucinations → the model recovers structure cleanly.
- A thick left tail on recall combined with a fat right tail of hallucinations → the model is guessing.

Splitting recall (coverage) from hallucination count (precision side) is deliberate: unlike F1 the two signals can move independently, so the plot surfaces *which* failure mode dominates.

### How this compares to other tools

| Tool | Strict intron chain | Distribution view | Precision/recall separated |
|------|---------------------|-------------------|----------------------------|
| gffcompare | Binary match | Corpus rate only | No |
| Mikado compare | Intron F1 | Corpus scalar | F1 only |
| EGASP | Intron-level S/Sp | Corpus scalar | Yes |
| **This benchmark** | **Yes (corpus P/R)** | **Per-transcript histograms** | **Recall and hallucinations plotted separately** |

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
| Per-transcript soft exon metrics (recall + hallucination distributions) | No (gffcompare reports corpus scalars only) |
| Transcript match structural taxonomy on label arrays | No (gffcompare class codes require GFF + isoform matching) |
| W&B / ML training-loop integration | No |
| Label-agnostic design (arbitrary token schemes) | No (all tools assume GFF with gene/mRNA/exon/CDS types) |
