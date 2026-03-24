# Benchmark Project — Agent Context

## What This Project Is

A Python library and CLI (`dna-segmentation-benchmark`) for evaluating **nucleotide-level DNA segmentation models** (gene finders like SegmentNT, Helixer, Tiberius, Augustus) against reference annotations (e.g., GENCODE). It bridges the gap between classical bioinformatics annotation comparison tools (gffcompare, Mikado) and ML training-loop evaluation.

Primary entry point: `dna-benchmark` CLI. Core API: `benchmark_gt_vs_pred_single` / `benchmark_gt_vs_pred_multiple` in `src/dna_segmentation_benchmark/eval/evaluate_predictors.py`.

## Metrics Computed

| Group (EvalMetrics) | Description |
|---|---|
| `INDEL` | 8-type structural error taxonomy: 5'/3' extensions, whole insertions, joins, 5'/3' deletions, whole deletions, splits |
| `REGION_DISCOVERY` | 4-level section P/R: neighborhood, internal, full\_coverage, perfect\_boundary — using 1:1 greedy max-overlap matching |
| `BOUNDARY_EXACTNESS` | IoU scores, exact boundary flags, 21×21 bias matrix (signed boundary residuals), 11×11 reliability matrix (cumulative recall at tolerances 0–10 bp) |
| `NUCLEOTIDE_CLASSIFICATION` | Per-base TP/TN/FP/FN → precision, recall, F1 (F1 only computed here) |
| `FRAMESHIFT` | Reading frame deviation mod-3 via sliding window codon intersection; only valid on single-transcript sequences |
| State transitions (always) | GT-transition confusion matrix + false-transition analysis |

## Key Design Decisions

- **Label-agnostic**: `LabelConfig` Pydantic model with arbitrary `dict[int, str]` token mapping; pre-built `BEND_LABEL_CONFIG` for the BEND benchmark label set
- **No PyTorch**: pure NumPy
- **GFF3/GTF input**: Polars LazyFrame parser (`io_utils.py`); strand-aware transcript matching (`transcript_mapping.py`)
- **W&B integration**: optional extra (`pip install dna-segmentation-benchmark[wandb]`)
- **Mask support**: `mask_labels` parameter excludes scaffold/repeat regions from evaluation

## Benchmark Review vs. SOTA (gffcompare, Mikado compare, EGASP, SpliceAI)

### Unique Strengths (not in any competing tool)
1. **INDEL error taxonomy** — 8-type causal diagnosis of mismatch regions; gffcompare class codes only cover transcript-level relationships
2. **Boundary bias matrix** (21×21 signed residuals) — reveals systematic directional boundary errors; no other tool does this
3. **Reliability matrix** (11×11 cumulative recall at tolerances) — shows how quickly recall degrades as boundary tolerance tightens
4. **Frameshift analysis** — reading frame deviation; absent from gffcompare, Mikado, gt eval, EGASP
5. **State transition analysis** — confusion at GT class-change positions and false transitions at stable positions
6. **ML training-loop integration** — W&B adapter, matplotlib figures, numpy array API; competing tools are batch CLI only

### Weaknesses vs. SOTA

| Gap | Competing tool that has it | Impact |
|---|---|---|
| No alternative splicing / isoform support | gffcompare (class codes), Mikado (per-isoform), SQANTI3 (FSM/ISM/NIC/NNC) | Cannot fairly evaluate multi-isoform loci (>60% of human genes) |
| No transcript-level or gene-level metrics | gffcompare, Mikado, EGASP | Cannot assess complete gene reconstruction accuracy |
| F1 not reported at section level | Mikado compare | Harder to rank models |
| GFF parser is fragile (Polars separator trick) | gffcompare (C parser), Mikado (validated Python reader) | May fail on edge-case GFF3 from real-world tools |
| `splice_donor_label` / `splice_acceptor_label` unused | SpliceAI evaluation (AUPRC, top-k) | Splice site evaluation not implemented despite config support |
| Transcript matching uses naive 20% overlap threshold | Mikado (junction F1 maximization), gffcompare (reciprocal overlap + class codes) | Misassignment at complex/overlapping loci |
| No statistical comparison between models | LRGASP community benchmarks | Cannot determine if score differences are significant |
| No CDS vs UTR distinction | EGASP/ENCODE standard | Cannot separate coding exon accuracy from UTR accuracy |

### Comparison Table vs. Key Tools

| Capability | This | gffcompare | Mikado | EGASP | SQANTI3 |
|---|---|---|---|---|---|
| Nucleotide P/R/F1 | Yes | Yes | Yes | Yes | No |
| Section/exon P/R | 4 levels | 1 level | 2 levels | Exact only | Classification |
| Intron chain (gap chain) | **Yes** | Yes | Yes | Yes | No |
| Transcript level | No | Yes | Yes (3 thresholds) | Yes | Classification |
| Gene level | No | Yes | Yes | Yes | No |
| INDEL taxonomy | **8 types** | No | No | No | No |
| Boundary bias matrix | **Yes** | No | No | No | No |
| Frameshift | **Yes** | No | No | No | No |
| State transitions | **Yes** | No | No | No | No |
| Alt splicing | No | Class codes | Per-isoform | Limited | FSM/ISM/NIC/NNC |
| CDS vs UTR | Configurable | No | No | Yes | No |
| W&B / ML integration | **Yes** | No | No | No | No |

### Verdict
The benchmark's primary value is **diagnostic depth at the exon/section level** — particularly for evaluating DL segmentation models in the BEND benchmark ecosystem. The intron chain gap has been closed via the label-agnostic **gap chain** metric. The main remaining credibility gap is the absence of isoform-aware evaluation and transcript-/gene-level metrics.