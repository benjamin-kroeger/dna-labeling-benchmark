# Global (Annotation-Level) Metrics

The **global** family lives under the top-level `"global"` key of a GFF-benchmark
result (`results[method]["global"]`) and is the direct counterpart to
[gffcompare](https://ccb.jhu.edu/software/stringtie/gffcompare.shtml)'s summary.
Unlike the per-transcript metric families — which score matched (GT, prediction)
pairs in isolation — the global family aggregates over **all** transcripts,
including unmatched ones, so false-positive predictions and missed reference
transcripts both move the numbers. Each group reports counts plus derived
precision / recall / F1.

```{note}
This `"global"` → `"nucleotide"` key is **not** the same as the
[Nucleotide Classification](nucleotide_classification.md) family's `"nucleotide"`
output. The global one is a **union-based, genome-level** base rate (each genomic
base counted once across all isoforms, gffcompare-style). The per-transcript
`NUCLEOTIDE_CLASSIFICATION` one is computed per matched transcript and averaged.
Same word, different scope — read the enclosing key to tell them apart.
```

## The eight groups

| Key | Question it answers |
|-----|---------------------|
| `nucleotide` | Base-level exon-coverage accuracy (union over isoforms; gffcompare nucleotide Sn/Sp). |
| `exon` | How many exons are reconstructed exactly. Each unique `(seqid, strand, start, end)` interval is counted **once across all transcripts** (this diverges from gffcompare's per-isoform counting). |
| `exon_lenient` | Same as `exon` but terminal-exon **outer** boundaries need not match (only internal splice sites must be exact) — gffcompare's default `=` leniency. |
| `intron_chain` | How many multi-exon intron chains match exactly (gffcompare intron-chain Sn/Sp), by coordinate-exact matching over the full reference. |
| `transcript_exact` | How many transcripts match exactly, coordinate-exact (gffcompare transcript Sn/Sp), over the full reference. |
| `transcript` | Transcript recovery from the assignment step: Sn = matched ref / total ref, Sp = matched pred / total pred (mode-dependent; retained for backward compatibility). |
| `gene` | How many gene loci are detected. Transcripts are clustered into loci by overlap; a locus counts as matched if **any** of its transcripts got a counterpart. |
| `locus_isoform` | How completely multi-isoform loci are recovered — the fraction of each GT locus's isoforms that received a match. Most meaningful with `FULL_DISCOVERY`. |

## Scoped vs unscoped

The families that depend on which label roles count as "exon"
(`nucleotide`, `exon`, `exon_lenient`, `intron_chain`, `transcript_exact`) nest
their results under a `"scopes"` key, so a `UTR_CDS_INTRON` run reports both a
`transcript_exon` and a `cds` view in one pass:

```python
results[method]["global"]["nucleotide"]["scopes"]
# {"transcript_exon": {...}, "cds": {...}}
```

The remaining families (`transcript`, `gene`, `locus_isoform`) are unscoped.

`intron_chain` and `transcript_exact` reproduce gffcompare's numbers by
coordinate-exact structure matching over the whole reference, **independent** of
the `--locus-matching` mode. See the [GFF benchmark guide](../getting_started/gff_benchmark.md)
for how `-R` / `-Q` (incomplete-annotation corrections) interact with these.
