# GFF/GTF Benchmark

Use {py:func}`dna_segmentation_benchmark.benchmark_from_gff` when your inputs
are annotation files rather than prebuilt arrays.

## Python API

```python
from dna_segmentation_benchmark import (
    BEND_LABEL_CONFIG,
    EvalMetrics,
    benchmark_from_gff,
    LocusMatchingMode,
)

results = benchmark_from_gff(
    gt_path="ground_truth.gtf",
    pred_paths={
        "segmentnt": "segmentnt.gtf",
        "augustus": "augustus.gtf",
    },
    label_config=BEND_LABEL_CONFIG,
    metrics=[
        EvalMetrics.REGION_DISCOVERY,
        EvalMetrics.BOUNDARY_EXACTNESS,
        EvalMetrics.STRUCTURAL_COHERENCE,
    ],
    # EXON_INTRON's default role map already maps both "exon" and "CDS" to the
    # exon role, so these maps are optional here — shown to illustrate the API.
    pred_feature_role_maps={
        "segmentnt": {"exon": "exon"},
        "augustus": {"CDS": "exon"},
    },
    exclude_features=["gene"],
    locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
    infer_introns=True,
)
```

The pipeline performs four steps:

1. parse GT and prediction files
2. map transcripts within loci
3. build paired label arrays
4. benchmark each predictor

## Result Shape

The pipeline returns one result block per predictor:

```python
{
    "segmentnt": {
        "aggregated": {...},
        "global": {...},
    },
    "augustus": {
        "aggregated": {...},
        "global": {...},
    },
}
```

`aggregated` is the same micro-averaged array benchmark result produced by
{py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_multiple`.

`global` covers file-level counts such as how much GT or prediction content was
unmatched. Global metric families that nest their scopes under a `scopes` key
(`nucleotide`, `exon`, `exon_lenient`, `intron_chain`, `transcript_exact`)
report per-scope views, so a `UTR_CDS_INTRON` run reports both
`transcript_exon` and `cds` views in one pass (the `transcript`, `gene`, and
`locus_isoform` families are unscoped):

```python
results["helixer"]["global"]["nucleotide"]["scopes"]
# {"transcript_exon": {...}, "cds": {...}}
```

## Feature Types and Roles

GFF/GTF feature names are **not** part of {py:class}`dna_segmentation_benchmark.LabelConfig`.
The Python pipeline maps them to benchmark roles through two arguments:
`gt_feature_role_map` (a `{feature_type: role}` map for the ground truth) and
`pred_feature_role_maps` (a flat `{feature_type: role}` map applied to every
predictor, or a nested `{predictor: {feature_type: role}}` map for per-predictor
parsing).

When omitted, mode-specific defaults are used:

- `EXON_INTRON` maps both `exon` and `CDS` to the exon role — so a tool such as
  Augustus that emits `CDS` rows instead of `exon` rows needs no extra
  configuration.
- `UTR_CDS_INTRON` maps `five_prime_UTR`, `CDS`, and `three_prime_UTR` to their
  respective roles.

A single exon type cannot express UTR vs CDS, so override with explicit
feature-role maps when a `UTR_CDS_INTRON` file uses non-default names:

```python
results = benchmark_from_gff(
    gt_path="ground_truth.gtf",
    pred_paths={"helixer": "helixer.gtf"},
    label_config=anatomy_config,  # AnnotationMode.UTR_CDS_INTRON
    gt_feature_role_map={
        "five_prime_UTR": "five_prime_utr",
        "CDS": "cds",
        "three_prime_UTR": "three_prime_utr",
    },
    # pred_feature_role_maps defaults to gt_feature_role_map; pass a dict
    # (or a nested {predictor: {...}} dict) to override per predictor.
)
```

Valid roles are `exon`, `five_prime_utr`, `cds`, `three_prime_utr`, `intron`,
`splice_donor`, and `splice_acceptor`, gated by the active mode and the labels
you configured.

## Locus Matching Modes

Use {py:class}`dna_segmentation_benchmark.LocusMatchingMode` to choose how GT
transcripts are paired with predictions inside one locus:

- `FULL_DISCOVERY`: maximize 1:1 matches across the locus
- `BEST_PER_LOCUS`: keep only the best pair per locus

`BEST_PER_LOCUS` is usually the better fit for single-transcript predictors.

### How `BEST_PER_LOCUS` scores each locus

Every GT locus is scored **exactly once per predictor**, so a predictor is
never rewarded for emitting more isoforms nor penalized for emitting fewer than
the reference. The per-transcript denominator is the full set of GT loci and is
directly comparable across predictors. Each (locus, predictor) pair resolves to
one of three entries:

- **Case A — match.** The predictor's single best-scoring (GT, prediction) pair
  in the locus is kept; all other isoforms and predictions in that locus are
  dropped.
- **Case C — overlap, wrong structure.** The predictor matched nothing in the
  locus but has a transcript overlapping it without a shared junction. That
  prediction is paired against the real GT isoform it best overlaps (junction-F1
  of 0), so a confidently-wrong call is penalized on its coinciding bases rather
  than silently dropped.
- **Case B — clean miss.** The predictor has no transcript overlapping the locus
  at all: a locus-level false negative against the representative (longest)
  isoform, paired with a null prediction.

Because each predictor accumulates independently and Case C only fires when the
predictor matched nothing in the locus, the GT isoform scored in Case C is never
also counted in a Case A entry for that same predictor — no GT base is
double-counted within a predictor's per-transcript numbers.

### Precision caveat: pair the per-transcript and global views

A prediction that overlaps **no** GT locus becomes an intergenic false positive.
But an *extra* prediction that overlaps a locus the predictor **already matched**
(Case A) is **dropped** from the per-transcript view: counting it would force a
second entry for a locus that is only allowed one, and double-count the GT bases
the matched pair already covers.

Those dropped bases are not lost — they are charged by the **global**
`nucleotide` precision, which is a union over all predicted bases and counts
each genomic base once regardless of how many transcripts cover it. As a result
per-transcript precision can read slightly optimistically relative to the global
numbers, so read the two views together. If you want every overlapping isoform
scored on its own, use `FULL_DISCOVERY` instead.

(incomplete-annotations-r-q)=
## Incomplete Annotations (`-R` / `-Q`)

By default every non-overlapping reference and prediction is counted, so the
benchmark behaves like gffcompare with **neither** `-R` nor `-Q`. Two flags
reproduce gffcompare's incomplete-annotation corrections:

- `ignore_novel_predictions` (CLI `-Q` / `--ignore-novel-predictions`) — drop
  predictions that overlap **no** GT transcript. Use when the ground truth is
  incomplete, so predictions in un-annotated regions stop counting as false
  positives (raises **precision**).
- `ignore_missed_reference` (CLI `-R` / `--ignore-missed-reference`) — drop GT
  transcripts that overlap **no** prediction. Use when the prediction set is
  incomplete, so reference genes the predictor never looked at stop counting as
  misses (raises **recall**).

```python
results = benchmark_from_gff(
    gt_path="ground_truth.gff3",
    pred_paths={"helixer": "helixer.gff3"},
    label_config=cfg,
    ignore_novel_predictions=True,   # -Q
    ignore_missed_reference=True,    # -R
)
```

Both default to `False`, so leaving them out reproduces today's numbers exactly.
The correction is orthogonal to the locus-matching mode — it changes which
transcripts enter the denominators, not how overlapping ones are paired — and
applies to **both** the global (annotation-level) metrics and the per-transcript
aggregated micro-metrics.

**What "overlap" means here.** Membership is decided by **transcript-span**
overlap on the same `(seqid, strand)` — any shared base, computed once on the
original files before any pruning. This is deliberately overlap-based, not
match-based: a prediction that overlaps a reference locus but reconstructs it
wrongly is **kept** (it is a real false positive), and only a truly intergenic
prediction is dropped under `-Q`; the reference side is symmetric under `-R`.

:::{note}
gffcompare's overlap test is closer to exon/locus level, whereas this uses the
whole transcript span. The two agree on clearly-novel and clearly-missed
transcripts; span overlap is slightly more lenient about what counts as
"overlapping" in crowded regions. It is a consistent approximation — the same
span notion the mapper already uses to assign predictions to loci — not a
bit-for-bit reproduction of gffcompare's filter.
:::

## Why Results Can Differ From gffcompare

Some metric names are intentionally familiar, especially `intron_chain`, but
the matching workflow is not the same as gffcompare's global transcript
comparison.

This benchmark first builds explicit GT-to-prediction transcript pairs inside
each locus and then computes the per-transcript metrics on those pairs. The
final precision and recall are aggregated from those local transcript-level
decisions.

gffcompare takes a more global view of the annotation set. Its reported matches
depend on the full reference/query transcript collections and on how competing
transcripts across a locus are resolved at the dataset level.

That means two things:

- the same prediction can be paired differently here than in gffcompare,
  especially in multi-isoform loci
- metrics with the same name can differ numerically because they are being
  aggregated from different match assignments

In practice:

- if you want a benchmark that reflects the model's quality on one matched
  transcript pair at a time, this benchmark is the intended view
- if you want the exact behavior of gffcompare's global transcript-resolution
  logic, use gffcompare as the reference tool for that comparison

The difference is usually smallest on simple one-to-one loci and largest on
crowded loci with several similar isoforms.

## CLI

The same workflow is available through `dna-benchmark`.

### Create a Config Template

Pick the annotation mode (see {doc}`annotation_modes`):

```bash
dna-benchmark init-config --mode exon_intron     --output label_config.yaml
dna-benchmark init-config --mode utr_cds_intron  --output anatomy_config.yaml
```

The generated YAML describes label meanings, not parser behavior.

`EXON_INTRON` template:

```yaml
annotation_mode: EXON_INTRON
background_label: 8
exon_label: 0
intron_label: 2
splice_donor_label: 1
splice_acceptor_label: 3
```

`UTR_CDS_INTRON` template:

```yaml
annotation_mode: UTR_CDS_INTRON
background_label: 8
cds_label: 0
five_prime_utr_label: 4
three_prime_utr_label: 5
intron_label: 2
splice_donor_label: 1
splice_acceptor_label: 3
```

### Mapping GFF/GTF feature names

The label config never names GFF/GTF feature types — that mapping lives in CLI
flags so the same labels can be reused across files with different conventions.

In `EXON_INTRON` mode, keep the simple exon-type flags:

- `--gt-exon-feature-type` (e.g. `exon`)
- `--pred-exon-feature-type` (e.g. `segmentnt:exon`, `augustus:CDS`)

In `UTR_CDS_INTRON` mode a single exon type cannot express UTR vs CDS, so use
**feature-role maps** instead. Each value is `feature_type:role`:

- `--gt-feature-role` — e.g. `--gt-feature-role CDS:cds`
- `--pred-feature-role` — plain `feature_type:role` for all predictors, or
  `predictor=feature_type:role` per predictor (e.g. `helixer=CDS:cds`)

Valid roles are `exon`, `five_prime_utr`, `cds`, `three_prime_utr`, `intron`,
`splice_donor`, `splice_acceptor` (subject to the mode and the labels you
configured). When no role flags are given in `UTR_CDS_INTRON` mode, a sensible
default map (`five_prime_UTR`, `CDS`, `three_prime_UTR`) is used.

### Run the Benchmark

`EXON_INTRON` (legacy exon-type flags):

```bash
dna-benchmark run \
  --gt ground_truth.gtf \
  --pred segmentnt:segmentnt.gtf \
  --pred augustus:augustus.gtf \
  --config label_config.yaml \
  --gt-exon-feature-type exon \
  --pred-exon-feature-type segmentnt:exon \
  --pred-exon-feature-type augustus:CDS \
  --exclude-features gene \
  --locus-matching best_per_locus \
  --infer-introns \
  --ignore-novel-predictions \
  --ignore-missed-reference \
  --metrics REGION_DISCOVERY \
  --metrics BOUNDARY_EXACTNESS \
  --metrics STRUCTURAL_COHERENCE \
  --output results.json
```

`UTR_CDS_INTRON` (feature-role maps):

```bash
dna-benchmark run \
  --gt ground_truth.gtf \
  --pred helixer:helixer.gtf \
  --config anatomy_config.yaml \
  --gt-feature-role five_prime_UTR:five_prime_utr \
  --gt-feature-role CDS:cds \
  --gt-feature-role three_prime_UTR:three_prime_utr \
  --exclude-features gene \
  --metrics REGION_DISCOVERY \
  --metrics BOUNDARY_EXACTNESS \
  --metrics NUCLEOTIDE_CLASSIFICATION \
  --output results.json
```

## Caveats

- If transcript mapping yields no overlapping loci, the pipeline raises an
  error instead of returning empty results.
- `infer_introns=True` affects both GT and prediction arrays before any metric
  family is computed.
- `-R` / `-Q` use transcript-span overlap, not gffcompare's exon-level overlap;
  see [Incomplete Annotations](#incomplete-annotations-r-q). Both default off, so
  results are unchanged unless you pass them.
- `PHASE_DRIFT` remains a transcript-level metric. It is a bad fit for partial
  or unmatched loci.
