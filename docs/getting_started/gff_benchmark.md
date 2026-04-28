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
    gt_exon_feature_types="exon",
    pred_exon_feature_types={
        "segmentnt": "exon",
        "augustus": "CDS",
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
        "per_transcript": {...},
        "global": {...},
    },
    "augustus": {
        "per_transcript": {...},
        "global": {...},
    },
}
```

`per_transcript` is the same aggregated array benchmark result produced by
{py:func}`dna_segmentation_benchmark.benchmark_gt_vs_pred_multiple`.

`global` covers file-level counts such as how much GT or prediction content was
unmatched.

## Exon Feature Types

The parser feature types are not part of {py:class}`dna_segmentation_benchmark.LabelConfig`.
They are explicit pipeline arguments:

- `gt_exon_feature_types`
- `pred_exon_feature_types`

That matters for tools such as Augustus that emit `CDS` rows instead of
`exon` rows.

## Locus Matching Modes

Use {py:class}`dna_segmentation_benchmark.LocusMatchingMode` to choose how GT
transcripts are paired with predictions inside one locus:

- `FULL_DISCOVERY`: maximize 1:1 matches across the locus
- `BEST_PER_LOCUS`: keep only the best pair per locus

`BEST_PER_LOCUS` is usually the better fit for single-transcript predictors.

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

```bash
dna-benchmark init-config --output label_config.yaml
```

The generated YAML describes label meanings, not parser behavior.

Typical contents:

```yaml
background_label: 8
exon_label: 0
intron_label: 2
splice_donor_label: 1
splice_acceptor_label: 3
```

Required fields:

- `background_label`
- `exon_label`

Optional fields:

- `intron_label`
- `splice_donor_label`
- `splice_acceptor_label`

Keep feature-type choices such as `exon` vs `CDS` in the CLI flags:

- `--gt-exon-feature-type`
- `--pred-exon-feature-type`

### Run the Benchmark

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
  --metrics REGION_DISCOVERY \
  --metrics BOUNDARY_EXACTNESS \
  --metrics STRUCTURAL_COHERENCE \
  --output results.json
```

## Caveats

- If transcript mapping yields no overlapping loci, the pipeline raises an
  error instead of returning empty results.
- `infer_introns=True` affects both GT and prediction arrays before any metric
  family is computed.
- `FRAMESHIFT` remains a transcript-level metric. It is a bad fit for partial
  or unmatched loci.
