# Annotation Modes

Every benchmark run is anchored to an explicit **annotation mode**. The mode
removes the old ambiguity around what the positive label meant (sometimes exon,
sometimes CDS) and makes metric scope a first-class, self-describing concept.

A {py:class}`gene_calling_benchmark.LabelConfig` always carries an
{py:class}`gene_calling_benchmark.AnnotationMode`. There is no implicit
default — you choose the mode, and the benchmark carries that choice through
validation, evaluation, plotting, logging, and serialized output.

## The two modes

### `EXON_INTRON`

One exonic label representing all exonic sequence of interest, plus optional
intron and splice-site labels.

> *Did the model recover exon structure correctly?*

```python
from gene_calling_benchmark import AnnotationMode, LabelConfig

label_config = LabelConfig(
    annotation_mode=AnnotationMode.EXON_INTRON,
    background_label=8,
    exon_label=0,            # required
    intron_label=2,          # optional
    splice_donor_label=1,    # optional
    splice_acceptor_label=3, # optional
)
```

Good fit for classical exon/intron segmentation and tools that do not
distinguish UTR from CDS.

### `UTR_CDS_INTRON`

Distinct `5' UTR`, `CDS`, and `3' UTR` labels, plus optional intron and
splice-site labels.

> *Did the model recover transcript anatomy, including UTR and CDS structure?*

```python
from gene_calling_benchmark import AnnotationMode, LabelConfig

label_config = LabelConfig(
    annotation_mode=AnnotationMode.UTR_CDS_INTRON,
    background_label=8,
    cds_label=0,             # required
    five_prime_utr_label=4,  # required
    three_prime_utr_label=5, # required
    intron_label=2,          # optional
    splice_donor_label=1,    # optional
    splice_acceptor_label=3, # optional
)
```

The integer tokens are an internal array-encoding detail. When you don't care
about the specific values (typical for GFF/GTF workflows), use the canonical
factory defaults instead of assigning tokens by hand:

```python
exon_config    = LabelConfig.default_exon_intron()     # background/exon/intron/splice
anatomy_config = LabelConfig.default_utr_cds_intron()  # explicit UTR/CDS/intron/splice
```

Good fit for full transcript annotation models, explicit terminus evaluation,
reliable CDS-only metrics such as `PHASE_DRIFT`, and UTR/CDS boundary analysis.

## Validation rules

`LabelConfig` rejects invalid mode/label combinations at construction time:

- all defined label integers must be unique
- `EXON_INTRON` requires `exon_label` and forbids `cds_label`,
  `five_prime_utr_label`, `three_prime_utr_label`
- `UTR_CDS_INTRON` requires `cds_label`, `five_prime_utr_label`,
  `three_prime_utr_label` and forbids `exon_label`
- `splice_donor_label` and `splice_acceptor_label` must both be set or both
  omitted

## Evaluation scopes

A **scope** is the positive-token set a metric operates on:

| Scope | `EXON_INTRON` | `UTR_CDS_INTRON` |
|-------|---------------|------------------|
| `transcript_exon` | `{exon_label}` | `{five_prime_utr, cds, three_prime_utr}` |
| `cds` | not available | `{cds_label}` |

The `evaluation_scope` field selects which scope the **per-transcript** metrics
use:

```python
from gene_calling_benchmark import AnnotationMode, BenchmarkScope, LabelConfig

cds_config = LabelConfig(
    annotation_mode=AnnotationMode.UTR_CDS_INTRON,
    evaluation_scope=BenchmarkScope.CDS,   # per-transcript metrics run on CDS only
    background_label=8,
    cds_label=0,
    five_prime_utr_label=4,
    three_prime_utr_label=5,
    intron_label=2,
)
```

`evaluation_scope` defaults to `transcript_exon`. `BenchmarkScope.CDS` is only
valid in `UTR_CDS_INTRON`.

**Global** (file-level) metrics from the GFF/GTF pipeline ignore
`evaluation_scope` and instead report *every* available scope, so a
`UTR_CDS_INTRON` run emits both `transcript_exon` and `cds` global views in one
pass.

## Metric availability by mode

| Metric family | `EXON_INTRON` | `UTR_CDS_INTRON` |
|---|---|---|
| Region Discovery | yes (`transcript_exon`) | yes (`transcript_exon` or `cds`) |
| Boundary Exactness | yes | yes |
| Nucleotide Classification | yes (binary) | yes (binary per scope) |
| Structural Coherence | yes | yes |
| Diagnostic Depth | yes | yes |
| INDEL | yes | yes |
| Transition Analysis | yes | yes (UTR/CDS transitions become visible) |
| Phase Drift | **no** | yes, with `evaluation_scope=cds` |

`PHASE_DRIFT` raises a clear error outside `UTR_CDS_INTRON` + `cds` scope rather
than silently scoring UTR-contaminated masks.

## Scoped output

The annotation mode is recorded in the result payload, and scope is explicit:

```python
{
    "metadata": {
        "annotation_mode": "UTR_CDS_INTRON",
        "evaluation_scope": "transcript_exon",
    },
    "REGION_DISCOVERY": {...},     # per-transcript: the configured scope
    "STRUCTURAL_COHERENCE": {...},
    ...
}
```

Global metrics nest every scope under a `scopes` key:

```python
results["augustus"]["global"]["nucleotide"]["scopes"]
# {"transcript_exon": {...}, "cds": {...}}
```

## Where the mode flows

The same mode is honoured everywhere:

- **Array benchmark** — {doc}`array_benchmark`
- **GFF/GTF pipeline** — {doc}`gff_benchmark` (feature-role maps translate GFF
  feature names such as `CDS`/`five_prime_UTR` into mode roles)
- **Plotting** — {py:func}`gene_calling_benchmark.compare_multiple_predictions`
  renders one figure per scope and includes the scope in figure keys
- **W&B logging** — scalar/media logging picks the default scope per group
- **CLI** — `gene-benchmark init-config --mode {exon_intron,utr_cds_intron}` and
  `gene-benchmark run` (feature roles via `--gt-feature-role` /
  `--pred-feature-role` in `UTR_CDS_INTRON`)
