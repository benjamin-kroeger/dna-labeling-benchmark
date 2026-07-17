# Datasets

Benchmark datasets (a reference **FASTA** + a **GFF3** annotation) are hosted on
[Zenodo](https://zenodo.org) and downloaded on demand. Each dataset is pinned to an immutable Zenodo *record id*,
so the bytes you get are reproducible, and every file is verified against its
recorded **decompressed-content** digest after download. Files are cached under
`$DNASB_DATA_HOME` (default: your platform user cache) and reused on subsequent
calls.

The annotation is shipped as **GFF3** (not GTF) so it carries distinct
`five_prime_UTR` / `three_prime_UTR` features and can drive either annotation
mode — `EXON_INTRON` or `UTR_CDS_INTRON`. See {doc}`annotation_modes`.

## Python API

```python
from gene_calling_benchmark import list_datasets, get_dataset_info, load_dataset

list_datasets()                   # -> ['zenodo_test']
get_dataset_info("zenodo_test")   # DatasetSpec, no download

ds = load_dataset("zenodo_test")  # downloads + verifies on first use
ds.fasta        # PosixPath('.../zenodo_test.fasta')  — reference sequence
ds.annotation   # PosixPath('.../zenodo_test.gff3')   — ground-truth annotation
```

`ds.fasta` is the sequence you run your own gene caller on; it is **not** consumed
by the benchmark itself. The benchmark compares your prediction GFF/GTF against
`ds.annotation` (the ground truth):

```python
from gene_calling_benchmark import benchmark_from_gff, LabelConfig, AnnotationMode

results = benchmark_from_gff(
    gt_path=ds.annotation,
    pred_paths={"augustus": "augustus.gff3"},
    label_config=LabelConfig(annotation_mode=AnnotationMode.EXON_INTRON, ...),
)
```

## Command line

```bash
gene-benchmark datasets list               # names + one-line descriptions
gene-benchmark datasets info zenodo_test   # record id, license, files (no download)
gene-benchmark datasets get  zenodo_test   # download + print cached paths

# Use a registry dataset directly as ground truth (downloaded on demand):
gene-benchmark run \
    --dataset zenodo_test \
    --pred augustus:augustus.gff3 \
    --config label_config.yaml
```

`--dataset NAME` is mutually exclusive with `--gt PATH`: pass one or the other.

## Cache location

Downloads land in `$DNASB_DATA_HOME` if set, otherwise the platform user cache
(e.g. `~/.cache/gene-calling-benchmark` on Linux). Delete that directory to
force a re-download.
