"""Tests for the dataset loader against the ``zenodo_test`` sandbox entry.

Registry parsing, lookup, spec fields, and URL construction run offline. The
actual download round-trip is marked ``network`` and skips gracefully when
``sandbox.zenodo.org`` is unreachable.
"""

from pathlib import Path

import pytest

from dna_segmentation_benchmark.datasets import (
    DatasetSpec,
    LoadedDataset,
    get_dataset_info,
    list_datasets,
    load_dataset,
)
from dna_segmentation_benchmark.datasets._fetch import _hash_file, _verify_checksum
from dna_segmentation_benchmark.datasets._registry import FileSpec, load_registry

requests = pytest.importorskip("requests")  # transitive dep via pooch


def test_registry_parses():
    registry = load_registry()
    assert "zenodo_test" in registry
    assert isinstance(registry["zenodo_test"], DatasetSpec)


def test_list_datasets_includes_zenodo_test():
    assert "zenodo_test" in list_datasets()


def test_get_dataset_info_unknown():
    with pytest.raises(KeyError):
        get_dataset_info("does_not_exist")


def test_zenodo_test_spec_is_sandbox_remote():
    spec = get_dataset_info("zenodo_test")
    assert spec.sandbox is True
    assert spec.record_id == "512849"
    assert spec.fasta.filename == "zenodo_test.fasta"
    assert spec.annotation.filename == "zenodo_test.gff3"
    assert spec.fasta.checksum.startswith("md5:")
    assert spec.zenodo_base_url() == "https://sandbox.zenodo.org/api/records/512849/files"


@pytest.mark.network
def test_load_zenodo_test_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("DNASB_DATA_HOME", str(tmp_path))
    try:
        ds = load_dataset("zenodo_test")
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"sandbox.zenodo.org unreachable: {exc}")
    assert isinstance(ds, LoadedDataset)
    assert ds.fasta.exists() and ds.fasta.suffix == ".fasta"
    assert ds.annotation.exists() and ds.annotation.suffix == ".gff3"
    assert ds.labels is None
    fa, annotation = ds.as_tuple()
    assert isinstance(fa, Path) and isinstance(annotation, Path)
    assert ds.fasta.read_text().startswith(">")
    assert ds.annotation.read_text().startswith("##gff-version")


def test_remote_spec_url_construction():
    spec = DatasetSpec(
        name="x",
        description="",
        fasta=FileSpec(
            filename="sequence.fa.gz", checksum="md5:00", local_name="sequence.fa", decompress=True
        ),
        annotation=FileSpec(
            filename="annotation.gff3.gz", checksum="md5:00", local_name="annotation.gff3", decompress=True
        ),
        record_id="123456",
    )
    assert spec.zenodo_base_url() == "https://zenodo.org/api/records/123456/files"


def test_sandbox_url_construction():
    spec = DatasetSpec(
        name="x",
        description="",
        fasta=FileSpec(filename="sequence.fa.gz", checksum="md5:00", local_name="sequence.fa", decompress=True),
        annotation=FileSpec(filename="annotation.gff3.gz", checksum="md5:00", local_name="annotation.gff3", decompress=True),
        record_id="123456",
        sandbox=True,
    )
    assert spec.zenodo_base_url() == "https://sandbox.zenodo.org/api/records/123456/files"


def test_missing_record_id_rejects_url():
    spec = DatasetSpec(
        name="b",
        description="",
        fasta=FileSpec(filename="x.fa", checksum="", local_name="x.fa"),
        annotation=FileSpec(filename="x.gtf", checksum="", local_name="x.gtf"),
    )
    with pytest.raises(ValueError):
        spec.zenodo_base_url()


def test_verify_checksum_roundtrip(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"ACGT\n")
    digest = _hash_file(p, "sha256")
    _verify_checksum(p, f"sha256:{digest}")  # correct digest -> no raise
    _verify_checksum(p, "")  # empty checksum -> skipped, no raise


def test_verify_checksum_mismatch(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"ACGT\n")
    with pytest.raises(ValueError):
        _verify_checksum(p, "sha256:" + "0" * 64)
