"""Dataset registry: parses ``registry.yaml`` into typed specs.

A dataset is a (FASTA, annotation) pair plus optional ground-truth labels,
hosted in a Zenodo record. A published Zenodo record is immutable, so the
``record_id`` pins the exact bytes — reproducible by construction. The package
ships the registry; users only need a local cache + network.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterator

import yaml


@dataclass(frozen=True)
class FileSpec:
    """One downloadable artifact within a dataset."""

    filename: str  # file name within the Zenodo record, e.g. "sequence.fa.gz"
    checksum: str  # digest of the *decompressed* content, "<alg>:<hex>" (e.g. "sha256:...")
    local_name: str  # final on-disk filename after any decompression
    decompress: bool = False  # gunzip after download if True


@dataclass(frozen=True)
class DatasetSpec:
    """A named benchmark dataset hosted in a Zenodo record."""

    name: str
    description: str
    fasta: FileSpec
    annotation: FileSpec
    labels: FileSpec | None = None
    record_id: str | None = None  # Zenodo record id — immutable, pins reproducibility
    sandbox: bool = False  # resolve against sandbox.zenodo.org (for testing uploads)
    assembly: str | None = None
    license: str | None = None
    citation: str | None = None

    def zenodo_base_url(self) -> str:
        if not self.record_id:
            raise ValueError(f"dataset {self.name!r} is missing record_id")
        host = "sandbox.zenodo.org" if self.sandbox else "zenodo.org"
        return f"https://{host}/api/records/{self.record_id}/files"

    def files(self) -> Iterator[tuple[str, FileSpec]]:
        yield "fasta", self.fasta
        yield "annotation", self.annotation
        if self.labels is not None:
            yield "labels", self.labels


def _file_from_dict(d: dict) -> FileSpec:
    filename = d["filename"]
    decompress = bool(d.get("decompress", filename.endswith(".gz")))
    default_local = filename
    if decompress and default_local.endswith(".gz"):
        default_local = default_local[: -len(".gz")]
    return FileSpec(
        filename=filename,
        checksum=d["checksum"],
        local_name=d.get("local_name", default_local),
        decompress=decompress,
    )


def _spec_from_dict(name: str, d: dict) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        description=d.get("description", ""),
        fasta=_file_from_dict(d["fasta"]),
        annotation=_file_from_dict(d["annotation"]),
        labels=_file_from_dict(d["labels"]) if "labels" in d else None,
        record_id=d.get("record_id"),
        sandbox=bool(d.get("sandbox", False)),
        assembly=d.get("assembly"),
        license=d.get("license"),
        citation=d.get("citation"),
    )


def _registry_path() -> Path:
    return Path(str(resources.files("dna_segmentation_benchmark.datasets") / "registry.yaml"))


def load_registry(path: Path | None = None) -> dict[str, DatasetSpec]:
    """Parse the YAML registry into ``{name: DatasetSpec}``."""
    target = path or _registry_path()
    raw = yaml.safe_load(target.read_text()) or {}
    datasets = raw.get("datasets", {})
    return {name: _spec_from_dict(name, d) for name, d in datasets.items()}
