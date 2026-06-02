"""Dataset registry: parses ``registry.yaml`` into typed specs.

A dataset is a (FASTA, GTF) pair plus optional ground-truth labels, hosted on a
Hugging Face Hub *dataset* repo and pinned to a specific commit SHA. The
package ships the registry; users only need a local cache + network.
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

    remote_path: str  # path inside the HF repo, e.g. "chr22/sequence.fa.gz"
    sha256: str  # checksum of the *remote* (possibly compressed) file
    local_name: str  # final on-disk filename after any decompression
    decompress: bool = False  # gunzip after download if True


@dataclass(frozen=True)
class DatasetSpec:
    """A named benchmark dataset.

    ``builtin=True`` skips network fetching and resolves files from the
    ``builtin/`` directory inside the package — used for tiny CI / tutorial
    datasets.
    """

    name: str
    description: str
    fasta: FileSpec
    gtf: FileSpec
    labels: FileSpec | None = None
    repo_id: str | None = None  # e.g. "benjaminkroeger/dnasb-data"
    revision: str | None = None  # commit SHA — pin for reproducibility
    assembly: str | None = None
    license: str | None = None
    citation: str | None = None
    builtin: bool = False

    def hf_base_url(self) -> str:
        if self.builtin:
            raise ValueError(f"dataset {self.name!r} is builtin and has no remote URL")
        if not self.repo_id or not self.revision:
            raise ValueError(f"dataset {self.name!r} is missing repo_id or revision")
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/{self.revision}"

    def files(self) -> Iterator[tuple[str, FileSpec]]:
        yield "fasta", self.fasta
        yield "gtf", self.gtf
        if self.labels is not None:
            yield "labels", self.labels


def _file_from_dict(d: dict) -> FileSpec:
    remote = d["remote_path"]
    decompress = bool(d.get("decompress", remote.endswith(".gz")))
    default_local = remote.rsplit("/", 1)[-1]
    if decompress and default_local.endswith(".gz"):
        default_local = default_local[: -len(".gz")]
    return FileSpec(
        remote_path=remote,
        sha256=d["sha256"],
        local_name=d.get("local_name", default_local),
        decompress=decompress,
    )


def _spec_from_dict(name: str, d: dict) -> DatasetSpec:
    builtin = bool(d.get("builtin", False))
    return DatasetSpec(
        name=name,
        description=d.get("description", ""),
        fasta=_file_from_dict(d["fasta"]),
        gtf=_file_from_dict(d["gtf"]),
        labels=_file_from_dict(d["labels"]) if "labels" in d else None,
        repo_id=d.get("repo_id"),
        revision=d.get("revision"),
        assembly=d.get("assembly"),
        license=d.get("license"),
        citation=d.get("citation"),
        builtin=builtin,
    )


def _registry_path() -> Path:
    return Path(str(resources.files("dna_segmentation_benchmark.datasets") / "registry.yaml"))


def load_registry(path: Path | None = None) -> dict[str, DatasetSpec]:
    """Parse the YAML registry into ``{name: DatasetSpec}``."""
    target = path or _registry_path()
    raw = yaml.safe_load(target.read_text()) or {}
    datasets = raw.get("datasets", {})
    return {name: _spec_from_dict(name, d) for name, d in datasets.items()}
