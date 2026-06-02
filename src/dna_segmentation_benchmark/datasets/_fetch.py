"""Pooch-backed downloader for benchmark datasets.

Each dataset gets its own pooch instance under
``$DNASB_DATA_HOME`` (or platform-default cache via ``appdirs``). Files are
fetched from Hugging Face Hub via SHA-pinned ``resolve/<commit>`` URLs,
verified by sha256, and gunzipped on demand.
"""

from __future__ import annotations

import gzip
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pooch

from ._registry import DatasetSpec, FileSpec


@dataclass(frozen=True)
class LoadedDataset:
    """Resolved on-disk paths for a benchmark dataset."""

    name: str
    fasta: Path
    gtf: Path
    labels: Path | None
    spec: DatasetSpec

    def as_tuple(self) -> tuple[Path, Path]:
        """Convenience: ``(fasta, gtf)`` for tools that take exactly those two."""
        return self.fasta, self.gtf


def _cache_root() -> Path:
    env = os.environ.get("DNASB_DATA_HOME")
    if env:
        return Path(env).expanduser()
    return Path(pooch.os_cache("dna-segmentation-benchmark"))


def _builtin_dir() -> Path:
    from importlib import resources

    return Path(str(resources.files("dna_segmentation_benchmark.datasets") / "builtin"))


class _Gunzip:
    """Pooch processor: decompress ``.gz`` once, return the decompressed path.

    Pooch ships ``pooch.Decompress`` but it keeps the ``.decomp`` suffix; we
    want a clean filename like ``sequence.fa`` so downstream tools (samtools
    faidx, pyfaidx) produce the conventional ``sequence.fa.fai`` index.
    """

    def __init__(self, target_name: str):
        self.target_name = target_name

    def __call__(self, fname: str, action: str, pooch_inst: pooch.Pooch) -> str:
        out = Path(fname).parent / self.target_name
        if action in ("download", "update") or not out.exists():
            with gzip.open(fname, "rb") as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return str(out)


def _build_pooch(spec: DatasetSpec) -> pooch.Pooch:
    base_url = spec.hf_base_url()
    registry: dict[str, str] = {}
    for _role, fs in spec.files():
        registry[fs.remote_path] = f"sha256:{fs.sha256}"
    return pooch.create(
        path=_cache_root() / spec.name,
        base_url=base_url + "/",
        registry=registry,
        env="DNASB_DATA_HOME",
    )


def _resolve_builtin(spec: DatasetSpec, fs: FileSpec) -> Path:
    p = _builtin_dir() / fs.local_name
    if not p.exists():
        raise FileNotFoundError(f"builtin file missing for {spec.name!r}: {p}")
    return p


def fetch(spec: DatasetSpec) -> LoadedDataset:
    """Download (or reuse cached) dataset files and return resolved paths."""
    if spec.builtin:
        labels = _resolve_builtin(spec, spec.labels) if spec.labels else None
        return LoadedDataset(
            name=spec.name,
            fasta=_resolve_builtin(spec, spec.fasta),
            gtf=_resolve_builtin(spec, spec.gtf),
            labels=labels,
            spec=spec,
        )

    pup = _build_pooch(spec)
    paths: dict[str, Path] = {}
    for role, fs in spec.files():
        processor = _Gunzip(fs.local_name) if fs.decompress else None
        local = pup.fetch(fs.remote_path, processor=processor, progressbar=True)
        paths[role] = Path(local)

    return LoadedDataset(
        name=spec.name,
        fasta=paths["fasta"],
        gtf=paths["gtf"],
        labels=paths.get("labels"),
        spec=spec,
    )
