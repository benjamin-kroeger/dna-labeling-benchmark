"""Downloader for benchmark datasets hosted on Zenodo.

Each dataset is cached under ``$DNASB_DATA_HOME`` (or the platform-default cache
via ``appdirs``) and fetched from a Zenodo record's immutable file-content URLs.
Integrity is checked against the **decompressed content** digest recorded in the
registry: gzip is non-deterministic, so a compressed blob's digest is not a
stable identity for the biological data.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pooch

from ._registry import DatasetSpec


@dataclass(frozen=True)
class LoadedDataset:
    """Resolved on-disk paths for a benchmark dataset."""

    name: str
    fasta: Path
    annotation: Path
    labels: Path | None
    spec: DatasetSpec

    def as_tuple(self) -> tuple[Path, Path]:
        """Convenience: ``(fasta, annotation)`` for tools that take exactly those two."""
        return self.fasta, self.annotation


def _cache_root() -> Path:
    env = os.environ.get("DNASB_DATA_HOME")
    if env:
        return Path(env).expanduser()
    return Path(pooch.os_cache("dna-segmentation-benchmark"))


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


def _hash_file(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(path: Path, checksum: str) -> None:
    """Verify *path* against a ``"<algorithm>:<hex>"`` checksum string."""
    if not checksum:
        return  # no checksum recorded (e.g. local sandbox testing)
    algorithm, _, expected = checksum.partition(":")
    if not expected:  # bare digest → assume sha256
        algorithm, expected = "sha256", checksum
    actual = _hash_file(path, algorithm)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"checksum mismatch for {path.name}: expected {algorithm}:{expected}, got {algorithm}:{actual}"
        )


def fetch(spec: DatasetSpec) -> LoadedDataset:
    """Download (or reuse cached) dataset files and return resolved paths."""
    base_url = spec.zenodo_base_url()
    cache = _cache_root() / spec.name
    paths: dict[str, Path] = {}
    for role, fs in spec.files():
        url = f"{base_url}/{fs.filename}/content"
        if fs.decompress:
            # gzip is non-deterministic, so the recorded digest is of the
            # decompressed content: let pooch fetch the blob unverified, then
            # check the decompressed file ourselves.
            local = Path(
                pooch.retrieve(
                    url=url,
                    known_hash=None,
                    fname=fs.filename,
                    path=cache,
                    processor=_Gunzip(fs.local_name),
                    progressbar=True,
                )
            )
            _verify_checksum(local, fs.checksum)
        else:
            # Stored uncompressed: the downloaded bytes are the content, so the
            # recorded digest is pooch's known_hash directly.
            local = Path(
                pooch.retrieve(
                    url=url,
                    known_hash=fs.checksum or None,
                    fname=fs.local_name,
                    path=cache,
                    progressbar=True,
                )
            )
        paths[role] = local

    return LoadedDataset(
        name=spec.name,
        fasta=paths["fasta"],
        annotation=paths["annotation"],
        labels=paths.get("labels"),
        spec=spec,
    )
