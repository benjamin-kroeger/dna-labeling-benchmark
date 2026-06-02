"""Benchmark dataset loader (seaborn-style).

Examples
--------
>>> from dna_segmentation_benchmark.datasets import load_dataset, list_datasets
>>> list_datasets()
['toy_ecoli', 'gencode_v49_chr22']
>>> ds = load_dataset("toy_ecoli")
>>> ds.fasta, ds.gtf  # doctest: +SKIP
(PosixPath('.../sequence.fa'), PosixPath('.../annotation.gtf'))

Datasets are pinned to a Hugging Face Hub commit SHA in ``registry.yaml``;
files are downloaded on first use, sha256-verified, and cached under
``$DNASB_DATA_HOME`` (default: platform user cache).
"""

from __future__ import annotations

from ._fetch import LoadedDataset, fetch
from ._registry import DatasetSpec, load_registry

__all__ = [
    "DatasetSpec",
    "LoadedDataset",
    "load_dataset",
    "list_datasets",
    "get_dataset_info",
]


def list_datasets() -> list[str]:
    """Names of all datasets in the registry, in declaration order."""
    return list(load_registry().keys())


def get_dataset_info(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``name`` without downloading."""
    registry = load_registry()
    if name not in registry:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(registry)}")
    return registry[name]


def load_dataset(name: str) -> LoadedDataset:
    """Fetch (or reuse cached) dataset and return resolved file paths."""
    return fetch(get_dataset_info(name))
