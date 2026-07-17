"""Benchmark dataset loader (seaborn-style).

Examples
--------
>>> from gene_calling_benchmark.datasets import load_dataset, list_datasets
>>> list_datasets()
['zenodo_test']
>>> ds = load_dataset("zenodo_test")
>>> ds.fasta, ds.annotation  # doctest: +SKIP
(PosixPath('.../zenodo_test.fasta'), PosixPath('.../zenodo_test.gff3'))

Datasets are pinned to an immutable Zenodo record id in ``registry.yaml``;
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
