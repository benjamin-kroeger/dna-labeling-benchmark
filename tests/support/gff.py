"""GFF3 helpers shared across the test suite.

* ``write_gff`` — write a list of ready-made GFF3 rows to a file (the one-line
  boilerplate that several tests used to hand-roll).
* ``UTR_ROLE_MAP`` / ``UTR_ROLE_MAP_NO_CDS`` — the feature→role maps repeated by
  the UTR/CDS tests.
* The DataFrame (de)serialisation + perturbation helpers used to derive
  known-answer predictions by mutating the ground truth (controlled mutants).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dna_segmentation_benchmark.io_utils import DEFAULT_TRANSCRIPT_TYPES

# Feature→role maps for UTR_CDS_INTRON tests. The prediction side commonly has no
# CDS feature (its UTRs abut directly), hence the CDS-free variant.
UTR_ROLE_MAP = {
    "five_prime_UTR": "five_prime_utr",
    "CDS": "cds",
    "three_prime_UTR": "three_prime_utr",
}
UTR_ROLE_MAP_NO_CDS = {
    "five_prime_UTR": "five_prime_utr",
    "three_prime_UTR": "three_prime_utr",
}


def write_gff(path, rows: list[str]) -> str:
    """Write GFF3 ``rows`` (tab-joined feature lines) under a ``##gff-version 3``
    header and return the path as a string."""
    Path(path).write_text("##gff-version 3\n" + "\n".join(rows) + "\n")
    return str(path)


def write_gff3(df: pd.DataFrame, path, transcript_types: list[str] | None = None) -> str:
    """Serialise a normalised ``collect_gff`` DataFrame back to a GFF3 file.

    Reconstructs transcript rows (``ID=<gff_id>``) and their child features
    (``Parent=<transcript gff_id>``).  Gene rows (no parent) are dropped; the
    pipeline only needs transcripts and their children.
    """
    transcript_types = transcript_types or list(DEFAULT_TRANSCRIPT_TYPES)
    lines = ["##gff-version 3"]

    for row in df.itertuples(index=False):
        if row.type in transcript_types:
            lines.append(
                "\t".join(
                    [str(row.seqid), "test", str(row.type), str(int(row.start)),
                     str(int(row.end)), ".", str(row.strand), ".", f"ID={row.gff_id}"]
                )
            )

    counters: dict[tuple, int] = {}
    for row in df.itertuples(index=False):
        if row.type in transcript_types or pd.isna(row.parent):
            continue
        idx = counters.get((row.parent, row.type), 0)
        counters[(row.parent, row.type)] = idx + 1
        lines.append(
            "\t".join(
                [str(row.seqid), "test", str(row.type), str(int(row.start)),
                 str(int(row.end)), ".", str(row.strand), ".",
                 f"ID={row.parent}.{row.type}.{idx};Parent={row.parent}"]
            )
        )

    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


def transcript_subset(df: pd.DataFrame, transcript_id: str) -> pd.DataFrame:
    """Return the transcript row + child rows for one transcript."""
    is_tx = df["type"].isin(DEFAULT_TRANSCRIPT_TYPES) & (df["gff_id"] == transcript_id)
    is_child = ~df["type"].isin(DEFAULT_TRANSCRIPT_TYPES) & (df["parent"] == transcript_id)
    return df[is_tx | is_child].copy()


def drop_internal_exon(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the middle exon (and any CDS inside it) — simulates exon skipping."""
    exons = df[df["type"] == "exon"].sort_values("start")
    if len(exons) < 3:
        raise ValueError("drop_internal_exon needs a transcript with >= 3 exons.")
    middle = exons.iloc[len(exons) // 2]
    inside_cds = (
        (df["type"] == "CDS")
        & (df["start"] >= middle.start)
        & (df["end"] <= middle.end)
    )
    return df[~((df.index == middle.name) | inside_cds)].copy()


def extend_first_cds(df: pd.DataFrame, delta: int = 1) -> pd.DataFrame:
    """Extend the first CDS by ``delta`` bp — simulates a small CDS indel."""
    df = df.copy()
    first_cds = df[df["type"] == "CDS"].sort_values("start").index[0]
    df.loc[first_cds, "end"] = int(df.loc[first_cds, "end"]) + delta
    return df


def merge_first_two_exons(df: pd.DataFrame) -> pd.DataFrame:
    """Fuse the first two exons into one — simulates intron retention."""
    df = df.copy()
    exons = df[df["type"] == "exon"].sort_values("start")
    if len(exons) < 2:
        raise ValueError("merge_first_two_exons needs a transcript with >= 2 exons.")
    first, second = exons.iloc[0], exons.iloc[1]
    df.loc[first.name, "end"] = int(second.end)
    return df.drop(index=second.name)
