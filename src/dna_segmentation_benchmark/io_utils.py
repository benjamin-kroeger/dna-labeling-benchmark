"""I/O utilities for DNA segmentation benchmark.

Handles reading of GFF3 and GTF annotations via PyRanges. Format is
detected from the file extension (.gtf → GTF, everything else → GFF3).

Returned DataFrames use **1-based, inclusive** coordinates (GFF convention)
with unified ``gff_id`` and ``parent`` columns that work for both formats:

- GTF: both columns are set to ``transcript_id``
- GFF3: ``gff_id`` ← ``ID``, ``parent`` ← ``Parent``

.. note::

   Arrays are always in **genomic (left-to-right) orientation**, regardless
   of strand.  A minus-strand gene at positions 200-400 is stored with
   index 0 = position 200.  Consuming code must be aware of this convention.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges1 as pr

from .label_definition import LabelConfig

logger = logging.getLogger(__name__)

# Default GFF feature types that denote a "transcript" row.
DEFAULT_TRANSCRIPT_TYPES: list[str] = ["mRNA", "transcript"]


# ------------------------------------------------------------------
# Internal: GFF parsing
# ------------------------------------------------------------------


def _detect_format(path: Path) -> str:
    """Return ``'gtf'`` or ``'gff3'`` based on file extension."""
    suffixes = {s.lower() for s in path.suffixes}
    return "gtf" if ".gtf" in suffixes else "gff3"


def _normalise_pyranges_df(gr_df: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """Convert a PyRanges DataFrame to the internal normalised format.

    PyRanges uses 0-based half-open coordinates; this function converts
    them back to 1-based inclusive (GFF convention) and renames columns
    to the internal schema.

    Returns columns: ``seqid``, ``type``, ``start``, ``end``, ``strand``,
    ``gff_id``, ``parent``.
    """
    df = gr_df.rename(
        columns={
            "Chromosome": "seqid",
            "Feature": "type",
            "Strand": "strand",
        }
    )

    # PyRanges 0-based half-open → 1-based inclusive
    df["start"] = df["Start"] + 1
    df["end"] = df["End"]

    if fmt == "gtf":
        tid = df["transcript_id"] if "transcript_id" in df.columns else pd.NA
        df["gff_id"] = tid
        df["parent"] = tid
    else:
        df["gff_id"] = df["ID"] if "ID" in df.columns else pd.NA
        if "Parent" in df.columns:
            # Parent may be comma-separated; take the first value only.
            df["parent"] = df["Parent"].str.split(",").str[0]
        else:
            df["parent"] = pd.NA

    keep = ["seqid", "type", "start", "end", "strand", "gff_id", "parent"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


# ------------------------------------------------------------------
# Public: collect a full GFF into memory
# ------------------------------------------------------------------


def collect_gff(
    gff_path: str | Path,
    exclude_features: list[str] | None = None,
) -> pd.DataFrame:
    """Read and collect a GFF/GTF file into a materialised DataFrame.

    This is the canonical way to read a GFF file once and reuse the
    result for multiple transcript look-ups, avoiding repeated disk I/O.

    Parameters
    ----------
    gff_path : str | Path
        Path to the GFF3 or GTF file.
    exclude_features : list[str] | None
        Feature types to exclude (e.g. ``["gene"]``).

    Returns
    -------
    pd.DataFrame
        Columns: ``seqid``, ``type``, ``start``, ``end``, ``strand``,
        ``gff_id``, ``parent``.  Coordinates are 1-based, inclusive.
    """
    path = Path(gff_path)
    fmt = _detect_format(path)

    try:
        if fmt == "gtf":
            gr = pr.read_gtf(str(path))
        else:
            gr = pr.read_gff3(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {gff_path}: {exc}") from exc

    df = _normalise_pyranges_df(pd.DataFrame(gr), fmt)

    if exclude_features:
        df = df[~df["type"].isin(exclude_features)].reset_index(drop=True)

    null_coord_count = df[["start", "end"]].isna().any(axis=1).sum()
    if null_coord_count > 0:
        logger.warning(
            "%d row(s) in %s have unparseable coordinates and will be ignored.",
            null_coord_count,
            gff_path,
        )

    return df


# ------------------------------------------------------------------
# Public: per-transcript array construction
# ------------------------------------------------------------------


def read_gff_to_arrays(
    gff_path: str | Path,
    label_config: LabelConfig,
    exclude_features: list[str] | None = None,
    transcript_types: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Parse a GFF/GTF file and create **one array per transcript**.

    Each transcript (``mRNA`` / ``transcript``) defines a single
    evaluation unit.  Child features (``exon``, ``CDS``, etc.) are
    mapped into local coordinates within that transcript's span.

    .. note::

       All child features are painted with ``coding_label`` regardless
       of their actual GFF type (CDS, exon, UTR, etc.).  This produces
       a binary coding / non-coding mask, not a multi-label array.

    Parameters
    ----------
    gff_path : str | Path
        Path to the GFF3 or GTF file.
    label_config : LabelConfig
        Label configuration defining the integer tokens.
    exclude_features : list[str] | None
        GFF feature types to completely ignore (e.g. ``['gene']``).
    transcript_types : list[str] | None
        Feature types that define transcript boundaries.
        Defaults to :data:`DEFAULT_TRANSCRIPT_TYPES`.

    Returns
    -------
    dict[str, np.ndarray]
        ``{transcript_id}_{strand}`` → 1-D ``int32`` annotation array.

    Raises
    ------
    ValueError
        If ``coding_label`` is not defined in *label_config*.
    RuntimeError
        If the GFF file cannot be parsed.
    """
    if label_config.coding_label is None:
        raise ValueError("LabelConfig must have `coding_label` set to parse GFF features.")

    coding_val = label_config.coding_label
    bg_val = label_config.background_label
    transcript_types = transcript_types or DEFAULT_TRANSCRIPT_TYPES

    try:
        df = collect_gff(str(gff_path), exclude_features=exclude_features)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {gff_path}: {exc}") from exc

    transcript_df = df[df["type"].isin(transcript_types)]
    child_df = df[df["parent"].notna() & ~df["type"].isin(transcript_types)]

    if transcript_df.empty:
        logger.warning(
            "No transcript features (%s) found in %s. Returning empty result.",
            transcript_types,
            gff_path,
        )
        return {}

    annotations: dict[str, np.ndarray] = {}
    transcript_lookup: dict[str, tuple[int, str, np.ndarray]] = {}

    for row in transcript_df.itertuples(index=False):
        t_id = row.gff_id
        t_start = row.start
        t_end = row.end
        strand = row.strand

        if pd.isna(t_id) or pd.isna(t_start) or pd.isna(t_end):
            logger.debug(
                "Skipping transcript with missing field(s): id=%s, start=%s, end=%s",
                t_id,
                t_start,
                t_end,
            )
            continue

        t_start = int(t_start)
        t_end = int(t_end)
        array_len = t_end - t_start + 1  # GFF is 1-based inclusive
        key = f"{t_id}_{strand}"
        arr = np.full(array_len, bg_val, dtype=np.int32)
        annotations[key] = arr
        transcript_lookup[str(t_id)] = (t_start, strand, arr)

    for row in child_df.itertuples(index=False):
        parent_id = row.parent
        c_start = row.start
        c_end = row.end

        if pd.isna(parent_id) or pd.isna(c_start) or pd.isna(c_end):
            continue

        parent_id = str(parent_id)
        if parent_id not in transcript_lookup:
            continue

        t_start, _, arr = transcript_lookup[parent_id]
        c_start = int(c_start)
        c_end = int(c_end)

        local_start = max(0, c_start - t_start)
        local_end = min(len(arr), c_end - t_start + 1)

        if c_start < t_start or c_end > t_start + len(arr) - 1:
            logger.debug(
                "Child feature of %s extends outside transcript boundaries "
                "(child: %d-%d, transcript: %d-%d). Clipping to transcript span.",
                parent_id,
                c_start,
                c_end,
                t_start,
                t_start + len(arr) - 1,
            )

        if local_start < local_end:
            arr[local_start:local_end] = coding_val

    return annotations
