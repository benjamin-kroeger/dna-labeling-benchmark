"""I/O utilities for DNA segmentation benchmark.

Handles reading of GFF3 and GTF annotations via Polars.  All GFF parsing
is done through :func:`_lazy_read_gff` which recognises both GFF3
(``ID=``/``Parent=``) and GTF (``transcript_id``/``gene_id``) attribute
conventions.

.. note::

   Arrays are always in **genomic (left-to-right) orientation**, regardless
   of strand.  A minus-strand gene at positions 200-400 is stored with
   index 0 = position 200.  Consuming code must be aware of this convention.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from .label_definition import LabelConfig

logger = logging.getLogger(__name__)

# Default GFF feature types that denote a "transcript" row.
DEFAULT_TRANSCRIPT_TYPES: list[str] = ["mRNA", "transcript"]

# Minimum character length for a GFF/GTF line to be considered valid.
# A well-formed line has 9 tab-separated fields; even with single-char
# values and 8 tabs the minimum is 17 characters.  We use a conservative
# lower bound to filter blank or truncated lines without risking false
# positives.
_MIN_GFF_LINE_LENGTH = 10


# ------------------------------------------------------------------
# Internal: GFF parsing
# ------------------------------------------------------------------


def _lazy_read_gff(gff_path: str) -> pl.LazyFrame:
    """Build a Polars LazyFrame from a raw GFF3 or GTF file.

    The GFF/GTF format is notoriously variable.  To avoid ``ShapeError``
    from ragged columns we read each line as a single string, strip
    comments, then split by tab.

    Both GFF3 (``ID=``/``Parent=``) and GTF (``transcript_id``) attribute
    conventions are recognised.  For GTF files, ``transcript_id`` is used
    as both the feature identifier and the parent link (since in GTF a
    child exon carries the same ``transcript_id`` as its parent
    transcript row).

    Returns
    -------
    pl.LazyFrame
        Columns: ``seqid``, ``source``, ``type``, ``start`` (Int64),
        ``end`` (Int64), ``score``, ``strand``, ``phase``, ``attributes``,
        ``gff_id``, ``parent``.
    """
    col_name = "column_1"

    # Read entire file as a single-column DataFrame, then convert to lazy.
    # Using read_csv instead of scan_csv because scan_csv hangs with the
    # unit-separator trick needed to force single-column parsing.
    df = pl.read_csv(
        gff_path,
        has_header=False,
        separator="\x1f",  # never occurs in GFF → forces single column
        truncate_ragged_lines=True,
        quote_char=None,
    )

    ldf = (
        df.lazy()
        # Drop comments and blank / too-short lines
        .filter(~pl.col(col_name).str.starts_with("#"))
        .filter(pl.col(col_name).str.len_chars() >= _MIN_GFF_LINE_LENGTH)
        # Split by tab into the 9 canonical GFF columns
        .with_columns(
            pl.col(col_name).str.split_exact("\t", 8).alias("fields")
        )
        .unnest("fields")
        .rename({
            "field_0": "seqid",
            "field_1": "source",
            "field_2": "type",
            "field_3": "start",
            "field_4": "end",
            "field_5": "score",
            "field_6": "strand",
            "field_7": "phase",
            "field_8": "attributes",
        })
        .with_columns([
            pl.col("start").cast(pl.Int64, strict=False),
            pl.col("end").cast(pl.Int64, strict=False),
        ])
        # Extract identifiers for both GFF3 and GTF conventions.
        #   GFF3: ID=foo;Parent=bar
        #   GTF:  gene_id "G1"; transcript_id "T1";
        # For GTF, transcript_id serves as both the row's identity and
        # the parent link (children share the same transcript_id).
        .with_columns([
            pl.col("attributes")
              .str.extract(r"ID=([^;\s]+)")
              .alias("_gff3_id"),
            pl.col("attributes")
              .str.extract(r"Parent=([^;\s]+)")
              .alias("_gff3_parent"),
            pl.col("attributes")
              .str.extract(r'transcript_id "([^"]+)"')
              .alias("_gtf_tid"),
        ])
        .with_columns([
            pl.coalesce("_gff3_id", "_gtf_tid").alias("gff_id"),
            pl.coalesce("_gff3_parent", "_gtf_tid").alias("parent"),
        ])
        .drop(["_gff3_id", "_gff3_parent", "_gtf_tid"])
    )

    return ldf


# ------------------------------------------------------------------
# Public: collect a full GFF into memory
# ------------------------------------------------------------------


def collect_gff(
    gff_path: str | Path,
    exclude_features: list[str] | None = None,
) -> pl.DataFrame:
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
    pl.DataFrame
        Collected DataFrame with standard GFF columns plus ``gff_id``
        and ``parent``.
    """
    lf = _lazy_read_gff(str(gff_path))

    if exclude_features:
        lf = lf.filter(~pl.col("type").is_in(exclude_features))

    df = lf.collect()

    null_coord_count = df.filter(
        pl.col("start").is_null() | pl.col("end").is_null()
    ).height
    if null_coord_count > 0:
        logger.warning(
            "%d row(s) in %s have unparseable coordinates and will be "
            "ignored.",
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
        raise ValueError(
            "LabelConfig must have `coding_label` set to parse GFF "
            "features."
        )

    coding_val = label_config.coding_label
    bg_val = label_config.background_label
    transcript_types = transcript_types or DEFAULT_TRANSCRIPT_TYPES

    try:
        df = collect_gff(str(gff_path), exclude_features=exclude_features)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse {gff_path} with Polars: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Identify transcript rows → they define array boundaries
    # ------------------------------------------------------------------
    transcript_df = df.filter(pl.col("type").is_in(transcript_types))

    if transcript_df.is_empty():
        logger.warning(
            "No transcript features (%s) found in %s. "
            "Returning empty result.",
            transcript_types,
            gff_path,
        )
        return {}

    # ------------------------------------------------------------------
    # Identify child rows → everything that has a Parent and is not
    # itself a transcript-type row
    # ------------------------------------------------------------------
    child_df = df.filter(
        pl.col("parent").is_not_null()
        & (~pl.col("type").is_in(transcript_types))
    )

    # ------------------------------------------------------------------
    # Build per-transcript arrays (single pass)
    # ------------------------------------------------------------------
    annotations: dict[str, np.ndarray] = {}
    transcript_lookup: dict[str, tuple[int, str, np.ndarray]] = {}

    for row in transcript_df.iter_rows(named=True):
        t_id = row["gff_id"]
        t_start = row["start"]
        t_end = row["end"]
        strand = row["strand"]

        if t_id is None or t_start is None or t_end is None:
            logger.debug(
                "Skipping transcript with missing field(s): "
                "id=%s, start=%s, end=%s",
                t_id,
                t_start,
                t_end,
            )
            continue

        array_len = t_end - t_start + 1  # GFF is 1-based inclusive
        key = f"{t_id}_{strand}"
        arr = np.full(array_len, bg_val, dtype=np.int32)
        annotations[key] = arr
        transcript_lookup[t_id] = (t_start, strand, arr)

    # ------------------------------------------------------------------
    # Fill children into their parent transcript's local coordinates
    # ------------------------------------------------------------------
    for row in child_df.iter_rows(named=True):
        parent_id = row["parent"]
        c_start = row["start"]
        c_end = row["end"]

        if parent_id is None or c_start is None or c_end is None:
            continue

        if parent_id not in transcript_lookup:
            continue

        t_start, _, arr = transcript_lookup[parent_id]

        local_start = max(0, c_start - t_start)
        local_end = min(len(arr), c_end - t_start + 1)

        if c_start < t_start or c_end > t_start + len(arr) - 1:
            logger.debug(
                "Child feature of %s extends outside transcript "
                "boundaries (child: %d-%d, transcript: %d-%d). "
                "Clipping to transcript span.",
                parent_id,
                c_start,
                c_end,
                t_start,
                t_start + len(arr) - 1,
            )

        if local_start < local_end:
            arr[local_start:local_end] = coding_val

    return annotations