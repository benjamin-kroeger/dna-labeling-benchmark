"""Generate the dummy integration-test fixtures.

These are tiny, hand-crafted stand-ins that mimic the *format* of a trimmed
GENCODE GTF (ground truth) and an AUGUSTUS GFF prediction on ``chr2``.  They
let the integration suite run end-to-end offline without the multi-GB real
files.

Replace the generated files with real trimmed data when available:

* ``gencode_chr2_snippet.gtf`` — a few protein-coding genes sliced from the
  real GENCODE GTF (same ``chr2`` window as the Augustus prediction).
* ``augustus_chr2_snippet.gff`` — the matching Augustus prediction over that
  window.

The integration tests assert *qualitative* known-answers (identity → perfect,
exon-skip → recall drops, …) and structure/ranges, so they keep passing after
the swap as long as the files share overlapping ``chr2`` loci on the same
strand.

Run from the repo root::

    python tests/data/_make_dummy_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent


def _gtf_line(seqid, ftype, start, end, strand, frame, attrs) -> str:
    return "\t".join([seqid, "HAVANA", ftype, str(start), str(end), ".", strand, frame, attrs])


def _gff_line(seqid, ftype, start, end, strand, frame, attrs) -> str:
    return "\t".join([seqid, "AUGUSTUS", ftype, str(start), str(end), ".", strand, frame, attrs])


def _gencode_gene(gene_id, tx_id, name, strand, gene_span, exons, cds, utrs, start_codon, stop_codon):
    """Emit GENCODE-style GTF rows for one protein-coding transcript."""
    g_attr = f'gene_id "{gene_id}"; gene_type "protein_coding"; gene_name "{name}";'
    t_attr = (
        f'gene_id "{gene_id}"; transcript_id "{tx_id}"; gene_type "protein_coding"; '
        f'transcript_type "protein_coding"; gene_name "{name}";'
    )
    rows = [
        _gtf_line("chr2", "gene", gene_span[0], gene_span[1], strand, ".", g_attr),
        _gtf_line("chr2", "transcript", gene_span[0], gene_span[1], strand, ".", t_attr),
    ]
    for s, e in exons:
        rows.append(_gtf_line("chr2", "exon", s, e, strand, ".", t_attr))
    for s, e in cds:
        rows.append(_gtf_line("chr2", "CDS", s, e, strand, "0", t_attr))
    for s, e in utrs:  # GENCODE uses a bare "UTR" type (no 5'/3' distinction)
        rows.append(_gtf_line("chr2", "UTR", s, e, strand, ".", t_attr))
    rows.append(_gtf_line("chr2", "start_codon", start_codon[0], start_codon[1], strand, "0", t_attr))
    rows.append(_gtf_line("chr2", "stop_codon", stop_codon[0], stop_codon[1], strand, "0", t_attr))
    return rows


def _augustus_gene(gene_id, strand, gene_span, cds, start_codon=None, stop_codon=None):
    """Emit AUGUSTUS-style GFF3 rows for one (CDS-only) gene prediction."""
    tx_id = f"{gene_id}.t1"
    rows = [
        _gff_line("chr2", "gene", gene_span[0], gene_span[1], strand, ".", f"ID={gene_id}"),
        _gff_line("chr2", "transcript", gene_span[0], gene_span[1], strand, ".", f"ID={tx_id};Parent={gene_id}"),
    ]
    if start_codon is not None:
        rows.append(_gff_line("chr2", "start_codon", start_codon[0], start_codon[1], strand, "0", f"Parent={tx_id}"))
    for s, e in cds:
        rows.append(_gff_line("chr2", "CDS", s, e, strand, "0", f"ID={tx_id}.cds;Parent={tx_id}"))
    if stop_codon is not None:
        rows.append(_gff_line("chr2", "stop_codon", stop_codon[0], stop_codon[1], strand, "0", f"Parent={tx_id}"))
    return rows


def build_gencode() -> str:
    rows: list[str] = ["##description: dummy GENCODE-style snippet (replace with real trimmed data)"]
    # Gene A: + strand, 3 exons, protein-coding.
    rows += _gencode_gene(
        "ENSGA", "ENSTA", "GENEA", "+", (1000, 2800),
        exons=[(1000, 1200), (1500, 1700), (2500, 2800)],
        cds=[(1100, 1200), (1500, 1700), (2500, 2700)],
        utrs=[(1000, 1099), (2701, 2800)],
        start_codon=(1100, 1102), stop_codon=(2698, 2700),
    )
    # Gene B: - strand, 3 exons (biological 5' end is the high-coordinate end).
    rows += _gencode_gene(
        "ENSGB", "ENSTB", "GENEB", "-", (5000, 7000),
        exons=[(5000, 5300), (5600, 5800), (6500, 7000)],
        cds=[(5100, 5300), (5600, 5800), (6500, 6800)],
        utrs=[(5000, 5099), (6801, 7000)],
        start_codon=(6798, 6800), stop_codon=(5100, 5102),
    )
    # Gene C: + strand, single-exon gene.
    rows += _gencode_gene(
        "ENSGC", "ENSTC", "GENEC", "+", (8000, 8500),
        exons=[(8000, 8500)],
        cds=[(8050, 8400)],
        utrs=[(8000, 8049), (8401, 8500)],
        start_codon=(8050, 8052), stop_codon=(8398, 8400),
    )
    return "\n".join(rows) + "\n"


def build_augustus() -> str:
    rows: list[str] = ["##gff-version 3"]
    # g1 → Gene A: exact CDS chain match (no UTR, as Augustus is CDS-only).
    rows += _augustus_gene("g1", "+", (1100, 2700),
                           cds=[(1100, 1200), (1500, 1700), (2500, 2700)],
                           start_codon=(1100, 1102), stop_codon=(2698, 2700))
    # g2 → Gene B: matching minus-strand CDS chain.
    rows += _augustus_gene("g2", "-", (5100, 6800),
                           cds=[(5100, 5300), (5600, 5800), (6500, 6800)],
                           start_codon=(6798, 6800), stop_codon=(5100, 5102))
    # g3 → Gene C: single-exon prediction with a small 5' boundary error.
    rows += _augustus_gene("g3", "+", (8060, 8400), cds=[(8060, 8400)])
    # g4 → hallucinated gene with no GT overlap (pure false positive).
    rows += _augustus_gene("g4", "+", (9000, 9300), cds=[(9000, 9300)])
    return "\n".join(rows) + "\n"


def main() -> None:
    (FIXTURES_DIR / "gencode_chr2_snippet.gtf").write_text(build_gencode())
    (FIXTURES_DIR / "augustus_chr2_snippet.gff").write_text(build_augustus())
    print(f"Wrote dummy fixtures to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
