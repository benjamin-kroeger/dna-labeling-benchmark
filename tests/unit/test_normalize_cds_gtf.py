"""Regression tests for real_data.preprocess.normalize_cds_gtf."""

from thesis_code.real_data.preprocess import normalize_cds_gtf


def _transcripts(path):
    return [l.split("\t") for l in open(path) if "\ttranscript\t" in l]


def test_recycled_id_across_seqids_not_merged(tmp_path):
    """A transcript_id reused on two seqids must yield two transcripts, not one chimera.

    Tiberius restarts gene ids per sequence; keying CDS by transcript_id alone
    merged CDS from unrelated genes into one multi-megabase transcript.
    """
    src = tmp_path / "in.gtf"
    src.write_text(
        'chr1\tTib\tCDS\t100\t200\t.\t+\t0\tgene_id "g1"; transcript_id "g1.t1";\n'
        'chr2\tTib\tCDS\t5000\t5100\t.\t+\t0\tgene_id "g1"; transcript_id "g1.t1";\n'
    )
    txs = _transcripts(normalize_cds_gtf(src, tmp_path / "out.gff3"))
    assert len(txs) == 2
    assert max(int(t[4]) - int(t[3]) for t in txs) < 1000  # no cross-seqid span


def test_single_transcript_grouped_from_cds(tmp_path):
    """CDS sharing a (seqid, transcript_id) collapse into one spanning transcript."""
    src = tmp_path / "in.gtf"
    src.write_text(
        'chr1\tTib\tCDS\t100\t200\t.\t+\t0\ttranscript_id "t1";\n'
        'chr1\tTib\tCDS\t300\t400\t.\t+\t2\ttranscript_id "t1";\n'
    )
    dst = normalize_cds_gtf(src, tmp_path / "out.gff3")
    txs = _transcripts(dst)
    cds = [l for l in open(dst) if "\tCDS\t" in l]
    assert len(txs) == 1 and (int(txs[0][3]), int(txs[0][4])) == (100, 400)
    assert len(cds) == 2
