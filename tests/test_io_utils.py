"""Tests for the I/O utilities module.

Covers:
- GFF3 parsing and per-transcript array construction
- GTF parsing support
- Feature exclusion
- Strand isolation
- Boundary clipping logging
"""

import numpy as np
import pytest

from dna_segmentation_benchmark.io_utils import (
    collect_gff,
    read_gff_to_arrays,
)
from dna_segmentation_benchmark.label_definition import LabelConfig


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def simple_config():
    """Two-token label config (coding=0, background=1)."""
    return LabelConfig(
        labels={0: "CODING", 1: "BACKGROUND"},
        background_label=1,
        coding_label=0,
    )


@pytest.fixture
def hierarchical_gff(tmp_path):
    """GFF3 with proper gene -> mRNA -> exon/CDS hierarchy.

    gene1 (1-30, +strand) -> mRNA1 (1-30) -> CDS at 1-10, exon at 20-28
    gene2 (50-80, -strand) -> mRNA2 (50-80) -> CDS at 55-65
    """
    content = """\
##gff-version 3
seq1\tTest\tgene\t1\t30\t.\t+\t.\tID=gene1
seq1\tTest\tmRNA\t1\t30\t.\t+\t.\tID=mRNA1;Parent=gene1
seq1\tTest\tCDS\t1\t10\t.\t+\t0\tID=cds1;Parent=mRNA1
seq1\tTest\texon\t20\t28\t.\t+\t.\tID=exon1;Parent=mRNA1
seq1\tTest\tgene\t50\t80\t.\t-\t.\tID=gene2
seq1\tTest\tmRNA\t50\t80\t.\t-\t.\tID=mRNA2;Parent=gene2
seq1\tTest\tCDS\t55\t65\t.\t-\t0\tID=cds2;Parent=mRNA2
"""
    f = tmp_path / "test.gff"
    f.write_text(content)
    return str(f)


@pytest.fixture
def gtf_file(tmp_path):
    """GTF file with transcript_id / gene_id attribute convention.

    gene1 (chr1, +, 1-100) -> transcript T1 (1-100) -> CDS at 10-30
    """
    content = """\
chr1\tTest\tgene\t1\t100\t.\t+\t.\tgene_id "gene1"; gene_name "GENE1";
chr1\tTest\ttranscript\t1\t100\t.\t+\t.\tgene_id "gene1"; transcript_id "T1";
chr1\tTest\tCDS\t10\t30\t.\t+\t0\tgene_id "gene1"; transcript_id "T1";
chr1\tTest\texon\t50\t70\t.\t+\t.\tgene_id "gene1"; transcript_id "T1";
"""
    f = tmp_path / "test.gtf"
    f.write_text(content)
    return str(f)


# ------------------------------------------------------------------
# Tests: GFF3 parsing
# ------------------------------------------------------------------


def test_per_transcript_arrays(hierarchical_gff, simple_config):
    """Each mRNA produces its own array, not one per chromosome."""
    arrays = read_gff_to_arrays(
        hierarchical_gff,
        simple_config,
        exclude_features=["gene"],
    )

    assert "mRNA1_+" in arrays
    assert "mRNA2_-" in arrays

    # mRNA1: span 1-30 -> array length 30
    arr1 = arrays["mRNA1_+"]
    assert len(arr1) == 30

    # CDS 1-10 maps to local indices 0-9 (coding)
    np.testing.assert_array_equal(arr1[0:10], np.full(10, 0))
    # Gap 11-19 stays background
    np.testing.assert_array_equal(arr1[10:19], np.full(9, 1))
    # Exon 20-28 maps to local indices 19-27 (coding)
    np.testing.assert_array_equal(arr1[19:28], np.full(9, 0))
    # Position 29 stays background
    assert arr1[29] == 1


def test_minus_strand_coordinates(hierarchical_gff, simple_config):
    """Minus-strand arrays use genomic (left-to-right) coordinates."""
    arrays = read_gff_to_arrays(
        hierarchical_gff,
        simple_config,
        exclude_features=["gene"],
    )

    # mRNA2: span 50-80 -> array length 31
    arr2 = arrays["mRNA2_-"]
    assert len(arr2) == 31

    # CDS 55-65 maps to local indices 5-15 (coding)
    np.testing.assert_array_equal(arr2[0:5], np.full(5, 1))
    np.testing.assert_array_equal(arr2[5:16], np.full(11, 0))
    np.testing.assert_array_equal(arr2[16:31], np.full(15, 1))


def test_exclude_features(hierarchical_gff, simple_config):
    """Excluding CDS leaves transcript arrays with only exon children."""
    arrays = read_gff_to_arrays(
        hierarchical_gff,
        simple_config,
        exclude_features=["gene", "CDS"],
    )

    # mRNA1_+: CDS excluded -> positions 0-9 are background
    arr1 = arrays["mRNA1_+"]
    np.testing.assert_array_equal(arr1[0:10], np.full(10, 1))
    # Exon 20-28 -> local 19-27 still coding
    np.testing.assert_array_equal(arr1[19:28], np.full(9, 0))

    # mRNA2_-: CDS excluded, no other children -> all background
    arr2 = arrays["mRNA2_-"]
    np.testing.assert_array_equal(arr2, np.full(31, 1))


def test_strand_isolation(hierarchical_gff, simple_config):
    """Plus and minus strand features do not interfere."""
    arrays = read_gff_to_arrays(
        hierarchical_gff,
        simple_config,
        exclude_features=["gene"],
    )

    plus_keys = [k for k in arrays if k.endswith("_+")]
    minus_keys = [k for k in arrays if k.endswith("_-")]
    assert len(plus_keys) >= 1
    assert len(minus_keys) >= 1

    # Plus array has coding at positions 0-9 and 19-27
    arr_plus = arrays["mRNA1_+"]
    assert arr_plus[0] == 0
    assert arr_plus[19] == 0

    # Minus array has coding at positions 5-15
    arr_minus = arrays["mRNA2_-"]
    assert arr_minus[5] == 0
    assert arr_minus[0] == 1


# ------------------------------------------------------------------
# Tests: GTF support
# ------------------------------------------------------------------


def test_gtf_parsing(gtf_file, simple_config):
    """GTF files with transcript_id attributes are parsed correctly."""
    arrays = read_gff_to_arrays(
        gtf_file,
        simple_config,
        exclude_features=["gene"],
    )

    assert "T1_+" in arrays, (
        f"Expected 'T1_+' in keys, got {list(arrays.keys())}"
    )

    # T1: span 1-100 -> array length 100
    arr = arrays["T1_+"]
    assert len(arr) == 100

    # CDS 10-30 maps to local indices 9-29 (coding)
    np.testing.assert_array_equal(arr[0:9], np.full(9, 1))
    np.testing.assert_array_equal(arr[9:30], np.full(21, 0))

    # Exon 50-70 maps to local indices 49-69 (coding)
    np.testing.assert_array_equal(arr[30:49], np.full(19, 1))
    np.testing.assert_array_equal(arr[49:70], np.full(21, 0))
    np.testing.assert_array_equal(arr[70:100], np.full(30, 1))


# ------------------------------------------------------------------
# Tests: collect_gff
# ------------------------------------------------------------------


def test_collect_gff_returns_dataframe(hierarchical_gff):
    """collect_gff returns a DataFrame with expected columns."""
    df = collect_gff(hierarchical_gff)

    assert "seqid" in df.columns
    assert "gff_id" in df.columns
    assert "parent" in df.columns
    assert df.height > 0


def test_collect_gff_excludes_features(hierarchical_gff):
    """collect_gff respects exclude_features."""
    df_all = collect_gff(hierarchical_gff)
    df_no_gene = collect_gff(hierarchical_gff, exclude_features=["gene"])

    gene_count = df_all.filter(
        df_all["type"] == "gene"
    ).height

    assert gene_count > 0
    assert df_no_gene.height == df_all.height - gene_count