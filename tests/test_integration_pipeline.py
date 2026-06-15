"""End-to-end integration tests for the read-GFF benchmark pipeline.

Unlike the unit tests (which exercise ``collect_gff`` / ``map_transcripts`` /
``build_paired_arrays`` / the metric core in isolation), these drive the whole
``benchmark_from_gff`` chain from files: a trimmed GENCODE-style ground truth
and an Augustus-style prediction (dummy fixtures; swap in real trimmed data
later).  Assertions are *known-answer* and *qualitative* so they survive that
swap as long as the files share overlapping ``chr2`` loci on the same strand.
"""

from __future__ import annotations

import numpy as np
import pytest

from dna_segmentation_benchmark.eval.evaluate_predictors import EvalMetrics
from dna_segmentation_benchmark.io_utils import collect_gff
from dna_segmentation_benchmark.pipeline import benchmark_from_gff
from dna_segmentation_benchmark.transcript_mapping import (
    LocusMatchingMode,
    build_paired_arrays,
    map_transcripts,
)

EXON_INTRON_METRICS = [
    EvalMetrics.REGION_DISCOVERY,
    EvalMetrics.BOUNDARY_EXACTNESS,
    EvalMetrics.NUCLEOTIDE_CLASSIFICATION,
    EvalMetrics.STRUCTURAL_COHERENCE,
]


def _nucleotide(result: dict, predictor: str) -> dict:
    return result[predictor]["aggregated"]["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]


# ---------------------------------------------------------------------------
# Real-dialect end-to-end run (GENCODE GT vs Augustus prediction)
# ---------------------------------------------------------------------------


def test_pipeline_runs_end_to_end_gencode_vs_augustus(gencode_gtf, augustus_gff, exon_intron_config):
    """The full file→benchmark chain returns the documented result structure."""
    results = benchmark_from_gff(
        gt_path=gencode_gtf,
        pred_paths={"augustus": augustus_gff},
        label_config=exon_intron_config,
        metrics=EXON_INTRON_METRICS,
        locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
        infer_introns=True,
    )

    assert set(results) == {"augustus"}
    assert set(results["augustus"]) == {"aggregated", "global"}

    aggregated = results["augustus"]["aggregated"]
    for metric in EXON_INTRON_METRICS:
        assert metric.name in aggregated
    assert aggregated["metadata"]["annotation_mode"] == "EXON_INTRON"

    nuc = _nucleotide(results, "augustus")
    # Augustus is CDS-only: every predicted base is real (precision 1.0) but
    # it cannot recover the UTR portions of the GENCODE exons (recall < 1).
    assert nuc["precision"] == pytest.approx(1.0)
    assert 0.0 < nuc["recall"] < 1.0
    # Region discovery still finds every gene locus.
    assert results["augustus"]["aggregated"]["REGION_DISCOVERY"]["neighborhood_hit"]["recall"] == pytest.approx(1.0)


def test_full_discovery_penalises_hallucinated_prediction(gencode_gtf, augustus_gff, exon_intron_config):
    """The unmatched Augustus gene (g4) becomes a false positive under FULL_DISCOVERY."""
    results = benchmark_from_gff(
        gt_path=gencode_gtf,
        pred_paths={"augustus": augustus_gff},
        label_config=exon_intron_config,
        metrics=[EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        locus_matching_mode=LocusMatchingMode.FULL_DISCOVERY,
    )
    nuc = _nucleotide(results, "augustus")
    # The hallucinated locus contributes pure FP bases → precision drops below 1.
    assert nuc["precision"] < 1.0
    assert nuc["recall"] > 0.0


# ---------------------------------------------------------------------------
# Identity & negative controls
# ---------------------------------------------------------------------------


def test_identity_prediction_is_perfect(gencode_gtf, exon_intron_config):
    """Benchmarking the ground truth against itself yields a perfect score."""
    results = benchmark_from_gff(
        gt_path=gencode_gtf,
        pred_paths={"perfect": gencode_gtf},
        label_config=exon_intron_config,
        metrics=EXON_INTRON_METRICS,
        infer_introns=True,
    )
    nuc = _nucleotide(results, "perfect")
    assert nuc["precision"] == pytest.approx(1.0)
    assert nuc["recall"] == pytest.approx(1.0)
    assert nuc["f1"] == pytest.approx(1.0)

    region = results["perfect"]["aggregated"]["REGION_DISCOVERY"]
    assert region["perfect_boundary_hit"]["precision"] == pytest.approx(1.0)
    assert region["perfect_boundary_hit"]["recall"] == pytest.approx(1.0)
    # Every transcript chain matches exactly.
    assert results["perfect"]["aggregated"]["STRUCTURAL_COHERENCE"]["exact_match_rate"] == pytest.approx(1.0)


def test_empty_prediction_is_negative_control(gencode_gtf, exon_intron_config, tmp_path):
    """A prediction with no features scores zero precision and recall."""
    empty = tmp_path / "empty.gff3"
    empty.write_text("##gff-version 3\n")

    results = benchmark_from_gff(
        gt_path=gencode_gtf,
        pred_paths={"empty": str(empty)},
        label_config=exon_intron_config,
        metrics=[EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        locus_matching_mode=LocusMatchingMode.FULL_DISCOVERY,
    )
    nuc = _nucleotide(results, "empty")
    # No predicted bases anywhere: zero recall, and zero precision by convention.
    assert nuc["precision"] == pytest.approx(0.0)
    assert nuc["recall"] == pytest.approx(0.0)


def test_pipeline_is_deterministic(gencode_gtf, augustus_gff, exon_intron_config):
    """Two identical runs produce identical nucleotide metrics."""
    kwargs = dict(
        gt_path=gencode_gtf,
        pred_paths={"augustus": augustus_gff},
        label_config=exon_intron_config,
        metrics=[EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
        locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
    )
    assert _nucleotide(benchmark_from_gff(**kwargs), "augustus") == _nucleotide(
        benchmark_from_gff(**kwargs), "augustus"
    )


# ---------------------------------------------------------------------------
# Coordinate convention & GTF/GFF3 equivalence
# ---------------------------------------------------------------------------


def test_gff_coordinates_are_one_based_inclusive(gencode_gtf, augustus_gff, exon_intron_config):
    """A GFF feature at 1-based [start, end] paints exactly that local span.

    Gene A's transcript spans chr2:1000-2800 and its first exon is 1000-1200,
    so local indices 0..200 (201 bp) must carry the exon label.
    """
    gt_df = collect_gff(gencode_gtf)
    pred_df = collect_gff(augustus_gff)
    mappings = map_transcripts(
        gt_path=gencode_gtf,
        pred_paths={"augustus": augustus_gff},
        label_config=exon_intron_config,
        locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
    )
    gene_a = next(m for m in mappings if m.gt_id == "ENSTA")
    assert gene_a.gt_start == 1000 and gene_a.gt_end == 2800

    gt_array, _ = build_paired_arrays(
        mapping=gene_a,
        gt_df=gt_df,
        pred_dfs={"augustus": pred_df},
        label_config=exon_intron_config,
    )
    # 1-based inclusive: span length = end - start + 1.
    assert len(gt_array) == 2800 - 1000 + 1
    exon_label = exon_intron_config.exon_label
    # First exon 1000-1200 → local indices 0..200 inclusive (gene A is + strand).
    np.testing.assert_array_equal(gt_array[0:201], np.full(201, exon_label))
    # The following intron gap (1201-1499) is not exon.
    assert gt_array[201] != exon_label


def test_gtf_and_gff3_are_equivalent(gencode_gtf, exon_intron_config, gff3_tools, tmp_path):
    """The same annotation as GTF or GFF3 yields identical metrics."""
    # Derive a GFF3 from the GTF so this stays valid after a real-data swap.
    gff3_path = gff3_tools.write_gff3(collect_gff(gencode_gtf), tmp_path / "gt.gff3")

    common = dict(
        label_config=exon_intron_config,
        metrics=[EvalMetrics.NUCLEOTIDE_CLASSIFICATION, EvalMetrics.REGION_DISCOVERY],
    )
    from_gtf = benchmark_from_gff(gt_path=gencode_gtf, pred_paths={"self": gencode_gtf}, **common)
    from_gff3 = benchmark_from_gff(gt_path=gff3_path, pred_paths={"self": gff3_path}, **common)

    assert _nucleotide(from_gtf, "self") == _nucleotide(from_gff3, "self")


# ---------------------------------------------------------------------------
# Structural error modes (controlled mutants of Gene A)
# ---------------------------------------------------------------------------


def _gene_a_pred(gencode_gtf, gff3_tools, tmp_path, mutator=None, name="pred.gff3"):
    """Build a single-gene (Gene A) prediction file, optionally perturbed."""
    subset = gff3_tools.transcript_subset(collect_gff(gencode_gtf), gff3_tools.target_transcript)
    if mutator is not None:
        subset = mutator(subset)
    return gff3_tools.write_gff3(subset, tmp_path / name)


def _gene_a_recall(gencode_gtf, pred_path, label_config, metrics, infer_introns=False):
    results = benchmark_from_gff(
        gt_path=gencode_gtf,
        pred_paths={"pred": pred_path},
        label_config=label_config,
        metrics=metrics,
        locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
        infer_introns=infer_introns,
    )
    return results["pred"]["aggregated"]


def test_exon_skipping_reduces_recall(gencode_gtf, exon_intron_config, gff3_tools, tmp_path):
    """Dropping an internal exon lowers recall and breaks the exact chain match."""
    perfect = _gene_a_pred(gencode_gtf, gff3_tools, tmp_path, name="perfect.gff3")
    skipped = _gene_a_pred(gencode_gtf, gff3_tools, tmp_path, gff3_tools.drop_internal_exon, "skip.gff3")

    metrics = [EvalMetrics.NUCLEOTIDE_CLASSIFICATION, EvalMetrics.STRUCTURAL_COHERENCE]
    perfect_agg = _gene_a_recall(gencode_gtf, perfect, exon_intron_config, metrics, infer_introns=True)
    skipped_agg = _gene_a_recall(gencode_gtf, skipped, exon_intron_config, metrics, infer_introns=True)

    assert perfect_agg["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]["recall"] == pytest.approx(1.0)
    assert skipped_agg["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]["recall"] < 1.0
    # Perfect chain match for the unperturbed gene; broken once an exon is dropped.
    assert perfect_agg["STRUCTURAL_COHERENCE"]["exact_match_rate"] == pytest.approx(1.0)
    assert skipped_agg["STRUCTURAL_COHERENCE"]["exact_match_rate"] < 1.0


def test_intron_retention_adds_false_positive_bases(gencode_gtf, exon_intron_config, gff3_tools, tmp_path):
    """Merging two exons paints the intervening intron, creating FP bases."""
    retained = _gene_a_pred(gencode_gtf, gff3_tools, tmp_path, gff3_tools.merge_first_two_exons, "retain.gff3")
    agg = _gene_a_recall(
        gencode_gtf, retained, exon_intron_config,
        [EvalMetrics.NUCLEOTIDE_CLASSIFICATION],
    )
    nuc = agg["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]
    # The merged exon paints the intervening intron → predicted bases the GT
    # calls background, i.e. false positives → precision drops below 1.
    assert nuc["precision"] < 1.0
    # No GT exon base is lost, so recall is unaffected.
    assert nuc["recall"] == pytest.approx(1.0)


def test_cds_indel_adds_false_positive_at_cds_scope(gencode_gtf, cds_config, gff3_tools, tmp_path):
    """A 1 bp CDS extension is caught as a false-positive CDS base (CDS scope)."""
    perfect = _gene_a_pred(gencode_gtf, gff3_tools, tmp_path, name="cds_perfect.gff3")
    indel = _gene_a_pred(
        gencode_gtf, gff3_tools, tmp_path,
        lambda df: gff3_tools.extend_first_cds(df, delta=1),
        "cds_indel.gff3",
    )
    metrics = [EvalMetrics.NUCLEOTIDE_CLASSIFICATION, EvalMetrics.INDEL]
    perfect_agg = _gene_a_recall(gencode_gtf, perfect, cds_config, metrics)
    indel_agg = _gene_a_recall(gencode_gtf, indel, cds_config, metrics)
    perfect_nuc = perfect_agg["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]
    indel_nuc = indel_agg["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]

    assert perfect_nuc["precision"] == pytest.approx(1.0)
    # The extra CDS base is a false positive → precision drops below the clean run.
    assert indel_nuc["precision"] < perfect_nuc["precision"]
    # Recall is unaffected: no GT CDS base was removed.
    assert indel_nuc["recall"] == pytest.approx(perfect_nuc["recall"])
    # The indel surfaces in the INDEL metric's boundary breakdown.
    assert indel_agg["INDEL"]["by_boundary"]


# ---------------------------------------------------------------------------
# Annotation modes
# ---------------------------------------------------------------------------


def test_cds_scope_mode_runs_and_populates_metrics(gencode_gtf, augustus_gff, cds_config):
    """UTR_CDS_INTRON @ CDS scope evaluates Augustus on coding bases only."""
    results = benchmark_from_gff(
        gt_path=gencode_gtf,
        pred_paths={"augustus": augustus_gff},
        label_config=cds_config,
        metrics=[EvalMetrics.NUCLEOTIDE_CLASSIFICATION, EvalMetrics.REGION_DISCOVERY],
        locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
    )
    aggregated = results["augustus"]["aggregated"]
    assert aggregated["metadata"]["evaluation_scope"] == "cds"
    nuc = aggregated["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]
    # Augustus reproduces the CDS almost exactly (only Gene C's boundary is off).
    assert nuc["precision"] == pytest.approx(1.0)
    assert nuc["recall"] > 0.9


# ---------------------------------------------------------------------------
# Strand & coordinate correctness
# ---------------------------------------------------------------------------


def _strand_gene(path, strand, exons, txid):
    """Write a tiny single-transcript GFF3 with exon==CDS structure."""
    lines = ["##gff-version 3"]
    start, end = exons[0][0], exons[-1][1]
    lines.append(f"chr9\tT\ttranscript\t{start}\t{end}\t.\t{strand}\t.\tID={txid}")
    for i, (a, b) in enumerate(exons):
        lines.append(f"chr9\tT\texon\t{a}\t{b}\t.\t{strand}\t.\tID={txid}.e{i};Parent={txid}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_minus_strand_orientation_symmetry(exon_intron_config, tmp_path):
    """A 5'-end error scores identically on + and - strand genes.

    Both genes share the same (symmetric) genomic exon structure.  The
    prediction drops the *biological 5'* exon — the lowest-coordinate exon on
    the + strand, the highest-coordinate exon on the - strand.  Because
    ``build_paired_arrays`` normalises both to 5'→3' order, the two cases must
    yield identical metrics.  (Guards against the minus-strand orientation gap.)
    """
    exons = [(100, 200), (300, 400), (500, 600)]  # symmetric spacing

    gt_plus = _strand_gene(tmp_path / "gt_plus.gff3", "+", exons, "txp")
    gt_minus = _strand_gene(tmp_path / "gt_minus.gff3", "-", exons, "txm")
    # + strand 5' exon = lowest coords; - strand 5' exon = highest coords.
    pred_plus = _strand_gene(tmp_path / "pred_plus.gff3", "+", exons[1:], "pp")
    pred_minus = _strand_gene(tmp_path / "pred_minus.gff3", "-", exons[:-1], "pm")

    metrics = [EvalMetrics.NUCLEOTIDE_CLASSIFICATION, EvalMetrics.REGION_DISCOVERY]
    plus = benchmark_from_gff(gt_path=gt_plus, pred_paths={"x": pred_plus},
                              label_config=exon_intron_config, metrics=metrics)
    minus = benchmark_from_gff(gt_path=gt_minus, pred_paths={"x": pred_minus},
                               label_config=exon_intron_config, metrics=metrics)

    plus_nuc = _nucleotide(plus, "x")
    minus_nuc = _nucleotide(minus, "x")
    assert plus_nuc["recall"] == pytest.approx(minus_nuc["recall"])
    assert plus_nuc["precision"] == pytest.approx(minus_nuc["precision"])
    # The 5' exon (1/3 of the coding bases) is missed on both strands.
    assert minus_nuc["recall"] == pytest.approx(2 / 3)
