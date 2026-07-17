"""Regression test: CLI `run` and `benchmark_from_gff` must agree on all metrics.

The critical case is BEST_PER_LOCUS with a multi-isoform locus: one predictor
matches only one of two GT isoforms, so `mapping.fn_for_predictors` is set on
the locus-level FN entry.  Before the fix, the CLI was missing that branch and
could double-count or misattribute missed loci.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from gene_calling_benchmark.cli import cli
from gene_calling_benchmark.eval.evaluate_predictors import EvalMetrics
from gene_calling_benchmark.label_definition import BEND_LABEL_CONFIG
from gene_calling_benchmark.pipeline import benchmark_from_gff
from gene_calling_benchmark.transcript_mapping import LocusMatchingMode

from support.gff import write_gff


def _write_two_isoform_gt(path: Path) -> str:
    """Two isoforms of the same gene locus on chr1 (+strand).

    Isoform A: exons 100-200, 300-400  (junction: 200-300)
    Isoform B: exons 100-200, 500-600  (junction: 200-500 — different)

    A predictor that only matches isoform A will trigger fn_for_predictors=[pred]
    on the locus-FN entry for isoform B in BEST_PER_LOCUS mode.
    """
    return write_gff(
        path,
        [
            "chr1\tT\ttranscript\t100\t400\t.\t+\t.\tID=txA",
            "chr1\tT\texon\t100\t200\t.\t+\t.\tID=txA.e1;Parent=txA",
            "chr1\tT\texon\t300\t400\t.\t+\t.\tID=txA.e2;Parent=txA",
            "chr1\tT\ttranscript\t100\t600\t.\t+\t.\tID=txB",
            "chr1\tT\texon\t100\t200\t.\t+\t.\tID=txB.e1;Parent=txB",
            "chr1\tT\texon\t500\t600\t.\t+\t.\tID=txB.e2;Parent=txB",
        ],
    )


def _write_isoform_a_pred(path: Path) -> str:
    """Prediction matching only isoform A's junction (100-200, 300-400)."""
    return write_gff(
        path,
        [
            "chr1\tT\ttranscript\t100\t400\t.\t+\t.\tID=predA",
            "chr1\tT\texon\t100\t200\t.\t+\t.\tID=predA.e1;Parent=predA",
            "chr1\tT\texon\t300\t400\t.\t+\t.\tID=predA.e2;Parent=predA",
        ],
    )


@pytest.fixture
def two_isoform_gt(tmp_path):
    return _write_two_isoform_gt(tmp_path / "gt_two_isoforms.gff3")


@pytest.fixture
def isoform_a_pred(tmp_path):
    return _write_isoform_a_pred(tmp_path / "pred_a.gff3")


@pytest.fixture
def label_config():
    return BEND_LABEL_CONFIG


def _nuc(result: dict, predictor: str) -> dict:
    return result[predictor]["aggregated"]["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]


# ---------------------------------------------------------------------------
# Parity: benchmark_from_gff vs CLI run (BEST_PER_LOCUS)
# ---------------------------------------------------------------------------


def test_cli_and_pipeline_agree_best_per_locus(two_isoform_gt, isoform_a_pred, label_config, tmp_path):
    """CLI and benchmark_from_gff must return identical nucleotide metrics.

    This is the fn_for_predictors regression: before the fix the CLI skipped
    the `mapping.fn_for_predictors` branch so BEST_PER_LOCUS results diverged.
    """
    metrics = [EvalMetrics.NUCLEOTIDE_CLASSIFICATION, EvalMetrics.REGION_DISCOVERY]

    lib_result = benchmark_from_gff(
        gt_path=two_isoform_gt,
        pred_paths={"pred": isoform_a_pred},
        label_config=label_config,
        metrics=metrics,
        locus_matching_mode=LocusMatchingMode.BEST_PER_LOCUS,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "annotation_mode: EXON_INTRON\n"
        "background_label: 8\n"
        "exon_label: 0\n"
        "intron_label: 2\n"
        "splice_donor_label: 1\n"
        "splice_acceptor_label: 3\n"
    )
    output_path = tmp_path / "cli_out.json"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--gt", two_isoform_gt,
            "--pred", f"pred:{isoform_a_pred}",
            "--config", str(config_path),
            "--locus-matching", "best_per_locus",
            "--metrics", "NUCLEOTIDE_CLASSIFICATION",
            "--metrics", "REGION_DISCOVERY",
            "--output", str(output_path),
        ],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}\n{result.exception}"

    cli_result = json.loads(output_path.read_text())

    lib_nuc = _nuc(lib_result, "pred")
    cli_nuc = cli_result["pred"]["aggregated"]["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]

    assert cli_nuc["precision"] == pytest.approx(lib_nuc["precision"], abs=1e-9)
    assert cli_nuc["recall"] == pytest.approx(lib_nuc["recall"], abs=1e-9)
    assert cli_nuc["f1"] == pytest.approx(lib_nuc["f1"], abs=1e-9)


def test_cli_and_pipeline_agree_full_discovery(two_isoform_gt, isoform_a_pred, label_config, tmp_path):
    """Parity also holds under FULL_DISCOVERY (the simpler path, regression guard)."""
    metrics = [EvalMetrics.NUCLEOTIDE_CLASSIFICATION]

    lib_result = benchmark_from_gff(
        gt_path=two_isoform_gt,
        pred_paths={"pred": isoform_a_pred},
        label_config=label_config,
        metrics=metrics,
        locus_matching_mode=LocusMatchingMode.FULL_DISCOVERY,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "annotation_mode: EXON_INTRON\n"
        "background_label: 8\n"
        "exon_label: 0\n"
        "intron_label: 2\n"
        "splice_donor_label: 1\n"
        "splice_acceptor_label: 3\n"
    )
    output_path = tmp_path / "cli_out.json"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--gt", two_isoform_gt,
            "--pred", f"pred:{isoform_a_pred}",
            "--config", str(config_path),
            "--locus-matching", "full_discovery",
            "--metrics", "NUCLEOTIDE_CLASSIFICATION",
            "--output", str(output_path),
        ],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}\n{result.exception}"

    cli_result = json.loads(output_path.read_text())

    lib_nuc = _nuc(lib_result, "pred")
    cli_nuc = cli_result["pred"]["aggregated"]["NUCLEOTIDE_CLASSIFICATION"]["nucleotide"]

    assert cli_nuc["precision"] == pytest.approx(lib_nuc["precision"], abs=1e-9)
    assert cli_nuc["recall"] == pytest.approx(lib_nuc["recall"], abs=1e-9)
    assert cli_nuc["f1"] == pytest.approx(lib_nuc["f1"], abs=1e-9)
