"""Tests for STRUCTURAL_COHERENCE plotting (structural.py).

Focus: the boundary-shift distribution figure must surface its conditioning —
it is only fed by topology-correct transcripts, so the figure carries a
per-method eligibility caption (on the internal-vs-terminal box panel) that
flags methods whose tight-looking offset boxes rest on a tiny eligible set.
The transcript-match classification itself lives in its standalone figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering (conftest also sets this)

import matplotlib.pyplot as plt
import pandas as pd

from dna_segmentation_benchmark.plotting.metrics.structural import (
    plot_boundary_shift_distribution,
    plot_per_transcript_exon_recovery,
    plot_segment_count_delta,
    plot_transcript_match_distribution,
)

_COLUMNS = ["method_name", "metric_group", "scope", "metric_key", "value"]


def _make_df_sc(spec: dict[str, tuple[dict, list]]) -> pd.DataFrame:
    """Build a long-format STRUCTURAL_COHERENCE frame from a compact spec.

    ``spec`` maps method name → (transcript_match_distribution dict,
    boundary_shift_offsets record list).
    """
    rows = []
    for method, (match_dist, offsets) in spec.items():
        rows.append([method, "STRUCTURAL_COHERENCE", "cds", "transcript_match_distribution", match_dist])
        rows.append([method, "STRUCTURAL_COHERENCE", "cds", "boundary_shift_offsets", offsets])
    return pd.DataFrame(rows, columns=_COLUMNS)


def _axis_texts(fig: plt.Figure) -> list[str]:
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def test_boundary_shift_plot_surfaces_eligibility_denominator():
    # good_method: 90/100 transcripts topology-correct → high eligibility.
    # weak_method:  4/100 transcripts topology-correct → must be ⚠-flagged so a
    # tight offset box is not misread as skill.
    df_sc = _make_df_sc(
        {
            "good_method": (
                {
                    "exact": 70,
                    "boundary_shift_internal": 15,
                    "boundary_shift_terminal": 5,
                    "missing_segments": 5,
                    "missed": 5,
                },
                [
                    {"offset": 1, "position": "internal"},
                    {"offset": -2, "position": "internal"},
                    {"offset": 5, "position": "terminal"},
                ],
            ),
            "weak_method": (
                {
                    "exact": 1,
                    "boundary_shift_internal": 2,
                    "boundary_shift_terminal": 1,
                    "missing_segments": 10,
                    "no_overlap": 30,
                    "missed": 56,
                },
                [
                    {"offset": 1, "position": "internal"},
                    {"offset": 3, "position": "terminal"},
                ],
            ),
        }
    )

    fig = plot_boundary_shift_distribution(df_sc, class_name="CDS")

    assert fig is not None
    # Three offset panels (ECDF · signed density · internal/terminal box).
    assert len(fig.axes) >= 3

    texts = _axis_texts(fig)
    # Low-eligibility method carries the ⚠ flag and its a/b transcript count.
    assert any("⚠" in t and "weak_method" in t and "(4/100 tx)" in t for t in texts)
    # High-eligibility method is annotated without the warning flag.
    assert any(
        "good_method" in t and "(90/100 tx)" in t and "⚠" not in t for t in texts
    )
    plt.close(fig)


def test_boundary_shift_plot_returns_none_without_offsets():
    # transcript_match_distribution present but no boundary_shift_offsets rows.
    df_sc = pd.DataFrame(
        [["m", "STRUCTURAL_COHERENCE", "cds", "transcript_match_distribution", {"exact": 3}]],
        columns=_COLUMNS,
    )
    assert plot_boundary_shift_distribution(df_sc, class_name="CDS") is None


def test_transcript_match_distribution_still_renders_after_refactor():
    df_sc = _make_df_sc(
        {"m": ({"exact": 8, "missed": 2}, [{"offset": 1, "position": "internal"}])}
    )
    fig = plot_transcript_match_distribution(df_sc, class_name="CDS")
    assert fig is not None
    assert len(fig.axes) >= 1
    # Per-method total annotated above the single bar.
    assert any("n=10" in t for t in _axis_texts(fig))
    plt.close(fig)


def _row(method: str, metric_key: str, value) -> list:
    return [method, "STRUCTURAL_COHERENCE", "cds", metric_key, value]


def test_segment_count_delta_renders_and_returns_none_when_empty():
    df = pd.DataFrame(
        [
            _row("over", "segment_count_delta", {"mean": 1.5, "std": 0.3}),
            _row("under", "segment_count_delta", {"mean": -2.0, "std": 0.4}),
        ],
        columns=_COLUMNS,
    )
    fig = plot_segment_count_delta(df, class_name="CDS")
    assert fig is not None
    assert len(fig.axes) >= 1
    plt.close(fig)

    # No matching metric_key rows → nothing to plot.
    empty = pd.DataFrame([_row("m", "other_key", {"mean": 0.0})], columns=_COLUMNS)
    assert plot_segment_count_delta(empty, class_name="CDS") is None


def test_per_transcript_exon_recovery_renders_three_panels():
    df = pd.DataFrame(
        [
            _row("m", "exon_recall_per_transcript", [1.0, 0.5, 0.9]),
            _row("m", "exon_precision_per_transcript", [1.0, 0.8, 0.7]),
            _row("m", "false_exon_count_per_transcript", [0, 1, 2]),
        ],
        columns=_COLUMNS,
    )
    fig = plot_per_transcript_exon_recovery(df, class_name="CDS")
    assert fig is not None
    # recall · precision · false-exon panels.
    assert len(fig.axes) >= 3
    plt.close(fig)
