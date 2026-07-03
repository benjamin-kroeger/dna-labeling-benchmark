"""Tests for per-sequence (macro) precision/recall/F1.

Macro gives every transcript equal weight, so it diverges from the
length-dominated micro score; it is emitted only for metrics whose per-sequence
unit count varies (nucleotide, region discovery), never for the all-or-nothing
chain tiers where macro == micro by construction.
"""

from __future__ import annotations

import math

import pytest

from dna_segmentation_benchmark.eval.statistics import Counts, summarise_counts


def test_macro_differs_from_micro_under_size_heterogeneity():
    # One long, perfect transcript and one short, fully-missed transcript.
    # Micro recall is dominated by the long one; macro weights them equally.
    counts = [Counts(tp=100), Counts(tp=0, fn=10)]
    stat = summarise_counts(counts, include_macro=True)

    assert math.isclose(stat.recall, 100 / 110)          # micro: pooled
    assert math.isclose(stat.recall_macro, 0.5)          # macro: mean(1.0, 0.0)


def test_macro_equals_micro_when_units_uniform():
    # Identical per-transcript balance ⇒ macro and micro coincide.
    counts = [Counts(tp=10, fp=2, fn=3), Counts(tp=10, fp=2, fn=3)]
    stat = summarise_counts(counts, include_macro=True)

    assert math.isclose(stat.precision, stat.precision_macro)
    assert math.isclose(stat.recall, stat.recall_macro)


def test_macro_skips_undefined_denominators():
    # A degenerate all-zero transcript must not poison the macro mean (no 0/0).
    counts = [Counts(tp=5, fp=5), Counts(tp=0, fp=0, fn=0)]
    stat = summarise_counts(counts, include_macro=True)

    # Only the first transcript defines precision (0.5); the empty one is skipped.
    assert math.isclose(stat.precision_macro, 0.5)


def test_macro_absent_unless_requested():
    counts = [Counts(tp=1, fp=1, fn=1), Counts(tp=2)]
    keys = summarise_counts(counts).to_dict()  # include_macro defaults to False

    assert not any(k.endswith("_macro") or k.endswith("_macro_stderr") for k in keys)


def test_macro_has_bootstrap_stderr_with_two_or_more_sequences():
    counts = [Counts(tp=100), Counts(tp=0, fn=10)]
    out = summarise_counts(counts, include_macro=True).to_dict()

    # The two sequences have maximally divergent per-sequence recall (1.0 and
    # 0.0), so a correct bootstrap must show substantial spread — a hardcoded
    # zero or broken formula cannot reproduce this. Value is deterministic under
    # the fixed seed (default_rng(42)); pinned exactly rather than bounded.
    assert out["recall_macro_stderr"] == pytest.approx(0.3556669790689037)
