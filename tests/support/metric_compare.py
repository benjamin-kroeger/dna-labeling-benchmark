"""Deep metric-comparison machinery shared by the benchmark-driver tests.

``assert_metric_value_equal`` strictly compares a computed metric payload against
a hand-authored expectation (keys *and* values), with a few deliberate
accommodations (count bundles, bootstrap ``*_stderr``, the derived boundary
landscape).  ``METRIC_EVAL_DISPATCH`` maps each :class:`EvalMetrics` to the
per-metric checker the parametrized cases route through.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

from dna_segmentation_benchmark.eval.evaluate_predictors import EvalMetrics


# ------------------------------------------------------------------
# Per-metric evaluators
# ------------------------------------------------------------------


def _eval_region_discovery(expected, computed):
    """Strictly verify region-discovery hit counts (exact 1:1)."""
    computed = _default_scope_payload(expected, computed)
    assert_metric_value_equal(expected, computed, "REGION_DISCOVERY")


def _eval_boundary_exactness(expected, computed):
    """Strictly verify boundary-exactness metrics (exact 1:1)."""
    computed = _default_scope_payload(expected, computed)
    assert_metric_value_equal(expected, computed, "BOUNDARY_EXACTNESS")


def _eval_nucleotide_classification(expected, computed):
    """Strictly verify nucleotide-classification metrics (exact 1:1)."""
    computed = _default_scope_payload(expected, computed)
    assert_metric_value_equal(expected, computed, "NUCLEOTIDE_CLASSIFICATION")


def _eval_indel_metrics(expected_indel, computed_indel):
    """Compare boundary-typed INDEL output (``by_boundary`` run-length lists).

    ``by_boundary`` is compared strictly (each bucket holds a list of run
    *lengths*, compared order-insensitively because multiple runs at the same
    boundary/bucket carry no intrinsic ordering).  The opportunity denominators
    (``exon_opportunities``, ``n_gt_segments``, ``n_pred_segments``) are
    compared only when a fixture spells them out, so older fixtures that assert
    only ``by_boundary`` still pass.
    """
    computed_indel = _default_scope_payload(expected_indel, computed_indel)
    assert "by_boundary" in computed_indel, "INDEL output is missing 'by_boundary'"

    for denom_key in ("exon_opportunities", "n_gt_segments", "n_pred_segments"):
        if denom_key in expected_indel:
            assert expected_indel[denom_key] == computed_indel[denom_key], (
                f"INDEL {denom_key} differs: expected {expected_indel[denom_key]}, "
                f"got {computed_indel.get(denom_key)}"
            )

    expected_bb = expected_indel["by_boundary"]
    computed_bb = computed_indel["by_boundary"]
    assert set(expected_bb.keys()) == set(computed_bb.keys()), (
        f"INDEL boundary keys differ: missing {set(expected_bb) - set(computed_bb)}, "
        f"unexpected {set(computed_bb) - set(expected_bb)}"
    )
    for boundary, expected_buckets in expected_bb.items():
        computed_buckets = computed_bb[boundary]
        assert set(expected_buckets.keys()) == set(computed_buckets.keys()), (
            f"INDEL buckets differ for boundary {boundary}: expected "
            f"{set(expected_buckets)}, got {set(computed_buckets)}"
        )
        for bucket, expected_lengths in expected_buckets.items():
            assert sorted(computed_buckets[bucket]) == sorted(expected_lengths), (
                f"INDEL run lengths differ for {boundary}/{bucket}: "
                f"expected {expected_lengths}, got {computed_buckets[bucket]}"
            )


def _eval_phase_drift_metrics(expected_phase_drift, computed_phase_drift):
    if "scopes" in computed_phase_drift and "scopes" not in expected_phase_drift:
        scope_payload = _single_scope_payload(computed_phase_drift)
        computed_phase_drift = {"gt_frames": np.asarray(scope_payload["frames"])}
    _OPTIONAL_PHASE_DRIFT_KEYS = {"boundary_indel_total", "boundary_indel_in_frame", "n_skipped_non_divisible", "n_skipped_short", "n_skipped_no_overlap"}
    unexpected = set(computed_phase_drift.keys()) - set(expected_phase_drift.keys()) - _OPTIONAL_PHASE_DRIFT_KEYS
    assert not unexpected, f"Unexpected keys in computed phase drift: {unexpected}"
    assert set(expected_phase_drift.keys()) <= set(computed_phase_drift.keys()), (
        "The keys for the phase-drift metrics dont match."
    )
    for metric in expected_phase_drift:
        assert (expected_phase_drift[metric] == computed_phase_drift[metric]).all(), (
            "The computed frame assignment does not match the expected frame assignment."
        )


def _flatten_structural_scope(computed):
    """Merge a single-scope STRUCTURAL_COHERENCE payload into a flat dict."""
    if "scopes" in computed:
        scope_payload = _single_scope_payload(computed)
        return {
            **scope_payload,
            **{
                key: value
                for key, value in computed.items()
                if key != "scopes"
            },
        }
    return computed


def _eval_structural_coherence(expected, computed):
    """Strictly verify structural coherence metrics (exact 1:1)."""
    if "scopes" not in expected:
        computed = _flatten_structural_scope(computed)
    assert_metric_value_equal(expected, computed, "STRUCTURAL_COHERENCE")


def _eval_diagnostic_depth(expected, computed):
    """Strictly verify diagnostic depth metrics (exact 1:1)."""
    computed = _default_scope_payload(expected, computed)
    assert_metric_value_equal(expected, computed, "DIAGNOSTIC_DEPTH")


# ------------------------------------------------------------------
# Generic deep comparison
# ------------------------------------------------------------------

_COUNT_KEYS = {"tp", "fp", "fn", "tn"}


def _is_landscape_artifact(value) -> bool:
    """True for aggregate-only derived payloads that can't be hand-authored.

    The aggregated boundary ``fuzzy_metrics`` is the precision-landscape dict
    (``{max_range, bias_matrix, reliability_matrix}``).  Such derived values are
    checked structurally (present, non-None) rather than value-by-value, while
    the single-sequence ``fuzzy_metrics`` dict (raw ``boundary_offsets``) stays
    strictly compared.
    """
    if isinstance(value, dict) and "bias_matrix" in value and "reliability_matrix" in value:
        return True
    if isinstance(value, pd.DataFrame):
        return True
    if isinstance(value, (tuple, list)):
        return any(isinstance(item, pd.DataFrame) for item in value)
    return False


def _as_count_bundle(value):
    """Return ``{tp,fp,fn,tn}`` if *value* is a (possibly partial) count bundle.

    Raw :class:`Counts`, terse fixture dicts like ``{"tp": 1, "fn": 2}``, and
    full ``{tp,fp,fn,tn}`` dicts all normalise to the same four-field form so
    that count *values* are always checked in full, regardless of which subset
    a fixture spelled out.  Returns ``None`` for anything that is not a count
    bundle (e.g. summarised ``{precision, recall, ...}`` dicts).
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict) and value and set(value).issubset(_COUNT_KEYS):
        return {field: int(value.get(field, 0)) for field in ("tp", "fp", "fn", "tn")}
    return None


def assert_metric_value_equal(expected, computed, key_name: str):
    """Strictly compare a metric value 1:1 (keys *and* values).

    Dicts must have identical key sets (no missing, no extra), with two
    deliberate accommodations:

    * **count bundles** (``tp/fp/fn/tn``) are normalised so a fixture may spell
      out a subset, but every one of the four counts is still compared exactly;
    * **``*_stderr`` keys** carry non-deterministic bootstrap values, so they
      are not required in fixtures and only sanity-checked (float ≥ 0 or None).
    """
    # Count bundles: compare all four base counts exactly.
    expected_counts = _as_count_bundle(expected)
    computed_counts = _as_count_bundle(computed)
    if expected_counts is not None or computed_counts is not None:
        assert expected_counts is not None and computed_counts is not None, (
            f"Count-bundle type mismatch for {key_name}: expected {expected!r}, got {computed!r}"
        )
        assert expected_counts == computed_counts, (
            f"Count mismatch for {key_name}: expected {expected_counts}, got {computed_counts}"
        )
        return

    if dataclasses.is_dataclass(expected) and not isinstance(expected, type):
        expected = dataclasses.asdict(expected)
    if dataclasses.is_dataclass(computed) and not isinstance(computed, type):
        computed = dataclasses.asdict(computed)
    if isinstance(expected, dict):
        assert isinstance(computed, dict), f"Expected dict for {key_name}, got {type(computed)}"
        # *_stderr (random bootstrap), *_macro (additive equal-weight siblings)
        # and the derived landscape artifact are soft: not required in
        # fixtures, only sanity-checked.
        def _soft(key, container):
            return (
                isinstance(key, str) and (key.endswith("_stderr") or key.endswith("_macro"))
            ) or _is_landscape_artifact(container.get(key))

        expected_keys = {k for k in expected if not _soft(k, expected)}
        computed_keys = {k for k in computed if not _soft(k, computed)}
        assert expected_keys == computed_keys, (
            f"Key-set mismatch for {key_name}: "
            f"missing {expected_keys - computed_keys}, unexpected {computed_keys - expected_keys}"
        )
        for stderr_key in (k for k in computed if isinstance(k, str) and k.endswith("_stderr")):
            value = computed[stderr_key]
            assert value is None or (isinstance(value, (int, float)) and value >= 0), (
                f"Bootstrap stderr {key_name}.{stderr_key} should be a non-negative float, got {value!r}"
            )
        for soft_key in (k for k in computed if _is_landscape_artifact(computed.get(k))):
            assert computed[soft_key] is not None, f"Missing landscape artifact {key_name}.{soft_key}"
        for sub_key in expected_keys:
            assert_metric_value_equal(expected[sub_key], computed[sub_key], f"{key_name}.{sub_key}")
    elif isinstance(expected, np.ndarray):
        assert isinstance(computed, np.ndarray), f"Expected ndarray for {key_name}, got {type(computed)}"
        assert np.array_equal(expected, computed), (
            f"Array mismatch for {key_name}: expected {expected}, got {computed}"
        )
    elif isinstance(expected, pd.DataFrame):
        assert isinstance(computed, pd.DataFrame), f"Expected DataFrame for {key_name}, got {type(computed)}"
        assert expected.equals(computed), f"DataFrame mismatch for {key_name}"
    elif isinstance(expected, pd.Series):
        assert isinstance(computed, pd.Series), f"Expected Series for {key_name}, got {type(computed)}"
        assert expected.equals(computed), f"Series mismatch for {key_name}"
    elif isinstance(expected, tuple):
        assert isinstance(computed, tuple), f"Expected tuple for {key_name}, got {type(computed)}"
        assert len(computed) == len(expected), (
            f"Tuple length mismatch for {key_name}: expected {len(expected)}, got {len(computed)}"
        )
        for i, (exp_item, comp_item) in enumerate(zip(expected, computed)):
            assert_metric_value_equal(exp_item, comp_item, f"{key_name}[{i}]")
    elif isinstance(expected, list):
        assert isinstance(computed, list), f"Expected list for {key_name}, got {type(computed)}"
        assert len(computed) == len(expected), (
            f"List length mismatch for {key_name}: expected {len(expected)}, got {len(computed)}"
        )
        for i, (exp_item, comp_item) in enumerate(zip(expected, computed)):
            assert_metric_value_equal(exp_item, comp_item, f"{key_name}[{i}]")
    elif expected is None:
        assert computed is None, f"Expected None for {key_name}, got {computed}"
    elif isinstance(expected, float):
        assert math.isclose(expected, computed, abs_tol=0.001, rel_tol=0.011), (
            f"Float mismatch for {key_name}: expected {expected}, got {computed}"
        )
    else:
        assert expected == computed, (
            f"Value mismatch for {key_name}: expected {expected}, got {computed}"
        )


def _default_scope_payload(expected: dict, computed: dict) -> dict:
    """Unwrap a single default scope when legacy flat expectations are used."""
    if "scopes" not in computed or "scopes" in expected:
        return computed
    return _single_scope_payload(computed)


def _single_scope_payload(computed: dict) -> dict:
    """Return the transcript-exon scope when present, otherwise the sole scope."""
    scopes = computed["scopes"]
    if "transcript_exon" in scopes:
        return scopes["transcript_exon"]
    if len(scopes) != 1:
        raise AssertionError(f"Expected a single scope payload, got {list(scopes.keys())}")
    return next(iter(scopes.values()))


# ------------------------------------------------------------------
# Dispatch table
# ------------------------------------------------------------------

METRIC_EVAL_DISPATCH = {
    EvalMetrics.INDEL: _eval_indel_metrics,
    EvalMetrics.REGION_DISCOVERY: _eval_region_discovery,
    EvalMetrics.BOUNDARY_EXACTNESS: _eval_boundary_exactness,
    EvalMetrics.NUCLEOTIDE_CLASSIFICATION: _eval_nucleotide_classification,
    EvalMetrics.PHASE_DRIFT: _eval_phase_drift_metrics,
    EvalMetrics.STRUCTURAL_COHERENCE: _eval_structural_coherence,
    EvalMetrics.DIAGNOSTIC_DEPTH: _eval_diagnostic_depth,
}
