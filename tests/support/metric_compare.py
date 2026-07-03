"""Strict deep-equality for benchmark metric payloads.

Every value a fixture spells out is compared exactly: dict key sets must be
identical (no missing key, no extra key), sequences compare element-wise, and
floats compare within a tight tolerance. :class:`Counts` dataclasses are
normalised to their ``{tp, fp, fn, tn}`` dict so fixtures can be written as
plain dicts. There is **no** soft-skipping — if the computed output gains or
drops a key, or any number moves, the comparison fails. That is the point: the
old permissive comparator (partial count bundles, un-checked ``*_stderr`` /
``*_macro`` / landscape keys, silent scope unwrapping) hid exactly those
regressions, so it was deleted rather than kept.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd

# Deterministic output (seeded bootstrap, integer counts, exact float ratios)
# lets us compare tightly; this is far tighter than a "close enough" check.
_REL_TOL = 1e-9
_ABS_TOL = 1e-12


def _normalise(value):
    """Dataclasses (``Counts``) -> dict so fixtures can be written as dicts."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def assert_metric_value_equal(expected, computed, key_name: str):
    """Assert *computed* equals *expected* exactly, recursing into containers."""
    expected = _normalise(expected)
    computed = _normalise(computed)

    if isinstance(expected, dict):
        assert isinstance(computed, dict), f"{key_name}: expected dict, got {type(computed).__name__}"
        assert set(expected) == set(computed), (
            f"{key_name}: key-set mismatch — "
            f"missing {set(expected) - set(computed)}, extra {set(computed) - set(expected)}"
        )
        for sub_key in expected:
            assert_metric_value_equal(expected[sub_key], computed[sub_key], f"{key_name}.{sub_key}")
    elif isinstance(expected, np.ndarray) or isinstance(computed, np.ndarray):
        np.testing.assert_allclose(
            np.asarray(computed, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=_REL_TOL,
            atol=_ABS_TOL,
            err_msg=f"{key_name}: array mismatch",
        )
    elif isinstance(expected, pd.DataFrame):
        assert isinstance(computed, pd.DataFrame) and computed.equals(expected), f"{key_name}: DataFrame mismatch"
    elif isinstance(expected, pd.Series):
        assert isinstance(computed, pd.Series) and computed.equals(expected), f"{key_name}: Series mismatch"
    elif isinstance(expected, (list, tuple)):
        assert isinstance(computed, (list, tuple)), f"{key_name}: expected sequence, got {type(computed).__name__}"
        assert len(expected) == len(computed), (
            f"{key_name}: length mismatch — expected {len(expected)}, got {len(computed)}"
        )
        for i, (exp_item, comp_item) in enumerate(zip(expected, computed)):
            assert_metric_value_equal(exp_item, comp_item, f"{key_name}[{i}]")
    elif isinstance(expected, bool) or isinstance(computed, bool):
        assert expected == computed, f"{key_name}: expected {expected!r}, got {computed!r}"
    elif isinstance(expected, float) or isinstance(computed, float):
        assert math.isclose(expected, computed, rel_tol=_REL_TOL, abs_tol=_ABS_TOL), (
            f"{key_name}: expected {expected}, got {computed}"
        )
    elif expected is None:
        assert computed is None, f"{key_name}: expected None, got {computed!r}"
    else:
        assert expected == computed, f"{key_name}: expected {expected!r}, got {computed!r}"
