"""Summary statistics shared by the cross-sequence accumulators.

Pure numeric helpers: precision/recall (with bootstrap standard errors) and
distribution summaries (MAE/RMSE/mean).  They have no knowledge of metric
groups or result layout — that lives in :mod:`accumulators`.

This module also hosts the typed value objects (:class:`Counts`,
:class:`Stat`) that the accumulators collect and summarise.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def _compute_summary_statistics(
    tp: list, fn: list = None, fp: list = None, tn: list = None, n_bootstrap: int = 1000
) -> dict:
    """Compute precision and recall with bootstrap standard errors.

    Bootstrap resamples sequences (each element of tp/fp/fn is one sequence)
    to estimate the standard error of the micro-averaged precision and recall.
    f1_stderr is included when both fp and fn are provided.
    """
    precision = None
    recall = None
    precision_stderr = None
    recall_stderr = None
    f1_stderr = None

    if tp is None:
        return {"precision": None, "recall": None, "precision_stderr": None, "recall_stderr": None}

    tp_arr = np.array(tp, dtype=float)
    fp_arr = np.array(fp, dtype=float) if fp is not None else None
    fn_arr = np.array(fn, dtype=float) if fn is not None else None
    n = len(tp_arr)

    if fp_arr is not None:
        total_tp, total_fp = float(tp_arr.sum()), float(fp_arr.sum())
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    if fn_arr is not None:
        total_tp, total_fn = float(tp_arr.sum()), float(fn_arr.sum())
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    if n >= 2:
        rng = np.random.default_rng(42)
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        boot_tp = tp_arr[idx].sum(axis=1)

        boot_prec = None
        boot_rec = None

        if fp_arr is not None:
            boot_fp = fp_arr[idx].sum(axis=1)
            denom = boot_tp + boot_fp
            boot_prec = np.where(denom > 0, boot_tp / np.where(denom > 0, denom, 1.0), precision or 0.0)
            precision_stderr = float(np.std(boot_prec))

        if fn_arr is not None:
            boot_fn = fn_arr[idx].sum(axis=1)
            denom = boot_tp + boot_fn
            boot_rec = np.where(denom > 0, boot_tp / np.where(denom > 0, denom, 1.0), recall or 0.0)
            recall_stderr = float(np.std(boot_rec))

        if boot_prec is not None and boot_rec is not None:
            denom_f1 = boot_prec + boot_rec
            boot_f1 = np.where(
                denom_f1 > 0,
                2 * boot_prec * boot_rec / np.where(denom_f1 > 0, denom_f1, 1.0),
                0.0,
            )
            f1_stderr = float(np.std(boot_f1))

    result = {
        "precision": precision,
        "recall": recall,
        "precision_stderr": precision_stderr,
        "recall_stderr": recall_stderr,
    }
    if f1_stderr is not None:
        result["f1_stderr"] = f1_stderr
    return result


def _compute_distribution_stats(values: list, is_abs: bool = True) -> dict:
    """Compute MAE, RMSE, Mean for a list of values."""
    if not values:
        return {"count": 0, "mean": 0.0, "mae": 0.0, "rmse": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    # Handle tuples if any (though IoU is scalar)
    if values and isinstance(values[0], (tuple, list)):
        flattened = [item for sublist in values for item in sublist]
    else:
        flattened = values

    arr = np.array(flattened, dtype=float)

    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "mae": float(np.mean(np.abs(arr))) if is_abs else float(np.mean(arr)),
        "rmse": float(np.sqrt(np.mean(arr**2))),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Typed accumulators (prototype)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Counts:
    """A single TP/FP/FN/TN confusion bundle for one sequence or comparison.

    Replaces the bare ``{"tp": ..., "fp": ..., "fn": ...}`` dict.  Supports
    ``+`` (and ``sum``) so callers can aggregate counts directly instead of
    threading parallel ``tp``/``fp``/``fn`` lists through a generic merge.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        if not isinstance(other, Counts):
            return NotImplemented
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn)

    def __radd__(self, other):
        # Lets ``sum(counts)`` work (sum starts from int 0).
        if other == 0:
            return self
        return self.__add__(other)


@dataclasses.dataclass(frozen=True)
class Stat:
    """Summarised precision/recall (with bootstrap standard errors).

    Replaces the bare summary dict.  :meth:`to_dict` reproduces the exact key
    set that :func:`_compute_summary_statistics` emits, so it is a drop-in at
    the result boundary.
    """

    precision: float | None = None
    recall: float | None = None
    precision_stderr: float | None = None
    recall_stderr: float | None = None
    f1: float | None = None
    f1_stderr: float | None = None

    def to_dict(self) -> dict:
        result = {
            "precision": self.precision,
            "recall": self.recall,
            "precision_stderr": self.precision_stderr,
            "recall_stderr": self.recall_stderr,
        }
        # Match _compute_summary_statistics: these keys appear only when set.
        if self.f1_stderr is not None:
            result["f1_stderr"] = self.f1_stderr
        if self.f1 is not None:
            result["f1"] = self.f1
        return result


def summarise_counts(per_sequence: list[Counts], n_bootstrap: int = 1000) -> Stat:
    """Micro-average a list of per-sequence :class:`Counts` into a :class:`Stat`.

    Delegates the precision/recall + bootstrap-standard-error computation to
    :func:`_compute_summary_statistics` so the numbers are identical to the
    pre-prototype path.
    """
    counts = list(per_sequence)
    raw = _compute_summary_statistics(
        tp=[c.tp for c in counts],
        fn=[c.fn for c in counts],
        fp=[c.fp for c in counts],
        n_bootstrap=n_bootstrap,
    )
    return Stat(
        precision=raw["precision"],
        recall=raw["recall"],
        precision_stderr=raw["precision_stderr"],
        recall_stderr=raw["recall_stderr"],
        f1_stderr=raw.get("f1_stderr"),
    )
