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

# ponytail: cap resample size — (n_bootstrap, n) at n>100K allocates gigabytes.
# m-out-of-n bootstrap: SE_n = SE_m * sqrt(m/n), so the estimate stays correct.
_BOOT_CAP = 20_000


def _cap_boot(rng: np.random.Generator, n: int, n_bootstrap: int, *arrays):
    """Subsample arrays to _BOOT_CAP; return (arrays, scale, idx)."""
    n_eff = min(n, _BOOT_CAP)
    scale = 1.0
    if n_eff < n:
        sub = rng.choice(n, n_eff, replace=False)
        arrays = tuple(a[sub] if a is not None else None for a in arrays)
        scale = float(np.sqrt(n_eff / n))
    idx = rng.integers(0, n_eff, size=(n_bootstrap, n_eff))
    return arrays, scale, idx


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

        (tp_b, fp_b, fn_b), scale, idx = _cap_boot(rng, n, n_bootstrap, tp_arr, fp_arr, fn_arr)
        boot_tp = tp_b[idx].sum(axis=1)

        boot_prec = None
        boot_rec = None

        if fp_arr is not None:
            boot_fp = fp_b[idx].sum(axis=1)
            denom = boot_tp + boot_fp
            boot_prec = np.where(denom > 0, boot_tp / np.where(denom > 0, denom, 1.0), precision or 0.0)
            precision_stderr = float(np.std(boot_prec)) * scale

        if fn_arr is not None:
            boot_fn = fn_b[idx].sum(axis=1)
            denom = boot_tp + boot_fn
            boot_rec = np.where(denom > 0, boot_tp / np.where(denom > 0, denom, 1.0), recall or 0.0)
            recall_stderr = float(np.std(boot_rec)) * scale

        if boot_prec is not None and boot_rec is not None:
            denom_f1 = boot_prec + boot_rec
            boot_f1 = np.where(
                denom_f1 > 0,
                2 * boot_prec * boot_rec / np.where(denom_f1 > 0, denom_f1, 1.0),
                0.0,
            )
            f1_stderr = float(np.std(boot_f1)) * scale

    result = {
        "precision": precision,
        "recall": recall,
        "precision_stderr": precision_stderr,
        "recall_stderr": recall_stderr,
    }
    if precision is not None and recall is not None:
        result["f1"] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    if f1_stderr is not None:
        result["f1_stderr"] = f1_stderr
    return result


def _macro_means(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> tuple:
    """Per-sequence precision/recall/F1 averaged along the last axis.

    Each ``(tp, fp, fn)`` triple is one sequence.  Unlike the micro path (pool
    counts, *then* divide), this computes a ratio per sequence and takes the
    mean, so every sequence carries equal weight regardless of length.  A
    sequence whose denominator is undefined for a given ratio is skipped for
    that ratio only; the result is ``NaN`` when every sequence is undefined.

    Accepts arrays of any leading shape (1-D for a point estimate, ``(B, n)``
    for a bootstrap batch) and reduces over the last axis.
    """
    p_den = tp + fp
    r_den = tp + fn
    f_den = 2.0 * tp + fp + fn

    # For count data: tp=0 whenever p_den=0, so ratio = 0/1 = 0 for invalid
    # sequences — they contribute 0 to the sum without explicit NaN masking.
    p_s = (tp / np.maximum(p_den, 1)).sum(axis=-1)
    r_s = (tp / np.maximum(r_den, 1)).sum(axis=-1)
    f_s = (2.0 * tp / np.maximum(f_den, 1)).sum(axis=-1)

    p_n = (p_den > 0).sum(axis=-1)
    r_n = (r_den > 0).sum(axis=-1)
    f_n = (f_den > 0).sum(axis=-1)

    nan = np.nan
    return (
        np.where(p_n > 0, p_s / np.maximum(p_n, 1), nan),
        np.where(r_n > 0, r_s / np.maximum(r_n, 1), nan),
        np.where(f_n > 0, f_s / np.maximum(f_n, 1), nan),
    )


def _compute_macro_statistics(counts: list, n_bootstrap: int = 1000) -> dict:
    """Macro (per-sequence, equal-weight) precision/recall/F1 with bootstrap SE.

    Mirrors :func:`_compute_summary_statistics` (same seed, ``n >= 2`` guard)
    but averages per-sequence ratios instead of pooling counts.  Returns only
    the ``*_macro`` keys; standard-error keys are added when bootstrappable.
    """
    tp = np.array([c.tp for c in counts], dtype=float)
    fp = np.array([c.fp for c in counts], dtype=float)
    fn = np.array([c.fn for c in counts], dtype=float)
    n = len(tp)

    p_macro, r_macro, f_macro = _macro_means(tp, fp, fn)
    result: dict = {
        "precision_macro": None if np.isnan(p_macro) else float(p_macro),
        "recall_macro": None if np.isnan(r_macro) else float(r_macro),
        "f1_macro": None if np.isnan(f_macro) else float(f_macro),
    }

    if n >= 2:
        rng = np.random.default_rng(42)
        (tp_b, fp_b, fn_b), scale, idx = _cap_boot(rng, n, n_bootstrap, tp, fp, fn)
        boot_p, boot_r, boot_f = _macro_means(tp_b[idx], fp_b[idx], fn_b[idx])
        result["precision_macro_stderr"] = float(np.nanstd(boot_p)) * scale
        result["recall_macro_stderr"] = float(np.nanstd(boot_r)) * scale
        result["f1_macro_stderr"] = float(np.nanstd(boot_f)) * scale

    return result


def _bootstrap_ratio_stderr(numerator: list, denominator: list, n_bootstrap: int = 1000) -> float | None:
    """Standard error of a pooled ratio ``sum(numerator) / sum(denominator)``.

    Resamples sequences with replacement (each element of ``numerator`` /
    ``denominator`` is one sequence) to estimate the SE of the micro-averaged
    ratio, mirroring the bootstrap used for precision/recall in
    :func:`_compute_summary_statistics` (same seed and ``n >= 2`` guard).
    Returns ``None`` when there are fewer than two sequences or no opportunities
    (empty denominator).
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    n = len(num)
    if n < 2 or den.sum() == 0:
        return None
    rng = np.random.default_rng(42)
    (num_b, den_b), scale, idx = _cap_boot(rng, n, n_bootstrap, num, den)
    boot_num = num_b[idx].sum(axis=1)
    boot_den = den_b[idx].sum(axis=1)
    ratios = np.where(boot_den > 0, boot_num / np.where(boot_den > 0, boot_den, 1.0), 0.0)
    return float(np.std(ratios)) * scale


def _compute_distribution_stats(values: list) -> dict:
    """Compute MAE, RMSE, Mean for a list of values."""
    if not values:
        return {"count": 0, "mean": 0.0, "mae": 0.0, "rmse": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(values, dtype=float)

    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "mae": float(np.mean(np.abs(arr))),
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
    # Macro (per-sequence, equal-weight) siblings — populated only when the
    # caller asks for them (metrics whose per-sequence unit count varies).
    precision_macro: float | None = None
    recall_macro: float | None = None
    f1_macro: float | None = None
    precision_macro_stderr: float | None = None
    recall_macro_stderr: float | None = None
    f1_macro_stderr: float | None = None

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
        # Macro keys appear only when computed, so chain/all-or-nothing tiers
        # (where macro == micro) never carry a redundant duplicate.
        for key in (
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_macro_stderr",
            "recall_macro_stderr",
            "f1_macro_stderr",
        ):
            if getattr(self, key) is not None:
                result[key] = getattr(self, key)
        return result


def summarise_counts(per_sequence: list[Counts], n_bootstrap: int = 1000, include_macro: bool = False) -> Stat:
    """Micro-average a list of per-sequence :class:`Counts` into a :class:`Stat`.

    Delegates the precision/recall + bootstrap-standard-error computation to
    :func:`_compute_summary_statistics` so the numbers are identical to the
    pre-prototype path.

    When ``include_macro`` is set, the per-sequence (equal-weight) macro
    precision/recall/F1 are computed alongside and stored in the ``*_macro``
    fields.  Only metrics whose per-sequence unit count varies (nucleotide,
    region discovery) request this; for all-or-nothing chain tiers macro equals
    micro, so they leave it off and the redundant keys never appear.
    """
    counts = list(per_sequence)
    raw = _compute_summary_statistics(
        tp=[c.tp for c in counts],
        fn=[c.fn for c in counts],
        fp=[c.fp for c in counts],
        n_bootstrap=n_bootstrap,
    )
    macro = _compute_macro_statistics(counts, n_bootstrap=n_bootstrap) if include_macro else {}
    return Stat(
        precision=raw["precision"],
        recall=raw["recall"],
        precision_stderr=raw["precision_stderr"],
        recall_stderr=raw["recall_stderr"],
        f1=raw.get("f1"),
        f1_stderr=raw.get("f1_stderr"),
        precision_macro=macro.get("precision_macro"),
        recall_macro=macro.get("recall_macro"),
        f1_macro=macro.get("f1_macro"),
        precision_macro_stderr=macro.get("precision_macro_stderr"),
        recall_macro_stderr=macro.get("recall_macro_stderr"),
        f1_macro_stderr=macro.get("f1_macro_stderr"),
    )
