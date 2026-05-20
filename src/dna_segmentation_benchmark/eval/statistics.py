"""Summary statistics shared by the cross-sequence aggregator.

Pure numeric helpers: precision/recall (with bootstrap standard errors) and
distribution summaries (MAE/RMSE/mean).  They have no knowledge of metric
groups or result layout — that lives in :mod:`aggregation`.
"""

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
