import numpy as np


def recursive_merge(target: dict, source: dict) -> dict:
    """Recursively merge *source* into *target*, skipping ``None`` values."""
    for key, source_value in source.items():
        if source_value is None:
            continue

        if key not in target:
            if isinstance(source_value, dict):
                target[key] = {}
                recursive_merge(target[key], source_value)
            elif isinstance(source_value, list):
                target[key] = list(source_value)
            elif isinstance(source_value, np.ndarray):
                target[key] = source_value
            else:
                target[key] = [source_value]
        else:
            target_value = target[key]
            if isinstance(source_value, dict) and isinstance(target_value, dict):
                recursive_merge(target_value, source_value)
            elif isinstance(target_value, list):
                if isinstance(source_value, list):
                    target_value.extend(source_value)
                else:
                    target_value.append(source_value)
            elif isinstance(target_value, np.ndarray):
                target[key] += source_value
            else:
                target[key] = [target_value, source_value]
    return target


def _compute_summary_statistics(tp: list, fn: list = None, fp: list = None, tn: list = None) -> dict:
    """Compute precision and recall from aggregated confusion counts."""
    precision = None
    recall = None
    if tp is not None and fp is not None:
        total_tp = sum(tp)
        total_fp = sum(fp)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    if tp is not None and fn is not None:
        total_tp = sum(tp)
        total_fn = sum(fn)
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

    return {"precision": precision, "recall": recall}


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


def get_contiguous_groups(indices: np.ndarray) -> list[np.ndarray]:
    """Split *indices* into sub-arrays of contiguous runs."""
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    return np.split(indices, breaks)
