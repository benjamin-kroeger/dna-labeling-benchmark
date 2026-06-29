import numpy as np


def _sweep_cluster(items, *, key):
    """O(n log n) interval sweep — group items into overlapping clusters.

    key(item) must return (start, end). Items within each cluster overlap pairwise.
    """
    if not items:
        return []
    sorted_items = sorted(items, key=key)
    clusters, current = [], [sorted_items[0]]
    current_end = key(sorted_items[0])[1]
    for item in sorted_items[1:]:
        start, end = key(item)
        if start <= current_end:
            current.append(item)
            current_end = max(current_end, end)
        else:
            clusters.append(current)
            current, current_end = [item], end
    clusters.append(current)
    return clusters


def get_contiguous_groups(indices: np.ndarray) -> list[np.ndarray]:
    """Split *indices* into sub-arrays of contiguous runs."""
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    if breaks.size == 0:
        return [indices]
    return np.split(indices, breaks)
