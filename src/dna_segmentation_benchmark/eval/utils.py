import numpy as np


def get_contiguous_groups(indices: np.ndarray) -> list[np.ndarray]:
    """Split *indices* into sub-arrays of contiguous runs."""
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    return np.split(indices, breaks)
