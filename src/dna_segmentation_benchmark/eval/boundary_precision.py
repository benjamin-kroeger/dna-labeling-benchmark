import numpy as np


def _compute_boundary_precision_landscape(
    residuals: list[tuple[int, int]], total_gt_count: int, max_range: int = 10
) -> dict:
    """Compute two matrices for boundary evaluation as a JSON-serialisable dict.

    Returns ``{"max_range", "bias_matrix", "reliability_matrix"}`` where both
    matrices are plain nested lists with **rows = 5' dimension** and
    **columns = 3' dimension**.  The axis ticks are fully derivable from
    ``max_range`` (bias: ``-max_range..max_range``; tolerance: ``0..max_range``),
    so they are not stored here — the plotting layer reattaches the tick labels
    and axis names from ``max_range``.

    1. Bias Matrix: 2-D histogram of raw signed errors
       (``-max_range`` to ``+max_range``).
       Shows WHERE the model is shifting (Systemic Bias).
    2. Reliability Matrix: Cumulative Recall
       (``0`` to ``max_range``).
       Shows HOW MUCH standard 'Double Penalty' is reduced by tolerance.
    """
    if not residuals:
        return {
            "max_range": max_range,
            "bias_matrix": np.zeros((2 * max_range + 1, 2 * max_range + 1)).tolist(),
            "reliability_matrix": np.zeros((max_range + 1, max_range + 1)).tolist(),
            "sidedness": {
                "total": 0, "exact": 0, "one_sided": 0, "two_sided": 0,
                "one_sided_fraction": 0.0,
                "clipped_from_bias_matrix": 0,
            },
        }

    res_arr = np.array(residuals)  # Shape: (N, 2) — (5prime, 3prime) tuples

    # --- Matrix 1: Bias Matrix (The 'Scatter' Heatmap) ---
    bins = np.arange(-max_range, max_range + 2) - 0.5
    # np.histogram2d SILENTLY DROPS points outside the bin range, and the dropped
    # fraction is caller-dependent (heavy-tailed callers lose more), which would
    # confound any cross-caller one-sidedness read off this matrix. Clip a COPY so
    # out-of-window pairs saturate into the edge bins instead of vanishing — the
    # bias matrix then holds every matched pair. Clipping is one-sidedness-invariant
    # (it never turns a nonzero edge into zero). Reliability below keeps res_arr raw.
    res_clipped = np.clip(res_arr, -max_range, max_range)
    # np.histogram2d: x → rows (dim 0), y → cols (dim 1)
    # ==> rows = 5', cols = 3'
    bias_values, _, _ = np.histogram2d(x=res_clipped[:, 0], y=res_clipped[:, 1], bins=bins)

    # --- Matrix 2: Reliability Matrix (vectorized broadcast) ---
    tolerance_ticks = np.arange(max_range + 1)
    abs_res = np.abs(res_arr)
    # Broadcast: tolerance thresholds (T, 1) against residual values (N,)
    tol_5 = tolerance_ticks.reshape(-1, 1, 1)  # (T, 1, 1)
    tol_3 = tolerance_ticks.reshape(1, -1, 1)  # (1, T, 1)
    abs_5 = abs_res[:, 0].reshape(1, 1, -1)  # (1, 1, N)
    abs_3 = abs_res[:, 1].reshape(1, 1, -1)  # (1, 1, N)

    reliability_values = np.sum((abs_5 <= tol_5) & (abs_3 <= tol_3), axis=2).astype(float)
    if total_gt_count > 0:
        reliability_values /= total_gt_count

    # --- Scalar: one-sidedness decomposition (computed from RAW residuals, clip-free) ---
    # Cross-caller-comparable summary the bias matrix cannot give once clipped: of the
    # matched pairs with any boundary error, how many miss exactly one edge vs both.
    r5, r3 = res_arr[:, 0], res_arr[:, 1]
    exact = int(np.sum((r5 == 0) & (r3 == 0)))
    two_sided = int(np.sum((r5 != 0) & (r3 != 0)))
    one_sided = int(len(res_arr) - exact - two_sided)
    n_err = one_sided + two_sided
    sidedness = {
        "total": int(len(res_arr)),
        "exact": exact,
        "one_sided": one_sided,
        "two_sided": two_sided,
        "one_sided_fraction": (one_sided / n_err) if n_err else 0.0,
        # How much mass the ±max_range bias matrix dropped for this caller (audit trail).
        "clipped_from_bias_matrix": int(np.sum((np.abs(r5) > max_range) | (np.abs(r3) > max_range))),
    }

    return {
        "max_range": max_range,
        "bias_matrix": bias_values.tolist(),
        "reliability_matrix": reliability_values.tolist(),
        "sidedness": sidedness,
    }
