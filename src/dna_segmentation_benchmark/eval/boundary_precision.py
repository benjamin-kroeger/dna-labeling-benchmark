import pandas as pd
import numpy as np


def _compute_boundary_precision_landscape(
    residuals: list[tuple[int, int]], total_gt_count: int, max_range: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute two matrices for boundary evaluation.

    Both returned DataFrames use **rows = 5' dimension** and
    **columns = 3' dimension**, with their index/columns set to the
    corresponding bin centres so that downstream plotting code can
    render them directly.

    1. Bias Matrix: 2-D histogram of raw signed errors
       (``-max_range`` to ``+max_range``).
       Shows WHERE the model is shifting (Systemic Bias).
    2. Reliability Matrix: Cumulative Recall
       (``0`` to ``max_range``).
       Shows HOW MUCH standard 'Double Penalty' is reduced by tolerance.
    """
    bias_ticks = np.arange(-max_range, max_range + 1)
    tolerance_ticks = np.arange(max_range + 1)

    if not residuals:
        return (
            pd.DataFrame(
                np.zeros((2 * max_range + 1, 2 * max_range + 1)),
                index=pd.Index(bias_ticks, name="5' Residual (Pred − GT)"),
                columns=pd.Index(bias_ticks, name="3' Residual (Pred − GT)"),
            ),
            pd.DataFrame(
                np.zeros((max_range + 1, max_range + 1)),
                index=pd.Index(tolerance_ticks, name="5' Tolerance (bp)"),
                columns=pd.Index(tolerance_ticks, name="3' Tolerance (bp)"),
            ),
        )

    res_arr = np.array(residuals)  # Shape: (N, 2) — (5prime, 3prime) tuples

    # --- Matrix 1: Bias Matrix (The 'Scatter' Heatmap) ---
    bins = np.arange(-max_range, max_range + 2) - 0.5
    # np.histogram2d: x → rows (dim 0), y → cols (dim 1)
    # ==> rows = 5', cols = 3'
    bias_values, _, _ = np.histogram2d(x=res_arr[:, 0], y=res_arr[:, 1], bins=bins)
    bias_matrix = pd.DataFrame(
        bias_values,
        index=pd.Index(bias_ticks, name="5' Residual (Pred − GT)"),
        columns=pd.Index(bias_ticks, name="3' Residual (Pred − GT)"),
    )

    # --- Matrix 2: Reliability Matrix (vectorized broadcast) ---
    abs_res = np.abs(res_arr)
    # Broadcast: tolerance thresholds (T, 1) against residual values (N,)
    tol_5 = tolerance_ticks.reshape(-1, 1, 1)  # (T, 1, 1)
    tol_3 = tolerance_ticks.reshape(1, -1, 1)  # (1, T, 1)
    abs_5 = abs_res[:, 0].reshape(1, 1, -1)  # (1, 1, N)
    abs_3 = abs_res[:, 1].reshape(1, 1, -1)  # (1, 1, N)

    reliability_values = np.sum((abs_5 <= tol_5) & (abs_3 <= tol_3), axis=2).astype(float)
    if total_gt_count > 0:
        reliability_values /= total_gt_count

    reliability_matrix = pd.DataFrame(
        reliability_values,
        index=pd.Index(tolerance_ticks, name="5' Tolerance (bp)"),
        columns=pd.Index(tolerance_ticks, name="3' Tolerance (bp)"),
    )

    return bias_matrix, reliability_matrix
