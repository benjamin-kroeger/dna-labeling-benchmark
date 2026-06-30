"""Plotting functions for splice-site junction evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import _save_figure


def plot_splice_site_confusion_matrices(
    per_method_data: dict[str, dict],
    save_path: Path | None = None,
) -> plt.Figure | None:
    """One figure with one 2×2 heatmap per method.

    Rows = donor predicted (hit / miss), columns = acceptor predicted (hit / miss).
    Cell values are the summed counts across all evaluated sequences.
    """
    if not per_method_data:
        return None

    methods = list(per_method_data.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        ss = per_method_data[method]
        matrix = np.array([
            [ss.get("both_correct", 0), ss.get("donor_only", 0)],
            [ss.get("acceptor_only", 0), ss.get("neither", 0)],
        ])
        df_cm = pd.DataFrame(
            matrix,
            index=["donor hit", "donor miss"],
            columns=["acceptor hit", "acceptor miss"],
        )
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
        ax.set_title(method, fontsize=12)
        ax.set_xlabel("Acceptor", fontsize=10)
        ax.set_ylabel("Donor", fontsize=10)

    fig.suptitle("Splice-site junction confusion", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        import logging
        _save_figure(fig, save_path, logger=logging.getLogger(__name__))

    return fig


def plot_splice_site_pr_bar(
    per_method_data: dict[str, dict],
    save_path: Path | None = None,
) -> plt.Figure | None:
    """Grouped bar chart: donor/acceptor precision & recall, one bar per method.

    X-axis = metric (4 metrics), hue = method.
    """
    if not per_method_data:
        return None

    rows = []
    for method, ss in per_method_data.items():
        for metric in ("donor_precision", "donor_recall", "acceptor_precision", "acceptor_recall"):
            rows.append({"Method": method, "Metric": metric, "Score": ss.get(metric, 0.0)})

    df = pd.DataFrame(rows)

    metric_labels = {
        "donor_precision": "Donor\nPrecision",
        "donor_recall": "Donor\nRecall",
        "acceptor_precision": "Acceptor\nPrecision",
        "acceptor_recall": "Acceptor\nRecall",
    }
    df["Metric"] = df["Metric"].map(metric_labels)

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = list(per_method_data.keys())
    sns.barplot(
        data=df,
        x="Metric",
        y="Score",
        hue="Method",
        hue_order=methods,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, label_type="edge", padding=3, fmt="%.3f")

    ax.set_ylim(0, 1.1)
    ax.set_title("Splice-site precision & recall", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(title="Method", fontsize=9)
    fig.tight_layout()

    if save_path is not None:
        import logging
        _save_figure(fig, save_path, logger=logging.getLogger(__name__))

    return fig
