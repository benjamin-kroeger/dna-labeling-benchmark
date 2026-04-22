"""Plotting functions for state-transition analysis.

Provides two plot types:

1. **GT Transition Confusion Matrices** – one heatmap per source label showing
   where GT transitioned vs what the predictor did.
2. **False Transition Faceted Bar Charts** – one subplot per GT label showing
   grouped bars of transition types coloured by method.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ...label_definition import LabelConfig


def plot_transition_matrices(
        transition_failures: dict,
        label_config: LabelConfig,
        method_name: str,
) -> plt.Figure | None:
    """Plot a grid of GT-transition confusion matrices.

    Parameters
    ----------
    transition_failures : dict[int, np.ndarray]
        Per source-label confusion matrix (rows = GT target, cols = pred target).
    label_config : LabelConfig
        Resolves label IDs to human-readable names.
    method_name : str
        Method identifier used in the figure title.

    Returns
    -------
    Figure or None
        Returns ``None`` when *transition_failures* is empty.
    """
    num_matrices = len(transition_failures)
    if num_matrices == 0:
        return None

    cols = min(3, num_matrices)
    rows = math.ceil(num_matrices / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    if num_matrices == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    ordered_labels = [label_config.labels[x] for x in sorted(label_config.labels)]

    for i, (key, matrix) in enumerate(transition_failures.items()):
        ax = axes[i]
        matrix_df = pd.DataFrame(matrix, columns=ordered_labels, index=ordered_labels)

        sns.heatmap(matrix_df, annot=True, cmap="Blues", fmt="d", ax=ax, cbar=True)

        ax.set_title(f"From {label_config.labels[key]}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Ground truth label")

    for j in range(num_matrices, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{method_name} – GT Transition Confusion", fontsize=14)
    plt.tight_layout()
    return fig


def plot_false_transitions(
        per_method_data: dict[str, dict],
        label_config: LabelConfig,
) -> plt.Figure | None:
    """Multi-method false transition comparison using faceted subplots.

    One subplot per GT label, each showing grouped horizontal bars of
    transition-type counts coloured by method.  Clean and scannable at
    first glance.

    Parameters
    ----------
    per_method_data : dict[str, dict]
        Outer key = method name, inner dict must contain ``"matrices"``
        and ``"stable_position_counts"`` as produced by the evaluation pipeline.
    label_config : LabelConfig
        Resolves label IDs to human-readable names.

    Returns
    -------
    Figure or None
        Returns ``None`` when every method has zero false transitions.
    """
    if not per_method_data:
        return None

    # Resolve labels
    first_method = next(iter(per_method_data.values()))
    if "matrices" not in first_method:
        return None  # No false transition data available
    label_ids = sorted(first_method["matrices"].keys())
    ordered_labels = [label_config.labels[lid] for lid in label_ids]
    method_names = list(per_method_data.keys())

    # ---- Build long-format DataFrame ------------------------------------
    rows = []
    any_false = False

    for method_name, method_data in per_method_data.items():
        matrices = method_data["matrices"]

        for row_idx, label_id in enumerate(label_ids):
            target_vec = matrices[label_id]
            gt_label = label_config.labels[label_id]

            # Late catch-up (diagonal)
            catch_up = int(target_vec[row_idx])
            if catch_up > 0:
                any_false = True
            rows.append({
                "GT label": gt_label,
                "Type": "Late catch-up",
                "Count": catch_up,
                "Method": method_name,
            })

            # Spurious (off-diagonal)
            for col_idx, col_lid in enumerate(label_ids):
                if col_idx != row_idx:
                    count = int(target_vec[col_idx])
                    if count > 0:
                        any_false = True
                    target_name = label_config.labels[col_lid]
                    rows.append({
                        "GT label": gt_label,
                        "Type": f"Spurious → {target_name}",
                        "Count": count,
                        "Method": method_name,
                    })

    if not any_false:
        return None

    df = pd.DataFrame(rows)

    # Order types: "Late catch-up" first, then spurious targets
    type_order = ["Late catch-up"] + [
        f"Spurious → {label_config.labels[lid]}" for lid in label_ids
    ]
    # Keep only types present in data
    type_order = [t for t in type_order if t in df["Type"].values]

    num_labels = len(ordered_labels)
    num_types = len(type_order)

    # ---- Create faceted figure ------------------------------------------
    fig, axes = plt.subplots(
        1, num_labels,
        figsize=(6 * num_labels, max(4, num_types * 1.0 + 2)),
        sharey=True,
    )

    if num_labels == 1:
        axes = [axes]

    palette = sns.color_palette("tab10", n_colors=len(method_names))
    method_color_map = dict(zip(method_names, palette))

    for ax_idx, gt_label in enumerate(ordered_labels):
        ax = axes[ax_idx]
        df_label = df[df["GT label"] == gt_label].copy()

        sns.barplot(
            data=df_label,
            y="Type",
            x="Count",
            hue="Method",
            order=type_order,
            hue_order=method_names,
            palette=method_color_map,
            ax=ax,
            orient="h",
            edgecolor="white",
            linewidth=0.5,
        )

        # Annotate bars with counts
        for container in ax.containers:
            for bar in container:
                width = bar.get_width()
                if width > 0:
                    ax.text(
                        width, bar.get_y() + bar.get_height() / 2,
                        f" {int(width)}",
                        ha="left", va="center", fontsize=9,
                    )

        ax.set_title(f"Inside {gt_label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Count", fontsize=11)
        ax.set_ylabel("" if ax_idx > 0 else "Transition type")

        # Only show legend on the last subplot
        if ax_idx < num_labels - 1:
            ax.get_legend().remove()
        else:
            ax.legend(title="Method", fontsize=9, title_fontsize=10)

    fig.suptitle("False Transitions at GT-stable Positions", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig
