"""Plotting functions for state-transition analysis.

Provides two plot types:

1. **GT Transition Confusion Matrices** – one heatmap per source label showing
   where GT transitioned vs what the predictor did at that exact site.
2. **False Transition Faceted Bar Charts** – one subplot per GT label showing
   grouped bars of transition types coloured by method.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ...label_definition import LabelConfig


def plot_transition_matrices(
        transition_matrices: dict,
        label_config: LabelConfig,
        method_name: str,
) -> plt.Figure | None:
    """Plot GT-transition confusion matrices, one heatmap per source label.

    Parameters
    ----------
    transition_matrices : dict[int, np.ndarray]
        Per GT source label, a ``(L, L)`` matrix: rows = GT target,
        cols = predicted target; only source-matched transitions counted.
    label_config : LabelConfig
        Resolves label IDs to human-readable names.
    method_name : str
        Method identifier used in the figure title.

    Returns
    -------
    Figure or None
        Returns ``None`` when all matrices are empty.
    """

    if not transition_matrices or not any(np.any(m) for m in transition_matrices.values()):
        return None

    label_ids = sorted(label_config.labels.keys())
    ordered_labels = [label_config.labels[lid] for lid in label_ids]
    num_labels = len(label_ids)

    fig, axes = plt.subplots(1, num_labels, figsize=(5 * num_labels, 4))
    if num_labels == 1:
        axes = [axes]

    for ax_idx, src_id in enumerate(label_ids):
        src_name = label_config.labels[src_id]
        matrix = transition_matrices.get(src_id, np.zeros((num_labels, num_labels), dtype=np.int64))
        ax = axes[ax_idx]
        matrix_df = pd.DataFrame(matrix, columns=ordered_labels, index=ordered_labels)
        sns.heatmap(matrix_df, annot=True, cmap="Blues", fmt="d", ax=ax, cbar=True)
        ax.set_xlabel("Predicted target")
        ax.set_ylabel("GT target")
        ax.set_title(f"Source: {src_name}", fontsize=12)

    fig.suptitle(f"{method_name} – GT Transition Confusion (source-matched)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_false_transitions(
        per_method_data: dict[str, dict],
        label_config: LabelConfig,
) -> plt.Figure | None:
    """Multi-method false transition comparison using faceted subplots.

    One subplot per GT-stable label, each showing grouped horizontal bars of
    transition-type counts coloured by method.  Categories are ground-truth
    context-aware:

    * **Late catch-up** – pred was in the *previous* GT state and transitions
      to the current one (``pred_src == prev_GT``, ``pred_tgt == curr_GT``).
    * **Premature → X** – pred leaves the current GT state for the *next* GT
      state X too early (``pred_src == curr_GT``, ``pred_tgt == next_GT``).
    * **Spurious → X** – all other false pred transitions.

    Parameters
    ----------
    per_method_data : dict[str, dict]
        Outer key = method name, inner dict must contain ``"late_catchup"``,
        ``"premature"``, ``"spurious"``, and ``"stable_position_counts"`` as
        produced by the evaluation pipeline.
    label_config : LabelConfig
        Resolves label IDs to human-readable names.

    Returns
    -------
    Figure or None
        Returns ``None`` when every method has zero false transitions.
    """
    if not per_method_data:
        return None

    first_method = next(iter(per_method_data.values()))
    if "late_catchup" not in first_method:
        return None
    label_ids = sorted(first_method["late_catchup"].keys())
    ordered_labels = [label_config.labels[lid] for lid in label_ids]
    method_names = list(per_method_data.keys())

    # ---- Build long-format DataFrame ------------------------------------
    rows = []
    any_false = False

    for method_name, method_data in per_method_data.items():
        late_catchup = method_data["late_catchup"]
        premature    = method_data["premature"]
        spurious     = method_data["spurious"]

        for label_id in label_ids:
            gt_label = label_config.labels[label_id]

            lc_count = int(late_catchup[label_id].sum())
            if lc_count > 0:
                any_false = True
            rows.append({"GT label": gt_label, "Type": "Late catch-up", "Count": lc_count, "Method": method_name})

            for col_idx, col_lid in enumerate(label_ids):
                target_name = label_config.labels[col_lid]

                pm_count = int(premature[label_id][:, col_idx].sum())
                if pm_count > 0:
                    any_false = True
                rows.append({"GT label": gt_label, "Type": f"Premature → {target_name}", "Count": pm_count, "Method": method_name})

                sp_count = int(spurious[label_id][:, col_idx].sum())
                if sp_count > 0:
                    any_false = True
                rows.append({"GT label": gt_label, "Type": f"Spurious → {target_name}", "Count": sp_count, "Method": method_name})

    if not any_false:
        return None

    df = pd.DataFrame(rows)

    type_order = ["Late catch-up"]
    type_order.extend([f"Premature → {label_config.labels[lid]}" for lid in label_ids])
    type_order.extend([f"Spurious → {label_config.labels[lid]}" for lid in label_ids])
    type_order = [t for t in type_order if t in df["Type"].values]

    num_labels = len(ordered_labels)
    num_types = len(type_order)

    # ---- Create faceted figure ------------------------------------------
    fig, axes = plt.subplots(
        1,
        num_labels,
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
                        width,
                        bar.get_y() + bar.get_height() / 2,
                        f" {int(width)}",
                        ha="left",
                        va="center",
                        fontsize=9,
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
