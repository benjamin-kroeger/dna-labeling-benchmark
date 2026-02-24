import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ...label_definition import LabelConfig

def plot_transition_matrices(transition_failures: dict, label_config: LabelConfig, method_name:str):
    """
    Plots a grid of transition matrices from a dictionary.

    Args:
        transition_failures (dict): A dictionary where keys are titles/identifiers
                                    and values are 2D arrays/lists representing the matrices.
                                    :param label_config:
    """
    num_matrices = len(transition_failures)

    if num_matrices == 0:
        print("No matrices to plot. The dictionary is empty.")
        return

    # 1. Calculate the grid dimensions (roughly square)
    cols = 3
    rows = math.ceil(num_matrices / cols)

    # 2. Create the figure and subplots
    # Adjust figsize so each subplot gets a decent amount of space (e.g., 5x4 inches per subplot)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))

    # Flatten the axes array to easily iterate over it, regardless of dimensions
    if num_matrices == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 3. Iterate through the dictionary and plot each matrix
    for i, (key, matrix) in enumerate(transition_failures.items()):
        ax = axes[i]
        ordered_labels = [label_config.labels[x] for x in sorted(label_config.labels)]
        matrix_df = pd.DataFrame(matrix, columns=ordered_labels, index=ordered_labels)

        # Using seaborn's heatmap for nice color scaling and text annotations inside the cells
        sns.heatmap(matrix_df, annot=True, cmap="Blues", fmt=".2f", ax=ax, cbar=True)

        # Formatting the subplot
        ax.set_title(f"From {label_config.labels[key]}")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Expected label")

    # 4. Hide any empty subplots if the grid isn't perfectly filled
    for j in range(num_matrices, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(method_name, fontsize=14)
    # 5. Adjust layout so titles and labels don't overlap, then return
    plt.tight_layout()
    return fig
