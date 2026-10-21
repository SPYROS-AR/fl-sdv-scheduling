"""Plot Pareto-style views of completed hyperparameter sweeps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def identify_pareto_frontier(df, x_col, y_col, maximize_x=False, maximize_y=True):
    """Identify a simple two-objective Pareto frontier."""
    sorted_df = df.sort_values(by=x_col, ascending=not maximize_x)
    pareto_front = []
    current_best_y = -np.inf if maximize_y else np.inf

    for _, row in sorted_df.iterrows():
        y_val = row[y_col]
        is_better = y_val > current_best_y if maximize_y else y_val < current_best_y
        if is_better:
            pareto_front.append(row)
            current_best_y = y_val
    return pd.DataFrame(pareto_front, columns=df.columns)


def main(input_csv="final_sweep_rankings.csv"):
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Sweep rankings file not found: {path}")
    df = pd.read_csv(path)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    plots = [
        (
            "interference",
            "utility",
            "Compute Interference",
            "System Utility",
            "utility_vs_interference.pdf",
            True,
            "delta",
            "viridis",
        ),
        (
            "network_mbs",
            "drift",
            "Network Payload / Cost (MB)",
            "Model Drift",
            "drift_vs_network.pdf",
            False,
            "utility",
            "viridis",
        ),
        (
            "network_mbs",
            "utility",
            "Network Payload / Cost (MB)",
            "System Utility",
            "utility_vs_network.pdf",
            True,
            "delta",
            "viridis",
        ),
        (
            "drift",
            "utility",
            "Model Drift",
            "System Utility",
            "utility_vs_drift.pdf",
            True,
            "interference",
            "magma",
        ),
    ]

    for x_col, y_col, x_label, y_label, filename, maximize_y, color_col, cmap in plots:
        if x_col not in df or y_col not in df:
            print(f"Skipping {filename}: required columns are missing.")
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(
            df[x_col],
            df[y_col],
            c=df[color_col] if color_col in df else None,
            cmap=cmap,
            alpha=0.6,
            edgecolors="w",
            linewidth=0.5,
            label="Trial Runs",
        )
        if color_col in df:
            colorbar = plt.colorbar(scatter, ax=ax)
            colorbar.set_label(color_col.replace("_", " ").title())
        pareto = identify_pareto_frontier(
            df, x_col, y_col, maximize_x=False, maximize_y=maximize_y
        )
        if not pareto.empty:
            ax.plot(
                pareto[x_col],
                pareto[y_col],
                "r--o",
                linewidth=2,
                markersize=4,
                label="Pareto Frontier",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {filename}")


if __name__ == "__main__":
    main()
