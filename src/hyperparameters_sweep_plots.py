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


def main(input_csv=""):
    search_paths = [
        Path(input_csv) if input_csv else None,
        Path("final_sweep_rankings.csv"),
        Path("final_metrics/final_sweep_rankings.csv"),
    ]
    path = None
    for p in search_paths:
        if p and p.exists():
            path = p
            break
    if not path:
        raise FileNotFoundError(
            "Sweep rankings file not found in root or final_metrics/"
        )

    df = pd.read_csv(path)
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

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
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_dir / filename} and {filename}")

    fig_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    sc1 = ax1.scatter(
        df["interference"],
        df["utility"],
        c=df["delta"],
        cmap="viridis",
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
        label="Sweep Trials",
    )
    cb1 = plt.colorbar(sc1, ax=ax1)
    cb1.set_label(r"Learning Gain Weight ($\delta$)")
    pareto1 = identify_pareto_frontier(
        df, "interference", "utility", maximize_x=False, maximize_y=True
    )
    if not pareto1.empty:
        ax1.plot(
            pareto1["interference"],
            pareto1["utility"],
            "r--o",
            linewidth=2,
            markersize=5,
            label="Pareto Frontier",
        )
    ax1.set_xlabel(r"Compute Interference ($\sum c_i U_i^{\mathrm{crit}}$)")
    ax1.set_ylabel(r"Global System Utility ($\mathcal{U}$)")
    ax1.set_title(
        "(a) Utility vs. Compute Interference", fontsize=11, fontweight="bold"
    )
    ax1.legend(loc="lower right")

    sc2 = ax2.scatter(
        df["network_mbs"],
        df["drift"],
        c=df["utility"],
        cmap="viridis",
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
        label="Sweep Trials",
    )
    cb2 = plt.colorbar(sc2, ax=ax2)
    cb2.set_label("Global Utility")
    pareto2 = identify_pareto_frontier(
        df, "network_mbs", "drift", maximize_x=False, maximize_y=False
    )
    if not pareto2.empty:
        ax2.plot(
            pareto2["network_mbs"],
            pareto2["drift"],
            "r--o",
            linewidth=2,
            markersize=5,
            label="Pareto Frontier",
        )
    ax2.set_xlabel("Network Cost (MB)")
    ax2.set_ylabel(r"Model Drift ($\sum D_i$)")
    ax2.set_title("(b) Model Drift vs. Network Cost", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right")

    fig_comp.tight_layout()
    composite_pdf = output_dir / "pareto_frontier_tradeoffs.pdf"
    fig_comp.savefig(composite_pdf, dpi=300, bbox_inches="tight")
    fig_comp.savefig("pareto_frontier_tradeoffs.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_comp)
    print(f"Saved {composite_pdf} and pareto_frontier_tradeoffs.pdf")


if __name__ == "__main__":
    main()
