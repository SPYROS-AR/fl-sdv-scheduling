import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENTS = ["dynamic", "fixed_1", "fixed_3", "fixed_5", "fixed_10", "fixed_20"]

META_CONFIG = {
    "dynamic": {
        "label": "Proposed (Dynamic)",
        "color": "#E63946",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    "fixed_1": {
        "label": "Fixed: 1 Epoch",
        "color": "#1D3557",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "fixed_3": {
        "label": "Fixed: 3 Epochs",
        "color": "#457B9D",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "fixed_5": {
        "label": "Fixed: 5 Epochs",
        "color": "#2A9D8F",
        "linestyle": "-.",
        "linewidth": 1.5,
    },
    "fixed_10": {
        "label": "Fixed: 10 Epochs",
        "color": "#E9C46A",
        "linestyle": ":",
        "linewidth": 1.5,
    },
    "fixed_20": {
        "label": "Fixed: 20 Epochs",
        "color": "#F4A261",
        "linestyle": ":",
        "linewidth": 1.5,
    },
}


def setup_plot_style():
    """Configures global matplotlib settings for publication-quality plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
        }
    )


def apply_moving_average(series: pd.Series, window: int = 10) -> pd.Series:
    """Smooths volatile metrics (like loss) for clearer visualization."""
    return series.rolling(window=window, min_periods=1).mean()


def load_metrics_data(job_id: str) -> dict[str, pd.DataFrame]:
    """Loads CSV files for all experiments into a dictionary."""
    loaded_data = {}
    for exp in EXPERIMENTS:
        filename = Path(
            f"metrics_{exp}_{job_id}.csv" if job_id else f"metrics_{exp}.csv"
        )

        directory = Path("final_metrics")
        full_path = directory / filename
        if full_path.exists():
            loaded_data[exp] = pd.read_csv(full_path)
            print(f"Successfully loaded file: {full_path}")
        else:
            print(f"Warning: {full_path} not found. Skipping {exp}.")

    return loaded_data


def plot_learning_curve(data: dict[str, pd.DataFrame], output_dir: Path):
    """Figure 1: Learning Curve (Validation Loss vs Rounds)"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp, df in data.items():
        if "val_loss" in df.columns and "round" in df.columns:
            smoothed_loss = apply_moving_average(df["val_loss"], window=5)
            ax.plot(
                df["round"],
                smoothed_loss,
                label=META_CONFIG[exp]["label"],
                **{k: v for k, v in META_CONFIG[exp].items() if k != "label"},
            )

    ax.set_title("Convergence Profile over Communication Rounds")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Validation Loss (Moving Avg)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_learning_curve.pdf", dpi=300)
    plt.close(fig)


def plot_cost_efficiency(data: dict[str, pd.DataFrame], output_dir: Path):
    """Figure 2: Network Efficiency (Validation Loss vs Total Cost)"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for exp, df in data.items():
        if "total_c_net" in df.columns and "val_loss" in df.columns:
            smoothed_loss = apply_moving_average(df["val_loss"], window=5)
            markevery = max(1, len(df) // 10)

            ax.plot(
                df["total_c_net"],
                smoothed_loss,
                marker="o",
                markevery=markevery,
                markersize=6,
                label=META_CONFIG[exp]["label"],
                **{k: v for k, v in META_CONFIG[exp].items() if k != "label"},
            )

    ax.set_title("Network Cost Efficiency")
    ax.set_xlabel("Cumulative Network Cost (MB)")
    ax.set_ylabel("Validation Loss (Moving Avg)")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "fig2_cost_efficiency.pdf", dpi=300)
    plt.close(fig)


def plot_cumulative_tradeoffs(data: dict[str, pd.DataFrame], output_dir: Path):
    """Figure 3: Cumulative Trade-offs Grid (2x2)"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Cumulative System Metrics Over Time", fontsize=16)

    metrics = [
        ("total_utility", "System Utility", axs[0, 0]),
        ("total_c_net", "Network Overhead (MB)", axs[0, 1]),
        ("total_c_comp", "Compute Interference", axs[1, 0]),
        ("total_c_drift", "Model Divergence (Drift)", axs[1, 1]),
    ]

    for metric_col, y_label, ax in metrics:
        for exp, df in data.items():
            if metric_col in df.columns and "round" in df.columns:
                ax.plot(
                    df["round"],
                    df[metric_col],
                    label=META_CONFIG[exp]["label"],
                    **{k: v for k, v in META_CONFIG[exp].items() if k != "label"},
                )
        ax.set_xlabel("Round")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle="--", alpha=0.7)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout()
    fig.savefig(
        output_dir / "fig3_cumulative_tradeoffs.pdf", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def plot_client_states(data: dict[str, pd.DataFrame], output_dir: Path):
    """Figure 4: Client State Distribution (Stacked Bar)"""
    labels_list = []
    syncs, continues, waits, dropped = [], [], [], []

    for exp, df in data.items():
        labels_list.append(META_CONFIG[exp]["label"])
        syncs.append(df.get("syncs", pd.Series([0])).sum())
        continues.append(df.get("continues", pd.Series([0])).sum())
        waits.append(df.get("waits", pd.Series([0])).sum())
        dropped.append(df.get("dropped", pd.Series([0])).sum())

    x = np.arange(len(labels_list))
    width = 0.6

    syncs_arr = np.array(syncs)
    continues_arr = np.array(continues)
    waits_arr = np.array(waits)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, syncs_arr, width, label="SYNC (Upload)", color="#2A9D8F")
    ax.bar(
        x,
        continues_arr,
        width,
        bottom=syncs_arr,
        label="CONTINUE (Train locally)",
        color="#E9C46A",
    )
    ax.bar(
        x,
        waits_arr,
        width,
        bottom=syncs_arr + continues_arr,
        label="WAIT (Idle)",
        color="#F4A261",
    )
    ax.bar(
        x,
        dropped,
        width,
        bottom=syncs_arr + continues_arr + waits_arr,
        label="DROPPED (Network limit)",
        color="#E63946",
    )

    ax.set_title("Client Operational State Distribution Across Simulation")
    ax.set_ylabel("Total Client Occurrences")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=30, ha="right")
    ax.legend(title="Client State")

    fig.tight_layout()
    fig.savefig(output_dir / "fig4_client_states.pdf", dpi=300)
    plt.close(fig)


def main(job_id: str):
    setup_plot_style()

    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    loaded_data = load_metrics_data(job_id)
    if not loaded_data:
        print("\nNo metrics files found to plot. Run simulation first.")
        return

    print("\nGenerating plots...")
    plot_learning_curve(loaded_data, output_dir)
    plot_cost_efficiency(loaded_data, output_dir)
    plot_cumulative_tradeoffs(loaded_data, output_dir)
    plot_client_states(loaded_data, output_dir)

    print(
        f"Plotting complete! 4 PDF figures generated successfully in the '{output_dir}/' directory."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate simulation evaluation plots."
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default="",
        help="Optional Job ID to append to CSV filenames",
    )
    args = parser.parse_args()

    main(job_id=args.job_id)
