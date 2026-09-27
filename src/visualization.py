import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# Setup Plot Style for Publication
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'lines.linewidth': 2.0
})

def apply_moving_average(series, window=5):
    """Smooths volatile metrics (like loss) for clearer visualization."""
    return series.rolling(window=window, min_periods=1).mean()

def plot_from_files(job_id=""):
    experiments = ["dynamic", "fixed_1", "fixed_3", "fixed_5", "fixed_10", "fixed_20"]
    
    # Meta configuration for consistent coloring and line styles
    meta = {
        "dynamic":  {"label": "Proposed (Dynamic)", "color": "#E63946", "linestyle": "-", "linewidth": 3.0}, # Bold Red
        "fixed_1":  {"label": "Fixed: 1 Epoch", "color": "#1D3557", "linestyle": "--", "linewidth": 1.5},
        "fixed_3":  {"label": "Fixed: 3 Epochs", "color": "#457B9D", "linestyle": "--", "linewidth": 1.5},
        "fixed_5":  {"label": "Fixed: 5 Epochs", "color": "#2A9D8F", "linestyle": "-.", "linewidth": 1.5},
        "fixed_10": {"label": "Fixed: 10 Epochs", "color": "#E9C46A", "linestyle": ":", "linewidth": 1.5},
        "fixed_20": {"label": "Fixed: 20 Epochs", "color": "#F4A261", "linestyle": ":", "linewidth": 1.5}
    }
    
    loaded_data = {}
    for exp in experiments:
        filename = f"metrics_{exp}_{job_id}.csv" if job_id else f"metrics_{exp}.csv"
        if os.path.exists(filename):
            loaded_data[exp] = pd.read_csv(filename)
        else:
            print(f"Warning: {filename} not found. Skipping {exp}.")
            
    if not loaded_data:
        print("No metrics files found to plot. Run simulation first.")
        return

    # =======================================================
    # Figure 1: Learning Curve (Validation Loss vs Rounds)
    # =======================================================
    plt.figure(figsize=(8, 5))
    for exp, df in loaded_data.items():
        if 'val_loss' in df.columns and 'round' in df.columns:
            smoothed_loss = apply_moving_average(df['val_loss'], window=5)
            plt.plot(df['round'], smoothed_loss, 
                     color=meta[exp]["color"], 
                     linestyle=meta[exp]["linestyle"], 
                     linewidth=meta[exp]["linewidth"],
                     label=meta[exp]["label"])
            
    plt.title("Convergence Profile over Communication Rounds")
    plt.xlabel("Communication Round")
    plt.ylabel("Validation Loss (Moving Avg)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("fig1_learning_curve.pdf", dpi=300)
    plt.close()

    # =======================================================
    # Figure 2: Network Efficiency (Validation Loss vs Total Cost)
    # =======================================================
    plt.figure(figsize=(8, 5))
    for exp, df in loaded_data.items():
        if 'total_c_net' in df.columns and 'val_loss' in df.columns:
            smoothed_loss = apply_moving_average(df['val_loss'], window=5)
            # Plot only every Nth marker to reduce clutter
            markevery = max(1, len(df) // 10)
            plt.plot(df['total_c_net'], smoothed_loss, 
                     color=meta[exp]["color"], 
                     linestyle=meta[exp]["linestyle"], 
                     linewidth=meta[exp]["linewidth"],
                     marker='o', markevery=markevery, markersize=6,
                     label=meta[exp]["label"])
            
    plt.title("Network Cost Efficiency")
    plt.xlabel("Cumulative Network Cost (MB)")
    plt.ylabel("Validation Loss (Moving Avg)")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig("fig2_cost_efficiency.pdf", dpi=300)
    plt.close()

    # =======================================================
    # Figure 3: Cumulative Trade-offs Grid (2x2)
    # =======================================================
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Cumulative System Metrics Over Time", fontsize=16)
    
    metrics = [
        ('total_utility', 'System Utility', axs[0, 0]),
        ('total_c_net', 'Network Overhead (MB)', axs[0, 1]),
        ('total_c_comp', 'Compute Interference', axs[1, 0]),
        ('total_c_drift', 'Model Divergence (Drift)', axs[1, 1])
    ]
    
    for metric_col, y_label, ax in metrics:
        for exp, df in loaded_data.items():
            if metric_col in df.columns and 'round' in df.columns:
                ax.plot(df['round'], df[metric_col], 
                        color=meta[exp]["color"], 
                        linestyle=meta[exp]["linestyle"], 
                        linewidth=meta[exp]["linewidth"],
                        label=meta[exp]["label"])
        ax.set_xlabel("Round")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Place one master legend outside the subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.savefig("fig3_cumulative_tradeoffs.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    # =======================================================
    # Figure 4: Client State Distribution (Stacked Bar)
    # =======================================================
    # This visualizes how the dynamic engine shifts states compared to fixed
    state_metrics = ['syncs', 'continues', 'waits', 'dropped']
    labels_list = []
    syncs, continues, waits, dropped = [], [], [], []

    for exp, df in loaded_data.items():
        labels_list.append(meta[exp]["label"])
        # Sum the occurrences across all rounds
        syncs.append(df['syncs'].sum() if 'syncs' in df.columns else 0)
        continues.append(df['continues'].sum() if 'continues' in df.columns else 0)
        waits.append(df['waits'].sum() if 'waits' in df.columns else 0)
        dropped.append(df['dropped'].sum() if 'dropped' in df.columns else 0)

    x = np.arange(len(labels_list))
    width = 0.6

    plt.figure(figsize=(10, 6))
    p1 = plt.bar(x, syncs, width, label='SYNC (Upload)', color='#2A9D8F')
    p2 = plt.bar(x, continues, width, bottom=syncs, label='CONTINUE (Train locally)', color='#E9C46A')
    p3 = plt.bar(x, waits, width, bottom=np.array(syncs)+np.array(continues), label='WAIT (Idle)', color='#F4A261')
    p4 = plt.bar(x, dropped, width, bottom=np.array(syncs)+np.array(continues)+np.array(waits), label='DROPPED (Network limit)', color='#E63946')

    plt.title("Client Operational State Distribution Across Simulation")
    plt.ylabel("Total Client Occurrences")
    plt.xticks(x, labels_list, rotation=30, ha='right')
    plt.legend(title="Client State")
    plt.tight_layout()
    plt.savefig("fig4_client_states.pdf", dpi=300)
    plt.close()

    print("Plotting complete! 4 PDF figures generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default="")
    args = parser.parse_args()
    plot_from_files(job_id=args.job_id)