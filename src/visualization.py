import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def _plot_bar_chart(labels, values, colors, title, ylabel, filename):
    """Generic helper to generate and save bar charts."""
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=colors)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_from_files(job_id=""):
    experiments = ["dynamic", "fixed_1", "fixed_3", "fixed_5", "fixed_10", "fixed_20"]
    meta = {
        "dynamic":  {"label": "Proposed (Dynamic)", "color": "blue"},
        "fixed_1":  {"label": "Fixed: 1 Epoch", "color": "red"},
        "fixed_3":  {"label": "Fixed: 3 Epochs", "color": "purple"},
        "fixed_5":  {"label": "Fixed: 5 Epochs", "color": "green"},
        "fixed_10": {"label": "Fixed: 10 Epochs", "color": "orange"},
        "fixed_20": {"label": "Fixed: 20 Epochs", "color": "cyan"}
    }
    
    loaded_data = {}
    for exp in experiments:
        filename = f"metrics_{exp}_{job_id}.csv" if job_id else f"metrics_{exp}.csv"
        if os.path.exists(filename):
            loaded_data[exp] = pd.read_csv(filename)
        else:
            print(f"Warning: {filename} not found. Skipping.")
            
    if not loaded_data:
        print("No metrics files found to plot.")
        return
        
    # Plot 1: Line Chart (Loss vs Cost)
    plt.figure(figsize=(10, 6))
    for exp, df in loaded_data.items():
        if 'total_c_net' in df.columns and 'val_loss' in df.columns:
            plt.plot(df['total_c_net'], df['val_loss'], marker='o', 
                     color=meta[exp]["color"], label=meta[exp]["label"])
            
    plt.title(f"Validation Loss vs Total Network Cost")
    plt.xlabel("Total Network Cost (MBs)")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"results_graph.png")
    plt.close()
    
    # Extract data for bar charts
    labels = [meta[exp]["label"] for exp in loaded_data.keys()]
    colors = [meta[exp]["color"] for exp in loaded_data.keys()]
    
    # Plots 2-4: Bar Charts
    _plot_bar_chart(labels, [df['total_c_comp'].iloc[-1] for df in loaded_data.values()], colors,
                    f"Total Compute Interference", 
                    "Accumulated Interference Score", f"interference_graph.png")
                    
    _plot_bar_chart(labels, [df['total_c_drift'].iloc[-1] for df in loaded_data.values()], colors,
                    f"Accumulated Model Drift over Simulation", 
                    "Total Model Drift", f"drift_graph.png")
                    
    _plot_bar_chart(labels, [df['total_utility'].iloc[-1] for df in loaded_data.values()], colors,
                    f"Global System Utility", 
                    "Cumulative Utility Score", f"utility_graph.png")
                    
    print("Plotting complete! Images saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default="")
    args = parser.parse_args()
    
    plot_from_files(job_id=args.job_id)