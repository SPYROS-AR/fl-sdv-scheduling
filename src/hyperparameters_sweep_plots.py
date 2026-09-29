import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def identify_pareto_frontier(df, x_col, y_col, maximize_x=False, maximize_y=True):
    """
    Identifies the Pareto frontier of a dataset.
    Handles combinations of maximizing or minimizing X and Y.
    """
    sorted_df = df.sort_values(by=x_col, ascending=not maximize_x)
    pareto_front = []
    
    if maximize_y:
        current_best_y = -np.inf
    else:
        current_best_y = np.inf
        
    for index, row in sorted_df.iterrows():
        y_val = row[y_col]
        if maximize_y:
            if y_val > current_best_y:
                pareto_front.append(row)
                current_best_y = y_val
        else:
            if y_val < current_best_y:
                pareto_front.append(row)
                current_best_y = y_val
                
    return pd.DataFrame(pareto_front)

# ==========================================
# Load the Dataset
# ==========================================
df = pd.read_csv('final_sweep_rankings.csv')

# ==========================================
# Setup Plot Style for IEEE Paper
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# ==========================================
# Plot 1 (For Paper): Utility vs Compute Interference
# ==========================================
fig, ax = plt.subplots(figsize=(6, 4))
scatter1 = ax.scatter(df['interference'], df['utility'], c=df['delta'], cmap='viridis', 
                      alpha=0.6, edgecolors='w', linewidth=0.5, label='Trial Runs')
cbar1 = plt.colorbar(scatter1, ax=ax)
cbar1.set_label(r'Learning Gain Weight ($\delta$)')

pareto1 = identify_pareto_frontier(df, 'interference', 'utility', maximize_x=False, maximize_y=True)
ax.plot(pareto1['interference'], pareto1['utility'], color='red', linestyle='--', 
        linewidth=2, marker='o', markersize=6, label='Pareto Frontier')

ax.set_xlabel('Compute Interference')
ax.set_ylabel('System Utility')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('utility_vs_interference.pdf', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# Plot 2 (For Paper): Model Drift vs Network Cost
# ==========================================
fig, ax = plt.subplots(figsize=(6, 4))
scatter2 = ax.scatter(df['network_mbs'], df['drift'], c=df['utility'], cmap='viridis', 
                      alpha=0.6, edgecolors='w', linewidth=0.5, label='Trial Runs')
cbar2 = plt.colorbar(scatter2, ax=ax)
cbar2.set_label('System Utility')

# Minimize Network Cost (X) and Minimize Model Drift (Y)
pareto2 = identify_pareto_frontier(df, 'network_mbs', 'drift', maximize_x=False, maximize_y=False)
ax.plot(pareto2['network_mbs'], pareto2['drift'], color='red', linestyle='--', 
        linewidth=2, marker='o', markersize=6, label='Pareto Frontier')

ax.set_xlabel('Network Cost (MB/s)')
ax.set_ylabel('Model Drift (Divergence)')
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig('drift_vs_network.pdf', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# Plot 3 (Proof): Utility vs Network Cost
# ==========================================
fig, ax = plt.subplots(figsize=(6, 4))
scatter3 = ax.scatter(df['network_mbs'], df['utility'], c=df['delta'], cmap='viridis', 
                      alpha=0.6, edgecolors='w', linewidth=0.5, label='Trial Runs')
cbar3 = plt.colorbar(scatter3, ax=ax)
cbar3.set_label(r'Learning Gain Weight ($\delta$)')

pareto3 = identify_pareto_frontier(df, 'network_mbs', 'utility', maximize_x=False, maximize_y=True)
ax.plot(pareto3['network_mbs'], pareto3['utility'], color='red', linestyle='--', 
        linewidth=2, marker='o', markersize=6, label='Pareto Frontier')

ax.set_xlabel('Network Cost (MB/s)')
ax.set_ylabel('System Utility')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('utility_vs_network.pdf', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# Plot 4 (Proof): Utility vs Model Drift
# ==========================================
fig, ax = plt.subplots(figsize=(6, 4))
scatter4 = ax.scatter(df['drift'], df['utility'], c=df['interference'], cmap='magma', 
                      alpha=0.6, edgecolors='w', linewidth=0.5, label='Trial Runs')
cbar4 = plt.colorbar(scatter4, ax=ax)
cbar4.set_label('Compute Interference')

pareto4 = identify_pareto_frontier(df, 'drift', 'utility', maximize_x=False, maximize_y=True)
ax.plot(pareto4['drift'], pareto4['utility'], color='red', linestyle='--', 
        linewidth=2, marker='o', markersize=6, label='Pareto Frontier')

ax.set_xlabel('Model Drift (Divergence)')
ax.set_ylabel('System Utility')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('utility_vs_drift.pdf', dpi=300, bbox_inches='tight')
plt.close()