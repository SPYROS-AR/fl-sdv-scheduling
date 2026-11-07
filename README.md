# Federated Learning in SDVs: Optimizing Communication Frequency

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-FL-792ee5.svg)](https://flower.ai/)

Official PyTorch and Flower (`flwr`) implementation for the paper **"Federated Learning in SDVs: Optimizing Communication Frequency"**.

This repository implements an online, lightweight runtime policy engine for Federated Learning (FL) across fleets of **Software-Defined Vehicles (SDVs)**. Unlike conventional vehicular FL frameworks that execute fixed local epochs and assume isolated onboard hardware, our engine dynamically governs local training depth and wireless upload frequency based on **available safety-critical CPU slack**, **instantaneous V2I link throughput**, and **short-horizon exponential forecasts of local validation loss decay**.

---

## Key Features & Architecture

* **Three-State Vehicular Execution Engine (`WAIT`, `CONTINUE`, `SYNC`):**
  * **`WAIT` ($e_i = 0, p_i = 0$):** Preempts FL execution and yields processor cycles to safety-critical Advanced Driver-Assistance Systems (ADAS) when critical CPU occupancy is high ($U_i^{\text{crit}}(t) > 0.80$) or when poor V2I channel throughput ($B_i(t) < 20.0\text{ Mbps}$) coincides with flat prior validation gain.
  * **`CONTINUE` ($e_i > 0, p_i = 0$):** Trains locally during wireless uplink congestion while persisting model weights on disk across Ray actor instantiations (`.client_cache/`), maintaining a secondary reference buffer to compute exact $L_2$ parameter drift and FedProx proximal gradients relative to the evolving global model.
  * **`SYNC` ($e_i > 0, p_i = 1$):** Uploads parameter updates when link throughput is sufficient or when safety-valve thresholds on accumulated $L_2$ drift ($0.85 D_i^{\text{max}}$) or consecutive local rounds ($K_{\text{max}} = 5$) are reached.
* **Bounded Exponential Gain Predictor (`BoundedExpFit`):** Fits a box-constrained three-parameter exponential decay curve $L(k) \approx L_\infty + a \exp(-bk)$ via `scipy.optimize.curve_fit` to smoothed local validation loss traces, halting local epochs before diminishing returns and client drift set in.
* **GroupNorm ResNet-9 Architecture:** Custom 9-layer Residual Network ($2,441,546$ FP32 parameters, $9.31\text{ MiB}$ payload) replacing standard Batch Normalization with Group Normalization (8 groups) to prevent running-statistic divergence under extreme non-IID data skew.
* **Realistic V2X & Mixed-Criticality Trace Driven:** Simulates 100,000 vehicular mobility and Multi-access Edge Computing (MEC) states incorporating Shannon spectral efficiency, cellular handovers, peak-hour traffic, and dynamic cell buffer limits.

---

## Main Experimental Results

Evaluated over **250 communication rounds** across $N = 10$ vehicles on non-IID CIFAR-10 partitions (`DirichletPartitioner`, $\alpha_{\text{Dir}} \in \{0.1, 0.3, 0.5\}$):

| Policy / Scheme | Test Acc ($\alpha=0.1$) | Test Acc ($\alpha=0.3$) | Test Acc ($\alpha=0.5$) | Uplink Payload (MB) | Buffer Drops | Compute Contention ($\sum c_i U_i^{\text{crit}}$) | Parameter Drift ($\alpha=0.1 / 0.5$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fixed FedAvg ($E = 1$) | 68.98% | 77.09% | 79.33% | 23,272.2 | 567 | 100.1 | 3,661.7 / 3,019.9 |
| Fixed FedAvg ($E = 5$) | **72.51%** | **77.81%** | **78.70%** | 23,272.2 | 567 | 500.3 | 4,319.3 / 2,546.5 |
| Fixed FedAvg ($E = 10$) | 72.12% | 77.32% | 78.05% | 23,272.2 | 567 | 1,000.6 | 4,474.1 / 2,494.4 |
| Fixed FedProx ($E = 5, \mu = 0.01$) | 71.13% | 77.10% | 78.63% | 23,272.2 | 567 | 500.3 | 2,956.9 / 2,016.3 |
| **Dynamic FedAvg (Ours)** | 70.18% | 76.76% | 78.60% | **16,718.7** (-28.2%) | **136** (-76.0%) | **265.6–385.2** | 4,103.2 / 2,598.7 |
| **Dynamic FedProx (Ours)** | **72.02%** | 76.90% | 77.88% | **16,700.1** (-28.2%) | **140** (-75.3%) | **261.9–383.0** | **2,724.2 / 2,149.4** |

---

## Repository Structure

```text
├── src/
│   ├── main.py                        # CLI entry point for running FL simulations
│   ├── scheduler.py                   # Flower simulation launcher, ClientFactory & centralized eval
│   ├── client.py                      # FLVehicle client, 3-state machine & BoundedExpFit predictor
│   ├── server.py                      # DynamicFLStrategy, cell buffer envelope & round CSV logger
│   ├── model.py                       # ResNet9FL architecture with GroupNorm blocks
│   ├── constants.py                   # System, networking, and utility policy hyperparameters
│   ├── data_setup.py                  # External V2X trace repo provisioning & DirichletPartitioner
│   ├── data_processing.py             # Shannon throughput & critical ADAS workload preprocessing
│   ├── setup_data.py                  # One-time dataset initialization script
│   ├── sweep_worker.py                # Worker script for hyperparameter calibration sweeps
│   ├── compare_sweep.py               # Aggregator for ranking hyperparameter sweep trials
│   ├── hyperparameters_sweep_plots.py # Pareto frontier trade-off plotting script
│   └── visualization.py               # Publication figure generator & LaTeX Table II exporter
├── plot_sweeps.py                     # Interactive Plotly parallel-coordinates sweep visualizer
├── submit_final.sh                    # Slurm job array script for full 250-round evaluations
├── submit_sweep.sh                    # Slurm job array script for 100-trial hyperparameter sweeps
└── submit_aggregator.sh               # Slurm script to aggregate sweep outputs
```

---

## Installation & Setup

### 1. Create Environment & Install Dependencies
We recommend Python 3.10+ with a virtual environment or Conda:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Initialize the V2X Network & Workload Trace
Before running simulations, provision and preprocess the 100,000-state vehicular network dataset (`dataset/data/cleaned_network_dataset.csv`):

```bash
python src/setup_data.py
```
*Note: `src/main.py` will also automatically run this step if the cleaned CSV is not found.*

---

## Running Experiments

### Option A: Local Execution (Without Slurm)
You can run individual policies or the full benchmark suite directly on any workstation using `src/main.py`:

```bash
# Run Dynamic FedAvg for 250 rounds under extreme non-IID (alpha = 0.1)
python src/main.py --rounds 250 --dynamic_only --dirichlet-alpha 0.1 --job-id a01

# Run Dynamic FedProx (mu = 0.01)
python src/main.py --rounds 250 --dynamic-fedprox --fedprox-mu 0.01 --dirichlet-alpha 0.1 --job-id a01

# Run Static FedAvg baseline (e.g., E = 5)
python src/main.py --rounds 250 --fixed 5 --dirichlet-alpha 0.1 --job-id a01

# Run Static FedProx baseline (E = 5, mu = 0.01)
python src/main.py --rounds 250 --fedprox-only --fedprox-epochs 5 --fedprox-mu 0.01 --dirichlet-alpha 0.1 --job-id a01
```

To reproduce **all three Dirichlet regimes ($\alpha_{\text{Dir}} \in \{0.1, 0.3, 0.5\}$)** sequentially on a single machine without Slurm:

```bash
for a in 0.1 0.3 0.5; do
  jid="a${a//./}"
  python src/main.py --rounds 250 --dirichlet-alpha $a --job-id $jid
done
```

### Option B: Slurm Cluster Execution (`submit_*.sh`)
The provided `.sh` scripts (`submit_final.sh`, `submit_sweep.sh`, `submit_aggregator.sh`) are **Slurm batch scripts** designed to run parallel job arrays using `$SLURM_ARRAY_TASK_ID`.

> **Important Note Before Running `.sh` Scripts:**
> 1. **Cluster Partition Name:** The scripts are pre-configured with `#SBATCH --partition=yoda` for our institutional HPC cluster. Before submitting via `sbatch`, update `--partition=yoda` (and `--gres=gpu:1` / `--cpus-per-task`) in the `.sh` headers to match your cluster's partition name.
> 2. **Do Not Run Directly with `bash` Unless Setting `$SLURM_ARRAY_TASK_ID`:** Because `submit_final.sh` and `submit_sweep.sh` index into experiments using `$SLURM_ARRAY_TASK_ID`, launching them as plain shell scripts (`bash submit_final.sh`) without Slurm will not loop through the array. To test a specific array index locally via bash, pass the variable inline: `SLURM_ARRAY_TASK_ID=0 bash submit_final.sh`.

```bash
# Run all 18 benchmark configurations (6 policies x 3 Dirichlet alphas) in parallel on Slurm
sbatch submit_final.sh
```

### Hyperparameter Calibration Sweep
To reproduce the 50-round hyperparameter sweep over $(\alpha, \beta, \gamma, \delta)$:

```bash
# 1. Submit the 100-task Slurm array (reads parameter combinations from params.txt)
sbatch submit_sweep.sh

# 2. Aggregate and rank trials into final_sweep_rankings.csv
python src/compare_sweep.py

# 3. Generate Pareto frontier PDFs and interactive parallel coordinates HTML
python src/hyperparameters_sweep_plots.py
python plot_sweeps.py
```

---

## Generating Plots & LaTeX Tables

Round-level telemetry is logged automatically to `metrics_<exp_name>_<job_id>.csv`. Once runs complete, generate all single-regime figures, $1 \times 3$ multi-panel Dirichlet comparison plots, and the LaTeX code for Table II (`plots/table2_results.tex`):

```bash
# Auto-discover all job groups (a01, a03, a05) and export plots + LaTeX table
python src/visualization.py --export-table

# Or plot a specific Dirichlet regime only
python src/visualization.py --job-id a01
```

All high-resolution PDF figures are saved to the `plots/` directory:
* `fig1_dirichlet_comparison.pdf` — Centralized Accuracy vs. Bidirectional Payload (MB)
* `fig1b_uplink_dirichlet_comparison.pdf` — Centralized Accuracy vs. Uplink-Only Payload (MB)
* `fig2_accuracy_vs_wall_time.pdf` — Centralized Accuracy vs. Wall-Clock Time (s)
* `fig3_cumulative_compute_interference.pdf` — Cumulative ADAS Compute Contention
* `fig4_dirichlet_drift_comparison.pdf` — Mean $L_2$ Parameter Drift vs. Round
* `fig5_client_state_distribution.pdf` — Stacked Distribution of Client States (`SYNC`, `CONTINUE`, `WAIT`, `DROPPED`)
* `fig6_utility_dirichlet_comparison.pdf` — Cumulative Fleet Utility vs. Round

## License & Acknowledgments

Developed at the **Department of Informatics and Telematics, Harokopio University of Athens**. The underlying V2X mobility and MEC workload generator builds upon the [`CLOUDNET2026-QoS-Offloading`](https://github.com/icsa-hua/CLOUDNET2026-QoS-Offloading) framework.
