# Federated Learning in Software-Defined Vehicles (SDVs): Optimizing Communication Frequency

This repository contains the implementation, experimental evaluation framework, and manuscript for the research paper: **"Federated Learning in SDVs: Optimizing Communication Frequency"** by Spyridon Arvanitis.

## Overview

Deploying Federated Learning (FL) on Software-Defined Vehicles (SDVs) requires balancing wireless transmission overhead, client model drift, and onboard compute resource contention with safety-critical vehicle workloads (such as ADAS perception and trajectory planning). 

Conventional vehicular FL frameworks typically assume dedicated onboard compute and rely on static local training intervals, which can lead to uplink cellular congestion and compute starvation for co-located real-time services. This project implements a runtime heuristic policy engine that dynamically regulates local training epochs and client upload schedules based on available compute slack, instantaneous link throughput, and predicted local learning gain.

### Key Capabilities
- **Three-State Runtime Policy Engine:** Clients transition dynamically among `WAIT` (idle to yield execution to safety tasks or save energy when link is poor and gain is negligible), `CONTINUE` (autonomous local training with persistent local weights), and `SYNC` (parameter upload).
- **Bounded Non-Linear Extrapolation:** Predicts empirical validation loss reduction via bounded exponential curve fitting ($L(e) \approx L_\infty + A e^{-B e}$) before executing local epochs.
- **Physical Parameter & Network Accounting:** Exact FP32 parameter footprint accounting ($9.31$ MiB for ResNet-9) and true vector Euclidean parameter divergence ($\|\mathbf{w}_{\text{local}} - \mathbf{w}_{\text{global}}\|_2$).
- **Algorithmic Baselines & Heterogeneity:** Benchmarks against static epoch baselines ($E \in \{1, 5, 10\}$) and FedProx under symmetric Dirichlet non-IID partitioning ($\alpha_{\text{Dir}} \in \{0.1, 0.3, 0.5\}$).

---

## Repository Structure

```text
.
├── final_metrics/                # Final evaluation metrics CSVs and Pareto sweep rankings
├── metrics/                      # Round-level CSV logs from hyperparameter exploration runs
├── paper.tex                     # Camera-ready IEEE conference LaTeX manuscript
├── plots/                        # Generated publication evaluation figures (PDF)
│   ├── fig1_accuracy_vs_transmitted_data.pdf
│   ├── fig2_accuracy_vs_wall_time.pdf
│   ├── fig3_cumulative_compute_interference.pdf
│   └── fig4_model_drift.pdf
├── scripts/                      # Slurm cluster submission & analysis scripts
│   ├── submit_final.sh           # Main Slurm array job for 250-round baseline suite
│   ├── submit_sweep.sh           # Slurm array script for hyperparameter sweeps
│   ├── submit_aggregator.sh      # Slurm script for parsing sweep outputs
│   ├── generate_params.py        # Generates hyperparameter combinations (params.txt)
│   └── plot_sweeps.py            # Generates interactive Plotly parallel coordinates
├── src/
│   ├── client.py                 # FLVehicle client implementation (3-state dynamic & FedProx)
│   ├── server.py                 # DynamicFLStrategy and round-level CSV metrics logger
│   ├── scheduler.py              # Flower simulation orchestrator and ClientFactory caching
│   ├── model.py                  # ResNet-9 with Group Normalization (CIFAR-10 & GTSRB)
│   ├── constants.py              # System, networking, and utility policy hyperparameters
│   ├── data_setup.py             # External network dataset & Dirichlet non-IID partitioning
│   ├── data_processing.py        # Network dataset cleaning and vehicular load modeling
│   ├── setup_data.py             # Network dataset initialization and preprocessing
│   ├── main.py                   # Main CLI entry point for local and cluster simulations
│   ├── sweep_worker.py           # Single-worker runner for hyperparameter sweeps
│   ├── compare_sweep.py          # Parses sweep outputs and Pareto ranks parameter sets
│   ├── visualization.py          # Generates evaluation PDF figures (Figures 1–4)
│   └── hyperparameters_sweep_plots.py # Generates sweep trade-off plots
├── sweep_outputs/                # JSON trial records from individual sweep worker runs
└── requirements.txt              # Python package dependencies
```

---

## Setup and Installation

1. **Environment Setup (Python 3.10+):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Network Dataset Initialization:**
   The simulation utilizes empirical V2X channel and vehicular load traces generated from the external dataset:
   ```bash
   python src/setup_data.py
   ```

---

## Running Locally

Simulations can be executed directly using `src/main.py`. The runner automatically detects available GPUs and falls back to CPU execution if CUDA is unavailable.

### 1. Run Dynamic Policy
```bash
python src/main.py --rounds 50 --dataset cifar10 --dirichlet-alpha 0.1 --dynamic_only --job-id a01
```

### 2. Run Baselines
- **Fixed Epoch FedAvg ($E = 5$):**
  ```bash
  python src/main.py --rounds 50 --dataset cifar10 --dirichlet-alpha 0.1 --fixed 5 --job-id a01
  ```
- **Fixed Epoch FedProx ($E = 5, \mu = 0.01$):**
  ```bash
  python src/main.py --rounds 50 --dataset cifar10 --dirichlet-alpha 0.1 --fedprox-only --fedprox-epochs 5 --fedprox-mu 0.01 --job-id a01
  ```
- **Dynamic FedProx ($\mu = 0.01$):**
  ```bash
  python src/main.py --rounds 50 --dataset cifar10 --dirichlet-alpha 0.1 --dynamic-fedprox --fedprox-mu 0.01 --job-id a01
  ```

### 3. Generate Evaluation Figures
After running experiments, generate camera-ready PDF figures into `plots/`:
```bash
python src/visualization.py --job-id a01
```

---

## Running on Slurm Cluster (`yoda`)

To dispatch full 250-round baseline evaluations across Dirichlet partitions ($\alpha_{\text{Dir}} \in \{0.1, 0.3, 0.5\}$):

```bash
# Submit baseline suite for alpha = 0.1 (Dynamic, Fixed 1, Fixed 5, Fixed 10, FedProx 5)
sbatch scripts/submit_final.sh 0.1

# Submit baseline suite for alpha = 0.3
sbatch scripts/submit_final.sh 0.3

# Submit baseline suite for alpha = 0.5
sbatch scripts/submit_final.sh 0.5
```

Outputs are logged as `metrics_<experiment>_<job-id>.csv` and can be visualized via `python src/visualization.py --job-id a01`.

---

## Citation & Paper

If you use this codebase or framework in your research, please cite:
```bibtex
@inproceedings{arvanitis2026flsdv,
  title={Federated Learning in SDVs: Optimizing Communication Frequency},
  author={Arvanitis, Spyridon},
  booktitle={IEEE Conference Proceedings},
  year={2026}
}
```
