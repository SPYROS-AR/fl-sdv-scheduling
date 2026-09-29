# Federated Learning in Software-Defined Vehicles (SDVs): Optimizing Communication Frequency

This repository contains the implementation and experimental evaluation code for the research paper: **"Federated Learning in SDVs: Optimizing Communication Frequency"** by Spyridon Arvanitis.

## Overview

Deploying Federated Learning (FL) on Software-Defined Vehicles (SDVs) requires balancing wireless transmission overhead, client model drift, and onboard compute resource contention with safety-critical vehicle workloads (such as ADAS perception and trajectory planning). 

Conventional vehicular FL frameworks typically assume dedicated onboard compute and rely on static local training intervals, which can lead to uplink cellular congestion and compute starvation for co-located real-time service. This project provides a runtime heuristic policy engine that dynamically regulates local training epochs and client upload schedules based on available compute slack, instantaneous link throughput, and observed local loss progress.

---

## Repository Structure

```text
.
├── src/
│   ├── client.py                 # FLVehicle client implementation (dynamic and fixed policies)
│   ├── server.py                 # DynamicFLStrategy and CSV metrics logging
│   ├── scheduler.py              # Flower simulation orchestrator and client factory
│   ├── model.py                  # ResNet-9 architecture with Group Normalization
│   ├── constants.py              # Simulation, networking, and utility hyperparameters
│   ├── data_setup.py             # External repository cloning and Dirichlet data partitioning
│   ├── data_processing.py        # Network dataset cleaning and feature extraction
│   ├── main.py                   # Main CLI entry point for simulations
│   ├── sweep_worker.py           # Single-worker script for hyperparameter sweeps
│   ├── compare_sweep.py          # Parses sweep outputs and ranks parameter sets
│   ├── visualization.py          # Generates evaluation PDF figures (Figures 1–4)
│   ├── hyperparameters_sweep_plots.py # Generates sweep plots
│   └── setup_data.py             # One-time data initialization script
├── generate_params.py            # Generates hyperparameter combinations (params.txt)
├── plot_sweeps.py                # Generates interactive Plotly parallel coordinates plot
├── submit_sweep.sh               # Slurm array script for hyperparameter sweeps
├── submit_aggregator.sh          # Slurm script for aggregating sweep results
├── submit_final.sh               # Slurm array script for final comparative evaluation
└── requirements.txt              # Python package dependencies
```

---

## Setup and Initialization

1. **Create a Python environment (Python 3.10+ required):**
   You can use Conda (recommended), standard `venv`, or a fast resolver like `uv`.

   **Option A: Using Conda (Recommended)**
   ```bash
   conda create -n fl_env python=3.10
   conda activate fl_env
   pip install -r requirements.txt
   ```
   **Option B: Using standard Python venv**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   **Option C: Using uv (Fastest)**
   ```bash
   uv venv -p 3.10
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

2. **Initialize the dataset:**
   This step clones the external network dataset repository (`CLOUDNET2026-QoS-Offloading.git`) and partitions the CIFAR-10 dataset using a Dirichlet distribution ($\alpha_{Dir} = 0.2$) across $10$ vehicles:
   ```bash
   python src/setup_data.py
   ```

---

## Running Locally

To run experiments directly via Python without a cluster scheduler, use `src/main.py`.

### 1. Run Simulations

Run only the dynamic policy:
```bash
python src/main.py --rounds 250 --dynamic_only
```
Run a specific fixed-epoch baseline (e.g., 3 epochs):
```bash
python src/main.py --rounds 250 --fixed 3
```
Run all baseline experiments sequentially (Dynamic and Fixed 1, 3, 5, 10, 20):
```bash
python src/main.py --rounds 50
```

### 2. Generate Evaluation Plots

After running the simulation data generation, produce the evaluation PDF figures (saved in the `plots/` directory):
```bash
python src/visualization.py
```

---

## Running on a Slurm Cluster

To execute large-scale hyperparameter sweeps or full comparative evaluations on a Slurm cluster, use the provided shell scripts.

### 1. Hyperparameter Sweeps

1. **Generate the parameter file (`params.txt`):**
   ```bash
   python generate_params.py --num_sweeps 100
   ```
2. **Submit the parameter sweep array job:**
   ```bash
   sbatch submit_sweep.sh
   ```
3. **Aggregate the sweep results once finished:**
   ```bash
   sbatch submit_aggregator.sh
   ```
4. **Generate interactive sweep visualizations (Plotly HTML):**
   ```bash
   python plot_sweeps.py
   ```

### 2. Final Comparative Evaluation

To run the complete comparative benchmark across the proposed dynamic policy and all fixed-epoch baselines (1, 3, 5, 10, 20 epochs) as a parallel array job:
```bash
sbatch submit_final.sh
```
*Note: This script dispatches array tasks (0 to 5) running `main.py --rounds 250` in parallel using allocated GPU resources.*
