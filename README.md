# Federated Learning in SDVs: Optimizing Communication Frequency

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-FL-792ee5.svg)](https://flower.ai/)

Official PyTorch and Flower (`flwr`) implementation for the paper **"Federated Learning in SDVs: Optimizing Communication Frequency"**[cite: 2, 6].

This repository implements an online, lightweight runtime policy engine for Federated Learning (FL) across fleets of **Software-Defined Vehicles (SDVs)**. Unlike conventional vehicular FL frameworks that execute fixed local epochs and assume isolated onboard hardware, our engine dynamically governs local training depth and wireless upload frequency based on **available safety-critical CPU slack**, **instantaneous V2I link throughput**, and **short-horizon exponential forecasts of local validation loss decay**.

---

## Key Features & Architecture

* **Three-State Vehicular Execution Engine (`WAIT`, `CONTINUE`, `SYNC`):**
  * **`WAIT` ($e_i = 0, p_i = 0$):** Preempts FL execution and yields processor cycles to safety-critical Advanced Driver-Assistance Systems (ADAS) when critical CPU occupancy is high ($U_i^{\text{crit}}(t) > 0.80$) or when poor V2I channel throughput ($B_i(t) < 20.0\text{ Mbps}$) coincides with flat prior validation gain[cite: 5, 6, 7, 8].
  * **`CONTINUE` ($e_i > 0, p_i = 0$):** Trains locally during wireless uplink congestion while persisting model weights on disk across Ray actor instantiations (`.client_cache/`), maintaining a secondary reference buffer to compute exact $L_2$ parameter drift and FedProx proximal gradients relative to the evolving global model[cite: 1, 6].
  * **`SYNC` ($e_i > 0, p_i = 1$):** Uploads parameter updates when link throughput is sufficient or when safety-valve thresholds on accumulated $L_2$ drift ($0.85 D_i^{\text{max}}$) or consecutive local rounds ($K_{\text{max}} = 5$) are reached[cite: 5, 6].
* **Bounded Exponential Gain Predictor (`BoundedExpFit`):** Fits a box-constrained three-parameter exponential decay curve $L(k) \approx L_\infty + a \exp(-bk)$ via `scipy.optimize.curve_fit` to smoothed local validation loss traces, halting local epochs before diminishing returns and client drift set in[cite: 1, 6].
* **GroupNorm ResNet-9 Architecture:** Custom 9-layer Residual Network ($2,441,546$ FP32 parameters, $9.31\text{ MiB}$ payload) replacing standard Batch Normalization with Group Normalization (8 groups) to prevent running-statistic divergence under extreme non-IID data skew[cite: 1, 6].
* **Realistic V2X & Mixed-Criticality Trace Driven:** Simulates 100,000 vehicular mobility and Multi-access Edge Computing (MEC) states incorporating Shannon spectral efficiency, cellular handovers, peak-hour traffic, and dynamic cell buffer limits[cite: 1, 6].

---

## Main Experimental Results

Evaluated over **250 communication rounds** across $N = 10$ vehicles on non-IID CIFAR-10 partitions (`DirichletPartitioner`, $\alpha_{\text{Dir}} \in \{0.1, 0.3, 0.5\}$)[cite: 6, 8]:

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
├── submit_final.sh                    # Slurm array script for full 250-round evaluations (18 jobs)
├── submit_sweep.sh                    # Slurm array script for 100-trial hyperparameter sweeps
└── submit_aggregator.sh               # Slurm script to aggregate sweep outputs