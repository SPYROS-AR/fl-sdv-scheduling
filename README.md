# FL-SDV Scheduling

A simulation environment for Federated Learning (FL) resource scheduling in Software-Defined Vehicles (SDVs)

## System Architecture and Capabilities

The framework has transitioned to a dataset-driven mathematical simulation pipeline, moving away from live physical SUMO telemetry. The current workflow includes:

1. **Data Setup:** Automatically clones the [`CLOUDNET2026-QoS-Offloading`](https://github.com/icsa-hua/CLOUDNET2026-QoS-Offloading.git) repository and generates network dataset samples based on configured vehicle and sample constants.
2. **Data Processing:** Loads and cleans the generated network data (`network_dataset.csv`). It dynamically calculates the critical compute load (`u_crit_t`) for each vehicle utilizing factors such as car speed, urban scenarios (dense vs. sparse), and active network handovers.
3. **Federated Learning Scheduler:** Utilizes a `HeuristicFLScheduler` to allocate compute resources (`c_i`, `e_i`) and decide communication states (`p_i`). The scheduler optimizes a custom cost function that balances:
   - **Network Cost:** Evaluated using the vehicle's model size and the dynamic MEC channel bandwidth.
   - **Compute Cost:** Derived from the compute allocation and the vehicle's critical load.
   - **Model Drift:** Tracks model backlog/divergence and dynamically updates state constraints to trigger global syncs.

## Project Structure

- `main.py`: The entry point for the simulation. Orchestrates data generation, data processing, and scheduling logic.
- `constants.py`: Contains global configuration variables (e.g., `NUM_SAMPLES`, `NUM_VEHICLES`).
- `data_setup.py`: Automates the cloning of the dataset repository and executes the dataset generation script via a subprocess.
- `data_processing.py`: Formats boolean columns, computes the critical load heuristic, and exports a cleaned dataset.
- `scheduler.py`: Contains the `Vehicle` class representing FL clients, and the `FLScheduler` class which manages decision-making and generates visual plots of aggregate drift over time.

## How to Run

### Prerequisites
Ensure you have Python 3 installed along with the required libraries:
```bash
pip install pandas numpy matplotlib
```
### Execution
Run the full pipeline (dataset generation, processing, and simulation execution) using:

```Bash
python3 main.py
```