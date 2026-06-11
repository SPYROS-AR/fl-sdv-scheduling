# FL-SDV Scheduling

A hybrid simulation environment bridging physical vehicular traffic dynamics (SUMO) with a Python-based Federated Learning (FL) resource scheduler.

# System Model Status
The framework currently executes a complete physical-to-mathematical pipeline at every simulation step:
1. **Telemetry Extraction:** Captures live (X,Y) coordinates and velocity from active SUMO vehicles via the TraCI API.
2. **RF Signal Modeling:** Calculates dynamic Uplink Bandwidth *B_i(t)* for each vehicle using a Linear-Distance Path Loss model relative to a central Edge Server.
3. **FL Client Filtering:** Automatically drops clients from the active FL pool if they drive beyond the physical communication boundary.
Additional edge-compute variables are currently in active development.

## How to run

### Installation & Execution
Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/SPYROS-AR/fl-sdv-scheduling.git
cd fl-sdv-scheduling
```
### Option 1: Local execution (Recommended for UI)
1. Setup Environment

```Bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
2. Configure UI

Open src/main.py and set:
USE_GUI = True

3. Run Simulation

Ensure you are in the project root directory:

```Bash
python3 src/main.py
```
### Option 2: Using docker (Headless)

Use this method for isolated execution and fast data collection.
The simulation runs entirely in the background. 

Simply use the provided docker-compose.yml file and run this command.
```Bash
docker-compose up
```
*Note:* The GUI cannot be used in the container.

## Known Limitations

***Simplified RF Physics:*** The simulation currently uses a linear-distance path loss model to calculate bandwidth.
Because real-world RF signals degrade exponentially, this artificially inflates bandwidth at medium distances.
This will be replaced with a more realistic loss function in the future.
