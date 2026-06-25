import math
from traci_bridge import TraciBridge
import numpy as np


USE_GUI = False
SUMO_CONFIG = "sumo.env/simulation.sumocfg"

# MEC Environment Parameters
MEC_X, MEC_Y = 200.0, 200.0
MAX_BANDWIDTH = 50.0
MAX_DISTANCE = 2000.0
MIN_DISTANCE = 50.0

FREQ_MHZ = 5900.0
Pt = 23.0  # dBm
FLOOR_NOISE = -100.0  # dBm
BANDWITDH = 20.0  # MHz


def calculate_uplink_data_rate(veh_x: float, veh_y: float) -> float:
    """
    Translates physical 2d location into uplink bandwidth (Mbps).
    """
    # Calculate Euclidean distance to edge server
    distance_to_server = math.sqrt((MEC_X - veh_x) ** 2 + (MEC_Y - veh_y) ** 2)

    # Boundary checks
    if distance_to_server >= MAX_DISTANCE:
        return 0.0
    if distance_to_server <= MIN_DISTANCE:
        return MAX_BANDWIDTH

    # Calculate Path Loss (dB) using the standard FSPL formula
    fspl_db = 20 * math.log10(distance_to_server) + 20 * math.log10(FREQ_MHZ) - 27.55

    # Received Power (dBm)
    pr_dbm = Pt - fspl_db

    # SNR (dB) and linear conversion
    snr_db = pr_dbm - FLOOR_NOISE
    snr_linear = 10 ** (snr_db / 10)

    # Shannon Capacity (Mbps)
    capacity = BANDWITDH * math.log2(1 + snr_linear)

    # Hardware limit
    return min(capacity, MAX_BANDWIDTH)


def main():
    bridge = TraciBridge(SUMO_CONFIG, USE_GUI)

    print("Connecting to SUMO enviroment")
    bridge.start()

    step_count = 0
    try:
        # Execute the time-horizon horizon loop
        while bridge.is_simulation_running():
            bridge.step()

            # Ingest live vehicle states from the environment
            telemetry = bridge.get_vehicle_telemetry()

            # Map physical metrics to system state variables
            for veh_id, data in telemetry.items():
                bandwidth = calculate_uplink_data_rate(data["x"], data["y"])

                # Filter out passive nodes and log active client parameters
                if bandwidth > 0:
                    print(
                        f"[Step {step_count:04d}] Client: {veh_id} | "
                        f"Coords: ({data['x']:.1f}, {data['y']:.1f}) | "
                        f"Velocity: {data['speed']:.1f} m/s | "
                        f"B_i(t): {bandwidth:.2f} Mbps"
                    )

            step_count += 1

    except Exception as error:
        print(f"Runtime processing error caught: {error}")
    finally:
        # Ensure cleanup is triggered
        print("Teminating connection")  # DEBUG
        bridge.close()
        print("Process ended.")


if __name__ == "__main__":
    main()
