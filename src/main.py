import math
from turtle import distance
from traci_bridge import TraciBridge
import numpy as np
import random

USE_GUI = False
SUMO_CONFIG = "sumo.env/simulation.sumocfg"

# MEC Environment Parameters
MEC_X, MEC_Y = 200.0, 200.0
MAX_BANDWIDTH = 250.0  # (Mbps)
MAX_DISTANCE = 2000.0  # (meters)
MIN_DISTANCE = 10.0  # (meters)


c = 3e8  # speed of light (m/s)
FREQ_MHZ = 5900.0
Pt = 23.0  # dBm
FLOOR_NOISE = -100.0  # dBm
BANDWIDTH = 20.0  # MHz
PATH_LOSS_EXP = 3.5  # (Theoretic for shadowed urban tower)
SHADOWING_STD_DEV = 6.0  # (db)
D0 = 1.0  # Reference distance (meters)


# Compute standard fspl for reference distance D0 once
PL_d0 = 20 * math.log10(4 * math.pi * D0 * FREQ_MHZ * 1e6 / c)


def calculate_uplink_data_rate(veh_x: float, veh_y: float) -> float:
    """
    Translates physical 2d location into uplink bandwidth (Mbps).
    """
    # Calculate Euclidean distance to edge server
    dist = math.sqrt((MEC_X - veh_x) ** 2 + (MEC_Y - veh_y) ** 2)

    # Edge case where vehicle is next to the server
    dist = max(dist, 1.0)

    # Boundary checks
    if dist >= MAX_DISTANCE:
        return 0.0
    if dist <= MIN_DISTANCE:
        return MAX_BANDWIDTH

    # FIX: Add shadowing to vehicle objects and update every few seconds
    # Gaussian shadowing
    shadowing = random.gauss(0, SHADOWING_STD_DEV)

    # --- LOG-DISTANCE PATH LOSS ---

    # Calculate Path Loss (dB) using Log-distance path loss
    path_loss_db = PL_d0 + (10 * PATH_LOSS_EXP * math.log10(dist / D0)) + shadowing

    # Received Power (dBm)
    pr_dbm = Pt - path_loss_db

    # SNR (dB) and linear conversion
    snr_db = pr_dbm - FLOOR_NOISE
    snr_linear = 10 ** (snr_db / 10)

    # Shannon Capacity (Mbps)
    capacity = BANDWIDTH * math.log2(1 + snr_linear)

    print(f"Dist: {dist:.1f}m | SNR: {snr_db:.1f}dB")  # DEBUG
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
