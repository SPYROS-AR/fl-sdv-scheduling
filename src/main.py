import math


from traci_bridge import TraciBridge

USE_GUI = False
SUMO_CONFIG = "sumo.env/simulation.sumocfg"

# RSU Environment Parameters
RSU_X = 200.0  # Center of sumo grid
RSU_Y = 200.0

MAX_BANDWIDTH = 50.0  # Maximum capacity in Mbps
MAX_DISTANCE = 200.0  # Out-of-range boundary threshold
MIN_DISTANCE = 50.0  # Full-signal boundary threshold


def calculate_bandwidth(veh_x, veh_y):
    """
    Translates phyical 2d location into uplink Bandwitdh (B_i)
    """
    # Calculate euclidean distance to edge server
    distance_to_server = math.sqrt((RSU_X - veh_x) ** 2 + (RSU_Y - veh_y) ** 2)

    if distance_to_server > MAX_DISTANCE:
        return 0.0
    if distance_to_server < MIN_DISTANCE:
        return MAX_BANDWIDTH

    # FIX: Use accurate power loss function
    scale = 1.0 - ((distance_to_server - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE))
    return MAX_BANDWIDTH * scale


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
                bandwidth = calculate_bandwidth(data["x"], data["y"])

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
