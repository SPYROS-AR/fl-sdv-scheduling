import traci
import sumolib


class TraciBridge:
    def __init__(self, config_path, use_gui=False):
        self.config_path = config_path
        binary_name = "sumo_gui" if use_gui else "sumo"
        self.sumo_binary = sumolib.checkBinary(binary_name)

    def start(self):
        traci.start([self.sumo_binary, "-c", self.config_path])

    def step(self):
        traci.simulationStep()

    def get_vehicle_telemetry(self):
        telemetry = {}
        vehicle_ids = traci.vehicle.getIDList()

        for veh_id in vehicle_ids:
            pos = traci.vehicle.getPosition(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)
            telemetry[veh_id] = {"x": pos[0], "y": pos[1], "speed": speed}
        return telemetry

    def is_simulation_running(self):
        return traci.simulation.getMinExpectedNumber() > 0

    def close(self):
        traci.close()
