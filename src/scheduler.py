from enum import Enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import constants
from client import FLVehicle


# --- SCHEDULER ---

class FLScheduler:

    def __init__(self, vehicles: list, alpha=1.0, beta=1.0, gamma=1.0, 
                 total_cell_bandwidth=constants.TOTAL_BANDWIDTH, sync_threshold_ratio=0.5):
        
        self.vehicles = vehicles # Vehicles
        
        # Cost function weights
        self.alpha = alpha  # Weight for network cost
        self.beta = beta    # Weight for compute interference cost
        self.gamma = gamma  # Weight for model drift cost
        
        # Network and Strategy parameters
        self.cell_bandwidth = total_cell_bandwidth
        self.sync_threshold_ratio = sync_threshold_ratio
        
        self.history = []

    def step(self):
        pass

    def _calculate_global_costs(self, step_results):
        pass
    
    def plot_history(self):
        pass






def run_simulation(df: pd.DataFrame, num_vehicles: int, num_samples: int):
    pass

def init_Flower():
    pass


# Action to take 
class Action(Enum):
    CONTINUE = 1
    SYNC = 2
    WAIT = 3