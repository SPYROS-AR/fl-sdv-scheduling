import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import constants

# --- Vehicle ---

class Vehicle:

    def __init__(self, vehicle_id: int, model_size_mb: float = 50.0, 
                 epochs_per_compute: int = 10, drift_growth_rate: float = 0.5, 
                 drift_reduction_rate: float = 10.0, max_drift: float = constants.MAX_ALLOWED_DRIFT):
        self.id = vehicle_id
        self.M = model_size_mb # M_i
        self.rho = epochs_per_compute # \rho_i
        self.a = drift_growth_rate # a_i
        self.b = drift_reduction_rate # b_i: Drift
        self.D_max = max_drift # D_i^{max}
        
        # Dynamic State
        self.current_drift = 0.0 # D_i(t)


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

    def step(self, t: int, env_states: list) -> list:

        # Compute allocation per vehicle
        step_decisions = self._allocate_compute(env_states)
        
        # Communication decisions
        self._decide_communication(step_decisions)
        
        # Cost calculation and state update
        self._calculate_costs_and_update_state(t, step_decisions)
        
        return step_decisions

    def _allocate_compute(self, env_states: list) -> list:

        decisions = []
        for i, vehicle in enumerate(self.vehicles):
            state = env_states[i]
            u_crit = state['u_crit_t']       # U_crit(t)
            bandwidth = state['bandwidth']   # B_i(t)
            
            # Constraint: c_i(t) <= 1 - U_crit(t)
            compute_allocation_c = max(0.0, 1.0 - u_crit)
            
            # Constraint: e_i(t) <= rho_i * c_i(t)
            local_epochs_e = int(np.floor(vehicle.rho * compute_allocation_c))
            
            decisions.append({
                'vehicle': vehicle,
                'state_u_crit': u_crit, 
                'state_bw': bandwidth,
                'decision_c': compute_allocation_c, # c_i(t)
                'decision_e': local_epochs_e, # e_i(t)
                'decision_p': 0  # p_i(t)
            })
        return decisions

    def _decide_communication(self, decisions: list):

        # Sort vehicles by drift
        decisions.sort(key=lambda x: x['vehicle'].current_drift, reverse=True)
        
        current_cell_load = 0.0
        
        for decision in decisions:
            v = decision['vehicle']
            drift_threshold = v.D_max * self.sync_threshold_ratio
            
            # If drift is large enough, attempt transmission
            if v.current_drift > drift_threshold:
                # Constraint: Sum(p_i * M_i) <= B_cell(t)
                if current_cell_load + v.M <= self.cell_bandwidth:
                    decision['decision_p'] = 1  # Execute transmission
                    current_cell_load += v.M

    def _calculate_costs_and_update_state(self, t: int, decisions: list):

        total_net_cost, total_comp_cost, total_drift_cost = 0.0, 0.0, 0.0
        
        for d in decisions:
            v = d['vehicle']
            
            # C_net(t) = p_i * (M_i / B_i) 
            cost_net = d['decision_p'] * (v.M / max(d['state_bw'], 0.1))
            
            # C_comp(t) = c_i * U_crit
            cost_comp = d['decision_c'] * d['state_u_crit']
            
            # C_drift(t) = D_i(t)
            cost_drift = v.current_drift
            
            total_net_cost += cost_net
            total_comp_cost += cost_comp
            total_drift_cost += cost_drift
            

            # D_i(t+1) = D_i(t) + a_i*e_i(t) - b_i*p_i(t)
            new_drift = v.current_drift + (v.a * d['decision_e']) - (v.b * d['decision_p'])
            
            v.current_drift = max(0.0, min(new_drift, v.D_max))

        # Total objective
        total_cost = (self.alpha * total_net_cost) + \
                     (self.beta * total_comp_cost) + \
                     (self.gamma * total_drift_cost)

        self.history.append({
            'time': t,
            'cost_net': total_net_cost,
            'cost_comp': total_comp_cost,
            'cost_drift': total_drift_cost,
            'total_cost': total_cost
        })

    def plot_history(self):
        times = [entry['time'] for entry in self.history]
        drift_costs = [entry['cost_drift'] for entry in self.history]

        plt.figure(figsize=(10, 6))
        
        plt.plot(times, drift_costs, label='Total Drift', color='crimson', linewidth=2)
        
        plt.title('System Drift Over Time', fontsize=14)
        plt.xlabel('Time Step (t)', fontsize=12)
        plt.ylabel('Aggregate Drift Cost', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.show()




def run_simulation(df: pd.DataFrame, num_vehicles: int, num_samples: int):

    total_time_steps = int(num_samples / num_vehicles) 
    
    # Init vehicles
    vehicles = [Vehicle(vehicle_id=i) for i in range(num_vehicles)]
    
    # Init Scheduler
    scheduler = FLScheduler(vehicles, alpha=1.0, beta=0.5, gamma=2.0)
    
    print(f"Starting FL Scheduling Simulation for {total_time_steps} time steps...\n")
    
    for t in range(total_time_steps):
        
        # Gather environment states from the dataset for step t
        env_states = []
        for i in range(num_vehicles):
            row_idx = (t * num_vehicles + i) % len(df)
            row = df.iloc[row_idx] 
            
            env_states.append({
                'bandwidth': row['mec_channel_bw_mhz'],
                'u_crit_t': row['u_crit_t']
            })
            
        decisions = scheduler.step(t, env_states)
        
        # Print results
        if t % 1000 == 0:
            print(f"--- Time Step t={t} ---")
            for d in decisions:
                v_id = d['vehicle'].id
                print(f"Vehicle {v_id} | Compute (c): {d['decision_c']:.2f} | Epochs (e): {d['decision_e']} "
                      f"| Transmitted (p): {d['decision_p']} | New Drift: {d['vehicle'].current_drift:.1f}")
            
            current_cost = scheduler.history[-1]['total_cost']
            print(f"> Total Cost for t={t}: {current_cost:.2f}\n")
    scheduler.plot_history()        