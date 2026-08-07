import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Optional, Union
import pandas as pd
import random

class DynamicFLStrategy(fl.server.strategy.FedAvg):
    def __init__(self, df_network: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_network = df_network
        
        # Global Optimization Objective Metrics
        self.total_c_net = 0.0
        self.total_c_comp = 0.0
        self.total_c_drift = 0.0
        self.total_gain = 0.0
        
        # Operational States Tracking
        self.total_syncs = 0
        self.total_continues = 0
        self.total_waits = 0
        self.total_dropped = 0
        
        # Network Capacity Constraint: B^{cell}
        self.B_cell_max = 400.0  

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_configs = super().configure_fit(server_round, parameters, client_manager)
        
        custom_configs = []
        for idx, (client_proxy, fit_ins) in enumerate(client_configs):
            row_idx = ((server_round - 1) * len(client_configs) + idx) % len(self.df_network)
            sample = self.df_network.iloc[row_idx]
            
            config_dict = fit_ins.config.copy()
            config_dict["u_crit"] = float(sample["u_crit_t"])
            config_dict["bandwidth"] = float(sample["effective_throughput_mbps"])
            
            config_dict["tau_continue"] = 0.1     
            config_dict["drift_threshold"] = 5.0  
            config_dict["gain_threshold"] = 0.01  
            config_dict["bw_threshold"] = 30.0
            
            new_fit_ins = FitIns(parameters=fit_ins.parameters, config=config_dict)
            custom_configs.append((client_proxy, new_fit_ins))
            
        return custom_configs

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        valid_results = []
        current_bw_used = 0.0
        model_size_mb = 50.0  # M_i
        
        round_syncs, round_continues, round_waits, round_dropped = 0, 0, 0, 0
        round_c_net, round_c_comp, round_c_drift, round_gain = 0.0, 0.0, 0.0, 0.0
        
        random.shuffle(results)
        
        for client, fit_res in results:
            metrics = fit_res.metrics
            status = metrics.get("status", "WAIT")
            
            # Interference and Gain apply to both SYNC and CONTINUE
            round_c_comp += metrics.get("c_comp", 0.0)
            round_gain += metrics.get("gain", 0.0)
            
            if fit_res.num_examples == 0:
                if status == "WAIT":
                    round_waits += 1
                    self.total_waits += 1
                elif status == "CONTINUE":
                    round_continues += 1
                    self.total_continues += 1
            else:
                # Client decided to SYNC (p_i = 1). Verify Network Capacity Constraint
                if current_bw_used + model_size_mb <= self.B_cell_max:
                    valid_results.append((client, fit_res))
                    round_syncs += 1
                    self.total_syncs += 1
                    
                    current_bw_used += model_size_mb
                    round_c_net += metrics.get("c_net", 0.0)
                    round_c_drift += metrics.get("c_drift", 0.0)
                else:
                    round_dropped += 1
                    self.total_dropped += 1
                    
        # Update Global Metrics
        self.total_c_comp += round_c_comp
        self.total_c_net += round_c_net
        self.total_c_drift += round_c_drift
        self.total_gain += round_gain
        
        print(f"\n[Round {server_round}] SYNC: {round_syncs} | CONTINUE: {round_continues} | WAIT: {round_waits} | DROPPED: {round_dropped}")
        print(f"[Round {server_round}] Cell Load: {current_bw_used}/{self.B_cell_max} MB")
        
        if not valid_results:
            return None, {}
            
        return super().aggregate_fit(server_round, valid_results, failures)