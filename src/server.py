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
        
        # Metrics
        self.total_mbs_sent = 0.0
        self.total_syncs = 0
        self.total_waits = 0
        self.total_dropped = 0 
        self.total_interference = 0.0

        self.B_cell_max = 400.0  

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_configs = super().configure_fit(server_round, parameters, client_manager)
        
        custom_configs = []
        for client_proxy, fit_ins in client_configs:
            sample = self.df_network.sample(1).iloc[0]
            
            # Dictionary for System Model
            config_dict = fit_ins.config.copy()
            config_dict["u_crit"] = float(sample["u_crit_t"])             # U_crit(t)
            config_dict["bandwidth"] = float(sample["mec_channel_bw_mhz"]) # B_i(t)
            
            config_dict["tau_continue"] = 0.1     # Min utility for CONTINUE
            config_dict["drift_threshold"] = 5.0  # D_max for forced SYNC
            config_dict["gain_threshold"] = 0.01  # G_min for forced SYNC
            config_dict["bw_threshold"] = 30.0
            
            new_fit_ins = FitIns(parameters=fit_ins.parameters, config=config_dict)
            custom_configs.append((client_proxy, new_fit_ins))
            
        return custom_configs

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        valid_results = []
        round_syncs = 0
        round_waits = 0
        round_dropped = 0
        
        current_bw_used = 0.0
        model_size_mb = 50.0 # M_i

        random.shuffle(results)
        
        for client, fit_res in results:
            
            if fit_res.num_examples == 0:
                round_waits += 1
                self.total_waits += 1
            else:
                
                if current_bw_used + model_size_mb <= self.B_cell_max:
                    valid_results.append((client, fit_res))
                    round_syncs += 1
                    self.total_syncs += 1
                    current_bw_used += model_size_mb
                    self.total_mbs_sent += model_size_mb
                    
                    interference = fit_res.metrics.get("interference", 0.0)
                    self.total_interference += interference
                else:
                    round_dropped += 1
                    self.total_dropped += 1

        print(f"\n[Round {server_round}] SYNC: {round_syncs} | WAIT: {round_waits} | DROPPED (Network): {round_dropped}")
        print(f"[Round {server_round}] Total Data: {self.total_mbs_sent} MB | Cell Load: {current_bw_used}/{self.B_cell_max} MB")
        
        if not valid_results:
            print(f"[Round {server_round}] No valid SYNCs. Keeping global model.")
            return None, {}

        return super().aggregate_fit(server_round, valid_results, failures)