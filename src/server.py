import csv
import math
import os
import random
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from constants import MAX_ALLOWED_DRIFT, TOTAL_BANDWIDTH, MODEL_SIZE_MB, TAU_CONTINUE, GAIN_THRESHOLD, BW_THRESHOLD

class CSVLogger:
    def __init__(self, exp_name: str):
        self.csv_file = f"metrics_{exp_name}.csv"
        self.pending_metrics = {}
        self._initialize_file()

    def _initialize_file(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round", "syncs", "continues", "waits", "dropped", 
                    "c_net", "c_comp", "c_drift", "gain", "utility",
                    "total_c_net", "total_c_comp", "total_c_drift", "total_gain", "total_utility", "val_loss"
                ])

    def stage_round_metrics(self, round_num: int, metrics_row: list):
        self.pending_metrics[round_num] = metrics_row

    def commit_round_metrics(self, round_num: int, val_loss: float):
        if round_num in self.pending_metrics:
            row = self.pending_metrics.pop(round_num)
            row.append(val_loss if val_loss is not None else "")
            with open(self.csv_file, 'a', newline='') as f:
                csv.writer(f).writerow(row)

class DynamicFLStrategy(fl.server.strategy.FedAvg):
    def __init__(self, df_network: pd.DataFrame, alpha=0.002, beta=1.0, gamma=0.05, delta=20.0, exp_name="dynamic", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_network = df_network
        self.alpha, self.beta = alpha, beta
        self.gamma, self.delta = gamma, delta
        
        self.logger = CSVLogger(exp_name)
        
        self.total_c_net = self.total_c_comp = self.total_c_drift = self.total_gain = self.total_utility = 0.0
        self.total_syncs = self.total_continues = self.total_waits = self.total_dropped = 0

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        client_configs = super().configure_fit(server_round, parameters, client_manager)
        
        custom_configs = []
        for idx, (client_proxy, fit_ins) in enumerate(client_configs):
            row_idx = ((server_round - 1) * len(client_configs) + idx) % len(self.df_network)
            sample = self.df_network.iloc[row_idx]
            
            config_dict = fit_ins.config.copy()
            config_dict.update({
                "u_crit": float(sample["u_crit_t"]),
                "bandwidth": float(sample["effective_throughput_mbps"]),
                "tau_continue": TAU_CONTINUE,
                "drift_threshold": MAX_ALLOWED_DRIFT,
                "gain_threshold": GAIN_THRESHOLD,
                "bw_threshold": BW_THRESHOLD
            })
            custom_configs.append((client_proxy, FitIns(parameters=fit_ins.parameters, config=config_dict)))
            
        return custom_configs

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        valid_results = []
        current_bw_used = 0.0
         
        r_syncs = r_continues = r_waits = r_dropped = 0
        r_c_net = r_c_comp = r_c_drift = r_gain = 0.0
        
        current_B_cell_max = TOTAL_BANDWIDTH * (0.8 + 0.2 * math.sin(server_round))
        random.shuffle(results)
        
        for client, fit_res in results:
            metrics = fit_res.metrics
            status = metrics.get("status", "WAIT")
            
            r_c_comp += float(metrics.get("c_comp", 0.0))
            r_gain += float(metrics.get("gain", 0.0))
            
            if fit_res.num_examples == 0:
                if status == "WAIT":
                    r_waits += 1
                elif status == "CONTINUE":
                    r_continues += 1
            else:
                if current_bw_used + MODEL_SIZE_MB <= current_B_cell_max:
                    valid_results.append((client, fit_res))
                    r_syncs += 1
                    current_bw_used += MODEL_SIZE_MB
                    r_c_net += float(metrics.get("c_net", 0.0))
                    r_c_drift += float(metrics.get("c_drift", 0.0))
                else:
                    r_dropped += 1
                    
            r_utility = (self.delta * r_gain) - (self.alpha * r_c_net) - (self.beta * r_c_comp) - (self.gamma * r_c_drift)
            
        self._update_global_metrics(r_syncs, r_continues, r_waits, r_dropped, r_c_net, r_c_comp, r_c_drift, r_gain, r_utility)
        self._print_round_summary(server_round, r_syncs, r_continues, r_waits, r_dropped, current_bw_used, current_B_cell_max, r_utility, r_c_drift)
        
        self.logger.stage_round_metrics(server_round, [
            server_round, r_syncs, r_continues, r_waits, r_dropped,
            r_c_net, r_c_comp, r_c_drift, r_gain, r_utility,
            self.total_c_net, self.total_c_comp, self.total_c_drift, self.total_gain, self.total_utility
        ])
        
        if not valid_results:
            return None, {}
            
        return super().aggregate_fit(server_round, valid_results, failures)

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, fl.common.EvaluateRes]], failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self.logger.commit_round_metrics(server_round, aggregated_loss)
        return aggregated_loss, metrics

    def _update_global_metrics(self, syncs, continues, waits, dropped, net, comp, drift, gain, util):
        self.total_syncs += syncs
        self.total_continues += continues
        self.total_waits += waits
        self.total_dropped += dropped
        self.total_c_net += net
        self.total_c_comp += comp
        self.total_c_drift += drift
        self.total_gain += gain
        self.total_utility += util

    def _print_round_summary(self, round_num, syncs, continues, waits, dropped, bw_used, bw_max, util, drift):
        print(f"\n[Round {round_num}] SYNC: {syncs} | CONTINUE: {continues} | WAIT: {waits} | DROPPED: {dropped}")
        print(f"[Round {round_num}] Cell Load: {bw_used:.1f}/{bw_max:.1f} MB")
        print(f"[Round {round_num}] Utility: {util:.4f} | Drift: {drift:.4f}")