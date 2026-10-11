import csv
import math
import os
import random

import flwr as fl
import pandas as pd
from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

from constants import (
    BW_THRESHOLD,
    GAIN_THRESHOLD,
    MAX_ALLOWED_DRIFT,
    MODEL_SIZE_MB,
    TAU_CONTINUE,
    TOTAL_BANDWIDTH,
)


class CSVLogger:
    def __init__(self, exp_name: str):
        self.csv_file = f"metrics_{exp_name}.csv"
        self.pending_metrics = {}
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "round",
                        "syncs",
                        "continues",
                        "waits",
                        "dropped",
                        "c_net",
                        "c_comp",
                        "c_drift",
                        "gain",
                        "utility",
                        "total_c_net",
                        "total_c_comp",
                        "total_c_drift",
                        "total_gain",
                        "total_utility",
                        "val_loss",
                    ]
                )

    def stage_round_metrics(self, round_num: int, row: list):
        self.pending_metrics[round_num] = row

    def commit_round_metrics(self, round_num: int, val_loss: float | None):
        if round_num in self.pending_metrics:
            row = self.pending_metrics.pop(round_num)
            row.append(val_loss if val_loss is not None else "")
            with open(self.csv_file, "a", newline="") as f:
                csv.writer(f).writerow(row)


class DynamicFLStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        df_network: pd.DataFrame,
        alpha: float = 0.002,
        beta: float = 1.0,
        gamma: float = 0.05,
        delta: float = 20.0,
        exp_name: str = "dynamic",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.df_network = df_network
        self.alpha, self.beta = alpha, beta
        self.gamma, self.delta = gamma, delta
        self.logger = CSVLogger(exp_name)

        self.total_c_net = self.total_c_comp = self.total_c_drift = 0.0
        self.total_gain = self.total_utility = 0.0
        self.total_syncs = self.total_continues = self.total_waits = (
            self.total_dropped
        ) = 0

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        base_configs = super().configure_fit(server_round, parameters, client_manager)
        custom_configs = []
        n_clients = len(base_configs)

        for idx, (proxy, fit_ins) in enumerate(base_configs):
            row_idx = ((server_round - 1) * n_clients + idx) % len(self.df_network)
            row = self.df_network.iloc[row_idx]
            cfg = fit_ins.config.copy()
            cfg.update(
                {
                    "u_crit": float(row["u_crit_t"]),
                    "bandwidth": float(row["effective_throughput_mbps"]),
                    "tau_continue": TAU_CONTINUE,
                    "drift_threshold": MAX_ALLOWED_DRIFT,
                    "gain_threshold": GAIN_THRESHOLD,
                    "bw_threshold": BW_THRESHOLD,
                }
            )
            custom_configs.append(
                (proxy, FitIns(parameters=fit_ins.parameters, config=cfg))
            )

        return custom_configs

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        valid_results = []
        current_bw_used = 0.0
        r_syncs = r_continues = r_waits = r_dropped = 0
        r_c_net = r_c_comp = r_c_drift = r_gain = 0.0

        # Periodic cell capacity envelope to evaluate behavior under fluctuating channel conditions
        current_b_max = TOTAL_BANDWIDTH * (0.8 + 0.2 * math.sin(server_round))
        random.shuffle(results)

        for client, res in results:
            metrics = res.metrics
            status = metrics.get("status", "WAIT")
            r_c_comp += float(metrics.get("c_comp", 0.0))
            r_gain += float(metrics.get("gain", 0.0))

            if res.num_examples == 0:
                if status == "WAIT":
                    r_waits += 1
                elif status == "CONTINUE":
                    r_continues += 1
            else:
                if current_bw_used + MODEL_SIZE_MB <= current_b_max:
                    valid_results.append((client, res))
                    r_syncs += 1
                    current_bw_used += MODEL_SIZE_MB
                    r_c_net += float(metrics.get("c_net", 0.0))
                    r_c_drift += float(metrics.get("c_drift", 0.0))
                else:
                    r_dropped += 1

        r_utility = (
            (self.delta * r_gain)
            - (self.alpha * r_c_net)
            - (self.beta * r_c_comp)
            - (self.gamma * r_c_drift)
        )

        self._record_round(
            r_syncs,
            r_continues,
            r_waits,
            r_dropped,
            r_c_net,
            r_c_comp,
            r_c_drift,
            r_gain,
            r_utility,
        )

        self.logger.stage_round_metrics(
            server_round,
            [
                server_round,
                r_syncs,
                r_continues,
                r_waits,
                r_dropped,
                r_c_net,
                r_c_comp,
                r_c_drift,
                r_gain,
                r_utility,
                self.total_c_net,
                self.total_c_comp,
                self.total_c_drift,
                self.total_gain,
                self.total_utility,
            ],
        )

        if not valid_results:
            return None, {}
        return super().aggregate_fit(server_round, valid_results, failures)

    def aggregate_evaluate(self, server_round: int, results, failures):
        agg_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        self.logger.commit_round_metrics(server_round, agg_loss)
        return agg_loss, metrics

    def _record_round(
        self, syncs, continues, waits, dropped, net, comp, drift, gain, util
    ):
        self.total_syncs += syncs
        self.total_continues += continues
        self.total_waits += waits
        self.total_dropped += dropped
        self.total_c_net += net
        self.total_c_comp += comp
        self.total_c_drift += drift
        self.total_gain += gain
        self.total_utility += util
