import csv
import math
import os
import random
import time

import flwr as fl
import pandas as pd
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
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
    """Round-level CSV logger with explicit communication/time/accuracy fields."""

    HEADER = [
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
        "transmitted_mb",
        "total_transmitted_mb",
        "mean_model_drift",
        "round_time_s",
        "wall_time_s",
        "val_loss",
        "global_accuracy",
    ]

    def __init__(self, exp_name: str):
        self.csv_file = f"metrics_{exp_name}.csv"
        self.pending_metrics = {}
        # Start a clean run file so old schemas or duplicated rounds are not appended.
        with open(self.csv_file, "w", newline="") as file:
            csv.writer(file).writerow(self.HEADER)

    def stage_round_metrics(self, round_num: int, row: list):
        self.pending_metrics[round_num] = row

    def commit_round_metrics(
        self,
        round_num: int,
        val_loss,
        global_accuracy,
        round_time_s: float,
        wall_time_s: float,
    ):
        if round_num not in self.pending_metrics:
            return
        row = self.pending_metrics.pop(round_num)
        row.extend(
            [
                round_time_s,
                wall_time_s,
                val_loss if val_loss is not None else "",
                global_accuracy if global_accuracy is not None else "",
            ]
        )
        with open(self.csv_file, "a", newline="") as file:
            csv.writer(file).writerow(row)


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
        if df_network.empty:
            raise ValueError("Network-condition dataframe must not be empty.")
        required_columns = {"u_crit_t", "effective_throughput_mbps"}
        missing = required_columns.difference(df_network.columns)
        if missing:
            raise ValueError(
                f"Network dataframe missing required columns: {sorted(missing)}"
            )

        self.df_network = df_network.reset_index(drop=True)
        self.alpha, self.beta = alpha, beta
        self.gamma, self.delta = gamma, delta
        self.logger = CSVLogger(exp_name)

        self.start_time = time.perf_counter()
        self.round_started_at = {}
        self.round_downlink_mb = {}
        self.model_payload_mb = 0.0
        self.total_transmitted_mb = 0.0

        self.total_c_net = self.total_c_comp = self.total_c_drift = 0.0
        self.total_gain = self.total_utility = 0.0
        self.total_syncs = self.total_continues = self.total_waits = 0
        self.total_dropped = 0

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        base_configs = super().configure_fit(server_round, parameters, client_manager)
        self.round_started_at[server_round] = time.perf_counter()

        # Raw parameter-array payload size (MiB; labeled MB in output for readability).
        arrays = parameters_to_ndarrays(parameters)
        self.model_payload_mb = sum(array.nbytes for array in arrays) / (1024**2)
        downlink_mb = len(base_configs) * self.model_payload_mb
        self.round_downlink_mb[server_round] = downlink_mb
        self.total_transmitted_mb += downlink_mb

        custom_configs = []
        n_clients = len(base_configs)
        if n_clients == 0:
            return custom_configs

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
        drift_sum = 0.0
        drift_clients = 0
        upload_clients = 0

        # Randomized admission order under the modeled cell-capacity envelope.
        current_b_max = TOTAL_BANDWIDTH * (0.8 + 0.2 * math.sin(server_round))
        random.shuffle(results)

        for client, res in results:
            metrics = res.metrics
            status = metrics.get("status", "WAIT")
            r_c_comp += float(metrics.get("c_comp", 0.0))
            r_gain += float(metrics.get("gain", 0.0))

            if int(metrics.get("epochs_trained", 0)) > 0:
                drift_sum += float(metrics.get("c_drift", 0.0))
                drift_clients += 1

            # A non-empty update reached the server, even if admission later drops it.
            if res.num_examples > 0:
                upload_clients += 1

            if res.num_examples == 0:
                if status == "WAIT":
                    r_waits += 1
                elif status == "CONTINUE":
                    r_continues += 1
            else:
                payload = (
                    self.model_payload_mb
                    if self.model_payload_mb > 0
                    else MODEL_SIZE_MB
                )
                if current_bw_used + payload <= current_b_max:
                    valid_results.append((client, res))
                    r_syncs += 1
                    current_bw_used += payload
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

        downlink_mb = self.round_downlink_mb.pop(server_round, 0.0)
        upload_mb = upload_clients * self.model_payload_mb
        round_transmitted_mb = downlink_mb + upload_mb
        self.total_transmitted_mb += upload_mb
        mean_model_drift = drift_sum / drift_clients if drift_clients else ""

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
                round_transmitted_mb,
                self.total_transmitted_mb,
                mean_model_drift,
            ],
        )

        if not valid_results:
            return None, {}
        return super().aggregate_fit(server_round, valid_results, failures)

    def aggregate_evaluate(self, server_round: int, results, failures):
        agg_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        now = time.perf_counter()
        round_start = self.round_started_at.pop(server_round, now)
        round_time_s = now - round_start
        wall_time_s = now - self.start_time
        self.logger.commit_round_metrics(
            round_num=server_round,
            val_loss=agg_loss,
            global_accuracy=metrics.get("accuracy"),
            round_time_s=round_time_s,
            wall_time_s=wall_time_s,
        )
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
