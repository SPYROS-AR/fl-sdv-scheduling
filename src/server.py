import csv
import math
import random
import time

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import FitIns, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

import policy
from constants import (
    ALPHA,
    BETA,
    DELTA,
    GAMMA,
    KAPPA_MBIT_PER_MIB,
    MODEL_SIZE_MB,
    NUM_VEHICLES,
    SIM_SAMPLES_PER_SECOND,
    STICKY_MEAN_DWELL,
    TOTAL_BANDWIDTH,
    VAL_COST_RATIO,
)


class CSVLogger:
    """Round-level CSV logger (one row per round)."""

    HEADER = [
        "round", "syncs", "continues", "waits", "dropped",
        "c_net", "c_comp", "c_drift", "gain", "utility",
        "total_c_net", "total_c_comp", "total_c_drift", "total_gain", "total_utility",
        "uplink_mb", "total_uplink_mb", "transmitted_mb", "total_transmitted_mb",
        "mean_model_drift",
        # added by the audit fixes
        "dropped_capacity", "dropped_deadline",
        "sync_link", "sync_drift", "sync_gain", "sync_staleness",
        "wait_compute", "wait_link", "wait_not_selected",
        "uplink_airtime_s", "uplink_straggler_s", "sim_round_time_s", "sim_time_s",
        # committed after evaluation
        "round_time_s", "wall_time_s", "val_loss", "global_accuracy", "centralized_accuracy",
    ]

    def __init__(self, exp_name: str):
        self.csv_file = f"metrics_{exp_name}.csv"
        self.pending_metrics = {}
        with open(self.csv_file, "w", newline="") as file:
            csv.writer(file).writerow(self.HEADER)

    def stage_round_metrics(self, round_num: int, row: list):
        self.pending_metrics[round_num] = row

    def commit_round_metrics(self, round_num, val_loss, global_accuracy, round_time_s,
                             wall_time_s, centralized_accuracy=None):
        if round_num not in self.pending_metrics:
            return
        row = self.pending_metrics.pop(round_num)
        row.extend([
            round_time_s, wall_time_s,
            val_loss if val_loss is not None else "",
            global_accuracy if global_accuracy is not None else "",
            centralized_accuracy if centralized_accuracy is not None else "",
        ])
        with open(self.csv_file, "a", newline="") as file:
            csv.writer(file).writerow(row)


class ClientLogger:
    """Per-vehicle, per-round decisions (clients_<exp>.csv)."""

    HEADER = [
        "round", "partition_id", "status", "accepted", "drop_reason", "epochs",
        "n_train", "n_val", "val_passes", "bandwidth_mbps", "u_crit", "drift", "gain",
        "c_comp", "c_net", "was_in_continue", "continue_count", "sync_reason",
        "wait_reason", "t_comp_s", "t_up_s",
    ]

    def __init__(self, exp_name: str):
        self.csv_file = f"clients_{exp_name}.csv"
        with open(self.csv_file, "w", newline="") as file:
            csv.writer(file).writerow(self.HEADER)

    def write(self, rows: list[list]):
        with open(self.csv_file, "a", newline="") as file:
            csv.writer(file).writerows(rows)


def scenario_codes(df: pd.DataFrame) -> np.ndarray:
    """Integer scenario label per trace row (from one-hot or raw column)."""
    cols = [c for c in df.columns if c.startswith("scenario_")]
    if cols:
        return df[cols].to_numpy().argmax(axis=1)
    if "scenario" in df.columns:
        return pd.factorize(df["scenario"])[0]
    return np.zeros(len(df), dtype=int)


class TraceTable:
    """Maps (round, vehicle) to a trace sample.

    iid    : sample ((round-1)*N + vehicle) mod len(df), i.e. the same samples per round
             as the original code, but assigned by partition id instead of Flower's
             random sampling order.
    sticky : each vehicle stays in a road environment for a geometric dwell time
             (mean ``mean_dwell`` rounds) and draws its samples from that environment,
             so weak-link and high-load periods persist across rounds.
    """

    def __init__(self, df, num_vehicles, mode="iid", seed=0, mean_dwell=STICKY_MEAN_DWELL,
                 ucrit_offset=0.0):
        if mode not in ("iid", "sticky"):
            raise ValueError("trace mode must be 'iid' or 'sticky'")
        self.u = np.minimum(0.95, df["u_crit_t"].to_numpy(float) + float(ucrit_offset))
        self.b = df["effective_throughput_mbps"].to_numpy(float)
        self.n = int(num_vehicles)
        self.mode = mode
        self._cache = {}
        if mode == "sticky":
            self._rng = np.random.default_rng(policy.derive_seed(seed, 4242))
            codes = scenario_codes(df)
            self._labels = np.unique(codes)
            self._probs = np.array([(codes == s).mean() for s in self._labels])
            self._pools = {s: np.flatnonzero(codes == s) for s in self._labels}
            self._perm = {}
            self._state = []
            for v in range(self.n):
                for s in self._labels:
                    self._perm[(v, s)] = [self._rng.permutation(self._pools[s]), 0]
                s0 = self._rng.choice(self._labels, p=self._probs)
                self._state.append([s0, int(self._rng.geometric(1.0 / mean_dwell))])
            self._mean_dwell = mean_dwell
            self._generated = 0

    def rows(self, server_round: int) -> list[int]:
        if server_round in self._cache:
            return self._cache[server_round]
        if self.mode == "iid":
            rows = [((server_round - 1) * self.n + v) % len(self.u) for v in range(self.n)]
            self._cache[server_round] = rows
            return rows
        while self._generated < server_round:
            self._generated += 1
            rows = []
            for v in range(self.n):
                s, remaining = self._state[v]
                if remaining <= 0:
                    s = self._rng.choice(self._labels, p=self._probs)
                    remaining = int(self._rng.geometric(1.0 / self._mean_dwell))
                perm, ptr = self._perm[(v, s)]
                rows.append(int(perm[ptr % len(perm)]))
                self._perm[(v, s)][1] = ptr + 1
                self._state[v] = [s, remaining - 1]
            self._cache[self._generated] = rows
        return self._cache[server_round]

    def values(self, server_round: int):
        rows = self.rows(server_round)
        return [float(self.u[r]) for r in rows], [float(self.b[r]) for r in rows]


def _fmt(values) -> str:
    return ",".join(f"{v:.6f}" for v in values)


class DynamicFLStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        df_network: pd.DataFrame,
        alpha: float = ALPHA,
        beta: float = BETA,
        gamma: float = GAMMA,
        delta: float = DELTA,
        exp_name: str = "dynamic",
        *args,
        num_vehicles: int = NUM_VEHICLES,
        seed: int = 0,
        trace_mode: str = "iid",
        sticky_mean_dwell: float = STICKY_MEAN_DWELL,
        ucrit_offset: float = 0.0,
        upload_deadline_s: float | None = None,
        sim_samples_per_s: float = SIM_SAMPLES_PER_SECOND,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if df_network.empty:
            raise ValueError("Network-condition dataframe must not be empty.")
        missing = {"u_crit_t", "effective_throughput_mbps"}.difference(df_network.columns)
        if missing:
            raise ValueError(f"Network dataframe missing required columns: {sorted(missing)}")
        self.df_network = df_network.reset_index(drop=True)
        self.trace = TraceTable(self.df_network, num_vehicles, trace_mode, seed,
                                sticky_mean_dwell, ucrit_offset)
        self.alpha, self.beta = alpha, beta
        self.gamma, self.delta = gamma, delta
        self.seed = int(seed)
        self.upload_deadline_s = upload_deadline_s
        self.sim_samples_per_s = float(sim_samples_per_s)
        self.logger = CSVLogger(exp_name)
        self.client_logger = ClientLogger(exp_name)
        self.start_time = time.perf_counter()
        self.round_started_at = {}
        self.round_n_clients = {}
        self.model_payload_mb = 0.0

        self.total_uplink_mb = 0.0
        self.total_transmitted_mb = 0.0
        self.total_c_net = self.total_c_comp = self.total_c_drift = 0.0
        self.total_gain = self.total_utility = 0.0
        self.total_syncs = self.total_continues = self.total_waits = 0
        self.total_dropped = 0
        self.total_sim_time_s = 0.0
        self.last_centralized_accuracy: float | None = None

    def evaluate(self, server_round: int, parameters: Parameters):
        result = super().evaluate(server_round, parameters)
        self.last_centralized_accuracy = result[1].get("centralized_accuracy") if result else None
        return result

    def configure_fit(self, server_round, parameters, client_manager):
        base_configs = super().configure_fit(server_round, parameters, client_manager)
        self.round_started_at[server_round] = time.perf_counter()
        arrays = parameters_to_ndarrays(parameters)
        self.model_payload_mb = sum(array.nbytes for array in arrays) / (1024**2)
        self.round_n_clients[server_round] = len(base_configs)
        u_values, b_values = self.trace.values(server_round)
        custom_configs = []
        for proxy, fit_ins in base_configs:
            cfg = fit_ins.config.copy()
            cfg.update({
                "server_round": server_round,
                "trace_u_crit": _fmt(u_values),
                "trace_bandwidth": _fmt(b_values),
            })
            custom_configs.append((proxy, FitIns(parameters=fit_ins.parameters, config=cfg)))
        return custom_configs

    def _sim_times(self, metrics, uploaded: bool, payload_mb: float):
        epochs = int(metrics.get("epochs_trained", 0))
        work = epochs * int(metrics.get("n_train", 0)) + (
            int(metrics.get("val_passes", 0)) * int(metrics.get("n_val", 0)) * VAL_COST_RATIO
        )
        share = max(0.05, 1.0 - float(metrics.get("u_crit", 0.0)))
        t_comp = work / (self.sim_samples_per_s * share)
        t_up = (KAPPA_MBIT_PER_MIB * payload_mb / max(0.1, float(metrics.get("bandwidth", 0.1)))
                if uploaded else 0.0)
        return t_comp, t_up

    def aggregate_fit(self, server_round, results, failures):
        valid_results = []
        current_bw_used = 0.0
        r = dict(syncs=0, continues=0, waits=0, dropped=0, dropped_capacity=0,
                 dropped_deadline=0, sync_link=0, sync_drift=0, sync_gain=0,
                 sync_staleness=0, wait_compute=0, wait_link=0, wait_not_selected=0)
        r_c_net = r_c_comp = r_c_drift = r_gain = 0.0
        drift_sum, drift_clients, upload_clients = 0.0, 0, 0
        airtime, straggler, round_sim = 0.0, 0.0, 0.0
        payload = self.model_payload_mb if self.model_payload_mb > 0 else MODEL_SIZE_MB
        current_b_max = TOTAL_BANDWIDTH * (0.8 + 0.2 * math.sin(server_round))

        # Deterministic order: by partition id, then a seeded shuffle for drop selection
        ordered = sorted(results, key=lambda cr: (int(cr[1].metrics.get("partition_id", -1)), cr[0].cid))
        random.Random(policy.derive_seed(self.seed, server_round)).shuffle(ordered)

        client_rows = []
        for client, res in ordered:
            m = res.metrics
            status = m.get("status", "WAIT")
            r_c_comp += float(m.get("c_comp", 0.0))
            if int(m.get("epochs_trained", 0)) > 0:
                drift_sum += float(m.get("c_drift", 0.0))
                drift_clients += 1
            uploaded = res.num_examples > 0
            t_comp, t_up = self._sim_times(m, uploaded, payload)
            accepted, drop_reason = "", ""

            if not uploaded:
                if status == "WAIT":
                    r["waits"] += 1
                    key = f"wait_{m.get('wait_reason', '')}"
                    if key in r:
                        r[key] += 1
                elif status == "CONTINUE":
                    r["continues"] += 1
                    r_gain += float(m.get("gain", 0.0))
                    r_c_drift += float(m.get("c_drift", 0.0))
            else:
                upload_clients += 1
                r_c_net += float(m.get("c_net", 0.0))
                r_c_drift += float(m.get("c_drift", 0.0))
                if self.upload_deadline_s is not None and t_up > self.upload_deadline_s:
                    t_up = float(self.upload_deadline_s)
                    accepted, drop_reason = 0, "deadline"
                    r["dropped"] += 1
                    r["dropped_deadline"] += 1
                elif current_bw_used + payload <= current_b_max:
                    valid_results.append((client, res))
                    current_bw_used += payload
                    accepted = 1
                    r["syncs"] += 1
                    r_gain += float(m.get("gain", 0.0))
                    key = f"sync_{m.get('sync_reason', '')}"
                    if key in r:
                        r[key] += 1
                else:
                    accepted, drop_reason = 0, "capacity"
                    r["dropped"] += 1
                    r["dropped_capacity"] += 1
                airtime += t_up
                straggler = max(straggler, t_up)
            round_sim = max(round_sim, t_comp + t_up)
            client_rows.append([
                server_round, m.get("partition_id", ""), status, accepted, drop_reason,
                m.get("epochs_trained", 0), m.get("n_train", ""), m.get("n_val", ""),
                m.get("val_passes", ""), m.get("bandwidth", ""), m.get("u_crit", ""),
                m.get("c_drift", 0.0), m.get("gain", 0.0), m.get("c_comp", 0.0),
                m.get("c_net", 0.0), m.get("was_in_continue", 0), m.get("continue_count", ""),
                m.get("sync_reason", ""), m.get("wait_reason", ""), round(t_comp, 4), round(t_up, 4),
            ])
        self.client_logger.write(sorted(client_rows, key=lambda row: (row[1] if row[1] != "" else -1)))

        r_utility = (self.delta * r_gain - self.alpha * r_c_net
                     - self.beta * r_c_comp - self.gamma * r_c_drift)
        self.total_syncs += r["syncs"]
        self.total_continues += r["continues"]
        self.total_waits += r["waits"]
        self.total_dropped += r["dropped"]
        self.total_c_net += r_c_net
        self.total_c_comp += r_c_comp
        self.total_c_drift += r_c_drift
        self.total_gain += r_gain
        self.total_utility += r_utility
        self.total_sim_time_s += round_sim

        n_configured = self.round_n_clients.pop(server_round, len(results))
        downlink_mb = n_configured * self.model_payload_mb
        uplink_mb = upload_clients * self.model_payload_mb
        self.total_uplink_mb += uplink_mb
        self.total_transmitted_mb += downlink_mb + uplink_mb
        mean_model_drift = drift_sum / drift_clients if drift_clients else ""

        self.logger.stage_round_metrics(server_round, [
            server_round, r["syncs"], r["continues"], r["waits"], r["dropped"],
            r_c_net, r_c_comp, r_c_drift, r_gain, r_utility,
            self.total_c_net, self.total_c_comp, self.total_c_drift, self.total_gain,
            self.total_utility, uplink_mb, self.total_uplink_mb, downlink_mb + uplink_mb,
            self.total_transmitted_mb, mean_model_drift,
            r["dropped_capacity"], r["dropped_deadline"], r["sync_link"], r["sync_drift"],
            r["sync_gain"], r["sync_staleness"], r["wait_compute"], r["wait_link"],
            r["wait_not_selected"], airtime, straggler, round_sim, self.total_sim_time_s,
        ])

        if not valid_results:
            return None, {}
        return super().aggregate_fit(server_round, valid_results, failures)

    def aggregate_evaluate(self, server_round: int, results, failures):
        agg_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        now = time.perf_counter()
        round_start = self.round_started_at.pop(server_round, now)
        self.logger.commit_round_metrics(
            round_num=server_round, val_loss=agg_loss,
            global_accuracy=metrics.get("accuracy"),
            round_time_s=now - round_start, wall_time_s=now - self.start_time,
            centralized_accuracy=self.last_centralized_accuracy,
        )
        return agg_loss, metrics
