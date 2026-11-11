"""Tests for src/server.py: trace mapping, drops, logging (no training)."""

import numpy as np
import pandas as pd
import pytest
from flwr.common import Code, FitRes, GetParametersRes, Status, ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy

from constants import KAPPA_MBIT_PER_MIB
from server import ClientLogger, CSVLogger, DynamicFLStrategy, TraceTable

PAYLOAD = 93.08876037597656 / 10  # MiB per upload in the submitted runs (from the CSVs)
N = 10


def make_df(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    scen = rng.integers(0, 3, n)
    df = pd.DataFrame({
        "u_crit_t": rng.uniform(0.28, 0.72, n),
        "effective_throughput_mbps": rng.uniform(0.5, 80.0, n),
    })
    for k, name in enumerate(["highway", "urban_dense", "rural"]):
        df[f"scenario_{name}"] = (scen == k).astype(int)
    return df


class Proxy:
    def __init__(self, cid):
        self.cid = str(cid)


class FakeClientProxy(ClientProxy):
    def get_properties(self, ins, timeout, group_id):
        raise NotImplementedError

    def get_parameters(self, ins, timeout, group_id):
        return GetParametersRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters([]))

    def fit(self, ins, timeout, group_id):
        raise NotImplementedError

    def evaluate(self, ins, timeout, group_id):
        raise NotImplementedError

    def reconnect(self, ins, timeout, group_id):
        raise NotImplementedError


def fit_result(pid, status="SYNC", uploaded=True, **extra):
    metrics = {
        "status": status, "partition_id": pid, "server_round": 1,
        "epochs_trained": 0 if status == "WAIT" else 3, "c_comp": 0.12, "c_net": 0.5,
        "c_drift": 1.5, "gain": 0.05, "was_in_continue": 0, "continue_count": 0,
        "bandwidth": 30.0, "u_crit": 0.4, "n_train": 400, "n_val": 100, "val_passes": 4,
        "sync_reason": "link" if status == "SYNC" else "", "wait_reason": "",
    }
    metrics.update(extra)
    arrays = [np.full(4, float(pid), dtype=np.float32)] if uploaded else []
    res = FitRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(arrays),
                 num_examples=100 if uploaded else 0, metrics=metrics)
    return Proxy(pid), res


def make_strategy(df=None, **kwargs):
    strategy = DynamicFLStrategy(
        df_network=make_df() if df is None else df, exp_name="t", num_vehicles=N,
        fraction_fit=1.0, min_fit_clients=N, min_available_clients=N, **kwargs,
    )
    strategy.model_payload_mb = PAYLOAD
    return strategy


# --------------------------------------------------------------------------- #
# Trace mapping
# --------------------------------------------------------------------------- #
def test_iid_trace_uses_the_same_samples_per_round_as_the_submitted_code():
    df = make_df()
    table = TraceTable(df, N, "iid")
    for r in (1, 2, 250, 600):
        assert table.rows(r) == [((r - 1) * N + v) % len(df) for v in range(N)]
    u, b = table.values(3)
    assert u == pytest.approx(df["u_crit_t"].iloc[20:30].tolist())
    assert b == pytest.approx(df["effective_throughput_mbps"].iloc[20:30].tolist())


def test_ucrit_offset_is_capped():
    df = make_df()
    u, _ = TraceTable(df, N, "iid", ucrit_offset=0.3).values(1)
    expected = np.minimum(0.95, df["u_crit_t"].iloc[:N].to_numpy() + 0.3)
    assert u == pytest.approx(expected.tolist())


def test_sticky_trace_is_deterministic_and_persistent():
    df = make_df()
    scen = df[[c for c in df.columns if c.startswith("scenario_")]].to_numpy().argmax(1)
    a = TraceTable(df, N, "sticky", seed=1, mean_dwell=10)
    b = TraceTable(df, N, "sticky", seed=1, mean_dwell=10)
    c = TraceTable(df, N, "sticky", seed=2, mean_dwell=10)
    rounds = range(1, 1501)
    rows_a = [a.rows(r) for r in rounds]
    assert rows_a == [b.rows(r) for r in rounds]
    assert rows_a != [c.rows(r) for r in rounds]
    # environment of each vehicle persists: mean run length close to the mean dwell
    runs = []
    for v in range(N):
        seq = [scen[rows[v]] for rows in rows_a]
        length = 1
        for prev, cur in zip(seq, seq[1:]):
            if cur == prev:
                length += 1
            else:
                runs.append(length)
                length = 1
    # a switch can land on the same environment, so observed runs are a bit longer
    assert 9.0 < np.mean(runs) < 18.0
    # random access to an earlier round returns the cached rows
    assert a.rows(5) == rows_a[4]


def test_invalid_trace_mode():
    with pytest.raises(ValueError):
        TraceTable(make_df(), N, "markov")


def test_configure_fit_broadcasts_the_round_vector(in_tmp):
    strategy = make_strategy()
    cm = SimpleClientManager()
    for cid in range(N):
        cm.register(FakeClientProxy(str(cid)))
    params = ndarrays_to_parameters([np.zeros(4, dtype=np.float32)])
    configs = strategy.configure_fit(7, params, cm)
    assert len(configs) == N
    u, b = strategy.trace.values(7)
    for _, fit_ins in configs:
        cfg = fit_ins.config
        assert cfg["server_round"] == 7
        assert [float(x) for x in cfg["trace_u_crit"].split(",")] == pytest.approx(u, abs=1e-6)
        assert [float(x) for x in cfg["trace_bandwidth"].split(",")] == pytest.approx(b, abs=1e-6)


# --------------------------------------------------------------------------- #
# Drops
# --------------------------------------------------------------------------- #
def test_capacity_drops_reproduce_the_submitted_runs(in_tmp):
    """Ten uploads per round for 250 rounds -> 567 drops, as in every fixed baseline."""
    strategy = make_strategy()
    for r in range(1, 251):
        strategy.round_n_clients[r] = N
        strategy.aggregate_fit(r, [fit_result(pid) for pid in range(N)], [])
    assert strategy.total_dropped == 567
    assert strategy.total_syncs == 2500 - 567
    assert strategy.total_uplink_mb == pytest.approx(2500 * PAYLOAD)


def test_drop_selection_does_not_depend_on_result_order(in_tmp):
    r = 3  # 95*(0.8+0.2*sin 3) = 78.7 MiB -> 8 of 10 uploads accepted
    results = [fit_result(pid) for pid in range(N)]
    s1, s2 = make_strategy(seed=5), make_strategy(seed=5)
    s1.round_n_clients[r] = s2.round_n_clients[r] = N
    p1, _ = s1.aggregate_fit(r, list(results), [])
    p2, _ = s2.aggregate_fit(r, list(reversed(results)), [])
    from flwr.common import parameters_to_ndarrays
    assert np.allclose(parameters_to_ndarrays(p1)[0], parameters_to_ndarrays(p2)[0])
    assert s1.total_dropped == s2.total_dropped == 2


def test_deadline_drops_depend_on_the_link(in_tmp):
    strategy = make_strategy(upload_deadline_s=5.0)
    strategy.round_n_clients[1] = 2
    slow = fit_result(0, bandwidth=5.0)   # 8.39*9.31/5  = 15.6 s > 5 s
    fast = fit_result(1, bandwidth=40.0)  # 8.39*9.31/40 =  1.95 s
    strategy.aggregate_fit(1, [slow, fast], [])
    log = pd.read_csv("clients_t.csv")
    assert log.set_index("partition_id").loc[0, "drop_reason"] == "deadline"
    assert log.set_index("partition_id").loc[1, "accepted"] == 1
    assert log.set_index("partition_id").loc[0, "t_up_s"] == pytest.approx(5.0)
    assert log.set_index("partition_id").loc[1, "t_up_s"] == pytest.approx(
        KAPPA_MBIT_PER_MIB * PAYLOAD / 40.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# Logs
# --------------------------------------------------------------------------- #
def test_round_and_client_logs_have_consistent_columns(in_tmp):
    strategy = make_strategy()
    strategy.round_n_clients[1] = N
    results = [fit_result(pid) for pid in range(6)]
    results += [fit_result(6, "CONTINUE", uploaded=False, sync_reason="")]
    results += [fit_result(pid, "WAIT", uploaded=False, wait_reason="link", c_comp=0.0)
                for pid in (7, 8)]
    results += [fit_result(9, "WAIT", uploaded=False, wait_reason="compute", c_comp=0.0)]
    strategy.aggregate_fit(1, results, [])
    strategy.logger.commit_round_metrics(1, 0.9, 0.5, 1.2, 1.2, 0.48)

    rounds = pd.read_csv("metrics_t.csv")
    assert list(rounds.columns) == CSVLogger.HEADER
    row = rounds.iloc[0]
    assert (row.syncs, row.continues, row.waits, row.dropped) == (6, 1, 3, 0)
    assert (row.wait_link, row.wait_compute, row.sync_link) == (2, 1, 6)
    assert row.uplink_mb == pytest.approx(6 * PAYLOAD)
    assert row.transmitted_mb == pytest.approx(16 * PAYLOAD)
    assert row.uplink_straggler_s == pytest.approx(KAPPA_MBIT_PER_MIB * PAYLOAD / 30.0, rel=1e-6)

    clients = pd.read_csv("clients_t.csv")
    assert list(clients.columns) == ClientLogger.HEADER
    assert clients["partition_id"].tolist() == list(range(N))
    assert clients["status"].value_counts().to_dict() == {"SYNC": 6, "WAIT": 3, "CONTINUE": 1}


def test_simulated_compute_time_uses_the_slack_share(in_tmp):
    strategy = make_strategy(sim_samples_per_s=1000.0)
    m = {"epochs_trained": 3, "n_train": 400, "n_val": 100, "val_passes": 4,
         "u_crit": 0.5, "bandwidth": 20.0}
    t_comp, t_up = strategy._sim_times(m, uploaded=True, payload_mb=PAYLOAD)
    work = 3 * 400 + 4 * 100 / 3
    assert t_comp == pytest.approx(work / (1000.0 * 0.5))
    assert t_up == pytest.approx(KAPPA_MBIT_PER_MIB * PAYLOAD / 20.0)
