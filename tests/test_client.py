"""Tests for src/client.py with real (CPU) training on tiny synthetic data."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import policy  # noqa: E402
from client import FLVehicle  # noqa: E402

N_VEHICLES = 4


class Tiny(Dataset):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 3, 32, 32, generator=g)
        self.y = torch.randint(0, 3, (n,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"img": self.x[i], "label": int(self.y[i])}


def make_vehicle(pid=0, params=None, spec=None, seed=0, exp="t", algorithm="fedavg"):
    train = DataLoader(Tiny(64, 1 + pid), batch_size=32, shuffle=True,
                       generator=torch.Generator())
    val = DataLoader(Tiny(32, 100 + pid), batch_size=32, shuffle=False)
    return FLVehicle(pid, train, val, params=params, spec=spec, seed=seed, exp_name=exp,
                     algorithm=algorithm)


def cfg(rnd, pid, u, b):
    us, bs = [0.3] * N_VEHICLES, [50.0] * N_VEHICLES
    us[pid], bs[pid] = u, b
    return {"server_round": rnd, "trace_u_crit": ",".join(map(str, us)),
            "trace_bandwidth": ",".join(map(str, bs))}


def spec(kind, epochs=None, participation=1.0, ablation=""):
    return {"kind": kind, "epochs": epochs, "participation": participation,
            "ablation": ablation}


@pytest.fixture()
def global_params():
    torch.manual_seed(0)
    return make_vehicle(pid=3, exp="init").get_parameters()


def test_dynamic_syncs_on_a_good_link(in_tmp, global_params):
    v = make_vehicle()
    params, n, m = v.fit(global_params, cfg(1, 0, 0.4, 50.0))
    assert m["status"] == "SYNC" and m["sync_reason"] == policy.SYNC_LINK
    assert n == 64 and len(params) == len(global_params)
    assert 1 <= m["epochs_trained"] <= 6  # slack budget floor(10 * 0.6) = 6
    assert m["val_passes"] == m["epochs_trained"] + 1
    assert m["bandwidth"] == 50.0 and m["u_crit"] == 0.4 and m["partition_id"] == 0
    assert m["c_comp"] == pytest.approx(m["epochs_trained"] / 10 * 0.4)


def test_compute_starvation_waits(in_tmp, global_params):
    _, n, m = make_vehicle().fit(global_params, cfg(1, 0, 0.85, 50.0))
    assert (m["status"], m["wait_reason"], n, m["c_comp"]) == ("WAIT", "compute", 0, 0.0)


def test_wait_keeps_a_pending_continue_trajectory(in_tmp, global_params):
    """Audit fix 5.2: WAIT used to discard the weights of a CONTINUE trajectory."""
    p = policy.PolicyParams(gain_threshold=10.0)  # any prior gain counts as "low"
    v1 = make_vehicle(params=p)
    _, n, m1 = v1.fit(global_params, cfg(1, 0, 0.4, 5.0))  # weak link, no prior gain
    assert (m1["status"], n, m1["continue_count"]) == ("CONTINUE", 0, 1)
    local = [t.detach().clone() for t in v1.model.state_dict().values()]

    v2 = make_vehicle(params=p)  # new object, as after Ray actor recycling
    _, _, m2 = v2.fit(global_params, cfg(2, 0, 0.4, 5.0))  # weak link, low prior gain
    assert (m2["status"], m2["wait_reason"], m2["was_in_continue"]) == ("WAIT", "link", 1)
    assert v2._in_continue and v2._continue_count == 2  # the WAIT counts towards K_max
    saved = torch.load(v2._state_file, weights_only=False)
    assert all(torch.equal(a, b) for a, b in zip(local, saved["model_state"].values()))

    v3 = make_vehicle(params=p)
    restored = list(v3.model.state_dict().values())
    assert all(torch.equal(a, b) for a, b in zip(local, restored))  # resumes local weights
    _, n3, m3 = v3.fit(global_params, cfg(3, 0, 0.4, 50.0))  # link recovers
    assert (m3["status"], m3["sync_reason"], m3["was_in_continue"]) == ("SYNC", "link", 1)
    assert n3 == 64 and not v3._in_continue and v3._continue_count == 0


def test_seeded_runs_are_reproducible(in_tmp, global_params):
    s = spec("fixed", epochs=1)
    a, _, _ = make_vehicle(spec=s, seed=7, exp="a").fit(global_params, cfg(4, 0, 0.4, 50.0))
    b, _, _ = make_vehicle(spec=s, seed=7, exp="b").fit(global_params, cfg(4, 0, 0.4, 50.0))
    c, _, _ = make_vehicle(spec=s, seed=8, exp="c").fit(global_params, cfg(4, 0, 0.4, 50.0))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, z) for x, z in zip(a, c))


@pytest.mark.parametrize("kind,epochs,u,b,expected_status,expected_epochs,reason", [
    ("fixed", 2, 0.85, 50.0, "SYNC", 2, "always"),  # ignores the slack
    ("fixedcap", 10, 0.75, 50.0, "SYNC", 2, "always"),  # floor(10 * 0.25) = 2
    ("slackfill", None, 0.75, 50.0, "SYNC", 2, "always"),
    ("threshold", 2, 0.4, 5.0, "WAIT", 0, "link"),
    ("threshold", 2, 0.4, 50.0, "SYNC", 2, "always"),
    ("fixedcap", 3, 0.95, 50.0, "WAIT", 0, "compute"),  # floor(10 * 0.05) = 0
])
def test_baselines(in_tmp, global_params, kind, epochs, u, b, expected_status,
                   expected_epochs, reason):
    _, n, m = make_vehicle(spec=spec(kind, epochs)).fit(global_params, cfg(1, 0, u, b))
    assert (m["status"], m["epochs_trained"]) == (expected_status, expected_epochs)
    assert (m["sync_reason"] if m["status"] == "SYNC" else m["wait_reason"]) == reason
    assert (n > 0) == (expected_status == "SYNC")


def test_random_participation_is_deterministic(in_tmp, global_params):
    v = make_vehicle(spec=spec("randpart", epochs=1, participation=0.5), seed=3)
    statuses = [v.fit(global_params, cfg(r, 0, 0.4, 50.0))[2]["status"] for r in range(1, 9)]
    expected = ["SYNC" if policy.participates(3, r, 0, 0.5) else "WAIT" for r in range(1, 9)]
    assert statuses == expected and "SYNC" in statuses and "WAIT" in statuses


def test_no_stop_ablation_trains_the_whole_budget(in_tmp, global_params):
    v = make_vehicle(spec=spec("dynamic", ablation="no_stop"))
    _, _, m = v.fit(global_params, cfg(1, 0, 0.6, 50.0))
    assert m["epochs_trained"] == 4 and m["val_passes"] == 2


def test_no_continue_ablation_waits_on_weak_links(in_tmp, global_params):
    v = make_vehicle(spec=spec("dynamic", ablation="no_continue"))
    _, n, m = v.fit(global_params, cfg(1, 0, 0.4, 5.0))
    assert (m["status"], m["wait_reason"], n) == ("WAIT", "link", 0)


@pytest.mark.parametrize("estimate", ["last", "curve"])
def test_utility_stop_rule_and_fedprox_run(in_tmp, global_params, estimate):
    p = policy.PolicyParams(stop_rule="utility", tau_continue=0.75, gain_estimate=estimate)
    v = make_vehicle(params=p, algorithm="fedprox")
    _, n, m = v.fit(global_params, cfg(1, 0, 0.4, 50.0))
    assert m["status"] == "SYNC" and m["epochs_trained"] >= 1 and n == 64


def test_unknown_scheme_rejected(in_tmp):
    with pytest.raises(ValueError):
        make_vehicle(spec=spec("greedy", epochs=1))
