"""Tests for the command-line layer: experiment names, main.py wiring, result aggregation."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

import main
import policy
from scheduler import parse_experiment

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Experiment names
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,kind,epochs,part,ablation,algorithm", [
    ("dynamic", "dynamic", None, 1.0, "", "fedavg"),
    ("dynamic_fedprox", "dynamic", None, 1.0, "", "fedprox"),
    ("dynamic_nostop", "dynamic", None, 1.0, "no_stop", "fedavg"),
    ("dynamic_nocontinue_fedprox", "dynamic", None, 1.0, "no_continue", "fedprox"),
    ("fixed_5", "fixed", 5, 1.0, "", "fedavg"),
    ("fedprox_5", "fixed", 5, 1.0, "", "fedprox"),
    ("fixedcap_10", "fixedcap", 10, 1.0, "", "fedavg"),
    ("slackfill", "slackfill", None, 1.0, "", "fedavg"),
    ("threshold_5", "threshold", 5, 1.0, "", "fedavg"),
    ("randpart_5_72", "randpart", 5, 0.72, "", "fedavg"),
    ("slackfill_fedprox", "slackfill", None, 1.0, "", "fedprox"),
])
def test_experiment_names(name, kind, epochs, part, ablation, algorithm):
    spec, alg = parse_experiment(name)
    assert (spec["kind"], spec["epochs"], spec["participation"], spec["ablation"], alg) == (
        kind, epochs, part, ablation, algorithm)


@pytest.mark.parametrize("name", ["greedy", "fixed_0", "fixed_x", "randpart_5_0",
                                  "randpart_5_150", "dynamic_fast"])
def test_bad_experiment_names(name):
    with pytest.raises(ValueError):
        parse_experiment(name)


# --------------------------------------------------------------------------- #
# main.py wiring (run_simulation replaced by a stub)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def captured(in_tmp, monkeypatch):
    data = in_tmp / "dataset" / "data"
    data.mkdir(parents=True)
    pd.DataFrame({"u_crit_t": [0.4] * 20, "effective_throughput_mbps": [30.0] * 20}).to_csv(
        data / "cleaned_network_dataset.csv", index=False)
    calls = {}

    def fake_run_simulation(df, num_vehicles, **kwargs):
        calls.update(kwargs, num_vehicles=num_vehicles, rows=len(df))
        return {}

    monkeypatch.setattr(main, "run_simulation", fake_run_simulation)
    return calls


def test_defaults_reproduce_the_submitted_configuration(captured):
    main.main(["--dynamic_only", "--job-id", "a01"])
    p = captured["params"]
    assert captured["experiments"] == ["dynamic"] and captured["job_id"] == "a01"
    assert (p.stop_rule, p.gain_estimate, p.tau_continue) == ("legacy", "last", 0.1)
    assert (p.delta, p.beta, p.gamma, p.gain_threshold, p.bw_threshold) == (
        27.0, 1.0, 0.006, 0.03, 20.0)
    assert (captured["trace_mode"], captured["ucrit_offset"], captured["tag"]) == ("iid", 0.0, "")
    assert captured["upload_deadline_s"] is None and captured["seed"] == 0


def test_revised_configuration_and_seed_suffix(captured):
    main.main(["--experiments", "dynamic", "slackfill", "--job-id", "a01", "--seed", "3",
               "--stop-rule", "utility", "--gain-estimate", "curve", "--tag=-rev",
               "--trace-mode", "sticky", "--upload-deadline-s", "4"])
    p = captured["params"]
    assert captured["job_id"] == "a01_s3" and captured["seed"] == 3
    assert (p.stop_rule, p.gain_estimate, p.tau_continue) == ("utility", "curve", 0.75)
    assert captured["experiments"] == ["dynamic", "slackfill"]
    assert (captured["tag"], captured["trace_mode"], captured["upload_deadline_s"]) == (
        "-rev", "sticky", 4.0)


@pytest.mark.parametrize("argv", [["--tag=a_b"], ["--rounds", "0"], ["--delta", "0"],
                                  ["--gain-estimate", "median"]])
def test_invalid_arguments_are_rejected(captured, argv):
    with pytest.raises(SystemExit):
        main.main(argv)


# --------------------------------------------------------------------------- #
# Aggregation (numbers of the revised paper, from the committed CSVs)
# --------------------------------------------------------------------------- #
def load_aggregator():
    spec = importlib.util.spec_from_file_location(
        "aggregate_results", REPO / "scripts" / "aggregate_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_file_name_pattern():
    agg = load_aggregator()
    m = agg.NAME.match("metrics_dynamic-rev_a01_s3.csv")
    assert (m["exp"], m["alpha"], m["seed"]) == ("dynamic-rev", "a01", "3")
    m = agg.NAME.match("metrics_randpart_5_72_a05.csv")
    assert (m["exp"], m["alpha"], m["seed"]) == ("randpart_5_72", "a05", None)
    assert agg.alpha_value("a01") == 0.1 and agg.alpha_value("a05") == 0.5
    assert agg.NAME.match("clients_dynamic_a01_s3.csv") is None


@pytest.mark.skipif(not (REPO / "final_metrics").is_dir(), reason="final_metrics/ not present")
def test_aggregator_reproduces_the_revised_tables(tmp_path):
    agg = load_aggregator()
    agg.main([str(REPO / "final_metrics"), "--out", str(tmp_path), "--targets", "70", "75"])
    s = pd.read_csv(tmp_path / "summary.csv").set_index(["exp", "alpha"])
    expected = {
        ("dynamic_fedprox", 0.1, "acc_lastk_mean"): 73.74,
        ("fixed_10", 0.1, "acc_lastk_mean"): 73.11,
        ("fixed_1", 0.5, "acc_lastk_mean"): 79.28,
        ("dynamic_fedprox", 0.1, "uplink_to_70_mean"): 3332.6,
        ("fixed_5", 0.1, "uplink_to_70_mean"): 3630.5,
        ("fixed_5", 0.5, "uplink_to_75_mean"): 1303.2,
        ("dynamic", 0.1, "uplink_mib_mean"): 16718.7,
        ("fixed_5", 0.1, "drops_mean"): 567,
        ("dynamic", 0.1, "compute_mean"): 385.2,
        ("dynamic", 0.1, "drift_per_update_mean"): 1.81,
        ("fedprox_5", 0.5, "drift_per_update_mean"): 0.81,
    }
    for (exp, alpha, col), value in expected.items():
        tolerance = 0.01 if value < 100 else 0.1  # rounding of the published values
        assert s.loc[(exp, alpha), col] == pytest.approx(value, abs=tolerance), (exp, alpha, col)
    assert (tmp_path / "table_results.tex").read_text().count("\\\\") >= 18
