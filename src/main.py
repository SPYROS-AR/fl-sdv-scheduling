import argparse
import os

import pandas as pd
import torch

import policy
from constants import (
    ALPHA,
    BETA,
    BW_THRESHOLD,
    DELTA,
    DIRICHLET_ALPHA,
    FEDPROX_MU,
    GAIN_THRESHOLD,
    GAMMA,
    MAX_ALLOWED_DRIFT,
    MAX_CONTINUE_ROUNDS,
    NUM_VEHICLES,
    SIM_SAMPLES_PER_SECOND,
    STICKY_MEAN_DWELL,
    TAU_CONTINUE,
    TAU_CONTINUE_UTILITY,
)
from scheduler import EXPERIMENT_HELP, run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run federated-learning experiments for SDV scheduling."
    )
    default_gpu = 0.1 if torch.cuda.is_available() else 0.0
    parser.add_argument("--gpus_per_client", type=float, default=default_gpu,
                        help="Fraction of GPU per client")
    parser.add_argument("--rounds", type=int, default=50, help="Number of FL rounds")
    parser.add_argument("--dynamic_only", action="store_true", help="Run only dynamic FedAvg")
    parser.add_argument("--dynamic-fedprox", "--dynamic_fedprox", action="store_true",
                        help="Run only dynamic FedProx")
    parser.add_argument("--fixed", type=int, help="Run fixed-epoch FedAvg, e.g. --fixed 3")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help=f"Explicit experiment list: {EXPERIMENT_HELP}")
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    parser.add_argument("--dirichlet-alpha", type=float, choices=[0.1, 0.2, 0.3, 0.5],
                        default=DIRICHLET_ALPHA)
    parser.add_argument("--fedprox-mu", type=float, default=FEDPROX_MU)
    parser.add_argument("--fedprox-epochs", type=int, default=5)
    parser.add_argument("--fedprox-only", action="store_true")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Network weight (accounting only)")
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--job-id", "--job_id", dest="job_id", default="",
                        help="Identifier appended to each metrics filename (e.g. a01)")
    # --- audit fixes ---
    parser.add_argument("--seed", type=int, default=None,
                        help="Run seed (init, data order, drops). Appends _s<seed> to file names.")
    parser.add_argument("--num-vehicles", type=int, default=NUM_VEHICLES)
    parser.add_argument("--stop-rule", choices=list(policy.STOP_RULES), default="legacy",
                        help="legacy = submitted runs; utility = endogenous gain floor")
    parser.add_argument("--gain-estimate", choices=list(policy.GAIN_ESTIMATES), default="last",
                        help="last = submitted runs; curve = slope of the fitted curve")
    parser.add_argument("--tau-continue", type=float, default=None,
                        help=f"Default {TAU_CONTINUE} (legacy) or {TAU_CONTINUE_UTILITY} (utility)")
    parser.add_argument("--bw-threshold", type=float, default=BW_THRESHOLD, help="Mbit/s")
    parser.add_argument("--gain-threshold", type=float, default=GAIN_THRESHOLD)
    parser.add_argument("--drift-threshold", type=float, default=MAX_ALLOWED_DRIFT)
    parser.add_argument("--max-continue-rounds", type=int, default=MAX_CONTINUE_ROUNDS)
    parser.add_argument("--trace-mode", choices=["iid", "sticky"], default="iid")
    parser.add_argument("--sticky-dwell", type=float, default=STICKY_MEAN_DWELL,
                        help="Mean rounds per road environment in sticky mode")
    parser.add_argument("--ucrit-offset", type=float, default=0.0,
                        help="Stress test: add to U_crit (capped at 0.95)")
    parser.add_argument("--upload-deadline-s", type=float, default=None,
                        help="Drop uploads whose airtime exceeds this many seconds")
    parser.add_argument("--sim-samples-per-s", type=float, default=SIM_SAMPLES_PER_SECOND,
                        help="Training throughput assumed by the simulated V2X clock")
    parser.add_argument("--tag", default="",
                        help="Suffix for the experiment name in output files, e.g. --tag=-rev "
                             "-> metrics_dynamic-rev_a01_s3.csv")
    parser.add_argument("--synthetic-data", action="store_true",
                        help="Random images instead of CIFAR-10 (smoke tests only)")
    return parser


def select_experiments(args) -> list[str]:
    if args.experiments:
        return list(args.experiments)
    if args.dynamic_only:
        return ["dynamic"]
    if args.dynamic_fedprox:
        return ["dynamic_fedprox"]
    if args.fedprox_only:
        return [f"fedprox_{args.fedprox_epochs}"]
    if args.fixed is not None:
        return [f"fixed_{args.fixed}"]
    return ["dynamic", "fixed_1", "fixed_3", "fixed_5", "fixed_10", "fixed_20",
            f"fedprox_{args.fedprox_epochs}", "dynamic_fedprox"]


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("rounds", "fedprox_epochs", "num_vehicles", "max_continue_rounds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.fixed is not None and args.fixed <= 0:
        parser.error("--fixed must be a positive integer")
    for name in ("fedprox_mu", "alpha", "beta", "gamma", "bw_threshold", "gain_threshold"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.delta <= 0:
        parser.error("--delta must be positive")

    tau = args.tau_continue
    if tau is None:
        tau = TAU_CONTINUE_UTILITY if args.stop_rule == "utility" else TAU_CONTINUE
    params = policy.PolicyParams(
        beta=args.beta, gamma=args.gamma, delta=args.delta, tau_continue=tau,
        gain_threshold=args.gain_threshold, drift_threshold=args.drift_threshold,
        bw_threshold=args.bw_threshold, max_continue_rounds=args.max_continue_rounds,
        stop_rule=args.stop_rule, gain_estimate=args.gain_estimate,
    )
    if "_" in args.tag:
        parser.error("--tag must not contain '_' (use '-', e.g. -rev)")
    job_id = args.job_id
    seed = 0 if args.seed is None else args.seed
    if args.seed is not None:
        job_id = f"{job_id}_s{args.seed}" if job_id else f"s{args.seed}"

    print("Starting network dataset setup and simulation...")
    cleaned_path = "dataset/data/cleaned_network_dataset.csv"
    if os.path.isfile(cleaned_path):
        print(f"Using pre-processed network dataset: {cleaned_path}")
        dataframe = pd.read_csv(cleaned_path)
    else:
        from data_processing import load_and_clean_dataset
        from data_setup import clone_and_run_dataset

        clone_and_run_dataset()
        dataframe = load_and_clean_dataset(filepath="dataset/data/network_dataset.csv")

    experiments = select_experiments(args)
    print(f"Experiments: {experiments} | seed={seed} | stop rule={args.stop_rule} "
          f"| gain estimate={args.gain_estimate}")
    return run_simulation(
        dataframe, args.num_vehicles, num_rounds=args.rounds, alpha=args.alpha,
        beta=args.beta, gamma=args.gamma, delta=args.delta, experiments=experiments,
        gpus_per_client=args.gpus_per_client, dataset_name=args.dataset,
        dirichlet_alpha=args.dirichlet_alpha, fedprox_mu=args.fedprox_mu, job_id=job_id,
        seed=seed, params=params, trace_mode=args.trace_mode,
        sticky_mean_dwell=args.sticky_dwell, ucrit_offset=args.ucrit_offset,
        upload_deadline_s=args.upload_deadline_s, sim_samples_per_s=args.sim_samples_per_s,
        synthetic_data=args.synthetic_data, tag=args.tag,
    )


if __name__ == "__main__":
    main()
