import argparse
import json
import os

import pandas as pd
import torch

from constants import NUM_VEHICLES
from scheduler import run_simulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=50)
    default_gpu = 0.025 if torch.cuda.is_available() else 0.0
    parser.add_argument("--gpus_per_client", type=float, default=default_gpu)
    parser.add_argument("--dataset", choices=["cifar10", "gtsrb"], default="cifar10")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.2)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    args = parser.parse_args()

    df = pd.read_csv("dataset/data/cleaned_network_dataset.csv")
    results = run_simulation(
        df,
        NUM_VEHICLES,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        experiments=["dynamic"],
        gpus_per_client=args.gpus_per_client,
        job_id=args.job_id,
        num_rounds=args.rounds,
        dataset_name=args.dataset,
        dirichlet_alpha=args.dirichlet_alpha,
        fedprox_mu=args.fedprox_mu,
    )

    results["dynamic"]["params"] = {
        "job_id": args.job_id,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "delta": args.delta,
        "dataset": args.dataset,
        "dirichlet_alpha": args.dirichlet_alpha,
        "fedprox_mu": args.fedprox_mu,
    }

    output_dir = "sweep_outputs"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"result_{args.job_id}.json")
    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Worker {args.job_id} finished successfully. Saved to {out_file}")
