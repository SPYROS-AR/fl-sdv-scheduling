import argparse

import torch

from constants import (
    ALPHA,
    BETA,
    DELTA,
    DIRICHLET_ALPHA,
    FEDPROX_MU,
    GAMMA,
    NUM_VEHICLES,
)
from data_processing import load_and_clean_dataset
from data_setup import clone_and_run_dataset
from scheduler import run_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Run federated-learning experiments for SDV scheduling."
    )
    default_gpu = 0.1 if torch.cuda.is_available() else 0.0
    parser.add_argument(
        "--gpus_per_client",
        type=float,
        default=default_gpu,
        help="Fraction of GPU per client",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Number of federated learning rounds",
    )
    parser.add_argument(
        "--dynamic_only",
        action="store_true",
        help="Run only the dynamic FedAvg policy",
    )
    parser.add_argument(
        "--dynamic-fedprox",
        "--dynamic_fedprox",
        action="store_true",
        help="Run only the dynamic FedProx policy",
    )
    parser.add_argument(
        "--fixed",
        type=int,
        help="Run a fixed-epoch FedAvg experiment, e.g. --fixed 3",
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "gtsrb"],
        default="cifar10",
        help="Image dataset for federated learning",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        choices=[0.1, 0.2, 0.3, 0.5],
        default=DIRICHLET_ALPHA,
        help="Dirichlet concentration parameter for non-IID client partitions",
    )
    parser.add_argument(
        "--fedprox-mu",
        type=float,
        default=FEDPROX_MU,
        help="FedProx proximal coefficient (mu)",
    )
    parser.add_argument(
        "--fedprox-epochs",
        type=int,
        default=5,
        help="Local epochs for the fixed FedProx baseline",
    )
    parser.add_argument(
        "--fedprox-only",
        action="store_true",
        help="Run only the fixed-epoch FedProx baseline",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="Network penalty weight in utility function",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=BETA,
        help="Compute interference penalty weight in utility function",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=GAMMA,
        help="Model drift penalty weight in utility function",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=DELTA,
        help="Information gain reward weight in utility function",
    )
    parser.add_argument(
        "--job-id",
        "--job_id",
        dest="job_id",
        default="",
        help="Optional identifier appended to each metrics filename",
    )
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.fixed is not None and args.fixed <= 0:
        parser.error("--fixed must be a positive integer")
    if args.fedprox_epochs <= 0:
        parser.error("--fedprox-epochs must be positive")
    if args.fedprox_mu < 0:
        parser.error("--fedprox-mu must be non-negative")
    if args.alpha < 0:
        parser.error("--alpha must be non-negative")
    if args.beta < 0:
        parser.error("--beta must be non-negative")
    if args.gamma < 0:
        parser.error("--gamma must be non-negative")
    if args.delta < 0:
        parser.error("--delta must be non-negative")

    print("Starting network dataset setup and simulation...")
    clone_and_run_dataset()
    dataframe = load_and_clean_dataset(filepath="dataset/data/network_dataset.csv")

    if args.dynamic_only:
        experiments = ["dynamic"]
    elif args.dynamic_fedprox:
        experiments = ["dynamic_fedprox"]
    elif args.fedprox_only:
        experiments = [f"fedprox_{args.fedprox_epochs}"]
    elif args.fixed is not None:
        experiments = [f"fixed_{args.fixed}"]
    else:
        experiments = [
            "dynamic",
            "fixed_1",
            "fixed_3",
            "fixed_5",
            "fixed_10",
            "fixed_20",
            f"fedprox_{args.fedprox_epochs}",
            "dynamic_fedprox",
        ]

    print(f"Experiments: {experiments}")
    run_simulation(
        dataframe,
        NUM_VEHICLES,
        num_rounds=args.rounds,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        experiments=experiments,
        gpus_per_client=args.gpus_per_client,
        dataset_name=args.dataset,
        dirichlet_alpha=args.dirichlet_alpha,
        fedprox_mu=args.fedprox_mu,
        job_id=args.job_id,
    )


if __name__ == "__main__":
    main()
