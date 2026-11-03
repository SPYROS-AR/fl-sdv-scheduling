import os
import subprocess
import sys

import flwr_datasets

from constants import DATASET, DIRICHLET_ALPHA, NUM_SAMPLES, NUM_VEHICLES, SEED


class EnvironmentBuilder:
    """Manages external repository cloning and dependency provisioning."""

    REPO_URL = "https://github.com/icsa-hua/CLOUDNET2026-QoS-Offloading.git"
    TARGET_FOLDER = "dataset"

    @classmethod
    def _run_command(cls, command: list, cwd: str = None, quiet: bool = False):
        kwargs = {"cwd": cwd, "check": True}
        if quiet:
            kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT})
        try:
            subprocess.run(command, **kwargs)
        except subprocess.CalledProcessError as exc:
            print(f"Command execution failed: {exc}")
            raise

    @classmethod
    def setup_external_repo(cls):
        dataset_path = os.path.join(cls.TARGET_FOLDER, "data", "network_dataset.csv")
        if os.path.isfile(dataset_path):
            print(f"Using existing network dataset: {dataset_path}")
            return
        if os.path.exists(cls.TARGET_FOLDER) and not os.path.isdir(
            os.path.join(cls.TARGET_FOLDER, ".git")
        ):
            import shutil

            print(
                f"Directory '{cls.TARGET_FOLDER}' exists but lacks git repository data. Re-initializing..."
            )
            shutil.rmtree(cls.TARGET_FOLDER)
        if not os.path.exists(cls.TARGET_FOLDER):
            print(f"Cloning repository from {cls.REPO_URL}...")
            cls._run_command(["git", "clone", cls.REPO_URL, cls.TARGET_FOLDER])
        if os.path.isfile(dataset_path):
            print(f"Using existing network dataset: {dataset_path}")
            return
        print("Generating network dataset (skipping unused optional dependencies)...")
        cls._run_command(
            [
                sys.executable,
                "-W",
                "ignore",
                "-m",
                "src.dataset.generate",
                "--n",
                str(NUM_SAMPLES),
            ],
            cwd=cls.TARGET_FOLDER,
            quiet=False,
        )
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                "The network dataset generator completed without creating "
                f"{dataset_path}. Check the generator output above."
            )
        print("External setup complete!")


def clone_and_run_dataset():
    """Provision the network-condition dataset used by the SDV simulation."""
    EnvironmentBuilder.setup_external_repo()


def setup_dataset(
    dataset_name: str = "cifar10",
    alpha: float = DIRICHLET_ALPHA,
):
    """Build the requested federated dataset with a configurable Dirichlet alpha."""
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    name = dataset_name.lower()
    if name not in {"cifar10", DATASET.lower()}:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Choose 'cifar10'."
        )
    partitioner = flwr_datasets.partitioner.DirichletPartitioner(
        num_partitions=NUM_VEHICLES,
        partition_by="label",
        alpha=alpha,
        min_partition_size=10,
        self_balancing=False,
        seed=SEED,
    )
    return flwr_datasets.FederatedDataset(
        dataset=DATASET,
        partitioners={"train": partitioner},
    )
