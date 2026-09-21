import os
import sys
import subprocess
import flwr_datasets
from constants import DATASET, NUM_SAMPLES, NUM_VEHICLES, DIRICHLET_ALPHA

class EnvironmentBuilder:
    """Manages external repository cloning and dependency provisioning."""
    REPO_URL = "https://github.com/icsa-hua/CLOUDNET2026-QoS-Offloading.git"
    TARGET_FOLDER = "dataset"

    @classmethod
    def _run_command(cls, command: list, cwd: str = None, quiet: bool = False):
        """Wrapper for robust subprocess execution."""
        kwargs = {"cwd": cwd, "check": True}
        if quiet:
            kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT})
        try:
            subprocess.run(command, **kwargs)
        except subprocess.CalledProcessError as e:
            print(f"Command execution failed: {e}")
            sys.exit(1)

    @classmethod
    def setup_external_repo(cls):
        if not os.path.exists(cls.TARGET_FOLDER):
            print(f"Cloning repository from {cls.REPO_URL}...")
            cls._run_command(["git", "clone", cls.REPO_URL, cls.TARGET_FOLDER])

        print("Installing repository dependencies...")
        cls._run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=cls.TARGET_FOLDER)

        print("Generating network dataset...")
        cls._run_command(
            [sys.executable, "-W", "ignore", "-m", "src.dataset.generate", "--n", str(NUM_SAMPLES)],
            cwd=cls.TARGET_FOLDER, quiet=True
        )
        print("External setup complete!")

def clone_and_run_dataset():
    """Legacy entrypoint mapping."""
    EnvironmentBuilder.setup_external_repo()

def setup_dataset() -> flwr_datasets.FederatedDataset:
    """Provisions the federated data chunks via Dirichlet distribution."""
    partitioner = flwr_datasets.partitioner.DirichletPartitioner(
        num_partitions=NUM_VEHICLES,
        partition_by="label",
        alpha=DIRICHLET_ALPHA,
        min_partition_size=10
    )
    return flwr_datasets.FederatedDataset(dataset=DATASET, partitioners={"train": partitioner})