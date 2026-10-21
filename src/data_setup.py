import os
import subprocess
import sys

import flwr_datasets
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import GTSRB

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
        if not os.path.exists(cls.TARGET_FOLDER):
            print(f"Cloning repository from {cls.REPO_URL}...")
            cls._run_command(["git", "clone", cls.REPO_URL, cls.TARGET_FOLDER])

        # The network dataset generator only needs the core project dependencies
        # (NumPy, pandas, and PyTorch). Installing the external repository's full
        # requirements also installs optional YOLO/OpenCV tooling, which this FL
        # experiment does not use and which can fail on some Python versions.
        # Install this project's root requirements.txt once in the active venv.
        dataset_path = os.path.join(cls.TARGET_FOLDER, "data", "network_dataset.csv")
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


def dirichlet_partition_indices(
    labels,
    num_partitions: int,
    alpha: float,
    min_partition_size: int = 10,
    seed: int = SEED,
):
    """Create deterministic label-skewed partitions using a symmetric Dirichlet."""
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least 1.")

    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)

    for _ in range(1000):
        partitions = [[] for _ in range(num_partitions)]
        for class_id in np.unique(labels):
            class_indices = np.flatnonzero(labels == class_id)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(
                np.full(num_partitions, alpha, dtype=np.float64)
            )
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(
                int
            )
            for client_id, indices in enumerate(np.split(class_indices, split_points)):
                partitions[client_id].extend(indices.tolist())

        if min(map(len, partitions)) >= min_partition_size:
            for indices in partitions:
                rng.shuffle(indices)
            return partitions

    raise RuntimeError(
        "Could not create partitions meeting min_partition_size after 1000 attempts. "
        "Try a larger alpha or a smaller minimum size."
    )


class DictImageSubset(Dataset):
    """Expose a torchvision subset in the {'img': ..., 'label': ...} batch format."""

    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = list(map(int, indices))
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base_dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return {"img": image, "label": int(label)}


class GTSRBFederatedDataset:
    """GTSRB training data split across clients with a Dirichlet label skew."""

    def __init__(
        self,
        num_partitions: int,
        alpha: float,
        root: str = "dataset/vision",
        seed: int = SEED,
    ):
        self.train_dataset = GTSRB(root=root, split="train", download=True)

        labels = getattr(self.train_dataset, "targets", None)
        if labels is None:
            samples = getattr(self.train_dataset, "_samples", None)
            if samples is None:
                raise RuntimeError(
                    "Unable to read GTSRB training labels from this torchvision version."
                )
            labels = [target for _, target in samples]

        self.client_indices = dirichlet_partition_indices(
            labels=labels,
            num_partitions=num_partitions,
            alpha=alpha,
            min_partition_size=10,
            seed=seed,
        )


def setup_dataset(
    dataset_name: str = "cifar10",
    alpha: float = DIRICHLET_ALPHA,
):
    """Build the requested federated dataset with a configurable Dirichlet alpha."""
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive.")

    name = dataset_name.lower()
    if name == "gtsrb":
        return GTSRBFederatedDataset(
            num_partitions=NUM_VEHICLES,
            alpha=alpha,
            seed=SEED,
        )

    if name not in {"cifar10", DATASET.lower()}:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Choose 'cifar10' or 'gtsrb'."
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
