import logging
import shutil
from collections import OrderedDict
from pathlib import Path

import datasets
import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, GTSRB
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import model
from client import FLVehicle
from constants import (
    ALPHA,
    BATCH_SIZE,
    BETA,
    DELTA,
    DEVICE,
    DIRICHLET_ALPHA,
    FEDPROX_MU,
    GAMMA,
    SEED,
    TEST_SPLIT,
)
from data_setup import DictImageSubset, setup_dataset
from server import DynamicFLStrategy

datasets.utils.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, filename="results.log", filemode="a")


def weighted_accuracy(metrics):
    """Return sample-weighted validation accuracy across evaluated clients."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples <= 0:
        return {"accuracy": 0.0}
    accuracy = (
        sum(
            num_examples * float(client_metrics.get("accuracy", 0.0))
            for num_examples, client_metrics in metrics
        )
        / total_examples
    )
    return {"accuracy": accuracy}


def build_centralized_eval_fn(dataset_name: str = "cifar10"):
    """Return a centralized evaluation function for the global model."""
    dataset_name = dataset_name.lower()
    transforms = []
    if dataset_name == "gtsrb":
        transforms.append(Resize((32, 32)))
        num_classes = 43
    else:
        num_classes = 10
    transforms.extend([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    eval_transforms = Compose(transforms)

    if dataset_name == "gtsrb":
        test_dataset = GTSRB(
            root="dataset/vision",
            split="test",
            download=True,
            transform=eval_transforms,
        )
    else:
        test_dataset = CIFAR10(
            root="dataset/vision", train=False, download=True, transform=eval_transforms
        )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    def evaluate(
        server_round: int,
        parameters_ndarrays: list[np.ndarray],
        config: dict[str, fl.common.Scalar],
    ) -> tuple[float, dict] | None:
        net = model.load_model(num_classes=num_classes)
        state_dict = OrderedDict(
            (key, torch.as_tensor(value, device=DEVICE))
            for key, value in zip(net.state_dict().keys(), parameters_ndarrays)
        )
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE, dtype=torch.long)
                logits = net(images)
                batch_size = labels.size(0)
                total_loss += criterion(logits, labels).item() * batch_size
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_examples += batch_size
        if total_examples == 0:
            return None
        loss = total_loss / total_examples
        accuracy = total_correct / total_examples
        return loss, {"centralized_accuracy": float(accuracy)}

    return evaluate


class ClientFactory:
    def __init__(
        self,
        dataset,
        config: dict,
        dataset_name: str = "cifar10",
        exp_name: str = "dynamic",
    ):
        self.dataset = dataset
        self.config = config
        self.dataset_name = dataset_name.lower()
        self.exp_name = exp_name
        transforms = []
        if self.dataset_name == "gtsrb":
            transforms.append(Resize((32, 32)))
        transforms.extend([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transforms = Compose(transforms)
        self.clients = {}

    def _apply_transforms(self, batch):
        batch["img"] = [self.transforms(image) for image in batch["img"]]
        return batch

    def build_client(self, cid: str) -> fl.client.Client:
        client_id = int(cid)
        if client_id in self.clients:
            return self.clients[client_id].to_client()

        if self.dataset_name == "gtsrb":
            indices = np.asarray(self.dataset.client_indices[client_id]).copy()
            rng = np.random.default_rng(SEED + client_id)
            rng.shuffle(indices)
            if len(indices) < 2:
                raise ValueError(
                    f"GTSRB client {client_id} has fewer than two examples."
                )
            num_val = max(1, int(round(len(indices) * TEST_SPLIT)))
            num_val = min(num_val, len(indices) - 1)
            val_indices, train_indices = indices[:num_val], indices[num_val:]
            train_dataset = DictImageSubset(
                self.dataset.train_dataset, train_indices, self.transforms
            )
            val_dataset = DictImageSubset(
                self.dataset.train_dataset, val_indices, self.transforms
            )
            num_classes = 43
        else:
            partition = self.dataset.load_partition(
                client_id, "train"
            ).train_test_split(
                test_size=TEST_SPLIT,
                seed=SEED,
            )
            partition = partition.with_transform(self._apply_transforms)
            train_dataset = partition["train"]
            val_dataset = partition["test"]
            num_classes = 10

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        vehicle = FLVehicle(
            client_id,
            train_loader,
            val_loader,
            num_classes=num_classes,
            algorithm=self.config["algorithm"],
            fedprox_mu=self.config["fedprox_mu"],
            exp_name=self.exp_name,
        )
        vehicle.alpha = self.config["alpha"]
        vehicle.beta = self.config["beta"]
        vehicle.gamma = self.config["gamma"]
        vehicle.delta = self.config["delta"]
        vehicle.mode = self.config["mode"]
        self.clients[client_id] = vehicle
        return vehicle.to_client()


def run_simulation(
    df: pd.DataFrame,
    num_vehicles: int,
    num_rounds: int = 50,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
    experiments=None,
    gpus_per_client: float = 0.1,
    job_id: str = "",
    dataset_name: str = "cifar10",
    dirichlet_alpha: float = DIRICHLET_ALPHA,
    fedprox_mu: float = FEDPROX_MU,
):
    """Run each experiment against the same federated partition and network trace."""
    if experiments is None:
        experiments = ["dynamic"]
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive.")
    if fedprox_mu < 0:
        raise ValueError("fedprox_mu must be non-negative.")

    print(
        f"Loading federated dataset: {dataset_name}, Dirichlet alpha={dirichlet_alpha}"
    )
    federated_dataset = setup_dataset(
        dataset_name=dataset_name,
        alpha=dirichlet_alpha,
    )

    all_results = {}
    for exp in experiments:
        exp_name = f"{exp}_{job_id}" if job_id else exp

        # Clean any stale disk cache from previous runs of the same experiment name
        cache_dir = Path(".client_cache") / exp_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        if exp == "dynamic_fedprox":
            mode = "dynamic"
            algorithm = "fedprox"
        elif exp.startswith("fedprox_"):
            local_epochs = int(exp.split("_", 1)[1])
            if local_epochs <= 0:
                raise ValueError(f"Invalid FedProx epoch count in experiment '{exp}'.")
            mode = f"fixed_{local_epochs}"
            algorithm = "fedprox"
        elif exp == "dynamic" or exp.startswith("fixed_"):
            mode = exp
            algorithm = "fedavg"
        else:
            raise ValueError(f"Unsupported experiment name: {exp}")

        client_config = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "mode": mode,
            "algorithm": algorithm,
            "fedprox_mu": fedprox_mu,
        }

        factory = ClientFactory(
            federated_dataset,
            client_config,
            dataset_name=dataset_name,
            exp_name=exp_name,
        )

        strategy = DynamicFLStrategy(
            df_network=df,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            exp_name=exp_name,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_vehicles,
            min_available_clients=num_vehicles,
            evaluate_metrics_aggregation_fn=weighted_accuracy,
            evaluate_fn=build_centralized_eval_fn(dataset_name),
        )

        history = fl.simulation.start_simulation(
            client_fn=factory.build_client,
            num_clients=num_vehicles,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": gpus_per_client},
        )

        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)

        losses = (
            [value for _, value in history.losses_distributed]
            if getattr(history, "losses_distributed", None)
            else []
        )
        all_results[exp] = {
            "mbs": strategy.total_transmitted_mb,
            "uplink_mbs": strategy.total_uplink_mb,
            "network_cost": strategy.total_c_net,
            "losses": losses,
            "interference": strategy.total_c_comp,
            "drift": strategy.total_c_drift,
            "utility": strategy.total_utility,
            "waits": strategy.total_waits,
            "continues": strategy.total_continues,
            "syncs": strategy.total_syncs,
            "dropped": strategy.total_dropped,
            "dataset": dataset_name,
            "dirichlet_alpha": dirichlet_alpha,
            "algorithm": algorithm,
            "mode": mode,
            "fedprox_mu": fedprox_mu if algorithm == "fedprox" else 0.0,
        }
    return all_results
