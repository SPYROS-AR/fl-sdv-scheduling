import logging
import re
import shutil
from collections import OrderedDict
from pathlib import Path

import datasets
import flwr as fl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor

import model
import policy
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
    SIM_SAMPLES_PER_SECOND,
    STICKY_MEAN_DWELL,
    TEST_SPLIT,
)
from server import DynamicFLStrategy

datasets.utils.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, filename="results.log", filemode="a")


# --------------------------------------------------------------------------- #
# Experiment names
# --------------------------------------------------------------------------- #
EXPERIMENT_HELP = (
    "dynamic | dynamic_nostop | dynamic_nocontinue | fixed_E | fedprox_E | "
    "fixedcap_E | slackfill | threshold_E | randpart_E_P (P in percent); "
    "append _fedprox to any name for the FedProx variant"
)


def parse_experiment(exp: str) -> tuple[dict, str]:
    """Map an experiment name to (scheme spec, algorithm)."""
    algorithm = "fedavg"
    name = exp
    if name.endswith("_fedprox"):
        algorithm, name = "fedprox", name[: -len("_fedprox")]
    spec = {"kind": None, "epochs": None, "participation": 1.0, "ablation": ""}
    if name == "dynamic":
        spec["kind"] = "dynamic"
    elif name in ("dynamic_nostop", "dynamic_nocontinue"):
        spec["kind"] = "dynamic"
        spec["ablation"] = "no_stop" if name.endswith("nostop") else "no_continue"
    elif name == "slackfill":
        spec["kind"] = "slackfill"
    elif m := re.fullmatch(r"(fixed|fedprox|fixedcap|threshold)_(\d+)", name):
        kind, epochs = m.group(1), int(m.group(2))
        if kind == "fedprox":
            kind, algorithm = "fixed", "fedprox"
        spec.update(kind=kind, epochs=epochs)
    elif m := re.fullmatch(r"randpart_(\d+)_(\d+)", name):
        spec.update(kind="randpart", epochs=int(m.group(1)), participation=int(m.group(2)) / 100)
    else:
        raise ValueError(f"Unsupported experiment name '{exp}'. Use: {EXPERIMENT_HELP}")
    if spec["epochs"] is not None and spec["epochs"] <= 0:
        raise ValueError(f"Epoch count must be positive in '{exp}'.")
    if not 0.0 < spec["participation"] <= 1.0:
        raise ValueError(f"Participation must be in (0, 100] percent in '{exp}'.")
    return spec, algorithm


def weighted_accuracy(metrics):
    """Return sample-weighted validation accuracy across evaluated clients."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples <= 0:
        return {"accuracy": 0.0}
    accuracy = sum(n * float(m.get("accuracy", 0.0)) for n, m in metrics) / total_examples
    return {"accuracy": accuracy}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class SyntheticFederatedDataset:
    """Tiny label-skewed random-image dataset for smoke tests (no download)."""

    def __init__(self, num_partitions: int, per_partition: int = 60, seed: int = 0):
        from PIL import Image

        rng = np.random.default_rng(seed)
        self.parts = {}
        for pid in range(num_partitions):
            main = [pid % 10, (pid + 1) % 10]
            labels = [int(rng.choice(main)) if rng.random() < 0.8 else int(rng.integers(10))
                      for _ in range(per_partition)]
            images = [Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
                      for _ in range(per_partition)]
            self.parts[pid] = datasets.Dataset.from_dict({"img": images, "label": labels})

    def load_partition(self, partition_id: int, split: str = "train"):
        return self.parts[int(partition_id)]


def build_centralized_eval_fn(dataset_name: str = "cifar10", synthetic: bool = False):
    """Return a centralized evaluation function for the global model."""
    if synthetic:
        gen = torch.Generator().manual_seed(0)
        test_dataset = TensorDataset(torch.randn(200, 3, 32, 32, generator=gen),
                                     torch.randint(0, 10, (200,), generator=gen))
    else:
        from torchvision.datasets import CIFAR10

        eval_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset = CIFAR10(root="dataset/vision", train=False, download=True,
                               transform=eval_transforms)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    def evaluate(server_round, parameters_ndarrays, config):
        net = model.load_model(num_classes=10)
        state_dict = OrderedDict(
            (key, torch.as_tensor(value, device=DEVICE))
            for key, value in zip(net.state_dict().keys(), parameters_ndarrays)
        )
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss, total_correct, total_examples = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE, dtype=torch.long)
                logits = net(images)
                total_loss += criterion(logits, labels).item() * labels.size(0)
                total_correct += (logits.argmax(dim=1) == labels).sum().item()
                total_examples += labels.size(0)
        if total_examples == 0:
            return None
        return total_loss / total_examples, {"centralized_accuracy": total_correct / total_examples}

    return evaluate


class ClientFactory:
    def __init__(self, dataset, config: dict, dataset_name: str = "cifar10",
                 exp_name: str = "dynamic"):
        self.dataset = dataset
        self.config = config
        self.dataset_name = dataset_name.lower()
        self.exp_name = exp_name
        self.transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.clients = {}

    def _apply_transforms(self, batch):
        batch["img"] = [self.transforms(image) for image in batch["img"]]
        return batch

    def build_client(self, cid: str) -> fl.client.Client:
        client_id = int(cid)
        if client_id in self.clients:
            return self.clients[client_id].to_client()
        partition = self.dataset.load_partition(client_id, "train").train_test_split(
            test_size=TEST_SPLIT, seed=SEED
        )
        partition = partition.with_transform(self._apply_transforms)
        # A dedicated generator, re-seeded per (seed, vehicle, round, epoch) by the client
        train_loader = DataLoader(partition["train"], batch_size=BATCH_SIZE, shuffle=True,
                                  generator=torch.Generator())
        val_loader = DataLoader(partition["test"], batch_size=BATCH_SIZE, shuffle=False)
        vehicle = FLVehicle(
            client_id, train_loader, val_loader, num_classes=10,
            algorithm=self.config["algorithm"], fedprox_mu=self.config["fedprox_mu"],
            exp_name=self.exp_name, params=self.config["params"], spec=self.config["spec"],
            seed=self.config["seed"],
        )
        self.clients[client_id] = vehicle
        return vehicle.to_client()


def initial_parameters(seed: int) -> fl.common.Parameters:
    """Seeded initial global model (the server no longer asks a random client)."""
    torch.manual_seed(policy.derive_seed(seed, 1))
    net = model.load_model(num_classes=10)
    return fl.common.ndarrays_to_parameters(
        [v.detach().cpu().numpy() for v in net.state_dict().values()]
    )


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
    seed: int = 0,
    params: policy.PolicyParams | None = None,
    trace_mode: str = "iid",
    sticky_mean_dwell: float = STICKY_MEAN_DWELL,
    ucrit_offset: float = 0.0,
    upload_deadline_s: float | None = None,
    sim_samples_per_s: float = SIM_SAMPLES_PER_SECOND,
    synthetic_data: bool = False,
    tag: str = "",
):
    """Run each experiment against the same federated partition and network trace."""
    experiments = experiments or ["dynamic"]
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive.")
    if fedprox_mu < 0:
        raise ValueError("fedprox_mu must be non-negative.")
    params = params or policy.PolicyParams(beta=beta, gamma=gamma, delta=delta)
    if (params.beta, params.gamma, params.delta) != (beta, gamma, delta):
        raise ValueError("PolicyParams weights must match beta/gamma/delta.")
    specs = {exp: parse_experiment(exp) for exp in experiments}  # fail fast on typos

    if synthetic_data:
        print("Using a synthetic dataset (smoke test; no download)")
        federated_dataset = SyntheticFederatedDataset(num_vehicles, seed=seed)
    else:
        from data_setup import setup_dataset

        print(f"Loading federated dataset: {dataset_name}, Dirichlet alpha={dirichlet_alpha}")
        federated_dataset = setup_dataset(dataset_name=dataset_name, alpha=dirichlet_alpha)
    eval_fn = build_centralized_eval_fn(dataset_name, synthetic=synthetic_data)

    all_results = {}
    for exp in experiments:
        label = f"{exp}{tag}"
        exp_name = f"{label}_{job_id}" if job_id else label
        cache_dir = Path(".client_cache") / exp_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        spec, algorithm = specs[exp]
        client_config = {"algorithm": algorithm, "fedprox_mu": fedprox_mu, "params": params,
                         "spec": spec, "seed": seed}
        factory = ClientFactory(federated_dataset, client_config, dataset_name, exp_name)
        strategy = DynamicFLStrategy(
            df_network=df, alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            exp_name=exp_name, num_vehicles=num_vehicles, seed=seed, trace_mode=trace_mode,
            sticky_mean_dwell=sticky_mean_dwell, ucrit_offset=ucrit_offset,
            upload_deadline_s=upload_deadline_s, sim_samples_per_s=sim_samples_per_s,
            fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=num_vehicles,
            min_evaluate_clients=num_vehicles, min_available_clients=num_vehicles,
            evaluate_metrics_aggregation_fn=weighted_accuracy, evaluate_fn=eval_fn,
            initial_parameters=initial_parameters(seed),
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
        losses = [v for _, v in getattr(history, "losses_distributed", None) or []]
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
            "sim_time_s": strategy.total_sim_time_s,
            "dataset": dataset_name,
            "dirichlet_alpha": dirichlet_alpha,
            "algorithm": algorithm,
            "mode": spec["kind"],
            "fedprox_mu": fedprox_mu if algorithm == "fedprox" else 0.0,
        }
    return all_results
