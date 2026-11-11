from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import model
import policy
from constants import (
    DEVICE,
    FEDPROX_MU,
    LEARNING_RATE,
    MODEL_SIZE_MB,
    MOMENTUM,
)


def _default_spec(mode: str = "dynamic") -> dict:
    return {"kind": mode, "epochs": None, "participation": 1.0, "ablation": ""}


class FLVehicle(fl.client.NumPyClient):
    """One SDV client.

    ``spec`` describes the scheduling scheme (built by scheduler.parse_experiment):
    kind in {"dynamic", "fixed", "fixedcap", "slackfill", "threshold", "randpart"},
    epochs (baselines), participation (randpart), ablation in {"", "no_stop",
    "no_continue"} (dynamic only).
    """

    def __init__(
        self,
        partition_id: int,
        trainloader: DataLoader,
        valloader: DataLoader,
        epochs_per_compute: int = 10,
        num_classes: int = 10,
        algorithm: str = "fedavg",
        fedprox_mu: float = FEDPROX_MU,
        exp_name: str = "dynamic",
        params: policy.PolicyParams | None = None,
        spec: dict | None = None,
        seed: int = 0,
    ):
        self.id = int(partition_id)
        self.trainloader = trainloader
        self.valloader = valloader
        self.M = MODEL_SIZE_MB
        self.rho = int(epochs_per_compute)
        self.params = params or policy.PolicyParams(rho=self.rho)
        self.spec = spec or _default_spec()
        self.seed = int(seed)
        self.model = model.load_model(num_classes=num_classes).to(DEVICE)
        self.num_classes = num_classes
        self._temp_global_model = model.load_model(num_classes=num_classes).to(DEVICE)
        self._in_continue = False
        self._last_gain = None
        self._continue_count = 0
        self.criterion = nn.CrossEntropyLoss()
        self.algorithm = algorithm.lower()
        self.fedprox_mu = float(fedprox_mu)
        self.exp_name = exp_name

        if self.algorithm not in {"fedavg", "fedprox"}:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        if self.fedprox_mu < 0:
            raise ValueError("FedProx mu must be non-negative.")
        if self.spec["kind"] != "dynamic" and self.spec["kind"] not in policy.BASELINE_KINDS:
            raise ValueError(f"Unsupported scheme: {self.spec['kind']}")

        self._cache_dir = Path(".client_cache") / self.exp_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._cache_dir / f"client_{self.id}.pt"
        self._load_persistent_state()

    # ------------------------------------------------------------------ #
    # Persistent state (survives Ray actor recycling)
    # ------------------------------------------------------------------ #
    def _load_persistent_state(self):
        if self._state_file.is_file():
            try:
                ckpt = torch.load(self._state_file, map_location=DEVICE, weights_only=False)
                self._in_continue = bool(ckpt.get("in_continue", False))
                self._last_gain = ckpt.get("last_gain", None)
                self._continue_count = int(ckpt.get("continue_count", 0))
                if self._in_continue and ckpt.get("model_state") is not None:
                    self.model.load_state_dict(ckpt["model_state"], strict=True)
            except Exception:
                self._in_continue = False
                self._last_gain = None
                self._continue_count = 0

    def _save_persistent_state(self):
        ckpt = {
            "in_continue": self._in_continue,
            "last_gain": self._last_gain,
            "continue_count": self._continue_count,
            "model_state": self.model.state_dict() if self._in_continue else None,
        }
        torch.save(ckpt, self._state_file)

    # ------------------------------------------------------------------ #
    # Flower API
    # ------------------------------------------------------------------ #
    def get_parameters(self, config=None):
        return [value.detach().cpu().numpy() for value in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            (key, torch.as_tensor(value, device=DEVICE))
            for key, value in zip(self.model.state_dict().keys(), parameters)
        )
        if len(state_dict) != len(self.model.state_dict()):
            raise ValueError("Received parameter count does not match model state.")
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self._load_persistent_state()
        state = self._extract_state(config)
        # Deterministic dropout / init-dependent randomness for this (seed, vehicle, round)
        torch.manual_seed(policy.derive_seed(self.seed, self.id, state["server_round"]))
        proximal_mu = self.fedprox_mu if self.algorithm == "fedprox" else 0.0

        if not self._in_continue:
            self.set_parameters(parameters)
            global_weights = [p.detach().clone() for p in self.model.parameters()]
        else:
            # Continue the local trajectory; the new global model is only a reference
            state_dict = OrderedDict(
                (key, torch.as_tensor(value, device=DEVICE))
                for key, value in zip(self._temp_global_model.state_dict().keys(), parameters)
            )
            self._temp_global_model.load_state_dict(state_dict, strict=True)
            global_weights = [p.detach().clone() for p in self._temp_global_model.parameters()]

        if self.spec["kind"] == "dynamic":
            return self._run_dynamic(global_weights, state, proximal_mu)
        return self._run_baseline(global_weights, state, proximal_mu)

    def evaluate(self, parameters, config):
        state_dict = OrderedDict(
            (key, torch.as_tensor(value, device=DEVICE))
            for key, value in zip(self._temp_global_model.state_dict().keys(), parameters)
        )
        self._temp_global_model.load_state_dict(state_dict, strict=True)
        loss, accuracy = self._validate_model(self._temp_global_model)
        return loss, len(self.valloader.dataset), {"accuracy": float(accuracy)}

    # ------------------------------------------------------------------ #
    # Schemes
    # ------------------------------------------------------------------ #
    def _new_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    def _run_baseline(self, global_weights, state, proximal_mu):
        kind = self.spec["kind"]
        self._in_continue = False
        self._continue_count = 0
        if kind == "randpart" and not policy.participates(
            self.seed, state["server_round"], self.id, float(self.spec["participation"])
        ):
            return self._wait(state, policy.WAIT_NOT_SELECTED, 0)
        if kind == "threshold" and state["bandwidth"] < self.params.bw_threshold:
            return self._wait(state, policy.WAIT_LINK, 0)
        epochs = policy.baseline_epochs(kind, self.spec["epochs"], state["u_crit"], self.rho)
        if epochs <= 0:
            return self._wait(state, policy.WAIT_COMPUTE, 0)

        optimizer = self._new_optimizer()
        init_loss = self._validate()
        for epoch_idx in range(epochs):
            self._train_epoch(optimizer, global_weights, proximal_mu, state["server_round"], epoch_idx)
        drift = self._compute_drift(global_weights)
        curr_loss = self._validate()
        gain = max(0.0, init_loss - curr_loss)
        self._last_gain = gain
        self._save_persistent_state()
        metrics = self._metrics(
            state, policy.SYNC, epochs, drift, gain, 0, val_passes=2,
            c_net=self.M / max(0.1, state["bandwidth"]), sync_reason=policy.SYNC_ALWAYS,
        )
        return self.get_parameters(), len(self.trainloader.dataset), metrics

    def _run_dynamic(self, global_weights, state, proximal_mu):
        p = self.params
        was_in_continue = int(bool(self._in_continue))

        wait_reason = policy.admission(
            state["u_crit"], state["bandwidth"], self._last_gain, p,
            no_continue=(self.spec["ablation"] == "no_continue"),
        )
        if wait_reason:
            # Fix (audit 5.2): a WAIT keeps a pending CONTINUE trajectory. The weights
            # stay on disk and the round counts towards staleness (K_max).
            if self._in_continue:
                self._continue_count += 1
            self._save_persistent_state()
            return self._wait(state, wait_reason, was_in_continue)

        optimizer = self._new_optimizer()
        init_loss = self._validate()
        val_passes = 1
        e_max = max(1, min(self.rho, policy.slack_epoch_budget(state["u_crit"], self.rho)))
        no_stop = self.spec["ablation"] == "no_stop"
        history = [init_loss]
        epoch, curr_loss = 0, init_loss

        while epoch < e_max:
            self._train_epoch(optimizer, global_weights, proximal_mu, state["server_round"], epoch)
            epoch += 1
            if no_stop:
                continue
            curr_loss = self._validate()
            val_passes += 1
            policy.ema_append(history, curr_loss, p.ema_weight)
            if epoch < 2 or epoch >= e_max:
                continue
            expected_gain = policy.expected_gain(history, p)
            drift = self._compute_drift(global_weights)
            if policy.should_stop(expected_gain, state["u_crit"], drift, p):
                break

        if no_stop:
            curr_loss = self._validate()
            val_passes += 1

        actual_gain = max(0.0, init_loss - curr_loss)
        self._last_gain = actual_gain
        final_drift = self._compute_drift(global_weights)
        status, self._continue_count, sync_reason = policy.post_training_decision(
            state["bandwidth"], final_drift, actual_gain, bool(was_in_continue),
            self._continue_count, p,
        )
        self._in_continue = status == policy.CONTINUE
        self._save_persistent_state()

        is_sync = status == policy.SYNC
        metrics = self._metrics(
            state, status, epoch, final_drift, actual_gain, was_in_continue,
            val_passes=val_passes,
            c_net=(self.M / max(0.1, state["bandwidth"])) if is_sync else 0.0,
            sync_reason=sync_reason,
        )
        if is_sync:
            return self.get_parameters(), len(self.trainloader.dataset), metrics
        return [], 0, metrics  # CONTINUE: weights kept locally

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _extract_state(self, config):
        server_round = int(config.get("server_round", 0))
        u_crit = self._pick(config, "trace_u_crit", "u_crit", 0.2)
        bandwidth = self._pick(config, "trace_bandwidth", "bandwidth", 10.0)
        return {
            "server_round": server_round,
            "u_crit": u_crit,
            "max_c_i": max(0.0, 1.0 - u_crit),
            "bandwidth": bandwidth,
        }

    def _pick(self, config, vector_key, scalar_key, default):
        """Per-vehicle value from the round's trace vector (indexed by partition id)."""
        if vector_key in config:
            values = [float(v) for v in str(config[vector_key]).split(",") if v != ""]
            if self.id < len(values):
                return values[self.id]
            raise ValueError(f"{vector_key} has {len(values)} entries; vehicle {self.id} missing")
        return float(config.get(scalar_key, default))

    def _wait(self, state, reason, was_in_continue):
        metrics = self._metrics(state, policy.WAIT, 0, 0.0, 0.0, was_in_continue, val_passes=0)
        metrics["wait_reason"] = reason
        return [], 0, metrics

    def _metrics(self, state, status, epochs, drift, gain, was_in_continue,
                 val_passes, c_net=0.0, sync_reason=""):
        return {
            "status": status,
            "partition_id": self.id,
            "server_round": state["server_round"],
            "epochs_trained": int(epochs),
            "c_comp": (epochs / self.rho) * state["u_crit"],
            "c_net": float(c_net),
            "c_drift": float(drift),
            "gain": float(gain),
            "was_in_continue": int(was_in_continue),
            "continue_count": int(self._continue_count),
            "bandwidth": float(state["bandwidth"]),
            "u_crit": float(state["u_crit"]),
            "n_train": int(len(self.trainloader.dataset)),
            "n_val": int(len(self.valloader.dataset)),
            "val_passes": int(val_passes),
            "sync_reason": sync_reason,
            "wait_reason": "",
        }

    def _train_epoch(self, optimizer, global_weights=None, proximal_mu: float = 0.0,
                     server_round: int = 0, epoch_idx: int = 0):
        generator = getattr(self.trainloader, "generator", None)
        if generator is not None:  # deterministic data order per (seed, vehicle, round, epoch)
            generator.manual_seed(policy.derive_seed(self.seed, self.id, server_round, epoch_idx))
        self.model.train()
        for batch in self.trainloader:
            x = batch["img"].to(DEVICE)
            y = batch["label"].to(DEVICE, dtype=torch.long)
            optimizer.zero_grad()
            loss = self.criterion(self.model(x), y)
            if proximal_mu > 0.0:
                if global_weights is None:
                    raise ValueError("FedProx requires frozen global weights.")
                proximal_term = torch.zeros((), device=DEVICE)
                for local_param, global_param in zip(self.model.parameters(), global_weights):
                    proximal_term = proximal_term + torch.sum((local_param - global_param) ** 2)
                loss = loss + 0.5 * proximal_mu * proximal_term
            loss.backward()
            optimizer.step()

    def _validate(self):
        loss, _ = self._validate_model(self.model)
        return loss

    def _validate_model(self, eval_model):
        eval_model.eval()
        total_loss, total_correct, total_examples = 0.0, 0, 0
        with torch.no_grad():
            for batch in self.valloader:
                x = batch["img"].to(DEVICE)
                y = batch["label"].to(DEVICE, dtype=torch.long)
                logits = eval_model(x)
                batch_size = y.size(0)
                total_loss += self.criterion(logits, y).item() * batch_size
                total_correct += (logits.argmax(dim=1) == y).sum().item()
                total_examples += batch_size
        if total_examples == 0:
            return float("nan"), float("nan")
        return total_loss / total_examples, total_correct / total_examples

    def _predict_loss(self, history):  # kept for backward compatibility
        return policy.predict_next_loss(history)

    def _compute_drift(self, global_weights):
        with torch.no_grad():
            sum_sq = torch.zeros((), device=DEVICE)
            for local_param, global_param in zip(self.model.parameters(), global_weights):
                sum_sq += torch.sum((local_param - global_param) ** 2)
            return torch.sqrt(sum_sq).item()
