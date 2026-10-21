import warnings
from collections import OrderedDict

import flwr as fl
import numpy as np
import scipy.optimize
import torch
from scipy.optimize import OptimizeWarning
from torch import nn
from torch.utils.data import DataLoader

import model
from constants import (
    ALPHA,
    BETA,
    DELTA,
    DEVICE,
    EMA_WEIGHT,
    FEDPROX_MU,
    GAMMA,
    LEARNING_RATE,
    MODEL_SIZE_MB,
    MOMENTUM,
)


class FLVehicle(fl.client.NumPyClient):
    def __init__(
        self,
        partition_id: int,
        trainloader: DataLoader,
        valloader: DataLoader,
        epochs_per_compute: int = 10,
        num_classes: int = 10,
        algorithm: str = "fedavg",
        fedprox_mu: float = FEDPROX_MU,
    ):
        self.id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.M = MODEL_SIZE_MB  # Keep policy model size separate from measured payload bytes.
        self.rho = epochs_per_compute
        self.alpha, self.beta = ALPHA, BETA
        self.gamma, self.delta = GAMMA, DELTA
        self.model = model.load_model(num_classes=num_classes).to(DEVICE)
        self.num_classes = num_classes
        self._temp_global_model = model.load_model(num_classes=num_classes).to(DEVICE)
        self._in_continue = False
        self._last_gain = None
        self.criterion = nn.CrossEntropyLoss()
        self.mode = "dynamic"

        self.algorithm = algorithm.lower()
        self.fedprox_mu = float(fedprox_mu)
        if self.algorithm not in {"fedavg", "fedprox"}:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        if self.fedprox_mu < 0:
            raise ValueError("FedProx mu must be non-negative.")

    def get_parameters(self, config=None):
        return [
            value.detach().cpu().numpy() for value in self.model.state_dict().values()
        ]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            (
                key,
                torch.as_tensor(value, device=DEVICE),
            )
            for key, value in zip(self.model.state_dict().keys(), parameters)
        )
        if len(state_dict) != len(self.model.state_dict()):
            raise ValueError("Received parameter count does not match model state.")
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        state = self._extract_state(config)
        proximal_mu = self.fedprox_mu if self.algorithm == "fedprox" else 0.0

        if not getattr(self, "_in_continue", False):
            # Fresh start: load latest global parameters into local model
            self.set_parameters(parameters)
            global_weights = [p.detach().clone() for p in self.model.parameters()]
        else:
            # Continuing local trajectory: preserve local weights in self.model,
            # but update global reference weights for drift and proximal regularization
            state_dict = OrderedDict(
                (key, torch.as_tensor(value, device=DEVICE))
                for key, value in zip(self._temp_global_model.state_dict().keys(), parameters)
            )
            self._temp_global_model.load_state_dict(state_dict, strict=True)
            global_weights = [p.detach().clone() for p in self._temp_global_model.parameters()]

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
        )
        init_loss = self._validate()

        if self.mode.startswith("fixed_"):
            return self._run_fixed(
                optimizer, global_weights, init_loss, state, proximal_mu
            )
        return self._run_dynamic(
            parameters, optimizer, global_weights, init_loss, state, proximal_mu
        )

    def evaluate(self, parameters, config):
        state_dict = OrderedDict(
            (key, torch.as_tensor(value, device=DEVICE))
            for key, value in zip(self._temp_global_model.state_dict().keys(), parameters)
        )
        self._temp_global_model.load_state_dict(state_dict, strict=True)
        loss, accuracy = self._validate_model(self._temp_global_model)
        return loss, len(self.valloader.dataset), {"accuracy": float(accuracy)}

    def _run_fixed(self, optimizer, global_weights, init_loss, state, proximal_mu):
        self._in_continue = False
        epochs = int(self.mode.split("_", 1)[1])
        for _ in range(epochs):
            self._train_epoch(optimizer, global_weights, proximal_mu)

        drift = self._compute_drift(global_weights)
        curr_loss = self._validate()
        gain = max(0.0, init_loss - curr_loss)
        self._last_gain = gain

        metrics = {
            "status": "SYNC",
            "epochs_trained": epochs,
            "c_comp": (epochs / self.rho) * state["u_crit"],
            "c_net": self.M / max(0.1, state["bandwidth"]),
            "c_drift": drift,
            "gain": gain,
        }
        return self.get_parameters(), len(self.trainloader.dataset), metrics

    def _run_dynamic(
        self, parameters, optimizer, global_weights, init_loss, state, proximal_mu
    ):
        # 1. Compute Starvation: FL yields to higher-priority safety workloads
        if state["max_c_i"] < (1.0 / self.rho):
            self._in_continue = False
            return [], 0, self._empty_metrics("WAIT")

        # 2. Poor Channel Bandwidth AND Low Potential Learning Gain (WAIT State)
        is_bw_poor = state["bandwidth"] < state["bw_threshold"]
        prior_gain = getattr(self, "_last_gain", None)
        if is_bw_poor and prior_gain is not None and prior_gain <= state["gain_threshold"]:
            self._in_continue = False
            return [], 0, self._empty_metrics("WAIT")

        history = [init_loss]
        epoch, curr_loss = 0, init_loss

        while True:
            if ((epoch + 1.0) / self.rho) > state["max_c_i"]:
                break

            self._train_epoch(optimizer, global_weights, proximal_mu)
            epoch += 1
            curr_loss = self._validate()
            smoothed = EMA_WEIGHT * curr_loss + (1.0 - EMA_WEIGHT) * history[-1]
            history.append(smoothed)

            if epoch < 2:
                continue
            if epoch >= self.rho:
                break

            predicted_loss = self._predict_loss(history)
            expected_gain = max(0.0, history[-1] - predicted_loss)
            drift = self._compute_drift(global_weights)
            c_comp_marginal = (1.0 / self.rho) * state["u_crit"]
            marginal_utility = (
                self.delta * expected_gain
                - self.beta * c_comp_marginal
                - self.gamma * drift
            )

            if (
                drift > state["drift_threshold"]
                or expected_gain < state["gain_threshold"]
                or marginal_utility < state["tau_continue"]
            ):
                break

        actual_gain = max(0.0, init_loss - curr_loss)
        self._last_gain = actual_gain

        if epoch == 0:
            self._in_continue = False
            return [], 0, self._empty_metrics("WAIT")

        is_sync = state["bandwidth"] >= state["bw_threshold"]

        # Transition to WAIT if bandwidth is constrained and progress was negligible
        if not is_sync and actual_gain <= state["gain_threshold"]:
            self._in_continue = False
            return [], 0, self._empty_metrics("WAIT")

        status = "SYNC" if is_sync else "CONTINUE"
        self._in_continue = (status == "CONTINUE")

        metrics = {
            "status": status,
            "epochs_trained": epoch,
            "c_comp": (epoch / self.rho) * state["u_crit"],
            "c_net": (self.M / max(0.1, state["bandwidth"])) if is_sync else 0.0,
            "c_drift": self._compute_drift(global_weights) if epoch > 0 else 0.0,
            "gain": actual_gain,
        }
        if is_sync:
            return self.get_parameters(), len(self.trainloader.dataset), metrics
        # CONTINUE: keep local model weights; send no upload in this round
        return [], 0, metrics

    def _extract_state(self, config):
        u_crit = float(config.get("u_crit", 0.2))
        return {
            "u_crit": u_crit,
            "max_c_i": max(0.0, 1.0 - u_crit),
            "bandwidth": float(config.get("bandwidth", 10.0)),
            "tau_continue": float(config.get("tau_continue", 0.1)),
            "drift_threshold": float(config.get("drift_threshold", 5.0)),
            "gain_threshold": float(config.get("gain_threshold", 0.01)),
            "bw_threshold": float(config.get("bw_threshold", 20.0)),
        }

    def _empty_metrics(self, status):
        return {
            "status": status,
            "epochs_trained": 0,
            "c_comp": 0.0,
            "c_net": 0.0,
            "c_drift": 0.0,
            "gain": 0.0,
        }

    def _train_epoch(self, optimizer, global_weights=None, proximal_mu: float = 0.0):
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
                for local_param, global_param in zip(
                    self.model.parameters(), global_weights
                ):
                    proximal_term = proximal_term + torch.sum(
                        (local_param - global_param) ** 2
                    )
                loss = loss + 0.5 * proximal_mu * proximal_term

            loss.backward()
            optimizer.step()

    def _validate(self):
        loss, _ = self._validate_model(self.model)
        return loss

    def _validate_metrics(self):
        return self._validate_model(self.model)

    def _validate_model(self, eval_model):
        eval_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
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

    @staticmethod
    def _exp_curve(x, l_inf, a, b):
        return l_inf + a * np.exp(-b * x)

    def _predict_loss(self, history):
        if len(history) < 3:
            return float(history[-1])

        x_data = np.arange(len(history), dtype=np.float64)
        y_data = np.array(history, dtype=np.float64)

        l_inf_init = max(0.0, float(np.min(y_data) * 0.8))
        a_init = max(1e-3, float(y_data[0] - l_inf_init))
        b_init = 0.5
        p0 = [l_inf_init, a_init, b_init]
        bounds = ([0.0, 0.0, 1e-4], [float(np.min(y_data)), np.inf, 10.0])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = scipy.optimize.curve_fit(
                    self._exp_curve, x_data, y_data, p0=p0, bounds=bounds, maxfev=1500
                )
            next_step = float(len(history))
            pred = float(self._exp_curve(next_step, *popt))
            return min(float(y_data[-1]), max(0.0, pred))
        except (RuntimeError, ValueError, FloatingPointError):
            delta_last = float(y_data[-1] - y_data[-2])
            if delta_last < 0:
                return max(0.0, float(y_data[-1] + 0.5 * delta_last))
            return float(y_data[-1])

    def _compute_drift(self, global_weights):
        with torch.no_grad():
            sum_sq = torch.zeros((), device=DEVICE)
            for local_param, global_param in zip(self.model.parameters(), global_weights):
                sum_sq += torch.sum((local_param - global_param) ** 2)
            return torch.sqrt(sum_sq).item()
