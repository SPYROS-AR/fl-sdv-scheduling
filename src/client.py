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
    ):
        self.id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.M = MODEL_SIZE_MB
        self.rho = epochs_per_compute
        self.alpha, self.beta = ALPHA, BETA
        self.gamma, self.delta = GAMMA, DELTA
        self.model = model.load_model().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.mode = "dynamic"

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.model.state_dict().keys(), parameters)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)
        return [torch.tensor(v).to(DEVICE) for v in parameters]

    def fit(self, parameters, config):
        global_weights = self.set_parameters(parameters)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
        )
        state = self._extract_state(config)
        init_loss = self._validate()

        if self.mode.startswith("fixed"):
            return self._run_fixed(optimizer, global_weights, init_loss, state)
        return self._run_dynamic(
            parameters, optimizer, global_weights, init_loss, state
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self._validate()
        return loss, len(self.valloader.dataset), {"loss": loss}

    def _run_fixed(self, optimizer, global_weights, init_loss, state):
        epochs = int(self.mode.split("_")[1])
        for _ in range(epochs):
            self._train_epoch(optimizer)

        metrics = {
            "status": "SYNC",
            "epochs_trained": epochs,
            "c_comp": (epochs / self.rho) * state["u_crit"],
            "c_net": self.M / max(0.1, state["bandwidth"]),
            "c_drift": self._compute_drift(global_weights),
            "gain": init_loss - self._validate(),
        }
        return self.get_parameters(), len(self.trainloader.dataset), metrics

    def _run_dynamic(self, parameters, optimizer, global_weights, init_loss, state):
        if state["max_c_i"] < (1.0 / self.rho):
            return parameters, 0, self._empty_metrics("WAIT")

        history = [init_loss]
        epoch, curr_loss = 0, init_loss

        while True:
            if ((epoch + 1.0) / self.rho) > state["max_c_i"]:
                break

            self._train_epoch(optimizer)
            epoch += 1
            curr_loss = self._validate()
            smoothed = EMA_WEIGHT * curr_loss + (1.0 - EMA_WEIGHT) * history[-1]
            history.append(smoothed)

            if epoch < 2:
                continue
            if epoch >= self.rho:
                break

            predicted_loss = self._predict_loss(history)
            expected_gain = history[-1] - predicted_loss
            drift = self._compute_drift(global_weights)
            utility = self.delta * expected_gain - self.gamma * drift

            if (
                drift > state["drift_threshold"]
                or expected_gain < state["gain_threshold"]
                or utility < state["tau_continue"]
            ):
                break

        is_sync = state["bandwidth"] >= state["bw_threshold"]
        status = "SYNC" if is_sync else "CONTINUE"

        metrics = {
            "status": status,
            "epochs_trained": epoch,
            "c_comp": (epoch / self.rho) * state["u_crit"],
            "c_net": (self.M / max(0.1, state["bandwidth"])) if is_sync else 0.0,
            "c_drift": self._compute_drift(global_weights) if epoch > 0 else 0.0,
            "gain": (init_loss - curr_loss) if epoch > 0 else 0.0,
        }

        return (
            self.get_parameters() if is_sync else parameters,
            len(self.trainloader.dataset) if is_sync else 0,
            metrics,
        )

    def _extract_state(self, config):
        u_crit = config.get("u_crit", 0.2)
        return {
            "u_crit": u_crit,
            "max_c_i": max(0.0, 1.0 - u_crit),
            "bandwidth": config.get("bandwidth", 10.0),
            "tau_continue": config.get("tau_continue", 0.1),
            "drift_threshold": config.get("drift_threshold", 5.0),
            "gain_threshold": config.get("gain_threshold", 0.01),
            "bw_threshold": config.get("bw_threshold", 20.0),
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

    def _train_epoch(self, optimizer):
        self.model.train()
        for batch in self.trainloader:
            x, y = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            self.criterion(self.model(x), y).backward()
            optimizer.step()

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.valloader:
                x, y = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                total_loss += self.criterion(self.model(x), y).item()
        return total_loss / max(1, len(self.valloader))

    @staticmethod
    def _exp_curve(x, l_inf, a, b):
        return l_inf + a * np.exp(-b * x)

    def _predict_loss(self, history):
        if len(history) < 3:
            return history[-1]
        x_data = np.arange(1, len(history) + 1)
        y_data = np.array(history)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = scipy.optimize.curve_fit(
                    self._exp_curve, x_data, y_data, maxfev=2000
                )
            return self._exp_curve(len(history) + 1, *popt)
        except (RuntimeError, ValueError):
            return history[-1]

    def _compute_drift(self, global_weights):
        with torch.no_grad():
            return sum(
                torch.norm(loc - glob).item()
                for loc, glob in zip(self.model.parameters(), global_weights)
            )
