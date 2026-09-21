import logging
import warnings
from collections import OrderedDict
import flwr as fl
from flwr.common.logger import log
import numpy as np
import scipy.optimize
from scipy.optimize import OptimizeWarning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import model
from constants import ALPHA, BETA, GAMMA, DELTA, DEVICE, MODEL_SIZE_MB, LEARNING_RATE, MOMENTUM, EMA_WEIGHT

class FLVehicle(fl.client.NumPyClient):
    def __init__(self, partition_id: int, trainloader: DataLoader, valloader: DataLoader,
                 epochs_per_compute: int = 10):
        self.id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.M = MODEL_SIZE_MB
        self.rho = epochs_per_compute
        
        self.alpha, self.beta = ALPHA, BETA
        self.gamma, self.delta = GAMMA, DELTA
        
        self.model = model.load_model().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        return [torch.tensor(v).to(DEVICE) for v in parameters]

    def fit(self, parameters, config):
        log(logging.INFO, f"DEBUG: Training on -> {next(self.model.parameters()).device}")
        global_weights = self.set_parameters(parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        
        system_state = self._extract_system_state(config)
        initial_loss = self._validate()
        
        if hasattr(self, 'mode') and self.mode.startswith("fixed"):
            return self._run_fixed_policy(optimizer, global_weights, initial_loss, system_state)
            
        return self._run_dynamic_policy(parameters, optimizer, global_weights, initial_loss, system_state)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self._validate()
        return loss, len(self.valloader.dataset), {"loss": loss}

    # --- Policy Implementations ---

    def _run_fixed_policy(self, optimizer, global_weights, initial_loss, state):
        epochs_to_run = int(self.mode.split("_")[1])
        
        for _ in range(epochs_to_run):
            self._train_epoch(optimizer)
            
        metrics = {
            "status": "SYNC",
            "epochs_trained": epochs_to_run,
            "c_comp": (epochs_to_run / self.rho) * state["u_crit"],
            "c_net": 1.0 * (self.M / max(0.1, state["bandwidth"])),
            "c_drift": self._compute_drift(global_weights), 
            "gain": initial_loss - self._validate()
        }
        return self.get_parameters(), len(self.trainloader.dataset), metrics

    def _run_dynamic_policy(self, parameters, optimizer, global_weights, initial_loss, state):
        if state["max_c_i"] < (1.0 / self.rho):
            return parameters, 0, self._empty_metrics("WAIT")

        loss_history = [initial_loss]
        epoch, current_loss = 0, initial_loss
        
        while True:
            if ((epoch + 1.0) / self.rho) > state["max_c_i"]:
                break
                
            self._train_epoch(optimizer)
            epoch += 1
            
            current_loss = self._validate()
            loss_history.append(EMA_WEIGHT * current_loss + (1.0 - EMA_WEIGHT) * loss_history[-1])
            
            if epoch < 2:
                continue

            if epoch >= self.rho:
                break
                
            expected_gain = loss_history[-1] - self._predict_loss(loss_history)
            drift = self._compute_drift(global_weights)
            utility = self.delta * expected_gain - self.gamma * drift
            
            if (drift > state["drift_threshold"]) or (expected_gain < state["gain_threshold"]) or (utility < state["tau_continue"]):
                break
            
        is_sync = state["bandwidth"] >= state["bw_threshold"]
        status = "SYNC" if is_sync else "CONTINUE"
        
        metrics = {
            "status": status,
            "epochs_trained": epoch,
            "c_comp": (epoch / self.rho) * state["u_crit"],
            "c_net": (1.0 if is_sync else 0.0) * (self.M / max(0.1, state["bandwidth"])),
            "c_drift": self._compute_drift(global_weights) if epoch > 0 else 0.0,
            "gain": initial_loss - current_loss if epoch > 0 else 0.0
        }
        
        return self.get_parameters() if is_sync else parameters, (len(self.trainloader.dataset) if is_sync else 0), metrics

    # --- Helpers ---
    
    def _extract_system_state(self, config):
        u_crit = config.get("u_crit", 0.2)
        return {
            "u_crit": u_crit,
            "max_c_i": max(0.0, 1.0 - u_crit),
            "bandwidth": config.get("bandwidth", 10.0),
            "tau_continue": config.get("tau_continue", 0.1),
            "drift_threshold": config.get("drift_threshold", 5.0),
            "gain_threshold": config.get("gain_threshold", 0.01),
            "bw_threshold": config.get("bw_threshold", 20.0)
        }

    def _empty_metrics(self, status):
        return {"status": status, "epochs_trained": 0, "c_comp": 0.0, "c_net": 0.0, "c_drift": 0.0, "gain": 0.0}

    def _train_epoch(self, optimizer):
        self.model.train()
        for batch in self.trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)
            loss.backward()
            optimizer.step()

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.valloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                total_loss += self.criterion(self.model(images), labels).item()
        return total_loss / len(self.valloader)

    def _exp_func(self, x, L_inf, alpha, beta):
        return L_inf + alpha * np.exp(-beta * x)

    def _predict_loss(self, loss_history):
        if len(loss_history) < 3: return loss_history[-1]
        x_data, y_data = np.arange(1, len(loss_history) + 1), np.array(loss_history)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = scipy.optimize.curve_fit(self._exp_func, x_data, y_data, maxfev=5000)
            return self._exp_func(len(loss_history) + 1, *popt)
        except RuntimeError:
            return y_data[-1]

    def _compute_drift(self, global_weights):
        with torch.no_grad():
            return sum(torch.norm(local - global_w).item() for local, global_w in zip(self.model.parameters(), global_weights))