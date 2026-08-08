import flwr as fl
from flwr.common import Context
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
import scipy.optimize
import warnings
from scipy.optimize import OptimizeWarning
from constants import ALPHA, BETA, GAMMA, DELTA, DEVICE
import model

class FLVehicle(fl.client.NumPyClient):
    def __init__(self, partition_id: int, trainloader: DataLoader, valloader: DataLoader, 
                 model_size_mb: float = 50.0, epochs_per_compute: int = 10):
        self.id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.M = model_size_mb
        self.rho = epochs_per_compute
        
        self.alpha = ALPHA
        self.beta = BETA
        self.gamma = GAMMA
        self.delta = DELTA
        
        self.model = model.load_model().to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        return [torch.tensor(v).to(DEVICE) for v in parameters]

    def fit(self, parameters, config):
        global_weights = self.set_parameters(parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        # System State S_i(t)
        u_crit = config.get("u_crit", 0.2)
        bandwidth = config.get("bandwidth", 10.0)
        
        tau_continue = config.get("tau_continue", 0.1)
        drift_threshold = config.get("drift_threshold", 5.0)
        gain_threshold = config.get("gain_threshold", 0.01)
        bw_threshold = config.get("bw_threshold", 30.0)
        
        max_c_i = max(0.0, 1.0 - u_crit)
        
        # Fixed Policy
        if hasattr(self, 'mode') and self.mode.startswith("fixed"):
            epochs_to_run = int(self.mode.split("_")[1])
            
            # Force train for the exact number of fixed epochs
            for _ in range(epochs_to_run):
                self._train_epoch(optimizer)
                
            c_i = epochs_to_run / self.rho
            c_comp = c_i * u_crit
            c_net = 1.0 * (self.M / max(0.1, bandwidth))
            c_drift = self._compute_drift(global_weights)
            
            metrics = {
                "status": "SYNC",
                "epochs_trained": epochs_to_run,
                "c_comp": c_comp,
                "c_net": c_net,
                "c_drift": c_drift,
                "gain": 0.0
            }
            return self.get_parameters(config={}), len(self.trainloader.dataset), metrics
        # ---------------------------------------------------------

        # WAIT (e_i = 0, p_i = 0
        if max_c_i < (1.0 / self.rho):
            return parameters, 0, {
                "status": "WAIT", "epochs_trained": 0,
                "c_comp": 0.0, "c_net": 0.0, "c_drift": 0.0, "gain": 0.0
            }

        # Dynamic Policy
        loss_history = []
        epoch = 0
        drift = 0.0
        
        initial_loss = self._validate()
        loss_history.append(initial_loss)
        
        while True:
            # Training-Compute Coupling
            required_c_i = (epoch + 1.0) / self.rho
            if required_c_i > max_c_i:
                break
                
            self._train_epoch(optimizer)
            epoch += 1
            
            current_loss = self._validate()
            smoothed_loss = 0.3 * current_loss + 0.7 * loss_history[-1]
            loss_history.append(smoothed_loss)
            
            if epoch < 2:
                continue
            if epoch >= self.rho:
                break
                
            predicted_next_loss = self._predict_loss(loss_history)
            expected_gain = loss_history[-1] - predicted_next_loss
            drift = self._compute_drift(global_weights)
            
            # Local Utility = \delta * G_i(t) - \gamma * D_i(t)
            utility = self._compute_utility(expected_gain, drift)
            
            # Drift Stability Constraint and utility drop
            if (drift > drift_threshold) or (expected_gain < gain_threshold) or (utility < tau_continue):
                break
                
        if bandwidth >= bw_threshold:
            status = "SYNC"
            p_i = 1.0
        else:
            status = "CONTINUE"
            p_i = 0.0
            
        c_i = epoch / self.rho 
        c_comp = c_i * u_crit 
        c_net = p_i * (self.M / max(0.1, bandwidth))
        c_drift = self._compute_drift(global_weights) if epoch > 0 else 0.0
        gain = initial_loss - current_loss if epoch > 0 else 0.0
        
        metrics = {
            "status": status,
            "epochs_trained": epoch,
            "c_comp": c_comp,
            "c_net": c_net,
            "c_drift": c_drift,
            "gain": gain
        }
        
        if status == "CONTINUE":
            # e_i > 0, p_i = 0)
            return parameters, 0, metrics
            
        # SYNC (p_i = 1)
        return self.get_parameters(config={}), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self._validate()
        return loss, len(self.valloader.dataset), {"loss": loss}

    # Helpers
    def _train_epoch(self, optimizer):
        self.model.train()
        for batch in self.trainloader:
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.valloader:
                images = batch["img"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        return total_loss / len(self.valloader)

    def _exp_func(self, x, L_inf, alpha, beta):
        return L_inf + alpha * np.exp(-beta * x)

    def _predict_loss(self, loss_history):
        if len(loss_history) < 3:
            return loss_history[-1]
            
        x_data = np.arange(1, len(loss_history) + 1)
        y_data = np.array(loss_history)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = scipy.optimize.curve_fit(self._exp_func, x_data, y_data, maxfev=5000)
                
            next_epoch = len(loss_history) + 1
            return self._exp_func(next_epoch, *popt)
        except RuntimeError:
            return y_data[-1]

    def _compute_drift(self, global_weights):
        drift = 0.0
        with torch.no_grad():
            for local_param, global_param in zip(self.model.parameters(), global_weights):
                drift += torch.norm(local_param - global_param).item()
        return drift

    def _compute_utility(self, gain, drift):
        return self.delta * gain - self.gamma * drift