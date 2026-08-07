import flwr as fl
from flwr.common import Context
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np
import scipy.optimize

from constants import ALPHA, BETA, GAMMA, DELTA, DEVICE
import model

class FLVehicle(fl.client.NumPyClient):
    def __init__(self, partition_id: int, trainloader: DataLoader, valloader: DataLoader, 
                 model_size_mb: float = 50.0, epochs_per_compute: int = 10):
        # System model
        self.id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.M = model_size_mb
        self.rho = epochs_per_compute
        
        self.alpha = ALPHA
        self.beta = BETA
        self.gamma = GAMMA
        self.delta = DELTA
        
        # PyTorch Setup
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
        
        # Read the dynamic variables from the server
        u_crit = config.get("u_crit", 0.2)
        bandwidth = config.get("bandwidth", 10.0)
        tau_continue = config.get("tau_continue", 0.1)
        drift_threshold = config.get("drift_threshold", 5.0)
        gain_threshold = config.get("gain_threshold", 0.01)
        bw_threshold = config.get("bw_threshold", 30.0)

        max_c_i = max(0.0, 1.0 - u_crit)

        # Fixed policy
        if hasattr(self, 'mode') and self.mode.startswith("fixed"):
            epochs_to_run = int(self.mode.split("_")[1])
            for _ in range(epochs_to_run):
                self._train_epoch(optimizer)
            
            drift = self._compute_drift(global_weights)
            required_cpu = epochs_to_run / self.rho
            interference = required_cpu * u_crit 
            
            return self.get_parameters(config={}), len(self.trainloader.dataset), {
                "status": "SYNC", 
                "epochs_trained": epochs_to_run, 
                "drift": drift,
                "interference": interference
            }

        # Dynamic Policy
        loss_history = []
        epoch = 0
        status = "CONTINUE"
        
        initial_loss = self._validate()
        loss_history.append(initial_loss)

        while True:
            required_cpu_for_next_epoch = (epoch + 1.0) / self.rho
            if required_cpu_for_next_epoch > max_c_i:
                status = "SYNC"
                break

            self._train_epoch(optimizer)
            epoch += 1
            
            current_loss = self._validate()
            smoothed_loss = 0.3 * current_loss + 0.7 * loss_history[-1]
            loss_history.append(smoothed_loss)

            if epoch < 2:
                continue

            if epoch >= self.rho:
                status = "SYNC"
                break
                
            predicted_next_loss = self._predict_loss(loss_history)
            expected_gain = loss_history[-1] - predicted_next_loss
            drift = self._compute_drift(global_weights)
            
            utility = self._compute_utility(expected_gain, drift)
            
            if drift > drift_threshold or expected_gain < gain_threshold:
                status = "SYNC"
                break
                
            if bandwidth < bw_threshold and utility < tau_continue:
                status = "WAIT"
                break

        # Compute final interference
        final_cpu_used = epoch / self.rho
        interference = final_cpu_used * u_crit

        if status == "WAIT" or epoch == 0:
            return parameters, 0, {"status": "WAIT", "epochs_trained": epoch, "drift": 0.0, "interference": 0.0}
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {
            "status": "SYNC", 
            "epochs_trained": epoch, 
            "drift": drift,
            "interference": interference
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self._validate()
        # Return loss
        return loss, len(self.valloader.dataset), {"loss": loss}

    # Helpers
    def _train_epoch(self, optimizer):
        self.model.train()
        for batch in self.valloader:
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
        if len(loss_history) < 2:
            return loss_history[-1]
        x_data = np.arange(1, len(loss_history) + 1)
        y_data = np.array(loss_history)
        try:
            popt, _ = scipy.optimize.curve_fit(self._exp_func, x_data, y_data, maxfev=5000)
            next_epoch = len(loss_history) + 1
            predicted_loss = self._exp_func(next_epoch, *popt)
            return predicted_loss
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