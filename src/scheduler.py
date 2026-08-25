import flwr as fl
from flwr.common import Context
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
from data_setup import setup_dataset
from client import FLVehicle
from server import DynamicFLStrategy
import logging
import datasets
from constants import NUM_ROUNDS

# Silence HuggingFace local offline cache notifications
datasets.utils.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, filename='results.log', filemode='a')

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def run_simulation(df: pd.DataFrame, num_vehicles: int,
                   num_samples: int, alpha: float=0.002, beta: float=1.0,
                   gamma: float=0.05, delta: float=20.0,
                   run_only_dynamic: bool=False,
                   gpus_per_client: float=0.1):
    print("Loading Dataset...")
    fds = setup_dataset()
    

    experiments = ["dynamic"] if run_only_dynamic else ["dynamic", "fixed_1", "fixed_3", "fixed_5", "fixed_10", "fixed_20"]
    all_results = {}

    for exp in experiments:
        def client_fn(context: Context) -> fl.client.Client:
            vehicle = FLVehicle(partition_id=int(cid), trainloader=trainloader, valloader=valloader)
            # Inject weights directly into the client
            vehicle.alpha = alpha
            vehicle.beta = beta
            vehicle.gamma = gamma
            vehicle.delta = delta
            vehicle.mode = exp
            return vehicle.to_client()

        # Pass weights to the strategy
        strategy = DynamicFLStrategy(
            df_network=df, alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            fraction_fit=1.0, fraction_evaluate=1.0, 
            min_fit_clients=num_vehicles, min_available_clients=num_vehicles,
        )

        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_vehicles,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus":gpus_per_client},
        )
        
        losses = [val[1] for val in history.losses_distributed]
        
        all_results[exp] = {
            "mbs": strategy.total_c_net,
            "losses": [val[1] for val in history.losses_distributed],
            "interference": strategy.total_c_comp,
            "drift": strategy.total_c_drift,
            "utility": strategy.total_utility
        }

    # Return the dictionary instead of strictly plotting
    return all_results

def plot_history(all_results):
    plt.figure(figsize=(10, 6))
    colors = {"dynamic": "blue", "fixed_1": "red", "fixed_3": "purple", "fixed_5": "green", "fixed_10": "orange", "fixed_20": "cyan"}
    labels = {"dynamic": "Proposed (Dynamic)", "fixed_1": "Fixed: 1 Epoch", "fixed_3": "Fixed: 3 Epochs", "fixed_5": "Fixed: 5 Epochs", "fixed_10": "Fixed: 10 Epochs", "fixed_20": "Fixed: 20 Epochs"}
    
    for exp, data in all_results.items():
        losses = data["losses"]
        total_mbs = data["mbs"]
        rounds = len(losses)
        mbs_per_round = total_mbs / rounds if rounds > 0 else 0
        x_axis_mbs = [mbs_per_round * (i+1) for i in range(rounds)]
        plt.plot(x_axis_mbs, losses, marker='o', color=colors[exp], label=labels[exp])
        
    plt.title("Validation Loss vs Total Network Cost")
    plt.xlabel("Total Network Cost")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig("results_graph.png")
    print("Graph saved as 'results_graph.png'!")

def plot_interference(all_results):
    plt.figure(figsize=(8, 5))
    colors_map = {"dynamic": "blue", "fixed_1": "red", "fixed_3": "purple", "fixed_5": "green", "fixed_10": "orange", "fixed_20": "cyan"}
    labels_map = {"dynamic": "Proposed (Dynamic)", "fixed_1": "Fixed: 1 Epoch", "fixed_3": "Fixed: 3 Epochs", "fixed_5": "Fixed: 5 Epochs", "fixed_10": "Fixed: 10 Epochs", "fixed_20": "Fixed: 20 Epochs"}
    
    experiments = list(all_results.keys())
    interferences = [all_results[exp]["interference"] for exp in experiments]
    bar_labels = [labels_map[exp] for exp in experiments]
    bar_colors = [colors_map[exp] for exp in experiments]
    
    plt.bar(bar_labels, interferences, color=bar_colors)
    plt.title("Total Compute Interference on Safety-Critical Tasks")
    plt.ylabel("Accumulated Interference Score")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() 
    plt.savefig("interference_graph.png")
    print("Interference graph saved as 'interference_graph.png'!")


def plot_drift(all_results):
    plt.figure(figsize=(8, 5))
    colors_map = {"dynamic": "blue", "fixed_1": "red", "fixed_3": "purple", "fixed_5": "green", "fixed_10": "orange", "fixed_20": "cyan"}
    labels_map = {"dynamic": "Proposed (Dynamic)", "fixed_1": "Fixed: 1 Epoch", "fixed_3": "Fixed: 3 Epochs", "fixed_5": "Fixed: 5 Epochs", "fixed_10": "Fixed: 10 Epochs", "fixed_20": "Fixed: 20 Epochs"}
    
    experiments = list(all_results.keys())
    drifts = [all_results[exp]["drift"] for exp in experiments]
    bar_labels = [labels_map[exp] for exp in experiments]
    bar_colors = [colors_map[exp] for exp in experiments]
    
    plt.bar(bar_labels, drifts, color=bar_colors)
    plt.title("Accumulated Model Drift over Simulation")
    plt.ylabel("Total Model Drift (L2 Norm Distance)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() 
    plt.savefig("drift_graph.png")
    print("Drift graph saved as 'drift_graph.png'!")

def plot_utility(all_results):
    plt.figure(figsize=(8, 5))
    colors_map = {"dynamic": "blue", "fixed_1": "red", "fixed_3": "purple", "fixed_5": "green", "fixed_10": "orange", "fixed_20": "cyan"}
    labels_map = {"dynamic": "Proposed (Dynamic)", "fixed_1": "Fixed: 1 Epoch", "fixed_3": "Fixed: 3 Epochs", "fixed_5": "Fixed: 5 Epochs", "fixed_10": "Fixed: 10 Epochs", "fixed_20": "Fixed: 20 Epochs"}
    
    experiments = list(all_results.keys())
    utilities = [all_results[exp]["utility"] for exp in experiments]
    bar_labels = [labels_map[exp] for exp in experiments]
    bar_colors = [colors_map[exp] for exp in experiments]
    
    plt.bar(bar_labels, utilities, color=bar_colors)
    plt.title("Global System Utility (Learning Gain vs Costs)")
    plt.ylabel("Cumulative Utility Score")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout() 
    plt.savefig("utility_graph.png")
    print("Utility graph saved as 'utility_graph.png'!")