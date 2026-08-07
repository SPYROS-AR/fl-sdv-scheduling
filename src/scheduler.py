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
logging.basicConfig(
    level=logging.INFO,
    filename='results.log',
    filemode='a'
)
# Basic transforms for CIFAR-10 images
pytorch_transforms = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def run_simulation(df: pd.DataFrame, num_vehicles: int, num_samples: int):
    print("Loading Dataset...")
    fds = setup_dataset()
    
    # List of experiments to run for the evaluation
    experiments = ["dynamic", "fixed_1", "fixed_3", "fixed_5", "fixed_10", "fixed_20"]
    
    # Dictionary to store results for the final plot
    all_results = {}

    # Loop to run the Flower simulation
    for exp in experiments:
        print(f"\n========================================")
        print(f" STARTING EXPERIMENT: {exp}")
        print(f"========================================\n")
        

        def client_fn(context: Context) -> fl.client.Client:
            cid = str(context.node_config["partition-id"]) 
            
            partition = fds.load_partition(int(cid), "train")
            split = partition.train_test_split(test_size=0.15, seed=42)
            
            train_data = split["train"].with_transform(apply_transforms)
            val_data = split["test"].with_transform(apply_transforms)
            
            trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
            valloader = DataLoader(val_data, batch_size=32, shuffle=False)
            
            # Set epochs_per_compute to 10 as the maximum limit for dynamic mode
            vehicle = FLVehicle(
                partition_id=int(cid),
                trainloader=trainloader,
                valloader=valloader,
                model_size_mb=50.0,
                epochs_per_compute=10 
            )
            vehicle.mode = exp 
            return vehicle.to_client()

        strategy = DynamicFLStrategy(
            df_network=df,
            fraction_fit=1.0,
            fraction_evaluate=1.0, 
            min_fit_clients=num_vehicles,
            min_available_clients=num_vehicles,
        )
        
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_vehicles,
            config=fl.server.ServerConfig(num_rounds=40),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
        )
        
        # Save the metrics for the final graph
        total_mbs = strategy.total_mbs_sent
        
        # Extract loss values from the returned history object
        losses = [val[1] for val in history.losses_distributed]
        
        all_results[exp] = {
            "mbs": total_mbs,
            "losses": losses,
            "interference": strategy.total_interference
        }

    print("\nAll experiments finished! Generating the plot...")
    logging.info(all_results)
    plot_history(all_results)
    plot_interference(all_results)


def plot_history(all_results):
    plt.figure(figsize=(10, 6))
    
    colors = {
        "dynamic": "blue", 
        "fixed_1": "red", 
        "fixed_3": "purple", 
        "fixed_5": "green", 
        "fixed_10": "orange", 
        "fixed_20": "cyan"
    }
    
    labels = {
        "dynamic": "Proposed (Dynamic)", 
        "fixed_1": "Fixed: 1 Epoch", 
        "fixed_3": "Fixed: 3 Epochs", 
        "fixed_5": "Fixed: 5 Epochs", 
        "fixed_10": "Fixed: 10 Epochs", 
        "fixed_20": "Fixed: 20 Epochs"
    }
    
    for exp, data in all_results.items():
        losses = data["losses"]
        total_mbs = data["mbs"]
        
        # Distribute total MBs evenly across rounds for the X-axis mapping
        rounds = len(losses)
        mbs_per_round = total_mbs / rounds if rounds > 0 else 0
        x_axis_mbs = [mbs_per_round * (i+1) for i in range(rounds)]
        
        # Now labels[exp] will successfully look up the string key
        plt.plot(x_axis_mbs, losses, marker='o', color=colors[exp], label=labels[exp])

    plt.title("Validation Loss vs Total Megabytes Sent")
    plt.xlabel("Total Data Sent (MBs)")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.legend()
    
    plt.savefig("results_graph.png")
    print("Graph saved as 'results_graph.png'!")
def plot_interference(all_results):
    plt.figure(figsize=(8, 5))
    
    colors_map = {"dynamic": "blue", "fixed_1": "red", "fixed_3": "purple", "fixed_5": "green", "fixed_10": "orange", "fixed_20": "cyan"}
    labels_map = {"dynamic": "Proposed (Dynamic)", "fixed_1": "Fixed: 1 Epoch", "fixed_3": "Fixed: 3 Epochs",
                   "fixed_5": "Fixed: 5 Epochs", "fixed_10": "Fixed: 10 Epochs", "fixed_20": "Fixed: 20 Epochs"}
    
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