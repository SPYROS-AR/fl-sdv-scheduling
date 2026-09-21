import logging
import datasets
import flwr as fl
import pandas as pd
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from data_setup import setup_dataset
from client import FLVehicle
from server import DynamicFLStrategy
from constants import BATCH_SIZE, TEST_SPLIT, SEED

datasets.utils.logging.set_verbosity_error()
logging.basicConfig(level=logging.INFO, filename='results.log', filemode='a')

class ClientFactory:
    def __init__(self, dataset, config: dict):
        self.dataset = dataset
        self.config = config
        self.transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def _apply_transforms(self, batch):
        batch["img"] = [self.transforms(img) for img in batch["img"]]
        return batch

    def build_client(self, cid: str) -> fl.client.Client:
        partition = self.dataset.load_partition(int(cid), "train").train_test_split(test_size=TEST_SPLIT, seed=SEED)
        partition = partition.with_transform(self._apply_transforms)
        
        train_loader = DataLoader(partition["train"], batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(partition["test"], batch_size=BATCH_SIZE, shuffle=False)
        
        vehicle = FLVehicle(int(cid), train_loader, val_loader)
        
        vehicle.alpha, vehicle.beta = self.config['alpha'], self.config['beta']
        vehicle.gamma, vehicle.delta = self.config['gamma'], self.config['delta']
        vehicle.mode = self.config['mode']
        
        return vehicle.to_client()

def run_simulation(df: pd.DataFrame, num_vehicles: int, num_samples: int, num_rounds: int = 50,
                   alpha: float=0.002, beta: float=1.0, gamma: float=0.05, delta: float=20.0,
                   run_only_dynamic: bool=False, gpus_per_client: float=0.1, 
                   job_id: str = ""):
                   
    print("Loading Federated Dataset...")
    fds = setup_dataset()
    experiments = ["dynamic"] if run_only_dynamic else ["dynamic", "fixed_1", "fixed_3", "fixed_5", "fixed_10", "fixed_20"]
    all_results = {}
    
    for exp in experiments:
        exp_name = f"{exp}_{job_id}" if job_id else exp
        client_config = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'delta': delta, 'mode': exp}
        
        factory = ClientFactory(fds, client_config)
        strategy = DynamicFLStrategy(
            df_network=df, alpha=alpha, beta=beta, gamma=gamma, delta=delta, exp_name=exp_name,
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=num_vehicles, min_available_clients=num_vehicles,
        )
        
        history = fl.simulation.start_simulation(
            client_fn=factory.build_client,
            num_clients=num_vehicles,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": gpus_per_client},
        )
        
        losses = [val[1] for val in history.losses_distributed] if getattr(history, 'losses_distributed', None) else []
        all_results[exp] = {
            "mbs": strategy.total_c_net,
            "losses": losses,
            "interference": strategy.total_c_comp,
            "drift": strategy.total_c_drift,
            "utility": strategy.total_utility
        }
        
    return all_results