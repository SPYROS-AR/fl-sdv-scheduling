import subprocess
import os
import sys
import flwr_datasets

from constants import DATASET, NUM_SAMPLES, NUM_VEHICLES, DIRICHLET_ALPHA


def clone_and_run_dataset():
    repo_url = "https://github.com/icsa-hua/CLOUDNET2026-QoS-Offloading.git"
    
    target_folder = "dataset"  
    
    if not os.path.exists(target_folder):
        print(f"Cloning repository from {repo_url} into '{target_folder}'...")
        try:
            subprocess.run(["git", "clone", repo_url, target_folder], check=True)
            print("Clone successful!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            return
        
    repo_root = target_folder
    module_name = "src.dataset.generate"
    

    try:
        subprocess.run(
        [sys.executable, "-W", "ignore", "-m", module_name, "--n", str(NUM_SAMPLES)],
        cwd=repo_root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
        )
        
        print("Dataset generation successful!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate dataset: {e}")
        
        
def setup_dataset()-> flwr_datasets.FederatedDataset:
    """
    Sets up the federated dataset for the vehicles using a Dirichlet partitioning strategy
    """
    
    partitioner = flwr_datasets.partitioner.DirichletPartitioner(
            num_partitions=NUM_VEHICLES,
            partition_by="label",
            alpha=DIRICHLET_ALPHA,
            min_partition_size=10
        )
    
    return flwr_datasets.FederatedDataset(dataset=DATASET, partitioners={"train": partitioner})