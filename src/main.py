from data_setup import clone_and_run_dataset
from data_processing import load_and_clean_dataset
from scheduler import run_simulation
from constants import NUM_VEHICLES, NUM_SAMPLES

import os

os.environ["HF_DATASETS_OFFLINE"] = "1" 

def main():
    print("Starting dataset setup and simulation...")
    clone_and_run_dataset()
    dataframe = load_and_clean_dataset(filepath="dataset/data/network_dataset.csv")
    run_simulation(dataframe, NUM_VEHICLES, NUM_SAMPLES)


if __name__ == "__main__":
    main()