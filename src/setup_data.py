from data_processing import load_and_clean_dataset
from data_setup import clone_and_run_dataset

if __name__ == "__main__":
    print("Running one-time network dataset setup...")
    clone_and_run_dataset()
    load_and_clean_dataset(filepath="dataset/data/network_dataset.csv")
    print("Setup complete! Data is ready for simulation workers.")
