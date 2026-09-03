import argparse

from data_setup import clone_and_run_dataset
from data_processing import load_and_clean_dataset
from scheduler import run_simulation, plot_drift, plot_history, plot_interference, plot_utility
from constants import NUM_VEHICLES, NUM_SAMPLES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["fl_cnn_balanced", "fl_cnn"], help="Model to use for federated learning")
    parser.add_argument("--gpus_per_client", type=float, default=0.1, help="Fraction of GPU per client")
    args = parser.parse_args()
    print("Starting dataset setup and simulation...")
    clone_and_run_dataset()
    dataframe = load_and_clean_dataset(filepath="dataset/data/network_dataset.csv")
    results, model_name = run_simulation(dataframe, NUM_VEHICLES, NUM_SAMPLES, model_name=args.model, gpus_per_client=args.gpus_per_client)
    plot_utility(results, model_name)
    plot_history(results, model_name)
    plot_interference(results, model_name)
    plot_drift(results, model_name)


if __name__ == "__main__":
    main()