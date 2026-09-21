import argparse
from data_setup import clone_and_run_dataset
from data_processing import load_and_clean_dataset
from scheduler import run_simulation
from constants import NUM_VEHICLES, NUM_SAMPLES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus_per_client", type=float, default=0.1, help="Fraction of GPU per client")
    parser.add_argument("--rounds", type=int, default=50, help="Number of federated learning rounds")
    parser.add_argument("--dynamic_only", action="store_true", help="Run only the dynamic policy")
    args = parser.parse_args()

    print("Starting dataset setup and simulation...")
    clone_and_run_dataset()
    dataframe = load_and_clean_dataset(filepath="dataset/data/network_dataset.csv")

    run_simulation(
        dataframe, 
        NUM_VEHICLES, 
        NUM_SAMPLES, 
        num_rounds=args.rounds, 
        run_only_dynamic=args.dynamic_only,
        gpus_per_client=args.gpus_per_client,
    )

if __name__ == "__main__":
    main()