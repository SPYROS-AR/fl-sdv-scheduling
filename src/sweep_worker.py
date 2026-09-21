# sweep_worker.py
import argparse
import json
import pandas as pd
from filelock import FileLock
from scheduler import run_simulation
from constants import NUM_VEHICLES, NUM_SAMPLES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=50, help="Number of federated learning rounds")
    parser.add_argument("--gpus_per_client", type=float, default=0.025, help="Fraction of GPU per client")
    
    args = parser.parse_args()
    
    df = pd.read_csv("dataset/data/cleaned_network_dataset.csv")
    
    results = run_simulation(
        df,
        NUM_VEHICLES, 
        NUM_SAMPLES, 
        alpha=args.alpha, 
        beta=args.beta, 
        gamma=args.gamma, 
        delta=args.delta, 
        run_only_dynamic=True,
        gpus_per_client=args.gpus_per_client,
        job_id=args.job_id,
        num_rounds=args.rounds,
    )
                                      
    results["dynamic"]["params"] = {
        "job_id": args.job_id,
        "alpha": args.alpha, 
        "beta": args.beta, 
        "gamma": args.gamma, 
        "delta": args.delta
    }
    
    out_file = "all_sweep_results.jsonl"
    lock_file = f"{out_file}.lock"
    
    # Cross-platform safe append
    with FileLock(lock_file):
        with open(out_file, "a") as f:
            f.write(json.dumps(results) + "\n")
            
    print(f"Worker {args.job_id} finished successfully. Appended to {out_file}")