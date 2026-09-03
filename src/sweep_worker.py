import argparse
import json
import pandas as pd
from scheduler import run_simulation
from constants import NUM_VEHICLES, NUM_SAMPLES

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--job_id", type=str, required=True)
    
    parser.add_argument("--gpus_per_client", type=float, default=0.025, help="Fraction of GPU per client")
    
    args = parser.parse_args()
    
    # Ensure dataset exists before workers start pulling
    df = pd.read_csv("dataset/data/cleaned_network_dataset.csv")
    
    # Run only the 'dynamic' policy to evaluate this specific weight combination
    results, model_name = run_simulation(
        df, 
        NUM_VEHICLES, 
        NUM_SAMPLES, 
        alpha=args.alpha, 
        beta=args.beta, 
        gamma=args.gamma, 
        delta=args.delta, 
        run_only_dynamic=True,
        gpus_per_client=args.gpus_per_client
    )
                             
    # Attach the parameters to the payload
    results["dynamic"]["params"] = {
        "alpha": args.alpha, 
        "beta": args.beta, 
        "gamma": args.gamma, 
        "delta": args.delta
    }
    
    out_file = f"results_sweep_{args.job_id}.json"
    with open(out_file, "w") as f:
        json.dump(results, f)
        
    print(f"Worker {args.job_id} finished successfully. Saved to {out_file}")