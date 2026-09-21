# compare_sweep.py
import json
import pandas as pd

def parse_sweep_line(line: str) -> dict:
    """Extracts relevant metric payload from a sweep result JSON line."""
    try:
        content = json.loads(line)
        if "dynamic" not in content:
            return None
            
        res, params = content["dynamic"], content["dynamic"].get("params", {})
        losses = res.get("losses", [])
        
        return {
            "job_id": params.get("job_id"),
            "alpha": params.get("alpha"),
            "beta": params.get("beta"),
            "gamma": params.get("gamma"),
            "delta": params.get("delta"),
            "utility": round(res.get("utility", 0.0), 4),
            "final_loss": round(losses[-1], 4) if losses else None,
            "interference": round(res.get("interference", 0.0), 2),
            "drift": round(res.get("drift", 0.0), 2),
            "network_mbs": round(res.get("mbs", 0.0), 2)
        }
    except Exception as e:
        print(f"Error parsing line: {e}")
        return None

def main():
    file_path = "all_sweep_results.jsonl"
    print(f"Reading experiment results from {file_path}...\n")
    
    try:
        with open(file_path, 'r') as f:
            data = [parsed for line in f if (parsed := parse_sweep_line(line)) is not None]
    except FileNotFoundError:
        print(f"No result file found at {file_path}! Ensure workers finished successfully.")
        return
        
    if not data:
        print("No valid data could be extracted.")
        return
        
    # Rank: Highest Utility -> Lowest Final Loss
    df_ranked = pd.DataFrame(data).sort_values(by=["utility", "final_loss"], ascending=[False, True])
    
    print("================ TOP 5 PARAMETER COMBINATIONS ================")
    print(df_ranked.head(5).to_string(index=False))
    print("==============================================================\n")
    
    df_ranked.to_csv("final_sweep_rankings.csv", index=False)
    print("Full ranked results successfully saved to 'final_sweep_rankings.csv'.")

if __name__ == "__main__":
    main()