import glob
import json

import pandas as pd


def parse_sweep_file(file_path: str) -> dict | None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = json.load(file)
        if "dynamic" not in content:
            return None
        result = content["dynamic"]
        params = result.get("params", {})
        losses = result.get("losses", [])
        return {
            "job_id": params.get("job_id"),
            "alpha": params.get("alpha"),
            "beta": params.get("beta"),
            "gamma": params.get("gamma"),
            "delta": params.get("delta"),
            "utility": round(result.get("utility", 0.0), 4),
            "final_loss": round(losses[-1], 4) if losses else None,
            "interference": round(result.get("interference", 0.0), 2),
            "drift": round(result.get("drift", 0.0), 2),
            "network_mbs": round(result.get("mbs", 0.0), 2),
            "uplink_mbs": round(result.get("uplink_mbs", 0.0), 2),
        }
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"Error parsing {file_path}: {exc}")
        return None


def main():
    files = glob.glob("sweep_outputs/result_*.json")
    print(f"Reading {len(files)} experiment results from sweep_outputs/...\n")
    data = [parsed for path in files if (parsed := parse_sweep_file(path)) is not None]
    if not data:
        print("No valid data could be extracted.")
        return

    # Minimize final_loss (ascending=True), Maximize utility (ascending=False)
    df_ranked = pd.DataFrame(data).sort_values(
        by=["final_loss", "utility"], ascending=[True, False]
    )
    print("================ TOP 5 PARAMETER COMBINATIONS ================")
    print(df_ranked.head(5).to_string(index=False))
    print("==============================================================\n")
    df_ranked.to_csv("final_sweep_rankings.csv", index=False)
    print("Full ranked results saved to 'final_sweep_rankings.csv'.")


if __name__ == "__main__":
    main()
