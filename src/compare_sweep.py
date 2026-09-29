import json
import glob
import pandas as pd


def parse_sweep_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            content = json.load(f)

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
            "network_mbs": round(res.get("mbs", 0.0), 2),
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None


def main():
    files = glob.glob("sweep_outputs/result_*.json")
    print(f"Reading {len(files)} experiment results from sweep_outputs/...\n")

    data = [parsed for f in files if (parsed := parse_sweep_file(f)) is not None]

    if not data:
        print("No valid data could be extracted.")
        return

    df_ranked = pd.DataFrame(data).sort_values(
        by=["final_loss", "utility"],
        ascending=[True, True],
    )

    print("================ TOP 5 PARAMETER COMBINATIONS ================")
    print(df_ranked.head(5).to_string(index=False))
    print("==============================================================\n")

    df_ranked.to_csv("final_sweep_rankings.csv", index=False)
    print("Full ranked results successfully saved to 'final_sweep_rankings.csv'.")


if __name__ == "__main__":
    main()
