import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


META_CONFIG = {
    "dynamic": {
        "label": "Dynamic FedAvg",
        "color": "#E63946",
        "linestyle": "-",
        "linewidth": 2.0,
    },
    "fixed_1": {
        "label": "Fixed FedAvg: 1 Epoch",
        "color": "#1D3557",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "fixed_3": {
        "label": "Fixed FedAvg: 3 Epochs",
        "color": "#457B9D",
        "linestyle": "--",
        "linewidth": 1.5,
    },
    "fixed_5": {
        "label": "Fixed FedAvg: 5 Epochs",
        "color": "#2A9D8F",
        "linestyle": "-.",
        "linewidth": 1.5,
    },
    "fixed_10": {
        "label": "Fixed FedAvg: 10 Epochs",
        "color": "#E9C46A",
        "linestyle": ":",
        "linewidth": 1.5,
    },
    "fixed_20": {
        "label": "Fixed FedAvg: 20 Epochs",
        "color": "#F4A261",
        "linestyle": ":",
        "linewidth": 1.5,
    },
    "fedprox_5": {
        "label": "Fixed FedProx: 5 Epochs",
        "color": "#6A4C93",
        "linestyle": "-.",
        "linewidth": 2.0,
    },
    "dynamic_fedprox": {
        "label": "Dynamic + FedProx",
        "color": "#1982C4",
        "linestyle": "-",
        "linewidth": 2.0,
    },
}


def setup_plot_style():
    """Apply common publication-oriented Matplotlib defaults."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 2.0,
        }
    )


def _style_for(exp_name: str):
    if exp_name in META_CONFIG:
        return META_CONFIG[exp_name]
    # Supports a non-default CLI value such as fedprox_3 or fedprox_10.
    if exp_name.startswith("fedprox_"):
        return {
            "label": f"Fixed FedProx: {exp_name.split('_', 1)[1]} Epochs",
            "color": "#6A4C93",
            "linestyle": "-.",
            "linewidth": 2.0,
        }
    return {
        "label": exp_name.replace("_", " ").title(),
        "linestyle": "-",
        "linewidth": 1.5,
    }


def load_metrics_data(job_id: str = "") -> dict[str, pd.DataFrame]:
    """Load metrics from the current directory, falling back to final_metrics/."""
    loaded_data = {}
    search_dirs = [Path("."), Path("final_metrics")]
    found_paths = {}

    if job_id:
        pattern = f"metrics_*_{job_id}.csv"
    else:
        pattern = "metrics_*.csv"

    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            # Avoid loading the same experiment twice when both locations contain it.
            if path.name in found_paths:
                continue
            stem = path.stem
            exp_name = stem[len("metrics_") :]
            if job_id:
                if not exp_name.endswith(f"_{job_id}"):
                    continue
                exp_name = exp_name[: -(len(job_id) + 1)]
            else:
                # Ignore job-specific files when no job ID was requested.
                known = {
                    "dynamic",
                    "fixed_1",
                    "fixed_3",
                    "fixed_5",
                    "fixed_10",
                    "fixed_20",
                    "dynamic_fedprox",
                }
                is_fedprox = (
                    exp_name.startswith("fedprox_")
                    and exp_name.split("_")[-1].isdigit()
                )
                if exp_name not in known and not is_fedprox:
                    continue
            found_paths[path.name] = (exp_name, path)

    for exp_name, path in found_paths.values():
        try:
            loaded_data[exp_name] = pd.read_csv(path)
            print(f"Loaded: {path}")
        except (OSError, pd.errors.ParserError) as exc:
            print(f"Warning: could not read {path}: {exc}")

    return loaded_data


def plot_required_metrics(data: dict[str, pd.DataFrame], output_dir: Path):
    """Generate accuracy-vs-cost/time, cumulative compute, and drift figures."""
    plot_specs = [
        {
            "x": "total_transmitted_mb",
            "y": "global_accuracy",
            "filename": "fig1_accuracy_vs_transmitted_data.pdf",
            "title": "Global Accuracy vs. Total Transmitted Data",
            "xlabel": "Cumulative Transmitted Model Payload (MB)",
            "ylabel": "Global Validation Accuracy (%)",
            "percent": True,
        },
        {
            "x": "wall_time_s",
            "y": "global_accuracy",
            "filename": "fig2_accuracy_vs_wall_time.pdf",
            "title": "Global Accuracy vs. Wall-Clock Time",
            "xlabel": "Elapsed Wall-Clock Time (seconds)",
            "ylabel": "Global Validation Accuracy (%)",
            "percent": True,
        },
        {
            "x": "round",
            "y": "total_c_comp",
            "filename": "fig3_cumulative_compute_interference.pdf",
            "title": "Cumulative Compute Interference",
            "xlabel": "Communication Round",
            "ylabel": "Cumulative Compute Interference",
            "percent": False,
        },
        {
            "x": "round",
            "y": "mean_model_drift",
            "filename": "fig4_model_drift.pdf",
            "title": "Mean Local-to-Global Model Drift",
            "xlabel": "Communication Round",
            "ylabel": "Mean Parameter Divergence (sum of layer L2 norms)",
            "percent": False,
        },
    ]

    for spec in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = 0
        for exp_name, frame in data.items():
            if spec["x"] not in frame.columns or spec["y"] not in frame.columns:
                print(
                    f"Skipping {exp_name} for {spec['filename']}: required columns missing"
                )
                continue
            series = frame[[spec["x"], spec["y"]]].copy()
            series[spec["x"]] = pd.to_numeric(series[spec["x"]], errors="coerce")
            series[spec["y"]] = pd.to_numeric(series[spec["y"]], errors="coerce")
            series = series.dropna().sort_values(spec["x"])
            if series.empty:
                continue

            y_values = (
                series[spec["y"]] * 100.0 if spec["percent"] else series[spec["y"]]
            )
            style = _style_for(exp_name)
            ax.plot(
                series[spec["x"]],
                y_values,
                label=style["label"],
                **{key: val for key, val in style.items() if key != "label"},
            )
            plotted += 1

        ax.set_title(spec["title"])
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, linestyle="--", alpha=0.5)
        if plotted:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "No compatible metrics found. Run the updated experiments first.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        fig.tight_layout()
        destination = output_dir / spec["filename"]
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {destination}")


def main(job_id: str = ""):
    setup_plot_style()
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    loaded_data = load_metrics_data(job_id)
    if not loaded_data:
        print("No metrics files found. Run the simulation before plotting.")
        return
    plot_required_metrics(loaded_data, output_dir)
    print(f"Four evaluation figures saved in {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SDV federated-learning metrics.")
    parser.add_argument(
        "--job_id",
        "--job-id",
        dest="job_id",
        type=str,
        default="",
        help="Optional run identifier used in metrics filenames",
    )
    main(job_id=parser.parse_args().job_id)
