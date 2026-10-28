import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

ALPHA_METADATA = {
    "a01": {
        "title": r"$\alpha_{\mathrm{Dir}} = 0.1$ (Extreme Non-IID)",
        "label": r"$\alpha = 0.1$",
    },
    "a03": {
        "title": r"$\alpha_{\mathrm{Dir}} = 0.3$ (Moderate Non-IID)",
        "label": r"$\alpha = 0.3$",
    },
    "a05": {
        "title": r"$\alpha_{\mathrm{Dir}} = 0.5$ (Mild Non-IID)",
        "label": r"$\alpha = 0.5$",
    },
    "gtsrb_a01": {
        "title": r"GTSRB $\alpha_{\mathrm{Dir}} = 0.1$ (Extreme Non-IID)",
        "label": r"GTSRB $\alpha = 0.1$",
    },
    "gtsrb_a03": {
        "title": r"GTSRB $\alpha_{\mathrm{Dir}} = 0.3$ (Moderate Non-IID)",
        "label": r"GTSRB $\alpha = 0.3$",
    },
    "gtsrb_a05": {
        "title": r"GTSRB $\alpha_{\mathrm{Dir}} = 0.5$ (Mild Non-IID)",
        "label": r"GTSRB $\alpha = 0.5$",
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


def _style_for(exp_name: str) -> dict:
    if exp_name in META_CONFIG:
        return dict(META_CONFIG[exp_name])
    if exp_name.startswith("fedprox_"):
        parts = exp_name.split("_")
        epochs = parts[1] if len(parts) > 1 else "?"
        return {
            "label": f"Fixed FedProx: {epochs} Epochs",
            "color": "#6A4C93",
            "linestyle": "-.",
            "linewidth": 2.0,
        }
    if exp_name.startswith("fixed_"):
        parts = exp_name.split("_")
        epochs = parts[1] if len(parts) > 1 else "?"
        return {
            "label": f"Fixed FedAvg: {epochs} Epochs",
            "linestyle": "--",
            "linewidth": 1.5,
        }
    return {
        "label": exp_name.replace("_", " ").title(),
        "linestyle": "-",
        "linewidth": 1.5,
    }


def repair_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Synthesize legacy columns if missing so older experiment runs can be visualized."""
    df = df.copy()
    if "total_c_comp" not in df.columns and "c_comp" in df.columns:
        df["total_c_comp"] = (
            pd.to_numeric(df["c_comp"], errors="coerce").fillna(0.0).cumsum()
        )
    if "mean_model_drift" not in df.columns and "c_drift" in df.columns:
        syncs = (
            pd.to_numeric(df.get("syncs", 1), errors="coerce").fillna(1).replace(0, 1)
        )
        df["mean_model_drift"] = pd.to_numeric(df["c_drift"], errors="coerce") / syncs
    if "total_transmitted_mb" not in df.columns:
        if "transmitted_mb" in df.columns:
            df["total_transmitted_mb"] = (
                pd.to_numeric(df["transmitted_mb"], errors="coerce")
                .fillna(0.0)
                .cumsum()
            )
        elif "syncs" in df.columns and "round" in df.columns:
            round_mb = (10 * 9.31) + (
                pd.to_numeric(df["syncs"], errors="coerce").fillna(0.0) * 9.31
            )
            df["total_transmitted_mb"] = round_mb.cumsum()
    if "total_uplink_mb" not in df.columns:
        if "uplink_mb" in df.columns:
            df["total_uplink_mb"] = (
                pd.to_numeric(df["uplink_mb"], errors="coerce").fillna(0.0).cumsum()
            )
        elif "syncs" in df.columns:
            df["total_uplink_mb"] = (
                pd.to_numeric(df["syncs"], errors="coerce").fillna(0.0) * 9.31
            ).cumsum()
    return df


def discover_job_ids(search_dirs: list[Path]) -> list[str]:
    """Find all distinct job_id suffixes from metrics_*_<job_id>.csv files."""
    job_ids = set()
    known_prefixes = {
        "dynamic",
        "fixed_1",
        "fixed_3",
        "fixed_5",
        "fixed_10",
        "fixed_20",
        "dynamic_fedprox",
        "fedprox_1",
        "fedprox_3",
        "fedprox_5",
        "fedprox_10",
        "fedprox_20",
    }
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.glob("metrics_*.csv"):
            stem = path.stem[len("metrics_") :]
            parts = stem.split("_")
            for sep_idx in range(len(parts) - 1, 0, -1):
                prefix = "_".join(parts[:sep_idx])
                suffix = "_".join(parts[sep_idx:])
                if prefix in known_prefixes:
                    job_ids.add(suffix)
                    break
    order = {
        "a01": 1,
        "a03": 2,
        "a05": 3,
        "gtsrb_a01": 4,
        "gtsrb_a03": 5,
        "gtsrb_a05": 6,
    }
    return sorted(job_ids, key=lambda x: (order.get(x, 99), x))


def load_metrics_data(job_id: str = "") -> dict[str, pd.DataFrame]:
    """Load metrics matching a specific job_id, falling back across directories."""
    loaded_data = {}
    search_dirs = [Path("."), Path("final_metrics")]
    pattern = f"metrics_*_{job_id}.csv" if job_id else "metrics_*.csv"
    found_paths = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            if path.name in found_paths:
                continue
            stem = path.stem
            exp_name = stem[len("metrics_") :]
            if job_id:
                if not exp_name.endswith(f"_{job_id}"):
                    continue
                exp_name = exp_name[: -(len(job_id) + 1)]
            else:
                if any(
                    exp_name.endswith(f"_{jid}")
                    for jid in ["a01", "a03", "a05", "gtsrb_a01", "gtsrb_a03", "gtsrb_a05"]
                ):
                    continue
            found_paths[path.name] = (exp_name, path)

    for exp_name, path in found_paths.values():
        try:
            df = pd.read_csv(path)
            loaded_data[exp_name] = repair_dataframe_columns(df)
            print(f"Loaded: {path}")
        except (OSError, pd.errors.ParserError) as exc:
            print(f"Warning: could not read {path}: {exc}")
    return loaded_data


def plot_required_metrics(
    data: dict[str, pd.DataFrame], output_dir: Path, job_id: str = ""
):
    """Generate accuracy-vs-cost/time, cumulative compute, and drift figures."""
    plot_specs = [
        {
            "x": "total_transmitted_mb",
            "y": "global_accuracy",
            "filename": "fig1_accuracy_vs_transmitted_data.pdf",
            "title": "Global Accuracy vs. Total Transmitted Data",
            "xlabel": "Cumulative Transmitted Model Payload (MB)",
            "ylabel": "Global Test Accuracy (%)",
            "percent": True,
        },
        {
            "x": "total_uplink_mb",
            "y": "global_accuracy",
            "filename": "fig1b_accuracy_vs_uplink_data.pdf",
            "title": "Global Accuracy vs. Uplink Transmitted Data",
            "xlabel": "Cumulative Uplink Model Payload (MB)",
            "ylabel": "Global Test Accuracy (%)",
            "percent": True,
        },
        {
            "x": "wall_time_s",
            "y": "global_accuracy",
            "filename": "fig2_accuracy_vs_wall_time.pdf",
            "title": "Global Accuracy vs. Wall-Clock Time",
            "xlabel": "Elapsed Wall-Clock Time (seconds)",
            "ylabel": "Global Test Accuracy (%)",
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
            "ylabel": "Mean Parameter Divergence (L2 distance)",
            "percent": False,
        },
        {
            "x": "round",
            "y": "total_utility",
            "filename": "fig6_cumulative_utility.pdf",
            "title": "Cumulative System Utility",
            "xlabel": "Communication Round",
            "ylabel": "Cumulative System Utility",
            "percent": False,
        },
    ]

    for spec in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = 0
        for exp_name, frame in data.items():
            y_col = spec["y"]
            if y_col == "global_accuracy" and "centralized_accuracy" in frame.columns:
                if (
                    pd.to_numeric(frame["centralized_accuracy"], errors="coerce")
                    .notna()
                    .any()
                ):
                    y_col = "centralized_accuracy"
            if spec["x"] not in frame.columns or y_col not in frame.columns:
                print(
                    f"Skipping {exp_name} for {spec['filename']}: required column '{y_col}' or '{spec['x']}' missing"
                )
                continue
            series = frame[[spec["x"], y_col]].copy()
            series[spec["x"]] = pd.to_numeric(series[spec["x"]], errors="coerce")
            series[y_col] = pd.to_numeric(series[y_col], errors="coerce")
            series = series.dropna().sort_values(spec["x"])
            if series.empty:
                continue
            y_values = series[y_col] * 100.0 if spec["percent"] else series[y_col]
            style = _style_for(exp_name)
            ax.plot(
                series[spec["x"]],
                y_values,
                label=style["label"],
                **{key: val for key, val in style.items() if key != "label"},
            )
            plotted += 1

        title_suffix = (
            f" ({ALPHA_METADATA[job_id]['title']})"
            if job_id in ALPHA_METADATA
            else (f" ({job_id})" if job_id else "")
        )
        ax.set_title(f"{spec['title']}{title_suffix}")
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, linestyle="--", alpha=0.5)
        if plotted:
            ax.legend(loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "No compatible metrics found for this plot.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        fig.tight_layout()
        base_name = spec["filename"]
        if job_id:
            p = Path(base_name)
            suffixed_dest = output_dir / f"{p.stem}_{job_id}{p.suffix}"
            fig.savefig(suffixed_dest, dpi=300, bbox_inches="tight")
            print(f"Saved: {suffixed_dest}")
        if not job_id or job_id == "a01":
            main_dest = output_dir / base_name
            fig.savefig(main_dest, dpi=300, bbox_inches="tight")
            print(f"Saved: {main_dest}")
        plt.close(fig)


def plot_client_state_distribution(
    data: dict[str, pd.DataFrame], output_dir: Path, job_id: str = ""
):
    """Generate a stacked bar chart of client operational states across the simulation."""
    order = [
        "fixed_1",
        "fixed_3",
        "fixed_5",
        "fixed_10",
        "fixed_20",
        "fedprox_1",
        "fedprox_3",
        "fedprox_5",
        "fedprox_10",
        "dynamic",
        "dynamic_fedprox",
    ]
    present_exps = [e for e in order if e in data]
    present_exps += [e for e in data.keys() if e not in present_exps]
    if not present_exps:
        return

    labels = []
    syncs, continues, waits, dropped = [], [], [], []
    for exp in present_exps:
        df = data[exp]
        labels.append(_style_for(exp)["label"])
        syncs.append(
            int(pd.to_numeric(df.get("syncs", 0), errors="coerce").fillna(0).sum())
        )
        continues.append(
            int(pd.to_numeric(df.get("continues", 0), errors="coerce").fillna(0).sum())
        )
        waits.append(
            int(pd.to_numeric(df.get("waits", 0), errors="coerce").fillna(0).sum())
        )
        dropped.append(
            int(pd.to_numeric(df.get("dropped", 0), errors="coerce").fillna(0).sum())
        )

    x = np.arange(len(labels))
    width = 0.55
    fig, ax = plt.subplots(figsize=(8, 5))
    syncs_arr = np.array(syncs)
    continues_arr = np.array(continues)
    waits_arr = np.array(waits)
    dropped_arr = np.array(dropped)

    ax.bar(x, syncs_arr, width, label="SYNC (Upload)", color="#2A9D8F")
    ax.bar(
        x,
        continues_arr,
        width,
        bottom=syncs_arr,
        label="CONTINUE (Local Training)",
        color="#E9C46A",
    )
    ax.bar(
        x,
        waits_arr,
        width,
        bottom=syncs_arr + continues_arr,
        label="WAIT (Idle / Yield Slack)",
        color="#F4A261",
    )
    ax.bar(
        x,
        dropped_arr,
        width,
        bottom=syncs_arr + continues_arr + waits_arr,
        label="DROPPED (Buffer Limit)",
        color="#E63946",
    )

    title_suffix = (
        f" ({ALPHA_METADATA[job_id]['title']})"
        if job_id in ALPHA_METADATA
        else (f" ({job_id})" if job_id else "")
    )
    ax.set_title(
        f"Distribution of Client Operational States{title_suffix}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Total Client Occurrences")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.legend(title="Operational State", loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    fig.tight_layout()

    filename = "fig5_client_state_distribution.pdf"
    if job_id:
        suffixed_dest = output_dir / f"fig5_client_state_distribution_{job_id}.pdf"
        fig.savefig(suffixed_dest, dpi=300, bbox_inches="tight")
        print(f"Saved: {suffixed_dest}")
    if not job_id or job_id == "a01":
        main_dest = output_dir / filename
        fig.savefig(main_dest, dpi=300, bbox_inches="tight")
        print(f"Saved: {main_dest}")
    plt.close(fig)


def plot_multi_panel_dirichlet(
    all_jobs_data: dict[str, dict[str, pd.DataFrame]], output_dir: Path
):
    """Generate 1x3 side-by-side subplot figures across Dirichlet alphas (0.1, 0.3, 0.5) for CIFAR-10 and GTSRB."""
    suites = [
        (
            [jid for jid in ["a01", "a03", "a05"] if jid in all_jobs_data],
            "fig1_dirichlet_comparison.pdf",
            "fig4_dirichlet_drift_comparison.pdf",
            "fig5_client_state_distribution_comparison.pdf",
        ),
        (
            [jid for jid in ["gtsrb_a01", "gtsrb_a03", "gtsrb_a05"] if jid in all_jobs_data],
            "fig1_gtsrb_dirichlet_comparison.pdf",
            "fig4_gtsrb_dirichlet_drift_comparison.pdf",
            "fig5_gtsrb_client_state_distribution_comparison.pdf",
        ),
    ]

    for target_alphas, fig1_name, fig4_name, fig5_name in suites:
        if len(target_alphas) < 2:
            continue
        n_cols = len(target_alphas)

        # 1. Multi-panel: Accuracy vs Cumulative Transmitted Payload
        fig, axes = plt.subplots(1, n_cols, figsize=(5.8 * n_cols, 4.8), sharey=True)
        if n_cols == 1:
            axes = [axes]
        legend_handles = {}
        for idx, jid in enumerate(target_alphas):
            ax = axes[idx]
            data = all_jobs_data[jid]
            for exp_name, frame in data.items():
                y_col = (
                    "centralized_accuracy"
                    if "centralized_accuracy" in frame.columns
                    and frame["centralized_accuracy"].dropna().shape[0] > 0
                    else "global_accuracy"
                )
                x_col = "total_transmitted_mb"
                if x_col not in frame.columns or y_col not in frame.columns:
                    continue
                series = frame[[x_col, y_col]].copy()
                series[x_col] = pd.to_numeric(series[x_col], errors="coerce")
                series[y_col] = pd.to_numeric(series[y_col], errors="coerce")
                series = series.dropna().sort_values(x_col)
                if series.empty:
                    continue
                style = _style_for(exp_name)
                (line,) = ax.plot(
                    series[x_col],
                    series[y_col] * 100.0,
                    label=style["label"],
                    **{k: v for k, v in style.items() if k != "label"},
                )
                if style["label"] not in legend_handles:
                    legend_handles[style["label"]] = line
            title = ALPHA_METADATA.get(jid, {}).get("title", f"Partition: {jid}")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Cumulative Payload (MB)", fontsize=11)
            if idx == 0:
                ax.set_ylabel("Global Test Accuracy (%)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
        if legend_handles:
            fig.legend(
                legend_handles.values(),
                legend_handles.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.06),
                ncol=min(len(legend_handles), 4),
                fontsize=10,
            )
        fig.tight_layout()
        comparison_pdf = output_dir / fig1_name
        fig.savefig(comparison_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved multi-panel comparison: {comparison_pdf}")
        plt.close(fig)

        # 2. Multi-panel: Accuracy vs Cumulative Uplink Payload (Pure Uplink)
        fig_up, axes_up = plt.subplots(1, n_cols, figsize=(5.8 * n_cols, 4.8), sharey=True)
        if n_cols == 1:
            axes_up = [axes_up]
        legend_handles_up = {}
        for idx, jid in enumerate(target_alphas):
            ax = axes_up[idx]
            data = all_jobs_data[jid]
            for exp_name, frame in data.items():
                y_col = (
                    "centralized_accuracy"
                    if "centralized_accuracy" in frame.columns
                    and frame["centralized_accuracy"].dropna().shape[0] > 0
                    else "global_accuracy"
                )
                x_col = "total_uplink_mb"
                if x_col not in frame.columns or y_col not in frame.columns:
                    continue
                series = frame[[x_col, y_col]].copy()
                series[x_col] = pd.to_numeric(series[x_col], errors="coerce")
                series[y_col] = pd.to_numeric(series[y_col], errors="coerce")
                series = series.dropna().sort_values(x_col)
                if series.empty:
                    continue
                style = _style_for(exp_name)
                (line,) = ax.plot(
                    series[x_col],
                    series[y_col] * 100.0,
                    label=style["label"],
                    **{k: v for k, v in style.items() if k != "label"},
                )
                if style["label"] not in legend_handles_up:
                    legend_handles_up[style["label"]] = line
            title = ALPHA_METADATA.get(jid, {}).get("title", f"Partition: {jid}")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Cumulative Uplink Payload (MB)", fontsize=11)
            if idx == 0:
                ax.set_ylabel("Global Test Accuracy (%)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
        if legend_handles_up:
            fig_up.legend(
                legend_handles_up.values(),
                legend_handles_up.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.06),
                ncol=min(len(legend_handles_up), 4),
                fontsize=10,
            )
        fig_up.tight_layout()
        uplink_fig_name = fig1_name.replace("fig1_", "fig1b_uplink_")
        uplink_comparison_pdf = output_dir / uplink_fig_name
        fig_up.savefig(uplink_comparison_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved multi-panel uplink comparison: {uplink_comparison_pdf}")
        plt.close(fig_up)

        # 3. Multi-panel: Cumulative Compute Interference vs Round
        fig_comp, axes_comp = plt.subplots(1, n_cols, figsize=(5.8 * n_cols, 4.8), sharey=True)
        if n_cols == 1:
            axes_comp = [axes_comp]
        legend_handles_comp = {}
        for idx, jid in enumerate(target_alphas):
            ax = axes_comp[idx]
            data = all_jobs_data[jid]
            for exp_name, frame in data.items():
                y_col = "total_c_comp"
                x_col = "round"
                if x_col not in frame.columns or y_col not in frame.columns:
                    continue
                series = frame[[x_col, y_col]].copy()
                series[x_col] = pd.to_numeric(series[x_col], errors="coerce")
                series[y_col] = pd.to_numeric(series[y_col], errors="coerce")
                series = series.dropna().sort_values(x_col)
                if series.empty:
                    continue
                style = _style_for(exp_name)
                (line,) = ax.plot(
                    series[x_col],
                    series[y_col],
                    label=style["label"],
                    **{k: v for k, v in style.items() if k != "label"},
                )
                if style["label"] not in legend_handles_comp:
                    legend_handles_comp[style["label"]] = line
            title = ALPHA_METADATA.get(jid, {}).get("title", f"Partition: {jid}")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Communication Round", fontsize=11)
            if idx == 0:
                ax.set_ylabel("Cumulative Compute Interference", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
        if legend_handles_comp:
            fig_comp.legend(
                legend_handles_comp.values(),
                legend_handles_comp.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.06),
                ncol=min(len(legend_handles_comp), 4),
                fontsize=10,
            )
        fig_comp.tight_layout()
        compute_fig_name = fig1_name.replace("fig1_", "fig3_compute_")
        compute_comparison_pdf = output_dir / compute_fig_name
        fig_comp.savefig(compute_comparison_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved multi-panel compute comparison: {compute_comparison_pdf}")
        plt.close(fig_comp)

        # 4. Multi-panel: Cumulative System Utility vs Round
        fig_ut, axes_ut = plt.subplots(1, n_cols, figsize=(5.8 * n_cols, 4.8), sharey=True)
        if n_cols == 1:
            axes_ut = [axes_ut]
        legend_handles_ut = {}
        for idx, jid in enumerate(target_alphas):
            ax = axes_ut[idx]
            data = all_jobs_data[jid]
            for exp_name, frame in data.items():
                y_col = "total_utility"
                x_col = "round"
                if x_col not in frame.columns or y_col not in frame.columns:
                    continue
                series = frame[[x_col, y_col]].copy()
                series[x_col] = pd.to_numeric(series[x_col], errors="coerce")
                series[y_col] = pd.to_numeric(series[y_col], errors="coerce")
                series = series.dropna().sort_values(x_col)
                if series.empty:
                    continue
                style = _style_for(exp_name)
                (line,) = ax.plot(
                    series[x_col],
                    series[y_col],
                    label=style["label"],
                    **{k: v for k, v in style.items() if k != "label"},
                )
                if style["label"] not in legend_handles_ut:
                    legend_handles_ut[style["label"]] = line
            title = ALPHA_METADATA.get(jid, {}).get("title", f"Partition: {jid}")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Communication Round", fontsize=11)
            if idx == 0:
                ax.set_ylabel("Cumulative System Utility", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
        if legend_handles_ut:
            fig_ut.legend(
                legend_handles_ut.values(),
                legend_handles_ut.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.06),
                ncol=min(len(legend_handles_ut), 4),
                fontsize=10,
            )
        fig_ut.tight_layout()
        utility_fig_name = fig1_name.replace("fig1_", "fig6_utility_")
        utility_comparison_pdf = output_dir / utility_fig_name
        fig_ut.savefig(utility_comparison_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved multi-panel utility comparison: {utility_comparison_pdf}")
        plt.close(fig_ut)

        # 5. Multi-panel: Model Drift vs Round
        fig2, axes2 = plt.subplots(1, n_cols, figsize=(5.8 * n_cols, 4.8), sharey=True)
        if n_cols == 1:
            axes2 = [axes2]
        legend_handles2 = {}
        for idx, jid in enumerate(target_alphas):
            ax = axes2[idx]
            data = all_jobs_data[jid]
            for exp_name, frame in data.items():
                y_col = "mean_model_drift"
                x_col = "round"
                if x_col not in frame.columns or y_col not in frame.columns:
                    continue
                series = frame[[x_col, y_col]].copy()
                series[x_col] = pd.to_numeric(series[x_col], errors="coerce")
                series[y_col] = pd.to_numeric(series[y_col], errors="coerce")
                series = series.dropna().sort_values(x_col)
                if series.empty:
                    continue
                style = _style_for(exp_name)
                (line,) = ax.plot(
                    series[x_col],
                    series[y_col],
                    label=style["label"],
                    **{k: v for k, v in style.items() if k != "label"},
                )
                if style["label"] not in legend_handles2:
                    legend_handles2[style["label"]] = line
            title = ALPHA_METADATA.get(jid, {}).get("title", f"Partition: {jid}")
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Communication Round", fontsize=11)
            if idx == 0:
                ax.set_ylabel("Mean Parameter Drift (L2 distance)", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.5)
        if legend_handles2:
            fig2.legend(
                legend_handles2.values(),
                legend_handles2.keys(),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.06),
                ncol=min(len(legend_handles2), 4),
                fontsize=10,
            )
        fig2.tight_layout()
        drift_comparison_pdf = output_dir / fig4_name
        fig2.savefig(drift_comparison_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved multi-panel drift comparison: {drift_comparison_pdf}")
        plt.close(fig2)

        # 3. Multi-panel: Client Operational State Distribution (Stacked Bars)
        fig3, axes3 = plt.subplots(1, n_cols, figsize=(6.0 * n_cols, 4.8), sharey=True)
        if n_cols == 1:
            axes3 = [axes3]
        order = [
            "fixed_1",
            "fixed_3",
            "fixed_5",
            "fixed_10",
            "fixed_20",
            "fedprox_1",
            "fedprox_3",
            "fedprox_5",
            "fedprox_10",
            "dynamic",
            "dynamic_fedprox",
        ]
        p_handles = []
        for idx, jid in enumerate(target_alphas):
            ax = axes3[idx]
            data = all_jobs_data[jid]
            present_exps = [e for e in order if e in data]
            present_exps += [e for e in data.keys() if e not in present_exps]
            labels = [_style_for(e)["label"] for e in present_exps]
            syncs = [
                int(pd.to_numeric(data[e].get("syncs", 0), errors="coerce").fillna(0).sum())
                for e in present_exps
            ]
            continues = [
                int(
                    pd.to_numeric(data[e].get("continues", 0), errors="coerce")
                    .fillna(0)
                    .sum()
                )
                for e in present_exps
            ]
            waits = [
                int(pd.to_numeric(data[e].get("waits", 0), errors="coerce").fillna(0).sum())
                for e in present_exps
            ]
            dropped = [
                int(
                    pd.to_numeric(data[e].get("dropped", 0), errors="coerce")
                    .fillna(0)
                    .sum()
                )
                for e in present_exps
            ]
            x = np.arange(len(labels))
            width = 0.55
            syncs_arr = np.array(syncs)
            continues_arr = np.array(continues)
            waits_arr = np.array(waits)
            dropped_arr = np.array(dropped)
            p1 = ax.bar(x, syncs_arr, width, label="SYNC (Upload)", color="#2A9D8F")
            p2 = ax.bar(
                x,
                continues_arr,
                width,
                bottom=syncs_arr,
                label="CONTINUE (Local Training)",
                color="#E9C46A",
            )
            p3 = ax.bar(
                x,
                waits_arr,
                width,
                bottom=syncs_arr + continues_arr,
                label="WAIT (Idle)",
                color="#F4A261",
            )
            p4 = ax.bar(
                x,
                dropped_arr,
                width,
                bottom=syncs_arr + continues_arr + waits_arr,
                label="DROPPED (Buffer Limit)",
                color="#E63946",
            )
            if not p_handles:
                p_handles = [p1, p2, p3, p4]
            title = ALPHA_METADATA.get(jid, {}).get("title", f"Partition: {jid}")
            ax.set_title(title, fontsize=12, fontweight="bold")
            if idx == 0:
                ax.set_ylabel("Total Client Occurrences", fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
            ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        fig3.legend(
            p_handles,
            [
                "SYNC (Upload)",
                "CONTINUE (Train Locally)",
                "WAIT (Idle / Yield Slack)",
                "DROPPED (Network Limit)",
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            ncol=4,
            fontsize=10,
        )
        fig3.tight_layout()
        states_comparison_pdf = output_dir / fig5_name
        fig3.savefig(states_comparison_pdf, dpi=300, bbox_inches="tight")
        print(f"Saved multi-panel state distribution: {states_comparison_pdf}")
        plt.close(fig3)


def export_latex_table(
    all_jobs_data: dict[str, dict[str, pd.DataFrame]],
    output_path: Path | None = None,
) -> str:
    """Generate and return publication-ready LaTeX Table II code from discovered jobs data."""
    schemes = [
        ("fixed_1", r"Fixed FedAvg ($E = 1$)"),
        ("fixed_5", r"Fixed FedAvg ($E = 5$)"),
        ("fixed_10", r"Fixed FedAvg ($E = 10$)"),
        ("fedprox_5", r"Fixed FedProx ($E = 5, \mu = 0.01$)"),
        ("dynamic", r"\textbf{Dynamic FedAvg (Ours)}"),
        ("dynamic_fedprox", r"\textbf{Dynamic FedProx (Ours)}"),
    ]

    lines = [
        r"\begin{table*}[t]",
        r"\caption{Comprehensive Benchmark Results Across Non-IID Dirichlet Partitions (250 Communication Rounds)}",
        r"\label{tab:evaluation_results}",
        r"\centering",
        r"\begin{tabular}{lccccccr}",
        r"\toprule",
        r"\textbf{Policy / Scheme} & \multicolumn{3}{c}{\textbf{Centralized Test Accuracy (\%)}} & \textbf{Cumulative Uplink} & \textbf{Upload} & \textbf{Compute Contention} & \textbf{Parameter Drift} \\",
        r" & $\alpha = 0.1$ & $\alpha = 0.3$ & $\alpha = 0.5$ & \textbf{Payload (MB)} & \textbf{Drops} & \textbf{Cost ($\sum c_i U_i^{\text{crit}}$)} & \textbf{Cost ($\sum D_i$, $\alpha=0.1 / 0.5$)} \\",
        r"\midrule",
    ]

    for key, name in schemes:
        accs = []
        for a in ["a01", "a03", "a05"]:
            if a in all_jobs_data and key in all_jobs_data[a]:
                df = all_jobs_data[a][key]
                y_col = (
                    "centralized_accuracy"
                    if "centralized_accuracy" in df.columns
                    and df["centralized_accuracy"].dropna().shape[0] > 0
                    else "global_accuracy"
                )
                acc = (
                    pd.to_numeric(df[y_col], errors="coerce").dropna().iloc[-1]
                    * 100.0
                )
                accs.append(f"{acc:.2f}")
            else:
                accs.append("-")

        data01 = all_jobs_data.get("a01", {}).get(key)
        data05 = all_jobs_data.get("a05", {}).get(key)

        if data01 is not None and not data01.empty:
            last01 = data01.iloc[-1]
            up_mb = f"{float(last01.get('total_uplink_mb', 0)):,.1f}"
            drops = f"{int(pd.to_numeric(data01.get('dropped', 0), errors='coerce').fillna(0).sum())}"

            if key in {"dynamic", "dynamic_fedprox"}:
                comp_vals = []
                for a in ["a01", "a03", "a05"]:
                    if a in all_jobs_data and key in all_jobs_data[a]:
                        comp_vals.append(
                            float(
                                all_jobs_data[a][key]
                                .iloc[-1]
                                .get("total_c_comp", 0)
                            )
                        )
                if comp_vals:
                    comp = f"{min(comp_vals):.1f}--{max(comp_vals):.1f}"
                else:
                    comp = f"{float(last01.get('total_c_comp', 0)):.1f}"
            else:
                comp = f"{float(last01.get('total_c_comp', 0)):.1f}"

            drift01 = f"{float(last01.get('total_c_drift', 0)):,.1f}"
        else:
            up_mb = drops = comp = drift01 = "-"

        drift05 = (
            f"{float(data05.iloc[-1].get('total_c_drift', 0)):,.1f}"
            if data05 is not None and not data05.empty
            else "-"
        )
        drift_str = f"{drift01} / {drift05}"

        lines.append(
            f"{name} & {accs[0]} & {accs[1]} & {accs[2]} & {up_mb} & {drops} & {comp} & {drift_str} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    latex_code = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(latex_code)
        print(f"Exported LaTeX table to {output_path}")
    return latex_code


def main(job_id: str = "", export_table: bool = False):
    setup_plot_style()
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    search_dirs = [Path("."), Path("final_metrics")]
    if job_id:
        data = load_metrics_data(job_id)
        if not data:
            print(f"No metrics files found matching job-id '{job_id}'.")
            return
        plot_required_metrics(data, output_dir, job_id=job_id)
        plot_client_state_distribution(data, output_dir, job_id=job_id)
    else:
        discovered_job_ids = discover_job_ids(search_dirs)
        all_jobs_data = {}
        if discovered_job_ids:
            print(f"Discovered job groups: {discovered_job_ids}")
            for jid in discovered_job_ids:
                print(f"\n--- Processing Job ID: {jid} ---")
                job_data = load_metrics_data(jid)
                if job_data:
                    all_jobs_data[jid] = job_data
                    plot_required_metrics(job_data, output_dir, job_id=jid)
                    plot_client_state_distribution(job_data, output_dir, job_id=jid)
            if len(all_jobs_data) >= 2:
                print(
                    "\n--- Generating Multi-Panel Dirichlet Alpha Comparison Figures ---"
                )
                plot_multi_panel_dirichlet(all_jobs_data, output_dir)
            if export_table or len(all_jobs_data) >= 2:
                print("\n--- Generating LaTeX Table II ---")
                tbl = export_latex_table(all_jobs_data, output_dir / "table2_results.tex")
                print(tbl)
        else:
            print("No suffixed metrics found. Loading default unsuffixed metrics...")
            data = load_metrics_data()
            if not data:
                print("No metrics files found. Run the simulation before plotting.")
                return
            plot_required_metrics(data, output_dir)
            plot_client_state_distribution(data, output_dir)
    print(f"\nEvaluation figures saved in {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot SDV federated-learning metrics.")
    parser.add_argument(
        "--job_id",
        "--job-id",
        dest="job_id",
        type=str,
        default="",
        help="Optional run identifier (e.g. 'a01', 'a03', 'a05'). Omit to auto-discover all runs.",
    )
    parser.add_argument(
        "--export-table",
        action="store_true",
        help="Export LaTeX Table II summarizing benchmark results across alphas.",
    )
    args = parser.parse_args()
    main(job_id=args.job_id, export_table=args.export_table)
