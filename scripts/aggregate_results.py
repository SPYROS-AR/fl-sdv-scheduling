#!/usr/bin/env python3
"""Aggregate metrics_<exp>_<aXX>[_s<seed>].csv files over seeds.

Per run it computes the quantities reported in the revised paper (mean accuracy of the
last k rounds instead of the final round, uplink MiB, upload attempts and drops,
compute interference, drift per update, uplink MiB needed to *sustain* a target
accuracy, simulation and V2X-clock times, objective value). Across seeds it reports
mean, standard deviation, n, and the half-width of the 95% t-confidence interval.

Examples
--------
  python scripts/aggregate_results.py final_metrics --out results/
  python scripts/aggregate_results.py runs/ --targets 70 75 --plot
  # two directories holding the same configurations, counted as two seeds:
  python scripts/aggregate_results.py final_metrics final_metrics.bak --dir-as-seed
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

NAME = re.compile(r"^metrics_(?P<exp>.+?)_(?P<alpha>a\d+)(?:_s(?P<seed>\d+))?\.csv$")

# column -> (label in the LaTeX table, number format)
TABLE_COLUMNS = {
    "acc_lastk": ("Acc. last-$k$ (\\%)", "{:.2f}"),
    "uplink_mib": ("Uplink (MiB)", "{:,.0f}"),
    "drops": ("Drops", "{:.0f}"),
    "compute": ("$\\sum C_{\\text{comp}}$", "{:.1f}"),
    "drift_per_update": ("Drift/update", "{:.2f}"),
}


def alpha_value(tag: str) -> float:
    digits = tag[1:]
    return float(f"{digits[0]}.{digits[1:]}") if len(digits) > 1 else float(digits)


def discover(dirs: list[Path], dir_as_seed: bool) -> pd.DataFrame:
    rows = []
    for d_idx, directory in enumerate(dirs):
        for path in sorted(directory.glob("metrics_*.csv")):
            m = NAME.match(path.name)
            if not m:
                continue
            seed = int(m["seed"]) if m["seed"] is not None else (d_idx if dir_as_seed else 0)
            rows.append({"exp": m["exp"], "alpha_tag": m["alpha"],
                         "alpha": alpha_value(m["alpha"]), "seed": seed, "path": path})
    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit(f"No metrics_<exp>_<aXX>[_s<seed>].csv files in {dirs}")
    dup = runs.duplicated(["exp", "alpha_tag", "seed"], keep=False)
    if dup.any():
        raise SystemExit("Duplicate (experiment, alpha, seed):\n"
                         + runs[dup][["exp", "alpha_tag", "seed", "path"]].to_string())
    return runs


def sustain_point(df: pd.DataFrame, target: float, window: int, column: str) -> float:
    acc = 100.0 * df["centralized_accuracy"].astype(float)
    smooth = acc.rolling(window).mean()
    hit = np.flatnonzero(smooth.to_numpy() >= target)
    return float(df[column].iloc[hit[0]]) if len(hit) else float("nan")


def run_metrics(path: Path, last_k: int, targets: list[float], window: int) -> dict:
    df = pd.read_csv(path)
    last = df.iloc[-1]
    acc = 100.0 * df["centralized_accuracy"].astype(float)
    updates = df["continues"].sum() + df["syncs"].sum() + df["dropped"].sum()
    out = {
        "rounds": len(df),
        "acc_lastk": acc.tail(last_k).mean(),
        "acc_lastk_sd": acc.tail(last_k).std(),
        "acc_final": acc.iloc[-1],
        "uplink_mib": float(last["total_uplink_mb"]),
        "transmitted_mib": float(last["total_transmitted_mb"]),
        "attempts": int(df["syncs"].sum() + df["dropped"].sum()),
        "drops": int(df["dropped"].sum()),
        "continues": int(df["continues"].sum()),
        "waits": int(df["waits"].sum()),
        "compute": float(last["total_c_comp"]),
        "drift_per_update": float(last["total_c_drift"]) / updates if updates else float("nan"),
        "objective": float(last["total_utility"]),
        "wall_time_s": float(last["wall_time_s"]),
    }
    for col in ("sim_time_s",):  # columns written by the patched server only
        out[col] = float(last[col]) if col in df.columns else float("nan")
    out["straggler_sum_s"] = (float(df["uplink_straggler_s"].sum())
                              if "uplink_straggler_s" in df.columns else float("nan"))
    for t in targets:
        out[f"uplink_to_{t:g}"] = sustain_point(df, t, window, "total_uplink_mb")
        out[f"round_to_{t:g}"] = sustain_point(df, t, window, "round")
    return out


def ci_halfwidth(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))


def summarize(per_run: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for (exp, alpha), group in per_run.groupby(["exp", "alpha"], sort=True):
        row = {"exp": exp, "alpha": alpha, "n_seeds": len(group)}
        for col in metric_cols:
            vals = group[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
            row[f"{col}_sd"] = np.nanstd(vals, ddof=1) if np.isfinite(vals).sum() > 1 else np.nan
            row[f"{col}_ci95"] = ci_halfwidth(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_cell(mean: float, ci: float, pattern: str) -> str:
    if not np.isfinite(mean):
        return "--"
    text = pattern.format(mean).replace(",", "\\,")
    if np.isfinite(ci):
        text += "$\\pm$" + pattern.format(ci).replace(",", "\\,")
    return text


def latex_table(summary: pd.DataFrame, targets: list[float], last_k: int) -> str:
    cols = dict(TABLE_COLUMNS)
    cols["acc_lastk"] = (f"Acc. last-{last_k} (\\%)", "{:.2f}")
    for t in targets:
        cols[f"uplink_to_{t:g}"] = (f"Uplink to {t:g}\\% (MiB)", "{:,.0f}")
    lines = [
        "% Generated by scripts/aggregate_results.py: mean $\\pm$ 95% CI half-width over seeds",
        "\\begin{tabular}{@{}ll" + "c" * (len(cols) + 1) + "@{}}",
        "\\toprule",
        "$\\alpha_{\\text{Dir}}$ & Scheme & $n$ & " + " & ".join(v[0] for v in cols.values())
        + " \\\\",
        "\\midrule",
    ]
    for alpha, block in summary.groupby("alpha", sort=True):
        for _, r in block.iterrows():
            cells = [fmt_cell(r[f"{c}_mean"], r[f"{c}_ci95"], f) for c, (_, f) in cols.items()]
            exp = r["exp"].replace("_", "\\_")
            lines.append(f"{alpha:g} & {exp} & {int(r['n_seeds'])} & " + " & ".join(cells)
                         + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def plot_accuracy_vs_uplink(runs: pd.DataFrame, out_dir: Path, window: int,
                            ylim: tuple[float, float] | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for alpha, block in runs.groupby("alpha", sort=True):
        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        all_means = []
        for exp, group in block.groupby("exp", sort=True):
            curves = []
            for path in group["path"]:
                df = pd.read_csv(path)
                x = df["total_uplink_mb"].to_numpy(float)
                y = (100.0 * df["centralized_accuracy"].astype(float)).rolling(
                    window, min_periods=1).mean().to_numpy()
                curves.append((x, y))
            x_max = min(c[0][-1] for c in curves)
            grid = np.linspace(0.0, x_max, 200)
            ys = np.array([np.interp(grid, x, y, left=np.nan) for x, y in curves])
            mean = np.nanmean(ys, axis=0)
            all_means.append(mean)
            (line,) = ax.plot(grid, mean, label=f"{exp} (n={len(curves)})", lw=1.4)
            if len(curves) >= 3:  # with n = 2 the t-interval is too wide to be informative
                half = np.array([ci_halfwidth(col) for col in ys.T])
                ax.fill_between(grid, np.clip(mean - half, 0, 100), np.clip(mean + half, 0, 100),
                                alpha=0.2, color=line.get_color(), lw=0)
        if ylim is None:  # zoom on the informative range, past the first-round transient
            values = np.concatenate([m[len(m) // 20:] for m in all_means])
            values = values[np.isfinite(values)]
            if len(values):
                ylim = (max(0.0, np.percentile(values, 5) - 5.0), min(100.0, values.max() + 3.0))
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Cumulative uplink (MiB)")
        ax.set_ylabel(f"Test accuracy, {window}-round mean (%)")
        ax.set_title(f"$\\alpha_{{Dir}}={alpha:g}$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = out_dir / f"acc_vs_uplink_a{str(alpha).replace('.', '')}.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path, help="directories with metrics CSVs")
    ap.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    ap.add_argument("--last-k", type=int, default=10, help="rounds averaged for accuracy")
    ap.add_argument("--targets", type=float, nargs="*", default=[70.0, 75.0],
                    help="accuracy targets (percent) for the uplink-to-sustain metric")
    ap.add_argument("--window", type=int, default=5, help="moving-average window (rounds)")
    ap.add_argument("--dir-as-seed", action="store_true",
                    help="files without _s<seed>: use the directory index as the seed")
    ap.add_argument("--plot", action="store_true", help="accuracy-vs-uplink PDFs per alpha")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="y-axis range of the plots (default: automatic zoom)")
    args = ap.parse_args(argv)

    runs = discover(args.dirs, args.dir_as_seed)
    metrics = [run_metrics(p, args.last_k, args.targets, args.window) for p in runs["path"]]
    per_run = pd.concat([runs.reset_index(drop=True), pd.DataFrame(metrics)], axis=1)
    metric_cols = [c for c in pd.DataFrame(metrics).columns if c != "rounds"]
    summary = summarize(per_run, metric_cols)

    args.out.mkdir(parents=True, exist_ok=True)
    per_run.drop(columns="path").to_csv(args.out / "per_run.csv", index=False)
    summary.to_csv(args.out / "summary.csv", index=False)
    (args.out / "table_results.tex").write_text(latex_table(summary, args.targets, args.last_k))
    written = (plot_accuracy_vs_uplink(runs, args.out, args.window, args.ylim)
               if args.plot else [])

    show = ["exp", "alpha", "n_seeds", "acc_lastk_mean", "acc_lastk_ci95", "uplink_mib_mean",
            "drops_mean", "compute_mean", "drift_per_update_mean"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[show].round(3).to_string(index=False))
    print(f"\nWrote {args.out}/per_run.csv, summary.csv, table_results.tex"
          + (f" and {len(written)} plot(s)" if written else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
