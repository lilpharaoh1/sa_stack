"""
Compare car-follow experiment runs: plots + LaTeX table + terminal summary.

Usage:
    # Auto-discover all runs
    python scripts/debug/plot_car_follow.py

    # Specific directories
    python scripts/debug/plot_car_follow.py dir1/ dir2/ dir3/

    # Save figure
    python scripts/debug/plot_car_follow.py --output comparison.pdf

    # Print LaTeX table
    python scripts/debug/plot_car_follow.py --latex
"""

import sys
import os
import json
import argparse
from typing import Dict, List, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "car_follow_results")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_runs(root: str = RESULTS_DIR) -> List[str]:
    """Find all run directories containing episode.json."""
    runs = []
    if not os.path.isdir(root):
        return runs
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isfile(os.path.join(d, "episode.json")):
            runs.append(d)
    return runs


def load_run(run_dir: str) -> dict:
    """Load episode data and metadata from a run directory."""
    with open(os.path.join(run_dir, "episode.json")) as f:
        episode = json.load(f)
    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {}
    return {"episode": episode, "meta": meta, "dir": run_dir}


def run_label(meta: dict) -> str:
    """Short human-readable label for a run."""
    human = meta.get("human", "static")
    inf = meta.get("inference", "none")
    intv = meta.get("intervention", "none")
    return f"{human} / {inf} / {intv}"


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def extract_metrics(episode: dict, meta: dict) -> dict:
    """Compute summary statistics from one episode."""
    human_a = np.array(episode["human_accel"])
    exec_a = np.array(episode["executed_accel"])
    jerk = np.array(episode["jerk"])
    intervened = np.array(episode["intervened"])
    kl_vals = [v for v in episode["kl_divergence"] if v is not None]
    distance = np.array([v for v in episode["distance"] if v is not None])

    accel_dev = np.abs(human_a - exec_a)
    jerk_abs = np.abs(jerk)

    d_target = meta.get("target_distance", 12.0)
    d_safe = meta.get("d_safe", 8.0)

    def stats(arr):
        if len(arr) == 0:
            return {"mean": 0, "std": 0, "max": 0, "p95": 0, "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max": float(np.max(arr)),
            "p95": float(np.percentile(arr, 95)) if len(arr) > 1
                   else float(arr[0]),
            "n": len(arr),
        }

    # --- Tracking performance ---
    if len(distance) > 0:
        d_error = distance - d_target
        d_error_abs = np.abs(d_error)
        d_error_pct = d_error_abs / d_target * 100.0  # % of target
        d_rmse = float(np.sqrt(np.mean(d_error ** 2)))
        d_violation_rate = float(np.mean(distance < d_safe))
    else:
        d_error_abs = np.array([])
        d_error_pct = np.array([])
        d_rmse = 0.0
        d_violation_rate = 0.0

    m = {
        # Agency
        "accel_dev": stats(accel_dev),
        "jerk": stats(jerk_abs),
        "intervention_rate": float(np.mean(intervened)),
        "intervention_count": int(np.sum(intervened)),
        # Belief
        "kl": stats(np.array(kl_vals)) if kl_vals else stats(np.array([])),
        "kl_final": float(kl_vals[-1]) if kl_vals else None,
        # Tracking
        "d_error_pct": stats(d_error_pct),
        "d_rmse": d_rmse,
        "d_violation_rate": d_violation_rate,
        "d_target": d_target,
        "d_safe": d_safe,
        # Meta
        "n_steps": len(human_a),
        # Raw arrays for time-series plots
        "raw_accel_dev": accel_dev.tolist(),
        "raw_jerk": jerk_abs.tolist(),
        "raw_kl": kl_vals,
        "raw_distance": distance.tolist(),
        "raw_human_vel_err": [v for v in episode.get("human_vel_err", [])
                              if v is not None],
    }
    return m


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------

def print_summary(runs: List[dict]):
    """Print a text comparison table to the terminal."""
    labels = [run_label(r["meta"]) for r in runs]
    ml = max(len(l) for l in labels)

    # --- Agency & Comfort ---
    print(f"\n{'Agency & Comfort':=^{ml + 80}}")
    hdr = (f"{'Method':<{ml}}  | "
           f"{'|Da| mean':>9} {'max':>7} {'p95':>7}  "
           f"{'|Jrk| mean':>10} {'max':>7} {'p95':>7}  "
           f"{'Intv%':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        m = r["metrics"]
        ad = m["accel_dev"]
        jk = m["jerk"]
        print(f"{run_label(r['meta']):<{ml}}  | "
              f"{ad['mean']:9.4f} {ad['max']:7.4f} {ad['p95']:7.4f}  "
              f"{jk['mean']:10.4f} {jk['max']:7.4f} {jk['p95']:7.4f}  "
              f"{100*m['intervention_rate']:5.1f}%")

    # --- Tracking ---
    print(f"\n{'Tracking Performance':=^{ml + 80}}")
    hdr2 = (f"{'Method':<{ml}}  | "
            f"{'Err% mean':>9} {'max':>7} {'p95':>7}  "
            f"{'RMSE(m)':>8}  "
            f"{'d<d_safe%':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in runs:
        m = r["metrics"]
        dp = m["d_error_pct"]
        print(f"{run_label(r['meta']):<{ml}}  | "
              f"{dp['mean']:9.2f} {dp['max']:7.2f} {dp['p95']:7.2f}  "
              f"{m['d_rmse']:8.3f}  "
              f"{100*m['d_violation_rate']:8.1f}%")

    # --- Belief ---
    print(f"\n{'Belief':=^{ml + 80}}")
    hdr3 = (f"{'Method':<{ml}}  | "
            f"{'KL mean':>8} {'KL p95':>8} {'KL final':>9}")
    print(hdr3)
    print("-" * len(hdr3))
    for r in runs:
        m = r["metrics"]
        kl_final = f"{m['kl_final']:.4f}" if m["kl_final"] is not None else "N/A"
        print(f"{run_label(r['meta']):<{ml}}  | "
              f"{m['kl']['mean']:8.4f} {m['kl']['p95']:8.4f} {kl_final:>9}")

    print()


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def print_latex(runs: List[dict], bold: bool = True):
    """Print a LaTeX comparison table."""
    n = len(runs)
    labels = [run_label(r["meta"]).replace("_", r"\_") for r in runs]

    # Build column spec with | before baseline columns (none, always_policy)
    _baselines = {"none", "always_policy"}
    col_parts = ["l"]
    prev_baseline = False
    for r in runs:
        is_baseline = r["meta"].get("intervention") in _baselines
        if is_baseline and not prev_baseline:
            col_parts.append("|r")
        else:
            col_parts.append("r")
        prev_baseline = is_baseline
    cols = "".join(col_parts)
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Car-follow experiment comparison}")
    print(r"\label{tab:carfollow}")
    print(r"\begin{tabular}{" + cols + "}")
    print(r"\toprule")
    print("Metric & " + " & ".join(labels) + r" \\")
    print(r"\midrule")

    def _row(name, key_path, fmt=".4f", lower_is_better=True):
        vals = []
        raw = []
        for r in runs:
            m = r["metrics"]
            parts = key_path.split(".")
            v = m
            for p in parts:
                v = v[p] if isinstance(v, dict) and p in v else None
                if v is None:
                    break
            raw.append(v)
            if v is None:
                vals.append("---")
            elif fmt == "pct":
                vals.append(f"{v * 100:.1f}" + r"\%")
            else:
                vals.append(f"{v:{fmt}}")

        if bold and len(vals) > 1:
            # Exclude baselines (none, always_policy) from bolding
            numeric = [(i, raw[i]) for i in range(len(raw))
                       if raw[i] is not None
                       and runs[i]["meta"].get("intervention")
                       not in ("none", "always_policy")]
            if numeric:
                best_fn = min if lower_is_better else max
                best_i = best_fn(numeric, key=lambda x: x[1])[0]
                vals[best_i] = r"\textbf{" + vals[best_i] + "}"

        print(f"{name} & " + " & ".join(vals) + r" \\")

    # --- Agency ---
    print(r"\multicolumn{" + str(n + 1) + r"}{l}{\textit{Agency}} \\")
    _row(r"$|\Delta a|$ mean", "accel_dev.mean")
    _row(r"$|\Delta a|$ p95", "accel_dev.p95")
    _row(r"$|\Delta a|$ max", "accel_dev.max")
    _row(r"Intervention \%", "intervention_rate", fmt="pct")
    print(r"\midrule")

    # --- Comfort ---
    print(r"\multicolumn{" + str(n + 1) + r"}{l}{\textit{Comfort}} \\")
    _row(r"$|\mathrm{jerk}|$ mean", "jerk.mean")
    _row(r"$|\mathrm{jerk}|$ p95", "jerk.p95")
    _row(r"$|\mathrm{jerk}|$ max", "jerk.max")
    print(r"\midrule")

    # --- Tracking ---
    print(r"\multicolumn{" + str(n + 1) + r"}{l}{\textit{Tracking}} \\")
    _row(r"Distance error (\%) mean", "d_error_pct.mean", fmt=".2f")
    _row(r"Distance error (\%) p95", "d_error_pct.p95", fmt=".2f")
    _row(r"Distance error (\%) max", "d_error_pct.max", fmt=".2f")
    _row(r"RMSE (m)", "d_rmse", fmt=".3f")
    _row(r"Safety violation \%", "d_violation_rate", fmt="pct")
    print(r"\midrule")

    # --- Belief ---
    print(r"\multicolumn{" + str(n + 1) + r"}{l}{\textit{Belief}} \\")
    _row(r"KL mean", "kl.mean")
    _row(r"KL final", "kl_final")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(runs: List[dict], output: str = None):
    """2x4 comparison grid across runs."""
    n = len(runs)
    labels = [run_label(r["meta"]) for r in runs]
    x = np.arange(n)
    width = 0.6
    colours = plt.cm.Set2(np.linspace(0, 1, max(n, 3)))

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    def _bar(ax, vals, errs, title, ylabel, fmt=".3f"):
        bars = ax.bar(x, vals, width, yerr=errs if any(e > 0 for e in errs) else None,
                       color=colours[:n], edgecolor="white", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=7)

    # --- Row 0: Agency & Comfort ---

    # |Accel dev| — grouped mean/max/p95
    ax = axes[0, 0]
    w = 0.25
    for j, (stat, clr) in enumerate(zip(["mean", "p95", "max"],
                                         ["steelblue", "salmon", "mediumpurple"])):
        vals = [r["metrics"]["accel_dev"][stat] for r in runs]
        ax.bar(x + (j - 1) * w, vals, w, color=clr, edgecolor="white",
               label=stat)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title("|Accel Deviation|", fontsize=10)
    ax.set_ylabel("m/s²", fontsize=9)
    ax.legend(fontsize=7)

    # |Jerk| — grouped mean/max/p95
    ax = axes[0, 1]
    for j, (stat, clr) in enumerate(zip(["mean", "p95", "max"],
                                         ["steelblue", "salmon", "mediumpurple"])):
        vals = [r["metrics"]["jerk"][stat] for r in runs]
        ax.bar(x + (j - 1) * w, vals, w, color=clr, edgecolor="white",
               label=stat)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title("|Jerk|", fontsize=10)
    ax.set_ylabel("m/s³", fontsize=9)
    ax.legend(fontsize=7)

    # Intervention rate
    vals = [r["metrics"]["intervention_rate"] * 100 for r in runs]
    _bar(axes[0, 2], vals, [0] * n, "Intervention Rate", "%", fmt=".1f")

    # --- Row 1: Tracking & Belief ---

    # Distance error %  — grouped mean/max/p95
    ax = axes[1, 0]
    for j, (stat, clr) in enumerate(zip(["mean", "p95", "max"],
                                         ["steelblue", "salmon", "mediumpurple"])):
        vals = [r["metrics"]["d_error_pct"][stat] for r in runs]
        ax.bar(x + (j - 1) * w, vals, w, color=clr, edgecolor="white",
               label=stat)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title("Distance Tracking Error", fontsize=10)
    ax.set_ylabel("% of $d^*$", fontsize=9)
    ax.legend(fontsize=7)

    # Safety violation rate + RMSE (dual bar)
    ax = axes[1, 1]
    vals_viol = [r["metrics"]["d_violation_rate"] * 100 for r in runs]
    vals_rmse = [r["metrics"]["d_rmse"] for r in runs]
    bars1 = ax.bar(x - 0.15, vals_viol, 0.3, color="salmon",
                    edgecolor="white", label="$d < d_{safe}$ %")
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.15, vals_rmse, 0.3, color="steelblue",
                     edgecolor="white", label="RMSE (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_title("Safety & RMSE", fontsize=10)
    ax.set_ylabel("violation %", fontsize=9, color="salmon")
    ax2.set_ylabel("RMSE (m)", fontsize=9, color="steelblue")
    lines = [bars1, bars2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7,
              loc="upper right")

    # KL over time
    ax_kl = axes[1, 2]
    for i, r in enumerate(runs):
        kl_raw = r["metrics"]["raw_kl"]
        if kl_raw:
            ax_kl.plot(kl_raw, color=colours[i], linewidth=1.2,
                       label=labels[i], alpha=0.8)
    ax_kl.set_title("KL Divergence over time", fontsize=10)
    ax_kl.set_xlabel("step", fontsize=9)
    ax_kl.set_ylabel("KL(oracle || inferred)", fontsize=9)
    ax_kl.legend(fontsize=7, loc="upper right")

    # Human velocity error over time
    ax_hve = axes[0, 3]
    for i, r in enumerate(runs):
        hve = r["metrics"]["raw_human_vel_err"]
        if hve:
            ax_hve.plot(hve, color=colours[i], linewidth=1.2,
                        label=labels[i], alpha=0.8)
    ax_hve.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax_hve.set_title("Human velocity error over time", fontsize=10)
    ax_hve.set_xlabel("step", fontsize=9)
    ax_hve.set_ylabel("$\\kappa$", fontsize=9)
    ax_hve.legend(fontsize=7, loc="upper right")

    # Following distance over time
    ax_dist = axes[1, 3]
    for i, r in enumerate(runs):
        d_raw = r["metrics"]["raw_distance"]
        if d_raw:
            ax_dist.plot(d_raw, color=colours[i], linewidth=1.2,
                         label=labels[i], alpha=0.8)
    d_target = runs[0]["metrics"].get("d_target", 12.0)
    d_safe = runs[0]["metrics"].get("d_safe", 8.0)
    ax_dist.axhline(d_target, color="green", linewidth=1, linestyle="--",
                     alpha=0.6, label=f"$d^*$ = {d_target:.0f}m")
    ax_dist.axhline(d_safe, color="red", linewidth=1, linestyle="--",
                     alpha=0.6, label=f"$d_{{safe}}$ = {d_safe:.0f}m")
    ax_dist.set_title("Following distance over time", fontsize=10)
    ax_dist.set_xlabel("step", fontsize=9)
    ax_dist.set_ylabel("distance (m)", fontsize=9)
    ax_dist.legend(fontsize=7, loc="upper right")

    fig.suptitle("Car-Follow Experiment Comparison", fontsize=13, y=1.01)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output}")
    else:
        backend = matplotlib.get_backend().lower()
        if backend != "agg":
            plt.show(block=True)
        else:
            fallback = "car_follow_comparison.pdf"
            fig.savefig(fallback, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {fallback}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare car-follow experiment runs")
    parser.add_argument("dirs", nargs="*",
                        help="Run directories to compare (default: auto-discover)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save comparison figure to path")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX table")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plot, only print tables")
    parser.add_argument("--bold", action="store_true", default=True,
                        help="Bold best values in LaTeX table (default: True)")
    parser.add_argument("--human", type=str, default=None,
                        choices=["static", "rbf"],
                        help="Filter runs by human type (default: show all)")
    parser.add_argument("--inference", type=str, default=None,
                        help="Filter runs by inference type")
    parser.add_argument("--intervention", type=str, default=None,
                        help="Filter runs by intervention type")
    return parser.parse_args()


def _matches_filters(meta: dict, args) -> bool:
    """Check if a run's metadata matches the CLI filters."""
    if args.human and meta.get("human", "static") != args.human:
        return False
    if args.inference and meta.get("inference") != args.inference:
        return False
    if args.intervention and meta.get("intervention") != args.intervention:
        return False
    return True


def main():
    args = parse_args()

    # Discover or use explicit dirs
    if args.dirs:
        run_dirs = args.dirs
    else:
        run_dirs = discover_runs()
        if not run_dirs:
            print(f"No runs found in {RESULTS_DIR}")
            print("Run car_follow_example.py first, or pass directories explicitly.")
            sys.exit(1)

    print(f"Loading {len(run_dirs)} run(s)...")

    runs = []
    for d in run_dirs:
        try:
            run = load_run(d)
            if not _matches_filters(run["meta"], args):
                continue
            run["metrics"] = extract_metrics(run["episode"], run["meta"])
            runs.append(run)
            human_type = run["meta"].get("human", "static")
            print(f"  [{human_type}] {run_label(run['meta']):30s}  "
                  f"({run['metrics']['n_steps']} steps)  {d}")
        except Exception as e:
            print(f"  SKIP {d}: {e}")

    if not runs:
        filters = []
        if args.human:
            filters.append(f"human={args.human}")
        if args.inference:
            filters.append(f"inference={args.inference}")
        if args.intervention:
            filters.append(f"intervention={args.intervention}")
        filter_str = ", ".join(filters) if filters else "none"
        print(f"No runs matched filters ({filter_str}).")
        sys.exit(1)

    # Terminal summary
    print_summary(runs)

    # LaTeX table
    if args.latex:
        print_latex(runs, bold=args.bold)

    # Plot
    if not args.no_plot:
        plot_comparison(runs, output=args.output)


if __name__ == "__main__":
    main()
