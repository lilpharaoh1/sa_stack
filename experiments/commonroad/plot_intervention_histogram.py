"""
Plot histogram of action deviation magnitude across CommonRoad car-follow runs.

Compares the "agency" of two or more solutions: how much each method's
intervention deviates from the human's intended action.

X-axis: |a_human - a_exec| (m/s^2) in buckets of 0.1.
Y-axis: relative frequency.

Usage:
    # Compare two runs by directory name (auto-resolved under results/)
    python experiments/commonroad/plot_intervention_histogram.py \
        exp2_merge_in_front_inf_oracle_int_cbf_wmean_human_static_seed21_20260410_120000 \
        exp2_merge_in_front_inf_oracle_int_cbf_contmean_human_static_seed21_20260410_120100

    # With custom labels and save to file
    python experiments/commonroad/plot_intervention_histogram.py run1/ run2/ \
        --labels "wmean" "contmean" -o hist.png

    # Only count steps where intervention actually happened
    python experiments/commonroad/plot_intervention_histogram.py run1/ run2/ \
        --intervened-only
"""

import sys
import os
import json
import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_episode(run_dir: str) -> dict:
    """Load episode.json (and metadata.json) from a run directory."""
    if not os.path.isabs(run_dir) and not os.path.isdir(run_dir):
        run_dir = os.path.join(RESULTS_DIR, run_dir)
    with open(os.path.join(run_dir, "episode.json")) as f:
        episode = json.load(f)
    meta_path = os.path.join(run_dir, "metadata.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return {"episode": episode, "meta": meta, "dir": run_dir}


def collect_deviations(episode: dict, intervened_only: bool = False,
                       eps: float = 1e-8) -> np.ndarray:
    """Collect |a_human - a_exec| at each step.

    If intervened_only is True, only count steps where the controller
    flagged an intervention.
    """
    human_a = np.array(episode["human_accel"], dtype=float)
    exec_a = np.array(episode["executed_accel"], dtype=float)
    devs = np.abs(human_a - exec_a)

    if intervened_only:
        intervened = np.array(episode["intervened"], dtype=bool)
        devs = devs[intervened]
    else:
        devs = devs[devs > eps]

    return devs


def auto_label(meta: dict, fallback: str) -> str:
    """Build a short label from metadata."""
    if not meta:
        return fallback
    human = meta.get("human", "?")
    inf = meta.get("inference", "?")
    intv = meta.get("intervention", "?")
    return f"{human}/{inf}/{intv}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Histogram of action deviation magnitude "
                    "for CommonRoad car-follow runs")
    p.add_argument("dirs", nargs="+",
                   help="Run directories (or names under results/)")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Labels (default: built from metadata)")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Save figure to file instead of showing")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--figsize", type=float, nargs=2, default=[10, 6],
                   metavar=("W", "H"))
    p.add_argument("--bin-width", type=float, default=0.1,
                   help="Histogram bin width (default: 0.1)")
    p.add_argument("--intervened-only", action="store_true",
                   help="Only include steps flagged as intervened")
    p.add_argument("--max-dev", type=float, default=None,
                   help="Clip x-axis at this max deviation")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve directories
    resolved = []
    for d in args.dirs:
        if os.path.isdir(d):
            resolved.append(d)
        elif os.path.isdir(os.path.join(RESULTS_DIR, d)):
            resolved.append(os.path.join(RESULTS_DIR, d))
        else:
            print(f"Warning: directory not found: {d}")
    if not resolved:
        print("No valid run directories.")
        sys.exit(1)

    # Load and collect deviations
    all_devs: List[np.ndarray] = []
    all_labels: List[str] = []
    for i, run_dir in enumerate(resolved):
        run = load_episode(run_dir)
        devs = collect_deviations(
            run["episode"], intervened_only=args.intervened_only)
        all_devs.append(devs)

        if args.labels and i < len(args.labels):
            label = args.labels[i]
        else:
            label = auto_label(run["meta"], os.path.basename(run_dir.rstrip("/")))
        all_labels.append(label)

        n_steps = len(run["episode"]["steps"])
        n_intv = int(np.sum(run["episode"]["intervened"]))
        print(f"  {label}:  {n_steps} steps, {n_intv} intervened, "
              f"{len(devs)} deviated  (mean dev = "
              f"{devs.mean() if len(devs) else 0.0:.3f}, "
              f"max = {devs.max() if len(devs) else 0.0:.3f})")

    if not any(len(d) for d in all_devs):
        print("No deviations to plot.")
        sys.exit(1)

    # Shared bins
    global_max = max((d.max() for d in all_devs if len(d) > 0), default=1.0)
    if args.max_dev is not None:
        global_max = min(global_max, args.max_dev)
    bins = np.arange(0.0, global_max + args.bin_width, args.bin_width)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
    for i, (devs, label) in enumerate(zip(all_devs, all_labels)):
        if len(devs) == 0:
            continue
        ax.hist(devs, bins=bins, alpha=0.55, label=label,
                color=colors[i % len(colors)],
                weights=np.ones(len(devs)) / len(devs),
                edgecolor="black", linewidth=0.5)

    ax.set_xlabel(r"Action deviation $|a_{\mathrm{human}} - a_{\mathrm{exec}}|$  (m/s$^2$)",
                  fontsize=11)
    ax.set_ylabel("Relative frequency", fontsize=11)
    title = "Distribution of intervention magnitude"
    if args.intervened_only:
        title += "  (intervened steps only)"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(left=0, right=global_max + args.bin_width)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
