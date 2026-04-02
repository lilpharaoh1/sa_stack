"""
Plot histogram of action deviation magnitude across all episodes.

X-axis: action deviation (L2) in buckets of 0.1.
Y-axis: relative frequency.

Supports multiple run directories for side-by-side comparison.

Usage:
    python scripts/experiments/plot_intervention_histogram.py results/my_run/
    python scripts/experiments/plot_intervention_histogram.py run1/ run2/ --labels "Greedy" "QCBF" -o hist.png
"""

import sys
import os
import argparse
from typing import List

import dill
import numpy as np
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", ".."))
sys.path.insert(0, _EXPERIMENTS_DIR)

from utils import ExperimentResult, RESULTS_DIR


def _load_run(run_dir: str) -> List[ExperimentResult]:
    """Load all episodes from a run directory."""
    pkl_path = os.path.join(run_dir, "results.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(RESULTS_DIR, run_dir, "results.pkl")
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    elif isinstance(data, ExperimentResult):
        return [data]
    return []


def _collect_deviations(episodes: List[ExperimentResult]) -> np.ndarray:
    """Collect all non-zero action deviations from episodes."""
    vals = []
    for result in episodes:
        for sr in result.steps:
            if sr.action_deviation is not None and sr.action_deviation > 1e-8:
                vals.append(sr.action_deviation)
    return np.array(vals) if vals else np.empty(0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot histogram of action deviation magnitude")
    parser.add_argument("dirs", nargs="+",
                        help="Run directories (or names under results/)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each run directory")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Save figure to file instead of showing")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 6],
                        metavar=("W", "H"))
    parser.add_argument("--bin-width", type=float, default=0.1,
                        help="Histogram bin width (default: 0.1)")
    return parser.parse_args()


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

    # Load data
    all_devs = []
    all_labels = []
    for i, run_dir in enumerate(resolved):
        episodes = _load_run(run_dir)
        if not episodes:
            print(f"Warning: no episodes in {run_dir}")
            continue
        devs = _collect_deviations(episodes)
        all_devs.append(devs)
        if args.labels and i < len(args.labels):
            all_labels.append(args.labels[i])
        else:
            all_labels.append(os.path.basename(run_dir.rstrip('/')))
        print(f"  {all_labels[-1]}: {len(episodes)} episodes, "
              f"{len(devs)} deviated steps")

    if not all_devs:
        print("No data to plot.")
        sys.exit(1)

    # Shared bins
    global_max = max(d.max() for d in all_devs if len(d) > 0)
    bins = np.arange(0, global_max + args.bin_width, args.bin_width)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    for devs, label in zip(all_devs, all_labels):
        if len(devs) == 0:
            continue
        ax.hist(devs, bins=bins, alpha=0.6, label=label,
                weights=np.ones(len(devs)) / len(devs),
                edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Action deviation (L2)', fontsize=11)
    ax.set_ylabel('Relative frequency', fontsize=11)
    ax.set_title('Distribution of Action Deviation Magnitude', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
