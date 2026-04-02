"""
Histogram comparison of per-step ego cost components across runs.

For each cost component (lateral, speed, accel, steering, heading, total),
shows overlaid histograms from each run so you can compare distributions.

Uses the same CLI interface as plot_intervention_histogram.py for consistency.

Usage:
    python scripts/experiments/plot_cost_scene_histograms.py run1/ run2/
    python scripts/experiments/plot_cost_scene_histograms.py run1/ run2/ --labels "Greedy" "QCBF" -o hist.png
    python scripts/experiments/plot_cost_scene_histograms.py run1/ run2/ -c accel
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

# ── NLP cost weights (SecondStagePlanner.DEFAULTS) ────────────────────
_W = {'w_d': 10.0, 'w_v': 0.01, 'w_a': 1.0, 'w_delta': 2.0,
      'w_phi': 2.0, 'v_target': 10.0}

# ── Component definitions ────────────────────────────────────────────
COMPONENTS = {
    'total':    {'label': 'Total'},
    'lateral':  {'label': r'Lateral ($w_d \cdot d^2$)'},
    'speed':    {'label': r'Speed ($w_v \cdot \Delta v^2$)'},
    'accel':    {'label': r'Accel ($w_a \cdot a^2$)'},
    'steering': {'label': r'Steering ($w_\delta \cdot \delta^2$)'},
    'heading':  {'label': r'Heading ($w_\phi \cdot \phi^2$)'},
}
COMPONENT_ORDER = ['total', 'lateral', 'speed', 'accel', 'steering', 'heading']


# ── Loading ───────────────────────────────────────────────────────────

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


def _collect_costs(episodes: List[ExperimentResult], component: str) -> np.ndarray:
    """Collect per-step cost values for one component from all episodes."""
    vals = []
    for ep in episodes:
        for sr in ep.steps:
            if sr.ego_frenet_state is None:
                continue
            s, d, phi, v = sr.ego_frenet_state[:4]
            a = sr.ego_acceleration if sr.ego_acceleration is not None else 0.0
            delta = sr.ego_steer_angle if sr.ego_steer_angle is not None else 0.0

            c_lat = _W['w_d'] * d ** 2
            c_spd = _W['w_v'] * (v - _W['v_target']) ** 2
            c_acc = _W['w_a'] * a ** 2
            c_str = _W['w_delta'] * delta ** 2
            c_hea = _W['w_phi'] * phi ** 2

            cost_map = {
                'lateral': c_lat,
                'speed': c_spd,
                'accel': c_acc,
                'steering': c_str,
                'heading': c_hea,
                'total': c_lat + c_spd + c_acc + c_str + c_hea,
            }
            vals.append(cost_map[component])
    return np.array(vals) if vals else np.empty(0)


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot histogram of per-step ego cost components")
    parser.add_argument("dirs", nargs="+",
                        help="Run directories (or names under results/)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each run directory")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Save figure to file instead of showing")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 6],
                        metavar=("W", "H"))
    parser.add_argument("--bin-width", type=float, default=None,
                        help="Histogram bin width (default: auto)")
    parser.add_argument("-c", "--component", type=str, default='total',
                        choices=COMPONENT_ORDER,
                        help="Cost component to show (default: total)")
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

    comp = args.component
    info = COMPONENTS[comp]

    # Load data
    all_costs = []
    all_labels = []
    for i, run_dir in enumerate(resolved):
        episodes = _load_run(run_dir)
        if not episodes:
            print(f"Warning: no episodes in {run_dir}")
            continue
        costs = _collect_costs(episodes, comp)
        all_costs.append(costs)
        if args.labels and i < len(args.labels):
            all_labels.append(args.labels[i])
        else:
            all_labels.append(os.path.basename(run_dir.rstrip('/')))
        print(f"  {all_labels[-1]}: {len(episodes)} episodes, "
              f"{len(costs)} steps, mean={costs.mean():.4f}")

    if not all_costs:
        print("No data to plot.")
        sys.exit(1)

    # Shared bins
    global_max = max(d.max() for d in all_costs if len(d) > 0)
    if args.bin_width is not None:
        bins = np.arange(0, global_max + args.bin_width, args.bin_width)
    else:
        bins = np.linspace(0, global_max, 41)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    for costs, label in zip(all_costs, all_labels):
        if len(costs) == 0:
            continue
        ax.hist(costs, bins=bins, alpha=0.6, label=label,
                weights=np.ones(len(costs)) / len(costs),
                edgecolor='black', linewidth=0.5)

    ax.set_xlabel(info['label'], fontsize=11)
    ax.set_ylabel('Relative frequency', fontsize=11)
    ax.set_title(f'Distribution of {info["label"]} Step Costs', fontsize=13)
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
