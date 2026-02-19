"""
Generate plots from experiment results for paper figures.

Loads a .pkl file produced by ind_belief_experiment.py (single run) or
belief_experiment.py (batch run) and generates publication-quality plots.

Usage:
    python scripts/experiments/plot_results.py results/belief_experiment1_batch_10.pkl
    python scripts/experiments/plot_results.py results/belief_experiment1_batch_10.pkl --out figures/

TODO: Implement plotting based on inspect_results.py analysis.
"""

import sys
import os
import argparse

import dill
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ind_belief_experiment import ExperimentResult


def parse_args():
    parser = argparse.ArgumentParser(description="Plot experiment results")
    parser.add_argument("path", type=str, help="Path to .pkl results file")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for figures (default: show interactively)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.path):
        print(f"File not found: {args.path}")
        sys.exit(1)

    with open(args.path, 'rb') as f:
        data = dill.load(f)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    elif isinstance(data, ExperimentResult):
        results = [data]
    else:
        print(f"Unknown format: {type(data)}")
        sys.exit(1)

    print(f"Loaded {len(results)} episodes from {args.path}")
    print("Plotting not yet implemented. Use inspect_results.py to explore the data first.")


if __name__ == "__main__":
    main()
