"""
Compare filter architectures when the true human is lingap.

Runs lingapkalman (matched), revertkalman (no gap term), and walkkalman (no reversion)
on the same lingap human and compares rollout prediction quality.

Usage:
    python experiments/commonroad/sweep_filter_comparison.py
    python experiments/commonroad/sweep_filter_comparison.py -e expA_acc_at_target
    python experiments/commonroad/sweep_filter_comparison.py -o filter_comparison.pdf
"""

import os
import sys
import json
import argparse
import subprocess
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
RUN_SCRIPT = os.path.join(SCRIPT_DIR, "run_carfollow.py")
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
os.makedirs(FIG_DIR, exist_ok=True)

# Import the shared rollout computation
sys.path.insert(0, SCRIPT_DIR)
from sweep_lingap_ablation import run_and_validate


FILTERS = [
    {
        "name": "lingapkalman (matched)",
        "color": "#2ecc71",
        "method": {
            "inference": "lingapkalman", "intervention": "none",
            "kf_Q": 0.1, "kf_R": 1.0,
            "kf_revert_alpha": 0.9, "kf_lingap_beta": 0.01,
        },
    },
    {
        "name": "revertkalman (no gap)",
        "color": "#3498db",
        "method": {
            "inference": "revertkalman", "intervention": "none",
            "kf_Q": 0.1, "kf_R": 1.0,
            "kf_revert_alpha": 0.9,
        },
    },
    {
        "name": "walkkalman (no reversion)",
        "color": "#e74c3c",
        "method": {
            "inference": "walkkalman", "intervention": "none",
            "kf_Q": 0.1, "kf_R": 1.0,
        },
    },
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare filter architectures on lingap human")
    p.add_argument("--experiment", "-e", default="expC_acc_far_start")
    p.add_argument("--human-alpha", type=float, default=0.9)
    p.add_argument("--human-beta", type=float, default=0.01)
    p.add_argument("--Q", type=float, default=0.1)
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--out", "-o", type=str, default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    human_cfg = {
        "type": "lingap",
        "human_revert_alpha": args.human_alpha,
        "human_lingap_beta": args.human_beta,
        "human_walk_Q": args.Q,
        "action_noise_std": np.sqrt(args.R),
    }

    print(f"Filter comparison on {args.experiment}")
    print(f"Human: lingap (alpha={args.human_alpha}, beta={args.human_beta}, "
          f"Q={args.Q}, action_noise={np.sqrt(args.R):.2f})\n")

    results = []
    for filt in FILTERS:
        print(f"  {filt['name']} ...", end="", flush=True)
        m = run_and_validate(args.experiment, filt["method"], human_cfg,
                             horizon=args.horizon, seed=args.seed)
        if m:
            results.append((filt, m))
            print(f"  OK  (RMSE@5s={m['rmse'][-1]:.3f}, "
                  f"cov@5s={m['coverage'][-1]*100:.0f}%)")
        else:
            print(f"  FAILED")

    if not results:
        print("No results.")
        sys.exit(1)

    # --- Figure: 4 panels ---
    fig, (ax_rmse, ax_cov, ax_sharp, ax_nees) = plt.subplots(
        4, 1, figsize=(10, 12), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.06,
                        hspace=0.25)

    for filt, m in results:
        color = filt["color"]
        label = filt["name"]
        h = m["horizons"]

        ax_rmse.plot(h, m["rmse"], color=color, linewidth=2, label=label)
        ax_cov.plot(h, m["coverage"] * 100, color=color, linewidth=2, label=label)
        ax_sharp.plot(h, m["sharpness"], color=color, linewidth=2, label=label)
        ax_nees.plot(h, m["nees"], color=color, linewidth=2, label=label)

    ax_rmse.set_ylabel("RMSE (m/s)")
    ax_rmse.set_title("Rollout RMSE vs horizon", fontsize=11)
    ax_rmse.legend(fontsize=9)
    ax_rmse.grid(alpha=0.2)

    ax_cov.axhline(95, color="green", linestyle=":", linewidth=1, alpha=0.5,
                   label="95% target")
    ax_cov.set_ylabel("Coverage (%)")
    ax_cov.set_ylim(0, 105)
    ax_cov.set_title("Coverage rate  ($\\pm 2\\sigma$ band)", fontsize=11)
    ax_cov.legend(fontsize=9)
    ax_cov.grid(alpha=0.2)

    ax_sharp.axhline(1, color="red", linestyle=":", linewidth=1, alpha=0.5,
                     label="ideal = 1")
    ax_sharp.set_ylabel("RMSE / $\\sqrt{P}$")
    ax_sharp.set_title("Sharpness  (< 1 = underconfident, > 1 = overconfident)",
                       fontsize=11)
    ax_sharp.legend(fontsize=9)
    ax_sharp.grid(alpha=0.2)

    ax_nees.axhline(1, color="red", linestyle=":", linewidth=1, alpha=0.5,
                    label="expected = 1")
    ax_nees.set_xlabel("Horizon (s)")
    ax_nees.set_ylabel("Normalized error")
    ax_nees.set_title("Normalized rollout error  (expect $\\approx 1$)", fontsize=11)
    ax_nees.legend(fontsize=9)
    ax_nees.grid(alpha=0.2)

    fig.suptitle(f"Filter comparison on lingap human  ({args.experiment})",
                 fontsize=13, fontweight="bold")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  {'Filter':<30s}  {'RMSE@5s':>8s}  {'Cov@5s':>7s}  "
          f"{'Sharp@5s':>9s}  {'NEES@5s':>8s}")
    print(f"{'='*70}")
    for filt, m in results:
        j = -1  # last horizon step
        print(f"  {filt['name']:<30s}  {m['rmse'][j]:8.3f}  "
              f"{m['coverage'][j]*100:6.1f}%  "
              f"{m['sharpness'][j]:9.3f}  {m['nees'][j]:8.3f}")
    print(f"{'='*70}")

    out_path = args.out or os.path.join(FIG_DIR, "filter_comparison.pdf")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
