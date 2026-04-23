"""
Verify that lingapkalman with beta=0 produces identical results to revertkalman.

Runs both filters on the same revert human (no gap term) and overlays
the rollout metrics. The curves should be identical.

Usage:
    python experiments/commonroad/compare_revert_lingap_beta0.py
    python experiments/commonroad/compare_revert_lingap_beta0.py -e expA_acc_at_target
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)
from sweep_lingap_ablation import run_and_validate


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare revertkalman vs lingapkalman(beta=0)")
    p.add_argument("--experiment", "-e", default="expC_acc_far_start")
    p.add_argument("--alpha", type=float, default=0.9)
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
        "type": "revert",
        "human_revert_alpha": args.alpha,
        "human_walk_Q": args.Q,
        "action_noise_std": np.sqrt(args.R),
    }

    filters = [
        {
            "name": "revertkalman",
            "color": "#3498db",
            "method": {
                "inference": "revertkalman", "intervention": "none",
                "kf_Q": args.Q, "kf_R": args.R,
                "kf_revert_alpha": args.alpha,
            },
        },
        {
            "name": "lingapkalman ($\\beta$=0)",
            "color": "#e74c3c",
            "method": {
                "inference": "lingapkalman", "intervention": "none",
                "kf_Q": args.Q, "kf_R": args.R,
                "kf_revert_alpha": args.alpha,
                "kf_lingap_beta": 0.0,
            },
        },
    ]

    print(f"Comparing revertkalman vs lingapkalman(beta=0)")
    print(f"Human: revert (alpha={args.alpha}, Q={args.Q})")
    print(f"Experiment: {args.experiment}\n")

    results = []
    for f in filters:
        print(f"  {f['name']} ...", end="", flush=True)
        m = run_and_validate(args.experiment, f["method"], human_cfg,
                             horizon=args.horizon, seed=args.seed)
        if m:
            results.append((f, m))
            print(f"  OK")
        else:
            print(f"  FAILED")

    if len(results) < 2:
        print("Need both runs to compare.")
        sys.exit(1)

    # Check numerical equivalence
    _, m1 = results[0]
    _, m2 = results[1]
    rmse_diff = np.nanmax(np.abs(m1["rmse"] - m2["rmse"]))
    cov_diff = np.nanmax(np.abs(m1["coverage"] - m2["coverage"]))
    sharp_diff = np.nanmax(np.abs(m1["sharpness"] - m2["sharpness"]))

    print(f"\n  Max differences:")
    print(f"    RMSE:      {rmse_diff:.6f}")
    print(f"    Coverage:  {cov_diff:.6f}")
    print(f"    Sharpness: {sharp_diff:.6f}")
    if max(rmse_diff, cov_diff, sharp_diff) < 1e-4:
        print(f"    → IDENTICAL (within numerical tolerance)")
    else:
        print(f"    → DIFFERENT")

    # Plot
    fig, (ax_rmse, ax_cov, ax_sharp) = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.08,
                        hspace=0.25)

    for f, m in results:
        h = m["horizons"]
        # Use different line styles so overlap is visible
        ls = "-" if f["name"] == "revertkalman" else "--"
        lw = 2.5 if f["name"] == "revertkalman" else 1.5
        ax_rmse.plot(h, m["rmse"], color=f["color"], linewidth=lw,
                     linestyle=ls, label=f["name"])
        ax_cov.plot(h, m["coverage"] * 100, color=f["color"], linewidth=lw,
                    linestyle=ls, label=f["name"])
        ax_sharp.plot(h, m["sharpness"], color=f["color"], linewidth=lw,
                      linestyle=ls, label=f["name"])

    ax_rmse.set_ylabel("RMSE (m/s)")
    ax_rmse.set_title("Rollout RMSE", fontsize=11)
    ax_rmse.legend(fontsize=9)
    ax_rmse.grid(alpha=0.2)

    ax_cov.axhline(95, color="green", linestyle=":", linewidth=1, alpha=0.5)
    ax_cov.set_ylabel("Coverage (%)")
    ax_cov.set_ylim(0, 105)
    ax_cov.set_title("Coverage rate", fontsize=11)
    ax_cov.legend(fontsize=9)
    ax_cov.grid(alpha=0.2)

    ax_sharp.axhline(1, color="red", linestyle=":", linewidth=1, alpha=0.5)
    ax_sharp.set_ylabel("RMSE / $\\sqrt{P}$")
    ax_sharp.set_xlabel("Horizon (s)")
    ax_sharp.set_title("Sharpness", fontsize=11)
    ax_sharp.legend(fontsize=9)
    ax_sharp.grid(alpha=0.2)

    status = "IDENTICAL" if max(rmse_diff, cov_diff, sharp_diff) < 1e-4 else "DIFFERENT"
    fig.suptitle(f"revertkalman vs lingapkalman($\\beta$=0)  —  {status}\n"
                 f"Human: revert ($\\alpha$={args.alpha}, Q={args.Q}, "
                 f"action noise={np.sqrt(args.R):.1f})   |   "
                 f"exp={args.experiment}",
                 fontsize=12, fontweight="bold")

    out_path = args.out or os.path.join(FIG_DIR, "revert_vs_lingap_beta0.pdf")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
