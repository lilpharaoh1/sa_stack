"""
Plot Riccati steady-state P_ss as a function of alpha, and overlay
actual covariance convergence curves from experiments at different alpha values.

Also shows the walkkalman (A=1, random walk) and iidkalman (A=0, i.i.d.)
baselines for comparison.

Usage:
    # Just plot the theoretical curves
    python experiments/commonroad/plot_alpha_sweep.py --theory-only

    # Run experiments at each alpha, then plot
    python experiments/commonroad/plot_alpha_sweep.py

    # Custom alphas
    python experiments/commonroad/plot_alpha_sweep.py --alphas 0.0 0.3 0.5 0.7 0.9 0.95 1.0

    # Save figure
    python experiments/commonroad/plot_alpha_sweep.py -o alpha_sweep.pdf
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
import tempfile

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RUN_SCRIPT = os.path.join(SCRIPT_DIR, "run_carfollow.py")
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")


def riccati_steady_state(Q, R, C=1.0, A=1.0):
    a2c2 = (A * C) ** 2
    b = Q * C ** 2 + R * (1.0 - A ** 2)
    c_coeff = -R * Q
    disc = b ** 2 - 4 * a2c2 * c_coeff
    return (-b + np.sqrt(disc)) / (2 * a2c2)


def run_sweep_experiment(experiment, method_cfg, human_config,
                         seed=21):
    """Run an experiment, return episode dict, clean up results dir."""
    from datetime import datetime
    ts = datetime.now().strftime("%H%M%S%f")
    method_name = f"_sweep_{ts}"
    method_path = os.path.join(SCRIPT_DIR, "configs", "methods",
                               f"{method_name}.json")
    with open(method_path, "w") as f:
        json.dump(method_cfg, f)

    cmd = [
        sys.executable, RUN_SCRIPT,
        "-e", experiment,
        "-m", method_name,
        "-hc", human_config,
        "--headless",
        "--seed", str(seed),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    os.remove(method_path)

    run_dir = None
    for line in result.stdout.splitlines():
        if "Results saved to:" in line:
            run_dir = line.split("Results saved to:")[1].strip()

    episode = None
    if run_dir and os.path.isdir(run_dir):
        ep_path = os.path.join(run_dir, "episode.json")
        if os.path.isfile(ep_path):
            with open(ep_path) as f:
                episode = json.load(f)
        shutil.rmtree(run_dir)

    return episode


def parse_args():
    p = argparse.ArgumentParser(
        description="Alpha sweep: Riccati P_ss vs alpha")
    p.add_argument("--alphas", nargs="*", type=float,
                   default=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
                   help="Alpha values to sweep")
    p.add_argument("--experiment", "-e", default="expA_acc_at_target")
    p.add_argument("--Q", type=float, default=0.1)
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--theory-only", action="store_true",
                   help="Only plot theoretical curves, no experiments")
    p.add_argument("--out", "-o", type=str, default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    Q, R = args.Q, args.R

    # --- Theoretical curve: continuous alpha range ---
    alpha_cont = np.linspace(0.01, 1.0, 200)
    Pss_cont = np.array([riccati_steady_state(Q, R, A=a) for a in alpha_cont])
    sqrtPss_cont = np.sqrt(Pss_cont)

    # Special cases
    Pss_walk = riccati_steady_state(Q, R, A=1.0)   # random walk (alpha=1)
    # For i.i.d., the "process noise" is effectively the full variance of eps.
    # With alpha=0: P_pred = Q, so P_ss = R*Q/(Q+R)
    Pss_iid = R * Q / (Q + R)

    # --- Figure setup ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1, 2]})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.12,
                        wspace=0.25)
    ax_theory = axes[0]
    ax_traces = axes[1]

    # --- Left panel: theoretical sqrt(P_ss) vs alpha ---
    ax_theory.plot(alpha_cont, sqrtPss_cont, "k-", linewidth=2,
                   label="$\\sqrt{P_{ss}}(\\alpha)$")
    ax_theory.axhline(np.sqrt(Pss_walk), color="red", linestyle="--",
                      linewidth=1, alpha=0.7,
                      label=f"walk ($\\alpha$=1): {np.sqrt(Pss_walk):.3f}")
    ax_theory.axhline(np.sqrt(Pss_iid), color="blue", linestyle="--",
                      linewidth=1, alpha=0.7,
                      label=f"i.i.d. ($\\alpha$=0): {np.sqrt(Pss_iid):.3f}")

    # Mark the sweep alphas
    cmap = cm.get_cmap("viridis", len(args.alphas))
    for i, alpha in enumerate(args.alphas):
        Pss_a = riccati_steady_state(Q, R, A=max(alpha, 0.001))
        ax_theory.plot(alpha, np.sqrt(Pss_a), "o", color=cmap(i),
                       markersize=8, zorder=5)

    ax_theory.set_xlabel("$\\alpha$", fontsize=12)
    ax_theory.set_ylabel("$\\sqrt{P_{ss}}$ (m/s)", fontsize=11)
    ax_theory.set_title(f"Riccati steady-state  (Q={Q}, R={R})", fontsize=11)
    ax_theory.legend(fontsize=8, loc="upper left")
    ax_theory.set_xlim(0, 1.05)
    ax_theory.grid(alpha=0.2)

    # --- Right panel: actual convergence traces ---
    if not args.theory_only:
        print(f"Running {len(args.alphas)} experiments...\n")
        for i, alpha in enumerate(args.alphas):
            color = cmap(i)
            label = f"$\\alpha$={alpha:.1f}"
            print(f"  alpha={alpha:.2f} ...", end="", flush=True)

            method_cfg = {
                "inference": "revertkalman",
                "intervention": "none",
                "kf_Q": Q, "kf_R": R,
                "kf_revert_alpha": alpha,
            }
            ep = run_sweep_experiment(
                args.experiment, method_cfg, "revert_default",
                seed=args.seed)

            if ep:
                kf_P = ep.get("kf_P", [])
                steps = [t for t, p in enumerate(kf_P) if p is not None]
                sqrtP = [np.sqrt(p) for p in kf_P if p is not None]
                if steps:
                    ax_traces.plot(steps, sqrtP, color=color, linewidth=1.5,
                                   label=label)
                    Pss_a = riccati_steady_state(Q, R, A=max(alpha, 0.001))
                    ax_traces.axhline(np.sqrt(Pss_a), color=color,
                                      linestyle=":", linewidth=0.8, alpha=0.5)
                print(f"  OK  (final √P={sqrtP[-1]:.4f})")
            else:
                print("  FAILED")

    ax_traces.set_xlabel("step", fontsize=11)
    ax_traces.set_ylabel("$\\sqrt{P}$ (m/s)", fontsize=11)
    ax_traces.set_title("Covariance convergence per $\\alpha$  "
                        "(dotted = $\\sqrt{P_{ss}}$)", fontsize=11)
    ax_traces.legend(fontsize=8, loc="upper right")
    ax_traces.grid(alpha=0.2)

    fig.suptitle("Alpha sweep: mean-reversion rate vs filter uncertainty",
                 fontsize=13, fontweight="bold")

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
