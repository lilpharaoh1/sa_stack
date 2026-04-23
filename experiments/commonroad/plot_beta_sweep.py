"""
Plot Riccati steady-state P_ss as a function of beta (gap sensitivity),
and overlay actual covariance convergence curves from experiments.

Shows that at beta=0, lingapkalman reduces to revertkalman.

Usage:
    python experiments/commonroad/plot_beta_sweep.py
    python experiments/commonroad/plot_beta_sweep.py --theory-only
    python experiments/commonroad/plot_beta_sweep.py --betas 0.0 0.005 0.01 0.02 0.05
    python experiments/commonroad/plot_beta_sweep.py -o beta_sweep.pdf
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


def run_sweep_experiment(experiment, method_cfg, human_cfg_dict,
                         seed=21):
    """Run an experiment, return episode dict, clean up results dir."""
    from datetime import datetime
    ts = datetime.now().strftime("%H%M%S%f")
    method_name = f"_sweep_{ts}"
    method_path = os.path.join(SCRIPT_DIR, "configs", "methods",
                               f"{method_name}.json")
    with open(method_path, "w") as f:
        json.dump(method_cfg, f)

    # If human_cfg_dict is a string, use it as a config name.
    # If it's a dict, write a temp human config.
    if isinstance(human_cfg_dict, str):
        human_name = human_cfg_dict
        human_path = None
    else:
        human_name = f"_sweep_human_{ts}"
        human_path = os.path.join(SCRIPT_DIR, "configs", "humans",
                                  f"{human_name}.json")
        with open(human_path, "w") as f:
            json.dump(human_cfg_dict, f)

    cmd = [
        sys.executable, RUN_SCRIPT,
        "-e", experiment,
        "-m", method_name,
        "-hc", human_name,
        "--headless",
        "--seed", str(seed),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    os.remove(method_path)
    if human_path:
        os.remove(human_path)

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
        description="Beta sweep: gap sensitivity vs filter uncertainty")
    p.add_argument("--betas", nargs="*", type=float,
                   default=[0.0, 0.005, 0.01, 0.02, 0.05],
                   help="Beta values to sweep")
    p.add_argument("--alpha", type=float, default=0.9,
                   help="Fixed alpha for all runs")
    p.add_argument("--experiment", "-e", default="expA_acc_at_target")
    p.add_argument("--Q", type=float, default=0.1)
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--theory-only", action="store_true")
    p.add_argument("--out", "-o", type=str, default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    Q, R, alpha = args.Q, args.R, args.alpha

    # The Riccati P_ss doesn't depend on beta (beta is in B, not A).
    # P_ss is the same for all beta values — it only depends on A, Q, R.
    # The difference with beta is in the *estimate trajectory*, not the covariance.
    P_ss = riccati_steady_state(Q, R, A=alpha)

    # --- Figure setup ---
    fig, (ax_cov, ax_dv) = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12,
                        wspace=0.25)

    cmap = plt.colormaps.get_cmap("viridis") if hasattr(plt, 'colormaps') else cm.get_cmap("viridis")
    n_betas = len(args.betas)
    colors = [cmap(i / max(n_betas - 1, 1)) for i in range(n_betas)]

    # Riccati reference (same for all beta since P_ss doesn't depend on beta)
    ax_cov.axhline(np.sqrt(P_ss), color="red", linestyle="--", linewidth=1.5,
                   label=f"$\\sqrt{{P_{{ss}}}}$ = {np.sqrt(P_ss):.4f}  "
                         f"(revertkalman, $\\alpha$={alpha})")

    def _plot_episode(ep, ax_c, ax_d, color, label, linestyle="-", lw=1.5):
        """Helper to plot covariance + dv traces from an episode."""
        kf_P = ep.get("kf_P", [])
        hve = ep.get("human_vel_err", [])
        kk = ep.get("kf_kappa", [])
        ls = ep.get("lead_speed", [])

        steps_p = [t for t, p in enumerate(kf_P) if p is not None]
        sqrtP = [np.sqrt(p) for p in kf_P if p is not None]
        if steps_p:
            ax_c.plot(steps_p, sqrtP, color=color, linewidth=lw,
                      linestyle=linestyle, label=label)

        steps_dv, true_dv, inf_dv = [], [], []
        for t in range(len(ep.get("steps", []))):
            v = ls[t] if t < len(ls) and ls[t] is not None else None
            h = hve[t] if t < len(hve) else None
            e = kk[t] if t < len(kk) and kk[t] is not None else None
            if v and h is not None:
                steps_dv.append(t)
                true_dv.append(h * v)
                inf_dv.append(e * v if e is not None else None)
        if steps_dv:
            ax_d.plot(steps_dv, true_dv, color=color,
                      linewidth=0.5, alpha=0.3, linestyle=linestyle)
            inf_s = [s for s, v in zip(steps_dv, inf_dv) if v is not None]
            inf_v = [v for v in inf_dv if v is not None]
            if inf_s:
                ax_d.plot(inf_s, inf_v, color=color, linewidth=lw,
                          linestyle=linestyle, label=label)

        return sqrtP[-1] if sqrtP else float('nan')

    if not args.theory_only:
        print(f"Running {n_betas} experiments (alpha={alpha}, Q={Q}, R={R})...\n")

        for i, beta in enumerate(args.betas):
            color = colors[i]
            label = f"$\\beta$={beta}"
            print(f"  beta={beta:.4f} ...", end="", flush=True)

            method_cfg = {
                "inference": "lingapkalman", "intervention": "none",
                "kf_Q": Q, "kf_R": R,
                "kf_revert_alpha": alpha, "kf_lingap_beta": beta,
            }
            human_cfg = {
                "type": "lingap", "human_revert_alpha": alpha,
                "human_lingap_beta": beta, "human_walk_Q": Q,
                "action_noise_std": np.sqrt(R),
            }
            ep = run_sweep_experiment(
                args.experiment, method_cfg, human_cfg, seed=args.seed)

            if ep:
                final = _plot_episode(ep, ax_cov, ax_dv, color, label)
                print(f"  OK  (final √P={final:.4f})")
            else:
                print("  FAILED")

        # --- Revertkalman baseline ---
        print(f"\n  revertkalman baseline ...", end="", flush=True)
        rv_method = {
            "inference": "revertkalman", "intervention": "none",
            "kf_Q": Q, "kf_R": R, "kf_revert_alpha": alpha,
        }
        rv_human = {
            "type": "revert", "human_revert_alpha": alpha,
            "human_walk_Q": Q, "action_noise_std": np.sqrt(R),
        }
        rv_ep = run_sweep_experiment(
            args.experiment, rv_method, rv_human, seed=args.seed)

        if rv_ep:
            final = _plot_episode(rv_ep, ax_cov, ax_dv, "black",
                                  "revertkalman", linestyle="--", lw=2)
            print(f"  OK  (final √P={final:.4f})")
        else:
            print("  FAILED")

    # Formatting
    ax_cov.set_xlabel("step", fontsize=11)
    ax_cov.set_ylabel("$\\sqrt{P}$ (m/s)", fontsize=11)
    ax_cov.set_title("Covariance convergence per $\\beta$", fontsize=11)
    ax_cov.legend(fontsize=8, loc="upper right")
    ax_cov.grid(alpha=0.2)

    ax_dv.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax_dv.set_xlabel("step", fontsize=11)
    ax_dv.set_ylabel("$\\delta v$ (m/s)", fontsize=11)
    ax_dv.set_title("Velocity error traces  (faint = true, solid = inferred)",
                    fontsize=11)
    ax_dv.legend(fontsize=8, loc="upper right")
    ax_dv.grid(alpha=0.2)

    fig.suptitle(f"Beta sweep: gap sensitivity  "
                 f"($\\alpha$={alpha}, Q={Q}, R={R})  —  "
                 f"$\\beta$=0 recovers revertkalman",
                 fontsize=13, fontweight="bold")

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
