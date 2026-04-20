"""
Run a battery of matched vs mismatched Kalman filter experiments and
produce a comparison figure showing convergence behaviour.

Demonstrates:
  1. Correct calibration  → P converges to Riccati P_ss
  2. Wrong R (overconfident) → P converges to wrong steady-state
  3. Wrong mean bias       → persistent estimation error
  4. Wrong process model   → filter lags behind dynamics

Usage:
    python experiments/commonroad/run_validation.py
    python experiments/commonroad/run_validation.py --save  # also save GIFs per run
    python experiments/commonroad/run_validation.py -o validation.pdf
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RUN_SCRIPT = os.path.join(SCRIPT_DIR, "run_carfollow.py")


# ---------------------------------------------------------------------------
#  Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "Correct (σ=1, R=1)",
        "experiment": "expA_acc_at_target",
        "human": "gaussian_std_1",
        "inference": "idkalman",
        "intervention": "none",
        "kf_R": 1.0,
        "kf_Q": 0.001,
        "kf_alpha": 0.0,
        "color": "#2ecc71",
        "description": "Filter R matches true noise σ²=1",
    },
    {
        "name": "Wrong R (σ=2, R=1)",
        "experiment": "expA_acc_at_target",
        "human": "gaussian_std_2",
        "inference": "idkalman",
        "intervention": "none",
        "kf_R": 1.0,
        "kf_Q": 0.001,
        "kf_alpha": 0.0,
        "color": "#e74c3c",
        "description": "True noise σ=2 but filter assumes R=1 (overconfident)",
    },
    {
        "name": "Wrong mean (μ=1, σ=1)",
        "experiment": "expA_acc_at_target",
        "human": "gaussian_meanstd_1_1",
        "inference": "idkalman",
        "intervention": "none",
        "kf_R": 1.0,
        "kf_Q": 0.001,
        "kf_alpha": 0.0,
        "color": "#f39c12",
        "description": "True noise N(1,1) but filter assumes zero mean",
    },
    {
        "name": "Wrong process (accel lead, α=0)",
        "experiment": "expB_acc_lead_accel",
        "human": "gaussian_std_1",
        "inference": "idkalman",
        "intervention": "none",
        "kf_R": 1.0,
        "kf_Q": 0.001,
        "kf_alpha": 0.0,
        "color": "#9b59b6",
        "description": "Lead accelerates but filter uses identity process",
    },
]


def riccati_steady_state(Q: float, R: float, C: float = 1.0) -> float:
    a = C ** 2
    disc = (Q * a) ** 2 + 4 * a * Q * R
    return (-Q * a + np.sqrt(disc)) / (2 * a)


def run_experiment(exp: dict, seed: int = 21, save_gif: bool = False) -> str:
    """Run one experiment via subprocess, return the results directory."""
    cmd = [
        sys.executable, RUN_SCRIPT,
        "-e", exp["experiment"],
        "--inference", exp["inference"],
        "--intervention", exp["intervention"],
        "--human", exp["human"],
        "--headless",
        "--seed", str(seed),
    ]
    if save_gif:
        cmd.append("--save")

    # Pass Kalman params via a temporary method config
    ts = datetime.now().strftime("%H%M%S%f")
    method_name = f"_validation_{ts}"
    method_path = os.path.join(SCRIPT_DIR, "configs", "methods",
                               f"{method_name}.json")
    method_cfg = {
        "inference": exp["inference"],
        "intervention": exp["intervention"],
        "human": exp["human"],
        "kf_R": exp.get("kf_R", 1.0),
        "kf_Q": exp.get("kf_Q", 0.001),
        "kf_alpha": exp.get("kf_alpha", 0.0),
    }
    with open(method_path, "w") as f:
        json.dump(method_cfg, f)

    cmd.extend(["-m", method_name])

    print(f"\n  Running: {exp['name']} ...")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.join(SCRIPT_DIR, "..", ".."))
    print(result.stdout[-300:] if result.stdout else "")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")

    # Clean up temp method config
    os.remove(method_path)

    # Find the results directory (last line with "Results saved to:")
    for line in result.stdout.splitlines():
        if "Results saved to:" in line:
            return line.split("Results saved to:")[1].strip()
    return ""


def load_episode(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "episode.json")) as f:
        return json.load(f)


def load_meta(run_dir: str) -> dict:
    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}


def parse_args():
    p = argparse.ArgumentParser(
        description="Kalman filter validation experiments")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Save comparison figure (default: show)")
    p.add_argument("--save", action="store_true",
                   help="Also save GIFs for each run")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Kalman Filter Validation Experiments")
    print("=" * 60)

    # Run all experiments
    results = []
    for exp in EXPERIMENTS:
        run_dir = run_experiment(exp, seed=args.seed, save_gif=args.save)
        if run_dir and os.path.isdir(run_dir):
            episode = load_episode(run_dir)
            meta = load_meta(run_dir)
            results.append({
                "exp": exp,
                "episode": episode,
                "meta": meta,
                "run_dir": run_dir,
            })
        else:
            print(f"  WARNING: No results for {exp['name']}")

    if not results:
        print("No results to plot.")
        sys.exit(1)

    # ------------------------------------------------------------------
    #  Comparison figure: 4 rows
    #    1. sqrt(P) convergence vs Riccati
    #    2. Estimation error (inferred - true) in δv
    #    3. Normalized innovation
    #    4. Absolute δv traces
    # ------------------------------------------------------------------
    fig, (ax_P, ax_err, ax_ni, ax_dv) = plt.subplots(
        4, 1, figsize=(12, 13), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.05,
                        hspace=0.25)

    for r in results:
        exp = r["exp"]
        ep = r["episode"]
        meta = r["meta"]
        color = exp["color"]
        label = exp["name"]

        kf_P_list = ep.get("kf_P", [])
        kf_kappa_list = ep.get("kf_kappa", [])
        human_vel_err = ep.get("human_vel_err", [])
        lead_speed_list = ep.get("lead_speed", [])
        ni_list = ep.get("norm_innovation", [])
        n = len(ep["steps"])

        # Compute Riccati for this experiment's parameters
        Q = meta.get("kf_Q", exp.get("kf_Q", 0.001))
        R = meta.get("kf_R", exp.get("kf_R", 1.0))
        P_ss = riccati_steady_state(Q, R, C=1.0)

        # Build per-step data
        steps, sqrt_P = [], []
        est_err = []  # inferred - true in δv (m/s)
        true_dv, inf_dv = [], []
        t_dv, t_true_dv, t_inf_dv = [], [], []
        ni_steps, ni_vals = [], []

        for t in range(n):
            kf_p = kf_P_list[t] if t < len(kf_P_list) else None
            if kf_p is not None:
                steps.append(t)
                sqrt_P.append(np.sqrt(kf_p))

            v_lead = lead_speed_list[t] if t < len(lead_speed_list) and lead_speed_list[t] is not None else None
            hve = human_vel_err[t] if t < len(human_vel_err) else None
            est = kf_kappa_list[t] if t < len(kf_kappa_list) and kf_kappa_list[t] is not None else None

            if v_lead is not None and hve is not None:
                true_dv_val = hve * v_lead
                t_true_dv.append(t)
                true_dv.append(true_dv_val)

                if est is not None:
                    inf_dv_val = est * v_lead
                    t_inf_dv.append(t)
                    inf_dv.append(inf_dv_val)
                    est_err.append(inf_dv_val - true_dv_val)
                    t_dv.append(t)

        # Collect normalized innovation
        for t in range(n):
            ni = ni_list[t] if t < len(ni_list) else None
            if ni is not None:
                ni_steps.append(t)
                ni_vals.append(ni)

        # Panel 1: sqrt(P) convergence
        ax_P.plot(steps, sqrt_P, color=color, linewidth=1.5, label=label)
        ax_P.axhline(np.sqrt(P_ss), color=color, linestyle="--",
                     linewidth=0.8, alpha=0.5)

        # Panel 2: estimation error
        if est_err:
            ax_err.plot(t_dv, est_err, color=color, linewidth=1.0,
                        alpha=0.7, label=label)

        # Panel 3: normalized innovation
        if ni_vals:
            ax_ni.plot(ni_steps, ni_vals, color=color, linewidth=0.8,
                       alpha=0.7, label=label)

        # Panel 4: true vs inferred δv
        if true_dv:
            ax_dv.plot(t_true_dv, true_dv, color=color, linewidth=0.5,
                       alpha=0.3)
        if inf_dv:
            ax_dv.plot(t_inf_dv, inf_dv, color=color, linewidth=1.5,
                       alpha=0.8, label=label)

    # Formatting
    ax_P.set_ylabel("$\\sqrt{P_{\\delta v}}$ (m/s)")
    ax_P.set_title("Covariance convergence  (dashed = Riccati $P_{ss}$ per experiment)",
                   fontsize=11)
    ax_P.legend(fontsize=8, loc="upper right")
    ax_P.set_ylim(bottom=0)
    ax_P.grid(alpha=0.2)

    ax_err.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax_err.set_ylabel("estimation error (m/s)")
    ax_err.set_title("Estimation error  (inferred $\\delta v$ − true $\\delta v$)",
                     fontsize=11)
    ax_err.legend(fontsize=8, loc="upper right")
    ax_err.grid(alpha=0.2)

    ax_ni.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax_ni.axhspan(-2, 2, color="green", alpha=0.05)
    ax_ni.axhline(2, color="green", linewidth=0.8, linestyle="--", alpha=0.4)
    ax_ni.axhline(-2, color="green", linewidth=0.8, linestyle="--", alpha=0.4)
    ax_ni.set_ylabel("$\\nu / \\sqrt{S}$")
    ax_ni.set_title("Normalized innovation  (green band = $\\pm 2\\sigma$ for well-calibrated filter)",
                    fontsize=11)
    ax_ni.legend(fontsize=8, loc="upper right")
    ax_ni.grid(alpha=0.2)

    ax_dv.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax_dv.set_xlabel("step")
    ax_dv.set_ylabel("$\\delta v$ (m/s)")
    ax_dv.set_title("Velocity error traces  (faint = true, solid = inferred)",
                    fontsize=11)
    ax_dv.legend(fontsize=8, loc="upper right")
    ax_dv.grid(alpha=0.2)

    fig.suptitle("Kalman Filter Validation: Matched vs Mismatched Assumptions",
                 fontsize=13, fontweight="bold")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  {'Experiment':<30s}  {'RMSE':>8s}  {'Mean err':>9s}  {'Final √P':>9s}")
    print(f"{'='*60}")
    for r in results:
        ep = r["episode"]
        meta = r["meta"]
        kf_P_list = ep.get("kf_P", [])
        kf_kappa_list = ep.get("kf_kappa", [])
        human_vel_err = ep.get("human_vel_err", [])
        lead_speed_list = ep.get("lead_speed", [])

        errs = []
        for t in range(len(ep["steps"])):
            v = lead_speed_list[t] if t < len(lead_speed_list) and lead_speed_list[t] is not None else None
            h = human_vel_err[t] if t < len(human_vel_err) else None
            e = kf_kappa_list[t] if t < len(kf_kappa_list) and kf_kappa_list[t] is not None else None
            if v is not None and h is not None and e is not None:
                errs.append((e - h) * v)

        final_P = [p for p in kf_P_list if p is not None]
        rmse = np.sqrt(np.mean(np.array(errs) ** 2)) if errs else float("nan")
        mean_err = np.mean(errs) if errs else float("nan")
        sqrt_P_final = np.sqrt(final_P[-1]) if final_P else float("nan")

        print(f"  {r['exp']['name']:<30s}  {rmse:8.4f}  {mean_err:+9.4f}  {sqrt_P_final:9.4f}")
    print(f"{'='*60}")

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"\n  Figure saved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
