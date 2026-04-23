"""
Ablation sweep for lingapkalman: misspecify alpha and beta independently.

Runs lingapkalman with varying alpha or beta while the human stays fixed (lingap_default).
Compares rollout prediction quality: RMSE, coverage, and RMSE/sqrt(P) (sharpness).

Usage:
    # Sweep alpha (beta fixed at default)
    python experiments/commonroad/sweep_lingap_ablation.py --sweep alpha

    # Sweep beta (alpha fixed at default)
    python experiments/commonroad/sweep_lingap_ablation.py --sweep beta

    # Both sweeps
    python experiments/commonroad/sweep_lingap_ablation.py --sweep both

    # Custom experiment
    python experiments/commonroad/sweep_lingap_ablation.py --sweep both -e expA_acc_at_target

    # Save
    python experiments/commonroad/sweep_lingap_ablation.py --sweep both -o ablation.pdf
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
VALIDATE_SCRIPT = os.path.join(SCRIPT_DIR, "validate_rollout.py")
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
os.makedirs(FIG_DIR, exist_ok=True)


def run_and_validate(experiment, method_cfg, human_cfg, horizon=50, seed=21):
    """Run experiment, compute rollout metrics, clean up. Returns metrics dict."""
    from datetime import datetime
    ts = datetime.now().strftime("%H%M%S%f")

    method_name = f"_sweep_{ts}"
    method_path = os.path.join(SCRIPT_DIR, "configs", "methods", f"{method_name}.json")
    with open(method_path, "w") as f:
        json.dump(method_cfg, f)

    if isinstance(human_cfg, str):
        human_name = human_cfg
        human_path = None
    else:
        human_name = f"_sweep_human_{ts}"
        human_path = os.path.join(SCRIPT_DIR, "configs", "humans", f"{human_name}.json")
        with open(human_path, "w") as f:
            json.dump(human_cfg, f)

    cmd = [sys.executable, RUN_SCRIPT, "-e", experiment,
           "-m", method_name, "-hc", human_name,
           "--headless", "--seed", str(seed)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    os.remove(method_path)
    if human_path:
        os.remove(human_path)

    run_dir = None
    for line in result.stdout.splitlines():
        if "Results saved to:" in line:
            run_dir = line.split("Results saved to:")[1].strip()

    if not run_dir or not os.path.isdir(run_dir):
        return None

    # Load episode and compute rollout metrics inline
    with open(os.path.join(run_dir, "episode.json")) as f:
        ep = json.load(f)
    with open(os.path.join(run_dir, "metadata.json")) as f:
        meta = json.load(f)

    shutil.rmtree(run_dir)

    return compute_rollout_metrics(ep, meta, horizon)


def compute_rollout_metrics(ep, meta, horizon):
    """Compute rollout RMSE, coverage, sharpness per horizon step."""
    dt = meta.get("dt", 0.1)
    inf_type = meta.get("inference", "")
    if inf_type in ("revertkalman", "lingapkalman"):
        alpha = meta.get("kf_revert_alpha", 0.9)
    elif inf_type == "iidkalman":
        alpha = 0.0
    else:
        alpha = 1.0
    Q = meta.get("kf_Q", 0.1)
    beta = meta.get("kf_lingap_beta", 0.0) if inf_type == "lingapkalman" else 0.0

    kf_vhat = ep.get("kf_vhat", [])
    kf_kappa = ep.get("kf_kappa", [])
    kf_P = ep.get("kf_P", [])
    lead_speed = ep.get("lead_speed", [])
    distance = ep.get("distance", [])
    human_vel_err = ep.get("human_vel_err", [])
    n_steps = len(ep["steps"])

    # Actual v_hat
    actual_vhat = []
    for t in range(n_steps):
        v = lead_speed[t] if t < len(lead_speed) and lead_speed[t] is not None else None
        h = human_vel_err[t] if t < len(human_vel_err) else None
        actual_vhat.append(v * (1.0 + h) if v and h is not None else None)

    # Lead acceleration
    a_leads = [0.0]
    for t in range(1, n_steps):
        v0 = lead_speed[t-1] if t-1 < len(lead_speed) and lead_speed[t-1] else None
        v1 = lead_speed[t] if t < len(lead_speed) and lead_speed[t] else None
        a_leads.append((v1 - v0) / dt if v0 and v1 else 0.0)

    N = min(horizon, n_steps - 1)
    sq_errors = [[] for _ in range(N + 1)]
    norm_errors = [[] for _ in range(N + 1)]
    in_band = [[] for _ in range(N + 1)]
    sharpness = [[] for _ in range(N + 1)]

    for t in range(n_steps):
        vh = kf_vhat[t] if t < len(kf_vhat) and kf_vhat[t] is not None else None
        if vh is None and t < len(kf_kappa) and kf_kappa[t] is not None and t < len(lead_speed) and lead_speed[t]:
            vh = lead_speed[t] * (1.0 + kf_kappa[t])
        p = kf_P[t] if t < len(kf_P) else None
        if vh is None or p is None or t + N >= n_steps:
            continue

        # Rollout
        vhat_j, P_j = vh, p
        for j in range(N + 1):
            act = actual_vhat[t + j]
            if act is not None:
                err = vhat_j - act
                sq_errors[j].append(err ** 2)
                if P_j > 1e-12:
                    norm_errors[j].append(err ** 2 / P_j)
                    in_band[j].append(1.0 if abs(err) <= 2 * np.sqrt(P_j) else 0.0)
                    sharpness[j].append(np.sqrt(err ** 2) / np.sqrt(P_j))

            if j < N:
                v_l = lead_speed[t+j] if t+j < len(lead_speed) and lead_speed[t+j] else 15.0
                a_l = a_leads[t+j] if t+j < len(a_leads) else 0.0
                gap = distance[t+j] if t+j < len(distance) and distance[t+j] else 20.0
                B = (1.0 - alpha) * v_l + a_l * dt + beta * max(gap, 0)
                vhat_j = alpha * vhat_j + B
                P_j = alpha ** 2 * P_j + Q

    horizons = np.arange(N + 1) * dt
    rmse = np.array([np.sqrt(np.mean(s)) if s else np.nan for s in sq_errors])
    coverage = np.array([np.mean(s) if s else np.nan for s in in_band])
    mean_nees = np.array([np.mean(s) if s else np.nan for s in norm_errors])
    mean_sharp = np.array([np.mean(s) if s else np.nan for s in sharpness])

    return {
        "horizons": horizons,
        "rmse": rmse,
        "coverage": coverage,
        "nees": mean_nees,
        "sharpness": mean_sharp,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Lingapkalman ablation sweep")
    p.add_argument("--sweep", choices=["alpha", "beta", "both"], default="both")
    p.add_argument("--target", choices=["filter", "human"], default="filter",
                   help="What to sweep: 'filter' fixes human and sweeps filter params, "
                        "'human' fixes filter and sweeps human params")
    p.add_argument("--experiment", "-e", default="expC_acc_far_start")
    p.add_argument("--alphas", nargs="*", type=float,
                   default=[0.5, 0.7, 0.9, 0.95, 1.0])
    p.add_argument("--betas", nargs="*", type=float,
                   default=[0.0, 0.005, 0.01, 0.02, 0.05])
    p.add_argument("--default-alpha", type=float, default=0.9)
    p.add_argument("--default-beta", type=float, default=0.01)
    p.add_argument("--Q", type=float, default=0.1)
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--out", "-o", type=str, default=None)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def plot_sweep(ax_rmse, ax_cov, ax_sharp, results, matched_label=None,
               cmap_name="viridis"):
    """Plot sweep results, highlighting the matched case."""
    n = len(results)
    cmap = plt.colormaps.get_cmap(cmap_name) if hasattr(plt, 'colormaps') else plt.cm.get_cmap(cmap_name)
    for i, (label, m, is_matched) in enumerate(results):
        color = cmap(i / max(n - 1, 1))
        lw = 3.0 if is_matched else 1.5
        ls = "-" if is_matched else "-"
        marker = "o" if is_matched else None
        markevery = 10 if is_matched else None
        suffix = " (matched)" if is_matched else ""
        ax_rmse.plot(m["horizons"], m["rmse"], color=color, linewidth=lw,
                     linestyle=ls, marker=marker, markevery=markevery,
                     markersize=4, label=label + suffix)
        ax_cov.plot(m["horizons"], m["coverage"] * 100, color=color,
                    linewidth=lw, linestyle=ls, marker=marker,
                    markevery=markevery, markersize=4, label=label + suffix)
        ax_sharp.plot(m["horizons"], m["sharpness"], color=color,
                      linewidth=lw, linestyle=ls, marker=marker,
                      markevery=markevery, markersize=4, label=label + suffix)


def _run_sweep(args, param_name, values, default_val,
               make_method_cfg, make_human_cfg):
    """Run a sweep over one parameter, return list of (label, metrics, is_matched)."""
    results = []
    for val in values:
        is_matched = abs(val - default_val) < 1e-8
        label = f"${param_name}$={val}"
        print(f"  {param_name}={val:.4f} ...", end="", flush=True)

        method_cfg = make_method_cfg(val)
        human_cfg = make_human_cfg(val)

        m = run_and_validate(args.experiment, method_cfg, human_cfg,
                             horizon=args.horizon, seed=args.seed)
        if m:
            results.append((label, m, is_matched))
            print(f"  OK" + ("  ← matched" if is_matched else ""))
        else:
            print(f"  FAILED")
    return results


def main():
    args = parse_args()

    target = args.target  # "filter" or "human"
    do_alpha = args.sweep in ("alpha", "both")
    do_beta = args.sweep in ("beta", "both")
    n_panels = (1 if do_alpha else 0) + (1 if do_beta else 0)

    # Fixed configs (the side that doesn't get swept)
    fixed_method = {
        "inference": "lingapkalman", "intervention": "none",
        "kf_Q": args.Q, "kf_R": args.R,
        "kf_revert_alpha": args.default_alpha,
        "kf_lingap_beta": args.default_beta,
    }
    fixed_human = {
        "type": "lingap",
        "human_revert_alpha": args.default_alpha,
        "human_lingap_beta": args.default_beta,
        "human_walk_Q": args.Q,
        "action_noise_std": np.sqrt(args.R),
    }

    sweep_side = "filter" if target == "filter" else "human"
    fixed_side = "human" if target == "filter" else "filter"

    print(f"Sweeping {sweep_side} params, {fixed_side} fixed")
    print(f"Matched: alpha={args.default_alpha}, beta={args.default_beta}, "
          f"Q={args.Q}, R={args.R}")
    print(f"Experiment: {args.experiment}\n")

    fig, all_axes = plt.subplots(3, n_panels, figsize=(7 * n_panels, 9),
                                  squeeze=False, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.08,
                        hspace=0.30, wspace=0.25)

    col = 0

    if do_alpha:
        print(f"Alpha sweep ({sweep_side})...\n")

        if target == "filter":
            def make_m(a):
                return {**fixed_method, "kf_revert_alpha": a}
            def make_h(a):
                return fixed_human
        else:
            def make_m(a):
                return fixed_method
            def make_h(a):
                return {**fixed_human, "human_revert_alpha": a}

        alpha_results = _run_sweep(
            args, "\\alpha", args.alphas, args.default_alpha,
            make_m, make_h)

        ax_r, ax_c, ax_s = all_axes[0, col], all_axes[1, col], all_axes[2, col]
        plot_sweep(ax_r, ax_c, ax_s, alpha_results)
        side_str = f"{sweep_side} $\\alpha$"
        fixed_str = (f"human $\\alpha$={args.default_alpha}" if target == "filter"
                     else f"filter $\\alpha$={args.default_alpha}")
        ax_r.set_title(f"Sweep {side_str}  ({fixed_str}, $\\beta$={args.default_beta})",
                       fontsize=10)
        col += 1

    if do_beta:
        print(f"\nBeta sweep ({sweep_side})...\n")

        if target == "filter":
            def make_m(b):
                return {**fixed_method, "kf_lingap_beta": b}
            def make_h(b):
                return fixed_human
        else:
            def make_m(b):
                return fixed_method
            def make_h(b):
                return {**fixed_human, "human_lingap_beta": b}

        beta_results = _run_sweep(
            args, "\\beta", args.betas, args.default_beta,
            make_m, make_h)

        ax_r, ax_c, ax_s = all_axes[0, col], all_axes[1, col], all_axes[2, col]
        plot_sweep(ax_r, ax_c, ax_s, beta_results, cmap_name="plasma")
        side_str = f"{sweep_side} $\\beta$"
        fixed_str = (f"human $\\beta$={args.default_beta}" if target == "filter"
                     else f"filter $\\beta$={args.default_beta}")
        ax_r.set_title(f"Sweep {side_str}  ($\\alpha$={args.default_alpha}, {fixed_str})",
                       fontsize=10)

    # Format all axes
    for j in range(n_panels):
        all_axes[0, j].set_ylabel("RMSE (m/s)")
        all_axes[0, j].legend(fontsize=7, loc="upper left")
        all_axes[0, j].grid(alpha=0.2)

        all_axes[1, j].axhline(95, color="green", linestyle=":", linewidth=1, alpha=0.5)
        all_axes[1, j].set_ylabel("Coverage (%)")
        all_axes[1, j].set_ylim(0, 105)
        all_axes[1, j].legend(fontsize=7, loc="lower left")
        all_axes[1, j].grid(alpha=0.2)

        all_axes[2, j].axhline(1, color="red", linestyle=":", linewidth=1, alpha=0.5)
        all_axes[2, j].set_ylabel("RMSE / $\\sqrt{P}$")
        all_axes[2, j].set_xlabel("Horizon (s)")
        all_axes[2, j].legend(fontsize=7, loc="upper left")
        all_axes[2, j].grid(alpha=0.2)

    # Add matched spec annotation
    fig.text(0.5, 0.96,
             f"Matched: $\\alpha$={args.default_alpha}, $\\beta$={args.default_beta}, "
             f"Q={args.Q}, R={args.R}   |   Sweeping: {sweep_side}",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(f"Lingapkalman ablation  ({args.experiment})",
                 fontsize=13, fontweight="bold", y=0.99)

    suffix = "filter" if target == "filter" else "human"
    out_path = args.out or os.path.join(FIG_DIR, f"lingap_ablation_{suffix}.pdf")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
