"""
Validate Kalman filter rollouts against actual trajectories.

At each timestep k, rolls the process model forward N steps (no observations)
and compares the predicted v_hat trajectory against what actually happened.

Produces:
  1. GIF: rollout fan (v_hat ± 2*sqrt(P)) vs actual, updated each step
  2. Static figure: coverage rate + normalized rollout error vs horizon step

Usage:
    python experiments/commonroad/validate_rollout.py results/run_dir/
    python experiments/commonroad/validate_rollout.py results/run_dir/ -o rollout.gif
    python experiments/commonroad/validate_rollout.py results/run_dir/ --horizon 30
    python experiments/commonroad/validate_rollout.py results/run_dir/ --show
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_run(run_dir):
    if not os.path.isabs(run_dir) and not os.path.isdir(run_dir):
        run_dir = os.path.join(RESULTS_DIR, run_dir)
    with open(os.path.join(run_dir, "episode.json")) as f:
        episode = json.load(f)
    meta_path = os.path.join(run_dir, "metadata.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return episode, meta, run_dir


def rollout_revert(vhat_0, P_0, alpha, Q, lead_speeds, a_leads, dt, N,
                   beta=0.0, gaps=None):
    """Roll the process model forward N steps with no observations.

    Returns:
        vhats: (N+1,) predicted v_hat trajectory (including initial)
        Ps:    (N+1,) predicted covariance trajectory
    """
    vhats = np.zeros(N + 1)
    Ps = np.zeros(N + 1)
    vhats[0] = vhat_0
    Ps[0] = P_0

    for j in range(N):
        v_lead = lead_speeds[j] if j < len(lead_speeds) else lead_speeds[-1]
        a_lead = a_leads[j] if j < len(a_leads) else 0.0
        B = (1.0 - alpha) * v_lead + a_lead * dt
        if beta > 0 and gaps is not None:
            gap = gaps[j] if j < len(gaps) else gaps[-1]
            B += beta * max(gap, 0.0)
        vhats[j + 1] = alpha * vhats[j] + B
        Ps[j + 1] = alpha ** 2 * Ps[j] + Q

    return vhats, Ps


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate Kalman filter rollouts vs actual trajectory")
    p.add_argument("run_dir",
                   help="Run directory (or name under results/)")
    p.add_argument("--horizon", "-n", type=int, default=50,
                   help="Rollout horizon in steps (default: 50)")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Output GIF path (default: rollout_validation.gif in run dir)")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    episode, meta, run_dir = load_run(args.run_dir)

    dt = meta.get("dt", 0.1)
    inf_type = meta.get("inference", "")
    if inf_type in ("revertkalman", "lingapkalman"):
        alpha = meta.get("kf_revert_alpha", 0.9)
    elif inf_type == "iidkalman":
        alpha = 0.0
    else:
        alpha = 1.0  # walkkalman, boltzmann_kalman
    Q = meta.get("kf_Q", 0.1)
    beta = meta.get("kf_lingap_beta", 0.0)
    if inf_type != "lingapkalman":
        beta = 0.0
    N = args.horizon

    kf_vhat = episode.get("kf_vhat", [])
    kf_P = episode.get("kf_P", [])
    lead_speed = episode.get("lead_speed", [])
    distance = episode.get("distance", [])
    human_vel_err = episode.get("human_vel_err", [])
    kf_kappa = episode.get("kf_kappa", [])
    n_steps = len(episode["steps"])

    # Build actual v_hat trajectory (true perceived velocity)
    actual_vhat = []
    for t in range(n_steps):
        v = lead_speed[t] if t < len(lead_speed) and lead_speed[t] is not None else None
        hve = human_vel_err[t] if t < len(human_vel_err) else None
        if v is not None and hve is not None:
            actual_vhat.append(v * (1.0 + hve))
        else:
            actual_vhat.append(None)

    # Build estimated v_hat trajectory
    est_vhat = []
    for t in range(n_steps):
        v = lead_speed[t] if t < len(lead_speed) and lead_speed[t] is not None else None
        kk = kf_kappa[t] if t < len(kf_kappa) and kf_kappa[t] is not None else None
        if v is not None and kk is not None:
            est_vhat.append(v * (1.0 + kk))
        else:
            est_vhat.append(None)

    # Estimate lead acceleration from consecutive speeds
    a_leads = [0.0]
    for t in range(1, n_steps):
        v0 = lead_speed[t - 1] if t - 1 < len(lead_speed) else None
        v1 = lead_speed[t] if t < len(lead_speed) else None
        if v0 is not None and v1 is not None:
            a_leads.append((v1 - v0) / dt)
        else:
            a_leads.append(0.0)

    # Find valid start points (where we have filter state)
    valid_starts = []
    for t in range(n_steps):
        vh = kf_vhat[t] if t < len(kf_vhat) else None
        p = kf_P[t] if t < len(kf_P) else None
        # Fallback: reconstruct vhat from kf_kappa + lead_speed
        if vh is None and kf_kappa[t] is not None and lead_speed[t] is not None:
            vh = lead_speed[t] * (1.0 + kf_kappa[t])
        if vh is not None and p is not None and t + N < n_steps:
            valid_starts.append(t)

    if not valid_starts:
        print("Not enough data for rollout validation.")
        sys.exit(1)

    print(f"  Run:     {os.path.basename(run_dir)}")
    print(f"  Filter:  {inf_type}, alpha={alpha}, Q={Q}, beta={beta}")
    print(f"  Horizon: {N} steps ({N*dt:.1f}s)")
    print(f"  Valid rollout starts: {len(valid_starts)}")

    # ---------------------------------------------------------------
    #  Compute rollouts at every valid start
    # ---------------------------------------------------------------
    # Per-horizon-step accumulators
    sq_errors = [[] for _ in range(N + 1)]
    norm_errors = [[] for _ in range(N + 1)]
    in_band = [[] for _ in range(N + 1)]

    all_rollouts = {}  # t → (vhats, Ps)

    for t in valid_starts:
        vh = kf_vhat[t] if t < len(kf_vhat) and kf_vhat[t] is not None else None
        if vh is None:
            vh = lead_speed[t] * (1.0 + kf_kappa[t])
        p = kf_P[t]

        future_ls = [lead_speed[t + j] if t + j < len(lead_speed) and lead_speed[t + j] is not None
                     else 15.0 for j in range(N)]
        future_al = [a_leads[t + j] if t + j < len(a_leads) else 0.0 for j in range(N)]
        future_gaps = [distance[t + j] if t + j < len(distance) and distance[t + j] is not None
                       else 20.0 for j in range(N)]

        pred_vhat, pred_P = rollout_revert(
            vh, p, alpha, Q, future_ls, future_al, dt, N,
            beta=beta, gaps=future_gaps)

        all_rollouts[t] = (pred_vhat, pred_P)

        for j in range(N + 1):
            act = actual_vhat[t + j] if t + j < len(actual_vhat) else None
            if act is not None:
                err = pred_vhat[j] - act
                sq_errors[j].append(err ** 2)
                if pred_P[j] > 1e-12:
                    norm_errors[j].append(err ** 2 / pred_P[j])
                    in_band[j].append(1.0 if abs(err) <= 2.0 * np.sqrt(pred_P[j]) else 0.0)

    # ---------------------------------------------------------------
    #  Static metrics figure
    # ---------------------------------------------------------------
    horizons = np.arange(N + 1)
    mse = np.array([np.mean(s) if s else np.nan for s in sq_errors])
    rmse = np.sqrt(mse)
    mean_nees = np.array([np.mean(s) if s else np.nan for s in norm_errors])
    coverage = np.array([np.mean(s) if s else np.nan for s in in_band])

    fig_static, (ax_rmse, ax_nees, ax_cov) = plt.subplots(
        3, 1, figsize=(10, 9), sharex=True)
    fig_static.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.08,
                               hspace=0.25)

    ax_rmse.plot(horizons * dt, rmse, "steelblue", linewidth=2)
    ax_rmse.set_ylabel("RMSE (m/s)")
    ax_rmse.set_title("Rollout RMSE vs horizon", fontsize=11)
    ax_rmse.grid(alpha=0.2)

    ax_nees.plot(horizons * dt, mean_nees, "#e74c3c", linewidth=2)
    ax_nees.axhline(1.0, color="red", linestyle=":", linewidth=1, alpha=0.6,
                    label="expected = 1")
    ax_nees.set_ylabel("Normalized error")
    ax_nees.set_title("Normalized rollout error  (expect $\\approx 1$ if P calibrated)",
                      fontsize=11)
    ax_nees.legend(fontsize=8)
    ax_nees.grid(alpha=0.2)

    ax_cov.plot(horizons * dt, coverage * 100, "#2ecc71", linewidth=2)
    ax_cov.axhline(95, color="green", linestyle=":", linewidth=1, alpha=0.6,
                   label="expected = 95%")
    ax_cov.set_ylim(0, 105)
    ax_cov.set_xlabel(f"Horizon (s)")
    ax_cov.set_ylabel("Coverage (%)")
    ax_cov.set_title("Coverage rate  ($\\pm 2\\sigma$ band, expect $\\approx 95\\%$)",
                     fontsize=11)
    ax_cov.legend(fontsize=8)
    ax_cov.grid(alpha=0.2)

    fig_static.suptitle(f"Rollout validation  ({inf_type}, $\\alpha$={alpha}, Q={Q})",
                        fontsize=13, fontweight="bold")

    metrics_path = os.path.join(run_dir, "rollout_metrics.pdf")
    fig_static.savefig(metrics_path, dpi=args.dpi, bbox_inches="tight")
    print(f"  Metrics saved -> {metrics_path}")
    plt.close(fig_static)

    # Print summary
    print(f"\n  {'Horizon':>8s}  {'RMSE':>8s}  {'Norm err':>9s}  {'Coverage':>9s}")
    for j in [0, 5, 10, 20, 30, 50]:
        if j <= N:
            print(f"  {j*dt:7.1f}s  {rmse[j]:8.4f}  {mean_nees[j]:9.4f}  "
                  f"{coverage[j]*100:8.1f}%")

    # ---------------------------------------------------------------
    #  Animated GIF: rollout fan at each timestep
    # ---------------------------------------------------------------
    fig_anim, (ax_traj, ax_fan) = plt.subplots(
        2, 1, figsize=(12, 7),
        gridspec_kw={"height_ratios": [1, 1]})
    fig_anim.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.08,
                             hspace=0.30)

    # Top: full episode v_hat trajectory with sliding rollout
    # Precompute full actual and estimated traces
    all_t = list(range(n_steps))
    act_s = [t for t in all_t if actual_vhat[t] is not None]
    act_v = [actual_vhat[t] for t in act_s]
    est_s = [t for t in all_t if est_vhat[t] is not None]
    est_v = [est_vhat[t] for t in est_s]

    ax_traj.plot(act_s, act_v, color="steelblue", linewidth=0.8, alpha=0.5,
                 label="actual $\\hat{v}$")
    ax_traj.plot(est_s, est_v, color="orange", linewidth=0.8, alpha=0.5,
                 label="estimated $\\hat{v}$")
    line_rollout, = ax_traj.plot([], [], color="#e74c3c", linewidth=2,
                                  label="rollout")
    fill_rollout = [None]
    cursor_traj = ax_traj.axvline(0, color="black", linewidth=0.8, alpha=0.4)

    all_vals = [v for v in act_v + est_v if v is not None]
    if all_vals:
        ax_traj.set_ylim(min(all_vals) - 2, max(all_vals) + 2)
    ax_traj.set_xlim(0, n_steps - 1)
    ax_traj.set_ylabel("$\\hat{v}$ (m/s)")
    ax_traj.set_title("Perceived velocity: actual vs rollout prediction", fontsize=11)
    ax_traj.legend(fontsize=8, loc="upper right")
    ax_traj.grid(alpha=0.2)

    # Bottom: rollout error fan for current timestep
    line_err, = ax_fan.plot([], [], color="#e74c3c", linewidth=1.5,
                            label="rollout error")
    fill_band = [None]
    ax_fan.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax_fan.set_xlim(0, N * dt)
    ax_fan.set_xlabel("Horizon (s)")
    ax_fan.set_ylabel("Error (m/s)")
    ax_fan.set_title("Rollout error at current step  (band = $\\pm 2\\sqrt{P}$)",
                     fontsize=11)
    ax_fan.legend(fontsize=8, loc="upper right")
    ax_fan.grid(alpha=0.2)

    # Only animate at valid start points (skip steps with no rollout)
    anim_frames = valid_starts[::2]  # every other valid start for speed
    if not anim_frames:
        anim_frames = valid_starts

    def update(frame_idx):
        t = anim_frames[frame_idx]
        pred_vhat, pred_P = all_rollouts[t]

        # Top: draw rollout from t forward
        roll_steps = [t + j for j in range(N + 1)]
        line_rollout.set_data(roll_steps, pred_vhat)

        if fill_rollout[0] is not None:
            fill_rollout[0].remove()
            fill_rollout[0] = None
        upper = pred_vhat + 2 * np.sqrt(pred_P)
        lower = pred_vhat - 2 * np.sqrt(pred_P)
        fill_rollout[0] = ax_traj.fill_between(
            roll_steps, lower, upper, color="#e74c3c", alpha=0.15)
        cursor_traj.set_xdata([t, t])

        time_s = t * dt
        ax_traj.set_title(f"Perceived velocity: actual vs rollout  (t={time_s:.1f}s)",
                          fontsize=11)

        # Bottom: error at each horizon step
        horizons_s = np.arange(N + 1) * dt
        errors = []
        for j in range(N + 1):
            act = actual_vhat[t + j] if t + j < len(actual_vhat) else None
            errors.append(pred_vhat[j] - act if act is not None else 0.0)
        errors = np.array(errors)

        line_err.set_data(horizons_s, errors)

        if fill_band[0] is not None:
            fill_band[0].remove()
            fill_band[0] = None
        band = 2 * np.sqrt(pred_P)
        fill_band[0] = ax_fan.fill_between(
            horizons_s, -band, band, color="#e74c3c", alpha=0.15)

        err_max = max(np.max(np.abs(errors)), np.max(band)) * 1.2
        ax_fan.set_ylim(-err_max, err_max)

    if args.show:
        anim = animation.FuncAnimation(
            fig_anim, update, frames=len(anim_frames),
            interval=int(1000 / args.fps), repeat=True)
        plt.show(block=True)
    else:
        out_path = args.out or os.path.join(run_dir, "rollout_validation.gif")
        print(f"\n  Rendering {len(anim_frames)} frames ...")
        anim = animation.FuncAnimation(
            fig_anim, update, frames=len(anim_frames),
            interval=int(1000 / args.fps), repeat=False)
        ext = os.path.splitext(out_path)[1].lower()
        writer = animation.FFMpegWriter(fps=args.fps) if ext == ".mp4" else animation.PillowWriter(fps=args.fps)
        anim.save(out_path, writer=writer, dpi=args.dpi)
        plt.close(fig_anim)
        print(f"  GIF saved -> {out_path}")


if __name__ == "__main__":
    main()
