"""
Combined replay of a CommonRoad car-follow episode: scene + beliefs + Riccati.

Three panels stacked vertically:
  1. Scene:   Bird's-eye view of ego + other vehicles.
  2. Beliefs: True kappa vs inferred kappa with +/- 2 sigma band.
  3. Riccati: sqrt(P) convergence to algebraic Riccati steady-state.

Usage:
    python experiments/commonroad/replay_combined.py results/run_dir/
    python experiments/commonroad/replay_combined.py results/run_dir/ -o combined.gif
    python experiments/commonroad/replay_combined.py results/run_dir/ --show
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D

VEH_LENGTH = 4.5
VEH_WIDTH = 1.8
LANE_WIDTH = 3.5
N_LANES = 3

OBSTACLE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
EGO_COLOR = "#27ae60"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_run(run_dir: str) -> tuple:
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


def build_title(meta: dict) -> str:
    exp = meta.get("experiment", "")
    inf = meta.get("inference", "?")
    intv = meta.get("intervention", "?")
    human = meta.get("human", "?")
    return f"{exp}  ({human} / {inf} / {intv})"


def draw_vehicle(ax, cx, cy, orient, color, alpha=0.7, lw=1.5, label=None):
    rect = mpatches.FancyBboxPatch(
        (-VEH_LENGTH / 2, -VEH_WIDTH / 2), VEH_LENGTH, VEH_WIDTH,
        boxstyle="round,pad=0.15",
        facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw)
    tr = Affine2D().rotate(orient).translate(cx, cy) + ax.transData
    rect.set_transform(tr)
    ax.add_patch(rect)
    if label:
        ax.annotate(label, xy=(cx, cy), fontsize=7, fontweight="bold",
                    ha="center", va="center", color="white", zorder=20)


def riccati_steady_state(Q: float, R: float, C: float = 1.0) -> float:
    """Algebraic Riccati steady-state for scalar Kalman.

    C²P² + QC²P - QR = 0
    P_ss = (-QC² + sqrt((QC²)² + 4C²QR)) / (2C²)
    """
    a = C ** 2
    disc = (Q * a) ** 2 + 4 * a * Q * R
    return (-Q * a + np.sqrt(disc)) / (2 * a)


def parse_args():
    p = argparse.ArgumentParser(
        description="Combined replay: scene + beliefs + Riccati")
    p.add_argument("run_dir",
                   help="Run directory (or name under results/)")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Output path (default: combined_replay.gif in run dir)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--view-half-x", type=float, default=60.0)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    episode, meta, run_dir = load_run(args.run_dir)
    title = build_title(meta)
    dt = meta.get("dt", 0.1)

    ego_positions = episode["ego_positions"]
    vehicle_positions = {k: v for k, v in episode["vehicle_positions"].items()}
    human_vel_err = episode.get("human_vel_err", [])
    inferred_mean = episode.get("inferred_mean", [])
    kf_kappa_list = episode.get("kf_kappa", [])
    kf_P_list = episode.get("kf_P", [])
    norm_innov_list = episode.get("norm_innovation", [])
    lead_speed_list = episode.get("lead_speed", [])
    distance_list = episode.get("distance", [])
    intervened_list = episode.get("intervened", [])
    n_frames = len(episode["steps"])

    true_vel_errors = episode.get("true_vel_errors", {})
    initial_kappa = list(true_vel_errors.values())[0] if true_vel_errors else None

    # Kalman params for Riccati
    Q = meta.get("kf_Q", 0.001)
    R = meta.get("kf_R", 1.0)
    inf_type = meta.get("inference", "")
    C = 1.0 if inf_type == "idkalman" else 15.0
    P_ss = riccati_steady_state(Q, R, C)
    has_kalman = any(p is not None for p in kf_P_list)

    print(f"  Run:    {os.path.basename(run_dir)}")
    print(f"  Title:  {title}")
    print(f"  Frames: {n_frames}  ({n_frames * dt:.1f} s)")
    if has_kalman:
        print(f"  Riccati P_ss = {P_ss:.6f},  sqrt(P_ss) = {np.sqrt(P_ss):.4f}")

    # -- Helper: get lead speed at step t --
    def _lead_speed(t):
        v = lead_speed_list[t] if t < len(lead_speed_list) else None
        return v if v is not None else 15.0

    # -- Pre-compute full traces in velocity error (m/s) space --
    # true velocity error = kappa * v_lead,  inferred velocity error = kf_kappa * v_lead
    # For idkalman, kf_P is already in velocity error space.
    # For boltzmann_kalman, kf_P is in kappa space so scale by v_lead².
    true_steps, true_eps = [], []
    inf_steps, inf_eps = [], []
    for t in range(n_frames):
        v = _lead_speed(t)
        hve = human_vel_err[t] if t < len(human_vel_err) else None
        if hve is not None:
            true_steps.append(t)
            true_eps.append(hve * v)  # kappa → velocity error
        est = kf_kappa_list[t] if t < len(kf_kappa_list) and kf_kappa_list[t] is not None else None
        if est is None and t < len(inferred_mean):
            est = inferred_mean[t]
        if est is not None:
            inf_steps.append(t)
            inf_eps.append(est * v)  # kappa → velocity error

    # Pre-compute sqrt(P) in velocity error space, and uncertainty band in velocity error
    P_steps, P_sqrt_vals = [], []
    band_steps, band_upper, band_lower = [], [], []
    for t in range(n_frames):
        kf_p = kf_P_list[t] if t < len(kf_P_list) else None
        if kf_p is not None:
            v = _lead_speed(t)
            # For idkalman, P is in velocity error space already.
            # For boltzmann_kalman, P is in kappa space → convert.
            if inf_type == "idkalman":
                P_eps = kf_p
            else:
                P_eps = kf_p * (v ** 2)
            P_steps.append(t)
            P_sqrt_vals.append(np.sqrt(P_eps))
            # Uncertainty band in velocity error space
            est_kappa = kf_kappa_list[t] if t < len(kf_kappa_list) and kf_kappa_list[t] is not None else None
            if est_kappa is not None:
                est_eps = est_kappa * v
                std_eps = np.sqrt(P_eps)
                band_steps.append(t)
                band_upper.append(est_eps + 2 * std_eps)
                band_lower.append(est_eps - 2 * std_eps)

    # Pre-compute normalized innovation trace
    ni_steps, ni_vals = [], []
    for t in range(n_frames):
        ni = norm_innov_list[t] if t < len(norm_innov_list) else None
        if ni is not None:
            ni_steps.append(t)
            ni_vals.append(ni)

    # -- Fixed axis limits for velocity error --
    initial_eps = initial_kappa * 15.0 if initial_kappa is not None else 0.0
    y_eps_top = abs(initial_eps) + 2.0 if initial_eps != 0 else 3.0
    y_eps_bot = -y_eps_top
    if true_eps:
        y_eps_top = max(y_eps_top, max(true_eps) + 0.5)
        y_eps_bot = min(y_eps_bot, min(true_eps) - 0.5)
    if band_upper:
        y_eps_top = max(y_eps_top, max(band_upper) + 0.2)
    if band_lower:
        y_eps_bot = min(y_eps_bot, min(band_lower) - 0.2)

    # -- Figure layout --
    fig = plt.figure(figsize=(14, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1],
                          hspace=0.35,
                          left=0.05, right=0.97, top=0.97, bottom=0.03)
    ax_scene = fig.add_subplot(gs[0])
    ax_belief = fig.add_subplot(gs[1])
    ax_ni = fig.add_subplot(gs[2], sharex=ax_belief)
    ax_riccati = fig.add_subplot(gs[3], sharex=ax_belief)

    # -- Static belief elements (velocity error = perceived - true, in m/s) --
    ax_belief.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    line_true, = ax_belief.plot([], [], color="steelblue", linewidth=1.5,
                                label="true $\\delta v$")
    line_inf, = ax_belief.plot([], [], color="orange", linewidth=1.5,
                               label="inferred $\\hat{\\delta v}$")
    fill_artist = [None]
    cursor_belief = ax_belief.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax_belief.set_xlim(0, n_frames - 1)
    ax_belief.set_ylim(y_eps_bot, y_eps_top)
    ax_belief.set_ylabel("$\\delta v$ (m/s)")
    ax_belief.legend(fontsize=8, loc="upper right")
    ax_belief.set_title("Perceived velocity error  ($v_{perceived} - v_{true}$)",
                        fontsize=10)

    # -- Normalized innovation running statistics + autocorrelation --
    # Reference lines
    ax_ni.axhline(0, color="steelblue", linewidth=1.0, linestyle=":",
                  alpha=0.6, label="expected $\\hat{\\mu}$ = 0")
    ax_ni.axhline(1, color="#e74c3c", linewidth=1.0, linestyle=":",
                  alpha=0.6, label="expected $\\hat{\\sigma}^2$ = 1")
    ax_ni.axhline(0, color="#2ecc71", linewidth=1.0, linestyle=":",
                  alpha=0.6)  # autocorr reference (same as mean ref)
    line_ni_mean, = ax_ni.plot([], [], color="steelblue", linewidth=2.0,
                               label="running $\\hat{\\mu}$")
    line_ni_var, = ax_ni.plot([], [], color="#e74c3c", linewidth=2.0,
                              label="running $\\hat{\\sigma}^2$")
    line_ni_acf, = ax_ni.plot([], [], color="#2ecc71", linewidth=2.0,
                              label="running autocorr(1)")
    cursor_ni = ax_ni.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax_ni.set_xlim(0, n_frames - 1)
    ax_ni.set_ylim(-1.0, 3.0)
    ax_ni.set_ylabel("value")
    ax_ni.legend(fontsize=7, loc="upper right", ncol=2)
    ax_ni.set_title("Normalized innovation statistics  "
                    "($\\hat{\\mu} \\to 0$,  $\\hat{\\sigma}^2 \\to 1$,  "
                    "autocorr $\\to 0$ if well-calibrated)",
                    fontsize=10)

    # -- Static Riccati elements --
    if has_kalman:
        ax_riccati.axhline(np.sqrt(P_ss), color="red", linestyle="--",
                           linewidth=1.5,
                           label=f"$\\sqrt{{P_{{ss}}}}$ = {np.sqrt(P_ss):.4f}")
    line_P, = ax_riccati.plot([], [], color="steelblue", linewidth=1.5,
                              label="$\\sqrt{P_{\\delta v}}$")
    cursor_riccati = ax_riccati.axvline(0, color="black", linewidth=0.8, alpha=0.4)
    ax_riccati.set_xlim(0, n_frames - 1)
    if P_sqrt_vals:
        ax_riccati.set_ylim(0, max(max(P_sqrt_vals), np.sqrt(P_ss)) * 1.15)
    else:
        ax_riccati.set_ylim(0, 1)
    ax_riccati.set_xlabel("step")
    ax_riccati.set_ylabel("$\\sqrt{P_{\\delta v}}$ (m/s)")
    ax_riccati.set_title(f"Kalman covariance convergence   (Q={Q}, R={R}, C={C})",
                         fontsize=10)
    ax_riccati.legend(fontsize=8, loc="upper right")

    # -- Scene helpers --
    trail_len = 30
    view_half_x = args.view_half_x

    def draw_scene(t):
        ax_scene.clear()
        for i in range(N_LANES + 1):
            y = i * LANE_WIDTH
            is_edge = (i == 0 or i == N_LANES)
            ax_scene.axhline(y, color="gray", linewidth=2 if is_edge else 1,
                             linestyle="-" if is_edge else "--", zorder=1)
        for i in range(N_LANES):
            ax_scene.axhspan(i * LANE_WIDTH, (i + 1) * LANE_WIDTH,
                             color="#f5f5f5", zorder=0)

        # Non-ego
        sorted_vids = sorted(vehicle_positions.keys())
        for idx, vid in enumerate(sorted_vids):
            traj = vehicle_positions[vid]
            ti = min(t, len(traj) - 1)
            vx, vy, vh, vv = traj[ti]
            color = OBSTACLE_COLORS[idx % len(OBSTACLE_COLORS)]
            draw_vehicle(ax_scene, vx, vy, vh, color)
            ax_scene.annotate(f"{vid}", xy=(vx, vy), fontsize=7,
                              fontweight="bold", ha="center", va="center",
                              color="white", zorder=20)
            ax_scene.annotate(f"{vv:.1f} m/s", xy=(vx, vy + VEH_WIDTH),
                              fontsize=6, ha="center", color=color, zorder=15)
            ts = max(0, t - trail_len)
            trail = traj[ts:ti + 1]
            if len(trail) > 1:
                ax_scene.plot([s[0] for s in trail], [s[1] for s in trail],
                              "-", color=color, alpha=0.25, linewidth=2, zorder=1)

        # Ego
        ti = min(t, len(ego_positions) - 1)
        ex, ey, eh, ev = ego_positions[ti]
        draw_vehicle(ax_scene, ex, ey, eh, EGO_COLOR, alpha=0.85, label="EGO")
        ax_scene.annotate(f"{ev:.1f} m/s", xy=(ex, ey + VEH_WIDTH),
                          fontsize=6, ha="center", color=EGO_COLOR, zorder=15)
        ts = max(0, t - trail_len)
        trail = ego_positions[ts:ti + 1]
        if len(trail) > 1:
            ax_scene.plot([s[0] for s in trail], [s[1] for s in trail],
                          "-", color=EGO_COLOR, alpha=0.4, linewidth=2, zorder=1)

        ax_scene.set_xlim(ex - view_half_x, ex + view_half_x)
        ax_scene.set_ylim(-2.0, N_LANES * LANE_WIDTH + 2.0)
        ax_scene.set_aspect("equal")
        ax_scene.set_xticks([])
        ax_scene.set_yticks([])

        # Title
        time_s = t * dt
        info_str = f"t = {time_s:.1f} s   v = {ev:.1f} m/s"
        d = distance_list[t] if t < len(distance_list) else None
        if d is not None:
            info_str += f"   d = {d:.1f} m"
        if t < len(intervened_list) and intervened_list[t]:
            info_str += "   [INTERVENED]"
        ax_scene.set_title(f"{title}    {info_str}", fontsize=10)

    def update(t):
        # Scene
        draw_scene(t)

        # Beliefs (velocity error, m/s) — reveal up to t
        idx_t = [i for i, s in enumerate(true_steps) if s <= t]
        if idx_t:
            line_true.set_data([true_steps[i] for i in idx_t],
                               [true_eps[i] for i in idx_t])
        idx_i = [i for i, s in enumerate(inf_steps) if s <= t]
        if idx_i:
            line_inf.set_data([inf_steps[i] for i in idx_i],
                              [inf_eps[i] for i in idx_i])

        if fill_artist[0] is not None:
            fill_artist[0].remove()
            fill_artist[0] = None
        band_idx = [i for i, s in enumerate(band_steps) if s <= t]
        if band_idx:
            fill_artist[0] = ax_belief.fill_between(
                [band_steps[i] for i in band_idx],
                [band_lower[i] for i in band_idx],
                [band_upper[i] for i in band_idx],
                color="orange", alpha=0.15)
        cursor_belief.set_xdata([t, t])

        # Normalized innovation — running mean, variance, autocorrelation
        idx_ni = [i for i, s in enumerate(ni_steps) if s <= t]
        if idx_ni:
            s_ni = [ni_steps[i] for i in idx_ni]
            v_ni = [ni_vals[i] for i in idx_ni]

            arr = np.array(v_ni)
            n_arr = np.arange(1, len(arr) + 1)
            cum_mean = np.cumsum(arr) / n_arr
            cum_var = np.cumsum(arr ** 2) / n_arr - cum_mean ** 2
            cum_var = np.maximum(cum_var, 0.0)

            # Running lag-1 autocorrelation
            cum_acf = np.zeros(len(arr))
            for j in range(1, len(arr)):
                # Σ(ν_i * ν_{i+1}) / Σ(ν_i²) up to index j
                num = np.sum(arr[:j] * arr[1:j + 1])
                den = np.sum(arr[:j + 1] ** 2)
                cum_acf[j] = num / den if den > 1e-12 else 0.0

            line_ni_mean.set_data(s_ni, cum_mean)
            line_ni_var.set_data(s_ni, cum_var)
            line_ni_acf.set_data(s_ni, cum_acf)

        cursor_ni.set_xdata([t, t])

        # Riccati — reveal up to t
        idx_p = [i for i, s in enumerate(P_steps) if s <= t]
        if idx_p:
            line_P.set_data([P_steps[i] for i in idx_p],
                            [P_sqrt_vals[i] for i in idx_p])
        cursor_riccati.set_xdata([t, t])

    if args.show:
        anim = animation.FuncAnimation(
            fig, update, frames=n_frames,
            interval=int(1000 / args.fps), repeat=True)
        plt.show(block=True)
    else:
        out_path = args.out
        if out_path is None:
            out_path = os.path.join(run_dir, "combined_replay.gif")

        print(f"  Rendering {n_frames} frames ...")
        anim = animation.FuncAnimation(
            fig, update, frames=n_frames,
            interval=int(1000 / args.fps), repeat=False)

        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".mp4":
            writer = animation.FFMpegWriter(fps=args.fps)
        else:
            writer = animation.PillowWriter(fps=args.fps)

        anim.save(out_path, writer=writer, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
