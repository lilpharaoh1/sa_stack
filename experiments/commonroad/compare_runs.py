"""
Compare multiple runs: one scene replay on top, selected metrics overlaid below.

The scene is taken from the first run (assumed representative).
Metric panels overlay all runs for comparison with auto-scaled y-axes.

Usage:
    # Compare two runs, show NEES
    python experiments/commonroad/compare_runs.py run1/ run2/ --metrics nees

    # Multiple metrics
    python experiments/commonroad/compare_runs.py run1/ run2/ --metrics nees dv riccati

    # Custom labels
    python experiments/commonroad/compare_runs.py run1/ run2/ \
        --labels "matched" "mismatched" --metrics nees -o comparison.gif

Available metrics: dv, ni_stats, nees, riccati
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
RUN_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

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


def run_label(meta, fallback):
    if not meta:
        return fallback
    inf = meta.get("inference", "?")
    intv = meta.get("intervention", "?")
    human = meta.get("human", "?")
    return f"{human}/{inf}/{intv}"


def riccati_steady_state(Q, R, C=1.0, A=1.0):
    a2c2 = (A * C) ** 2
    b = Q * C ** 2 + R * (1.0 - A ** 2)
    c_coeff = -R * Q
    disc = b ** 2 - 4 * a2c2 * c_coeff
    return (-b + np.sqrt(disc)) / (2 * a2c2)


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


def precompute(r):
    """Pre-compute all metric traces for one run."""
    ep = r["ep"]
    meta = r["meta"]
    inf_type = meta.get("inference", "")
    is_additive = inf_type in ("iidkalman", "walkkalman", "revertkalman", "lingapkalman")

    hve = ep.get("human_vel_err", [])
    kk = ep.get("kf_kappa", [])
    kf_P = ep.get("kf_P", [])
    ls = ep.get("lead_speed", [])
    ni = ep.get("norm_innovation", [])
    n = len(ep.get("steps", []))

    def _lv(t):
        v = ls[t] if t < len(ls) else None
        return v if v is not None else 15.0

    # dv
    true_s, true_v, inf_s, inf_v = [], [], [], []
    for t in range(n):
        v = _lv(t)
        h = hve[t] if t < len(hve) else None
        e = kk[t] if t < len(kk) and kk[t] is not None else None
        if h is not None:
            true_s.append(t); true_v.append(h * v)
        if e is not None:
            inf_s.append(t); inf_v.append(e * v)
    r["dv"] = {"true_s": true_s, "true_v": true_v,
               "inf_s": inf_s, "inf_v": inf_v}

    # ni
    ni_s = [t for t, v in enumerate(ni) if v is not None]
    ni_v = [v for v in ni if v is not None]
    r["ni"] = {"s": ni_s, "v": ni_v}

    # nees
    nees_s, nees_v = [], []
    for t in range(n):
        kf_p = kf_P[t] if t < len(kf_P) else None
        h = hve[t] if t < len(hve) else None
        e = kk[t] if t < len(kk) and kk[t] is not None else None
        if kf_p and h is not None and e is not None and kf_p > 1e-12:
            v = _lv(t)
            P_e = kf_p if is_additive else kf_p * v ** 2
            nees_s.append(t)
            nees_v.append((h * v - e * v) ** 2 / P_e)
    r["nees"] = {"s": nees_s, "v": nees_v}

    # riccati
    P_s = [t for t, p in enumerate(kf_P) if p is not None]
    P_v = [np.sqrt(p) for p in kf_P if p is not None]
    r["riccati"] = {"s": P_s, "v": P_v}

    Q = meta.get("kf_Q", 0.1)
    R_kf = meta.get("kf_R", 1.0)
    A_proc = meta.get("kf_revert_alpha", 1.0) if inf_type in ("revertkalman", "lingapkalman") else 1.0
    r["P_ss"] = riccati_steady_state(Q, R_kf, C=1.0, A=A_proc)


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare runs: one scene + overlaid metrics")
    p.add_argument("dirs", nargs="+",
                   help="Run directories (or names under results/)")
    p.add_argument("--metrics", nargs="+", default=["nees"],
                   choices=["dv", "ni_stats", "nees", "riccati"])
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--out", "-o", type=str, default=None)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--view-half-x", type=float, default=60.0)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    runs = []
    for i, d in enumerate(args.dirs):
        try:
            ep, meta, rd = load_run(d)
            label = args.labels[i] if args.labels and i < len(args.labels) else \
                    run_label(meta, os.path.basename(d.rstrip("/")))
            runs.append({"ep": ep, "meta": meta, "dir": rd, "label": label,
                         "color": RUN_COLORS[i % len(RUN_COLORS)]})
        except Exception as e:
            print(f"Warning: skipping {d}: {e}")

    if not runs:
        print("No valid runs.")
        sys.exit(1)

    for r in runs:
        precompute(r)

    n_metrics = len(args.metrics)
    n_frames = max(len(r["ep"]["steps"]) for r in runs)

    print(f"  Runs: {len(runs)},  Metrics: {args.metrics},  Frames: {n_frames}")

    # -- Figure: 1 scene + n_metrics --
    height_ratios = [2] + [1] * n_metrics
    fig = plt.figure(figsize=(14, 3 + 3 * n_metrics))
    gs = fig.add_gridspec(1 + n_metrics, 1, height_ratios=height_ratios,
                          hspace=0.35,
                          left=0.05, right=0.97, top=0.97, bottom=0.03)

    ax_scene = fig.add_subplot(gs[0])
    metric_axes = {}
    first_ax = None
    for j, m in enumerate(args.metrics):
        ax = fig.add_subplot(gs[1 + j], sharex=first_ax) if first_ax else fig.add_subplot(gs[1 + j])
        if first_ax is None:
            first_ax = ax
        metric_axes[m] = ax

    # -- Auto-scale y-axes from all data --
    def _all_vals(key, sub):
        vals = []
        for r in runs:
            vals.extend(r[key][sub])
        return vals

    # -- Set up metric axes --
    metric_lines = {m: {} for m in args.metrics}
    metric_cursors = {}

    for m, ax in metric_axes.items():
        ax.set_xlim(0, n_frames - 1)
        metric_cursors[m] = ax.axvline(0, color="black", linewidth=0.8, alpha=0.4)

        if m == "dv":
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            all_v = _all_vals("dv", "true_v") + _all_vals("dv", "inf_v")
            if all_v:
                margin = max(abs(max(all_v)), abs(min(all_v))) * 0.15
                ax.set_ylim(min(all_v) - margin, max(all_v) + margin)
            ax.set_ylabel("$\\delta v$ (m/s)")
            ax.set_title("Velocity error  (faint=true, solid=inferred)", fontsize=10)
            for r in runs:
                metric_lines[m][r["label"]] = {
                    "true": ax.plot([], [], color=r["color"], linewidth=0.5, alpha=0.3)[0],
                    "inf": ax.plot([], [], color=r["color"], linewidth=1.5, label=r["label"])[0],
                }
            ax.legend(fontsize=8, loc="upper right")

        elif m == "ni_stats":
            ax.axhline(0, color="grey", linewidth=1.0, linestyle=":", alpha=0.5)
            ax.axhline(1, color="grey", linewidth=1.0, linestyle=":", alpha=0.5)
            ax.set_ylim(-1.5, 4.0)
            ax.set_ylabel("value")
            ax.set_title("NI stats  ($\\hat{\\mu}\\to 0$, $\\hat{\\sigma}^2\\to 1$, acf$\\to 0$)", fontsize=10)
            for r in runs:
                metric_lines[m][r["label"]] = {
                    "mean": ax.plot([], [], color=r["color"], linewidth=1.5, linestyle="-")[0],
                    "var": ax.plot([], [], color=r["color"], linewidth=1.5, linestyle="--")[0],
                    "acf": ax.plot([], [], color=r["color"], linewidth=1.5, linestyle=":")[0],
                }
            # Manual legend
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], linestyle="-", color="grey", label="$\\hat{\\mu}$"),
                       Line2D([0], [0], linestyle="--", color="grey", label="$\\hat{\\sigma}^2$"),
                       Line2D([0], [0], linestyle=":", color="grey", label="acf(1)")]
            for r in runs:
                handles.append(Line2D([0], [0], color=r["color"], linewidth=2, label=r["label"]))
            ax.legend(handles=handles, fontsize=7, loc="upper right", ncol=2)

        elif m == "nees":
            ax.axhline(1.0, color="red", linewidth=1.0, linestyle=":", alpha=0.6,
                       label="expected = 1")
            all_nees = _all_vals("nees", "v")
            if all_nees:
                # Running averages will be smaller than raw, but show enough range
                ax.set_ylim(0, max(np.percentile(all_nees, 95) * 1.5, 3.0))
            else:
                ax.set_ylim(0, 5)
            ax.set_ylabel("NEES")
            ax.set_title("Normalized estimation error squared  (expect $\\approx 1$)", fontsize=10)
            for r in runs:
                metric_lines[m][r["label"]] = {
                    "raw": ax.plot([], [], color=r["color"], linewidth=0.5, alpha=0.2)[0],
                    "avg": ax.plot([], [], color=r["color"], linewidth=2.0, label=r["label"])[0],
                }
            ax.legend(fontsize=8, loc="upper right")

        elif m == "riccati":
            all_p = _all_vals("riccati", "v")
            all_pss = [np.sqrt(r["P_ss"]) for r in runs]
            if all_p:
                p_min = min(min(all_p), min(all_pss))
                p_max = max(max(all_p), max(all_pss))
                margin = max((p_max - p_min) * 0.15, p_min * 0.1)
                ax.set_ylim(max(0, p_min - margin), p_max + margin)
            ax.set_ylabel("$\\sqrt{P}$ (m/s)")
            ax.set_title("Kalman covariance convergence", fontsize=10)
            for r in runs:
                ax.axhline(np.sqrt(r["P_ss"]), color=r["color"],
                           linestyle=":", linewidth=0.8, alpha=0.5)
                metric_lines[m][r["label"]] = {
                    "P": ax.plot([], [], color=r["color"], linewidth=1.5, label=r["label"])[0],
                }
            ax.legend(fontsize=8, loc="upper right")

        ax.grid(alpha=0.2)

    # -- Scene (from first run) --
    scene_run = runs[0]
    trail_len = 30
    view_half_x = args.view_half_x

    def draw_scene(t):
        ax_scene.clear()
        ep = scene_run["ep"]
        ego_pos = ep["ego_positions"]
        veh_pos = ep["vehicle_positions"]
        dt = scene_run["meta"].get("dt", 0.1)

        for i in range(N_LANES + 1):
            y = i * LANE_WIDTH
            is_edge = (i == 0 or i == N_LANES)
            ax_scene.axhline(y, color="gray", linewidth=2 if is_edge else 1,
                             linestyle="-" if is_edge else "--", zorder=1)
        for i in range(N_LANES):
            ax_scene.axhspan(i * LANE_WIDTH, (i + 1) * LANE_WIDTH,
                             color="#f5f5f5", zorder=0)

        for idx, (vid, traj) in enumerate(sorted(veh_pos.items())):
            ti = min(t, len(traj) - 1)
            vx, vy, vh, vv = traj[ti]
            color = OBSTACLE_COLORS[idx % len(OBSTACLE_COLORS)]
            draw_vehicle(ax_scene, vx, vy, vh, color)
            ax_scene.annotate(f"{vv:.1f}", xy=(vx, vy + VEH_WIDTH),
                              fontsize=6, ha="center", color=color, zorder=15)
            ts = max(0, t - trail_len)
            trail = traj[ts:ti + 1]
            if len(trail) > 1:
                ax_scene.plot([s[0] for s in trail], [s[1] for s in trail],
                              "-", color=color, alpha=0.25, linewidth=2, zorder=1)

        ti = min(t, len(ego_pos) - 1)
        ex, ey, eh, ev = ego_pos[ti]
        draw_vehicle(ax_scene, ex, ey, eh, EGO_COLOR, alpha=0.85, label="EGO")
        ts = max(0, t - trail_len)
        trail = ego_pos[ts:ti + 1]
        if len(trail) > 1:
            ax_scene.plot([s[0] for s in trail], [s[1] for s in trail],
                          "-", color=EGO_COLOR, alpha=0.4, linewidth=2, zorder=1)

        ax_scene.set_xlim(ex - view_half_x, ex + view_half_x)
        ax_scene.set_ylim(-2.0, N_LANES * LANE_WIDTH + 2.0)
        ax_scene.set_aspect("equal")
        ax_scene.set_xticks([])
        ax_scene.set_yticks([])

        time_s = t * dt
        dist = ep.get("distance", [])
        d = dist[t] if t < len(dist) and dist[t] is not None else None
        info = f"t={time_s:.1f}s  v={ev:.1f}"
        if d is not None:
            info += f"  d={d:.1f}m"
        ax_scene.set_title(info, fontsize=10)

    def update(t):
        draw_scene(t)

        for m in args.metrics:
            for r in runs:
                lines = metric_lines[m][r["label"]]
                data = r[m] if m != "ni_stats" else r["ni"]

                if m == "dv":
                    d = r["dv"]
                    idx_t = [i for i, s in enumerate(d["true_s"]) if s <= t]
                    idx_i = [i for i, s in enumerate(d["inf_s"]) if s <= t]
                    if idx_t:
                        lines["true"].set_data([d["true_s"][i] for i in idx_t],
                                               [d["true_v"][i] for i in idx_t])
                    if idx_i:
                        lines["inf"].set_data([d["inf_s"][i] for i in idx_i],
                                              [d["inf_v"][i] for i in idx_i])

                elif m == "ni_stats":
                    d = r["ni"]
                    idx = [i for i, s in enumerate(d["s"]) if s <= t]
                    if len(idx) > 1:
                        arr = np.array([d["v"][i] for i in idx])
                        s_ni = [d["s"][i] for i in idx]
                        n_arr = np.arange(1, len(arr) + 1)
                        cm_ = np.cumsum(arr) / n_arr
                        cv = np.maximum(np.cumsum(arr**2)/n_arr - cm_**2, 0)
                        ca = np.zeros(len(arr))
                        for j in range(1, len(arr)):
                            num = np.sum(arr[:j] * arr[1:j+1])
                            den = np.sum(arr[:j+1]**2)
                            ca[j] = num / den if den > 1e-12 else 0
                        lines["mean"].set_data(s_ni, cm_)
                        lines["var"].set_data(s_ni, cv)
                        lines["acf"].set_data(s_ni, ca)

                elif m == "nees":
                    d = r["nees"]
                    idx = [i for i, s in enumerate(d["s"]) if s <= t]
                    if idx:
                        s_n = [d["s"][i] for i in idx]
                        v_n = [d["v"][i] for i in idx]
                        lines["raw"].set_data(s_n, v_n)
                        arr = np.array(v_n)
                        lines["avg"].set_data(s_n, np.cumsum(arr) / np.arange(1, len(arr) + 1))

                elif m == "riccati":
                    d = r["riccati"]
                    idx = [i for i, s in enumerate(d["s"]) if s <= t]
                    if idx:
                        lines["P"].set_data([d["s"][i] for i in idx],
                                            [d["v"][i] for i in idx])

            metric_cursors[m].set_xdata([t, t])

    if args.show:
        anim = animation.FuncAnimation(
            fig, update, frames=n_frames,
            interval=int(1000 / args.fps), repeat=True)
        plt.show(block=True)
    else:
        out_path = args.out or os.path.join(RESULTS_DIR, "comparison.gif")
        print(f"  Rendering {n_frames} frames ...")
        anim = animation.FuncAnimation(
            fig, update, frames=n_frames,
            interval=int(1000 / args.fps), repeat=False)
        ext = os.path.splitext(out_path)[1].lower()
        writer = animation.FFMpegWriter(fps=args.fps) if ext == ".mp4" else animation.PillowWriter(fps=args.fps)
        anim.save(out_path, writer=writer, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
