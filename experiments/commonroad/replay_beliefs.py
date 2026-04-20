"""
Replay belief evolution from a saved CommonRoad car-follow episode as a GIF.

Shows a time-series of true kappa vs inferred estimate (kf_kappa or mean),
with +/- 2 std shading from the continuous Kalman variance.

The full true and inferred traces are drawn from the start; a vertical
cursor sweeps across to indicate the current timestep.

Usage:
    # Save GIF (default: beliefs_replay.gif in run dir)
    python experiments/commonroad/replay_beliefs.py results/run_dir/

    # Custom output
    python experiments/commonroad/replay_beliefs.py results/run_dir/ -o beliefs.gif

    # Show in live window
    python experiments/commonroad/replay_beliefs.py results/run_dir/ --show

    # Faster
    python experiments/commonroad/replay_beliefs.py results/run_dir/ --fps 30
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Replay belief evolution as GIF from a saved episode")
    p.add_argument("run_dir",
                   help="Run directory (or name under results/)")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Output path (default: beliefs_replay.gif in run dir)")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--figsize", type=float, nargs=2, default=[8, 3],
                   metavar=("W", "H"))
    p.add_argument("--show", action="store_true",
                   help="Show in live window instead of saving")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    episode, meta, run_dir = load_run(args.run_dir)
    title = build_title(meta)
    dt = meta.get("dt", 0.1)

    human_vel_err = episode.get("human_vel_err", [])
    inferred_mean = episode.get("inferred_mean", [])
    kf_kappa = episode.get("kf_kappa", [])
    kf_P = episode.get("kf_P", [])
    n_frames = len(episode["steps"])

    # True velocity errors (initial value from config)
    true_vel_errors = episode.get("true_vel_errors", {})
    initial_kappa = None
    if true_vel_errors:
        initial_kappa = list(true_vel_errors.values())[0]

    print(f"  Run:    {os.path.basename(run_dir)}")
    print(f"  Title:  {title}")
    print(f"  Frames: {n_frames}  ({n_frames * dt:.1f} s)")

    # -- Pre-compute full traces (only where data exists) --
    all_steps = list(range(n_frames))

    true_steps, true_vals = [], []
    for t in all_steps:
        hve = human_vel_err[t] if t < len(human_vel_err) else None
        if hve is not None:
            true_steps.append(t)
            true_vals.append(hve)

    inf_steps, inf_vals, inf_upper, inf_lower = [], [], [], []
    for t in all_steps:
        est = kf_kappa[t] if t < len(kf_kappa) and kf_kappa[t] is not None else None
        if est is None:
            est = inferred_mean[t] if t < len(inferred_mean) and inferred_mean[t] is not None else None
        if est is not None:
            inf_steps.append(t)
            inf_vals.append(est)
            kf_p = kf_P[t] if t < len(kf_P) and kf_P[t] is not None else None
            if kf_p is not None:
                std = np.sqrt(kf_p)
                inf_upper.append(est + 2 * std)
                inf_lower.append(est - 2 * std)
            else:
                inf_upper.append(None)
                inf_lower.append(None)

    # -- Fixed axis limits --
    y_top = (initial_kappa if initial_kappa is not None else 0.5) + 0.05
    y_bottom = -0.05

    # -- Set up figure --
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.13)

    # Static elements
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    if initial_kappa is not None:
        ax.axhline(initial_kappa, color="red", linewidth=0.8,
                   linestyle="--", alpha=0.4,
                   label=f"$\\kappa_0$ = {initial_kappa:.2f}")

    # Full traces (drawn once, then revealed progressively)
    line_true, = ax.plot([], [], color="steelblue", linewidth=1.5,
                         label="true $\\kappa$")
    line_inf, = ax.plot([], [], color="orange", linewidth=1.5,
                        label="inferred")
    fill_artist = [None]

    # Cursor
    cursor = ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)

    ax.set_xlim(0, n_frames - 1)
    ax.set_ylim(y_bottom, y_top)
    ax.set_xlabel("step")
    ax.set_ylabel("$\\kappa$")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=10)

    def update(t):
        # Reveal true trace up to t
        idx_true = [i for i, s in enumerate(true_steps) if s <= t]
        if idx_true:
            line_true.set_data(
                [true_steps[i] for i in idx_true],
                [true_vals[i] for i in idx_true])

        # Reveal inferred trace up to t
        idx_inf = [i for i, s in enumerate(inf_steps) if s <= t]
        if idx_inf:
            line_inf.set_data(
                [inf_steps[i] for i in idx_inf],
                [inf_vals[i] for i in idx_inf])

            # Confidence band
            if fill_artist[0] is not None:
                fill_artist[0].remove()
                fill_artist[0] = None
            band = [(inf_steps[i], inf_lower[i], inf_upper[i])
                    for i in idx_inf
                    if inf_lower[i] is not None and inf_upper[i] is not None]
            if band:
                bs, bl, bu = zip(*band)
                fill_artist[0] = ax.fill_between(
                    bs, bl, bu, color="orange", alpha=0.15)

        cursor.set_xdata([t, t])

    if args.show:
        anim = animation.FuncAnimation(
            fig, update, frames=n_frames,
            interval=int(1000 / args.fps), repeat=True)
        plt.show(block=True)
    else:
        out_path = args.out
        if out_path is None:
            out_path = os.path.join(run_dir, "beliefs_replay.gif")

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
