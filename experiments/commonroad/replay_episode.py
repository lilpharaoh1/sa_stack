"""
Replay a saved CommonRoad car-follow episode and export as GIF or MP4.

Reads ego_positions and vehicle_positions from episode.json and renders
the scene frame-by-frame using the same visual style as run_carfollow.py.

Usage:
    # Replay and save GIF (auto-detects episode.json inside run dir)
    python experiments/commonroad/replay_episode.py results/run_dir/

    # Save as MP4 instead
    python experiments/commonroad/replay_episode.py results/run_dir/ -o replay.mp4

    # Custom output path
    python experiments/commonroad/replay_episode.py results/run_dir/ -o /tmp/my_replay.gif

    # Faster playback
    python experiments/commonroad/replay_episode.py results/run_dir/ --fps 30

    # Show in live window instead of saving
    python experiments/commonroad/replay_episode.py results/run_dir/ --show
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

# Constants matching run_experiment.py / run_carfollow.py
VEH_LENGTH = 4.5
VEH_WIDTH = 1.8
LANE_WIDTH = 3.5
N_LANES = 3

OBSTACLE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
EGO_COLOR = "#27ae60"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_run(run_dir: str) -> tuple:
    """Load episode.json and metadata.json from a run directory."""
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
    """Build a readable title from metadata."""
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


def draw_frame(ax, t, ego_positions, vehicle_positions, dt,
               title="", view_half_x=60.0,
               human_accel=None, executed_accel=None,
               distance=None, intervened=None):
    ax.clear()

    y_lo, y_hi = -2.0, N_LANES * LANE_WIDTH + 2.0

    # Lane markings
    for i in range(N_LANES + 1):
        y = i * LANE_WIDTH
        is_edge = (i == 0 or i == N_LANES)
        ax.axhline(y, color="gray", linewidth=2 if is_edge else 1,
                   linestyle="-" if is_edge else "--", zorder=1)
    for i in range(N_LANES):
        ax.axhspan(i * LANE_WIDTH, (i + 1) * LANE_WIDTH,
                   color="#f5f5f5", zorder=0)

    n_frames = len(ego_positions)
    trail_len = 30

    # Non-ego vehicles
    sorted_vids = sorted(vehicle_positions.keys())
    for idx, vid in enumerate(sorted_vids):
        traj = vehicle_positions[vid]
        ti = min(t, len(traj) - 1)
        vx, vy, vh, vv = traj[ti]
        color = OBSTACLE_COLORS[idx % len(OBSTACLE_COLORS)]
        draw_vehicle(ax, vx, vy, vh, color)
        ax.annotate(f"{vid}", xy=(vx, vy), fontsize=7, fontweight="bold",
                    ha="center", va="center", color="white", zorder=20)
        ax.annotate(f"{vv:.1f} m/s", xy=(vx, vy + VEH_WIDTH),
                    fontsize=6, ha="center", color=color, zorder=15)
        # Trail
        trail_start = max(0, t - trail_len)
        trail = traj[trail_start:ti + 1]
        if len(trail) > 1:
            ax.plot([s[0] for s in trail], [s[1] for s in trail],
                    "-", color=color, alpha=0.25, linewidth=2, zorder=1)

    # Ego vehicle
    ti = min(t, n_frames - 1)
    ex, ey, eh, ev = ego_positions[ti]
    draw_vehicle(ax, ex, ey, eh, EGO_COLOR, alpha=0.85, label="EGO")
    ax.annotate(f"{ev:.1f} m/s", xy=(ex, ey + VEH_WIDTH),
                fontsize=6, ha="center", color=EGO_COLOR, zorder=15)
    # Trail
    trail_start = max(0, t - trail_len)
    trail = ego_positions[trail_start:ti + 1]
    if len(trail) > 1:
        ax.plot([s[0] for s in trail], [s[1] for s in trail],
                "-", color=EGO_COLOR, alpha=0.4, linewidth=2, zorder=1)

    ax.set_xlim(ex - view_half_x, ex + view_half_x)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Title with diagnostics
    time_s = t * dt
    info_str = f"t = {time_s:.1f} s   v = {ev:.1f} m/s"
    if distance is not None and t < len(distance) and distance[t] is not None:
        info_str += f"   d = {distance[t]:.1f} m"
    if intervened is not None and t < len(intervened) and intervened[t]:
        info_str += "   [INTERVENED]"
    ax.set_title(f"{title}    {info_str}", fontsize=11)


def parse_args():
    p = argparse.ArgumentParser(
        description="Replay a CommonRoad car-follow episode as GIF/MP4")
    p.add_argument("run_dir",
                   help="Run directory (or name under results/)")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Output path (default: scene_replay.gif in run dir). "
                        "Extension determines format (.gif or .mp4)")
    p.add_argument("--fps", type=int, default=20,
                   help="Frames per second (default: 20)")
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--figsize", type=float, nargs=2, default=[14, 5],
                   metavar=("W", "H"))
    p.add_argument("--view-half-x", type=float, default=60.0,
                   help="Half-width of the x viewport around ego (default: 60)")
    p.add_argument("--show", action="store_true",
                   help="Show in a live window instead of saving")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.show:
        matplotlib.use("Agg")

    episode, meta, run_dir = load_run(args.run_dir)

    ego_positions = episode["ego_positions"]
    vehicle_positions = {k: v for k, v in episode["vehicle_positions"].items()}
    dt = meta.get("dt", 0.1)
    distance = episode.get("distance")
    intervened = episode.get("intervened")
    title = build_title(meta)
    n_frames = len(ego_positions)

    print(f"  Run:    {os.path.basename(run_dir)}")
    print(f"  Title:  {title}")
    print(f"  Frames: {n_frames}  ({n_frames * dt:.1f} s)")

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    fig.subplots_adjust(left=0.01, right=0.995, top=0.95, bottom=0.01)

    def update(frame):
        draw_frame(ax, frame, ego_positions, vehicle_positions, dt,
                   title=title, view_half_x=args.view_half_x,
                   distance=distance, intervened=intervened)

    if args.show:
        anim = animation.FuncAnimation(
            fig, update, frames=n_frames,
            interval=int(1000 / args.fps), repeat=True)
        plt.show(block=True)
    else:
        out_path = args.out
        if out_path is None:
            out_path = os.path.join(run_dir, "scene_replay.gif")

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
