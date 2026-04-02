"""
Side-by-side scene comparison of two runs on the same episode.

Renders each step as two subplots (left = run A, right = run B) so you can
visually compare trajectories, interventions, and agent positions frame by
frame.

Usage:
    # Render frames for episode 0:
    python scripts/experiments/render_scene_compare.py runA/ runB/ --frames out/

    # Single step:
    python scripts/experiments/render_scene_compare.py runA/ runB/ --step 20

    # Custom labels:
    python scripts/experiments/render_scene_compare.py runA/ runB/ \
        --labels "Greedy" "QCBF" --frames out/

    # Positions only (no planned paths):
    python scripts/experiments/render_scene_compare.py runA/ runB/ \
        --no-paths --frames out/
"""

import sys
import os
import argparse
from typing import Optional, Tuple

import dill
import numpy as np
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", ".."))
sys.path.insert(0, _EXPERIMENTS_DIR)

import igp2 as ip
from igp2.opendrive.plot_map import plot_map
from utils import ExperimentResult, RESULTS_DIR
from render_scene import render_scene_frame, _extract_ego_goal


def _load_result(path: str, episode: int) -> ExperimentResult:
    """Load a single ExperimentResult from a run directory or pkl file."""
    if os.path.isdir(path):
        pkl = os.path.join(path, "results.pkl")
    elif os.path.isdir(os.path.join(RESULTS_DIR, path)):
        pkl = os.path.join(RESULTS_DIR, path, "results.pkl")
    else:
        pkl = path

    with open(pkl, 'rb') as f:
        data = dill.load(f)

    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        if episode >= len(results):
            print(f"Episode {episode} out of range (have {len(results)})")
            sys.exit(1)
        return results[episode]
    elif isinstance(data, ExperimentResult):
        return data
    else:
        print(f"Unknown format: {type(data)}")
        sys.exit(1)


def render_compare_frame(sr_a, sr_b, scenario_map, fig, axes,
                         label_a="Run A", label_b="Run B",
                         **kwargs):
    """Render a side-by-side comparison frame."""
    for ax in axes:
        ax.clear()
        plot_map(scenario_map, ax=ax, markings=True)
        ax.set_aspect('equal')

    fps = kwargs.get('fps', 10)

    # Left panel
    render_scene_frame(sr_a, scenario_map, axes[0], **kwargs)
    step_title_a = f"{label_a}  |  step={sr_a.step}, t={sr_a.step/fps:.1f}s"
    if sr_a.ego_speed is not None:
        step_title_a += f", v={sr_a.ego_speed:.1f}m/s"
    axes[0].set_title(step_title_a, fontsize=10)

    # Right panel
    render_scene_frame(sr_b, scenario_map, axes[1], **kwargs)
    step_title_b = f"{label_b}  |  step={sr_b.step}, t={sr_b.step/fps:.1f}s"
    if sr_b.ego_speed is not None:
        step_title_b += f", v={sr_b.ego_speed:.1f}m/s"
    axes[1].set_title(step_title_b, fontsize=10)


def parse_args():
    p = argparse.ArgumentParser(
        description="Side-by-side scene comparison of two runs")
    p.add_argument("dirs", nargs=2,
                   help="Two run directories (or names under results/)")
    p.add_argument("--labels", nargs=2, default=None,
                   help="Labels for each run (default: directory names)")
    p.add_argument("--episode", "-e", type=int, default=0,
                   help="Episode index (default: 0)")
    p.add_argument("--step", "-s", type=int, default=None,
                   help="Single step to render (default: all steps)")
    p.add_argument("--frames", type=str, default=None,
                   help="Output directory for PNG frames")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Save single frame to file (use with --step)")
    p.add_argument("--no-paths", action="store_true",
                   help="Hide all planned paths (show positions only)")
    p.add_argument("--no-milp", action="store_true",
                   help="Hide MILP rollout paths")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--figsize", type=float, nargs=2, default=[24, 8],
                   metavar=("W", "H"))
    p.add_argument("--margin", type=float, default=40.0,
                   help="View margin around ego in metres (default: 40)")
    return p.parse_args()


def main():
    args = parse_args()

    result_a = _load_result(args.dirs[0], args.episode)
    result_b = _load_result(args.dirs[1], args.episode)

    if args.labels:
        label_a, label_b = args.labels
    else:
        label_a = os.path.basename(args.dirs[0].rstrip('/'))
        label_b = os.path.basename(args.dirs[1].rstrip('/'))

    n_a, n_b = len(result_a.steps), len(result_b.steps)
    n_steps = min(n_a, n_b)
    print(f"Run A ({label_a}): {n_a} steps  |  "
          f"Run B ({label_b}): {n_b} steps  |  "
          f"Comparing {n_steps} steps")

    # Load map from whichever result has it
    map_path = (result_a.config.get("scenario", {}).get("map_path")
                or result_b.config.get("scenario", {}).get("map_path"))
    if not map_path:
        print("No map_path in config.")
        sys.exit(1)
    scenario_map = ip.Map.parse_from_opendrive(map_path)

    ego_goal_a = _extract_ego_goal(result_a.config)
    ego_goal_b = _extract_ego_goal(result_b.config)

    fps = result_a.fps

    render_kwargs = dict(
        fps=fps,
        show_milp=not args.no_milp,
        show_paths=not args.no_paths,
        show_legend=True,
        margin=args.margin,
    )

    figsize = tuple(args.figsize)

    if args.step is not None:
        # Single frame mode
        if args.step >= n_steps:
            print(f"Step {args.step} out of range (max {n_steps - 1})")
            sys.exit(1)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        render_compare_frame(
            result_a.steps[args.step],
            result_b.steps[args.step],
            scenario_map, fig, axes,
            label_a=label_a, label_b=label_b,
            ego_goal=ego_goal_a, **render_kwargs)
        # Set ego_goal separately for right panel if different
        fig.tight_layout()

        if args.out:
            fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved to {args.out}")
            plt.close(fig)
        else:
            plt.show(block=True)
        return

    if args.frames is not None:
        # All frames mode
        os.makedirs(args.frames, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for i in range(n_steps):
            render_compare_frame(
                result_a.steps[i],
                result_b.steps[i],
                scenario_map, fig, axes,
                label_a=label_a, label_b=label_b,
                ego_goal=ego_goal_a, **render_kwargs)
            fig.tight_layout()

            frame_path = os.path.join(args.frames, f"frame_{i:04d}.png")
            fig.savefig(frame_path, dpi=args.dpi)

            if (i + 1) % 10 == 0 or (i + 1) == n_steps:
                print(f"  Rendered {i + 1}/{n_steps} frames")

        plt.close(fig)
        print(f"Frames saved to {args.frames}/")
        return

    # Default: show last step
    step_idx = n_steps - 1
    print(f"No --step or --frames given, showing last step ({step_idx})")
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    render_compare_frame(
        result_a.steps[step_idx],
        result_b.steps[step_idx],
        scenario_map, fig, axes,
        label_a=label_a, label_b=label_b,
        ego_goal=ego_goal_a, **render_kwargs)
    fig.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
