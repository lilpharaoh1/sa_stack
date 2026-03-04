"""
Render belief-aware scene views for all steps of an experiment episode.

Produces per-step scene snapshots (with P(hidden) darkening and colorbar)
and control dial plots.  Optionally assembles them into videos.

Usage:
    # Render all frames as PNGs
    python scripts/experiments/render_beliefs.py results/my_run.pkl --frames out/

    # Render all frames and assemble a video
    python scripts/experiments/render_beliefs.py results/my_run.pkl --video out/

    # Video, keep the individual PNGs too
    python scripts/experiments/render_beliefs.py results/my_run.pkl --video out/ --keep-frames

    # Single step (same as belief_snapshot.py)
    python scripts/experiments/render_beliefs.py results/my_run.pkl --step 30

    # Batch results, episode 2
    python scripts/experiments/render_beliefs.py results/my_run.pkl --video out/ -e 2
"""

import sys
import os
import argparse
from typing import Optional, Tuple

import dill
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip
from belief_utils import ExperimentResult, StepRecord
from belief_snapshot import (
    create_belief_snapshot,
    create_control_dial,
    create_combined_snapshot,
)


def _extract_ego_goal(config: dict) -> Optional['ip.BoxGoal']:
    """Extract the ego goal from an expanded config dict."""
    agents = config.get("agents", [])
    if not agents:
        return None
    ego_cfg = agents[0]
    goal_cfg = ego_cfg.get("goal", {}).get("box")
    if goal_cfg is None:
        return None
    return ip.BoxGoal(ip.Box(
        np.array(goal_cfg["center"]),
        goal_cfg["length"],
        goal_cfg["width"],
        goal_cfg.get("heading", 0.0),
    ))


def render_belief_frames(result: ExperimentResult,
                         scenario_map: 'ip.Map',
                         out_dir: str,
                         figsize: Tuple[float, float] = (8, 8),
                         dpi: int = 150,
                         keep_frames: bool = False,
                         video: bool = True,
                         margin: float = 40.0):
    """Render every step as belief snapshot + control dial PNGs.

    If *video* is True, assembles each set into an mp4.  Individual PNGs
    are deleted afterwards unless *keep_frames* is True.

    Args:
        result: Experiment result with step records.
        scenario_map: Parsed road map.
        out_dir: Output directory.
        figsize: Figure size for scene plots.
        dpi: Image resolution.
        keep_frames: Keep PNGs after creating videos.
        video: Assemble frames into mp4s.
        margin: View margin around ego in metres.
    """
    scene_dir = os.path.join(out_dir, "scene")
    dial_dir = os.path.join(out_dir, "dial")
    combined_dir = os.path.join(out_dir, "scene_dial")
    os.makedirs(scene_dir, exist_ok=True)
    os.makedirs(dial_dir, exist_ok=True)
    os.makedirs(combined_dir, exist_ok=True)

    n = len(result.steps)
    fps = result.fps
    ego_goal = _extract_ego_goal(result.config)

    # Snap figure size for video codec compatibility (divisible by 16)
    macro = 16
    w_px = int(round(figsize[0] * dpi / macro)) * macro
    h_px = int(round(figsize[1] * dpi / macro)) * macro
    scene_figsize = (w_px / dpi, h_px / dpi)

    dial_figsize = (5, 5)
    dw_px = int(round(dial_figsize[0] * dpi / macro)) * macro
    dh_px = int(round(dial_figsize[1] * dpi / macro)) * macro
    dial_figsize = (dw_px / dpi, dh_px / dpi)

    combined_figsize = (14, 7)
    cw_px = int(round(combined_figsize[0] * dpi / macro)) * macro
    ch_px = int(round(combined_figsize[1] * dpi / macro)) * macro
    combined_figsize = (cw_px / dpi, ch_px / dpi)

    scene_paths = []
    dial_paths = []
    combined_paths = []

    render_kwargs = dict(fps=fps, ego_goal=ego_goal, margin=margin)

    for i, sr in enumerate(result.steps):
        # --- Scene frame ---
        fig_scene, _ = create_belief_snapshot(
            sr, scenario_map,
            figsize=scene_figsize,
            **render_kwargs,
        )
        scene_path = os.path.join(scene_dir, f"frame_{i:04d}.png")
        fig_scene.savefig(scene_path, dpi=dpi)
        plt.close(fig_scene)
        scene_paths.append(scene_path)

        # --- Dial frame ---
        fig_dial, _ = create_control_dial(sr, figsize=dial_figsize)
        dial_path = os.path.join(dial_dir, f"frame_{i:04d}.png")
        fig_dial.savefig(dial_path, dpi=dpi)
        plt.close(fig_dial)
        dial_paths.append(dial_path)

        # --- Combined frame ---
        fig_comb, _ = create_combined_snapshot(
            sr, scenario_map,
            figsize=combined_figsize,
            **render_kwargs,
        )
        comb_path = os.path.join(combined_dir, f"frame_{i:04d}.png")
        fig_comb.savefig(comb_path, dpi=dpi)
        plt.close(fig_comb)
        combined_paths.append(comb_path)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  Rendered {i + 1}/{n} frames")

    print(f"Scene frames saved to {scene_dir}/")
    print(f"Dial  frames saved to {dial_dir}/")
    print(f"Combined frames saved to {combined_dir}/")

    if video:
        import imageio.v2 as imageio

        for label, paths, w, h in [
            ("scene", scene_paths, w_px, h_px),
            ("dial", dial_paths, dw_px, dh_px),
            ("scene_dial", combined_paths, cw_px, ch_px),
        ]:
            video_path = os.path.join(out_dir, f"{label}.mp4")
            print(f"Assembling {label} video ({w}x{h}px at {fps} fps)...")
            writer = imageio.get_writer(
                video_path, fps=fps,
                codec='libx264', pixelformat='yuv420p',
                output_params=['-profile:v', 'baseline',
                               '-level', '3.0'])
            for fp in paths:
                writer.append_data(imageio.imread(fp))
            writer.close()
            print(f"  Saved to {video_path}")

        if not keep_frames:
            for fp in scene_paths + dial_paths + combined_paths:
                os.remove(fp)
            os.rmdir(scene_dir)
            os.rmdir(dial_dir)
            os.rmdir(combined_dir)
            print(f"Cleaned up {len(scene_paths) + len(dial_paths) + len(combined_paths)} frame files")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render belief-aware scene views for an experiment episode")
    parser.add_argument("path", type=str, help="Path to .pkl results file")
    parser.add_argument("--step", "-s", type=int, default=None,
                        help="Render a single step (default: all steps)")
    parser.add_argument("--episode", "-e", type=int, default=0,
                        help="Episode index for batch results (default: 0)")
    parser.add_argument("--video", type=str, default=None,
                        help="Render all steps and assemble video to this dir")
    parser.add_argument("--frames", type=str, default=None,
                        help="Render all steps as PNG frames only (no video)")
    parser.add_argument("--keep-frames", action="store_true",
                        help="Keep individual PNG frames after creating video")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Save single-step figure to file instead of showing")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved images (default: 150)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[8, 8],
                        metavar=("W", "H"),
                        help="Figure size in inches (default: 8 8)")
    parser.add_argument("--margin", type=float, default=40.0,
                        help="View margin around ego in metres (default: 40)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.path):
        print(f"File not found: {args.path}")
        sys.exit(1)

    with open(args.path, 'rb') as f:
        data = dill.load(f)

    # Extract the ExperimentResult
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
        if args.episode >= len(results):
            print(f"Episode {args.episode} out of range "
                  f"(have {len(results)} episodes)")
            sys.exit(1)
        result = results[args.episode]
        print(f"Loaded batch file with {len(results)} episodes, "
              f"using episode {args.episode}")
    elif isinstance(data, ExperimentResult):
        result = data
        print("Loaded single episode result")
    else:
        print(f"Unknown format: {type(data)}")
        sys.exit(1)

    if not result.steps:
        print("No step records in this result.")
        sys.exit(1)

    # Load the map
    map_path = result.config.get("scenario", {}).get("map_path")
    if map_path is None:
        print("No map_path in config, cannot render.")
        sys.exit(1)

    scenario_map = ip.Map.parse_from_opendrive(map_path)

    print(f"Scenario: {result.scenario_name}  |  "
          f"Steps: {len(result.steps)}  |  "
          f"Solved: {result.solved}  |  Failed: {result.failed}")

    figsize = tuple(args.figsize)

    # --- Video mode ---
    if args.video is not None:
        print(f"Rendering {len(result.steps)} frames to {args.video}/")
        render_belief_frames(result, scenario_map, args.video,
                             figsize=figsize, dpi=args.dpi,
                             keep_frames=args.keep_frames,
                             video=True, margin=args.margin)
        return

    # --- Frames-only mode ---
    if args.frames is not None:
        print(f"Rendering {len(result.steps)} frames to {args.frames}/")
        render_belief_frames(result, scenario_map, args.frames,
                             figsize=figsize, dpi=args.dpi,
                             keep_frames=True, video=False,
                             margin=args.margin)
        return

    # --- Single step mode ---
    ego_goal = _extract_ego_goal(result.config)
    step_idx = args.step
    if step_idx is None:
        step_idx = len(result.steps) - 1
        print(f"No --step given, using last step ({step_idx})")
    elif step_idx >= len(result.steps):
        print(f"Step {step_idx} out of range (have {len(result.steps)} steps)")
        sys.exit(1)

    sr = result.steps[step_idx]
    print(f"Rendering step {sr.step} (index {step_idx})")

    fig_scene, _ = create_belief_snapshot(
        sr, scenario_map, figsize=figsize,
        fps=result.fps, ego_goal=ego_goal, margin=args.margin,
    )
    fig_dial, _ = create_control_dial(sr)

    if args.out:
        fig_scene.savefig(args.out, dpi=args.dpi)
        print(f"Saved scene to {args.out}")
        plt.close(fig_scene)

        base, ext = os.path.splitext(args.out)
        dial_path = f"{base}_dial{ext}"
        fig_dial.savefig(dial_path, dpi=args.dpi)
        print(f"Saved dial  to {dial_path}")
        plt.close(fig_dial)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
