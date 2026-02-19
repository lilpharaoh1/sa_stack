"""
Render a static scene view at a given simulation timestep.

Loads a .pkl results file and draws the road layout, all agents (ego,
dynamic, static), the true NLP rollout path, and the human NLP rollout
path (if present).

The core rendering function ``render_scene_frame`` can also be called
programmatically to generate video frames.

Usage:
    # Show scene at step 10
    python scripts/experiments/render_scene.py results/my_run.pkl --step 10

    # Save to file instead of showing
    python scripts/experiments/render_scene.py results/my_run.pkl --step 10 --out scene.png

    # Render all steps as video frames
    python scripts/experiments/render_scene.py results/my_run.pkl --video frames/

    # For batch results, select episode index
    python scripts/experiments/render_scene.py results/batch.pkl --step 10 --episode 3
"""

import sys
import os
import argparse
from typing import Optional, Dict, Tuple

import dill
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip
from igp2.opendrive.plot_map import plot_map
from igp2.core.util import calculate_multiple_bboxes
from ind_belief_experiment import ExperimentResult, StepRecord

# Colours
COLOUR_EGO = (0.2, 0.4, 0.9)
COLOUR_DYNAMIC = (0.85, 0.75, 0.3)
COLOUR_STATIC = (0.6, 0.6, 0.6)
COLOUR_TRUE_NLP = (0.0, 0.7, 0.0)
COLOUR_HUMAN_NLP = (0.9, 0.1, 0.1)
COLOUR_TRUE_MILP = (0.0, 0.5, 0.0)
COLOUR_HUMAN_MILP = (0.7, 0.0, 0.0)


def render_scene_frame(step_record: StepRecord,
                       scenario_map: 'ip.Map',
                       ax: plt.Axes,
                       *,
                       fps: int = 10,
                       ego_goal: Optional['ip.BoxGoal'] = None,
                       margin: float = 40.0,
                       show_milp: bool = True,
                       show_legend: bool = True,
                       title: Optional[str] = None):
    """Render a single scene frame onto the given axes.

    This is the core rendering function.  Call it directly to produce
    individual frames for video generation, or let the CLI handle it.

    Args:
        step_record: The StepRecord for the timestep to render.
        scenario_map: Parsed road map.
        ax: Matplotlib axes to draw on (should already have the map drawn
            if you want to avoid re-drawing the static map each frame).
        fps: Simulation FPS (used to compute simulation time).
        ego_goal: Optional ego goal to draw as a marker.
        margin: View margin around the ego in metres.
        show_milp: Whether to draw MILP rollout paths.
        show_legend: Whether to draw the legend.
        title: Optional title override.
    """
    sr = step_record

    # Centre the view on the ego
    if sr.ego_position is not None:
        ax.set_xlim(sr.ego_position[0] - margin, sr.ego_position[0] + margin)
        ax.set_ylim(sr.ego_position[1] - margin, sr.ego_position[1] + margin)

    # --- Draw ego goal ---
    if ego_goal is not None:
        corners = np.array(ego_goal.box.boundary)
        goal_poly = MplPolygon(corners, closed=True,
                               facecolor=(*COLOUR_EGO, 0.08),
                               edgecolor=(*COLOUR_EGO, 0.4),
                               linewidth=1.5, linestyle=':', zorder=2)
        ax.add_patch(goal_poly)

    # --- Draw static obstacles ---
    for aid, state in sr.static_obstacles.items():
        meta = getattr(state, 'metadata', None)
        vl = meta.length if meta else 4.5
        vw = meta.width if meta else 1.8
        corners = calculate_multiple_bboxes(
            [state.position[0]], [state.position[1]],
            vl, vw, state.heading)[0]
        poly = MplPolygon(corners, closed=True,
                          facecolor=(*COLOUR_STATIC, 0.5),
                          edgecolor=(*COLOUR_STATIC, 0.9),
                          linewidth=1.0, zorder=4)
        ax.add_patch(poly)

    # --- Draw dynamic agents ---
    for aid, state in sr.dynamic_agents.items():
        meta = getattr(state, 'metadata', None)
        vl = meta.length if meta else 4.5
        vw = meta.width if meta else 1.8
        corners = calculate_multiple_bboxes(
            [state.position[0]], [state.position[1]],
            vl, vw, state.heading)[0]
        poly = MplPolygon(corners, closed=True,
                          facecolor=(*COLOUR_DYNAMIC, 0.6),
                          edgecolor=(*COLOUR_DYNAMIC, 1.0),
                          linewidth=1.5, zorder=5)
        ax.add_patch(poly)

        # Speed annotation
        ax.annotate(f"{state.speed:.1f} m/s",
                    xy=(state.position[0], state.position[1] - 2.5),
                    fontsize=6, ha='center', color=COLOUR_DYNAMIC,
                    alpha=0.8, zorder=7)

        # Heading arrow
        arrow_len = 2.5
        dx = arrow_len * np.cos(state.heading)
        dy = arrow_len * np.sin(state.heading)
        ax.annotate("", xy=(state.position[0] + dx, state.position[1] + dy),
                    xytext=(state.position[0], state.position[1]),
                    arrowprops=dict(arrowstyle='->', color=COLOUR_DYNAMIC,
                                   lw=1.5),
                    zorder=6)

    # --- Draw ego vehicle ---
    if sr.ego_position is not None:
        ego_heading = sr.ego_heading if sr.ego_heading is not None else 0.0
        corners = calculate_multiple_bboxes(
            [sr.ego_position[0]], [sr.ego_position[1]],
            4.5, 1.8, ego_heading)[0]
        ego_poly = MplPolygon(corners, closed=True,
                              facecolor=(*COLOUR_EGO, 0.6),
                              edgecolor=(*COLOUR_EGO, 1.0),
                              linewidth=2.0, zorder=6)
        ax.add_patch(ego_poly)

        # Heading arrow
        arrow_len = 3.0
        dx = arrow_len * np.cos(ego_heading)
        dy = arrow_len * np.sin(ego_heading)
        ax.annotate("", xy=(sr.ego_position[0] + dx, sr.ego_position[1] + dy),
                    xytext=(sr.ego_position[0], sr.ego_position[1]),
                    arrowprops=dict(arrowstyle='->', color=COLOUR_EGO, lw=2),
                    zorder=7)

        # Speed
        spd = sr.ego_speed if sr.ego_speed is not None else 0.0
        ax.annotate(f"ego {spd:.1f} m/s",
                    xy=(sr.ego_position[0], sr.ego_position[1] + 3.0),
                    fontsize=7, fontweight='bold', ha='center',
                    color=COLOUR_EGO, zorder=8)

    # --- Draw true NLP rollout ---
    if sr.true_rollout is not None and len(sr.true_rollout) > 1:
        ax.plot(sr.true_rollout[:, 0], sr.true_rollout[:, 1],
                color=COLOUR_TRUE_NLP, linewidth=2.0, zorder=5,
                label='True NLP')

    # --- Draw human NLP rollout ---
    if sr.human_rollout is not None and len(sr.human_rollout) > 1:
        ax.plot(sr.human_rollout[:, 0], sr.human_rollout[:, 1],
                color=COLOUR_HUMAN_NLP, linewidth=2.0, zorder=5,
                label='Human NLP')

    # --- Draw MILP rollouts ---
    if show_milp:
        if sr.true_milp_rollout is not None and len(sr.true_milp_rollout) > 1:
            ax.plot(sr.true_milp_rollout[:, 0], sr.true_milp_rollout[:, 1],
                    color=COLOUR_TRUE_MILP, linewidth=1.5, linestyle='--',
                    zorder=4, label='True MILP')

        if sr.human_milp_rollout is not None and len(sr.human_milp_rollout) > 1:
            ax.plot(sr.human_milp_rollout[:, 0], sr.human_milp_rollout[:, 1],
                    color=COLOUR_HUMAN_MILP, linewidth=1.5, linestyle='--',
                    zorder=4, label='Human MILP')

    # --- Legend ---
    if show_legend:
        legend_handles = [
            MplPolygon([[0, 0]], closed=True,
                       facecolor=(*COLOUR_EGO, 0.6),
                       edgecolor=(*COLOUR_EGO, 1.0),
                       linewidth=2, label='Ego'),
            MplPolygon([[0, 0]], closed=True,
                       facecolor=(*COLOUR_DYNAMIC, 0.6),
                       edgecolor=(*COLOUR_DYNAMIC, 1.0),
                       linewidth=1.5, label='Dynamic agent'),
            MplPolygon([[0, 0]], closed=True,
                       facecolor=(*COLOUR_STATIC, 0.5),
                       edgecolor=(*COLOUR_STATIC, 0.9),
                       linewidth=1, label='Static obstacle'),
            Line2D([0], [0], color=COLOUR_TRUE_NLP, linewidth=2,
                   label='True NLP'),
            Line2D([0], [0], color=COLOUR_HUMAN_NLP, linewidth=2,
                   label='Human NLP'),
        ]
        if show_milp:
            legend_handles.extend([
                Line2D([0], [0], color=COLOUR_TRUE_MILP, linewidth=1.5,
                       linestyle='--', label='True MILP'),
                Line2D([0], [0], color=COLOUR_HUMAN_MILP, linewidth=1.5,
                       linestyle='--', label='Human MILP'),
            ])
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
                  framealpha=0.9)

    # --- Title ---
    if title is not None:
        ax.set_title(title, fontsize=11)
    else:
        sim_time = sr.step / fps
        status_parts = [f"step={sr.step}"]
        if sr.ego_speed is not None:
            status_parts.append(f"v={sr.ego_speed:.1f}m/s")
        status_parts.append(f"t={sim_time:.1f}s")
        ax.set_title("  |  ".join(status_parts), fontsize=11)


def create_scene_figure(step_record: StepRecord,
                        scenario_map: 'ip.Map',
                        figsize: Tuple[float, float] = (14, 8),
                        **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Create a complete figure with map and scene for a single timestep.

    This is a convenience wrapper around ``render_scene_frame`` that also
    draws the static map.  Returns the figure and axes so the caller can
    save or show it.

    Args:
        step_record: The StepRecord to render.
        scenario_map: Parsed road map.
        figsize: Figure size.
        **kwargs: Forwarded to ``render_scene_frame``.

    Returns:
        (fig, ax) tuple.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_map(scenario_map, ax=ax, markings=True)
    ax.set_aspect('equal')
    render_scene_frame(step_record, scenario_map, ax, **kwargs)
    fig.tight_layout()
    return fig, ax


def render_video_frames(result: ExperimentResult,
                        scenario_map: 'ip.Map',
                        out_dir: str,
                        figsize: Tuple[float, float] = (14, 8),
                        dpi: int = 150,
                        keep_frames: bool = False,
                        video: bool = True,
                        **kwargs):
    """Render every step of an episode as PNGs and optionally assemble a video.

    Frames are rendered to *out_dir*.  If *video* is True they are stitched into
    ``out_dir/video.mp4`` using imageio.  The individual frame PNGs are
    deleted afterwards unless *keep_frames* is True.

    Args:
        result: The experiment result.
        scenario_map: Parsed road map.
        out_dir: Output directory for frame images and video.
        figsize: Figure size.
        dpi: Image resolution.
        keep_frames: If True, keep the individual PNG frames after
            creating the video.
        video: If True, assemble frames into an mp4.  If False, only
            save the PNG frames.
        **kwargs: Forwarded to ``render_scene_frame``.
    """

    os.makedirs(out_dir, exist_ok=True)
    n = len(result.steps)
    fps = result.fps

    # Snap figure size so pixel dimensions are divisible by 16
    # (required by most video codecs)
    macro = 16
    w_px = int(round(figsize[0] * dpi / macro)) * macro
    h_px = int(round(figsize[1] * dpi / macro)) * macro
    figsize = (w_px / dpi, h_px / dpi)

    # Extract ego goal from config for display
    ego_goal = _extract_ego_goal(result.config)

    frame_paths = []
    for i, sr in enumerate(result.steps):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plot_map(scenario_map, ax=ax, markings=True)
        ax.set_aspect('equal')

        kw = dict(kwargs, fps=fps, ego_goal=ego_goal,
                  show_legend=(i == 0))
        render_scene_frame(sr, scenario_map, ax, **kw)
        fig.tight_layout()

        frame_path = os.path.join(out_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        frame_paths.append(frame_path)

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  Rendered {i + 1}/{n} frames")

    print(f"Frames saved to {out_dir}/")

    if video:
        import imageio.v2 as imageio

        video_path = os.path.join(out_dir, "video.mp4")
        print(f"Assembling video ({w_px}x{h_px}px at {fps} fps)...")
        writer = imageio.get_writer(video_path, fps=fps,
                                    codec='libx264', pixelformat='yuv420p',
                                    output_params=['-profile:v', 'baseline',
                                                   '-level', '3.0'])
        for fp in frame_paths:
            writer.append_data(imageio.imread(fp))
        writer.close()
        print(f"Video saved to {video_path}")

        if not keep_frames:
            for fp in frame_paths:
                os.remove(fp)
            print(f"Cleaned up {len(frame_paths)} frame files")


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a static scene view at a given simulation timestep")
    parser.add_argument("path", type=str, help="Path to .pkl results file")
    parser.add_argument("--step", "-s", type=int, default=None,
                        help="Timestep to render (default: last step)")
    parser.add_argument("--episode", "-e", type=int, default=0,
                        help="Episode index for batch results (default: 0)")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Save figure to file instead of showing")
    parser.add_argument("--video", type=str, default=None,
                        help="Render all steps as video to this directory")
    parser.add_argument("--frames", type=str, default=None,
                        help="Render all steps as PNG frames only (no video)")
    parser.add_argument("--keep-frames", action="store_true",
                        help="Keep individual PNG frames after creating video")
    parser.add_argument("--no-milp", action="store_true",
                        help="Hide MILP rollout paths")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved images (default: 150)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 8],
                        metavar=("W", "H"),
                        help="Figure size in inches, e.g. --figsize 7 4 "
                             "(default: 14 8)")
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
        print(f"Loaded single episode result")
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
    ego_goal = _extract_ego_goal(result.config)

    print(f"Scenario: {result.scenario_name}  |  "
          f"Steps: {len(result.steps)}  |  "
          f"Solved: {result.solved}  |  Failed: {result.failed}")

    render_kwargs = dict(
        fps=result.fps,
        ego_goal=ego_goal,
        show_milp=not args.no_milp,
    )

    figsize = tuple(args.figsize)

    # Video mode: render all frames + assemble video
    if args.video is not None:
        print(f"Rendering {len(result.steps)} frames to {args.video}/")
        render_video_frames(result, scenario_map, args.video,
                            figsize=figsize, dpi=args.dpi,
                            keep_frames=args.keep_frames,
                            **render_kwargs)
        return

    # Frames-only mode: render all frames as PNGs, no video
    if args.frames is not None:
        print(f"Rendering {len(result.steps)} frames to {args.frames}/")
        render_video_frames(result, scenario_map, args.frames,
                            figsize=figsize, dpi=args.dpi,
                            keep_frames=True, video=False,
                            **render_kwargs)
        return

    # Single frame mode
    step_idx = args.step
    if step_idx is None:
        step_idx = len(result.steps) - 1
        print(f"No --step given, using last step ({step_idx})")
    elif step_idx >= len(result.steps):
        print(f"Step {step_idx} out of range (have {len(result.steps)} steps)")
        sys.exit(1)

    sr = result.steps[step_idx]
    print(f"Rendering step {sr.step} (index {step_idx})")

    fig, ax = create_scene_figure(sr, scenario_map, figsize=figsize,
                                  **render_kwargs)

    if args.out:
        fig.savefig(args.out, dpi=args.dpi)
        print(f"Saved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
