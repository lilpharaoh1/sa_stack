"""
Render scene views with dynamic agents coloured by human awareness (phi).

Agents the human is more aware of (phi -> 1) appear bright; agents the human
is less aware of (phi -> 0) appear dark.  Uses the ``human_awareness`` field
from StepRecord, which stores the ground-truth Kalman awareness phi per agent.

Supports single-step, multi-frame, and --frames output modes.

Usage:
    # Single step:
    python scripts/experiments/render_kscene.py results/my_run/ --step 20

    # All frames as PNGs:
    python scripts/experiments/render_kscene.py results/my_run/ --frames out/

    # Save single step:
    python scripts/experiments/render_kscene.py results/my_run/ -s 20 -o scene.png
"""

import sys
import os
import argparse
from typing import Optional, Tuple

import dill
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon as MplPolygon

_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", ".."))
sys.path.insert(0, _EXPERIMENTS_DIR)

import igp2 as ip
from igp2.opendrive.plot_map import plot_map
from igp2.core.util import calculate_multiple_bboxes
from igp2.beliefcontrol.frenet import FrenetFrame
from utils import ExperimentResult, StepRecord, RESULTS_DIR
from render_scene import _draw_speed_path

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOUR_EGO = (0.2, 0.4, 0.9)
COLOUR_DYNAMIC = (0.85, 0.75, 0.3)
COLOUR_STATIC = (0.6, 0.6, 0.6)
COLOUR_HUMAN_NLP = (0.9, 0.1, 0.1)
COLOUR_INTERVENTION = (0.0, 0.7, 0.0)


def _darken(colour, factor):
    """Darken an RGB colour by *factor* (0 = black, 1 = unchanged)."""
    return tuple(c * factor for c in colour[:3])


def _draw_vehicle(ax, position, heading, length, width, facecolor,
                  edgecolor, linewidth=1.5, zorder=5, alpha=1.0):
    """Draw a vehicle bounding box polygon."""
    corners = calculate_multiple_bboxes(
        [position[0]], [position[1]], length, width, heading)[0]
    poly = MplPolygon(corners, closed=True,
                      facecolor=(*facecolor[:3], 0.6 * alpha),
                      edgecolor=(*edgecolor[:3], 1.0 * alpha),
                      linewidth=linewidth, zorder=zorder)
    ax.add_patch(poly)
    return poly


def render_kscene_frame(step_record: StepRecord,
                        scenario_map: 'ip.Map',
                        ax: plt.Axes,
                        *,
                        fps: int = 10,
                        ego_goal: Optional['ip.BoxGoal'] = None,
                        margin: float = 40.0,
                        show_paths: bool = True,
                        show_intervention: bool = True,
                        show_legend: bool = True,
                        title: Optional[str] = None):
    """Render a scene frame with agents coloured by human awareness.

    Brightness scales with phi (awareness probability):
      phi = 1.0 (fully aware)  -> full colour
      phi = 0.0 (unaware)      -> dark (0.2 brightness)

    Args:
        step_record: StepRecord for the timestep.
        scenario_map: Parsed road map.
        ax: Target axes.
        fps: Simulation FPS.
        ego_goal: Optional ego goal box.
        margin: View margin around ego in metres.
        show_paths: Whether to draw planned paths.
        show_legend: Whether to show the legend.
        title: Optional title override.
    """
    sr = step_record

    # Centre on ego
    if sr.ego_position is not None:
        ax.set_xlim(sr.ego_position[0] - margin, sr.ego_position[0] + margin)
        ax.set_ylim(sr.ego_position[1] - margin, sr.ego_position[1] + margin)

    # --- Ego goal ---
    if ego_goal is not None:
        corners = np.array(ego_goal.box.boundary)
        goal_poly = MplPolygon(corners, closed=True,
                               facecolor=(*COLOUR_EGO, 0.08),
                               edgecolor=(*COLOUR_EGO, 0.4),
                               linewidth=1.5, linestyle=':', zorder=2)
        ax.add_patch(goal_poly)

    # --- Static obstacles ---
    for aid, state in sr.static_obstacles.items():
        meta = getattr(state, 'metadata', None)
        vl = meta.length if meta else 4.5
        vw = meta.width if meta else 1.8
        _draw_vehicle(ax, state.position, state.heading, vl, vw,
                      COLOUR_STATIC, COLOUR_STATIC,
                      linewidth=1.0, zorder=4)

    # --- Dynamic agents (coloured by human awareness phi) ---
    awareness = sr.human_awareness or {}
    for aid, state in sr.dynamic_agents.items():
        meta = getattr(state, 'metadata', None)
        vl = meta.length if meta else 4.5
        vw = meta.width if meta else 1.8

        # phi: 1.0 = fully aware (bright), 0.0 = unaware (dark)
        phi = awareness.get(aid, 1.0)
        brightness = 0.2 + 0.8 * phi  # range [0.2, 1.0]

        face = _darken(COLOUR_DYNAMIC, brightness)
        edge = _darken(COLOUR_DYNAMIC, brightness)

        _draw_vehicle(ax, state.position, state.heading, vl, vw,
                      face, edge, linewidth=1.5, zorder=5)

        # Heading arrow
        arrow_colour = _darken(COLOUR_DYNAMIC, brightness)
        arrow_len = 2.5
        dx = arrow_len * np.cos(state.heading)
        dy = arrow_len * np.sin(state.heading)
        ax.annotate("", xy=(state.position[0] + dx, state.position[1] + dy),
                    xytext=(state.position[0], state.position[1]),
                    arrowprops=dict(arrowstyle='->', color=arrow_colour,
                                   lw=1.5),
                    zorder=6)

        # Phi label
        ax.annotate(f"$\\phi$={phi:.2f}",
                    xy=(state.position[0], state.position[1] - 4.0),
                    fontsize=7, ha='center', color=face,
                    alpha=0.9, zorder=7)

    # --- Ego vehicle ---
    if sr.ego_position is not None:
        ego_heading = sr.ego_heading if sr.ego_heading is not None else 0.0
        _draw_vehicle(ax, sr.ego_position, ego_heading, 4.5, 1.8,
                      COLOUR_EGO, COLOUR_EGO,
                      linewidth=2.0, zorder=6)

        arrow_len = 3.0
        dx = arrow_len * np.cos(ego_heading)
        dy = arrow_len * np.sin(ego_heading)
        ax.annotate("", xy=(sr.ego_position[0] + dx, sr.ego_position[1] + dy),
                    xytext=(sr.ego_position[0], sr.ego_position[1]),
                    arrowprops=dict(arrowstyle='->', color=COLOUR_EGO, lw=2),
                    zorder=7)

        spd = sr.ego_speed if sr.ego_speed is not None else 0.0
        ax.annotate(f"ego {spd:.1f} m/s",
                    xy=(sr.ego_position[0], sr.ego_position[1] + 3.0),
                    fontsize=7, fontweight='bold', ha='center',
                    color=COLOUR_EGO, zorder=8)

    # --- Paths ---
    if show_paths:
        if sr.human_rollout is not None and len(sr.human_rollout) > 1:
            _draw_speed_path(ax, sr.human_rollout[:, :2],
                             sr.human_rollout[:, 3],
                             COLOUR_HUMAN_NLP, linewidth=2.5, zorder=5,
                             label='Human')

        if show_intervention:
            ref_wp = getattr(sr, 'reference_waypoints', None)
            interv_states = sr.intervention_opt_states
            if interv_states is not None and ref_wp is not None and len(ref_wp) >= 2:
                frenet = FrenetFrame(ref_wp)
                K = len(interv_states)
                world = np.empty((K, 2))
                for i in range(K):
                    s_i, d_i = interv_states[i, 0], interv_states[i, 1]
                    w = frenet.frenet_to_world(s_i, d_i)
                    world[i] = [w['x'], w['y']]
                interv_speeds = interv_states[:, 3]
                _draw_speed_path(ax, world, interv_speeds,
                                 COLOUR_INTERVENTION, linewidth=2.0, zorder=6,
                                 label='Intervention')

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
        ]
        if show_paths:
            from matplotlib.lines import Line2D
            legend_handles.append(
                Line2D([0], [0], color=COLOUR_HUMAN_NLP, linewidth=2,
                       label='Human'))
            if show_intervention:
                legend_handles.append(
                    Line2D([0], [0], color=COLOUR_INTERVENTION, linewidth=2,
                           label='Intervention'))
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
                  framealpha=0.9)

    # --- Title ---
    if title is not None:
        ax.set_title(title, fontsize=11)
    else:
        sim_time = sr.step / fps
        parts = [f"step={sr.step}", f"t={sim_time:.1f}s"]
        if sr.ego_speed is not None:
            parts.append(f"v={sr.ego_speed:.1f}m/s")
        ax.set_title("  |  ".join(parts), fontsize=11)


def create_kscene_figure(step_record: StepRecord,
                         scenario_map: 'ip.Map',
                         figsize: Tuple[float, float] = (14, 8),
                         **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with map, awareness-coloured agents, and colorbar."""
    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes([0.02, 0.02, 0.82, 0.93])
    cbar_ax = fig.add_axes([0.86, 0.15, 0.03, 0.65])

    plot_map(scenario_map, ax=ax, markings=True,
             junction_color=(0, 0, 0, 0))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    render_kscene_frame(step_record, scenario_map, ax, **kwargs)

    # --- Awareness (phi) colour bar ---
    n_steps = 256
    phi_vals = np.linspace(0, 1, n_steps)
    colours_list = []
    for phi in phi_vals:
        b = 0.2 + 0.8 * phi
        colours_list.append((*_darken(COLOUR_DYNAMIC, b), 1.0))
    awareness_cmap = mcolors.ListedColormap(colours_list)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(cmap=awareness_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.set_ylabel(r"Human awareness $\phi$", fontsize=9)

    return fig, ax


def _extract_ego_goal(config: dict) -> Optional['ip.BoxGoal']:
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


def _load_result(path: str, episode: int) -> ExperimentResult:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Render scene with agents coloured by human awareness")
    p.add_argument("path", type=str,
                   help="Run directory or .pkl results file")
    p.add_argument("--step", "-s", type=int, default=None,
                   help="Single step to render (default: last step)")
    p.add_argument("--episode", "-e", type=int, default=0,
                   help="Episode index for batch results (default: 0)")
    p.add_argument("--frames", type=str, default=None,
                   help="Render all steps as PNG frames to this directory")
    p.add_argument("--out", "-o", type=str, default=None,
                   help="Save single-step figure to file")
    p.add_argument("--no-paths", action="store_true",
                   help="Hide planned paths")
    p.add_argument("--no-intervention", action="store_true",
                   help="Hide the intervention path")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--figsize", type=float, nargs=2, default=[14, 8],
                   metavar=("W", "H"))
    p.add_argument("--margin", type=float, default=40.0,
                   help="View margin around ego in metres (default: 40)")
    return p.parse_args()


def main():
    args = parse_args()

    result = _load_result(args.path, args.episode)

    if not result.steps:
        print("No step records.")
        sys.exit(1)

    map_path = result.config.get("scenario", {}).get("map_path")
    if not map_path:
        print("No map_path in config.")
        sys.exit(1)
    scenario_map = ip.Map.parse_from_opendrive(map_path)
    ego_goal = _extract_ego_goal(result.config)

    print(f"Scenario: {result.scenario_name}  |  "
          f"Steps: {len(result.steps)}  |  "
          f"Solved: {result.solved}  |  Failed: {result.failed}")

    figsize = tuple(args.figsize)
    render_kwargs = dict(
        fps=result.fps,
        ego_goal=ego_goal,
        show_paths=not args.no_paths,
        show_intervention=not args.no_intervention,
        margin=args.margin,
    )

    # --- All frames mode ---
    if args.frames is not None:
        os.makedirs(args.frames, exist_ok=True)
        n = len(result.steps)
        for i, sr in enumerate(result.steps):
            fig, ax = create_kscene_figure(sr, scenario_map,
                                           figsize=figsize, **render_kwargs)
            frame_path = os.path.join(args.frames, f"frame_{i:04d}.png")
            fig.savefig(frame_path, dpi=args.dpi)
            plt.close(fig)
            if (i + 1) % 10 == 0 or (i + 1) == n:
                print(f"  Rendered {i + 1}/{n} frames")
        print(f"Frames saved to {args.frames}/")
        return

    # --- Single step mode ---
    step_idx = args.step
    if step_idx is None:
        step_idx = len(result.steps) - 1
        print(f"No --step given, using last step ({step_idx})")
    elif step_idx >= len(result.steps):
        print(f"Step {step_idx} out of range (have {len(result.steps)})")
        sys.exit(1)

    sr = result.steps[step_idx]
    print(f"Rendering step {sr.step} (index {step_idx})")

    # Print awareness info
    if sr.human_awareness:
        for aid, phi in sr.human_awareness.items():
            print(f"  Agent {aid}: phi={phi:.3f}")

    fig, ax = create_kscene_figure(sr, scenario_map,
                                   figsize=figsize, **render_kwargs)

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
