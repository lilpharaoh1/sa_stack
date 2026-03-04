"""
Render a belief-aware snapshot at a given simulation timestep.

Produces two figures:
  1. Scene plot: road layout, ego, agents darkened by P(hidden)
  2. Control dial: arrow showing the intervention's acceleration and
     steering adjustments on centred axes with a halfway guide circle.

Usage:
    # Show at step 30
    python scripts/experiments/belief_snapshot.py results/my_run.pkl --step 30

    # Save to files (scene.png + scene_dial.png)
    python scripts/experiments/belief_snapshot.py results/my_run.pkl -s 30 -o scene.png

    # Batch results, episode 2
    python scripts/experiments/belief_snapshot.py results/batch.pkl -s 30 -e 2
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
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip
from igp2.opendrive.plot_map import plot_map
from igp2.core.util import calculate_multiple_bboxes
from belief_utils import ExperimentResult, StepRecord

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOUR_EGO = (0.2, 0.4, 0.9)
COLOUR_DYNAMIC = (0.85, 0.75, 0.3)
COLOUR_STATIC = (0.6, 0.6, 0.6)

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


def render_belief_snapshot(step_record: StepRecord,
                           scenario_map: 'ip.Map',
                           ax: plt.Axes,
                           *,
                           fps: int = 10,
                           ego_goal: Optional['ip.BoxGoal'] = None,
                           margin: float = 40.0,
                           show_legend: bool = True,
                           title: Optional[str] = None):
    """Render a single belief-aware snapshot onto *ax*.

    Args:
        step_record: StepRecord for the timestep.
        scenario_map: Parsed road map.
        ax: Target axes.
        fps: Simulation FPS.
        ego_goal: Optional ego goal box.
        margin: View margin around the ego in metres.
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

    # --- Dynamic agents (darkened by P(hidden)) ---
    marginals = sr.belief_marginals or {}
    for aid, state in sr.dynamic_agents.items():
        meta = getattr(state, 'metadata', None)
        vl = meta.length if meta else 4.5
        vw = meta.width if meta else 1.8

        p_hidden = marginals.get(aid, 0.0)

        # Darkening: 0.0-0.5 → full colour; 0.5-1.0 → darken linearly to 0.3
        if p_hidden <= 0.5:
            brightness = 1.0
        else:
            brightness = 1.0 - 0.7 * (p_hidden - 0.5) / 0.5  # 1.0 → 0.3

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

    # --- Ego vehicle ---
    if sr.ego_position is not None:
        ego_heading = sr.ego_heading if sr.ego_heading is not None else 0.0
        _draw_vehicle(ax, sr.ego_position, ego_heading, 4.5, 1.8,
                      COLOUR_EGO, COLOUR_EGO,
                      linewidth=2.0, zorder=6)

        # Heading arrow
        arrow_len = 3.0
        dx = arrow_len * np.cos(ego_heading)
        dy = arrow_len * np.sin(ego_heading)
        ax.annotate("", xy=(sr.ego_position[0] + dx, sr.ego_position[1] + dy),
                    xytext=(sr.ego_position[0], sr.ego_position[1]),
                    arrowprops=dict(arrowstyle='->', color=COLOUR_EGO, lw=2),
                    zorder=7)

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


def create_belief_snapshot(step_record: StepRecord,
                           scenario_map: 'ip.Map',
                           figsize: Tuple[float, float] = (14, 8),
                           **kwargs) -> Tuple[plt.Figure, plt.Axes]:
    """Create a complete figure with map and belief snapshot.

    Returns:
        (fig, ax) tuple.
    """
    fig = plt.figure(figsize=figsize)

    # Fixed axes position: leave a thin strip on the right for the colorbar
    ax = fig.add_axes([0.02, 0.02, 0.82, 0.93])  # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.86, 0.15, 0.03, 0.65])

    plot_map(scenario_map, ax=ax, markings=True,
             junction_color=(0, 0, 0, 0))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_clip_on(True)
    render_belief_snapshot(step_record, scenario_map, ax, **kwargs)

    # --- P(hidden) colour bar ---
    n_steps = 256
    p_vals = np.linspace(0, 1, n_steps)
    colours_list = []
    for p in p_vals:
        if p <= 0.5:
            b = 1.0
        else:
            b = 1.0 - 0.7 * (p - 0.5) / 0.5
        colours_list.append((*_darken(COLOUR_DYNAMIC, b), 1.0))
    hidden_cmap = mcolors.ListedColormap(colours_list)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(cmap=hidden_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.set_ylabel("P(hidden)", fontsize=9)

    return fig, ax


def render_control_dial(step_record: StepRecord, ax: plt.Axes):
    """Render the control-adjustment dial onto *ax*.

    Draws centred axes with guide ellipses and the intervention line.
    """
    sr = step_record

    # Axis ranges
    steer_lim = 0.45
    accel_lim = 3.0
    ax.set_xlim(-steer_lim, steer_lim)
    ax.set_ylim(-accel_lim, accel_lim)

    # Centre cross-hairs
    ax.axhline(0, color='grey', linewidth=0.5, zorder=1)
    ax.axvline(0, color='grey', linewidth=0.5, zorder=1)

    # Halfway guide ellipse
    half_ellipse = Ellipse((0, 0),
                           width=2 * steer_lim * 0.5,
                           height=2 * accel_lim * 0.5,
                           facecolor='none',
                           edgecolor='grey',
                           linestyle=':',
                           linewidth=1.5,
                           alpha=0.6,
                           zorder=2)
    ax.add_patch(half_ellipse)

    # Outer boundary ellipse
    outer_ellipse = Ellipse((0, 0),
                            width=2 * steer_lim,
                            height=2 * accel_lim,
                            facecolor='none',
                            edgecolor='grey',
                            linestyle='-',
                            linewidth=1.0,
                            alpha=0.3,
                            zorder=2)
    ax.add_patch(outer_ellipse)

    # Intervention line
    has_intervention = (sr.intervention_controls is not None
                        and len(sr.intervention_controls) > 0)

    if has_intervention:
        da = sr.intervention_controls[0, 0]
        ddelta = sr.intervention_controls[0, 1]
        colour = COLOUR_EGO

        ax.plot([0, ddelta], [0, da], '-', color=colour, lw=2.5, zorder=5)
        ax.plot(ddelta, da, 'o', color=colour, markersize=5, zorder=6)
    else:
        ax.plot(0, 0, 'o', color=(0.5, 0.5, 0.5), markersize=6, zorder=5)

    ax.set_xlabel(r"Steering adjustment [rad]", fontsize=9)
    ax.set_ylabel(r"Acceleration adjustment [m/s$^2$]", fontsize=9)

    status = "ACTIVE" if sr.intervention_active else "inactive"
    ax.set_title(f"Intervention control  (step {sr.step}, {status})",
                 fontsize=10)

    ax.grid(True, alpha=0.2)


def create_control_dial(step_record: StepRecord,
                        figsize: Tuple[float, float] = (5, 5),
                        ) -> Tuple[plt.Figure, plt.Axes]:
    """Create a standalone control-adjustment dial figure."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    render_control_dial(step_record, ax)
    fig.tight_layout()
    return fig, ax


def create_combined_snapshot(step_record: StepRecord,
                             scenario_map: 'ip.Map',
                             figsize: Tuple[float, float] = (14, 7),
                             **kwargs) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Create a figure with scene (left) and control dial (right) side by side.

    Returns:
        (fig, (ax_scene, ax_dial)) tuple.
    """
    fig = plt.figure(figsize=figsize)

    # Scene takes ~70% of width, dial takes ~25%, small gaps
    ax_scene = fig.add_axes([0.01, 0.02, 0.62, 0.93])
    cbar_ax = fig.add_axes([0.64, 0.15, 0.015, 0.65])
    ax_dial = fig.add_axes([0.72, 0.08, 0.26, 0.84])

    # --- Scene ---
    plot_map(scenario_map, ax=ax_scene, markings=True,
             junction_color=(0, 0, 0, 0))
    ax_scene.set_aspect('equal')
    ax_scene.set_xticks([])
    ax_scene.set_yticks([])
    ax_scene.set_clip_on(True)
    render_belief_snapshot(step_record, scenario_map, ax_scene, **kwargs)

    # P(hidden) colour bar
    n_steps = 256
    p_vals = np.linspace(0, 1, n_steps)
    colours_list = []
    for p in p_vals:
        if p <= 0.5:
            b = 1.0
        else:
            b = 1.0 - 0.7 * (p - 0.5) / 0.5
        colours_list.append((*_darken(COLOUR_DYNAMIC, b), 1.0))
    hidden_cmap = mcolors.ListedColormap(colours_list)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm = ScalarMappable(cmap=hidden_cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.set_ylabel("P(hidden)", fontsize=8)

    # --- Dial ---
    render_control_dial(step_record, ax_dial)

    return fig, (ax_scene, ax_dial)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a belief-aware snapshot at a given timestep")
    parser.add_argument("path", type=str, help="Path to .pkl results file")
    parser.add_argument("--step", "-s", type=int, default=None,
                        help="Timestep to render (default: last step)")
    parser.add_argument("--episode", "-e", type=int, default=0,
                        help="Episode index for batch results (default: 0)")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Save figure to file instead of showing")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved images (default: 150)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[8, 8],
                        metavar=("W", "H"),
                        help="Figure size in inches (default: 14 8)")
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

    # Select step
    step_idx = args.step
    if step_idx is None:
        step_idx = len(result.steps) - 1
        print(f"No --step given, using last step ({step_idx})")
    elif step_idx >= len(result.steps):
        print(f"Step {step_idx} out of range (have {len(result.steps)} steps)")
        sys.exit(1)

    sr = result.steps[step_idx]
    print(f"Rendering step {sr.step} (index {step_idx})")

    # Print belief info
    if sr.belief_marginals:
        for aid, p in sr.belief_marginals.items():
            gt = sr.belief_ground_truth.get(aid, "?") if sr.belief_ground_truth else "?"
            print(f"  Agent {aid}: P(hidden)={p:.3f}  ground_truth_visible={gt}")
    if sr.intervention_active:
        dev = sr.action_deviation if sr.action_deviation is not None else 0.0
        print(f"  Intervention ACTIVE (deviation={dev:.4f})")

    figsize = tuple(args.figsize)

    # --- Scene plot ---
    fig_scene, ax_scene = create_belief_snapshot(
        sr, scenario_map,
        figsize=figsize,
        fps=result.fps,
        ego_goal=ego_goal,
        margin=args.margin,
    )

    # --- Control dial plot ---
    fig_dial, ax_dial = create_control_dial(sr)

    if args.out:
        # Save scene
        fig_scene.savefig(args.out, dpi=args.dpi)
        print(f"Saved scene to {args.out}")
        plt.close(fig_scene)

        # Save dial alongside: foo.png → foo_dial.png
        base, ext = os.path.splitext(args.out)
        dial_path = f"{base}_dial{ext}"
        fig_dial.savefig(dial_path, dpi=args.dpi)
        print(f"Saved dial  to {dial_path}")
        plt.close(fig_dial)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
