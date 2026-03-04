"""
Plot the full episode trajectory on the road layout.

- Non-ego agents: trajectory points coloured by P(hidden) darkening
  (same scheme as belief_snapshot).
- Ego agent: trajectory points coloured with coolwarm — red when human
  is driving, blue when intervention is active.  Saturation scales with
  velocity (faster = more saturated).
- Static obstacles drawn as grey polygons.

Usage:
    python scripts/experiments/plot_intervention_scene.py results/my_run.pkl -o scene.png
    python scripts/experiments/plot_intervention_scene.py results/my_run.pkl -e 2
"""

import sys
import os
import argparse
from typing import Optional, Tuple

import dill
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
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

CMAP_COOLWARM = cm.get_cmap('coolwarm')


def _darken(colour, factor):
    """Darken an RGB colour by *factor* (0 = black, 1 = unchanged)."""
    return tuple(c * factor for c in colour[:3])


def _brightness_from_p_hidden(p_hidden):
    """Same darkening rule as belief_snapshot."""
    if p_hidden <= 0.5:
        return 1.0
    else:
        return 1.0 - 0.7 * (p_hidden - 0.5) / 0.5


def plot_intervention_scene(result: ExperimentResult,
                            scenario_map: 'ip.Map',
                            figsize: Tuple[float, float] = (14, 8),
                            margin: Optional[float] = None,
                            point_size: float = 8.0,
                            ego_goal: Optional['ip.BoxGoal'] = None,
                            ) -> Tuple[plt.Figure, plt.Axes]:
    """Plot full-episode trajectories on the road layout.

    Args:
        result: Experiment result with step records.
        scenario_map: Parsed road map.
        figsize: Figure size.
        margin: View margin around trajectory extent. None = auto-fit.
        point_size: Marker size for trajectory points.
        ego_goal: Optional ego goal box.

    Returns:
        (fig, ax) tuple.
    """
    steps = result.steps

    # --- Collect ego trajectory ---
    ego_xy = []
    ego_speeds = []
    ego_intervening = []
    for sr in steps:
        if sr.ego_position is not None:
            ego_xy.append(sr.ego_position)
            ego_speeds.append(sr.ego_speed if sr.ego_speed is not None else 0.0)
            ego_intervening.append(sr.intervention_active)
    ego_xy = np.array(ego_xy)  # (N, 2)
    ego_speeds = np.array(ego_speeds)
    ego_intervening = np.array(ego_intervening)

    # --- Collect non-ego agent trajectories ---
    # {aid: {'xy': [(x,y), ...], 'p_hidden': [float, ...]}}
    agent_data = {}
    for sr in steps:
        marginals = sr.belief_marginals or {}
        for aid, state in sr.dynamic_agents.items():
            if aid not in agent_data:
                agent_data[aid] = {'xy': [], 'p_hidden': []}
            agent_data[aid]['xy'].append(state.position)
            agent_data[aid]['p_hidden'].append(marginals.get(aid, 0.0))

    for aid in agent_data:
        agent_data[aid]['xy'] = np.array(agent_data[aid]['xy'])
        agent_data[aid]['p_hidden'] = np.array(agent_data[aid]['p_hidden'])

    # --- Figure ---
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.02, 0.02, 0.82, 0.93])
    cbar_ax = fig.add_axes([0.86, 0.15, 0.015, 0.30])
    hidden_cbar_ax = fig.add_axes([0.86, 0.55, 0.015, 0.30])

    plot_map(scenario_map, ax=ax, markings=True,
             junction_color=(0, 0, 0, 0))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # --- View bounds ---
    if margin is not None and len(ego_xy) > 0:
        cx = (ego_xy[:, 0].min() + ego_xy[:, 0].max()) / 2
        cy = (ego_xy[:, 1].min() + ego_xy[:, 1].max()) / 2
        span = max(ego_xy[:, 0].ptp(), ego_xy[:, 1].ptp()) / 2 + margin
        ax.set_xlim(cx - span, cx + span)
        ax.set_ylim(cy - span, cy + span)
    elif len(ego_xy) > 0:
        pad = 15.0
        all_x = list(ego_xy[:, 0])
        all_y = list(ego_xy[:, 1])
        for ad in agent_data.values():
            all_x.extend(ad['xy'][:, 0])
            all_y.extend(ad['xy'][:, 1])
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_y) - pad, max(all_y) + pad)

    # --- Ego goal ---
    if ego_goal is not None:
        corners = np.array(ego_goal.box.boundary)
        goal_poly = MplPolygon(corners, closed=True,
                               facecolor=(*COLOUR_EGO, 0.08),
                               edgecolor=(*COLOUR_EGO, 0.4),
                               linewidth=1.5, linestyle=':', zorder=2)
        ax.add_patch(goal_poly)

    # --- Static obstacles (from last step that has them) ---
    last_sr = steps[-1]
    for aid, state in last_sr.static_obstacles.items():
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

    # --- Non-ego agent trajectories ---
    for aid, ad in agent_data.items():
        xy = ad['xy']
        ph = ad['p_hidden']
        for j in range(len(xy)):
            b = _brightness_from_p_hidden(ph[j])
            c = _darken(COLOUR_DYNAMIC, b)
            ax.plot(xy[j, 0], xy[j, 1], 'o',
                    color=(*c, 0.7), markersize=point_size * 0.6,
                    zorder=3)

    # --- Ego trajectory ---
    if len(ego_xy) > 0:
        # Normalise speed to [0, 1]
        v_max = max(ego_speeds.max(), 1e-3)
        v_norm = ego_speeds / v_max

        for j in range(len(ego_xy)):
            if ego_intervening[j]:
                # Blue half: high speed → 0.05 (deep blue), low → 0.45 (pale)
                cval = 0.45 - v_norm[j] * 0.40
            else:
                # Red half: high speed → 0.95 (deep red), low → 0.55 (pale)
                cval = 0.55 + v_norm[j] * 0.40

            colour = CMAP_COOLWARM(cval)
            ax.plot(ego_xy[j, 0], ego_xy[j, 1], 'o',
                    color=colour, markersize=point_size,
                    zorder=5)

    # --- Ego velocity colourbar ---
    # Two-part colourbar: red half for human, blue half for intervention
    ego_norm = mcolors.Normalize(vmin=0, vmax=v_max)

    # Red (human) colourbar
    red_colours = [CMAP_COOLWARM(0.55 + t * 0.40) for t in np.linspace(0, 1, 256)]
    red_cmap = mcolors.ListedColormap(red_colours)
    sm_red = ScalarMappable(cmap=red_cmap, norm=ego_norm)
    sm_red.set_array([])
    cb_red = fig.colorbar(sm_red, cax=cbar_ax)
    cbar_ax.set_ylabel("Ego speed — human [m/s]", fontsize=8)

    # Blue (intervention) colourbar
    blue_colours = [CMAP_COOLWARM(0.45 - t * 0.40) for t in np.linspace(0, 1, 256)]
    blue_cmap = mcolors.ListedColormap(blue_colours)
    sm_blue = ScalarMappable(cmap=blue_cmap, norm=ego_norm)
    sm_blue.set_array([])
    cb_blue = fig.colorbar(sm_blue, cax=hidden_cbar_ax)
    hidden_cbar_ax.set_ylabel("Ego speed — intervention [m/s]", fontsize=8)

    return fig, ax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot full-episode intervention trajectory scene")
    parser.add_argument("path", type=str, help="Path to .pkl results file")
    parser.add_argument("--episode", "-e", type=int, default=0,
                        help="Episode index for batch results (default: 0)")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Save figure to file instead of showing")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved images (default: 150)")
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 8],
                        metavar=("W", "H"),
                        help="Figure size in inches (default: 14 8)")
    parser.add_argument("--margin", type=float, default=None,
                        help="View margin in metres (default: auto-fit)")
    parser.add_argument("--point-size", type=float, default=8.0,
                        help="Marker size for trajectory points (default: 8)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.path):
        print(f"File not found: {args.path}")
        sys.exit(1)

    with open(args.path, 'rb') as f:
        data = dill.load(f)

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

    map_path = result.config.get("scenario", {}).get("map_path")
    if map_path is None:
        print("No map_path in config, cannot render.")
        sys.exit(1)

    scenario_map = ip.Map.parse_from_opendrive(map_path)
    ego_goal = _extract_ego_goal(result.config)

    n_intervening = sum(1 for sr in result.steps if sr.intervention_active)
    print(f"Scenario: {result.scenario_name}  |  "
          f"Steps: {len(result.steps)}  |  "
          f"Intervening: {n_intervening}/{len(result.steps)}  |  "
          f"Solved: {result.solved}  |  Failed: {result.failed}")

    fig, ax = plot_intervention_scene(
        result, scenario_map,
        figsize=tuple(args.figsize),
        margin=args.margin,
        point_size=args.point_size,
        ego_goal=ego_goal,
    )

    if args.out:
        fig.savefig(args.out, dpi=args.dpi)
        print(f"Saved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
