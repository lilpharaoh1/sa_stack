"""
Plot aggregated intervention magnitude across all episodes in an experiment.

All ego trajectory points from every episode are overlaid on the road map,
coloured by acceleration deviation — |a_executed - a_human|.  Since the
ego follows the same reference path in every episode, points cluster along
the road — denser/darker regions indicate where interventions concentrate.

Usage:
    python scripts/experiments/plot_intervention_heatmap.py results/belief_experiment4_naive_agency_only_s21_n5/
    python scripts/experiments/plot_intervention_heatmap.py results/my_run/ -o heatmap.png
    python scripts/experiments/plot_intervention_heatmap.py run1/ run2/ --labels "Greedy" "QCBF"
"""

import sys
import os
import argparse
from typing import Optional, Tuple, List

import dill
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip
from igp2.opendrive.plot_map import plot_map
from igp2.core.util import calculate_multiple_bboxes
from belief_utils import ExperimentResult, StepRecord, RESULTS_DIR

COLOUR_STATIC = (0.6, 0.6, 0.6)
COLOUR_EGO = (0.2, 0.4, 0.9)


def _load_run(run_dir: str) -> List[ExperimentResult]:
    """Load all episodes from a run directory."""
    pkl_path = os.path.join(run_dir, "results.pkl")
    if not os.path.exists(pkl_path):
        # Maybe it's just a name, try RESULTS_DIR
        pkl_path = os.path.join(RESULTS_DIR, run_dir, "results.pkl")
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    elif isinstance(data, ExperimentResult):
        return [data]
    return []


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


def plot_intervention_heatmap(
        episodes: List[ExperimentResult],
        scenario_map: 'ip.Map',
        figsize: Tuple[float, float] = (14, 8),
        point_size: float = 6.0,
        alpha: float = 0.4,
        ego_goal: Optional['ip.BoxGoal'] = None,
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        vmax: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes, float]:
    """Plot aggregated ego trajectory points coloured by intervention magnitude.

    Args:
        episodes: List of ExperimentResult from one run.
        scenario_map: Parsed road map.
        figsize: Figure size (only used if ax is None).
        point_size: Marker size.
        alpha: Point transparency (lower = better for many overlapping points).
        ego_goal: Optional ego goal box.
        title: Optional plot title.
        ax: Optional existing axes to draw on.
        vmax: Optional shared max for colour normalisation.

    Returns:
        (fig, ax, mag_max) tuple.
    """
    # Collect all ego positions and acceleration deviations
    # Uses |a_executed - a_human| (same as plot_comparison accel deviation)
    ego_xy = []
    ego_magnitudes = []
    for result in episodes:
        for sr in result.steps:
            if sr.ego_position is not None:
                ego_xy.append(sr.ego_position)
                if sr.action_deviation_accel is not None:
                    mag = sr.action_deviation_accel
                else:
                    mag = 0.0
                ego_magnitudes.append(mag)

    ego_xy = np.array(ego_xy) if ego_xy else np.empty((0, 2))
    ego_magnitudes = np.array(ego_magnitudes)

    # Collect static obstacles from the first episode's last step
    static_obs = {}
    if episodes and episodes[0].steps:
        static_obs = episodes[0].steps[-1].static_obstacles

    # Figure
    own_fig = ax is None
    if own_fig:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.02, 0.02, 0.82, 0.93])
    else:
        fig = ax.figure

    plot_map(scenario_map, ax=ax, markings=True,
             junction_color=(0, 0, 0, 0))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # View bounds
    if len(ego_xy) > 0:
        pad = 15.0
        ax.set_xlim(ego_xy[:, 0].min() - pad, ego_xy[:, 0].max() + pad)
        ax.set_ylim(ego_xy[:, 1].min() - pad, ego_xy[:, 1].max() + pad)

    # Ego goal
    if ego_goal is not None:
        corners = np.array(ego_goal.box.boundary)
        goal_poly = MplPolygon(corners, closed=True,
                               facecolor=(*COLOUR_EGO, 0.08),
                               edgecolor=(*COLOUR_EGO, 0.4),
                               linewidth=1.5, linestyle=':', zorder=2)
        ax.add_patch(goal_poly)

    # Static obstacles
    for aid, state in static_obs.items():
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

    # Ego trajectory points coloured by magnitude
    mag_max = vmax if vmax is not None else max(ego_magnitudes.max(), 1e-6) if len(ego_magnitudes) > 0 else 1e-6
    if len(ego_xy) > 0:
        cmap = plt.cm.Purples
        norm = mcolors.Normalize(vmin=0.0, vmax=mag_max)
        # Sort by magnitude so high-magnitude points are drawn on top
        order = np.argsort(ego_magnitudes)
        ax.scatter(ego_xy[order, 0], ego_xy[order, 1],
                   c=ego_magnitudes[order], cmap=cmap, norm=norm,
                   s=point_size, alpha=alpha, edgecolors='none', zorder=5)

    if title:
        ax.set_title(title, fontsize=11)

    # Stats annotation (mean over all steps, consistent with plot_comparison)
    n_points = len(ego_magnitudes)
    n_intervening = int((ego_magnitudes > 1e-8).sum()) if n_points > 0 else 0
    n_episodes = len(episodes)
    stats_text = (f"{n_episodes} episodes, {n_points} points\n"
                  f"{n_intervening} deviated ({100*n_intervening/max(n_points,1):.1f}%)")
    if n_points > 0:
        stats_text += f"\nmean={ego_magnitudes.mean():.4f}, max={ego_magnitudes.max():.4f}"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=7, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
            zorder=10)

    return fig, ax, mag_max


def main():
    args = parse_args()

    run_dirs = args.dirs
    # Resolve directory names
    resolved = []
    for d in run_dirs:
        if os.path.isdir(d):
            resolved.append(d)
        elif os.path.isdir(os.path.join(RESULTS_DIR, d)):
            resolved.append(os.path.join(RESULTS_DIR, d))
        else:
            print(f"Warning: directory not found: {d}")
    run_dirs = resolved

    if not run_dirs:
        print("No valid run directories.")
        sys.exit(1)

    # Load episodes and determine map from first run
    all_runs = []
    all_labels = []
    scenario_map = None
    ego_goal = None

    for i, run_dir in enumerate(run_dirs):
        episodes = _load_run(run_dir)
        if not episodes:
            print(f"Warning: no episodes in {run_dir}")
            continue

        if scenario_map is None:
            map_path = episodes[0].config.get("scenario", {}).get("map_path")
            if map_path:
                scenario_map = ip.Map.parse_from_opendrive(map_path)
            ego_goal = _extract_ego_goal(episodes[0].config)

        all_runs.append(episodes)
        if args.labels and i < len(args.labels):
            all_labels.append(args.labels[i])
        else:
            all_labels.append(os.path.basename(run_dir.rstrip('/')))

    if not all_runs or scenario_map is None:
        print("No data to plot.")
        sys.exit(1)

    # Compute shared vmax across all runs for consistent colour scale
    global_max = 0.0
    for episodes in all_runs:
        for result in episodes:
            for sr in result.steps:
                if sr.action_deviation_accel is not None:
                    global_max = max(global_max, sr.action_deviation_accel)
    global_max = max(global_max, 1e-6)

    n_runs = len(all_runs)
    if n_runs == 1:
        fig, ax, _ = plot_intervention_heatmap(
            all_runs[0], scenario_map,
            figsize=tuple(args.figsize),
            point_size=args.point_size,
            alpha=args.alpha,
            ego_goal=ego_goal,
            title=all_labels[0],
            vmax=global_max,
        )
        # Add colourbar
        cbar_ax = fig.add_axes([0.86, 0.15, 0.015, 0.65])
        norm = mcolors.Normalize(vmin=0.0, vmax=global_max)
        sm = ScalarMappable(cmap=plt.cm.Purples, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
        cbar_ax.set_ylabel("Acceleration deviation (m/s²)", fontsize=9)
    else:
        fig, axes = plt.subplots(1, n_runs, figsize=(args.figsize[0] * n_runs / 2, args.figsize[1]))
        if n_runs == 1:
            axes = [axes]
        for i, (episodes, label) in enumerate(zip(all_runs, all_labels)):
            plot_intervention_heatmap(
                episodes, scenario_map,
                point_size=args.point_size,
                alpha=args.alpha,
                ego_goal=ego_goal,
                title=label,
                ax=axes[i],
                vmax=global_max,
            )
        # Shared colourbar
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
        norm = mcolors.Normalize(vmin=0.0, vmax=global_max)
        sm = ScalarMappable(cmap=plt.cm.Purples, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax)
        cbar_ax.set_ylabel("Acceleration deviation (m/s²)", fontsize=9)

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot aggregated intervention magnitude across all episodes")
    parser.add_argument("dirs", nargs="+",
                        help="Run directories (or names under results/)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each run directory (for multi-panel)")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Save figure to file instead of showing")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 8],
                        metavar=("W", "H"))
    parser.add_argument("--point-size", type=float, default=6.0,
                        help="Marker size (default: 6)")
    parser.add_argument("--alpha", type=float, default=0.4,
                        help="Point transparency (default: 0.4)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
