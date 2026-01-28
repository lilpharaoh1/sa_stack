"""Debug visualization for MacroGuidedRecognition.

Plots road layout with candidate trajectories colored by maneuver type,
the optimal trajectory, and the ego's observed history.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from math import ceil
from typing import Dict, List, Optional

from igp2.core.trajectory import Trajectory, VelocityTrajectory
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.planlibrary.maneuver import Maneuver

# Maneuver type name -> color
MANEUVER_COLORS = {
    "FollowLane": "green",
    "Turn": "blue",
    "GiveWay": "orange",
    "Stop": "red",
    "SwitchLaneLeft": "purple",
    "SwitchLaneRight": "magenta",
}
DEFAULT_COLOR = "cyan"

# Zoom margin around ego position (meters)
ZOOM_MARGIN = 50


def _maneuver_color(maneuver: Maneuver) -> str:
    """Get the plot color for a maneuver based on its type name."""
    name = type(maneuver).__name__
    return MANEUVER_COLORS.get(name, DEFAULT_COLOR)


def _plot_maneuver_segments(ax: plt.Axes, plan: List[Maneuver],
                            linestyle: str = "-", linewidth: float = 1.5,
                            alpha: float = 1.0, seen_types: set = None):
    """Plot a plan's maneuver segments with per-type coloring.

    Args:
        ax: Axes to plot on.
        plan: List of Maneuver objects, each with a .trajectory attribute.
        linestyle: Line style ('-' for solid, '--' for dashed).
        linewidth: Line width.
        alpha: Line transparency.
        seen_types: Set to track which maneuver types have been plotted
                    (for legend deduplication). Updated in-place.
    """
    if seen_types is None:
        seen_types = set()

    for maneuver in plan:
        if maneuver.trajectory is None or len(maneuver.trajectory.path) < 2:
            continue
        color = _maneuver_color(maneuver)
        name = type(maneuver).__name__
        path = maneuver.trajectory.path

        # Only add label for the first occurrence of each type (for legend)
        label = name if name not in seen_types else None
        seen_types.add(name)

        ax.plot(path[:, 0], path[:, 1],
                color=color, linestyle=linestyle, linewidth=linewidth,
                alpha=alpha, label=label)


def _setup_subplot(ax: plt.Axes, scenario_map: Map, ego_pos: np.ndarray,
                   goal: Goal, observed_trajectory: Optional[Trajectory]):
    """Set up a subplot with road layout, zoom, ego history, and goal marker.

    Args:
        ax: Axes to set up.
        scenario_map: Map to plot.
        ego_pos: Ego vehicle position for zoom centering.
        goal: Goal to mark.
        observed_trajectory: Ego observed trajectory (plotted in grey).
    """
    plot_map(scenario_map, markings=True, ax=ax)

    # Zoom to ego region
    ax.set_xlim(ego_pos[0] - ZOOM_MARGIN, ego_pos[0] + ZOOM_MARGIN)
    ax.set_ylim(ego_pos[1] - ZOOM_MARGIN, ego_pos[1] + ZOOM_MARGIN)

    # Plot ego observed trajectory
    # if observed_trajectory is not None and len(observed_trajectory.path) > 1:
    #     ax.plot(observed_trajectory.path[:, 0], observed_trajectory.path[:, 1],
    #             color="grey", linewidth=2, alpha=0.8, label="Observed")

    # Plot goal marker
    ax.plot(*goal.center, marker="x", color="black", markersize=10,
            markeredgewidth=2, zorder=10)


def plot_recognition_debug(
    scenario_map: Map,
    goal: Goal,
    agent_id: int,
    frame: Dict[int, AgentState],
    opt_trajectory: VelocityTrajectory,
    opt_plan: List[Maneuver],
    all_trajectories: List[VelocityTrajectory],
    all_plans: List[List[Maneuver]],
    observed_trajectory: Trajectory = None,
):
    """Plot debug visualization for MacroGuidedRecognition.

    Creates a figure with one subplot for the optimal trajectory and one for
    each candidate trajectory. Each subplot shows the road layout zoomed to the
    ego region, the ego's observed history, and the trajectory colored by
    maneuver type.

    The plot is displayed with plt.show() (blocking). Close the window to
    continue the simulation.

    Args:
        scenario_map: The road network map.
        goal: The goal being evaluated.
        agent_id: The ego agent ID.
        frame: Current frame with agent states.
        opt_trajectory: The optimal (benchmark) trajectory.
        opt_plan: The maneuver plan for the optimal trajectory.
        all_trajectories: Candidate trajectories from current position.
        all_plans: Maneuver plans for each candidate trajectory.
        observed_trajectory: The ego's observed trajectory so far.
    """
    n_candidates = len(all_plans)
    n_plots = 1 + n_candidates  # optimal + candidates
    n_cols = min(3, n_plots)
    n_rows = ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))

    # Ensure axes is always a flat array
    if n_plots == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    ego_pos = frame[agent_id].position

    # Track which maneuver types have been seen (for legend deduplication)
    seen_types = set()

    # --- Subplot 0: Optimal trajectory ---
    ax = axes[0]
    _setup_subplot(ax, scenario_map, ego_pos, goal, observed_trajectory)

    if opt_plan is not None:
        _plot_maneuver_segments(ax, opt_plan, linestyle="--", linewidth=2.5,
                                seen_types=seen_types)
        maneuver_names = [type(m).__name__ for m in opt_plan]
        ax.set_title(f"Optimal: {maneuver_names}", fontsize=9)
    else:
        ax.set_title("Optimal: N/A", fontsize=9)

    # --- Subplots 1..N: Candidate trajectories ---
    for i, (traj, plan) in enumerate(zip(all_trajectories, all_plans)):
        ax = axes[1 + i]
        _setup_subplot(ax, scenario_map, ego_pos, goal, observed_trajectory)
        _plot_maneuver_segments(ax, plan, linestyle="-", linewidth=1.5,
                                alpha=0.85, seen_types=seen_types)

        maneuver_names = [type(m).__name__ for m in plan]
        ax.set_title(f"Candidate {i}: {maneuver_names}", fontsize=9)

    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    # Build shared legend from all seen maneuver types
    legend_handles = []
    # Observed trajectory
    legend_handles.append(mlines.Line2D([], [], color="grey", linewidth=2, label="Observed"))
    # Maneuver types
    for name in sorted(seen_types):
        color = MANEUVER_COLORS.get(name, DEFAULT_COLOR)
        legend_handles.append(mlines.Line2D([], [], color=color, linewidth=2, label=name))

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(legend_handles), fontsize=9,
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        f"Agent {agent_id} — Goal: ({goal.center[0]:.0f}, {goal.center[1]:.0f})",
        fontsize=12
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])


def plot_velocity_profiles_debug(
    agent_id: int,
    opt_trajectory: VelocityTrajectory,
    opt_plan: List[Maneuver],
    all_trajectories: List[VelocityTrajectory],
    all_plans: List[List[Maneuver]],
    observed_trajectory: Optional[Trajectory] = None,
):
    """Plot velocity profiles for the optimal and candidate trajectories.

    Creates a separate figure showing velocity (m/s) vs path length (m)
    for the optimal trajectory and each candidate. Maneuver boundaries
    are marked with vertical dashed lines.

    Args:
        agent_id: The ego agent ID.
        opt_trajectory: The optimal (benchmark) trajectory.
        opt_plan: The maneuver plan for the optimal trajectory.
        all_trajectories: Candidate trajectories from current position.
        all_plans: Maneuver plans for each candidate trajectory.
        observed_trajectory: The ego's observed trajectory so far.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot observed trajectory velocity profile
    if observed_trajectory is not None and hasattr(observed_trajectory, 'velocity') \
            and len(observed_trajectory.velocity) > 1:
        obs_pathlength = _compute_pathlength(observed_trajectory.path)
        ax.plot(obs_pathlength, observed_trajectory.velocity,
                color="grey", linewidth=2.5, alpha=0.8, label="Observed")

    # Plot optimal trajectory velocity profile
    if opt_trajectory is not None and len(opt_trajectory.velocity) > 1:
        opt_pathlength = _compute_pathlength(opt_trajectory.path)
        opt_label = "Optimal"
        if opt_plan is not None:
            names = [type(m).__name__ for m in opt_plan]
            opt_label = f"Optimal: {names}"
        ax.plot(opt_pathlength, opt_trajectory.velocity,
                color="black", linewidth=2, linestyle="--", label=opt_label)

        # Mark maneuver boundaries on optimal
        _plot_maneuver_boundaries(ax, opt_plan, color="black", alpha=0.3)

    # Color cycle for candidates
    cmap = plt.cm.get_cmap("tab10")

    # Plot each candidate velocity profile
    for i, (traj, plan) in enumerate(zip(all_trajectories, all_plans)):
        if traj is None or len(traj.velocity) < 2:
            continue
        pathlength = _compute_pathlength(traj.path)
        names = [type(m).__name__ for m in plan]
        color = cmap(i % 10)
        ax.plot(pathlength, traj.velocity,
                color=color, linewidth=1.5, alpha=0.85,
                label=f"Candidate {i}: {names}")

    ax.set_xlabel("Path length (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title(f"Agent {agent_id} — Velocity Profiles")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()


def _compute_pathlength(path: np.ndarray) -> np.ndarray:
    """Compute cumulative path length from a 2D path array."""
    if len(path) < 2:
        return np.array([0.0])
    diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


def _plot_maneuver_boundaries(ax: plt.Axes, plan: List[Maneuver],
                               color: str = "black", alpha: float = 0.3):
    """Plot vertical dashed lines at maneuver boundaries along path length."""
    if plan is None:
        return
    cumulative = 0.0
    for maneuver in plan[:-1]:  # skip last (no boundary after it)
        if maneuver.trajectory is None or len(maneuver.trajectory.path) < 2:
            continue
        path = maneuver.trajectory.path
        segment_len = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        cumulative += segment_len
        ax.axvline(x=cumulative, color=color, linestyle=":", alpha=alpha)
