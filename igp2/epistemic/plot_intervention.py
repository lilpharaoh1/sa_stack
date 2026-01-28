"""Visualization for trajectory interventions.

Plots the predicted trajectory, MCTS trajectory, adjusted (intervention) trajectory,
observed history, and velocity profiles to help understand the intervention.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from math import ceil
from typing import Dict, List, Optional, Tuple

from igp2.core.trajectory import Trajectory, VelocityTrajectory
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map
from igp2.epistemic.intervention_optimizer import InterventionResult

# Color scheme for trajectories
COLORS = {
    "predicted": "#E74C3C",      # Red - what driver is doing (wrong)
    "mcts": "#27AE60",           # Green - optimal MCTS plan
    "adjusted": "#9B59B6",       # Purple - minimal intervention
    "observed": "#7F8C8D",       # Grey - history
    "optimal": "#3498DB",        # Blue - optimal benchmark
    "adjustment_arrow": "#E67E22",  # Orange - adjustment vectors
}

# Line styles
LINE_STYLES = {
    "predicted": {"linestyle": "-", "linewidth": 2.5, "alpha": 0.8},
    "mcts": {"linestyle": "--", "linewidth": 2.5, "alpha": 0.8},
    "adjusted": {"linestyle": "-", "linewidth": 3.0, "alpha": 0.9},
    "observed": {"linestyle": "-", "linewidth": 2.0, "alpha": 0.6},
    "optimal": {"linestyle": ":", "linewidth": 2.0, "alpha": 0.5},
}

# Zoom margin around ego position (meters)
ZOOM_MARGIN = 50


def plot_intervention_result(
    result: InterventionResult,
    scenario_map: Map = None,
    ego_pos: np.ndarray = None,
    goal: Goal = None,
    observed_trajectory: Trajectory = None,
    optimal_trajectory: VelocityTrajectory = None,
    show_adjustment_vectors: bool = True,
    show_velocity_profile: bool = True,
    title: str = None,
    figsize: Tuple[float, float] = None,
    ax: plt.Axes = None,
    show: bool = True
) -> plt.Figure:
    """Plot visualization of the intervention result.

    Creates a figure showing:
    - Road layout (if map provided)
    - Predicted trajectory (red) - what driver is doing
    - MCTS trajectory (green) - optimal plan
    - Adjusted trajectory (purple) - minimal intervention
    - Observed trajectory (grey) - history
    - Adjustment vectors showing position corrections
    - Velocity profile subplot

    Args:
        result: InterventionResult from the optimizer
        scenario_map: Road network map (optional, for road layout)
        ego_pos: Ego position for zoom centering (optional)
        goal: Goal marker (optional)
        observed_trajectory: Driver's observed trajectory history (optional)
        optimal_trajectory: Optimal benchmark trajectory (optional)
        show_adjustment_vectors: Whether to show position adjustment arrows
        show_velocity_profile: Whether to show velocity profile subplot
        title: Custom title (optional)
        figsize: Figure size (optional)
        ax: Existing axes to plot on (optional)
        show: Whether to call plt.show()

    Returns:
        The matplotlib Figure
    """
    if not result.success or result.trajectory is None:
        # Create a simple failure message plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Intervention failed:\n{result.message}",
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        if show:
            plt.show()
        return fig

    # Determine figure layout
    n_cols = 2 if show_velocity_profile else 1
    if figsize is None:
        figsize = (6 * n_cols, 6)

    if ax is None:
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]
        if show_velocity_profile:
            # Can't add velocity profile to existing axes
            show_velocity_profile = False

    # Main trajectory plot
    ax_traj = axes[0]

    # Plot map if available
    if scenario_map is not None:
        plot_map(scenario_map, markings=True, ax=ax_traj)

    # Determine ego position for centering
    if ego_pos is None:
        if observed_trajectory is not None and len(observed_trajectory.path) > 0:
            ego_pos = observed_trajectory.path[-1]
        elif result.predicted_trajectory is not None:
            ego_pos = result.predicted_trajectory.path[0]
        else:
            ego_pos = np.array([0, 0])

    # Set zoom
    ax_traj.set_xlim(ego_pos[0] - ZOOM_MARGIN, ego_pos[0] + ZOOM_MARGIN)
    ax_traj.set_ylim(ego_pos[1] - ZOOM_MARGIN, ego_pos[1] + ZOOM_MARGIN)

    # Plot trajectories
    _plot_trajectory(ax_traj, observed_trajectory, "observed", "Observed")
    _plot_trajectory(ax_traj, optimal_trajectory, "optimal", "Optimal benchmark")
    _plot_trajectory(ax_traj, result.predicted_trajectory, "predicted", "Predicted (current)")
    _plot_trajectory(ax_traj, result.mcts_trajectory, "mcts", "MCTS (target)")
    _plot_trajectory(ax_traj, result.trajectory, "adjusted", "Adjusted (intervention)")

    # Plot adjustment vectors
    if show_adjustment_vectors and result.trajectory is not None:
        _plot_adjustment_vectors(
            ax_traj,
            result.predicted_trajectory,
            result.trajectory
        )

    # Plot goal marker
    if goal is not None:
        ax_traj.plot(*goal.center, marker="*", color="gold", markersize=15,
                     markeredgecolor="black", markeredgewidth=1.5, zorder=100,
                     label="Goal")

    # Plot ego position marker
    ax_traj.plot(*ego_pos, marker="o", color="black", markersize=10,
                 markeredgecolor="white", markeredgewidth=2, zorder=101)

    # Add metrics annotation
    metrics_text = (
        f"Position adj: {result.position_adjustment:.2f} m\n"
        f"Velocity adj: {result.velocity_adjustment:.2f} m/s\n"
        f"Cost: {result.original_cost_diff:.3f} → {result.adjusted_cost_diff:.3f}\n"
        f"MCTS cost: {result.mcts_cost_diff:.3f}"
    )
    ax_traj.text(0.02, 0.98, metrics_text, transform=ax_traj.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Legend
    ax_traj.legend(loc='upper right', fontsize=8)

    # Title
    if title is None:
        title = f"Intervention: {result.message}"
    ax_traj.set_title(title, fontsize=11)

    ax_traj.set_xlabel("X (m)")
    ax_traj.set_ylabel("Y (m)")
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, alpha=0.3)

    # Velocity profile subplot
    if show_velocity_profile and len(axes) > 1:
        ax_vel = axes[1]
        _plot_velocity_profiles(
            ax_vel,
            result.predicted_trajectory,
            result.mcts_trajectory,
            result.trajectory,
            observed_trajectory
        )

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def _plot_trajectory(ax: plt.Axes, trajectory: Trajectory,
                     style_key: str, label: str):
    """Plot a single trajectory with the specified style."""
    if trajectory is None or len(trajectory.path) < 2:
        return

    style = LINE_STYLES.get(style_key, LINE_STYLES["observed"])
    color = COLORS.get(style_key, "grey")

    ax.plot(trajectory.path[:, 0], trajectory.path[:, 1],
            color=color, label=label, **style)

    # Add start/end markers
    ax.plot(trajectory.path[0, 0], trajectory.path[0, 1],
            'o', color=color, markersize=6, alpha=0.7)
    ax.plot(trajectory.path[-1, 0], trajectory.path[-1, 1],
            's', color=color, markersize=6, alpha=0.7)


def _plot_adjustment_vectors(ax: plt.Axes,
                             predicted: VelocityTrajectory,
                             adjusted: VelocityTrajectory,
                             n_arrows: int = 10):
    """Plot arrows showing position adjustments from predicted to adjusted.

    Args:
        ax: Axes to plot on
        predicted: Original predicted trajectory
        adjusted: Adjusted trajectory
        n_arrows: Number of arrows to draw
    """
    if predicted is None or adjusted is None:
        return

    # Resample to same length for comparison
    n_pred = len(predicted.path)
    n_adj = len(adjusted.path)

    if n_pred < 2 or n_adj < 2:
        return

    # Select evenly spaced indices
    indices = np.linspace(0, min(n_pred, n_adj) - 1, n_arrows, dtype=int)

    for idx in indices:
        if idx >= n_pred or idx >= n_adj:
            continue

        start = predicted.path[idx]
        end = adjusted.path[idx]
        diff = end - start

        # Only draw if there's a meaningful difference
        if np.linalg.norm(diff) < 0.1:
            continue

        ax.annotate(
            '', xy=end, xytext=start,
            arrowprops=dict(
                arrowstyle='->',
                color=COLORS["adjustment_arrow"],
                lw=1.5,
                alpha=0.7
            )
        )


def _plot_velocity_profiles(ax: plt.Axes,
                            predicted: VelocityTrajectory,
                            mcts: VelocityTrajectory,
                            adjusted: VelocityTrajectory,
                            observed: Trajectory = None):
    """Plot velocity profiles for all trajectories.

    Args:
        ax: Axes to plot on
        predicted: Predicted trajectory
        mcts: MCTS trajectory
        adjusted: Adjusted trajectory
        observed: Observed trajectory (optional)
    """
    ax.set_title("Velocity Profiles", fontsize=11)
    ax.set_xlabel("Path length (m)")
    ax.set_ylabel("Velocity (m/s)")

    # Plot each trajectory's velocity vs path length
    if observed is not None and len(observed.velocity) > 1:
        if hasattr(observed, 'pathlength'):
            s = observed.pathlength
        else:
            s = np.cumsum(np.append(0, np.linalg.norm(np.diff(observed.path, axis=0), axis=1)))
        ax.plot(s, observed.velocity, color=COLORS["observed"],
                label="Observed", **LINE_STYLES["observed"])

    # Offset path lengths for future trajectories
    if observed is not None and hasattr(observed, 'pathlength') and len(observed.pathlength) > 0:
        offset = observed.pathlength[-1]
    elif observed is not None and len(observed.path) > 1:
        offset = np.sum(np.linalg.norm(np.diff(observed.path, axis=0), axis=1))
    else:
        offset = 0

    if predicted is not None and len(predicted.velocity) > 1:
        s = predicted.pathlength + offset
        ax.plot(s, predicted.velocity, color=COLORS["predicted"],
                label="Predicted", **LINE_STYLES["predicted"])

    if mcts is not None and len(mcts.velocity) > 1:
        s = mcts.pathlength + offset
        ax.plot(s, mcts.velocity, color=COLORS["mcts"],
                label="MCTS", **LINE_STYLES["mcts"])

    if adjusted is not None and len(adjusted.velocity) > 1:
        s = adjusted.pathlength + offset
        ax.plot(s, adjusted.velocity, color=COLORS["adjusted"],
                label="Adjusted", **LINE_STYLES["adjusted"])

    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_intervention_comparison(
    result: InterventionResult,
    scenario_map: Map = None,
    ego_pos: np.ndarray = None,
    goal: Goal = None,
    observed_trajectory: Trajectory = None,
    title: str = None,
    show: bool = True
) -> plt.Figure:
    """Create a side-by-side comparison of predicted vs adjusted trajectories.

    Shows two subplots:
    - Left: Predicted trajectory (what driver is doing)
    - Right: Adjusted trajectory (minimal intervention)

    Args:
        result: InterventionResult from the optimizer
        scenario_map: Road network map
        ego_pos: Ego position for zoom centering
        goal: Goal marker
        observed_trajectory: Driver's observed trajectory
        title: Custom title
        show: Whether to call plt.show()

    Returns:
        The matplotlib Figure
    """
    if not result.success or result.trajectory is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Intervention failed:\n{result.message}",
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes)
        ax.axis('off')
        if show:
            plt.show()
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Determine ego position
    if ego_pos is None:
        if observed_trajectory is not None and len(observed_trajectory.path) > 0:
            ego_pos = observed_trajectory.path[-1]
        else:
            ego_pos = result.predicted_trajectory.path[0]

    # Left: Before intervention (predicted)
    ax_before = axes[0]
    if scenario_map is not None:
        plot_map(scenario_map, markings=True, ax=ax_before)

    ax_before.set_xlim(ego_pos[0] - ZOOM_MARGIN, ego_pos[0] + ZOOM_MARGIN)
    ax_before.set_ylim(ego_pos[1] - ZOOM_MARGIN, ego_pos[1] + ZOOM_MARGIN)

    _plot_trajectory(ax_before, observed_trajectory, "observed", "Observed")
    _plot_trajectory(ax_before, result.predicted_trajectory, "predicted", "Predicted")
    _plot_trajectory(ax_before, result.mcts_trajectory, "mcts", "MCTS target")

    if goal is not None:
        ax_before.plot(*goal.center, marker="*", color="gold", markersize=15,
                       markeredgecolor="black", markeredgewidth=1.5, zorder=100)

    ax_before.plot(*ego_pos, marker="o", color="black", markersize=10,
                   markeredgecolor="white", markeredgewidth=2, zorder=101)

    ax_before.set_title(f"Before: Predicted Plan\nCost diff: {result.original_cost_diff:.3f}",
                        fontsize=11)
    ax_before.set_xlabel("X (m)")
    ax_before.set_ylabel("Y (m)")
    ax_before.legend(loc='upper right', fontsize=8)
    ax_before.set_aspect('equal')
    ax_before.grid(True, alpha=0.3)

    # Right: After intervention (adjusted)
    ax_after = axes[1]
    if scenario_map is not None:
        plot_map(scenario_map, markings=True, ax=ax_after)

    ax_after.set_xlim(ego_pos[0] - ZOOM_MARGIN, ego_pos[0] + ZOOM_MARGIN)
    ax_after.set_ylim(ego_pos[1] - ZOOM_MARGIN, ego_pos[1] + ZOOM_MARGIN)

    _plot_trajectory(ax_after, observed_trajectory, "observed", "Observed")
    _plot_trajectory(ax_after, result.trajectory, "adjusted", "Adjusted")
    _plot_trajectory(ax_after, result.mcts_trajectory, "mcts", "MCTS target")

    # Show adjustment vectors
    _plot_adjustment_vectors(ax_after, result.predicted_trajectory, result.trajectory)

    if goal is not None:
        ax_after.plot(*goal.center, marker="*", color="gold", markersize=15,
                       markeredgecolor="black", markeredgewidth=1.5, zorder=100)

    ax_after.plot(*ego_pos, marker="o", color="black", markersize=10,
                   markeredgecolor="white", markeredgewidth=2, zorder=101)

    ax_after.set_title(f"After: Adjusted Plan\nCost diff: {result.adjusted_cost_diff:.3f}",
                       fontsize=11)
    ax_after.set_xlabel("X (m)")
    ax_after.set_ylabel("Y (m)")
    ax_after.legend(loc='upper right', fontsize=8)
    ax_after.set_aspect('equal')
    ax_after.grid(True, alpha=0.3)

    # Overall title
    if title is None:
        title = f"Intervention Comparison | Position adj: {result.position_adjustment:.2f}m, Velocity adj: {result.velocity_adjustment:.2f}m/s"
    fig.suptitle(title, fontsize=12)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def plot_velocity_colored_trajectory(
    trajectory: VelocityTrajectory,
    ax: plt.Axes = None,
    vmin: float = 0,
    vmax: float = 15,
    cmap: str = 'RdYlGn',
    linewidth: float = 3,
    label: str = None,
    show_colorbar: bool = True
) -> plt.Axes:
    """Plot a trajectory with color indicating velocity.

    Low velocity = red, high velocity = green.

    Args:
        trajectory: Trajectory to plot
        ax: Axes to plot on (creates new if None)
        vmin: Minimum velocity for colormap
        vmax: Maximum velocity for colormap
        cmap: Colormap name
        linewidth: Line width
        label: Label for the trajectory
        show_colorbar: Whether to show colorbar

    Returns:
        The axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    if trajectory is None or len(trajectory.path) < 2:
        return ax

    # Create line segments
    points = trajectory.path.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by velocity (use velocity at start of each segment)
    colors = trajectory.velocity[:-1]

    # Create LineCollection
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin, vmax))
    lc.set_array(colors)
    lc.set_linewidth(linewidth)

    line = ax.add_collection(lc)

    if show_colorbar:
        cbar = plt.colorbar(line, ax=ax)
        cbar.set_label('Velocity (m/s)')

    # Set axis limits
    ax.autoscale()
    ax.set_aspect('equal')

    return ax


def create_intervention_animation_frames(
    result: InterventionResult,
    scenario_map: Map = None,
    ego_pos: np.ndarray = None,
    goal: Goal = None,
    observed_trajectory: Trajectory = None,
    n_frames: int = 20
) -> List[plt.Figure]:
    """Create animation frames showing gradual transition from predicted to adjusted.

    Args:
        result: InterventionResult
        scenario_map: Road network map
        ego_pos: Ego position
        goal: Goal marker
        observed_trajectory: Observed trajectory
        n_frames: Number of frames for the animation

    Returns:
        List of matplotlib Figures
    """
    if not result.success or result.trajectory is None:
        return []

    frames = []
    pred = result.predicted_trajectory
    adj = result.trajectory

    # Resample to common length
    n_points = min(len(pred.path), len(adj.path))

    for i in range(n_frames):
        alpha = i / (n_frames - 1)  # 0 to 1

        # Interpolate between predicted and adjusted
        interp_path = (1 - alpha) * pred.path[:n_points] + alpha * adj.path[:n_points]
        interp_vel = (1 - alpha) * pred.velocity[:n_points] + alpha * adj.velocity[:n_points]
        interp_traj = VelocityTrajectory(interp_path, interp_vel)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        if scenario_map is not None:
            plot_map(scenario_map, markings=True, ax=ax)

        if ego_pos is None and observed_trajectory is not None:
            ego_pos = observed_trajectory.path[-1]
        elif ego_pos is None:
            ego_pos = pred.path[0]

        ax.set_xlim(ego_pos[0] - ZOOM_MARGIN, ego_pos[0] + ZOOM_MARGIN)
        ax.set_ylim(ego_pos[1] - ZOOM_MARGIN, ego_pos[1] + ZOOM_MARGIN)

        # Plot trajectories
        _plot_trajectory(ax, observed_trajectory, "observed", "Observed")
        _plot_trajectory(ax, result.mcts_trajectory, "mcts", "MCTS target")

        # Plot interpolated trajectory
        ax.plot(interp_traj.path[:, 0], interp_traj.path[:, 1],
                color=COLORS["adjusted"], linewidth=3, alpha=0.9,
                label=f"Transition ({alpha:.0%})")

        # Goal marker
        if goal is not None:
            ax.plot(*goal.center, marker="*", color="gold", markersize=15,
                    markeredgecolor="black", markeredgewidth=1.5, zorder=100)

        ax.set_title(f"Intervention Progress: {alpha:.0%}", fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        frames.append(fig)

    return frames
