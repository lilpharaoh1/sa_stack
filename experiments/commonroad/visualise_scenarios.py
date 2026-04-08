"""
Visualise the three CommonRoad car-following / merging scenarios.

For each scenario this script renders:
  - A multi-panel figure showing the road network + obstacle positions
    at several key time-steps, zoomed in to the action area.
  - A single-panel overview with full obstacle trajectories overlaid.

Figures are saved to experiments/commonroad/figures/.

Usage (from repo root, carla-igp2 env):
    python experiments/commonroad/visualise_scenarios.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.draw_params import MPDrawParams

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SCENARIOS = {
    "exp1_simple_acc": "Experiment 1: Simple ACC (Car Following)",
    "exp2_merge_in_front": "Experiment 2: Vehicle Merges in Front of Ego",
    "exp3_ego_merge": "Experiment 3: Ego Merges into Adjacent Lane",
}

SNAPSHOT_STEPS = [0, 40, 80, 120, 160, 199]

OBSTACLE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
EGO_COLOR = "#27ae60"
LANE_WIDTH = 3.5
N_LANES = 3
VEH_L, VEH_W = 4.5, 1.8


def load_scenario(name: str):
    path = os.path.join(SCENARIO_DIR, f"{name}.xml")
    return CommonRoadFileReader(path).open()


def _get_positions_at(scenario, pps, t: int):
    """Return a list of (x, y) positions of all actors at time-step *t*."""
    pts = []
    # ego initial
    for pp in pps.planning_problem_dict.values():
        if t == 0:
            pts.append(pp.initial_state.position)
    # obstacles
    for obs in scenario.obstacles:
        if t == 0:
            pts.append(obs.initial_state.position)
        elif obs.prediction is not None:
            idx = t - 1
            if 0 <= idx < len(obs.prediction.trajectory.state_list):
                pts.append(obs.prediction.trajectory.state_list[idx].position)
    return pts


def _get_all_trajectory_positions(scenario):
    """Return dict  obstacle_id → Nx2 array of positions (incl. initial)."""
    trajs = {}
    for obs in scenario.obstacles:
        positions = [obs.initial_state.position]
        if obs.prediction is not None:
            for s in obs.prediction.trajectory.state_list:
                positions.append(s.position)
        trajs[obs.obstacle_id] = np.array(positions)
    return trajs


def _draw_vehicle_rect(ax, cx, cy, orient, color, alpha=0.5, lw=1.5):
    """Draw a rotated vehicle rectangle centred at (cx, cy)."""
    rect = mpatches.FancyBboxPatch(
        (-VEH_L / 2, -VEH_W / 2), VEH_L, VEH_W,
        boxstyle="round,pad=0.15",
        facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw,
    )
    t = Affine2D().rotate(orient).translate(cx, cy) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)


# ------------------------------------------------------------------
#  Snapshot panels (zoomed to action region per timestep)
# ------------------------------------------------------------------
def render_snapshots(scenario, pps, title, save_name):
    n = len(SNAPSHOT_STEPS)
    fig, axes = plt.subplots(2, n // 2, figsize=(6 * (n // 2), 8),
                             sharex=False, sharey=True)
    axes = axes.flatten()

    y_lo = -2.0
    y_hi = N_LANES * LANE_WIDTH + 2.0
    view_half_x = 60.0  # show ±60 m around the centroid of actors

    for ax, t in zip(axes, SNAPSHOT_STEPS):
        # CommonRoad renderer for lanelets
        rnd = MPRenderer(ax=ax)
        dp = MPDrawParams()
        dp.time_begin = t
        scenario.lanelet_network.draw(rnd, draw_params=dp)

        # Draw obstacles via CR renderer
        dp.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "#e74c3c"
        dp.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#c0392b"
        for obs in scenario.obstacles:
            obs.draw(rnd, draw_params=dp)

        # Draw ego start marker (only at t=0)
        pps.draw(rnd)
        rnd.render()

        # Compute view window centred on actors
        pts = _get_positions_at(scenario, pps, t)
        if pts:
            cx = np.mean([p[0] for p in pts])
        else:
            cx = 250.0
        ax.set_xlim(cx - view_half_x, cx + view_half_x)
        ax.set_ylim(y_lo, y_hi)

        ax.set_title(f"t = {t * scenario.dt:.1f} s", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]", fontsize=9)

    axes[0].set_ylabel("y [m]", fontsize=9)
    axes[n // 2].set_ylabel("y [m]", fontsize=9)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIG_DIR, f"{save_name}_snapshots.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved snapshots → {save_name}_snapshots.png")


# ------------------------------------------------------------------
#  Trajectory overview (custom drawing, no CR renderer for obstacles)
# ------------------------------------------------------------------
def render_trajectory_overview(scenario, pps, title, save_name):
    trajs = _get_all_trajectory_positions(scenario)

    # Determine x-range that covers all motion
    all_x = []
    for pos_arr in trajs.values():
        all_x.extend(pos_arr[:, 0].tolist())
    for pp in pps.planning_problem_dict.values():
        all_x.append(pp.initial_state.position[0])
    x_min = min(all_x) - 20
    x_max = max(all_x) + 20
    y_lo = -2.0
    y_hi = N_LANES * LANE_WIDTH + 2.0

    fig, ax = plt.subplots(figsize=(max(14, (x_max - x_min) / 25), 4.5))

    # Draw lane boundaries manually (clean look)
    for i in range(N_LANES + 1):
        y = i * LANE_WIDTH
        is_edge = (i == 0 or i == N_LANES)
        ax.axhline(y, color="gray", linewidth=2 if is_edge else 1,
                   linestyle="-" if is_edge else "--", zorder=1)
    # Lane shading
    for i in range(N_LANES):
        ax.axhspan(i * LANE_WIDTH, (i + 1) * LANE_WIDTH,
                   color="#f0f0f0", zorder=0)

    # Ego initial position + goal
    for pp in pps.planning_problem_dict.values():
        pos = pp.initial_state.position
        _draw_vehicle_rect(ax, pos[0], pos[1],
                           pp.initial_state.orientation, EGO_COLOR, alpha=0.7)
        ax.annotate("EGO", xy=(pos[0], pos[1]), fontsize=7, fontweight="bold",
                    ha="center", va="center", color="white", zorder=20)

        for gs in pp.goal.state_list:
            if hasattr(gs, "position") and gs.position is not None:
                shape = gs.position
                if hasattr(shape, "center") and hasattr(shape, "length"):
                    cx, cy = shape.center
                    rect = mpatches.Rectangle(
                        (cx - shape.length / 2, cy - shape.width / 2),
                        shape.length, shape.width,
                        linewidth=2, edgecolor=EGO_COLOR,
                        facecolor=EGO_COLOR, alpha=0.12,
                        linestyle="--", label="Ego goal", zorder=2,
                    )
                    ax.add_patch(rect)

    # Obstacle trajectories
    obs_list = list(scenario.obstacles)
    for idx, obs in enumerate(obs_list):
        color = OBSTACLE_COLORS[idx % len(OBSTACLE_COLORS)]
        pos_arr = trajs[obs.obstacle_id]

        # Trajectory line
        ax.plot(pos_arr[:, 0], pos_arr[:, 1], "-", color=color,
                linewidth=2.5, alpha=0.6, label=f"Vehicle {obs.obstacle_id}",
                zorder=3)

        # Vehicle rectangles at evenly spaced positions
        n_rects = 6
        indices = np.linspace(0, len(pos_arr) - 1, n_rects, dtype=int)
        for j, ti in enumerate(indices):
            px, py = pos_arr[ti]
            orient = 0.0
            if ti > 0 and obs.prediction:
                si = min(ti - 1, len(obs.prediction.trajectory.state_list) - 1)
                orient = obs.prediction.trajectory.state_list[si].orientation
            alpha_v = 0.25 + 0.5 * (j / (n_rects - 1))  # fade in
            _draw_vehicle_rect(ax, px, py, orient, color, alpha=alpha_v, lw=1)

        # Time labels at start and end
        ax.annotate(f"t=0 s", xy=(pos_arr[0, 0], pos_arr[0, 1] + 1.2),
                    fontsize=7, ha="center", color=color, zorder=15)
        t_end = (len(pos_arr) - 1) * scenario.dt
        ax.annotate(f"t={t_end:.0f} s",
                    xy=(pos_arr[-1, 0], pos_arr[-1, 1] + 1.2),
                    fontsize=7, ha="center", color=color, zorder=15)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{save_name}_overview.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overview  → {save_name}_overview.png")


def main():
    for name, title in SCENARIOS.items():
        print(f"\nRendering {name} ...")
        scenario, pps = load_scenario(name)
        render_snapshots(scenario, pps, title, name)
        render_trajectory_overview(scenario, pps, title, name)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
