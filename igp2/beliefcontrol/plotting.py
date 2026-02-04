"""Plotting utilities for BeliefAgent visualisation.

Provides persistent, blitting-based matplotlib figures that draw the
static map once and efficiently update dynamic elements each frame.

* :class:`BeliefPlotter` — visualises the sample-based policy
  (candidate clouds + selected trajectory).
* :class:`OptimisationPlotter` — visualises the constraint-optimisation
  policy (optimised trajectory + vehicle footprints along it).
"""

import logging
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Ellipse as MplEllipse

from igp2.core.agentstate import AgentState, AgentMetadata
from igp2.core.goal import Goal
from igp2.core.trajectory import StateTrajectory
from igp2.core.util import calculate_multiple_bboxes
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map

logger = logging.getLogger(__name__)


class BeliefPlotter:
    """Manages a single persistent matplotlib figure for BeliefAgent.

    The static map, reference path, and goal are drawn once and cached
    as a bitmap.  Each call to :meth:`update` restores the cached
    background and redraws only the dynamic artists.

    Args:
        scenario_map: The road layout to draw.
        reference_waypoints: Concatenated A* reference path (N, 2).
        goal: Agent goal (for the goal marker).
    """

    def __init__(self,
                 scenario_map: Map,
                 reference_waypoints: np.ndarray,
                 goal: Optional[Goal] = None):
        self._scenario_map = scenario_map
        self._reference_waypoints = reference_waypoints
        self._goal = goal

        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._background = None  # cached static bitmap

    def _init(self):
        """Create figure, draw static content, cache background."""
        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(12, 8))
        plot_map(self._scenario_map, ax=self._ax, markings=True)

        if len(self._reference_waypoints) > 0:
            self._ax.plot(
                self._reference_waypoints[:, 0],
                self._reference_waypoints[:, 1],
                'g-', linewidth=2, label='Reference path', zorder=3,
            )

        if self._goal is not None:
            self._ax.plot(*self._goal.center, 'g*', markersize=12, zorder=6)

        self._ax.set_aspect('equal')
        self._ax.legend(loc='upper right', fontsize=8)
        self._fig.tight_layout()

        self._fig.canvas.draw()
        self._background = self._fig.canvas.copy_from_bbox(self._ax.bbox)

    def update(self,
               state: AgentState,
               candidates: List[np.ndarray],
               best_idx: int,
               trajectory_cl: StateTrajectory,
               agent_id: int,
               step: int):
        """Redraw dynamic content on the persistent figure.

        Args:
            state: Current ego state (for position and view centre).
            candidates: List of sampled trajectory arrays, each (H+1, 2).
            best_idx: Index of the selected best candidate.
            trajectory_cl: Closed-loop history trajectory of the agent.
            agent_id: Agent ID (for the title).
            step: Current simulation step (for the title).
        """
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init()

        ax = self._ax

        # Re-centre view
        margin = 40.0
        ax.set_xlim(state.position[0] - margin, state.position[0] + margin)
        ax.set_ylim(state.position[1] - margin, state.position[1] + margin)

        # Restore static background
        self._fig.canvas.restore_region(self._background)

        # --- dynamic artists ---
        dynamic = []

        # Candidate trajectories
        for i, traj in enumerate(candidates):
            if i == best_idx:
                continue
            line, = ax.plot(traj[:, 0], traj[:, 1],
                            color='orange', alpha=0.15, linewidth=0.5, zorder=2)
            ax.draw_artist(line)
            dynamic.append(line)

        # Best trajectory
        if candidates:
            best = candidates[best_idx]
            line, = ax.plot(best[:, 0], best[:, 1],
                            'r-', linewidth=2, label='Selected trajectory', zorder=5)
            ax.draw_artist(line)
            dynamic.append(line)

        # Agent history
        if len(trajectory_cl) > 1:
            hist = trajectory_cl.path
            line, = ax.plot(hist[:, 0], hist[:, 1],
                            'b-', linewidth=2, label='History', zorder=4)
            ax.draw_artist(line)
            dynamic.append(line)

        # Current position
        dot, = ax.plot(state.position[0], state.position[1],
                       'ko', markersize=6, zorder=6)
        ax.draw_artist(dot)
        dynamic.append(dot)

        ax.set_title(f"BeliefAgent {agent_id}  step={step}")

        # Blit and flush
        self._fig.canvas.blit(ax.bbox)
        self._fig.canvas.flush_events()

        # Clean up dynamic artists
        for artist in dynamic:
            artist.remove()


class OptimisationPlotter:
    """Persistent matplotlib figure for the constraint-optimisation policy.

    Draws the static map, reference path and goal once, then efficiently
    redraws per-frame:

    * The optimised trajectory as a line.
    * Vehicle footprint rectangles at evenly-spaced steps along the
      trajectory, coloured to indicate whether each footprint is inside
      the drivable area (green) or outside (red).
    * The agent's closed-loop history.
    * The current vehicle position and footprint.

    Args:
        scenario_map: The road layout to draw.
        reference_waypoints: Concatenated A* reference path (N, 2).
        metadata: Agent physical metadata (for vehicle dimensions).
        goal: Agent goal (for the goal marker).
        footprint_interval: Draw a vehicle footprint every N steps along
            the optimised trajectory.  Defaults to 5.
    """

    def __init__(self,
                 scenario_map: Map,
                 reference_waypoints: np.ndarray,
                 metadata: AgentMetadata,
                 goal: Optional[Goal] = None,
                 footprint_interval: int = 5):
        self._scenario_map = scenario_map
        self._reference_waypoints = reference_waypoints
        self._metadata = metadata
        self._goal = goal
        self._footprint_interval = max(1, footprint_interval)

        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._background = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _init(self):
        """Create figure, draw static content, cache background."""
        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(12, 8))
        plot_map(self._scenario_map, ax=self._ax, markings=True)

        if len(self._reference_waypoints) > 0:
            self._ax.plot(
                self._reference_waypoints[:, 0],
                self._reference_waypoints[:, 1],
                'g-', linewidth=2, label='Reference path', zorder=3,
            )

        if self._goal is not None:
            self._ax.plot(*self._goal.center, 'g*', markersize=12, zorder=6)

        self._ax.set_aspect('equal')
        self._ax.legend(loc='upper right', fontsize=8)
        self._fig.tight_layout()

        self._fig.canvas.draw()
        self._background = self._fig.canvas.copy_from_bbox(self._ax.bbox)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(self,
               state: AgentState,
               optimised_trajectory: np.ndarray,
               full_rollout: Optional[np.ndarray],
               trajectory_cl: StateTrajectory,
               agent_id: int,
               step: int,
               *,
               milp_trajectory: Optional[np.ndarray] = None,
               other_agents=None,
               obstacles=None,
               frenet=None,
               ego_length: float = 4.5,
               ego_width: float = 1.8,
               collision_margin: float = 0.5):
        """Redraw dynamic content for one simulation step.

        Args:
            state: Current ego state (for view centre and current footprint).
            optimised_trajectory: (H+1, 2) position-only trajectory from
                the optimiser.
            full_rollout: (H+1, 4) full state rollout [x, y, heading, speed]
                from the optimiser.  If provided, vehicle footprints are
                drawn along the trajectory.  May be None.
            trajectory_cl: Closed-loop history trajectory of the agent.
            agent_id: Agent ID (for the title).
            step: Current simulation step (for the title).
            milp_trajectory: (H+1, 2) position-only trajectory from the MILP
                stage.  If provided, drawn as a cyan dashed line.  May be None.
            other_agents: Dict {agent_id: AgentState} of other vehicles.
            obstacles: List of obstacle dicts from _predict_obstacles.
            frenet: _FrenetFrame instance for coordinate conversion.
            ego_length: Ego vehicle length (m).
            ego_width: Ego vehicle width (m).
            collision_margin: Extra safety margin around obstacles (m).
        """
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init()

        ax = self._ax

        # Re-centre view
        margin = 40.0
        ax.set_xlim(state.position[0] - margin, state.position[0] + margin)
        ax.set_ylim(state.position[1] - margin, state.position[1] + margin)

        # Restore static background
        self._fig.canvas.restore_region(self._background)

        dynamic = []

        # --- MILP trajectory line (warm-start) ---
        if milp_trajectory is not None and len(milp_trajectory) > 1:
            line, = ax.plot(
                milp_trajectory[:, 0], milp_trajectory[:, 1],
                'c--', linewidth=1.5, label='MILP trajectory', zorder=4,
            )
            ax.draw_artist(line)
            dynamic.append(line)

        # # --- Optimised trajectory line (NLP) ---
        # if optimised_trajectory is not None and len(optimised_trajectory) > 1:
        #     line, = ax.plot(
        #         optimised_trajectory[:, 0], optimised_trajectory[:, 1],
        #         'r-', linewidth=2, label='NLP trajectory', zorder=5,
        #     )
        #     ax.draw_artist(line)
        #     dynamic.append(line)

        # --- Compute unified timestep indices for all footprints ---
        # Use ego rollout length as the reference horizon
        horizon_len = len(full_rollout) if full_rollout is not None else 0
        if horizon_len == 0 and obstacles:
            # Fallback: use obstacle trajectory length
            horizon_len = len(obstacles[0].get('s', []))

        footprint_indices = []
        if horizon_len > 1:
            footprint_indices = list(range(0, horizon_len, self._footprint_interval))
            # Always include the last step
            if footprint_indices[-1] != horizon_len - 1:
                footprint_indices.append(horizon_len - 1)

        # --- Vehicle footprints along trajectory ---
        if full_rollout is not None and len(full_rollout) > 1:
            for idx in footprint_indices:
                if idx >= len(full_rollout):
                    continue
                x_k, y_k, heading_k = (full_rollout[idx, 0],
                                        full_rollout[idx, 1],
                                        full_rollout[idx, 2])
                corners = calculate_multiple_bboxes(
                    [x_k], [y_k],
                    self._metadata.length, self._metadata.width, heading_k,
                )[0]  # (4, 2)

                # Check drivability via map — if any corner has no
                # drivable lane nearby, colour the footprint red.
                in_drivable = all(
                    len(self._scenario_map.lanes_at(
                        c, drivable_only=True, max_distance=1.0)) > 0
                    for c in corners
                )
                colour = (0.2, 0.8, 0.2, 0.25) if in_drivable \
                    else (0.9, 0.2, 0.2, 0.35)

                patch = MplPolygon(
                    corners, closed=True,
                    facecolor=colour,
                    edgecolor=colour[:3] + (0.8,),
                    linewidth=0.8, zorder=4,
                )
                ax.add_patch(patch)
                ax.draw_artist(patch)
                dynamic.append(patch)

        # --- MILP collision avoidance rectangles (Paper's formulation) ---
        # Disabled - uncomment to visualize Minkowski-sum rectangles
        # if obstacles and frenet:
        #     for obs in obstacles:
        #         obs_half_L = obs['length'] / 2.0
        #         obs_half_W = obs['width'] / 2.0
        #         obs_s0 = float(obs['s'][0])
        #         _, _, _, road_angle_obs = frenet._interpolate(obs_s0)
        #         dh = obs.get('heading', road_angle_obs) - road_angle_obs
        #         obs_a = abs(obs_half_L * np.cos(dh)) + abs(obs_half_W * np.sin(dh)) + collision_margin
        #         obs_b = abs(obs_half_L * np.sin(dh)) + abs(obs_half_W * np.cos(dh)) + collision_margin
        #         half_s_rect = obs_a + ego_length / 2.0
        #         half_d_rect = obs_b + ego_width / 2.0
        #         s_arr, d_arr = obs['s'], obs['d']
        #         n_steps = len(s_arr)
        #         rect_indices = list(range(0, n_steps, self._footprint_interval))
        #         if rect_indices[-1] != n_steps - 1:
        #             rect_indices.append(n_steps - 1)
        #         for idx in rect_indices:
        #             w = frenet.frenet_to_world(float(s_arr[idx]), float(d_arr[idx]))
        #             _, _, _, road_angle = frenet._interpolate(float(s_arr[idx]))
        #             cos_r, sin_r = np.cos(road_angle), np.sin(road_angle)
        #             cx, cy = w['x'], w['y']
        #             corners_local = [(-half_s_rect, -half_d_rect), (+half_s_rect, -half_d_rect),
        #                              (+half_s_rect, +half_d_rect), (-half_s_rect, +half_d_rect)]
        #             corners_world = [[cx + ds*cos_r - dd*sin_r, cy + ds*sin_r + dd*cos_r]
        #                              for ds, dd in corners_local]
        #             rect = MplPolygon(corners_world, closed=True, facecolor=(0.8, 0.2, 0.2, 0.08),
        #                               edgecolor=(0.8, 0.2, 0.2, 0.6), linestyle='-', linewidth=1.0, zorder=2)
        #             ax.add_patch(rect)
        #             ax.draw_artist(rect)
        #             dynamic.append(rect)

        # --- NLP collision avoidance ellipses ---
        # Uses the same footprint_indices as ego vehicle for alignment
        if obstacles and frenet and footprint_indices:
            for obs in obstacles:
                # Project obstacle body-frame dimensions into Frenet (s, d)
                obs_half_L = obs['length'] / 2.0
                obs_half_W = obs['width'] / 2.0
                obs_s0 = float(obs['s'][0])
                _, _, _, road_angle_obs = frenet._interpolate(obs_s0)
                dh = obs.get('heading', road_angle_obs) - road_angle_obs
                rx = abs(obs_half_L * np.cos(dh)) + abs(obs_half_W * np.sin(dh)) + collision_margin
                ry = abs(obs_half_L * np.sin(dh)) + abs(obs_half_W * np.cos(dh)) + collision_margin

                # Use world positions if available
                world_pts = obs.get('world_positions', None)
                s_arr = obs['s']
                n_obs_steps = len(s_arr)

                # Color based on whether using planned or predicted trajectory
                uses_planned = obs.get('uses_planned_trajectory', False)
                if uses_planned:
                    facecolor = (0.0, 0.7, 0.0, 0.12)
                    edgecolor = (0.0, 0.5, 0.0, 0.5)
                else:
                    facecolor = (1.0, 0.6, 0.0, 0.12)
                    edgecolor = (0.8, 0.4, 0.0, 0.5)

                # Get headings array if available
                obs_headings = obs.get('headings', None)

                # Use unified footprint_indices for alignment with ego
                for idx in footprint_indices:
                    # Skip if index exceeds obstacle trajectory length
                    if idx >= n_obs_steps:
                        continue

                    # Use stored world position if available
                    if world_pts is not None and idx < len(world_pts):
                        cx, cy = world_pts[idx]
                    else:
                        # Fallback: convert from Frenet
                        w = frenet.frenet_to_world(float(s_arr[idx]),
                                                   float(obs['d'][idx]))
                        cx, cy = w['x'], w['y']

                    # Use computed heading if available, otherwise fall back to road tangent
                    if obs_headings is not None and idx < len(obs_headings):
                        ellipse_angle = obs_headings[idx]
                    else:
                        # Fallback: road tangent angle at this s position
                        _, _, _, ellipse_angle = frenet._interpolate(float(s_arr[idx]))

                    ellipse = MplEllipse(
                        xy=(cx, cy),
                        width=2 * rx, height=2 * ry,
                        angle=np.degrees(ellipse_angle),
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        linestyle='--', linewidth=0.8,
                        zorder=3,
                    )
                    ax.add_patch(ellipse)
                    ax.draw_artist(ellipse)
                    dynamic.append(ellipse)

        # --- Predicted/Planned obstacle trajectories ---
        # Truncate to horizon_len for alignment with ego trajectory
        if obstacles:
            for obs in obstacles:
                # Use stored world positions if available, otherwise convert from Frenet
                if 'world_positions' in obs:
                    world_pts = obs['world_positions']
                elif frenet is not None:
                    world_pts = frenet.frenet_to_world_batch(obs['s'], obs['d'])
                else:
                    continue

                # Truncate to match ego horizon for visual alignment
                if horizon_len > 0 and len(world_pts) > horizon_len:
                    world_pts = world_pts[:horizon_len]

                # Use different color for planned vs predicted trajectories
                uses_planned = obs.get('uses_planned_trajectory', False)
                if uses_planned:
                    # Green for actual planned trajectories
                    color = (0.0, 0.7, 0.0)
                    linestyle = '-'
                    linewidth = 1.5
                else:
                    # Orange dashed for constant-velocity predictions
                    color = (0.8, 0.4, 0.0)
                    linestyle = '--'
                    linewidth = 1.0

                line, = ax.plot(
                    world_pts[:, 0], world_pts[:, 1],
                    color=color, linestyle=linestyle,
                    linewidth=linewidth, alpha=0.7, zorder=5,
                )
                ax.draw_artist(line)
                dynamic.append(line)

        # --- True obstacle positions ---
        if other_agents:
            for aid, agent_state in other_agents.items():
                obs_meta = getattr(agent_state, 'metadata', None)
                obs_length = obs_meta.length if obs_meta else 4.5
                obs_width = obs_meta.width if obs_meta else 1.8
                is_static = (obs_meta is not None
                             and getattr(obs_meta, 'agent_type', '') == 'static')

                corners = calculate_multiple_bboxes(
                    [agent_state.position[0]], [agent_state.position[1]],
                    obs_length, obs_width, agent_state.heading,
                )[0]

                colour = (0.6, 0.6, 0.6, 0.4) if is_static \
                    else (0.0, 0.8, 0.8, 0.35)
                edge = (0.4, 0.4, 0.4, 0.8) if is_static \
                    else (0.0, 0.6, 0.6, 0.8)

                patch = MplPolygon(
                    corners, closed=True,
                    facecolor=colour, edgecolor=edge,
                    linewidth=1.0, zorder=6,
                )
                ax.add_patch(patch)
                ax.draw_artist(patch)
                dynamic.append(patch)

        # --- Agent history ---
        if trajectory_cl is not None and len(trajectory_cl) > 1:
            hist = trajectory_cl.path
            line, = ax.plot(
                hist[:, 0], hist[:, 1],
                'b-', linewidth=2, label='History', zorder=4,
            )
            ax.draw_artist(line)
            dynamic.append(line)

        # --- Current position + footprint ---
        dot, = ax.plot(
            state.position[0], state.position[1],
            'ko', markersize=6, zorder=7,
        )
        ax.draw_artist(dot)
        dynamic.append(dot)

        cur_corners = calculate_multiple_bboxes(
            [state.position[0]], [state.position[1]],
            self._metadata.length, self._metadata.width, state.heading,
        )[0]
        cur_patch = MplPolygon(
            cur_corners, closed=True,
            facecolor=(0.1, 0.1, 0.1, 0.3),
            edgecolor='black', linewidth=1.5, zorder=7,
        )
        ax.add_patch(cur_patch)
        ax.draw_artist(cur_patch)
        dynamic.append(cur_patch)

        ax.set_title(
            f"TwoStageOPT  Agent {agent_id}  step={step}")

        # Blit and flush
        self._fig.canvas.blit(ax.bbox)
        self._fig.canvas.flush_events()

        # Clean up dynamic artists
        for artist in dynamic:
            artist.remove()
