"""Plotting utilities for BeliefAgent visualisation.

Provides persistent, blitting-based matplotlib figures that draw the
static map once and efficiently update dynamic elements each frame.

* :class:`BeliefPlotter` — visualises the sample-based policy
  (candidate clouds + selected trajectory).
* :class:`OptimisationPlotter` — visualises the constraint-optimisation
  policy (optimised trajectory + vehicle footprints along it).
"""

import logging
from typing import List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon, Ellipse as MplEllipse, Patch

from igp2.core.agentstate import AgentState, AgentMetadata
from igp2.core.goal import Goal
from igp2.core.trajectory import StateTrajectory
from igp2.core.util import calculate_multiple_bboxes
from igp2.opendrive.map import Map
from igp2.opendrive.plot_map import plot_map

logger = logging.getLogger(__name__)

# Pool sizes for pre-allocated artists
_MAX_OBS_LINES = 12     # obstacle trajectory lines
_MAX_ELLIPSES = 60      # collision avoidance ellipses
_MAX_VEHICLE_PATCHES = 8  # vehicle bounding boxes
_MAX_TRUE_FOOTPRINTS = 30  # traffic agent footprints along true trajectory


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
        """Redraw dynamic content on the persistent figure."""
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init()

        ax = self._ax

        margin = 40.0
        ax.set_xlim(state.position[0] - margin, state.position[0] + margin)
        ax.set_ylim(state.position[1] - margin, state.position[1] + margin)

        self._fig.canvas.restore_region(self._background)

        dynamic = []

        for i, traj in enumerate(candidates):
            if i == best_idx:
                continue
            line, = ax.plot(traj[:, 0], traj[:, 1],
                            color='orange', alpha=0.15, linewidth=0.5, zorder=2)
            ax.draw_artist(line)
            dynamic.append(line)

        if candidates:
            best = candidates[best_idx]
            line, = ax.plot(best[:, 0], best[:, 1],
                            'r-', linewidth=2, zorder=5)
            ax.draw_artist(line)
            dynamic.append(line)

        if len(trajectory_cl) > 1:
            hist = trajectory_cl.path
            line, = ax.plot(hist[:, 0], hist[:, 1],
                            'b-', linewidth=2, zorder=4)
            ax.draw_artist(line)
            dynamic.append(line)

        dot, = ax.plot(state.position[0], state.position[1],
                       'ko', markersize=6, zorder=6)
        ax.draw_artist(dot)
        dynamic.append(dot)

        ax.set_title(f"BeliefAgent {agent_id}  step={step}")

        self._fig.canvas.blit(ax.bbox)
        self._fig.canvas.flush_events()

        for artist in dynamic:
            artist.remove()


class OptimisationPlotter:
    """Persistent matplotlib figure for the constraint-optimisation policy.

    Uses pre-allocated artist pools to avoid the overhead of creating and
    destroying matplotlib objects every frame.  All dynamic artists are
    created once in :meth:`_init`, then updated in-place each frame via
    ``set_data`` / ``set_center`` / ``set_xy`` / ``set_visible``.

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

        # Pre-allocated artist handles (created in _init)
        self._lines: Dict[str, Line2D] = {}
        self._obs_lines: List[Line2D] = []
        self._ellipses: List[MplEllipse] = []
        self._vehicle_patches: List[MplPolygon] = []
        self._true_footprints: List[MplPolygon] = []
        self._ego_patch: Optional[MplPolygon] = None

    def _init(self):
        """Create figure, draw static content, pre-allocate artists, cache background."""
        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(12, 8))
        ax = self._ax

        plot_map(self._scenario_map, ax=ax, markings=True)

        if len(self._reference_waypoints) > 0:
            ax.plot(
                self._reference_waypoints[:, 0],
                self._reference_waypoints[:, 1],
                'g-', linewidth=2, zorder=3,
            )

        if self._goal is not None:
            ax.plot(*self._goal.center, 'g*', markersize=12, zorder=6)

        # --- Static legend (proxy artists) ---
        legend_handles = [
            Line2D([0], [0], color='r', linewidth=2, label='Human NLP'),
            Line2D([0], [0], color=(0.0, 0.7, 0.0), linewidth=2, label='True NLP'),
            Line2D([0], [0], color='r', linewidth=1.5, linestyle='--', label='Human MILP'),
            Line2D([0], [0], color=(0.0, 0.7, 0.0), linewidth=1.5, linestyle='--',
                   label='True MILP'),
            Line2D([0], [0], color='g', linewidth=2, label='Reference'),
            Patch(facecolor=(0.0, 0.7, 0.0, 0.35), edgecolor=(0.0, 0.5, 0.0, 0.8),
                  label='Visible agent'),
            Patch(facecolor='none', edgecolor=(0.8, 0.2, 0.2, 0.8),
                  hatch='///', label='Hidden agent'),
            Patch(facecolor='none', edgecolor=(0.8, 0.2, 0.2, 0.8),
                  hatch='...', label='Biased agent'),
            Line2D([0], [0], color=(0.8, 0.2, 0.2), linewidth=1.5, linestyle='-.',
                   label='Biased prediction'),
            Line2D([0], [0], color=(0.0, 0.7, 0.0), linewidth=1.5, linestyle='--',
                   alpha=0.5, label='True prediction'),
            Patch(facecolor=(0.0, 0.7, 0.0, 0.15), edgecolor=(0.0, 0.6, 0.0, 0.6),
                  linestyle='--', label='True agent ellipse'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=6,
                  framealpha=0.8)

        # --- Pre-allocate fixed line artists ---
        empty = ([], [])
        self._lines = {
            'human_milp': ax.plot(*empty, color='r', ls='--', linewidth=1.5, zorder=4)[0],
            'human_nlp':  ax.plot(*empty, 'r-', linewidth=2, zorder=5)[0],
            'true_milp':  ax.plot(*empty, color=(0, 0.7, 0), ls='--', lw=1.5, zorder=4)[0],
            'true_nlp':   ax.plot(*empty, color=(0, 0.7, 0), ls='-', lw=2, zorder=5)[0],
            'dot':        ax.plot(*empty, 'ko', markersize=6, zorder=7)[0],
        }

        # --- Pre-allocate obstacle trajectory line pool ---
        self._obs_lines = []
        for _ in range(_MAX_OBS_LINES):
            l, = ax.plot(*empty, zorder=5)
            self._obs_lines.append(l)

        # --- Pre-allocate ellipse pool ---
        self._ellipses = []
        for _ in range(_MAX_ELLIPSES):
            e = MplEllipse((0, 0), 0, 0, visible=False, zorder=3)
            ax.add_patch(e)
            self._ellipses.append(e)

        # --- Pre-allocate vehicle patch pool ---
        self._vehicle_patches = []
        self._vehicle_speed_labels = []
        dummy = np.zeros((4, 2))
        for _ in range(_MAX_VEHICLE_PATCHES):
            p = MplPolygon(dummy, closed=True, visible=False, zorder=6)
            ax.add_patch(p)
            self._vehicle_patches.append(p)
            t = ax.text(0, 0, '', fontsize=6, ha='center', va='top',
                        alpha=0.9, visible=False, zorder=8)
            self._vehicle_speed_labels.append(t)

        # --- Pre-allocate true-path footprint ellipse pool ---
        self._true_footprints = []
        for _ in range(_MAX_TRUE_FOOTPRINTS):
            e = MplEllipse((0, 0), 0, 0, visible=False,
                           facecolor=(0.0, 0.7, 0.0, 0.15),
                           edgecolor=(0.0, 0.6, 0.0, 0.6),
                           linewidth=0.8, linestyle='--', zorder=6)
            ax.add_patch(e)
            self._true_footprints.append(e)

        # --- Ego footprint patch ---
        self._ego_patch = MplPolygon(
            dummy, closed=True, visible=False,
            facecolor=(0.1, 0.1, 0.1, 0.3), edgecolor='black',
            linewidth=1.5, zorder=7,
        )
        ax.add_patch(self._ego_patch)

        # --- Ego speed label ---
        self._ego_speed_label = ax.text(
            0, 0, '', fontsize=7, fontweight='bold', ha='center', va='bottom',
            color=(0.2, 0.4, 0.9), alpha=0.9, visible=False, zorder=8,
        )

        ax.set_aspect('equal')
        self._fig.tight_layout()

        self._fig.canvas.draw()
        self._background = self._fig.canvas.copy_from_bbox(ax.bbox)

    # ------------------------------------------------------------------

    def _hide_all(self):
        """Reset all pooled artists to invisible / empty."""
        for l in self._lines.values():
            l.set_data([], [])
        for l in self._obs_lines:
            l.set_data([], [])
            l.set_visible(False)
        for e in self._ellipses:
            e.set_visible(False)
        for p in self._vehicle_patches:
            p.set_visible(False)
        for t in self._vehicle_speed_labels:
            t.set_visible(False)
        for p in self._true_footprints:
            p.set_visible(False)
        self._ego_patch.set_visible(False)
        self._ego_speed_label.set_visible(False)

    def _draw_all(self):
        """Call draw_artist on every pre-allocated artist."""
        ax = self._ax
        for l in self._lines.values():
            ax.draw_artist(l)
        for l in self._obs_lines:
            if l.get_visible():
                ax.draw_artist(l)
        for e in self._ellipses:
            if e.get_visible():
                ax.draw_artist(e)
        for p in self._vehicle_patches:
            if p.get_visible():
                ax.draw_artist(p)
        for t in self._vehicle_speed_labels:
            if t.get_visible():
                ax.draw_artist(t)
        for p in self._true_footprints:
            if p.get_visible():
                ax.draw_artist(p)
        if self._ego_patch.get_visible():
            ax.draw_artist(self._ego_patch)
        if self._ego_speed_label.get_visible():
            ax.draw_artist(self._ego_speed_label)

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
               collision_margin: float = 0.5,
               true_rollout: Optional[np.ndarray] = None,
               true_milp_trajectory: Optional[np.ndarray] = None,
               all_other_agents=None,
               agent_beliefs=None,
               true_obstacles=None,
               true_frenet=None):
        """Redraw dynamic content for one simulation step."""
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init()

        ax = self._ax

        # Re-centre view
        margin = 40.0
        ax.set_xlim(state.position[0] - margin, state.position[0] + margin)
        ax.set_ylim(state.position[1] - margin, state.position[1] + margin)

        # Restore static background and reset all pooled artists
        self._fig.canvas.restore_region(self._background)
        self._hide_all()

        # --- Fixed lines ---
        if milp_trajectory is not None and len(milp_trajectory) > 1:
            self._lines['human_milp'].set_data(
                milp_trajectory[:, 0], milp_trajectory[:, 1])

        if optimised_trajectory is not None and len(optimised_trajectory) > 1:
            self._lines['human_nlp'].set_data(
                optimised_trajectory[:, 0], optimised_trajectory[:, 1])

        if true_milp_trajectory is not None and len(true_milp_trajectory) > 1:
            self._lines['true_milp'].set_data(
                true_milp_trajectory[:, 0], true_milp_trajectory[:, 1])

        if true_rollout is not None and len(true_rollout) > 1:
            self._lines['true_nlp'].set_data(
                true_rollout[:, 0], true_rollout[:, 1])

        self._lines['dot'].set_data([state.position[0]], [state.position[1]])

        # --- Compute footprint indices ---
        horizon_len = len(full_rollout) if full_rollout is not None else 0
        if horizon_len == 0 and obstacles:
            horizon_len = len(obstacles[0].get('s', []))

        footprint_indices = []
        if horizon_len > 1:
            footprint_indices = list(range(0, horizon_len, self._footprint_interval))
            if footprint_indices[-1] != horizon_len - 1:
                footprint_indices.append(horizon_len - 1)

        # --- Merge human + true obstacle predictions ---
        merged = {}  # aid -> (obs, obs_frenet, tag, true_obs_or_None)
        true_obs_by_aid = {}
        if true_obstacles:
            for obs in true_obstacles:
                true_obs_by_aid[obs.get('agent_id')] = obs
        if obstacles:
            for obs in obstacles:
                aid = obs.get('agent_id')
                belief = agent_beliefs.get(aid) if agent_beliefs and aid is not None else None
                if belief is not None and belief.velocity_error != 0.0:
                    merged[aid] = (obs, frenet, 'biased', true_obs_by_aid.get(aid))
                else:
                    merged[aid] = (obs, frenet, 'normal', None)
        for aid, obs in true_obs_by_aid.items():
            if aid not in merged:
                merged[aid] = (obs, true_frenet or frenet, 'hidden', None)

        # --- Collision avoidance ellipses ---
        ei = 0  # ellipse pool index
        if merged and footprint_indices:
            for aid, (obs, obs_frenet, tag, _) in merged.items():
                if obs_frenet is None:
                    continue
                half_L = obs['length'] / 2.0 + collision_margin
                half_W = obs['width'] / 2.0 + collision_margin
                world_pts = obs.get('world_positions')
                s_arr = obs['s']
                n_steps = len(s_arr)
                headings = obs.get('headings')

                if tag in ('hidden', 'biased'):
                    fc = (1, 1, 1, 0)
                    ec = (0.8, 0.2, 0.2, 0.5)
                    ls = ':'
                else:
                    if obs.get('uses_planned_trajectory', False):
                        fc = (0, 0.7, 0, 0.12)
                        ec = (0, 0.5, 0, 0.5)
                    else:
                        fc = (1, 0.6, 0, 0.12)
                        ec = (0.8, 0.4, 0, 0.5)
                    ls = '--'

                for idx in footprint_indices:
                    if idx >= n_steps or ei >= _MAX_ELLIPSES:
                        break
                    if world_pts is not None and idx < len(world_pts):
                        cx, cy = world_pts[idx]
                    else:
                        w = obs_frenet.frenet_to_world(
                            float(s_arr[idx]), float(obs['d'][idx]))
                        cx, cy = w['x'], w['y']

                    if headings is not None and idx < len(headings):
                        angle = headings[idx]
                    else:
                        _, _, _, angle = obs_frenet._interpolate(float(s_arr[idx]))

                    e = self._ellipses[ei]
                    e.set_center((cx, cy))
                    e.width = 2 * half_L
                    e.height = 2 * half_W
                    e.angle = np.degrees(angle)
                    e.set_facecolor(fc)
                    e.set_edgecolor(ec)
                    e.set_linestyle(ls)
                    e.set_linewidth(0.8)
                    e.set_visible(True)
                    ei += 1

        # --- Obstacle trajectory lines ---
        li = 0  # obs line pool index
        if merged:
            for aid, (obs, obs_frenet, tag, true_obs) in merged.items():
                # For biased agents, draw true trajectory underneath first
                if tag == 'biased' and true_obs is not None and li < _MAX_OBS_LINES:
                    true_pts = self._get_world_pts(true_obs, true_frenet or obs_frenet)
                    if true_pts is not None:
                        if horizon_len > 0 and len(true_pts) > horizon_len:
                            true_pts = true_pts[:horizon_len]
                        l = self._obs_lines[li]
                        l.set_data(true_pts[:, 0], true_pts[:, 1])
                        l.set_color((0, 0.7, 0))
                        l.set_linestyle('--')
                        l.set_linewidth(1.5)
                        l.set_alpha(0.5)
                        l.set_visible(True)
                        li += 1

                # Main trajectory
                if li >= _MAX_OBS_LINES:
                    break
                world_pts = self._get_world_pts(obs, obs_frenet)
                if world_pts is None:
                    continue
                if horizon_len > 0 and len(world_pts) > horizon_len:
                    world_pts = world_pts[:horizon_len]

                if tag in ('hidden', 'biased'):
                    color, ls, lw, alpha = (0.8, 0.2, 0.2), \
                        (':' if tag == 'hidden' else '-.'), 1.5, 1.0
                else:
                    if obs.get('uses_planned_trajectory', False):
                        color, ls, lw = (0, 0.7, 0), '-', 1.5
                    else:
                        color, ls, lw = (0.8, 0.4, 0), '--', 1.0
                    alpha = 0.7

                l = self._obs_lines[li]
                l.set_data(world_pts[:, 0], world_pts[:, 1])
                l.set_color(color)
                l.set_linestyle(ls)
                l.set_linewidth(lw)
                l.set_alpha(alpha)
                l.set_visible(True)
                li += 1

        # --- Vehicle bounding boxes ---
        draw_agents = all_other_agents if all_other_agents else other_agents
        pi = 0  # patch pool index
        if draw_agents:
            for aid, agent_state in draw_agents.items():
                if pi >= _MAX_VEHICLE_PATCHES:
                    break
                meta = getattr(agent_state, 'metadata', None)
                vl = meta.length if meta else 4.5
                vw = meta.width if meta else 1.8
                is_static = (meta is not None
                             and getattr(meta, 'agent_type', '') == 'static')

                belief = agent_beliefs.get(aid) if agent_beliefs else None
                has_effect = (belief is not None
                              and (not belief.visible or belief.velocity_error != 0.0))

                corners = calculate_multiple_bboxes(
                    [agent_state.position[0]], [agent_state.position[1]],
                    vl, vw, agent_state.heading,
                )[0]

                if is_static:
                    fc, ec, hatch, lw = (0.6, 0.6, 0.6, 0.4), (0.4, 0.4, 0.4, 0.8), '', 1.0
                elif not has_effect:
                    fc, ec, hatch, lw = (0, 0.7, 0, 0.35), (0, 0.5, 0, 0.8), '', 1.0
                elif not belief.visible:
                    fc, ec, hatch, lw = (1, 1, 1, 0), (0.8, 0.2, 0.2, 0.8), '///', 1.5
                else:
                    fc, ec, hatch, lw = (1, 1, 1, 0), (0.8, 0.2, 0.2, 0.8), '...', 1.5

                p = self._vehicle_patches[pi]
                p.set_xy(corners)
                p.set_facecolor(fc)
                p.set_edgecolor(ec)
                p.set_hatch(hatch)
                p.set_linewidth(lw)
                p.set_visible(True)

                # Speed annotation
                t = self._vehicle_speed_labels[pi]
                spd = agent_state.speed if hasattr(agent_state, 'speed') else 0.0
                t.set_position((agent_state.position[0],
                                agent_state.position[1] - 2.5))
                t.set_text(f"{spd:.1f} m/s")
                t.set_color(ec[:3] if len(ec) >= 3 else ec)
                t.set_visible(True)

                pi += 1

        # --- True-path ellipse footprints for biased traffic agents ---
        fi = 0  # true footprint pool index
        if merged and footprint_indices:
            for aid, (obs, obs_frenet, tag, true_obs) in merged.items():
                if tag != 'biased' or true_obs is None:
                    continue
                true_pts = self._get_world_pts(true_obs, true_frenet or obs_frenet)
                if true_pts is None:
                    continue
                true_headings = true_obs.get('headings')
                half_L = true_obs['length'] / 2.0 + collision_margin
                half_W = true_obs['width'] / 2.0 + collision_margin
                n_pts = len(true_pts)
                for idx in footprint_indices:
                    if idx >= n_pts or fi >= _MAX_TRUE_FOOTPRINTS:
                        break
                    cx, cy = true_pts[idx]
                    angle = true_headings[idx] if true_headings is not None and idx < len(true_headings) else 0.0
                    e = self._true_footprints[fi]
                    e.set_center((cx, cy))
                    e.width = 2 * half_L
                    e.height = 2 * half_W
                    e.angle = np.degrees(angle)
                    e.set_visible(True)
                    fi += 1

        # --- Ego footprint ---
        cur_corners = calculate_multiple_bboxes(
            [state.position[0]], [state.position[1]],
            self._metadata.length, self._metadata.width, state.heading,
        )[0]
        self._ego_patch.set_xy(cur_corners)
        self._ego_patch.set_visible(True)

        # Ego speed label
        spd = state.speed if hasattr(state, 'speed') else 0.0
        self._ego_speed_label.set_position(
            (state.position[0], state.position[1] + 3.0))
        self._ego_speed_label.set_text(f"ego {spd:.1f} m/s")
        self._ego_speed_label.set_visible(True)

        ax.set_title(f"BeliefAgent {agent_id}  step={step}")

        # Draw all pre-allocated artists and blit
        self._draw_all()
        self._fig.canvas.blit(ax.bbox)
        self._fig.canvas.flush_events()

    @staticmethod
    def _get_world_pts(obs, obs_frenet):
        """Extract world-coordinate positions from an obstacle dict."""
        if 'world_positions' in obs:
            return obs['world_positions']
        elif obs_frenet is not None:
            return obs_frenet.frenet_to_world_batch(obs['s'], obs['d'])
        return None


# Max candidate paths to pre-allocate in InferencePlotter (2^5 = 32)
_MAX_INFERENCE_LINES = 32
_MAX_INFERENCE_VEHICLES = 10


class InferencePlotter:
    """Live debug plot for belief inference comparisons.

    Top panel: observed ego trajectory vs candidate planned trajectories
    in world coordinates, with costs annotated at the end of each path.
    Bottom panel: velocity (speed) profiles over time for the same paths.

    Both panels use solid lines for the compared portion and faded
    (alpha=0.2) dashed lines for the future extension.

    Args:
        scenario_map: The road layout to draw.
        reference_waypoints: Concatenated A* reference path (N, 2).
        frenet: FrenetFrame for converting (s, d) to world coords.
        dt: Planning timestep in seconds (for velocity x-axis).
    """

    def __init__(self,
                 scenario_map: Map,
                 reference_waypoints: np.ndarray,
                 frenet: 'FrenetFrame',
                 dt: float = 0.1,
                 ego_length: float = 4.5,
                 ego_width: float = 1.8):
        self._scenario_map = scenario_map
        self._reference_waypoints = reference_waypoints
        self._frenet = frenet
        self._dt = dt
        self._ego_length = ego_length
        self._ego_width = ego_width

        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None
        self._vel_ax: Optional[plt.Axes] = None
        self._background = None

        # Pre-allocated artist handles (map panel)
        self._observed_line: Optional[Line2D] = None
        self._candidate_lines: List[Line2D] = []
        self._candidate_future_lines: List[Line2D] = []
        self._cost_labels: List[plt.Text] = []

        # Pre-allocated artist handles (velocity panel)
        self._vel_observed_line: Optional[Line2D] = None
        self._vel_candidate_lines: List[Line2D] = []
        self._vel_candidate_future_lines: List[Line2D] = []
        self._vel_window_line: Optional[Line2D] = None

        # Pre-allocated vehicle patches + labels (other agents)
        self._vehicle_patches: List[MplPolygon] = []
        self._vehicle_labels: List[plt.Text] = []
        self._ego_patch: Optional[MplPolygon] = None

    def _init(self):
        """Create figure, draw static content, pre-allocate artists."""
        plt.ion()
        self._fig, (self._ax, self._vel_ax) = plt.subplots(
            2, 1, figsize=(12, 10),
            gridspec_kw={'height_ratios': [3, 1]},
        )
        ax = self._ax
        vel_ax = self._vel_ax

        # --- Map panel ---
        plot_map(self._scenario_map, ax=ax, markings=True)

        if len(self._reference_waypoints) > 0:
            ax.plot(
                self._reference_waypoints[:, 0],
                self._reference_waypoints[:, 1],
                color=(0.5, 0.5, 0.5), linewidth=1, linestyle='--',
                alpha=0.5, zorder=2,
            )

        legend_handles = [
            Line2D([0], [0], color=(0.2, 0.4, 0.9), linewidth=3,
                   label='Observed'),
            Line2D([0], [0], color=(0.0, 0.7, 0.0), linewidth=2.5,
                   label='Best config'),
            Line2D([0], [0], color=(0.8, 0.4, 0.0), linewidth=1.2,
                   alpha=0.6, label='Other configs'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
                  framealpha=0.8)

        # Pre-allocate map artists
        self._observed_line, = ax.plot([], [], color=(0.2, 0.4, 0.9),
                                       linewidth=3, zorder=6)
        self._candidate_lines = []
        self._candidate_future_lines = []
        self._cost_labels = []
        for _ in range(_MAX_INFERENCE_LINES):
            l, = ax.plot([], [], linewidth=1.2, zorder=5)
            l.set_visible(False)
            self._candidate_lines.append(l)

            lf, = ax.plot([], [], linewidth=1.2, alpha=0.2, zorder=4)
            lf.set_visible(False)
            self._candidate_future_lines.append(lf)

            t = ax.text(0, 0, '', fontsize=6, ha='left', va='bottom',
                        alpha=0.9, visible=False, zorder=8,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.7,
                                  edgecolor='none'))
            self._cost_labels.append(t)

        # --- Pre-allocate vehicle patches for other agents ---
        dummy = np.zeros((4, 2))
        self._vehicle_patches = []
        self._vehicle_labels = []
        for _ in range(_MAX_INFERENCE_VEHICLES):
            p = MplPolygon(dummy, closed=True, visible=False,
                           linewidth=1.5, zorder=7)
            ax.add_patch(p)
            self._vehicle_patches.append(p)

            t = ax.text(0, 0, '', fontsize=7, ha='center', va='bottom',
                        fontweight='bold', visible=False, zorder=9,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.8,
                                  edgecolor='none'))
            self._vehicle_labels.append(t)

        # --- Ego vehicle patch ---
        self._ego_patch = MplPolygon(
            dummy, closed=True, visible=False,
            facecolor=(0.1, 0.1, 0.1, 0.3), edgecolor='black',
            linewidth=1.5, zorder=7)
        ax.add_patch(self._ego_patch)

        ax.set_aspect('equal')

        # --- Velocity panel ---
        vel_ax.set_xlabel('Time (s)', fontsize=8)
        vel_ax.set_ylabel('Speed (m/s)', fontsize=8)
        vel_ax.tick_params(labelsize=7)
        vel_ax.grid(True, alpha=0.3)

        self._vel_observed_line, = vel_ax.plot(
            [], [], color=(0.2, 0.4, 0.9), linewidth=2.5, zorder=6)
        self._vel_candidate_lines = []
        self._vel_candidate_future_lines = []
        for _ in range(_MAX_INFERENCE_LINES):
            l, = vel_ax.plot([], [], linewidth=1.2, zorder=5)
            l.set_visible(False)
            self._vel_candidate_lines.append(l)

            lf, = vel_ax.plot([], [], linewidth=1.2, alpha=0.2, zorder=4)
            lf.set_visible(False)
            self._vel_candidate_future_lines.append(lf)

        # Vertical line marking end of observation window
        self._vel_window_line = vel_ax.axvline(
            x=0, color='grey', linestyle=':', linewidth=1, alpha=0.5,
            visible=False, zorder=3)

        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._background = self._fig.canvas.copy_from_bbox(self._fig.bbox)

    def _hide_all(self):
        """Reset all dynamic artists."""
        self._observed_line.set_data([], [])
        for l in self._candidate_lines:
            l.set_data([], [])
            l.set_visible(False)
        for l in self._candidate_future_lines:
            l.set_data([], [])
            l.set_visible(False)
        for t in self._cost_labels:
            t.set_visible(False)

        # Vehicle patches + labels
        for p in self._vehicle_patches:
            p.set_visible(False)
        for t in self._vehicle_labels:
            t.set_visible(False)
        if self._ego_patch is not None:
            self._ego_patch.set_visible(False)

        # Velocity panel
        self._vel_observed_line.set_data([], [])
        for l in self._vel_candidate_lines:
            l.set_data([], [])
            l.set_visible(False)
        for l in self._vel_candidate_future_lines:
            l.set_data([], [])
            l.set_visible(False)
        self._vel_window_line.set_visible(False)

    def _draw_all(self):
        """Draw all visible dynamic artists on both panels."""
        ax = self._ax
        vel_ax = self._vel_ax

        # Map panel
        ax.draw_artist(self._observed_line)
        for l in self._candidate_future_lines:
            if l.get_visible():
                ax.draw_artist(l)
        for l in self._candidate_lines:
            if l.get_visible():
                ax.draw_artist(l)
        for t in self._cost_labels:
            if t.get_visible():
                ax.draw_artist(t)

        # Vehicle patches + labels
        for p in self._vehicle_patches:
            if p.get_visible():
                ax.draw_artist(p)
        for t in self._vehicle_labels:
            if t.get_visible():
                ax.draw_artist(t)
        if self._ego_patch is not None and self._ego_patch.get_visible():
            ax.draw_artist(self._ego_patch)

        # Velocity panel
        vel_ax.draw_artist(self._vel_observed_line)
        for l in self._vel_candidate_future_lines:
            if l.get_visible():
                vel_ax.draw_artist(l)
        for l in self._vel_candidate_lines:
            if l.get_visible():
                vel_ax.draw_artist(l)
        if self._vel_window_line.get_visible():
            vel_ax.draw_artist(self._vel_window_line)

    def update(self, observed_sd: np.ndarray, results: list,
               ego_position: np.ndarray, step: int,
               other_agent_states=None, marginals=None,
               ego_heading: float = 0.0):
        """Redraw the inference debug plot.

        Args:
            observed_sd: (K, 4) observed [s, d, vs, vd] trajectory.
            results: List of InferenceResult, sorted by cost.
            ego_position: Current ego world position [x, y] for view centring.
            step: Simulation step number.
            other_agent_states: Dict mapping agent_id -> AgentState.
            marginals: Dict mapping agent_id -> P(h_i | τ_obs).
            ego_heading: Ego world heading in radians.
        """
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init()

        ax = self._ax
        vel_ax = self._vel_ax

        # Centre map view on ego
        margin = 40.0
        ax.set_xlim(ego_position[0] - margin, ego_position[0] + margin)
        ax.set_ylim(ego_position[1] - margin, ego_position[1] + margin)

        self._fig.canvas.restore_region(self._background)
        self._hide_all()

        n_observed = len(observed_sd)
        dt = self._dt

        # --- Observed path (map + velocity) — only when comparing ---
        if n_observed > 0 and results:
            obs_world = self._frenet.frenet_to_world_batch(
                observed_sd[:, 0], observed_sd[:, 1])
            self._observed_line.set_data(obs_world[:, 0], obs_world[:, 1])

            # Observed speed profile
            obs_speed = np.sqrt(observed_sd[:, 2] ** 2 + observed_sd[:, 3] ** 2)
            obs_time = np.arange(n_observed) * dt
            self._vel_observed_line.set_data(obs_time, obs_speed)

        # Find max time and speed for velocity axis limits
        max_time = n_observed * dt
        max_speed = 0.0
        if n_observed > 0:
            max_speed = float(np.max(np.sqrt(
                observed_sd[:, 2] ** 2 + observed_sd[:, 3] ** 2)))

        # --- Candidate paths ---
        best_pos = results[0].pos_cost if results else float('inf')
        for i, r in enumerate(results):
            if i >= _MAX_INFERENCE_LINES:
                break

            l = self._candidate_lines[i]
            lf = self._candidate_future_lines[i]
            vl = self._vel_candidate_lines[i]
            vlf = self._vel_candidate_future_lines[i]

            if r.planned_sd is not None and r.milp_ok:
                n_planned = len(r.planned_sd)
                n_compare = min(n_observed, n_planned)

                # --- Map: compared portion ---
                sd_compared = r.planned_sd[:n_compare]
                plan_world = self._frenet.frenet_to_world_batch(
                    sd_compared[:, 0], sd_compared[:, 1])
                l.set_data(plan_world[:, 0], plan_world[:, 1])

                if r.pos_cost == best_pos:
                    color = (0.0, 0.7, 0.0)
                    lw = 2.5
                    alpha = 1.0
                    l.set_zorder(5)
                else:
                    color = (0.8, 0.4, 0.0)
                    lw = 1.2
                    alpha = 0.9
                    l.set_zorder(4)

                l.set_color(color)
                l.set_linewidth(lw)
                l.set_alpha(alpha)
                l.set_linestyle('-')

                # --- Map: future extension ---
                if n_compare < n_planned:
                    sd_future = r.planned_sd[n_compare - 1:]
                    future_world = self._frenet.frenet_to_world_batch(
                        sd_future[:, 0], sd_future[:, 1])
                    lf.set_data(future_world[:, 0], future_world[:, 1])
                    lf.set_color(color)
                    lf.set_linewidth(lw)
                    lf.set_linestyle('-')
                    lf.set_alpha(0.5)
                    lf.set_visible(True)

                # --- Velocity: compared portion ---
                if r.planned_vel is not None:
                    plan_speed = np.sqrt(
                        r.planned_vel[:n_compare, 0] ** 2 +
                        r.planned_vel[:n_compare, 1] ** 2)
                    plan_time = np.arange(n_compare) * dt
                    vl.set_data(plan_time, plan_speed)
                    vl.set_color(color)
                    vl.set_linewidth(lw)
                    vl.set_alpha(alpha)
                    vl.set_linestyle('-')
                    vl.set_visible(True)

                    max_speed = max(max_speed, float(np.max(plan_speed)))

                    # --- Velocity: future extension ---
                    if n_compare < len(r.planned_vel):
                        future_speed = np.sqrt(
                            r.planned_vel[n_compare - 1:, 0] ** 2 +
                            r.planned_vel[n_compare - 1:, 1] ** 2)
                        future_time = np.arange(
                            n_compare - 1, len(r.planned_vel)) * dt
                        vlf.set_data(future_time, future_speed)
                        vlf.set_color(color)
                        vlf.set_linewidth(lw)
                        vlf.set_linestyle('-')
                        vlf.set_alpha(0.5)
                        vlf.set_visible(True)

                        max_time = max(max_time,
                                       float(len(r.planned_vel) * dt))
                        max_speed = max(max_speed,
                                        float(np.max(future_speed)))

            l.set_visible(True)

        # --- Velocity axis limits and window marker ---
        vel_ax.set_xlim(-0.1, max_time + 0.5)
        vel_ax.set_ylim(-0.5, max_speed + 1.0)

        # Vertical line at end of observation window
        if n_observed > 0:
            window_t = (n_observed - 1) * dt
            self._vel_window_line.set_xdata([window_t, window_t])
            self._vel_window_line.set_visible(True)

        # --- Other agent vehicles (coloured by P(h_i | τ_obs) on coolwarm) ---
        cmap = plt.cm.coolwarm
        pi = 0
        if other_agent_states:
            if marginals is None:
                marginals = {}
            for aid, agent_state in other_agent_states.items():
                if pi >= _MAX_INFERENCE_VEHICLES:
                    break
                meta = getattr(agent_state, 'metadata', None)
                vl = meta.length if meta else 4.5
                vw = meta.width if meta else 1.8

                corners = calculate_multiple_bboxes(
                    [agent_state.position[0]], [agent_state.position[1]],
                    vl, vw, agent_state.heading,
                )[0]

                p = self._vehicle_patches[pi]
                p.set_xy(corners)

                if aid < 0:
                    # Static obstacles: ego-grey with alpha
                    p.set_facecolor((0.1, 0.1, 0.1, 0.6))
                    p.set_edgecolor((0.0, 0.0, 0.0, 0.8))
                else:
                    # Dynamic agents: coolwarm by P(h_i | τ_obs)
                    p_hidden = marginals.get(aid, 0.5)
                    color = cmap(p_hidden)
                    p.set_facecolor((*color[:3], 0.5))
                    p.set_edgecolor((*color[:3], 0.9))

                p.set_linewidth(1.5)
                p.set_visible(True)

                t = self._vehicle_labels[pi]
                t.set_position((agent_state.position[0],
                                agent_state.position[1] + vw / 2 + 1.5))
                if aid < 0:
                    t.set_text("")  # no label for static obstacles
                elif aid in marginals:
                    t.set_text(f"Agent {aid}\nP(h|τ)={marginals[aid]:.2f}")
                else:
                    t.set_text(f"Agent {aid}")
                t.set_color((0.15, 0.15, 0.15))
                t.set_visible(True)

                pi += 1

        # --- Ego vehicle (grey, matching OptimisationPlotter style) ---
        ego_corners = calculate_multiple_bboxes(
            [ego_position[0]], [ego_position[1]],
            self._ego_length, self._ego_width, ego_heading,
        )[0]
        self._ego_patch.set_xy(ego_corners)
        self._ego_patch.set_visible(True)

        # --- Title ---
        n_configs = len(results)
        n_ok = sum(1 for r in results if r.milp_ok)
        if results:
            bp, bv = results[0].pos_cost, results[0].vel_cost
            best_str = f"best pos={bp:.2f} vel={bv:.2f}"
        else:
            best_str = "no candidates"
        ax.set_title(
            f"Belief Inference  step={step}  |  "
            f"{n_configs} configs ({n_ok} OK)  |  "
            f"{best_str}",
            fontsize=10,
        )

        self._draw_all()
        self._fig.canvas.blit(self._fig.bbox)
        self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# InterventionPlotter
# ---------------------------------------------------------------------------

_MAX_INTERV_OBSTACLES = 10
_MAX_INTERV_ELLIPSES = 30
_MAX_INTERV_VEHICLES = 10


class InterventionPlotter:
    """Live debug plot for minimum-intervention visualisation.

    Three-panel layout:
    - Top: map with believed (red) vs intervened (blue) trajectories,
      obstacle ellipses, and vehicle bounding boxes.
    - Middle: acceleration comparison (ref vs opt).
    - Bottom: steering comparison (ref vs opt).

    Uses pre-allocated artist pools with blitting for efficiency.

    Args:
        scenario_map: The road layout to draw.
        reference_waypoints: Concatenated A* reference path (N, 2).
        frenet: FrenetFrame for converting (s, d) to world coords.
        dt: Planning timestep in seconds.
        ego_length: Ego vehicle length (m).
        ego_width: Ego vehicle width (m).
        collision_margin: Extra safety margin around obstacles (m).
    """

    def __init__(self,
                 scenario_map: Map,
                 reference_waypoints: np.ndarray,
                 frenet: 'FrenetFrame',
                 dt: float = 0.1,
                 ego_length: float = 4.5,
                 ego_width: float = 1.8,
                 collision_margin: float = 0.5):
        self._scenario_map = scenario_map
        self._reference_waypoints = reference_waypoints
        self._frenet = frenet
        self._dt = dt
        self._ego_length = ego_length
        self._ego_width = ego_width
        self._collision_margin = collision_margin

        self._fig: Optional[plt.Figure] = None
        self._ax_map: Optional[plt.Axes] = None
        self._ax_accel: Optional[plt.Axes] = None
        self._ax_steer: Optional[plt.Axes] = None
        self._background = None

        # Pre-allocated artists (created in _init)
        self._believed_line: Optional[Line2D] = None
        self._intervened_line: Optional[Line2D] = None
        self._ego_patch: Optional[MplPolygon] = None

        self._vehicle_patches: List[MplPolygon] = []
        self._vehicle_labels: List[plt.Text] = []
        self._ellipses: List[MplEllipse] = []

        # Control panels
        self._ref_accel_line: Optional[Line2D] = None
        self._opt_accel_line: Optional[Line2D] = None
        self._ref_steer_line: Optional[Line2D] = None
        self._opt_steer_line: Optional[Line2D] = None

    def _init(self):
        """Create figure, draw static content, pre-allocate artists."""
        plt.ion()
        self._fig, (self._ax_map, self._ax_accel, self._ax_steer) = plt.subplots(
            3, 1, figsize=(12, 12),
            gridspec_kw={'height_ratios': [3, 1, 1]},
        )
        ax = self._ax_map

        # --- Static map content ---
        plot_map(self._scenario_map, ax=ax, markings=True)

        if len(self._reference_waypoints) > 0:
            ax.plot(
                self._reference_waypoints[:, 0],
                self._reference_waypoints[:, 1],
                color=(0.5, 0.5, 0.5), linewidth=1, linestyle='--',
                alpha=0.5, zorder=2,
            )

        # Legend
        legend_handles = [
            Line2D([0], [0], color='r', linewidth=2, label='Believed trajectory'),
            Line2D([0], [0], color=(0.2, 0.4, 0.9), linewidth=2,
                   label='Intervened trajectory'),
            Patch(facecolor=(0.0, 0.7, 0.0, 0.35), edgecolor=(0.0, 0.5, 0.0, 0.8),
                  label='Visible agent'),
            Patch(facecolor='none', edgecolor=(0.8, 0.2, 0.2, 0.8),
                  hatch='///', label='Hidden agent'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
                  framealpha=0.8)

        # --- Pre-allocate map artists ---
        empty = ([], [])
        self._believed_line, = ax.plot(*empty, 'r-', linewidth=2, zorder=5)
        self._intervened_line, = ax.plot(
            *empty, color=(0.2, 0.4, 0.9), linewidth=2, zorder=6)

        # Ego patch
        dummy = np.zeros((4, 2))
        self._ego_patch = MplPolygon(
            dummy, closed=True, visible=False,
            facecolor=(0.1, 0.1, 0.1, 0.3), edgecolor='black',
            linewidth=1.5, zorder=7)
        ax.add_patch(self._ego_patch)

        # Vehicle patches + labels
        self._vehicle_patches = []
        self._vehicle_labels = []
        for _ in range(_MAX_INTERV_VEHICLES):
            p = MplPolygon(dummy, closed=True, visible=False,
                           linewidth=1.5, zorder=7)
            ax.add_patch(p)
            self._vehicle_patches.append(p)

            t = ax.text(0, 0, '', fontsize=7, ha='center', va='bottom',
                        fontweight='bold', visible=False, zorder=9,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.8,
                                  edgecolor='none'))
            self._vehicle_labels.append(t)

        # Ellipses for obstacle footprints
        self._ellipses = []
        for _ in range(_MAX_INTERV_ELLIPSES):
            e = MplEllipse((0, 0), 0, 0, visible=False, zorder=3)
            ax.add_patch(e)
            self._ellipses.append(e)

        ax.set_aspect('equal')

        # --- Acceleration panel ---
        accel_ax = self._ax_accel
        accel_ax.set_ylabel('Acceleration (m/s\u00b2)', fontsize=8)
        accel_ax.tick_params(labelsize=7)
        accel_ax.grid(True, alpha=0.3)

        self._ref_accel_line, = accel_ax.plot(
            *empty, 'r--', linewidth=1.5, label='Believed', zorder=5)
        self._opt_accel_line, = accel_ax.plot(
            *empty, color=(0.2, 0.4, 0.9), linewidth=1.5,
            label='Intervened', zorder=6)
        accel_ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

        # --- Steering panel ---
        steer_ax = self._ax_steer
        steer_ax.set_xlabel('Time (s)', fontsize=8)
        steer_ax.set_ylabel('Steering angle (rad)', fontsize=8)
        steer_ax.tick_params(labelsize=7)
        steer_ax.grid(True, alpha=0.3)

        self._ref_steer_line, = steer_ax.plot(
            *empty, 'r--', linewidth=1.5, label='Believed', zorder=5)
        self._opt_steer_line, = steer_ax.plot(
            *empty, color=(0.2, 0.4, 0.9), linewidth=1.5,
            label='Intervened', zorder=6)
        steer_ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._background = self._fig.canvas.copy_from_bbox(self._fig.bbox)

    def _hide_all(self):
        """Reset all dynamic artists."""
        self._believed_line.set_data([], [])
        self._intervened_line.set_data([], [])
        self._ego_patch.set_visible(False)

        for p in self._vehicle_patches:
            p.set_visible(False)
        for t in self._vehicle_labels:
            t.set_visible(False)
        for e in self._ellipses:
            e.set_visible(False)

        self._ref_accel_line.set_data([], [])
        self._opt_accel_line.set_data([], [])
        self._ref_steer_line.set_data([], [])
        self._opt_steer_line.set_data([], [])

    def _draw_all(self):
        """Draw all visible dynamic artists on all panels."""
        ax = self._ax_map
        ax.draw_artist(self._believed_line)
        ax.draw_artist(self._intervened_line)

        for e in self._ellipses:
            if e.get_visible():
                ax.draw_artist(e)
        for p in self._vehicle_patches:
            if p.get_visible():
                ax.draw_artist(p)
        for t in self._vehicle_labels:
            if t.get_visible():
                ax.draw_artist(t)
        if self._ego_patch.get_visible():
            ax.draw_artist(self._ego_patch)

        self._ax_accel.draw_artist(self._ref_accel_line)
        self._ax_accel.draw_artist(self._opt_accel_line)
        self._ax_steer.draw_artist(self._ref_steer_line)
        self._ax_steer.draw_artist(self._opt_steer_line)

    def update(self, ref_states, ref_controls, opt_states, opt_controls,
               intervention, success, believed_config, true_obstacles,
               ego_position, ego_heading, other_agent_states,
               step, dt):
        """Redraw the intervention debug plot.

        Args:
            ref_states: (H+1, 4) believed NLP states [s, d, phi, v].
            ref_controls: (H, 2) believed controls [a, delta].
            opt_states: (H+1, 4) intervened NLP states.
            opt_controls: (H, 2) intervened controls.
            intervention: (H, 2) control deviation (opt - ref).
            success: Whether the intervention NLP succeeded.
            believed_config: {agent_id: visible} dict.
            true_obstacles: List of obstacle dicts (all agents visible).
            ego_position: [x, y] ego world position.
            ego_heading: Ego world heading (radians).
            other_agent_states: Dict mapping agent_id -> AgentState.
            step: Simulation step number.
            dt: Planning timestep (seconds).
        """
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init()

        ax = self._ax_map

        # Centre map view on ego
        margin = 40.0
        ax.set_xlim(ego_position[0] - margin, ego_position[0] + margin)
        ax.set_ylim(ego_position[1] - margin, ego_position[1] + margin)

        self._fig.canvas.restore_region(self._background)
        self._hide_all()

        # --- Title ---
        if not success:
            title_str = f"Intervention  step={step}  |  FAILED"
        elif ref_states is None:
            title_str = f"Intervention  step={step}  |  NO DATA"
        else:
            du_norm = float(np.linalg.norm(intervention))
            max_da = float(np.max(np.abs(intervention[:, 0])))
            max_dd = float(np.max(np.abs(intervention[:, 1])))
            title_str = (f"Intervention  step={step}  |  OK  "
                         f"||du||={du_norm:.4f}  "
                         f"max|da|={max_da:.2f}  max|dd|={max_dd:.2f}")
        ax.set_title(title_str, fontsize=10)

        if ref_states is None or opt_states is None:
            self._draw_all()
            self._fig.canvas.blit(self._fig.bbox)
            self._fig.canvas.flush_events()
            return

        # --- Map panel: believed trajectory (red) ---
        ref_world = self._frenet.frenet_to_world_batch(
            ref_states[:, 0], ref_states[:, 1])
        self._believed_line.set_data(ref_world[:, 0], ref_world[:, 1])

        # --- Map panel: intervened trajectory (blue) ---
        opt_world = self._frenet.frenet_to_world_batch(
            opt_states[:, 0], opt_states[:, 1])
        self._intervened_line.set_data(opt_world[:, 0], opt_world[:, 1])

        # --- Obstacle ellipses at footprint intervals ---
        footprint_interval = 5
        H = len(ref_states) - 1
        footprint_indices = list(range(0, H + 1, footprint_interval))
        if footprint_indices[-1] != H:
            footprint_indices.append(H)

        ei = 0
        if true_obstacles:
            for obs in true_obstacles:
                half_L = obs['length'] / 2.0 + self._collision_margin
                half_W = obs['width'] / 2.0 + self._collision_margin
                world_pts = obs.get('world_positions')
                s_arr = obs['s']
                n_steps = len(s_arr)
                headings = obs.get('headings')

                fc = (1, 0.6, 0, 0.12)
                ec = (0.8, 0.4, 0, 0.5)
                ls = '--'

                for idx in footprint_indices:
                    if idx >= n_steps or ei >= _MAX_INTERV_ELLIPSES:
                        break
                    if world_pts is not None and idx < len(world_pts):
                        cx, cy = world_pts[idx]
                    else:
                        w = self._frenet.frenet_to_world(
                            float(s_arr[idx]), float(obs['d'][idx]))
                        cx, cy = w['x'], w['y']

                    if headings is not None and idx < len(headings):
                        angle = headings[idx]
                    else:
                        angle = 0.0

                    e = self._ellipses[ei]
                    e.set_center((cx, cy))
                    e.width = 2 * half_L
                    e.height = 2 * half_W
                    e.angle = np.degrees(angle)
                    e.set_facecolor(fc)
                    e.set_edgecolor(ec)
                    e.set_linestyle(ls)
                    e.set_linewidth(0.8)
                    e.set_visible(True)
                    ei += 1

        # --- Vehicle bounding boxes ---
        pi = 0
        if other_agent_states:
            for aid, agent_state in other_agent_states.items():
                if pi >= _MAX_INTERV_VEHICLES:
                    break
                meta = getattr(agent_state, 'metadata', None)
                vl = meta.length if meta else 4.5
                vw = meta.width if meta else 1.8

                corners = calculate_multiple_bboxes(
                    [agent_state.position[0]], [agent_state.position[1]],
                    vl, vw, agent_state.heading,
                )[0]

                p = self._vehicle_patches[pi]
                p.set_xy(corners)

                is_visible = believed_config.get(aid, True)
                if aid < 0:
                    # Static obstacle
                    fc = (0.6, 0.6, 0.6, 0.4)
                    ec = (0.4, 0.4, 0.4, 0.8)
                    hatch = ''
                elif is_visible:
                    fc = (0.0, 0.7, 0.0, 0.35)
                    ec = (0.0, 0.5, 0.0, 0.8)
                    hatch = ''
                else:
                    fc = (1, 1, 1, 0)
                    ec = (0.8, 0.2, 0.2, 0.8)
                    hatch = '///'

                p.set_facecolor(fc)
                p.set_edgecolor(ec)
                p.set_hatch(hatch)
                p.set_linewidth(1.5)
                p.set_visible(True)

                # Label
                t = self._vehicle_labels[pi]
                label_text = f"Agent {aid}" if aid >= 0 else ""
                if aid >= 0:
                    label_text += " (V)" if is_visible else " (H)"
                t.set_position((agent_state.position[0],
                                agent_state.position[1] + vw / 2 + 1.5))
                t.set_text(label_text)
                t.set_color((0.15, 0.15, 0.15))
                t.set_visible(True)

                pi += 1

        # --- Ego vehicle patch ---
        ego_corners = calculate_multiple_bboxes(
            [ego_position[0]], [ego_position[1]],
            self._ego_length, self._ego_width, ego_heading,
        )[0]
        self._ego_patch.set_xy(ego_corners)
        self._ego_patch.set_visible(True)

        # --- Acceleration panel ---
        H = len(ref_controls)
        t_arr = np.arange(H) * dt
        self._ref_accel_line.set_data(t_arr, ref_controls[:, 0])
        self._opt_accel_line.set_data(t_arr, opt_controls[:, 0])

        a_all = np.concatenate([ref_controls[:, 0], opt_controls[:, 0]])
        a_margin = max(0.5, (np.max(a_all) - np.min(a_all)) * 0.15)
        self._ax_accel.set_xlim(-0.1, t_arr[-1] + 0.1)
        self._ax_accel.set_ylim(np.min(a_all) - a_margin,
                                np.max(a_all) + a_margin)

        # --- Steering panel ---
        self._ref_steer_line.set_data(t_arr, ref_controls[:, 1])
        self._opt_steer_line.set_data(t_arr, opt_controls[:, 1])

        d_all = np.concatenate([ref_controls[:, 1], opt_controls[:, 1]])
        d_margin = max(0.05, (np.max(d_all) - np.min(d_all)) * 0.15)
        self._ax_steer.set_xlim(-0.1, t_arr[-1] + 0.1)
        self._ax_steer.set_ylim(np.min(d_all) - d_margin,
                                np.max(d_all) + d_margin)

        # --- Draw and blit ---
        self._draw_all()
        self._fig.canvas.blit(self._fig.bbox)
        self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# MCTSTreePlotter
# ---------------------------------------------------------------------------

_MAX_TREE_NODES = 2000
_MAX_TREE_EDGES = 2000


class MCTSTreePlotter:
    """Live debug plot showing the MCTS tree structure.

    Three-panel layout in Frenet coordinates:

    - **Left** (s vs d): each node is a dot, edges connect parent→child.
      Colour by depth, size by visit count.
    - **Top-right** (s vs phi): heading vs arc-length.
    - **Bottom-right** (s vs v): speed vs arc-length.

    Road boundaries are drawn as horizontal lines on the s-d panel.

    Args:
        frenet: FrenetFrame for coordinate info.
        horizon: Planning horizon (max depth).
        dt: Planning timestep.
    """

    def __init__(self, frenet, horizon: int, dt: float):
        self._frenet = frenet
        self._horizon = horizon
        self._dt = dt

        self._fig: Optional[plt.Figure] = None
        self._ax_sd: Optional[plt.Axes] = None
        self._ax_sphi: Optional[plt.Axes] = None
        self._ax_sv: Optional[plt.Axes] = None

    def _init(self):
        plt.ion()
        self._fig = plt.figure(figsize=(14, 8))
        gs = self._fig.add_gridspec(2, 2, width_ratios=[2, 1])
        self._ax_sd = self._fig.add_subplot(gs[:, 0])
        self._ax_sphi = self._fig.add_subplot(gs[0, 1])
        self._ax_sv = self._fig.add_subplot(gs[1, 1])

        self._ax_sd.set_xlabel('s (arc-length)', fontsize=9)
        self._ax_sd.set_ylabel('d (lateral)', fontsize=9)
        self._ax_sd.set_title('MCTS Tree: s vs d', fontsize=10)
        self._ax_sd.grid(True, alpha=0.3)

        self._ax_sphi.set_xlabel('s', fontsize=9)
        self._ax_sphi.set_ylabel('phi (rad)', fontsize=9)
        self._ax_sphi.set_title('s vs heading', fontsize=10)
        self._ax_sphi.grid(True, alpha=0.3)

        self._ax_sv.set_xlabel('s', fontsize=9)
        self._ax_sv.set_ylabel('v (m/s)', fontsize=9)
        self._ax_sv.set_title('s vs speed', fontsize=10)
        self._ax_sv.grid(True, alpha=0.3)

        self._fig.tight_layout()

    def update(self, root, road_left: np.ndarray, road_right: np.ndarray,
               step: int):
        """Redraw tree from the MCTS root node.

        Args:
            root: MCTSNode root of the search tree.
            road_left: (H+1,) left road boundary.
            road_right: (H+1,) right road boundary.
            step: Simulation step number.
        """
        if root is None:
            return

        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init()

        ax_sd = self._ax_sd
        ax_sphi = self._ax_sphi
        ax_sv = self._ax_sv

        # Clear previous
        ax_sd.cla()
        ax_sphi.cla()
        ax_sv.cla()

        ax_sd.set_xlabel('s (arc-length)', fontsize=9)
        ax_sd.set_ylabel('d (lateral)', fontsize=9)
        ax_sd.grid(True, alpha=0.3)
        ax_sphi.set_xlabel('s', fontsize=9)
        ax_sphi.set_ylabel('phi (rad)', fontsize=9)
        ax_sphi.grid(True, alpha=0.3)
        ax_sv.set_xlabel('s', fontsize=9)
        ax_sv.set_ylabel('v (m/s)', fontsize=9)
        ax_sv.grid(True, alpha=0.3)

        # BFS to collect all nodes and edges
        node_s, node_d, node_phi, node_v = [], [], [], []
        node_depth, node_visits = [], []
        edge_s, edge_d = [], []
        edge_sphi_s, edge_sphi_phi = [], []
        edge_sv_s, edge_sv_v = [], []

        queue = [root]
        n_nodes = 0
        while queue and n_nodes < _MAX_TREE_NODES:
            node = queue.pop(0)
            s, d, phi, v = node.state
            node_s.append(s)
            node_d.append(d)
            node_phi.append(phi)
            node_v.append(v)
            node_depth.append(node.depth)
            node_visits.append(node.visit_count)
            n_nodes += 1

            for child in node.children.values():
                if child is None:
                    continue
                cs, cd, cphi, cv = child.state
                # Edges as line segments (parent→child)
                edge_s.extend([s, cs, None])
                edge_d.extend([d, cd, None])
                edge_sphi_s.extend([s, cs, None])
                edge_sphi_phi.extend([phi, cphi, None])
                edge_sv_s.extend([s, cs, None])
                edge_sv_v.extend([v, cv, None])
                queue.append(child)

        if not node_s:
            return

        node_s = np.array(node_s)
        node_d = np.array(node_d)
        node_phi = np.array(node_phi)
        node_v = np.array(node_v)
        node_depth = np.array(node_depth)
        node_visits = np.array(node_visits, dtype=float)

        # Normalise visit counts for marker size
        max_vis = max(node_visits.max(), 1)
        sizes = 3 + 30 * (node_visits / max_vis)

        # Normalise depth for colour
        cmap = plt.cm.viridis
        max_d = max(node_depth.max(), 1)
        colours = cmap(node_depth / max_d)

        # --- s vs d panel ---
        # Road boundaries
        s_bounds = np.linspace(node_s.min(), node_s.max(), len(road_left))
        ax_sd.plot(s_bounds, road_left[:len(s_bounds)],
                   'k--', linewidth=1.5, alpha=0.6, label='road left')
        ax_sd.plot(s_bounds, road_right[:len(s_bounds)],
                   'k--', linewidth=1.5, alpha=0.6, label='road right')

        # Edges
        ax_sd.plot(edge_s, edge_d, color=(0.7, 0.7, 0.7), linewidth=0.4,
                   alpha=0.5, zorder=1)
        # Nodes
        ax_sd.scatter(node_s, node_d, c=colours, s=sizes, zorder=3,
                      edgecolors='none', alpha=0.8)
        # Root marker
        ax_sd.scatter([node_s[0]], [node_d[0]], c='red', s=80, zorder=5,
                      marker='*', edgecolors='black', linewidths=0.5)

        # --- s vs phi panel ---
        ax_sphi.plot(edge_sphi_s, edge_sphi_phi,
                     color=(0.7, 0.7, 0.7), linewidth=0.4, alpha=0.5, zorder=1)
        ax_sphi.scatter(node_s, node_phi, c=colours, s=sizes, zorder=3,
                        edgecolors='none', alpha=0.8)
        ax_sphi.scatter([node_s[0]], [node_phi[0]], c='red', s=80, zorder=5,
                        marker='*', edgecolors='black', linewidths=0.5)

        # --- s vs v panel ---
        ax_sv.plot(edge_sv_s, edge_sv_v,
                   color=(0.7, 0.7, 0.7), linewidth=0.4, alpha=0.5, zorder=1)
        ax_sv.scatter(node_s, node_v, c=colours, s=sizes, zorder=3,
                      edgecolors='none', alpha=0.8)
        ax_sv.scatter([node_s[0]], [node_v[0]], c='red', s=80, zorder=5,
                      marker='*', edgecolors='black', linewidths=0.5)

        # Titles with stats
        ax_sd.set_title(
            f'MCTS Tree (step={step})  |  {n_nodes} nodes, '
            f'max_depth={int(node_depth.max())}/{self._horizon}',
            fontsize=10)
        ax_sphi.set_title(
            f'phi range: [{node_phi.min():.3f}, {node_phi.max():.3f}] rad',
            fontsize=9)
        ax_sv.set_title(
            f'v range: [{node_v.min():.2f}, {node_v.max():.2f}] m/s',
            fontsize=9)

        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# MCTSTrajectoryPlotter
# ---------------------------------------------------------------------------

_MAX_MCTS_LINES = 50
_MAX_MCTS_VEHICLES = 10


class MCTSTrajectoryPlotter:
    """Live debug plot for MCTS trajectory generation.

    Four-panel layout:

    - **Top** (map): all MCTS trajectories in world coordinates, coloured
      by belief category.  Blue = all visible, red = any hidden,
      orange = unclear/empty config.  A dot marks the endpoint of each
      trajectory.  Other agents are drawn with bounding boxes.
    - **Upper-middle** (velocity): speed profiles over time, same colouring.
    - **Lower-middle** (steering): steering angle profiles over time.
    - **Bottom** (cost): horizontal bar chart of trajectory costs.

    Uses pre-allocated artist pools with blitting for efficiency.

    Args:
        scenario_map: The road layout to draw.
        reference_waypoints: Concatenated A* reference path (N, 2).
        frenet: FrenetFrame for (s, d) → world conversion.
        dt: Planning timestep in seconds.
        ego_length: Ego vehicle length (m).
        ego_width: Ego vehicle width (m).
    """

    # Belief category colours
    _COLOUR_VISIBLE = (0.12, 0.47, 0.71)   # blue  — all agents visible
    _COLOUR_HIDDEN  = (0.84, 0.15, 0.16)   # red   — any hidden agent
    _COLOUR_UNCLEAR = (1.0,  0.60, 0.05)   # orange — unclear / empty config

    @staticmethod
    def _belief_colour(config: Dict[int, bool]) -> tuple:
        """Return colour based on belief category."""
        if not config:
            return MCTSTrajectoryPlotter._COLOUR_UNCLEAR
        if all(config.values()):
            return MCTSTrajectoryPlotter._COLOUR_VISIBLE
        if any(not v for v in config.values()):
            return MCTSTrajectoryPlotter._COLOUR_HIDDEN
        return MCTSTrajectoryPlotter._COLOUR_UNCLEAR

    def __init__(self,
                 scenario_map: Map,
                 reference_waypoints: np.ndarray,
                 frenet: 'FrenetFrame',
                 dt: float = 0.1,
                 ego_length: float = 4.5,
                 ego_width: float = 1.8):
        self._scenario_map = scenario_map
        self._reference_waypoints = reference_waypoints
        self._frenet = frenet
        self._dt = dt
        self._ego_length = ego_length
        self._ego_width = ego_width

        self._fig: Optional[plt.Figure] = None
        self._ax_map: Optional[plt.Axes] = None
        self._ax_vel: Optional[plt.Axes] = None
        self._ax_steer: Optional[plt.Axes] = None
        self._ax_cost: Optional[plt.Axes] = None
        self._background = None

        # Pre-allocated pools
        self._traj_lines: List[Line2D] = []
        self._traj_endpoints: List[Line2D] = []
        self._vel_lines: List[Line2D] = []
        self._steer_lines: List[Line2D] = []
        self._cost_labels: List[plt.Text] = []
        self._vehicle_patches: List[MplPolygon] = []
        self._vehicle_labels: List[plt.Text] = []
        self._ego_patch: Optional[MplPolygon] = None

    def _init(self):
        plt.ion()
        self._fig, (self._ax_map, self._ax_vel,
                    self._ax_steer, self._ax_cost) = plt.subplots(
            4, 1, figsize=(13, 14),
            gridspec_kw={'height_ratios': [3, 1, 1, 1]},
        )
        ax = self._ax_map
        vel_ax = self._ax_vel
        steer_ax = self._ax_steer
        cost_ax = self._ax_cost

        # --- Static map ---
        plot_map(self._scenario_map, ax=ax, markings=True)
        if len(self._reference_waypoints) > 0:
            ax.plot(self._reference_waypoints[:, 0],
                    self._reference_waypoints[:, 1],
                    color=(0.5, 0.5, 0.5), linewidth=1, linestyle='--',
                    alpha=0.5, zorder=2)
        ax.set_aspect('equal')

        # --- Pre-allocate trajectory lines (map + velocity + steering) ---
        empty = ([], [])
        self._traj_lines = []
        self._traj_endpoints = []
        self._vel_lines = []
        self._steer_lines = []
        self._cost_labels = []
        for _ in range(_MAX_MCTS_LINES):
            l, = ax.plot(*empty, linewidth=1, zorder=4)
            l.set_visible(False)
            self._traj_lines.append(l)

            ep, = ax.plot(*empty, marker='o', markersize=4, linestyle='None',
                          zorder=5)
            ep.set_visible(False)
            self._traj_endpoints.append(ep)

            vl, = vel_ax.plot(*empty, linewidth=1, zorder=4)
            vl.set_visible(False)
            self._vel_lines.append(vl)

            sl, = steer_ax.plot(*empty, linewidth=1, zorder=4)
            sl.set_visible(False)
            self._steer_lines.append(sl)

            t = ax.text(0, 0, '', fontsize=5.5, ha='left', va='bottom',
                        alpha=0.85, visible=False, zorder=8,
                        bbox=dict(boxstyle='round,pad=0.12',
                                  facecolor='white', alpha=0.65,
                                  edgecolor='none'))
            self._cost_labels.append(t)

        # --- Vehicle patches ---
        dummy = np.zeros((4, 2))
        self._vehicle_patches = []
        self._vehicle_labels = []
        for _ in range(_MAX_MCTS_VEHICLES):
            p = MplPolygon(dummy, closed=True, visible=False,
                           linewidth=1.5, zorder=7)
            ax.add_patch(p)
            self._vehicle_patches.append(p)

            t = ax.text(0, 0, '', fontsize=7, ha='center', va='bottom',
                        fontweight='bold', visible=False, zorder=9,
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.8,
                                  edgecolor='none'))
            self._vehicle_labels.append(t)

        # Ego patch
        self._ego_patch = MplPolygon(
            dummy, closed=True, visible=False,
            facecolor=(0.1, 0.1, 0.1, 0.3), edgecolor='black',
            linewidth=1.5, zorder=7)
        ax.add_patch(self._ego_patch)

        # --- Velocity panel ---
        vel_ax.set_xlabel('Time (s)', fontsize=8)
        vel_ax.set_ylabel('Speed (m/s)', fontsize=8)
        vel_ax.tick_params(labelsize=7)
        vel_ax.grid(True, alpha=0.3)

        # --- Steering panel ---
        steer_ax.set_xlabel('Time (s)', fontsize=8)
        steer_ax.set_ylabel('Steering (rad)', fontsize=8)
        steer_ax.tick_params(labelsize=7)
        steer_ax.grid(True, alpha=0.3)

        # --- Cost panel ---
        cost_ax.set_xlabel('Position cost (L2)', fontsize=8)
        cost_ax.tick_params(labelsize=7)
        cost_ax.grid(True, axis='x', alpha=0.3)

        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._background = self._fig.canvas.copy_from_bbox(self._fig.bbox)

    def _hide_all(self):
        for l in self._traj_lines:
            l.set_data([], [])
            l.set_visible(False)
        for ep in self._traj_endpoints:
            ep.set_data([], [])
            ep.set_visible(False)
        for l in self._vel_lines:
            l.set_data([], [])
            l.set_visible(False)
        for l in self._steer_lines:
            l.set_data([], [])
            l.set_visible(False)
        for t in self._cost_labels:
            t.set_visible(False)
        for p in self._vehicle_patches:
            p.set_visible(False)
        for t in self._vehicle_labels:
            t.set_visible(False)
        if self._ego_patch is not None:
            self._ego_patch.set_visible(False)

    def _draw_all(self):
        ax = self._ax_map
        vel_ax = self._ax_vel
        steer_ax = self._ax_steer

        for l in self._traj_lines:
            if l.get_visible():
                ax.draw_artist(l)
        for ep in self._traj_endpoints:
            if ep.get_visible():
                ax.draw_artist(ep)
        for t in self._cost_labels:
            if t.get_visible():
                ax.draw_artist(t)
        for p in self._vehicle_patches:
            if p.get_visible():
                ax.draw_artist(p)
        for t in self._vehicle_labels:
            if t.get_visible():
                ax.draw_artist(t)
        if self._ego_patch is not None and self._ego_patch.get_visible():
            ax.draw_artist(self._ego_patch)

        for l in self._vel_lines:
            if l.get_visible():
                vel_ax.draw_artist(l)
        for l in self._steer_lines:
            if l.get_visible():
                steer_ax.draw_artist(l)

    @staticmethod
    def _belief_key(config: Dict[int, bool]) -> str:
        """Canonical string key for a belief config."""
        return ", ".join(f"{aid}:{'V' if v else 'H'}"
                         for aid, v in sorted(config.items()))

    def update(self, results: list, ego_position: np.ndarray,
               ego_heading: float, step: int,
               other_agent_states: Optional[Dict] = None,
               marginals: Optional[Dict[int, float]] = None):
        """Redraw the MCTS trajectory debug plot.

        Args:
            results: List of InferenceResult from MCTS inference,
                sorted by cost.
            ego_position: [x, y] ego world position.
            ego_heading: Ego world heading (radians).
            step: Simulation step number.
            other_agent_states: Dict mapping agent_id -> AgentState.
            marginals: Dict mapping agent_id -> P(hidden | τ_obs).
        """
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init()

        ax = self._ax_map
        vel_ax = self._ax_vel
        steer_ax = self._ax_steer
        cost_ax = self._ax_cost

        margin = 40.0
        ax.set_xlim(ego_position[0] - margin, ego_position[0] + margin)
        ax.set_ylim(ego_position[1] - margin, ego_position[1] + margin)

        self._fig.canvas.restore_region(self._background)
        self._hide_all()

        dt = self._dt

        # --- Draw trajectories (map + velocity + steering) ---
        max_time = 0.0
        max_speed = 0.0
        max_steer = 0.0

        belief_best_idx: Dict[str, int] = {}
        for i, r in enumerate(results):
            key = self._belief_key(r.config)
            if key not in belief_best_idx:
                belief_best_idx[key] = i

        for i, r in enumerate(results):
            if i >= _MAX_MCTS_LINES:
                break

            colour = self._belief_colour(r.config)
            is_overall_best = (i == 0)
            lw = 1.2
            alpha = 1.0
            zorder = 6 if is_overall_best else 4

            l = self._traj_lines[i]
            ep = self._traj_endpoints[i]
            vl = self._vel_lines[i]
            sl = self._steer_lines[i]

            if r.planned_sd is not None:
                # Map panel — trajectory line
                world = self._frenet.frenet_to_world_batch(
                    r.planned_sd[:, 0], r.planned_sd[:, 1])
                l.set_data(world[:, 0], world[:, 1])
                l.set_color(colour)
                l.set_linewidth(lw)
                l.set_alpha(alpha)
                l.set_zorder(zorder)
                l.set_linestyle('-')
                l.set_visible(True)

                # Endpoint marker
                end_x, end_y = world[-1]
                ep.set_data([end_x], [end_y])
                ep.set_color(colour)
                ep.set_zorder(zorder + 1)
                ep.set_visible(True)

                # Cost label at end of best-for-belief trajectories
                key = self._belief_key(r.config)
                if i == belief_best_idx.get(key):
                    t = self._cost_labels[i]
                    t.set_position((end_x + 0.5, end_y + 0.5))
                    t.set_text(f"{{{key}}}\ncost={r.pos_cost:.2f}")
                    t.set_color(colour)
                    t.set_visible(True)

                # Velocity panel
                if r.planned_vel is not None:
                    speed = np.sqrt(r.planned_vel[:, 0]**2 +
                                    r.planned_vel[:, 1]**2)
                    t_arr = np.arange(len(speed)) * dt
                    vl.set_data(t_arr, speed)
                    vl.set_color(colour)
                    vl.set_linewidth(lw)
                    vl.set_alpha(alpha)
                    vl.set_linestyle('-')
                    vl.set_visible(True)

                    max_time = max(max_time, float(t_arr[-1]) if len(t_arr) else 0)
                    max_speed = max(max_speed, float(np.max(speed)) if len(speed) else 0)

                # Steering panel
                if r.nlp_controls is not None and len(r.nlp_controls) > 0:
                    steer_vals = r.nlp_controls[:, 1]  # delta column
                    t_arr_s = np.arange(len(steer_vals)) * dt
                    sl.set_data(t_arr_s, steer_vals)
                    sl.set_color(colour)
                    sl.set_linewidth(lw)
                    sl.set_alpha(alpha)
                    sl.set_linestyle('-')
                    sl.set_visible(True)

                    max_time = max(max_time, float(t_arr_s[-1]) if len(t_arr_s) else 0)
                    max_steer = max(max_steer, float(np.max(np.abs(steer_vals))))

        # Velocity axis limits
        vel_ax.set_xlim(-0.1, max_time + 0.5)
        vel_ax.set_ylim(-0.5, max_speed + 1.0)

        # Steering axis limits
        steer_ax.set_xlim(-0.1, max_time + 0.5)
        steer_ax.set_ylim(-max_steer - 0.05, max_steer + 0.05)

        # --- Cost bar chart (redrawn each frame — small enough) ---
        cost_ax.cla()
        cost_ax.set_xlabel('Position cost (L2)', fontsize=8)
        cost_ax.tick_params(labelsize=7)
        cost_ax.grid(True, axis='x', alpha=0.3)

        if results:
            # Show up to top 15 trajectories
            n_bars = min(len(results), 15)
            bar_labels = []
            bar_costs = []
            bar_colours = []
            for i in range(n_bars):
                r = results[i]
                key = self._belief_key(r.config)
                is_best = (i == belief_best_idx.get(key, -1))
                label = f"{{{key}}}" if is_best else f"#{i}"
                bar_labels.append(label)
                bar_costs.append(r.pos_cost if np.isfinite(r.pos_cost) else 0)
                bar_colours.append(self._belief_colour(r.config))

            y_pos = np.arange(n_bars)
            cost_ax.barh(y_pos, bar_costs, color=bar_colours, alpha=0.8,
                         edgecolor='none', height=0.7)
            cost_ax.set_yticks(y_pos)
            cost_ax.set_yticklabels(bar_labels, fontsize=6)
            cost_ax.invert_yaxis()

        # --- Vehicles ---
        pi = 0
        if other_agent_states:
            if marginals is None:
                marginals = {}
            cmap = plt.cm.coolwarm
            for aid, agent_state in other_agent_states.items():
                if pi >= _MAX_MCTS_VEHICLES:
                    break
                meta = getattr(agent_state, 'metadata', None)
                vl = meta.length if meta else 4.5
                vw = meta.width if meta else 1.8

                corners = calculate_multiple_bboxes(
                    [agent_state.position[0]], [agent_state.position[1]],
                    vl, vw, agent_state.heading)[0]

                p = self._vehicle_patches[pi]
                p.set_xy(corners)

                if aid < 0:
                    p.set_facecolor((0.1, 0.1, 0.1, 0.6))
                    p.set_edgecolor((0.0, 0.0, 0.0, 0.8))
                else:
                    p_hidden = marginals.get(aid, 0.5)
                    color = cmap(p_hidden)
                    p.set_facecolor((*color[:3], 0.5))
                    p.set_edgecolor((*color[:3], 0.9))
                p.set_linewidth(1.5)
                p.set_visible(True)

                t = self._vehicle_labels[pi]
                t.set_position((agent_state.position[0],
                                agent_state.position[1] + vw / 2 + 1.5))
                if aid < 0:
                    t.set_text("")
                elif aid in marginals:
                    t.set_text(f"Agent {aid}\nP(h)={marginals[aid]:.2f}")
                else:
                    t.set_text(f"Agent {aid}")
                t.set_color((0.15, 0.15, 0.15))
                t.set_visible(True)
                pi += 1

        # Ego
        ego_corners = calculate_multiple_bboxes(
            [ego_position[0]], [ego_position[1]],
            self._ego_length, self._ego_width, ego_heading)[0]
        self._ego_patch.set_xy(ego_corners)
        self._ego_patch.set_visible(True)

        # --- Legend (fixed colour categories) ---
        legend_handles = [
            Line2D([0], [0], color=self._COLOUR_VISIBLE, linewidth=2,
                   label='All visible'),
            Line2D([0], [0], color=self._COLOUR_HIDDEN, linewidth=2,
                   label='Has hidden'),
            Line2D([0], [0], color=self._COLOUR_UNCLEAR, linewidth=2,
                   label='Unclear'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=6,
                  framealpha=0.8, title='Belief category', title_fontsize=7)

        # --- Title ---
        n_trajs = len(results)
        n_configs = len(belief_best_idx)
        best_cost = results[0].pos_cost if results else float('inf')
        ax.set_title(
            f"MCTS Trajectories  step={step}  |  "
            f"{n_trajs} trajectories, {n_configs} unique beliefs  |  "
            f"best cost={best_cost:.3f}",
            fontsize=10)

        self._draw_all()
        self._fig.canvas.blit(self._fig.bbox)
        self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# MCTSBeliefPlotter
# ---------------------------------------------------------------------------

class MCTSBeliefPlotter:
    """Live plot showing per-agent hidden probabilities from MCTS reasoning.

    Two-panel layout:

    - **Top**: bar chart of P(hidden | τ_obs) per agent, with a
      horizontal threshold line showing the hidden-detection boundary.
    - **Bottom**: time-series of P(hidden) for each tracked agent,
      showing how beliefs evolve over successive inference steps.

    Args:
        hidden_threshold: P(hidden) above which an agent is declared
            hidden. Drawn as a horizontal reference line.
        max_history: Number of past steps to show in the time series.
    """

    _AGENT_COLOURS = [
        (0.12, 0.47, 0.71),
        (1.0,  0.50, 0.05),
        (0.17, 0.63, 0.17),
        (0.84, 0.15, 0.16),
        (0.58, 0.40, 0.74),
        (0.55, 0.34, 0.29),
        (0.89, 0.47, 0.76),
        (0.50, 0.50, 0.50),
    ]

    def __init__(self,
                 hidden_threshold: float = 0.6,
                 max_history: int = 100):
        self._hidden_threshold = hidden_threshold
        self._max_history = max_history

        self._fig: Optional[plt.Figure] = None
        self._ax_bar: Optional[plt.Axes] = None
        self._ax_ts: Optional[plt.Axes] = None

        # History tracking
        self._history_steps: List[int] = []
        self._history_marginals: Dict[int, List[float]] = {}  # aid -> [p_hidden, ...]
        self._agent_colour_map: Dict[int, tuple] = {}

    def _get_agent_colour(self, aid: int) -> tuple:
        if aid not in self._agent_colour_map:
            idx = len(self._agent_colour_map)
            self._agent_colour_map[aid] = (
                self._AGENT_COLOURS[idx % len(self._AGENT_COLOURS)])
        return self._agent_colour_map[aid]

    def _init(self):
        plt.ion()
        self._fig, (self._ax_bar, self._ax_ts) = plt.subplots(
            2, 1, figsize=(10, 7),
            gridspec_kw={'height_ratios': [1, 1.5]},
        )
        self._fig.tight_layout(pad=3.0)

    def reset(self):
        """Clear history."""
        self._history_steps = []
        self._history_marginals = {}

    def update(self, marginals: Dict[int, float], step: int,
               n_trajectories: int = 0, n_beliefs: int = 0):
        """Redraw the MCTS belief plot.

        Args:
            marginals: {agent_id: P(hidden | τ_obs)} from MCTS inference.
            step: Simulation step number.
            n_trajectories: Number of MCTS trajectories generated.
            n_beliefs: Number of unique belief configs found.
        """
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._init()

        # Update history
        self._history_steps.append(step)
        for aid, p_h in marginals.items():
            if aid < 0:
                continue
            if aid not in self._history_marginals:
                # Backfill with 0.5 (unknown) for previous steps
                self._history_marginals[aid] = [0.5] * (len(self._history_steps) - 1)
            self._history_marginals[aid].append(p_h)

        # Agents seen before but not in current marginals → assume unchanged
        for aid in list(self._history_marginals.keys()):
            if aid not in marginals:
                prev = self._history_marginals[aid][-1] if self._history_marginals[aid] else 0.5
                self._history_marginals[aid].append(prev)

        # Trim history
        if len(self._history_steps) > self._max_history:
            excess = len(self._history_steps) - self._max_history
            self._history_steps = self._history_steps[excess:]
            for aid in self._history_marginals:
                self._history_marginals[aid] = self._history_marginals[aid][excess:]

        bar_ax = self._ax_bar
        ts_ax = self._ax_ts

        # ===== Top panel: bar chart =====
        bar_ax.cla()
        bar_ax.set_ylabel('P(hidden | τ_obs)', fontsize=9)
        bar_ax.set_ylim(0, 1.05)
        bar_ax.grid(True, axis='y', alpha=0.3)

        # Threshold line
        bar_ax.axhline(y=self._hidden_threshold, color='grey',
                        linestyle='--', linewidth=1.2, alpha=0.7,
                        label=f'Threshold ({self._hidden_threshold:.1f})')

        dynamic_aids = sorted(aid for aid in marginals if aid >= 0)
        if dynamic_aids:
            x_pos = np.arange(len(dynamic_aids))
            heights = [marginals[aid] for aid in dynamic_aids]
            colours = [self._get_agent_colour(aid) for aid in dynamic_aids]

            bars = bar_ax.bar(x_pos, heights, color=colours, alpha=0.85,
                              edgecolor='black', linewidth=0.8, width=0.6)

            # Annotate each bar
            for j, (aid, h) in enumerate(zip(dynamic_aids, heights)):
                status = "HIDDEN" if h > self._hidden_threshold else "visible"
                bar_ax.text(j, h + 0.02, f"{h:.2f}\n({status})",
                            ha='center', va='bottom', fontsize=7,
                            fontweight='bold' if h > self._hidden_threshold else 'normal',
                            color='red' if h > self._hidden_threshold else 'black')

            bar_ax.set_xticks(x_pos)
            bar_ax.set_xticklabels([f"Agent {aid}" for aid in dynamic_aids],
                                    fontsize=8)
        bar_ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
        bar_ax.set_title(
            f"MCTS Belief Inference  step={step}  |  "
            f"{n_trajectories} trajs, {n_beliefs} beliefs",
            fontsize=10)

        # ===== Bottom panel: time series =====
        ts_ax.cla()
        ts_ax.set_xlabel('Simulation step', fontsize=9)
        ts_ax.set_ylabel('P(hidden | τ_obs)', fontsize=9)
        ts_ax.set_ylim(-0.05, 1.05)
        ts_ax.grid(True, alpha=0.3)

        # Threshold line
        ts_ax.axhline(y=self._hidden_threshold, color='grey',
                       linestyle='--', linewidth=1.2, alpha=0.7)

        # Shade hidden region
        ts_ax.axhspan(self._hidden_threshold, 1.05, alpha=0.06,
                       color='red', zorder=0)
        ts_ax.axhspan(-0.05, self._hidden_threshold, alpha=0.06,
                       color='green', zorder=0)

        steps = self._history_steps
        for aid in sorted(self._history_marginals.keys()):
            vals = self._history_marginals[aid]
            colour = self._get_agent_colour(aid)
            n = min(len(steps), len(vals))
            ts_ax.plot(steps[:n], vals[:n],
                       color=colour, linewidth=1.8, alpha=0.9,
                       marker='o', markersize=3,
                       label=f"Agent {aid}")

        ts_ax.legend(loc='upper left', fontsize=7, framealpha=0.8)

        self._fig.tight_layout(pad=2.0)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
