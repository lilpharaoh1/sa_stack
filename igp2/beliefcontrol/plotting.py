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
        dummy = np.zeros((4, 2))
        for _ in range(_MAX_VEHICLE_PATCHES):
            p = MplPolygon(dummy, closed=True, visible=False, zorder=6)
            ax.add_patch(p)
            self._vehicle_patches.append(p)

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
        for p in self._true_footprints:
            p.set_visible(False)
        self._ego_patch.set_visible(False)

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
        for p in self._true_footprints:
            if p.get_visible():
                ax.draw_artist(p)
        if self._ego_patch.get_visible():
            ax.draw_artist(self._ego_patch)

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
