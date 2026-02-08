"""Control policies for BeliefAgent.

Two control policies are provided:

1. **SampleBased** — samples random action sequences, forward-simulates them
   through the bicycle model, picks the candidate closest to the reference
   path, and PID-tracks it.

2. **TwoStageOPT** — first solves a Mixed-Integer Linear Program (MILP)
   over a point-mass model to obtain a coarse trajectory, then converts
   it to bicycle-model controls and uses it as the warm-start for a
   constrained nonlinear program (NLP) that minimises distance to the
   reference path subject to kinematic and dynamic constraints.
"""

import logging
import time
from typing import List, Optional, Dict, Tuple

import numpy as np
import cvxpy as cp
from shapely.geometry import LineString
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.collections import PatchCollection

from igp2.core.agentstate import AgentState, AgentMetadata
from igp2.core.vehicle import Action
from igp2.opendrive.map import Map
from igp2.planlibrary.controller import PIDController

logger = logging.getLogger(__name__)


class SampleBased:
    """Sample-based trajectory planner with PID tracking.

    Each call to :meth:`select_action` samples ``n_samples`` candidate
    trajectories by forward-simulating random (acceleration, steer)
    sequences through the bicycle kinematic model, picks the candidate
    closest to the reference path, and produces a PID-tracked action
    toward that candidate.

    Args:
        fps: Simulation framerate.
        metadata: Agent physical metadata (wheelbase, limits, etc.).
        reference_waypoints: Concatenated A* reference path (N, 2).
        n_samples: Number of candidate trajectories per step.
        horizon: Number of simulation steps per candidate.
    """

    # PID gains (same as WaypointManeuver defaults)
    LATERAL_ARGS = {'K_P': 1.95, 'K_I': 0.2, 'K_D': 0.0}
    LONGITUDINAL_ARGS = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0.0}
    WAYPOINT_MARGIN = 1.0

    def __init__(self,
                 fps: int,
                 metadata: AgentMetadata,
                 reference_waypoints: np.ndarray,
                 n_samples: int = 50,
                 horizon: int = 40):
        self._fps = fps
        self._metadata = metadata
        self._reference_waypoints = reference_waypoints
        self._n_samples = n_samples
        self._horizon = horizon

        self._controller = PIDController(
            1.0 / fps, self.LATERAL_ARGS, self.LONGITUDINAL_ARGS,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def select_action(self, state: AgentState) -> tuple:
        """Sample trajectories, pick the best, and return a tracking action.

        Args:
            state: Current ego agent state.

        Returns:
            (action, candidates, best_idx) — the Action to execute,
            the list of sampled trajectory arrays, and the index of the
            selected best candidate.
        """
        candidates = self._sample_trajectories(state)
        errors = [self._trajectory_error(t) for t in candidates]
        best_idx = int(np.argmin(errors))
        action = self._track_trajectory(candidates[best_idx], state)
        return action, candidates, best_idx

    def reset(self):
        """Reset the PID controller state."""
        self._controller = PIDController(
            1.0 / self._fps, self.LATERAL_ARGS, self.LONGITUDINAL_ARGS,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _simulate_trajectory(self, state: AgentState,
                             accels: np.ndarray,
                             steers: np.ndarray) -> np.ndarray:
        """Forward-simulate the bicycle model for a sequence of actions.

        Returns:
            (horizon+1, 2) array of positions including the start.
        """
        dt = 1.0 / self._fps
        meta = self._metadata

        pos = state.position.copy()
        heading = float(state.heading)
        vel = float(state.speed)

        l_r = meta.wheelbase / 2 - (meta.rear_overhang - meta.front_overhang) / 2

        path = [pos.copy()]
        for a, s in zip(accels, steers):
            a = np.clip(a, -meta.max_acceleration, meta.max_acceleration)
            vel = max(0.0, vel + a * dt)
            beta = np.arctan(l_r * np.tan(s) / meta.wheelbase)
            dx = vel * np.cos(beta + heading) * dt
            dy = vel * np.sin(beta + heading) * dt
            pos = pos + np.array([dx, dy])
            d_theta = vel * np.tan(s) * np.cos(beta) / meta.wheelbase
            d_theta = np.clip(d_theta, -meta.max_angular_vel, meta.max_angular_vel)
            heading = (heading + d_theta * dt + np.pi) % (2 * np.pi) - np.pi
            path.append(pos.copy())

        return np.array(path)

    def _sample_trajectories(self, state: AgentState) -> List[np.ndarray]:
        """Generate candidate trajectories by sampling random action sequences."""
        max_accel = self._metadata.max_acceleration
        max_steer = 0.5  # radians (~29 degrees)
        trajectories = []
        for _ in range(self._n_samples):
            accels = np.random.uniform(-max_accel, max_accel, self._horizon)
            steers = np.random.uniform(-max_steer, max_steer, self._horizon)
            traj = self._simulate_trajectory(state, accels, steers)
            trajectories.append(traj)
        return trajectories

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _trajectory_error(self, traj: np.ndarray) -> float:
        """Mean closest-point distance from *traj* to the reference path."""
        if len(self._reference_waypoints) == 0:
            return float('inf')
        diffs = traj[:, None, :] - self._reference_waypoints[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        return float(np.mean(min_dists))

    # ------------------------------------------------------------------
    # PID tracking
    # ------------------------------------------------------------------

    def _track_trajectory(self, best_traj: np.ndarray,
                          state: AgentState) -> Action:
        """PID-track a waypoint on *best_traj* (mirrors WaypointManeuver)."""
        dists = np.linalg.norm(best_traj - state.position, axis=1)
        closest_idx = int(np.argmin(dists))

        if dists[-1] < self.WAYPOINT_MARGIN:
            target_idx = len(best_traj) - 1
        else:
            far = dists[closest_idx:]
            offset = np.argmax(far >= self.WAYPOINT_MARGIN)
            target_idx = closest_idx + offset

        target_wp = best_traj[target_idx]

        # Heading error
        direction = target_wp - state.position
        wp_heading = np.arctan2(direction[1], direction[0])
        heading_error = np.diff(np.unwrap([state.heading, wp_heading]))[0]

        # Velocity error
        target_speed = self._estimate_target_speed()
        vel_error = target_speed - state.speed

        acceleration, steering = self._controller.next_action(vel_error, heading_error)
        return Action(acceleration, steering, target_speed)

    def _estimate_target_speed(self) -> float:
        """Heuristic target speed from reference path point spacing."""
        if len(self._reference_waypoints) < 2:
            return 5.0
        diffs = np.diff(self._reference_waypoints, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        dt = 1.0 / self._fps
        vels = np.clip(segment_lengths / dt, 0.0,
                       self._metadata.max_acceleration * 10)
        return float(np.mean(vels))


# ======================================================================
# Frenet (reference-path) coordinate frame
# ======================================================================


class _FrenetFrame:
    """Coordinate transform between world (x, y) and Frenet (s, d) frames.

    The Frenet frame is defined by a reference path (sequence of waypoints):
      * **s** — arc-length distance along the path (longitudinal).
      * **d** — signed lateral offset from the path (positive = left).

    Precomputes cumulative arc lengths, unit tangents, and unit normals
    so that repeated transforms are fast.

    Args:
        waypoints: (N, 2) array of reference-path positions in world coords.
    """

    def __init__(self, waypoints: np.ndarray):
        if len(waypoints) < 2:
            raise ValueError("_FrenetFrame requires at least 2 waypoints")
        self._waypoints = np.asarray(waypoints, dtype=float)
        n = len(self._waypoints)

        # Segment vectors and lengths
        diffs = np.diff(self._waypoints, axis=0)  # (N-1, 2)
        seg_lengths = np.linalg.norm(diffs, axis=1)  # (N-1,)
        seg_lengths = np.maximum(seg_lengths, 1e-9)

        # Cumulative arc length at each waypoint
        self._arc_lengths = np.zeros(n)
        self._arc_lengths[1:] = np.cumsum(seg_lengths)
        self._total_length = self._arc_lengths[-1]

        # Unit tangent per segment (N-1, 2)
        self._seg_tangents = diffs / seg_lengths[:, None]

        # Unit normal per segment (rotate tangent 90° CCW → points left)
        self._seg_normals = np.column_stack([
            -self._seg_tangents[:, 1],
            self._seg_tangents[:, 0],
        ])

        # Tangent angle per segment
        self._seg_angles = np.arctan2(
            self._seg_tangents[:, 1], self._seg_tangents[:, 0],
        )

    # ------------------------------------------------------------------
    # World → Frenet
    # ------------------------------------------------------------------

    def world_to_frenet(self, x: float, y: float,
                        heading: float = None,
                        vx: float = None, vy: float = None):
        """Convert a world-frame point (and optionally heading / velocity)
        to Frenet coordinates.

        Args:
            x, y: World position.
            heading: World heading (rad). If given, a relative heading is
                returned.
            vx, vy: World velocity components. If given, longitudinal and
                lateral velocities in the Frenet frame are returned.

        Returns:
            A dict with keys ``'s'``, ``'d'``, and optionally
            ``'heading'``, ``'vs'``, ``'vd'``.
        """
        seg_idx, s, d = self._project(x, y)
        result = {'s': s, 'd': d}

        if heading is not None:
            angle = self._seg_angles[seg_idx]
            rel = (heading - angle + np.pi) % (2 * np.pi) - np.pi
            result['heading'] = rel

        if vx is not None and vy is not None:
            t = self._seg_tangents[seg_idx]
            n = self._seg_normals[seg_idx]
            result['vs'] = vx * t[0] + vy * t[1]
            result['vd'] = vx * n[0] + vy * n[1]

        return result

    def world_to_frenet_batch(self, positions: np.ndarray):
        """Convert an (M, 2) array of world positions to (M,) s and d arrays.

        Returns:
            (s_array, d_array) — each of shape (M,).
        """
        s_arr = np.empty(len(positions))
        d_arr = np.empty(len(positions))
        for i, (px, py) in enumerate(positions):
            _, s_arr[i], d_arr[i] = self._project(px, py)
        return s_arr, d_arr

    # ------------------------------------------------------------------
    # Frenet → World
    # ------------------------------------------------------------------

    def frenet_to_world(self, s: float, d: float,
                        heading: float = None,
                        vs: float = None, vd: float = None):
        """Convert Frenet coordinates back to world frame.

        Args:
            s: Arc-length position along path.
            d: Signed lateral offset (positive = left of path).
            heading: Relative heading in Frenet frame. If given, world
                heading is returned.
            vs, vd: Longitudinal / lateral velocity. If given, world
                velocity components are returned.

        Returns:
            A dict with keys ``'x'``, ``'y'``, and optionally
            ``'heading'``, ``'vx'``, ``'vy'``.
        """
        seg_idx, ref_x, ref_y, angle = self._interpolate(s)
        t = self._seg_tangents[seg_idx]
        n = self._seg_normals[seg_idx]

        world_x = ref_x + d * n[0]
        world_y = ref_y + d * n[1]
        result = {'x': world_x, 'y': world_y}

        if heading is not None:
            result['heading'] = (heading + angle + np.pi) % (2 * np.pi) - np.pi

        if vs is not None and vd is not None:
            result['vx'] = vs * t[0] + vd * n[0]
            result['vy'] = vs * t[1] + vd * n[1]

        return result

    def frenet_to_world_batch(self, s_arr: np.ndarray,
                              d_arr: np.ndarray) -> np.ndarray:
        """Convert arrays of (s, d) to an (M, 2) world-position array."""
        out = np.empty((len(s_arr), 2))
        for i in range(len(s_arr)):
            r = self.frenet_to_world(s_arr[i], d_arr[i])
            out[i] = [r['x'], r['y']]
        return out

    # ------------------------------------------------------------------
    # Road boundaries
    # ------------------------------------------------------------------

    def road_boundaries(self, s_values: np.ndarray,
                        scenario_map: Map,
                        search_radius: float = 5.0):
        """Query drivable-lane boundaries at sampled arc-length positions.

        For each *s* value, casts a perpendicular ray through the
        corresponding path point and intersects it with drivable lane
        boundaries to find left and right offsets.

        Args:
            s_values: (K,) array of longitudinal positions along the path.
            scenario_map: Map with drivable lane information.
            search_radius: Half-length of the perpendicular ray (metres).

        Returns:
            (d_left, d_right) — each (K,) arrays.  ``d_left`` is positive,
            ``d_right`` is negative, defining the corridor
            ``d_right <= d <= d_left`` at each step.
        """
        k = len(s_values)
        d_left = np.full(k, search_radius)
        d_right = np.full(k, -search_radius)

        for i, s in enumerate(s_values):
            seg_idx, rx, ry, _ = self._interpolate(s)
            n = self._seg_normals[seg_idx]

            # Perpendicular ray endpoints
            p_left = np.array([rx + search_radius * n[0],
                               ry + search_radius * n[1]])
            p_right = np.array([rx - search_radius * n[0],
                                ry - search_radius * n[1]])
            ray = LineString([p_right, p_left])

            # Find drivable lanes near this point
            ref_pt = np.array([rx, ry])
            lanes = scenario_map.lanes_at(ref_pt, drivable_only=True,
                                          max_distance=search_radius)
            if not lanes:
                continue

            # Intersect ray with each lane boundary, track extremes
            best_left = 0.0
            best_right = 0.0
            for lane in lanes:
                if lane.boundary is None:
                    continue
                ring = lane.boundary.exterior
                inter = ray.intersection(ring)
                if inter.is_empty:
                    continue
                # Collect intersection points
                pts = []
                if inter.geom_type == 'Point':
                    pts = [np.array([inter.x, inter.y])]
                elif inter.geom_type == 'MultiPoint':
                    pts = [np.array([p.x, p.y]) for p in inter.geoms]
                elif inter.geom_type == 'LineString':
                    pts = [np.array(c) for c in inter.coords]
                elif inter.geom_type in ('GeometryCollection',
                                         'MultiLineString'):
                    for g in inter.geoms:
                        if g.geom_type == 'Point':
                            pts.append(np.array([g.x, g.y]))
                        elif g.geom_type == 'LineString':
                            pts.extend(np.array(c) for c in g.coords)

                for pt in pts:
                    # Signed distance along normal direction
                    offset = np.dot(pt - ref_pt, n)
                    if offset >= 0:
                        best_left = max(best_left, offset)
                    else:
                        best_right = min(best_right, offset)

            if best_left > 0:
                d_left[i] = best_left
            if best_right < 0:
                d_right[i] = best_right

        return d_left, d_right

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_length(self) -> float:
        """Total arc length of the reference path."""
        return self._total_length

    @property
    def waypoints(self) -> np.ndarray:
        return self._waypoints

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project(self, x: float, y: float) -> Tuple[int, float, float]:
        """Project a world point onto the reference path.

        Returns:
            (seg_idx, s, d) — index of the closest segment, arc-length
            coordinate, and signed lateral offset.
        """
        pt = np.array([x, y])
        best_seg = 0
        best_s = 0.0
        best_d = 0.0
        best_dist_sq = float('inf')

        wps = self._waypoints
        for i in range(len(wps) - 1):
            a = wps[i]
            ab = wps[i + 1] - a
            seg_len_sq = np.dot(ab, ab)
            if seg_len_sq < 1e-18:
                t = 0.0
            else:
                t = np.clip(np.dot(pt - a, ab) / seg_len_sq, 0.0, 1.0)

            proj = a + t * ab
            diff = pt - proj
            dist_sq = np.dot(diff, diff)

            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_seg = i
                seg_len = np.sqrt(seg_len_sq)
                best_s = self._arc_lengths[i] + t * seg_len
                # Signed lateral offset: positive = left of path
                n = self._seg_normals[i]
                best_d = np.dot(diff, n)

        return best_seg, best_s, best_d

    def _interpolate(self, s: float) -> Tuple[int, float, float, float]:
        """Interpolate a point on the path at arc-length *s*.

        Returns:
            (seg_idx, x, y, angle) — segment index, world position,
            and tangent angle at that point.
        """
        s = np.clip(s, 0.0, self._total_length)

        # Find segment: arc_lengths[seg] <= s < arc_lengths[seg+1]
        idx = np.searchsorted(self._arc_lengths, s, side='right') - 1
        idx = np.clip(idx, 0, len(self._waypoints) - 2)

        seg_start_s = self._arc_lengths[idx]
        seg_len = self._arc_lengths[idx + 1] - seg_start_s
        if seg_len < 1e-9:
            t = 0.0
        else:
            t = (s - seg_start_s) / seg_len

        a = self._waypoints[idx]
        b = self._waypoints[idx + 1]
        pt = a + t * (b - a)
        return int(idx), float(pt[0]), float(pt[1]), float(self._seg_angles[idx])


# ======================================================================
# Two-stage optimisation policy (MILP + NLP)
# ======================================================================


class TwoStageOPT:
    """Two-stage Frenet-frame trajectory optimisation: MILP + CasADi NLP.

    All optimisation is performed in Frenet (reference-path) coordinates,
    where road boundaries become simple box constraints
    ``d_right <= d <= d_left``.

    **Stage 1 — MILP (coarse trajectory).**
    A smooth approximation of MILP over a point-mass model
    ``[s, d, vs, vd]`` in Frenet coordinates minimises L2 tracking error
    subject to ZOH dynamics, kinematic feasibility, road boundaries,
    and smooth penalty-based collision avoidance.

    **Stage 2 — NLP (refined trajectory).**
    A CasADi + IPOPT nonlinear program over the bicycle kinematic model
    ``[s, d, phi, v]`` in Frenet coordinates, with elliptical collision
    regions, road boundaries, jerk constraints, and the paper's cost
    function.

    The planning timestep ``dt`` (default 0.2 s) is independent of the
    simulation framerate.  Each MPC call plans at ``dt`` resolution over
    ``horizon`` steps (default 40 = 8 s), but only the first action is
    applied for one simulation step (receding-horizon MPC).

    **MILP and NLP parameters are configured separately** via ``milp_params``
    and ``nlp_params`` dicts. See ``MILP_DEFAULTS`` and ``NLP_DEFAULTS`` for
    available parameters and their defaults.

    MILP Parameters (point-mass model):
        - a_s_min/a_s_max: Longitudinal acceleration bounds (m/s^2)
        - a_d_min/a_d_max: Lateral acceleration bounds (m/s^2)
        - jerk_s_max: Longitudinal jerk max (m/s^3)
        - jerk_d_max: Lateral jerk max (m/s^3)
        - vs_min/vs_max: Longitudinal velocity bounds (m/s)
        - vd_min/vd_max: Lateral velocity bounds (m/s)
        - rho: Kinematic feasibility ratio (vs >= rho * |vd|)
        - w_s: Weight for longitudinal position tracking
        - w_d: Weight for lateral position tracking
        - w_v: Weight for velocity tracking
        - w_a_s: Weight for longitudinal acceleration
        - w_a_d: Weight for lateral acceleration

    NLP Parameters (bicycle model):
        - a_min/a_max: Acceleration bounds (m/s^2)
        - delta_max: Max steering angle magnitude (rad)
        - jerk_max: Max jerk (m/s^3)
        - v_min/v_max: Velocity bounds (m/s)
        - w_s: Weight for longitudinal position tracking
        - w_d: Weight for lateral position tracking
        - w_v: Weight for velocity tracking
        - w_a: Weight for acceleration norm
        - w_delta: Weight for steering angle norm

    Args:
        fps: Simulation framerate.
        metadata: Agent physical metadata (wheelbase, limits, etc.).
        reference_waypoints: Concatenated A* reference path (N, 2).
        scenario_map: Road layout for boundary queries. May be None.
        horizon: Number of planning steps (default 40).
        dt: Planning timestep in seconds (default 0.2).
        target_speed: Desired cruising speed (m/s).
        collision_margin: Extra safety margin around obstacles (m).
        big_m: Big-M constant for collision avoidance.
        milp_params: Dict of MILP-specific parameters (see MILP_DEFAULTS).
        nlp_params: Dict of NLP-specific parameters (see NLP_DEFAULTS).

    Example:
        >>> policy = TwoStageOPT(
        ...     fps=20, metadata=meta, reference_waypoints=waypoints,
        ...     milp_params={'rho': 2.0, 'w_s': 1.0, 'a_s_max': 2.0},
        ...     nlp_params={'delta_max': 0.3, 'w_d': 3.0, 'jerk_max': 0.3},
        ... )
    """

    # Paper default parameters
    DEFAULT_HORIZON = 20
    DEFAULT_DT = 0.2
    DEFAULT_BIG_M = 1000.0
    N_OBS_MAX = 10  # pre-allocated obstacle slots in NLP


    # # MILP default parameters (point-mass model with [s, d, vs, vd] state)
    # MILP_DEFAULTS = {
    #     'a_s_min': -3.0,      # Longitudinal acceleration min (m/s^2)
    #     'a_s_max': 3.0,       # Longitudinal acceleration max (m/s^2)
    #     'a_d_min': -0.5,      # Lateral acceleration min (m/s^2)
    #     'a_d_max': 0.5,       # Lateral acceleration max (m/s^2)
    #     'jerk_s_max': 0.5,    # Longitudinal jerk max (m/s^3)
    #     'jerk_d_max': 0.1,    # Lateral jerk max (m/s^3)
    #     'vs_min': 0.0,        # Longitudinal velocity min (m/s)
    #     'vs_max': 3.0,        # Longitudinal velocity max (m/s)
    #     'vd_min': -1.0,       # Lateral velocity min (m/s)
    #     'vd_max': 1.0,        # Lateral velocity max (m/s)
    #     'rho': 1.5,           # Kinematic feasibility ratio (vs >= rho * |vd|)
    #     'w_s': 0.9,           # Weight for longitudinal position tracking
    #     'w_d': 0.05,          # Weight for lateral position tracking
    #     'w_v': 0.5,           # Weight for velocity tracking
    #     'w_a_s': 0.4,        # Weight for longitudinal acceleration
    #     'w_a_d': 0.4,        # Weight for lateral acceleration
    # }

    # # NLP default parameters (bicycle model with [s, d, phi, v] state)
    # NLP_DEFAULTS = {
    #     'a_min': -3.0,        # Acceleration min (m/s^2)
    #     'a_max': 3.0,         # Acceleration max (m/s^2)
    #     'delta_max': 0.45,    # Max steering angle magnitude (rad)
    #     'jerk_max': 0.5,      # Max jerk (m/s^3)
    #     'v_min': 0.0,         # Velocity min (m/s)
    #     'v_max': 10.0,        # Velocity max (m/s)
    #     'w_s': 0.1,           # Weight for longitudinal position tracking
    #     'w_d': 0.05,           # Weight for lateral position tracking
    #     'w_v': 2.5,           # Weight for velocity tracking
    #     'w_a': 1.0,           # Weight for acceleration norm
    #     'w_delta': 2.0,       # Weight for steering angle norm
    # }

    # MILP default parameters (point-mass model with [s, d, vs, vd] state)
    MILP_DEFAULTS = {
        'a_s_min': -3.0,      # Longitudinal acceleration min (m/s^2)
        'a_s_max': 3.0,       # Longitudinal acceleration max (m/s^2)
        'a_d_min': -3.0,      # Lateral acceleration min (m/s^2)
        'a_d_max': 3.0,       # Lateral acceleration max (m/s^2)
        'jerk_s_max': 1.0,    # Longitudinal jerk max (m/s^3)
        'jerk_d_max': 1.0,    # Lateral jerk max (m/s^3)
        'vs_min': 0.0,        # Longitudinal velocity min (m/s)
        'vs_max': 8.0,       # Longitudinal velocity max (m/s)
        'vd_min': -8.0,       # Lateral velocity min (m/s)
        'vd_max': 8.0,        # Lateral velocity max (m/s)
        'rho': 1.5,           # Kinematic feasibility ratio (vs >= rho * |vd|)
        'w_s': 0.9,           # Weight for longitudinal position tracking
        'w_d': 10.0,          # Weight for lateral position tracking
        'w_v': 0.1,           # Weight for velocity tracking
        'w_a_s': 0.4,        # Weight for longitudinal acceleration
        'w_a_d': 0.4,        # Weight for lateral acceleration
    }

    # NLP default parameters (bicycle model with [s, d, phi, v] state)
    NLP_DEFAULTS = {
        'a_min': -3.0,        # Acceleration min (m/s^2)
        'a_max': 3.0,         # Acceleration max (m/s^2)
        'delta_max': 0.45,    # Max steering angle magnitude (rad)
        'delta_rate_max': 0.8,  # Max steering rate (rad/s)
        'jerk_max': 0.5,      # Max jerk (m/s^3)
        'v_min': 0.0,         # Velocity min (m/s)
        'v_max': 8.0,        # Velocity max (m/s)
        'w_s': 0.9,           # Weight for longitudinal position tracking
        'w_d': 500.0,           # Weight for lateral position tracking
        'w_v': 0.1,           # Weight for velocity tracking
        'w_a': 1.0,           # Weight for acceleration norm
        'w_delta': 2.0,       # Weight for steering angle norm
    }

    def __init__(self,
                 fps: int,
                 metadata: AgentMetadata,
                 reference_waypoints: np.ndarray,
                 scenario_map: Optional[Map] = None,
                 horizon: int = None,
                 dt: float = None,
                 target_speed: float = 5.0,
                 collision_margin: float = 0.5,
                 big_m: float = None,
                 milp_params: Optional[Dict] = None,
                 nlp_params: Optional[Dict] = None,
                 use_prev_nlp_on_fail: bool = True,
                 debug_plot_nlp: bool = True,
                 # Legacy params (for back-compat, override milp/nlp_params)
                 delta_max: float = None,
                 a_min: float = None,
                 a_max: float = None,
                 jerk_max: float = None,
                 v_max: float = None,
                 milp_rho: float = None,
                 w_x: float = None,
                 w_v: float = None,
                 w_y: float = None,
                 w_a: float = None,
                 w_delta: float = None,
                 max_steer=None, w_ref=None, w_speed=None, w_smooth=None,
                 **kwargs):
        self._fps = fps
        self._dt_sim = 1.0 / fps
        self._dt = dt if dt is not None else self.DEFAULT_DT
        self._metadata = metadata
        self._reference_waypoints = np.asarray(reference_waypoints, dtype=float)
        self._scenario_map = scenario_map
        self._horizon = horizon if horizon is not None else self.DEFAULT_HORIZON
        self._target_speed = target_speed
        self._big_m = big_m if big_m is not None else self.DEFAULT_BIG_M
        self._use_prev_nlp_on_fail = use_prev_nlp_on_fail
        self._debug_plot_nlp = debug_plot_nlp

        # Build MILP parameters from defaults + provided overrides
        self._milp = dict(self.MILP_DEFAULTS)
        if milp_params is not None:
            self._milp.update(milp_params)

        # Build NLP parameters from defaults + provided overrides
        self._nlp = dict(self.NLP_DEFAULTS)
        if nlp_params is not None:
            self._nlp.update(nlp_params)

        # Apply legacy parameter overrides (for backwards compatibility)
        if milp_rho is not None:
            self._milp['rho'] = milp_rho
        if v_max is not None:
            self._milp['vs_max'] = v_max  # Legacy v_max applies to longitudinal
            self._nlp['v_max'] = v_max
        if a_min is not None:
            self._milp['a_s_min'] = a_min
            self._milp['a_d_min'] = a_min
            self._nlp['a_min'] = a_min
        if a_max is not None:
            self._milp['a_s_max'] = a_max
            self._milp['a_d_max'] = a_max
            self._nlp['a_max'] = a_max
        if delta_max is not None:
            self._nlp['delta_max'] = delta_max
        if jerk_max is not None:
            self._nlp['jerk_max'] = jerk_max
        if w_x is not None:
            self._milp['w_s'] = w_x
            self._nlp['w_s'] = w_x
        if w_y is not None:
            self._milp['w_d'] = w_y
            self._nlp['w_d'] = w_y
        if w_v is not None:
            self._milp['w_v'] = w_v
            self._nlp['w_v'] = w_v
        if w_a is not None:
            self._milp['w_a_s'] = w_a
            self._milp['w_a_d'] = w_a
            self._nlp['w_a'] = w_a
        if w_delta is not None:
            self._nlp['w_delta'] = w_delta

        # Collision
        self._collision_margin = collision_margin
        self._ego_length = metadata.length
        self._ego_width = metadata.width
        self._wheelbase = metadata.wheelbase

        # Frenet frame (rebuilt when reference waypoints are available)
        self._frenet: Optional[_FrenetFrame] = None
        if len(self._reference_waypoints) >= 2:
            self._frenet = _FrenetFrame(self._reference_waypoints)

        # MPC state
        self._prev_milp_states: Optional[np.ndarray] = None  # (H+1, 4) Frenet
        self._prev_nlp_states: Optional[np.ndarray] = None   # (H+1, 4) Frenet
        self._prev_nlp_controls: Optional[np.ndarray] = None # (H, 2)
        self._last_rollout: Optional[np.ndarray] = None       # (H+1, 4) world
        self._last_milp_rollout: Optional[np.ndarray] = None  # (H+1, 2) world positions from MILP
        self._ref_start_idx: int = 0
        self._step_count: int = 0

        # Obstacle data (stored for plotter access)
        self._last_obstacles: Optional[List] = None
        self._last_other_agents: Optional[Dict] = None

        # Build CasADi NLP structure once
        self._nlp_solver = None
        self._nlp_built = False

        # Debug plotting figure (created on first use)
        self._debug_fig = None
        self._debug_axes = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, state: AgentState,
                      other_agents: Optional[Dict] = None,
                      agent_trajectories: Optional[Dict[int, np.ndarray]] = None) -> tuple:
        """MPC step: solve MILP + NLP in Frenet frame, return first action.

        Args:
            state: Current ego agent state.
            other_agents: Dict mapping agent_id -> AgentState for other
                vehicles (used for collision avoidance). May be None.
            agent_trajectories: Optional dict mapping agent_id -> (T, 2) array
                of planned world positions over the planning horizon. When
                provided, these trajectories are used instead of constant-velocity
                prediction for collision avoidance. May be None.

        Returns:
            (action, [trajectory_positions], 0)
        """
        if self._frenet is None:
            # No reference path — coast
            action = Action(acceleration=0.0, steer_angle=0.0,
                            target_speed=self._target_speed)
            return action, [np.array([state.position])], 0

        H = self._horizon
        dt = self._dt

        # Advance reference window
        self._advance_reference_window(state.position)

        # Convert ego state to Frenet
        frenet_state = self._state_to_frenet(state)

        # Sample road boundaries at each planning step
        s_values = np.array([frenet_state[0] + frenet_state[3] *
                             np.cos(frenet_state[2]) * k * dt
                             for k in range(H + 1)])
        # Clamp s values to path range
        s_values = np.clip(s_values, 0.0, self._frenet.total_length)
        road_left, road_right = self._sample_road_boundaries(s_values)

        # Predict obstacles in Frenet frame
        obstacles = self._predict_obstacles(other_agents, self._frenet, agent_trajectories)
        self._last_obstacles = obstacles
        self._last_other_agents = other_agents

        # Goal in Frenet: s_goal = total path length, d_goal = 0
        s_goal = self._frenet.total_length

        self._step_count += 1

        # --- Stage 1: MILP ---
        t_milp_start = time.time()
        milp_states = self._solve_milp(frenet_state, road_left, road_right,
                                       obstacles)
        t_milp = time.time() - t_milp_start

        if milp_states is None:
            print(f"[Step {self._step_count:4d}] MILP: FAILED ({t_milp*1000:.1f}ms)")
            action = Action(acceleration=0.0, steer_angle=0.0,
                            target_speed=self._target_speed)
            return action, [np.array([state.position])], 0

        # Prepare NLP warm-start from MILP solution
        warm_states, warm_controls = self._milp_to_nlp_warmstart(
            milp_states, frenet_state)
        self._prev_milp_states = milp_states.copy()

        # Store MILP trajectory in world coords for plotting
        self._last_milp_rollout = self._milp_states_to_world(milp_states)

        # --- Stage 2: NLP ---
        t_nlp_start = time.time()
        nlp_states, nlp_controls, nlp_ok = self._solve_nlp(
            frenet_state, warm_states, warm_controls,
            road_left, road_right, obstacles)
        t_nlp = time.time() - t_nlp_start

        if not nlp_ok:
            # NLP failed - decide which fallback to use
            if (self._use_prev_nlp_on_fail and
                self._prev_nlp_states is not None and
                self._prev_nlp_controls is not None):
                # Use previous NLP solution
                final_states = self._prev_nlp_states.copy()
                final_controls = self._prev_nlp_controls.copy()
                nlp_status = "FAILED(prev)"
            else:
                # Fall back to MILP warm-start
                final_states = warm_states
                final_controls = warm_controls
                nlp_status = "FAILED(milp)"
        else:
            final_states = nlp_states
            final_controls = nlp_controls
            nlp_status = "OK"

        # Print concise status
        print(f"[Step {self._step_count:4d}] MILP: OK ({t_milp*1000:.1f}ms) | NLP: {nlp_status} ({t_nlp*1000:.1f}ms)")

        # Print velocity profile
        velocities = final_states[:, 3]  # [s, d, phi, v] -> v is index 3
        v_str = " ".join([f"{v:.2f}" for v in velocities])
        print(f"  Velocity profile (H={len(velocities)-1}): [{v_str}]")

        # Print steering control profile
        steerings = final_controls[:, 1]  # [a, delta] -> delta is index 1
        delta_str = " ".join([f"{np.degrees(d):+.1f}" for d in steerings])
        print(f"  Steering profile (deg): [{delta_str}]")

        self._prev_nlp_states = final_states.copy()
        self._prev_nlp_controls = final_controls.copy()

        # Convert Frenet trajectory to world coordinates
        self._last_rollout = self._frenet_trajectory_to_world(final_states)
        trajectory = self._last_rollout[:, :2]

        # Extract first control action
        accel = float(final_controls[0, 0])
        steer = float(final_controls[0, 1])
        action = Action(
            acceleration=accel,
            steer_angle=steer,
            target_speed=self._target_speed,
        )
        return action, [trajectory], 0

    @property
    def last_rollout(self) -> Optional[np.ndarray]:
        """Full state rollout from the most recent optimisation.

        Returns:
            (H+1, 4) array of [x, y, heading, speed] in WORLD coords,
            or None if :meth:`select_action` has not been called yet.
        """
        return self._last_rollout

    @property
    def last_milp_rollout(self) -> Optional[np.ndarray]:
        """MILP trajectory from the most recent optimisation.

        Returns:
            (H+1, 2) array of [x, y] positions in WORLD coords,
            or None if :meth:`select_action` has not been called yet.
        """
        return self._last_milp_rollout

    @property
    def last_obstacles(self) -> Optional[List]:
        """Predicted obstacles from the most recent MPC step."""
        return self._last_obstacles

    @property
    def last_other_agents(self) -> Optional[Dict]:
        """Other-agent states passed to the most recent MPC step."""
        return self._last_other_agents

    @property
    def frenet_frame(self) -> Optional['_FrenetFrame']:
        """The Frenet coordinate frame built from the reference path."""
        return self._frenet

    @property
    def collision_margin(self) -> float:
        """Extra safety margin around obstacles (m)."""
        return self._collision_margin

    @property
    def ego_length(self) -> float:
        """Ego vehicle length (m)."""
        return self._ego_length

    @property
    def ego_width(self) -> float:
        """Ego vehicle width (m)."""
        return self._ego_width

    @property
    def milp_params(self) -> Dict:
        """MILP-specific parameters (copy of internal dict)."""
        return dict(self._milp)

    @property
    def nlp_params(self) -> Dict:
        """NLP-specific parameters (copy of internal dict)."""
        return dict(self._nlp)

    @property
    def horizon(self) -> int:
        """Planning horizon (number of steps)."""
        return self._horizon

    @property
    def dt(self) -> float:
        """Planning timestep (seconds)."""
        return self._dt

    def reset(self):
        """Reset all MPC state."""
        self._prev_milp_states = None
        self._prev_nlp_states = None
        self._prev_nlp_controls = None
        self._last_rollout = None
        self._ref_start_idx = 0
        self._nlp_built = False
        self._nlp_solver = None
        self._last_obstacles = None
        self._last_other_agents = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Frenet state conversion
    # ------------------------------------------------------------------

    def _state_to_frenet(self, state: AgentState) -> np.ndarray:
        """Convert AgentState to Frenet state [s, d, phi, v].

        Args:
            state: World-frame agent state.

        Returns:
            [s, d, phi, v] in Frenet frame.
        """
        f = self._frenet.world_to_frenet(
            float(state.position[0]), float(state.position[1]),
            heading=float(state.heading),
        )
        return np.array([f['s'], f['d'], f['heading'], float(state.speed)])

    def _frenet_trajectory_to_world(self, frenet_states: np.ndarray) -> np.ndarray:
        """Convert (K, 4) Frenet states [s, d, phi, v] to world [x, y, heading, speed].

        Args:
            frenet_states: (K, 4) array of [s, d, phi, v].

        Returns:
            (K, 4) array of [x, y, heading, speed] in world frame.
        """
        K = len(frenet_states)
        world = np.empty((K, 4))
        for i in range(K):
            s_i, d_i, phi_i, v_i = frenet_states[i]
            w = self._frenet.frenet_to_world(s_i, d_i, heading=phi_i)
            world[i] = [w['x'], w['y'], w['heading'], v_i]
        return world

    def _milp_states_to_world(self, milp_states: np.ndarray) -> np.ndarray:
        """Convert (K, 4) MILP states [s, d, vs, vd] to world [x, y] positions.

        Args:
            milp_states: (K, 4) array of [s, d, vs, vd].

        Returns:
            (K, 2) array of [x, y] in world frame.
        """
        K = len(milp_states)
        world = np.empty((K, 2))
        for i in range(K):
            s_i, d_i = milp_states[i, 0], milp_states[i, 1]
            w = self._frenet.frenet_to_world(s_i, d_i)
            world[i] = [w['x'], w['y']]
        return world

    # ------------------------------------------------------------------
    # Road boundary sampling
    # ------------------------------------------------------------------

    def _sample_road_boundaries(self, s_values: np.ndarray):
        """Sample road boundaries at given s positions.

        Returns:
            (d_left, d_right) arrays, each of shape (len(s_values),).
            d_left > 0 (left of centre), d_right < 0 (right of centre).
            If no map is available, defaults to +/- 3.5 m.
        """
        if self._scenario_map is not None and self._frenet is not None:
            # Use larger search radius to find opposite lanes too
            return self._frenet.road_boundaries(s_values, self._scenario_map,
                                                search_radius=15.0)
        # Fallback: standard lane width
        return (np.full(len(s_values), 3.5),
                np.full(len(s_values), -3.5))

    # ------------------------------------------------------------------
    # Obstacle prediction
    # ------------------------------------------------------------------

    def _predict_obstacles(self, other_agents, frenet, agent_trajectories=None,
                           trajectory_dt=None):
        """Predict obstacle positions over the planning horizon.

        Uses provided trajectories if available, otherwise falls back to
        constant-velocity propagation.

        IMPORTANT: The planning timestep (self._dt) may differ from the
        simulation timestep. Provided trajectories are assumed to be at
        the simulation timestep (trajectory_dt or 1/fps). This method
        resamples them to match the planning timestep.

        Args:
            other_agents: Dict {agent_id: AgentState} or None.
            frenet: _FrenetFrame instance.
            agent_trajectories: Optional dict {agent_id: (T, 2) array} of
                planned world positions at SIMULATION timestep. When provided,
                these are resampled to planning timestep.
            trajectory_dt: Timestep of provided trajectories (default: 1/fps).

        Returns:
            List of dicts, each with keys:
                's': (H+1,) arc-length positions at planning timestep
                'd': (H+1,) lateral positions at planning timestep
                'world_positions': (H+1, 2) world coordinates for plotting
                'length': vehicle length
                'width': vehicle width
                'heading': vehicle heading
                'agent_id': agent ID
                'uses_planned_trajectory': bool indicating if actual trajectory was used
        """
        if other_agents is None or frenet is None:
            return []

        H = self._horizon
        dt_plan = self._dt  # Planning timestep (e.g., 0.2s)
        dt_traj = trajectory_dt if trajectory_dt is not None else self._dt_sim  # Trajectory timestep (e.g., 0.05s)

        # Ratio for resampling: how many trajectory steps per planning step
        resample_ratio = dt_plan / dt_traj if dt_traj > 0 else 1.0

        obstacles = []

        for aid, agent_state in other_agents.items():
            pos = np.array(agent_state.position, dtype=float)
            vel = np.array(agent_state.velocity, dtype=float)

            uses_planned = False

            # Use provided trajectory if available, otherwise constant-velocity
            if agent_trajectories is not None and aid in agent_trajectories:
                provided_traj = np.asarray(agent_trajectories[aid], dtype=float)
                n_provided = len(provided_traj)

                # Resample trajectory from simulation timestep to planning timestep
                # Planning step k corresponds to time t = k * dt_plan
                # which is trajectory index i = k * dt_plan / dt_traj = k * resample_ratio
                world_positions = np.empty((H + 1, 2))
                for k in range(H + 1):
                    traj_idx_float = k * resample_ratio
                    traj_idx = int(traj_idx_float)

                    if traj_idx + 1 < n_provided:
                        # Linear interpolation between trajectory points
                        alpha = traj_idx_float - traj_idx
                        world_positions[k] = (1 - alpha) * provided_traj[traj_idx] + alpha * provided_traj[traj_idx + 1]
                    elif traj_idx < n_provided:
                        # Use last available point
                        world_positions[k] = provided_traj[traj_idx]
                    else:
                        # Extrapolate from last position using velocity
                        time_beyond = (k * dt_plan) - ((n_provided - 1) * dt_traj)
                        world_positions[k] = provided_traj[-1] + vel * time_beyond

                uses_planned = True
            else:
                # Fallback: constant-velocity prediction
                world_positions = np.empty((H + 1, 2))
                for k in range(H + 1):
                    world_positions[k] = pos + vel * k * dt_plan

            # Compute heading at each trajectory point from direction of motion
            headings = np.empty(H + 1)
            headings[0] = float(agent_state.heading)  # Use actual heading for first point
            for k in range(1, H + 1):
                dx = world_positions[k, 0] - world_positions[k - 1, 0]
                dy = world_positions[k, 1] - world_positions[k - 1, 1]
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    headings[k] = np.arctan2(dy, dx)
                else:
                    # No movement - use previous heading
                    headings[k] = headings[k - 1]

            # Transform to Frenet
            s_arr, d_arr = frenet.world_to_frenet_batch(world_positions)

            obs_meta = getattr(agent_state, 'metadata', None)
            obs_length = obs_meta.length if obs_meta is not None else 4.5
            obs_width = obs_meta.width if obs_meta is not None else 1.8

            obstacles.append({
                's': s_arr,
                'd': d_arr,
                'world_positions': world_positions,  # At planning timestep for plotting
                'headings': headings,  # Heading at each timestep
                'length': obs_length,
                'width': obs_width,
                'heading': float(agent_state.heading),  # Initial heading (kept for compatibility)
                'agent_id': aid,
                'uses_planned_trajectory': uses_planned,
            })

        return obstacles

    # ------------------------------------------------------------------
    # MPC helpers
    # ------------------------------------------------------------------

    def _advance_reference_window(self, position: np.ndarray):
        """Move ``_ref_start_idx`` to the nearest waypoint ahead."""
        if len(self._reference_waypoints) == 0:
            return
        dists = np.linalg.norm(
            self._reference_waypoints[self._ref_start_idx:] - position, axis=1)
        self._ref_start_idx += int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Stage 1: MILP (cvxpy)
    # ------------------------------------------------------------------

    def _solve_milp(self, frenet_state, road_left, road_right, obstacles):
        """Solve Stage 1 optimization using smooth approximation with CasADi + IPOPT.

        Based on "A Two-Stage Optimization-based Motion Planner for Safe Urban
        Driving" (Eiras et al.), but using smooth softplus approximation of max()
        instead of MILP formulation.

        Key aspects:
        - Ego vehicle is treated as a POINT MASS
        - Obstacles are RECTANGLES enlarged by ego dimensions (Minkowski sum)
        - Uses smooth μ-based collision avoidance (pass-left only):
            μ_i_k = softplus(s_min - s_k) + softplus(s_k - s_max) + softplus(d_min - d_k)
            where softplus(a) = (1/β) * log(1 + exp(β * a)) ≈ max(a, 0)
            Constraint: d_k >= d_max - M * μ_i_k

        Args:
            frenet_state: [s, d, phi, v] current state in Frenet.
            road_left: (H+1,) left boundary offsets (positive).
            road_right: (H+1,) right boundary offsets (negative).
            obstacles: List of obstacle dicts from _predict_obstacles.

        Returns:
            (H+1, 4) array of states [s, d, vs, vd] in Frenet,
            or None if solver failed.
        """
        H = self._horizon
        dt = self._dt
        M = 1e4  # Big-M from paper
        beta = 10.0  # Softplus sharpness (higher = closer to true max)

        # MILP-specific parameters
        milp = self._milp
        rho = milp['rho']
        a_s_min, a_s_max = milp['a_s_min'], milp['a_s_max']
        a_d_min, a_d_max = milp['a_d_min'], milp['a_d_max']
        jerk_s_max, jerk_d_max = milp['jerk_s_max'], milp['jerk_d_max']
        vs_min, vs_max = milp['vs_min'], milp['vs_max']
        vd_min, vd_max = milp['vd_min'], milp['vd_max']
        w_s, w_d, w_v = milp['w_s'], milp['w_d'], milp['w_v']
        w_a_s, w_a_d = milp['w_a_s'], milp['w_a_d']

        # Initial Frenet velocity components
        s0, d0, phi0, v0 = frenet_state
        vs0 = v0 * np.cos(phi0)
        vd0 = v0 * np.sin(phi0)

        N_obs = min(len(obstacles), self.N_OBS_MAX)

        # Goal reference: point vehicle would reach at target speed over the horizon
        v_goal = self._target_speed
        s_goal = min(s0 + v_goal * H * dt, self._frenet.total_length)

        # Ego dimensions for Minkowski sum
        dx = self._ego_length / 2.0
        dy = self._ego_width / 2.0

        # Softplus function: smooth approximation of max(a, 0)
        # Numerically stable form: softplus(a) = max(a,0) + (1/β)*log(1+exp(-β*|a|))
        # This avoids overflow when β*a is large positive
        def softplus(a):
            return ca.fmax(a, 0) + (1.0 / beta) * ca.log(1.0 + ca.exp(-beta * ca.fabs(a)))

        try:
            opti = ca.Opti()

            # ==================== DECISION VARIABLES ====================
            # States: [s, d, vs, vd] at each timestep
            S = opti.variable(4, H + 1)
            s = S[0, :]   # longitudinal position
            d = S[1, :]   # lateral position
            vs = S[2, :]  # longitudinal velocity
            vd = S[3, :]  # lateral velocity

            # Controls: [a_s, a_d] at each timestep
            U = opti.variable(2, H)
            a_s = U[0, :]  # longitudinal acceleration
            a_d = U[1, :]  # lateral acceleration

            # ==================== INITIAL STATE ====================
            opti.subject_to(s[0] == s0)
            opti.subject_to(d[0] == d0)
            opti.subject_to(vs[0] == vs0)
            opti.subject_to(vd[0] == vd0)

            # ==================== DYNAMICS (ZOH) ====================
            for k in range(H):
                opti.subject_to(s[k + 1] == s[k] + vs[k] * dt)
                opti.subject_to(d[k + 1] == d[k] + vd[k] * dt)
                opti.subject_to(vs[k + 1] == vs[k] + a_s[k] * dt)
                opti.subject_to(vd[k + 1] == vd[k] + a_d[k] * dt)

            # ==================== KINEMATIC FEASIBILITY ====================
            # vs >= rho * |vd| (ensures realistic turning)
            for k in range(H + 1):
                opti.subject_to(vs[k] >= rho * vd[k])
                opti.subject_to(vs[k] >= -rho * vd[k])

            # ==================== STATE BOUNDS ====================
            for k in range(H + 1):
                # Longitudinal velocity bounds
                opti.subject_to(vs[k] >= vs_min)
                opti.subject_to(vs[k] <= vs_max)
                # Lateral velocity bounds
                opti.subject_to(vd[k] >= vd_min)
                opti.subject_to(vd[k] <= vd_max)

            # ==================== CONTROL BOUNDS ====================
            for k in range(H):
                opti.subject_to(opti.bounded(a_s_min, a_s[k], a_s_max))
                opti.subject_to(opti.bounded(a_d_min, a_d[k], a_d_max))

            # ==================== JERK CONSTRAINTS ====================
            jerk_s_limit = jerk_s_max * dt
            jerk_d_limit = jerk_d_max * dt
            for k in range(H - 1):
                # Longitudinal jerk
                opti.subject_to(opti.bounded(-jerk_s_limit,
                                             a_s[k + 1] - a_s[k],
                                             jerk_s_limit))
                # Lateral jerk
                opti.subject_to(opti.bounded(-jerk_d_limit,
                                             a_d[k + 1] - a_d[k],
                                             jerk_d_limit))

            # ==================== ROAD BOUNDARIES ====================
            for k in range(H + 1):
                opti.subject_to(d[k] >= road_right[k])
                opti.subject_to(d[k] <= road_left[k])

            # ==================== COST FUNCTION ====================
            cost = 0.0

            # L2 tracking cost (smoother than L1 for NLP)
            for k in range(H + 1):
                ref_s = min(s0 + v_goal * k * dt, s_goal)
                ref_d = 0.0
                cost += w_s * (s[k] - ref_s) ** 2
                cost += w_d * (d[k] - ref_d) ** 2
                cost += w_v * (vs[k] - v_goal) ** 2

            # Control effort
            for k in range(H):
                cost += w_a_s * a_s[k]
                cost += w_a_d * a_d[k]

            # ==================== COLLISION AVOIDANCE ====================
            # Smooth penalty-based formulation that allows passing on ANY side
            #
            # Key idea: Add large penalty for being inside obstacle rectangle.
            # Being "outside" means: s < s_min OR s > s_max OR d < d_min OR d > d_max
            # Equivalently: max(s_min - s, s - s_max, d_min - d, d - d_max) >= 0
            #
            # We use smooth max (log-sum-exp) to find the maximum "escape distance"
            # and penalize when it's negative (inside the box).

            # Smooth max function using log-sum-exp (numerically stable)
            def smooth_max4(a, b, c, e):
                # Numerically stable log-sum-exp
                # smooth_max(a,b,c,e) ≈ max(a,b,c,e) as beta → ∞
                max_val = ca.fmax(ca.fmax(a, b), ca.fmax(c, e))
                return max_val + (1.0 / beta) * ca.log(
                    ca.exp(beta * (a - max_val)) +
                    ca.exp(beta * (b - max_val)) +
                    ca.exp(beta * (c - max_val)) +
                    ca.exp(beta * (e - max_val))
                )

            collision_penalty_weight = 1000.0  # Large weight for collision penalty
            safety_margin = 0.1  # Small positive margin for robustness

            for obs_idx in range(N_obs):
                obs = obstacles[obs_idx]

                # Obstacle dimensions in Frenet frame
                obs_half_L = obs['length'] / 2.0
                obs_half_W = obs['width'] / 2.0
                obs_s0 = float(obs['s'][0])
                _, _, _, road_angle = self._frenet._interpolate(obs_s0)
                dh = obs.get('heading', road_angle) - road_angle

                # Obstacle semi-axes (a = longitudinal, b = lateral) + collision margin
                obs_a = abs(obs_half_L * np.cos(dh)) + abs(obs_half_W * np.sin(dh)) + self._collision_margin
                obs_b = abs(obs_half_L * np.sin(dh)) + abs(obs_half_W * np.cos(dh)) + self._collision_margin

                for k in range(1, H + 1):  # Skip initial state
                    # Obstacle position at timestep k
                    s_obs = float(obs['s'][k] if k < len(obs['s']) else obs['s'][-1])
                    d_obs = float(obs['d'][k] if k < len(obs['d']) else obs['d'][-1])

                    # Enlarged rectangle bounds (Minkowski sum)
                    s_min = s_obs - obs_a - dx
                    s_max = s_obs + obs_a + dx
                    d_min = d_obs - obs_b - dy
                    d_max = d_obs + obs_b + dy

                    # Escape distances (positive = outside, negative = inside)
                    # escape_behind: positive when s < s_min (behind obstacle)
                    # escape_ahead: positive when s > s_max (ahead of obstacle)
                    # escape_right: positive when d < d_min (right of obstacle)
                    # escape_left: positive when d > d_max (left of obstacle)
                    escape_behind = s_min - s[k]
                    escape_ahead = s[k] - s_max
                    escape_right = d_min - d[k]
                    escape_left = d[k] - d_max

                    # Maximum escape distance (if > 0, we're outside in at least one direction)
                    max_escape = smooth_max4(escape_behind, escape_ahead, escape_right, escape_left)

                    # Soft constraint: penalize when max_escape < safety_margin
                    # violation = max(safety_margin - max_escape, 0)
                    violation = softplus(safety_margin - max_escape)
                    cost += collision_penalty_weight * violation ** 2

            # ==================== SET OBJECTIVE ====================
            opti.minimize(cost)

            # ==================== SOLVER OPTIONS ====================
            p_opts = {'expand': True, 'print_time': False}
            s_opts = {
                'max_iter': 500,
                'tol': 1e-4,
                'print_level': 0,
                'sb': 'yes',
            }
            opti.solver('ipopt', p_opts, s_opts)

            # ==================== INITIAL GUESS ====================
            # Simple straight-line trajectory at target speed
            for k in range(H + 1):
                opti.set_initial(s[k], s0 + v_goal * k * dt)
                opti.set_initial(d[k], d0)
                opti.set_initial(vs[k], v_goal)
                opti.set_initial(vd[k], 0)
            for k in range(H):
                opti.set_initial(a_s[k], 0)
                opti.set_initial(a_d[k], 0)

            # ==================== SOLVE ====================
            sol = opti.solve()

            # ==================== EXTRACT SOLUTION ====================
            s_val = sol.value(s).flatten()
            d_val = sol.value(d).flatten()
            vs_val = sol.value(vs).flatten()
            vd_val = sol.value(vd).flatten()

            milp_states = np.column_stack([s_val, d_val, vs_val, vd_val])
            return milp_states

        except Exception as e:
            logger.debug(f"Stage1 solver failed: {e}")

            # Diagnose constraint violations
            print(f"  MILP FAILURE DIAGNOSIS:")
            try:
                s_dbg = opti.debug.value(s).flatten()
                d_dbg = opti.debug.value(d).flatten()
                vs_dbg = opti.debug.value(vs).flatten()
                vd_dbg = opti.debug.value(vd).flatten()
                a_s_dbg = opti.debug.value(a_s).flatten()
                a_d_dbg = opti.debug.value(a_d).flatten()

                violations = []

                # Check initial state constraints
                if abs(s_dbg[0] - s0) > 1e-3:
                    violations.append(f"Initial s: {s_dbg[0]:.3f} != {s0:.3f}")
                if abs(d_dbg[0] - d0) > 1e-3:
                    violations.append(f"Initial d: {d_dbg[0]:.3f} != {d0:.3f}")
                if abs(vs_dbg[0] - vs0) > 1e-3:
                    violations.append(f"Initial vs: {vs_dbg[0]:.3f} != {vs0:.3f}")
                if abs(vd_dbg[0] - vd0) > 1e-3:
                    violations.append(f"Initial vd: {vd_dbg[0]:.3f} != {vd0:.3f}")

                # Check velocity bounds
                for k in range(H + 1):
                    if vs_dbg[k] < vs_min - 1e-3:
                        violations.append(f"vs[{k}]={vs_dbg[k]:.3f} < vs_min={vs_min}")
                    if vs_dbg[k] > vs_max + 1e-3:
                        violations.append(f"vs[{k}]={vs_dbg[k]:.3f} > vs_max={vs_max}")
                    if vd_dbg[k] < vd_min - 1e-3:
                        violations.append(f"vd[{k}]={vd_dbg[k]:.3f} < vd_min={vd_min}")
                    if vd_dbg[k] > vd_max + 1e-3:
                        violations.append(f"vd[{k}]={vd_dbg[k]:.3f} > vd_max={vd_max}")

                # Check kinematic feasibility: vs >= rho * |vd|
                for k in range(H + 1):
                    if vs_dbg[k] < rho * abs(vd_dbg[k]) - 1e-3:
                        violations.append(f"Kinematic[{k}]: vs={vs_dbg[k]:.3f} < rho*|vd|={rho * abs(vd_dbg[k]):.3f}")

                # Check road boundaries
                for k in range(H + 1):
                    if d_dbg[k] < road_right[k] - 1e-3:
                        violations.append(f"Road right[{k}]: d={d_dbg[k]:.3f} < {road_right[k]:.3f}")
                    if d_dbg[k] > road_left[k] + 1e-3:
                        violations.append(f"Road left[{k}]: d={d_dbg[k]:.3f} > {road_left[k]:.3f}")

                # Check control bounds
                for k in range(H):
                    if a_s_dbg[k] < a_s_min - 1e-3:
                        violations.append(f"a_s[{k}]={a_s_dbg[k]:.3f} < a_s_min={a_s_min}")
                    if a_s_dbg[k] > a_s_max + 1e-3:
                        violations.append(f"a_s[{k}]={a_s_dbg[k]:.3f} > a_s_max={a_s_max}")
                    if a_d_dbg[k] < a_d_min - 1e-3:
                        violations.append(f"a_d[{k}]={a_d_dbg[k]:.3f} < a_d_min={a_d_min}")
                    if a_d_dbg[k] > a_d_max + 1e-3:
                        violations.append(f"a_d[{k}]={a_d_dbg[k]:.3f} > a_d_max={a_d_max}")

                # Check jerk constraints
                for k in range(H - 1):
                    jerk_s = a_s_dbg[k + 1] - a_s_dbg[k]
                    jerk_d = a_d_dbg[k + 1] - a_d_dbg[k]
                    if abs(jerk_s) > jerk_s_limit + 1e-3:
                        violations.append(f"Jerk_s[{k}]={jerk_s:.3f} > limit={jerk_s_limit:.3f}")
                    if abs(jerk_d) > jerk_d_limit + 1e-3:
                        violations.append(f"Jerk_d[{k}]={jerk_d:.3f} > limit={jerk_d_limit:.3f}")

                if violations:
                    print(f"    Violated constraints ({len(violations)} total):")
                    for v in violations[:10]:  # Show first 10
                        print(f"      - {v}")
                    if len(violations) > 10:
                        print(f"      ... and {len(violations) - 10} more")
                else:
                    print(f"    No obvious constraint violations found in debug values")
                    print(f"    Initial state: s0={s0:.2f}, d0={d0:.2f}, vs0={vs0:.2f}, vd0={vd0:.2f}")
                    print(f"    Road bounds[0]: left={road_left[0]:.2f}, right={road_right[0]:.2f}")

            except Exception as debug_e:
                print(f"    (Could not extract debug values: {debug_e})")

            return None


    # ------------------------------------------------------------------
    # MILP -> NLP warm-start conversion
    # ------------------------------------------------------------------

    def _milp_to_nlp_warmstart(self, milp_states, frenet_state):
        """Convert MILP [s, d, vs, vd] to NLP [s, d, phi, v] + controls.

        Args:
            milp_states: (H+1, 4) Frenet MILP states.
            frenet_state: [s, d, phi, v] initial Frenet state.

        Returns:
            (nlp_states, nlp_controls) — (H+1, 4) and (H, 2) arrays.
        """
        H = self._horizon
        dt = self._dt
        L = self._wheelbase
        nlp = self._nlp

        s_arr = milp_states[:, 0]
        d_arr = milp_states[:, 1]
        vs_arr = milp_states[:, 2]
        vd_arr = milp_states[:, 3]

        # Speed and heading from velocity components
        speeds = np.sqrt(vs_arr**2 + vd_arr**2)
        phis = np.arctan2(vd_arr, vs_arr)
        phis[0] = frenet_state[2]  # use actual initial heading

        nlp_states = np.column_stack([s_arr, d_arr, phis, speeds])
        nlp_controls = np.zeros((H, 2))

        for k in range(H):
            # Acceleration
            a_k = np.clip((speeds[k + 1] - speeds[k]) / dt,
                          nlp['a_min'], nlp['a_max'])

            # Steering from heading change
            d_phi = (phis[k + 1] - phis[k] + np.pi) % (2 * np.pi) - np.pi
            spd = max(speeds[k], 0.5)
            sin_arg = np.clip(d_phi * L / (2.0 * spd * dt), -1.0, 1.0)
            delta_k = np.clip(np.arcsin(sin_arg),
                              -nlp['delta_max'], nlp['delta_max'])

            nlp_controls[k] = [a_k, delta_k]

        return nlp_states, nlp_controls

    # ------------------------------------------------------------------
    # Stage 2: NLP (CasADi + IPOPT)
    # ------------------------------------------------------------------

    def _solve_nlp(self, frenet_state, warm_states, warm_controls,
                   road_left, road_right, obstacles):
        """Solve the NLP using CasADi + IPOPT.

        Bicycle model in Frenet frame:
            s_{k+1} = s_k + v_k * cos(phi_k + delta_k) * dt
            d_{k+1} = d_k + v_k * sin(phi_k + delta_k) * dt
            phi_{k+1} = phi_k + (2*v_k/L) * sin(delta_k) * dt
            v_{k+1} = v_k + a_k * dt

        Args:
            frenet_state: [s, d, phi, v] initial state.
            warm_states: (H+1, 4) initial guess for states.
            warm_controls: (H, 2) initial guess for controls [a, delta].
            road_left: (H+1,) left boundary.
            road_right: (H+1,) right boundary.
            obstacles: List of obstacle dicts.

        Returns:
            (nlp_states, nlp_controls, success) — (H+1, 4), (H, 2), bool.
        """
        H = self._horizon
        dt = self._dt
        L = self._wheelbase

        # NLP-specific parameters
        nlp = self._nlp
        a_min, a_max = nlp['a_min'], nlp['a_max']
        delta_max = nlp['delta_max']
        delta_rate_max = nlp['delta_rate_max']
        jerk_max = nlp['jerk_max']
        v_min, v_max = nlp['v_min'], nlp['v_max']
        w_s, w_d, w_v = nlp['w_s'], nlp['w_d'], nlp['w_v']
        w_a, w_delta = nlp['w_a'], nlp['w_delta']

        try:
            opti = ca.Opti()

            # Decision variables
            S = opti.variable(4, H + 1)   # [s; d; phi; v] at each step
            U = opti.variable(2, H)        # [a; delta] at each step

            # Parameters for obstacle positions (pre-allocated slots)
            N_obs = min(len(obstacles), self.N_OBS_MAX)

            # Goal: point ahead if driving at target speed for horizon
            s0 = frenet_state[0]
            v_goal = self._target_speed
            s_max = self._frenet.total_length

            # --- Cost function ---
            cost = 0.0
            for k in range(H + 1):
                # Reference s: where vehicle would be at timestep k driving at target speed
                ref_s = min(s0 + v_goal * k * dt, s_max)
                cost += w_s * (S[0, k] - ref_s)**2
                cost += w_d * S[1, k]**2
                cost += w_v * (S[3, k] - v_goal)**2
            for k in range(H):
                cost += w_a * U[0, k]**2
                cost += w_delta * U[1, k]**2
            opti.minimize(cost)

            # --- Initial state constraint ---
            opti.subject_to(S[0, 0] == frenet_state[0])
            opti.subject_to(S[1, 0] == frenet_state[1])
            opti.subject_to(S[2, 0] == frenet_state[2])
            opti.subject_to(S[3, 0] == frenet_state[3])

            # --- Dynamics constraints ---
            for k in range(H):
                opti.subject_to(
                    S[0, k + 1] == S[0, k] + S[3, k] * ca.cos(S[2, k] + U[1, k]) * dt)
                opti.subject_to(
                    S[1, k + 1] == S[1, k] + S[3, k] * ca.sin(S[2, k] + U[1, k]) * dt)
                opti.subject_to(
                    S[2, k + 1] == S[2, k] + (2.0 * S[3, k] / L) * ca.sin(U[1, k]) * dt)
                opti.subject_to(
                    S[3, k + 1] == S[3, k] + U[0, k] * dt)

            # --- Road boundary constraints (corner-based) ---
            # Check all four corners of the rotated ego vehicle stay within road
            half_L = self._ego_length / 2.0
            half_W = self._ego_width / 2.0

            for k in range(H + 1):
                cos_phi = ca.cos(S[2, k])
                sin_phi = ca.sin(S[2, k])

                # Check all four corners: (sl, sw) in {(1,1), (1,-1), (-1,1), (-1,-1)}
                for sl, sw in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    # Corner lateral position in Frenet frame
                    c_d = S[1, k] + sl * half_L * sin_phi + sw * half_W * cos_phi

                    opti.subject_to(c_d >= road_right[k])
                    opti.subject_to(c_d <= road_left[k])

            # --- Control bounds ---
            for k in range(H):
                opti.subject_to(opti.bounded(a_min, U[0, k], a_max))
                opti.subject_to(opti.bounded(-delta_max, U[1, k], delta_max))

            # --- Speed bounds ---
            for k in range(H + 1):
                opti.subject_to(opti.bounded(v_min, S[3, k], v_max))

            # --- Jerk constraints ---
            jerk_limit = jerk_max * dt
            for k in range(H - 1):
                opti.subject_to(opti.bounded(-jerk_limit,
                                             U[0, k + 1] - U[0, k],
                                             jerk_limit))

            # --- Steering rate constraints ---
            delta_rate_limit = delta_rate_max * dt
            for k in range(H - 1):
                opti.subject_to(opti.bounded(-delta_rate_limit,
                                             U[1, k + 1] - U[1, k],
                                             delta_rate_limit))

            # --- Elliptical collision avoidance (corner-based) ---
            # Based on Eiras et al. Section III-3, Equations 9-10
            #
            # OBSTACLE REPRESENTATION (Eq. 9):
            #   Each obstacle i at timestep k is an ellipse with:
            #   - Center: (s^i_k, d^i_k) in Frenet coordinates
            #   - Orientation: φ^i_k (heading relative to road)
            #   - Semi-axes: (a^i_k, b^i_k) where a is along length, b along width
            #
            # EGO VEHICLE CORNERS (Eq. 6):
            #   c^α(z_k) = R(φ_k) × [α_l * l/2, α_w * w/2]^T + [s_k, d_k]^T
            #   where α = (α_l, α_w) ∈ {(±1, ±1)} for the 4 corners
            #   R(φ) = [[cos(φ), -sin(φ)], [sin(φ), cos(φ)]]
            #
            # COLLISION CONSTRAINT (Eq. 10): g^{i,α}(z_k) >= 1
            #   g = d^T × R(φ^i)^T × S × R(φ^i) × d
            #   where d = corner - obstacle_center, S = diag(1/a^2, 1/b^2)
            #
            # Equivalently in obstacle body frame:
            #   d_body = R(-φ^i) × d
            #   g = (d_body_x / a)^2 + (d_body_y / b)^2 >= 1
            #
            # Interpretation: g > 1 means corner is OUTSIDE ellipse (safe)

            half_L = self._ego_length / 2.0  # l/2
            half_W = self._ego_width / 2.0   # w/2

            for obs_idx in range(N_obs):
                obs = obstacles[obs_idx]

                # Ellipse semi-axes (Eq. 9): a^i_k, b^i_k
                # a = along obstacle length, b = along obstacle width
                # collision_margin acts as safety buffer (like uncertainty in paper)
                a_i = obs['length'] / 2.0 + self._collision_margin
                b_i = obs['width'] / 2.0 + self._collision_margin

                # Obstacle heading φ^i relative to Frenet s-axis
                obs_s0 = float(obs['s'][0])
                _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
                phi_i = obs.get('heading', road_angle_obs) - road_angle_obs

                # Precompute rotation to obstacle body frame: R(-φ^i)
                cos_neg_phi_i = np.cos(-phi_i)
                sin_neg_phi_i = np.sin(-phi_i)

                for k in range(1, H + 1):
                    # Obstacle center (s^i_k, d^i_k) at timestep k
                    s_obs_k = float(obs['s'][k]) if k < len(obs['s']) else float(obs['s'][-1])
                    d_obs_k = float(obs['d'][k]) if k < len(obs['d']) else float(obs['d'][-1])

                    # Ego heading φ_k at timestep k
                    cos_phi_k = ca.cos(S[2, k])
                    sin_phi_k = ca.sin(S[2, k])

                    # Check all 4 corners: α = (α_l, α_w) ∈ {(±1, ±1)}
                    for alpha_l, alpha_w in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        # Corner position c^α (Eq. 6):
                        # c = R(φ_k) × [α_l * l/2, α_w * w/2]^T + [s_k, d_k]^T
                        c_s = S[0, k] + alpha_l * half_L * cos_phi_k - alpha_w * half_W * sin_phi_k
                        c_d = S[1, k] + alpha_l * half_L * sin_phi_k + alpha_w * half_W * cos_phi_k

                        # Vector from obstacle center to corner: d = c - (s^i_k, d^i_k)
                        d_s = c_s - s_obs_k
                        d_d = c_d - d_obs_k

                        # Rotate to obstacle body frame: d_body = R(-φ^i) × d
                        d_body_x = cos_neg_phi_i * d_s + sin_neg_phi_i * d_d
                        d_body_y = -sin_neg_phi_i * d_s + cos_neg_phi_i * d_d

                        # Ellipse containment function g (Eq. 10):
                        # g = (d_body_x / a)^2 + (d_body_y / b)^2
                        # Constraint: g >= 1 (corner must be outside ellipse)
                        g = (d_body_x / a_i)**2 + (d_body_y / b_i)**2
                        opti.subject_to(g >= 1.0)

            # --- Initial guess ---
            opti.set_initial(S, warm_states.T)
            opti.set_initial(U, warm_controls.T)

            # --- Solver options ---
            p_opts = {'expand': True, 'print_time': False}
            s_opts = {
                'max_iter': 500,
                'warm_start_init_point': 'yes',
                'tol': 1e-4,
                'print_level': 0,
                'sb': 'yes',
            }
            opti.solver('ipopt', p_opts, s_opts)

            # --- Solve ---
            sol = opti.solve()

            nlp_states = sol.value(S).T   # (H+1, 4)
            nlp_controls = sol.value(U).T  # (H, 2)

            # Debug plotting if enabled
            if self._debug_plot_nlp and N_obs > 0:
                self._plot_nlp_debug(nlp_states, obstacles, road_left, road_right,
                                     frenet_state, title_suffix=f" (step {self._step_count})")

            return nlp_states, nlp_controls, True

        except Exception as e:
            logger.debug("NLP solver failed: %s", e)

            # Diagnose constraint violations
            print(f"  NLP FAILURE DIAGNOSIS:")
            try:
                S_dbg = opti.debug.value(S).T  # (H+1, 4)
                U_dbg = opti.debug.value(U).T  # (H, 2)

                s_dbg = S_dbg[:, 0]
                d_dbg = S_dbg[:, 1]
                phi_dbg = S_dbg[:, 2]
                v_dbg = S_dbg[:, 3]
                a_dbg = U_dbg[:, 0]
                delta_dbg = U_dbg[:, 1]

                violations = []

                # Check initial state constraints
                if abs(s_dbg[0] - frenet_state[0]) > 1e-3:
                    violations.append(f"Initial s: {s_dbg[0]:.3f} != {frenet_state[0]:.3f}")
                if abs(d_dbg[0] - frenet_state[1]) > 1e-3:
                    violations.append(f"Initial d: {d_dbg[0]:.3f} != {frenet_state[1]:.3f}")
                if abs(phi_dbg[0] - frenet_state[2]) > 1e-3:
                    violations.append(f"Initial phi: {phi_dbg[0]:.3f} != {frenet_state[2]:.3f}")
                if abs(v_dbg[0] - frenet_state[3]) > 1e-3:
                    violations.append(f"Initial v: {v_dbg[0]:.3f} != {frenet_state[3]:.3f}")

                # Check speed bounds
                for k in range(H + 1):
                    if v_dbg[k] < v_min - 1e-3:
                        violations.append(f"v[{k}]={v_dbg[k]:.3f} < v_min={v_min}")
                    if v_dbg[k] > v_max + 1e-3:
                        violations.append(f"v[{k}]={v_dbg[k]:.3f} > v_max={v_max}")

                # Check control bounds
                for k in range(H):
                    if a_dbg[k] < a_min - 1e-3:
                        violations.append(f"a[{k}]={a_dbg[k]:.3f} < a_min={a_min}")
                    if a_dbg[k] > a_max + 1e-3:
                        violations.append(f"a[{k}]={a_dbg[k]:.3f} > a_max={a_max}")
                    if abs(delta_dbg[k]) > delta_max + 1e-3:
                        violations.append(f"|delta[{k}]|={abs(delta_dbg[k]):.3f} > delta_max={delta_max}")

                # Check jerk constraints
                jerk_limit = jerk_max * dt
                for k in range(H - 1):
                    jerk = abs(a_dbg[k + 1] - a_dbg[k])
                    if jerk > jerk_limit + 1e-3:
                        violations.append(f"Jerk[{k}]={jerk:.3f} > limit={jerk_limit:.3f}")

                # Check steering rate constraints
                delta_rate_limit = delta_rate_max * dt
                for k in range(H - 1):
                    delta_rate = abs(delta_dbg[k + 1] - delta_dbg[k])
                    if delta_rate > delta_rate_limit + 1e-3:
                        violations.append(f"DeltaRate[{k}]={delta_rate:.3f} > limit={delta_rate_limit:.3f}")

                # Check road boundary constraints (corner-based)
                half_L = self._ego_length / 2.0
                half_W = self._ego_width / 2.0
                for k in range(H + 1):
                    cos_phi = np.cos(phi_dbg[k])
                    sin_phi = np.sin(phi_dbg[k])
                    for sl, sw in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        c_d = d_dbg[k] + sl * half_L * sin_phi + sw * half_W * cos_phi
                        if c_d < road_right[k] - 1e-3:
                            violations.append(f"Road right[{k}] corner({sl},{sw}): c_d={c_d:.3f} < {road_right[k]:.3f}")
                        if c_d > road_left[k] + 1e-3:
                            violations.append(f"Road left[{k}] corner({sl},{sw}): c_d={c_d:.3f} > {road_left[k]:.3f}")

                # Check collision constraints (ellipse)
                for obs_idx in range(N_obs):
                    obs = obstacles[obs_idx]
                    obs_half_L = obs['length'] / 2.0
                    obs_half_W = obs['width'] / 2.0
                    rx = obs_half_L + self._collision_margin
                    ry = obs_half_W + self._collision_margin

                    obs_s0 = float(obs['s'][0])
                    _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
                    obs_heading_frenet = obs.get('heading', road_angle_obs) - road_angle_obs
                    cos_theta = np.cos(-obs_heading_frenet)
                    sin_theta = np.sin(-obs_heading_frenet)

                    for k in range(1, H + 1):
                        s_obs = float(obs['s'][k] if k < len(obs['s']) else obs['s'][-1])
                        d_obs = float(obs['d'][k] if k < len(obs['d']) else obs['d'][-1])

                        cos_phi = np.cos(phi_dbg[k])
                        sin_phi = np.sin(phi_dbg[k])

                        for sl, sw in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            c_s = s_dbg[k] + sl * half_L * cos_phi - sw * half_W * sin_phi
                            c_d = d_dbg[k] + sl * half_L * sin_phi + sw * half_W * cos_phi
                            ds = c_s - s_obs
                            dd = c_d - d_obs
                            x_body = cos_theta * ds + sin_theta * dd
                            y_body = -sin_theta * ds + cos_theta * dd
                            ellipse_val = (x_body / rx)**2 + (y_body / ry)**2
                            if ellipse_val < 1.0 - 1e-3:
                                violations.append(f"Collision obs{obs_idx}[{k}] corner({sl},{sw}): ellipse={ellipse_val:.3f} < 1.0")

                if violations:
                    print(f"    Violated constraints ({len(violations)} total):")
                    for v in violations[:10]:  # Show first 10
                        print(f"      - {v}")
                    if len(violations) > 10:
                        print(f"      ... and {len(violations) - 10} more")
                else:
                    print(f"    No obvious constraint violations found in debug values")
                    print(f"    Initial: s={frenet_state[0]:.2f}, d={frenet_state[1]:.2f}, phi={frenet_state[2]:.2f}, v={frenet_state[3]:.2f}")
                    print(f"    Ego dimensions: L={2*half_L:.2f}, W={2*half_W:.2f}")
                    print(f"    Road bounds[0]: left={road_left[0]:.2f}, right={road_right[0]:.2f}, width={road_left[0]-road_right[0]:.2f}")

                    # Show obstacle info and gap analysis
                    if N_obs > 0:
                        print(f"    Obstacles ({N_obs}):")
                        for obs_idx in range(min(N_obs, 3)):
                            obs = obstacles[obs_idx]
                            obs_d0 = float(obs['d'][0])
                            obs_half_W = obs['width'] / 2.0
                            # Obstacle occupies [obs_d0 - obs_half_W, obs_d0 + obs_half_W]
                            obs_left = obs_d0 + obs_half_W + self._collision_margin
                            obs_right = obs_d0 - obs_half_W - self._collision_margin
                            gap_left = road_left[0] - obs_left  # Gap between obstacle and left road edge
                            gap_right = obs_right - road_right[0]  # Gap between obstacle and right road edge
                            print(f"      Obs{obs_idx}: d={obs_d0:.2f}, W={obs['width']:.2f}, gaps: left={gap_left:.2f}, right={gap_right:.2f}")
                            # Check if ego can fit in either gap
                            ego_needed = 2 * half_W + 0.1  # Width needed + small margin
                            if gap_left < ego_needed and gap_right < ego_needed:
                                print(f"      WARNING: Ego needs {ego_needed:.2f}m but gaps are too small!")

            except Exception as debug_e:
                print(f"    (Could not extract debug values: {debug_e})")

            return warm_states, warm_controls, False

    def _plot_nlp_debug(self, nlp_states, obstacles, road_left, road_right,
                        frenet_state, title_suffix=""):
        """Plot NLP solution for debugging: trajectory, ego corners, obstacle ellipses.

        All values are plotted in raw Frenet coordinates (s, d) without any transforms.
        This helps diagnose coordinate frame issues. Updates in place without creating new windows.

        Args:
            nlp_states: (H+1, 4) array with columns [s, d, phi, v].
            obstacles: List of obstacle dicts with 's', 'd', 'length', 'width', 'heading'.
            road_left: (H+1,) left road boundary in d.
            road_right: (H+1,) right road boundary in d.
            frenet_state: [s, d, phi, v] initial state.
            title_suffix: Optional suffix for plot title.
        """
        H = len(nlp_states) - 1
        half_L = self._ego_length / 2.0
        half_W = self._ego_width / 2.0

        # Extract trajectory
        s_traj = nlp_states[:, 0]
        d_traj = nlp_states[:, 1]
        phi_traj = nlp_states[:, 2]

        # Create figure on first call, reuse afterwards
        if self._debug_fig is None or not plt.fignum_exists(self._debug_fig.number):
            plt.ion()  # Interactive mode
            self._debug_fig, self._debug_axes = plt.subplots(1, 2, figsize=(16, 8))
            self._debug_fig.show()

        fig = self._debug_fig
        ax1, ax2 = self._debug_axes

        # Clear axes for redraw
        ax1.clear()
        ax2.clear()

        # --- Left plot: Full trajectory view in Frenet (s, d) ---
        ax1.set_title(f"NLP Solution in Frenet Frame (s, d){title_suffix}")
        ax1.set_xlabel("s (longitudinal)")
        ax1.set_ylabel("d (lateral)")

        # Plot road boundaries
        ax1.fill_between(s_traj, road_right, road_left, alpha=0.2, color='gray', label='Road')
        ax1.plot(s_traj, road_left, 'k--', linewidth=1, label='Road left')
        ax1.plot(s_traj, road_right, 'k--', linewidth=1, label='Road right')

        # Plot trajectory centerline
        ax1.plot(s_traj, d_traj, 'b-', linewidth=2, label='Ego trajectory', zorder=5)
        ax1.scatter(s_traj[0], d_traj[0], c='green', s=100, marker='o', zorder=10, label='Start')
        ax1.scatter(s_traj[-1], d_traj[-1], c='red', s=100, marker='x', zorder=10, label='End')

        # Plot ego vehicle corners at each timestep
        for k in range(0, H + 1, max(1, H // 10)):  # Sample every ~10% of horizon
            cos_phi = np.cos(phi_traj[k])
            sin_phi = np.sin(phi_traj[k])

            # Compute 4 corners in Frenet frame (Eq. 6)
            corners = []
            for alpha_l, alpha_w in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:  # CCW order
                c_s = s_traj[k] + alpha_l * half_L * cos_phi - alpha_w * half_W * sin_phi
                c_d = d_traj[k] + alpha_l * half_L * sin_phi + alpha_w * half_W * cos_phi
                corners.append([c_s, c_d])

            poly = Polygon(corners, closed=True, fill=False, edgecolor='blue',
                           linewidth=1, alpha=0.5 + 0.5 * k / H)
            ax1.add_patch(poly)

        # Plot obstacle ellipses
        for obs_idx, obs in enumerate(obstacles):
            # Ellipse semi-axes
            a_i = obs['length'] / 2.0 + self._collision_margin
            b_i = obs['width'] / 2.0 + self._collision_margin

            # Obstacle heading in Frenet frame
            obs_s0 = float(obs['s'][0])
            _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
            phi_i = obs.get('heading', road_angle_obs) - road_angle_obs
            phi_i_deg = np.degrees(phi_i)

            # Plot ellipse at each timestep (sample a few)
            for k in range(0, min(H + 1, len(obs['s'])), max(1, H // 5)):
                s_obs_k = float(obs['s'][k]) if k < len(obs['s']) else float(obs['s'][-1])
                d_obs_k = float(obs['d'][k]) if k < len(obs['d']) else float(obs['d'][-1])

                # Ellipse center and dimensions (raw values, no transform)
                ellipse = Ellipse((s_obs_k, d_obs_k), width=2*a_i, height=2*b_i,
                                  angle=phi_i_deg, fill=False, edgecolor='red',
                                  linewidth=2, alpha=0.3 + 0.7 * k / H)
                ax1.add_patch(ellipse)

                # Mark center
                ax1.scatter(s_obs_k, d_obs_k, c='red', s=30, marker='+', alpha=0.5)

            # Label
            ax1.annotate(f'Obs{obs_idx}', (float(obs['s'][0]), float(obs['d'][0])),
                         fontsize=8, color='red')

        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)

        # --- Right plot: Zoomed view showing corner-ellipse distances ---
        ax2.set_title(f"Corner-Ellipse Collision Check (g values){title_suffix}")
        ax2.set_xlabel("Timestep k")
        ax2.set_ylabel("g value (>= 1 is safe)")

        # Compute g values for each corner and each obstacle
        for obs_idx, obs in enumerate(obstacles):
            a_i = obs['length'] / 2.0 + self._collision_margin
            b_i = obs['width'] / 2.0 + self._collision_margin

            obs_s0 = float(obs['s'][0])
            _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
            phi_i = obs.get('heading', road_angle_obs) - road_angle_obs
            cos_neg_phi_i = np.cos(-phi_i)
            sin_neg_phi_i = np.sin(-phi_i)

            corner_labels = [(1, 1, 'FR'), (1, -1, 'FL'), (-1, 1, 'RR'), (-1, -1, 'RL')]
            for alpha_l, alpha_w, corner_name in corner_labels:
                g_values = []
                for k in range(1, H + 1):
                    s_obs_k = float(obs['s'][k]) if k < len(obs['s']) else float(obs['s'][-1])
                    d_obs_k = float(obs['d'][k]) if k < len(obs['d']) else float(obs['d'][-1])

                    cos_phi_k = np.cos(phi_traj[k])
                    sin_phi_k = np.sin(phi_traj[k])

                    # Corner position (Eq. 6)
                    c_s = s_traj[k] + alpha_l * half_L * cos_phi_k - alpha_w * half_W * sin_phi_k
                    c_d = d_traj[k] + alpha_l * half_L * sin_phi_k + alpha_w * half_W * cos_phi_k

                    # Vector from obstacle center to corner
                    d_s = c_s - s_obs_k
                    d_d = c_d - d_obs_k

                    # Rotate to obstacle body frame
                    d_body_x = cos_neg_phi_i * d_s + sin_neg_phi_i * d_d
                    d_body_y = -sin_neg_phi_i * d_s + cos_neg_phi_i * d_d

                    # Ellipse containment function g (Eq. 10)
                    g = (d_body_x / a_i)**2 + (d_body_y / b_i)**2
                    g_values.append(g)

                ax2.plot(range(1, H + 1), g_values,
                         label=f'Obs{obs_idx} {corner_name}', alpha=0.7)

        # Safe threshold line
        ax2.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Safe threshold (g=1)')
        ax2.legend(loc='upper right', fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)

        # Update the figure in place
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)  # Small pause to allow GUI update
