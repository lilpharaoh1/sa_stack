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
from typing import List, Optional, Dict, Tuple

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csc_matrix
from shapely.geometry import LineString
import casadi as ca

from igp2.core.agentstate import AgentState, AgentMetadata
from igp2.core.vehicle import Action
from igp2.opendrive.map import Map
from igp2.planlibrary.controller import PIDController
import time

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
    A Mixed-Integer Linear Program over a point-mass model
    ``[s, d, vs, vd]`` in Frenet coordinates minimises L1 tracking error
    subject to ZOH dynamics, kinematic feasibility, road boundaries,
    and Big-M rectangle collision avoidance.

    **Stage 2 — NLP (refined trajectory).**
    A CasADi + IPOPT nonlinear program over the bicycle kinematic model
    ``[s, d, phi, v]`` in Frenet coordinates, with elliptical collision
    regions, road boundaries, jerk constraints, and the paper's cost
    function.

    The planning timestep ``dt`` (default 0.1 s) is independent of the
    simulation framerate.  Each MPC call plans at ``dt`` resolution over
    ``horizon`` steps (default 40 = 4 s), but only the first action is
    applied for one simulation step (receding-horizon MPC).

    Args:
        fps: Simulation framerate.
        metadata: Agent physical metadata (wheelbase, limits, etc.).
        reference_waypoints: Concatenated A* reference path (N, 2).
        scenario_map: Road layout for boundary queries. May be None.
        horizon: Number of planning steps (default 40).
        dt: Planning timestep in seconds (default 0.1).
        target_speed: Desired cruising speed (m/s).
        milp_rho: Kinematic feasibility ratio for MILP.
        big_m: Big-M constant for collision avoidance.
        collision_margin: Extra safety margin around obstacles (m).

    MILP constraints (point-mass Frenet model).
    x = longitudinal (along path, Frenet s), y = lateral (Frenet d):

        milp_a_x_min: Min longitudinal acceleration.
        milp_a_x_max: Max longitudinal acceleration.
        milp_a_y_min: Min lateral acceleration.
        milp_a_y_max: Max lateral acceleration.
        milp_jerk_x_max: Longitudinal jerk limit (None = no constraint).
        milp_jerk_y_max: Lateral jerk limit (None = no constraint).
        milp_v_min: Min longitudinal velocity.
        milp_v_max: Max longitudinal velocity.

    MILP cost weights (L1 objective):

        milp_w_x: Weight on longitudinal position tracking.
        milp_w_v: Weight on speed tracking.
        milp_w_y: Weight on lateral deviation.
        milp_w_a: Weight on acceleration.

    NLP constraints (bicycle model):

        nlp_a_min: Min acceleration.
        nlp_a_max: Max acceleration.
        nlp_delta_max: Max steering angle.
        nlp_steer_rate_max: Max steering rate (None = no constraint).
        nlp_jerk_max: Max jerk.
        nlp_v_min: Min speed.
        nlp_v_max: Max speed.

    NLP cost weights (quadratic objective):

        nlp_w_x: Weight on longitudinal tracking error.
        nlp_w_v: Weight on speed tracking error.
        nlp_w_y: Weight on lateral deviation.
        nlp_w_a: Weight on acceleration.
        nlp_w_delta: Weight on steering.
    """

    DEFAULT_HORIZON = 40
    DEFAULT_DT = 0.1
    DEFAULT_RHO = 1.5
    DEFAULT_BIG_M = 1000.0
    N_OBS_MAX = 10  # pre-allocated obstacle slots in NLP

    def __init__(self,
                 fps: int,
                 metadata: AgentMetadata,
                 reference_waypoints: np.ndarray,
                 scenario_map: Optional[Map] = None,
                 horizon: int = None,
                 dt: float = None,
                 target_speed: float = 5.0,
                 milp_rho: float = 1.5,
                 big_m: float = None,
                 collision_margin: float = 0.9,
                 # MILP constraints
                 milp_a_x_min: float = -3.0,
                 milp_a_x_max: float = 3.0,
                 milp_a_y_min: float = -0.5,
                 milp_a_y_max: float = 0.5,
                 milp_jerk_x_max: float = 0.5,
                 milp_jerk_y_max: float = 0.1,
                 milp_v_min: float = 0.0,
                 milp_v_max: float = 3.0,
                 # MILP cost weights (L1 objective)
                 milp_w_x: float = 0.9,
                 milp_w_v: float = 0.5,
                 milp_w_y: float = 0.05,
                 milp_w_a: float = 0.4,
                 # NLP constraints
                 nlp_a_min: float = -3.0,
                 nlp_a_max: float = 3.0,
                 nlp_delta_max: float = 0.45,
                 nlp_steer_rate_max: float = 0.18,
                 nlp_jerk_max: float = 0.5,
                 nlp_v_min: float = 0.0,
                 nlp_v_max: float = 10,
                 # NLP cost weights
                 nlp_w_x: float = 0.1,
                 nlp_w_v: float = 2.5,
                 nlp_w_y: float = 0.05,
                 nlp_w_a: float = 1.0,
                 nlp_w_delta: float = 2.0,
                 # Legacy params (ignored for back-compat)
                 max_steer=None, w_ref=None, w_speed=None, w_smooth=None,
                 w_x=None, w_v=None, w_y=None, w_a=None, w_delta=None,
                 a_min=None, a_max=None, delta_max=None, jerk_max=None,
                 v_max=None,
                 **kwargs):
        self._fps = fps
        self._dt_sim = 1.0 / fps
        self._dt = dt if dt is not None else self.DEFAULT_DT
        self._metadata = metadata
        self._reference_waypoints = np.asarray(reference_waypoints, dtype=float)
        self._scenario_map = scenario_map
        self._horizon = horizon if horizon is not None else self.DEFAULT_HORIZON
        self._target_speed = target_speed

        # MILP constraints
        self._milp_a_x_min = milp_a_x_min
        self._milp_a_x_max = milp_a_x_max
        self._milp_a_y_min = milp_a_y_min
        self._milp_a_y_max = milp_a_y_max
        self._milp_jerk_x_max = milp_jerk_x_max
        self._milp_jerk_y_max = milp_jerk_y_max
        self._milp_v_min = milp_v_min
        self._milp_v_max = milp_v_max

        # MILP cost weights (L1)
        self._milp_w_x = milp_w_x
        self._milp_w_v = milp_w_v
        self._milp_w_y = milp_w_y
        self._milp_w_a = milp_w_a

        # NLP constraints
        self._nlp_a_min = nlp_a_min
        self._nlp_a_max = nlp_a_max
        self._nlp_delta_max = nlp_delta_max
        self._nlp_steer_rate_max = nlp_steer_rate_max
        self._nlp_jerk_max = nlp_jerk_max
        self._nlp_v_min = nlp_v_min
        self._nlp_v_max = nlp_v_max

        # NLP cost weights
        self._nlp_w_x = nlp_w_x
        self._nlp_w_v = nlp_w_v
        self._nlp_w_y = nlp_w_y
        self._nlp_w_a = nlp_w_a
        self._nlp_w_delta = nlp_w_delta

        self._milp_rho = milp_rho if milp_rho is not None else self.DEFAULT_RHO
        self._big_m = big_m if big_m is not None else self.DEFAULT_BIG_M

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
        self._ref_start_idx: int = 0

        # Obstacle data (stored for plotter access)
        self._last_obstacles: Optional[List] = None
        self._last_other_agents: Optional[Dict] = None

        # Build CasADi NLP structure once
        self._nlp_solver = None
        self._nlp_built = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_action(self, state: AgentState,
                      other_agents: Optional[Dict] = None) -> tuple:
        """MPC step: solve MILP + NLP in Frenet frame, return first action.

        Args:
            state: Current ego agent state.
            other_agents: Dict mapping agent_id -> AgentState for other
                vehicles (used for collision avoidance). May be None.

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
        obstacles = self._predict_obstacles(other_agents, self._frenet)
        self._last_obstacles = obstacles
        self._last_other_agents = other_agents

        # Goal in Frenet: s_goal = total path length, d_goal = 0
        s_goal = self._frenet.total_length

        # --- Stage 1: MILP ---
        before = time.time()
        milp_states = self._solve_milp(frenet_state, road_left, road_right,
                                       obstacles)
        milp_time = time.time() - before

        milp_ok = milp_states is not None

        # Prepare NLP warm-start
        if milp_ok:
            warm_states, warm_controls = self._milp_to_nlp_warmstart(
                milp_states, frenet_state)
            self._prev_milp_states = milp_states.copy()
        elif self._prev_nlp_states is not None:
            warm_states, warm_controls = self._shift_previous_solution()
        else:
            # No previous solution — default initial guess
            warm_states = np.zeros((H + 1, 4))
            for k in range(H + 1):
                warm_states[k, 0] = frenet_state[0] + frenet_state[3] * k * dt
                warm_states[k, 1] = frenet_state[1]
                warm_states[k, 2] = frenet_state[2]
                warm_states[k, 3] = frenet_state[3]
            warm_controls = np.zeros((H, 2))

        # --- Stage 2: NLP ---
        before_nlp = time.time()
        nlp_states, nlp_controls, nlp_ok = self._solve_nlp(
            frenet_state, warm_states, warm_controls,
            road_left, road_right, obstacles)
        nlp_time = time.time() - before_nlp

        total_time = time.time() - before
        logger.debug("MPC solve: MILP=%.3fs(%s) NLP=%.3fs(%s) total=%.3fs",
                     milp_time, "ok" if milp_ok else "fail",
                     nlp_time, "ok" if nlp_ok else "fail",
                     total_time)

        # --- Fallback logic ---
        if nlp_ok:
            final_states = nlp_states
            final_controls = nlp_controls
            self._prev_nlp_states = nlp_states.copy()
            self._prev_nlp_controls = nlp_controls.copy()
        elif milp_ok:
            # NLP failed but MILP succeeded — use MILP-converted controls
            final_states = warm_states
            final_controls = warm_controls
            self._prev_nlp_states = warm_states.copy()
            self._prev_nlp_controls = warm_controls.copy()
            logger.debug("NLP failed, using MILP warm-start directly")
        elif self._prev_nlp_states is not None:
            # Both failed — use shifted previous solution
            final_states, final_controls = self._shift_previous_solution()
            self._prev_nlp_states = final_states.copy()
            self._prev_nlp_controls = final_controls.copy()
            logger.debug("Both MILP and NLP failed, using shifted previous")
        else:
            # No previous solution — coast
            final_states = np.zeros((H + 1, 4))
            for k in range(H + 1):
                final_states[k, 0] = frenet_state[0] + frenet_state[3] * k * dt
                final_states[k, 1] = frenet_state[1]
                final_states[k, 2] = frenet_state[2]
                final_states[k, 3] = frenet_state[3]
            final_controls = np.zeros((H, 2))
            self._prev_nlp_states = final_states.copy()
            self._prev_nlp_controls = final_controls.copy()
            logger.debug("No solution available, coasting")

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
            return self._frenet.road_boundaries(s_values, self._scenario_map)
        # Fallback: standard lane width
        return (np.full(len(s_values), 3.5),
                np.full(len(s_values), -3.5))

    # ------------------------------------------------------------------
    # Obstacle prediction
    # ------------------------------------------------------------------

    def _predict_obstacles(self, other_agents, frenet):
        """Predict obstacle positions over the planning horizon.

        Constant-velocity propagation in world frame, then Frenet transform.

        Args:
            other_agents: Dict {agent_id: AgentState} or None.
            frenet: _FrenetFrame instance.

        Returns:
            List of dicts, each with keys:
                's': (H+1,) arc-length positions
                'd': (H+1,) lateral positions
                'length': vehicle length
                'width': vehicle width
        """
        if other_agents is None or frenet is None:
            return []

        H = self._horizon
        dt = self._dt
        obstacles = []

        for aid, agent_state in other_agents.items():
            pos = np.array(agent_state.position, dtype=float)
            vel = np.array(agent_state.velocity, dtype=float)

            # Constant-velocity world positions
            world_positions = np.empty((H + 1, 2))
            for k in range(H + 1):
                world_positions[k] = pos + vel * k * dt

            # Transform to Frenet
            s_arr, d_arr = frenet.world_to_frenet_batch(world_positions)

            obs_meta = getattr(agent_state, 'metadata', None)
            obs_length = obs_meta.length if obs_meta is not None else 4.5
            obs_width = obs_meta.width if obs_meta is not None else 1.8

            obstacles.append({
                's': s_arr,
                'd': d_arr,
                'length': obs_length,
                'width': obs_width,
                'heading': float(agent_state.heading),
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

    def _shift_previous_solution(self):
        """Shift the previous NLP solution forward by one planning step.

        Returns:
            (states, controls) — shifted arrays.
        """
        H = self._horizon
        states = self._prev_nlp_states
        controls = self._prev_nlp_controls

        new_states = np.empty((H + 1, 4))
        new_controls = np.empty((H, 2))

        # Shift states forward (drop first, duplicate last)
        new_states[:-1] = states[1:]
        new_states[-1] = states[-1]

        # Shift controls forward (drop first, duplicate last)
        new_controls[:-1] = controls[1:]
        new_controls[-1] = controls[-1]

        return new_states, new_controls

    # ------------------------------------------------------------------
    # Stage 1: MILP (scipy.optimize.milp)
    # ------------------------------------------------------------------

    def _solve_milp(self, frenet_state, road_left, road_right, obstacles):
        """Solve MILP in Frenet frame with point-mass model.

        Decision variables: states [s, d, vs, vd] at each step,
        controls [as, ad] at each step, L1 slack for tracking,
        and binary variables for Big-M collision avoidance.

        Args:
            frenet_state: [s, d, phi, v] current state in Frenet.
            road_left: (H+1,) left boundary offsets (positive).
            road_right: (H+1,) right boundary offsets (negative).
            obstacles: List of obstacle dicts from _predict_obstacles.

        Returns:
            (H+1, 4) array of MILP states [s, d, vs, vd] in Frenet,
            or None if MILP failed.
        """
        H = self._horizon
        dt = self._dt
        rho = self._milp_rho
        M = self._big_m

        # Initial Frenet velocity components
        s0, d0, phi0, v0 = frenet_state
        vs0 = v0 * np.cos(phi0)
        vd0 = v0 * np.sin(phi0)

        N_obs = min(len(obstacles), self.N_OBS_MAX)

        # --- Decision variable layout ---
        # States:    4*(H+1)  [s, d, vs, vd] at each step
        # Controls:  2*H      [as, ad] at each step
        # L1 slack:  2*(H+1)  [es, ed] for tracking error
        # Binaries:  4*N_obs*H for Big-M collision avoidance
        n_state = 4 * (H + 1)
        n_ctrl = 2 * H
        n_slack = 2 * (H + 1)
        n_binary = 4 * N_obs * H
        n_vars = n_state + n_ctrl + n_slack + n_binary

        s_off = 0
        c_off = n_state
        e_off = n_state + n_ctrl
        b_off = n_state + n_ctrl + n_slack

        # Goal reference: s_goal at each step, d_goal = 0
        s_goal = self._frenet.total_length
        v_goal = self._target_speed

        # --- Objective: minimise sum of L1 slack ---
        c_vec = np.zeros(n_vars)
        c_vec[e_off:e_off + n_slack] = 1.0

        # --- Count constraint rows ---
        n_init = 4
        n_dyn = 4 * H
        n_kin = 2 * (H + 1)
        n_road = 2 * (H + 1)
        n_l1 = 4 * (H + 1)
        n_coll = 5 * N_obs * H  # 4 side + 1 sum per obs per step
        n_rows = n_init + n_dyn + n_kin + n_road + n_l1 + n_coll

        A = lil_matrix((n_rows, n_vars))
        lb = np.zeros(n_rows)
        ub = np.zeros(n_rows)

        row = 0

        # --- Initial state ---
        z_init = np.array([s0, d0, vs0, vd0])
        for i in range(4):
            A[row, s_off + i] = 1.0
            lb[row] = z_init[i]
            ub[row] = z_init[i]
            row += 1

        # --- ZOH dynamics ---
        for k in range(H):
            sk = s_off + 4 * k
            sk1 = s_off + 4 * (k + 1)
            ck = c_off + 2 * k

            # s_{k+1} = s_k + vs_k * dt
            A[row, sk1 + 0] = 1.0
            A[row, sk + 0] = -1.0
            A[row, sk + 2] = -dt
            lb[row] = 0.0; ub[row] = 0.0; row += 1

            # d_{k+1} = d_k + vd_k * dt
            A[row, sk1 + 1] = 1.0
            A[row, sk + 1] = -1.0
            A[row, sk + 3] = -dt
            lb[row] = 0.0; ub[row] = 0.0; row += 1

            # vs_{k+1} = vs_k + as_k * dt
            A[row, sk1 + 2] = 1.0
            A[row, sk + 2] = -1.0
            A[row, ck + 0] = -dt
            lb[row] = 0.0; ub[row] = 0.0; row += 1

            # vd_{k+1} = vd_k + ad_k * dt
            A[row, sk1 + 3] = 1.0
            A[row, sk + 3] = -1.0
            A[row, ck + 1] = -dt
            lb[row] = 0.0; ub[row] = 0.0; row += 1

        # --- Kinematic feasibility: vs >= rho * |vd| ---
        for k in range(H + 1):
            sk = s_off + 4 * k
            # vs - rho * vd >= 0
            A[row, sk + 2] = 1.0
            A[row, sk + 3] = -rho
            lb[row] = 0.0; ub[row] = np.inf; row += 1
            # vs + rho * vd >= 0
            A[row, sk + 2] = 1.0
            A[row, sk + 3] = rho
            lb[row] = 0.0; ub[row] = np.inf; row += 1

        # --- Road boundaries: d_right <= d <= d_left ---
        for k in range(H + 1):
            sk = s_off + 4 * k
            # d_k >= d_right[k]
            A[row, sk + 1] = 1.0
            lb[row] = road_right[k]; ub[row] = np.inf; row += 1
            # d_k <= d_left[k]  =>  -d_k >= -d_left[k]
            A[row, sk + 1] = -1.0
            lb[row] = -road_left[k]; ub[row] = np.inf; row += 1

        # --- L1 slack: e >= |state_ref - state| (for s and d tracking) ---
        for k in range(H + 1):
            sk = s_off + 4 * k
            ek = e_off + 2 * k
            # Reference s at step k: linearly interpolate toward s_goal
            ref_s = s0 + v_goal * k * dt
            ref_s = min(ref_s, s_goal)
            ref_d = 0.0  # stay centred

            # es >= s - ref_s  =>  es - s >= -ref_s
            A[row, ek + 0] = 1.0
            A[row, sk + 0] = -1.0
            lb[row] = -ref_s; ub[row] = np.inf; row += 1

            # es >= -(s - ref_s)  =>  es + s >= ref_s
            A[row, ek + 0] = 1.0
            A[row, sk + 0] = 1.0
            lb[row] = ref_s; ub[row] = np.inf; row += 1

            # ed >= d - ref_d  =>  ed - d >= -ref_d
            A[row, ek + 1] = 1.0
            A[row, sk + 1] = -1.0
            lb[row] = -ref_d; ub[row] = np.inf; row += 1

            # ed >= -(d - ref_d)  =>  ed + d >= ref_d
            A[row, ek + 1] = 1.0
            A[row, sk + 1] = 1.0
            lb[row] = ref_d; ub[row] = np.inf; row += 1

        # --- Collision avoidance (Big-M) ---
        for obs_idx in range(N_obs):
            obs = obstacles[obs_idx]

            # Project obstacle body-frame dimensions into Frenet (s, d)
            obs_half_L = obs['length'] / 2.0
            obs_half_W = obs['width'] / 2.0
            obs_s0 = float(obs['s'][0])
            _, _, _, road_angle = self._frenet._interpolate(obs_s0)
            dh = obs.get('heading', road_angle) - road_angle
            half_s_obs = abs(obs_half_L * np.cos(dh)) + abs(obs_half_W * np.sin(dh))
            half_d_obs = abs(obs_half_L * np.sin(dh)) + abs(obs_half_W * np.cos(dh))

            half_s = self._ego_length / 2.0 + half_s_obs + self._collision_margin
            half_d = self._ego_width / 2.0 + half_d_obs + self._collision_margin

            for k in range(H):
                sk = s_off + 4 * (k + 1)  # ego state at k+1 (skip initial)
                b_base = b_off + 4 * (obs_idx * H + k)

                s_obs = obs['s'][k + 1] if k + 1 < len(obs['s']) else obs['s'][-1]
                d_obs = obs['d'][k + 1] if k + 1 < len(obs['d']) else obs['d'][-1]

                # s_ego - s_obs >= half_s - M * b1
                # => s_ego + M*b1 >= half_s + s_obs
                A[row, sk + 0] = 1.0
                A[row, b_base + 0] = M
                lb[row] = half_s + s_obs; ub[row] = np.inf; row += 1

                # s_obs - s_ego >= half_s - M * b2
                # => -s_ego + M*b2 >= half_s - s_obs
                A[row, sk + 0] = -1.0
                A[row, b_base + 1] = M
                lb[row] = half_s - s_obs; ub[row] = np.inf; row += 1

                # d_ego - d_obs >= half_d - M * b3
                A[row, sk + 1] = 1.0
                A[row, b_base + 2] = M
                lb[row] = half_d + d_obs; ub[row] = np.inf; row += 1

                # d_obs - d_ego >= half_d - M * b4
                A[row, sk + 1] = -1.0
                A[row, b_base + 3] = M
                lb[row] = half_d - d_obs; ub[row] = np.inf; row += 1

                # sum(b_i) <= 3  =>  at least one constraint active
                for j in range(4):
                    A[row, b_base + j] = 1.0
                lb[row] = -np.inf; ub[row] = 3.0; row += 1

        # Trim unused rows
        A = A[:row]
        lb = lb[:row]
        ub = ub[:row]

        # --- Variable bounds ---
        var_lb = np.full(n_vars, -np.inf)
        var_ub = np.full(n_vars, np.inf)

        # State bounds: milp_v_min <= vs <= milp_v_max
        for k in range(H + 1):
            var_lb[s_off + 4 * k + 2] = self._milp_v_min
            var_ub[s_off + 4 * k + 2] = self._milp_v_max

        # Control bounds (separate s and d)
        var_lb[c_off:c_off + n_ctrl:2] = self._milp_a_x_min    # as lower
        var_ub[c_off:c_off + n_ctrl:2] = self._milp_a_x_max    # as upper
        var_lb[c_off + 1:c_off + n_ctrl:2] = self._milp_a_y_min  # ad lower
        var_ub[c_off + 1:c_off + n_ctrl:2] = self._milp_a_y_max  # ad upper

        # Slack bounds: >= 0
        var_lb[e_off:e_off + n_slack] = 0.0

        # Binary bounds: [0, 1]
        var_lb[b_off:] = 0.0
        var_ub[b_off:] = 1.0

        # --- Integrality ---
        integrality = np.zeros(n_vars)
        integrality[b_off:] = 1  # binary variables

        # --- Solve ---
        constraints = LinearConstraint(csc_matrix(A), lb, ub)

        try:
            result = milp(
                c=c_vec,
                constraints=constraints,
                integrality=integrality,
                bounds=Bounds(var_lb, var_ub),
                options={'time_limit': 2.0},
            )
        except Exception as e:
            logger.debug("MILP solver exception: %s", e)
            return None

        if not result.success:
            logger.debug("MILP did not converge: %s", result.message)
            return None

        # Extract states: (H+1, 4) [s, d, vs, vd]
        milp_states = result.x[s_off:s_off + n_state].reshape(H + 1, 4)
        return milp_states

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
                          self._nlp_a_min, self._nlp_a_max)

            # Steering from heading change
            d_phi = (phis[k + 1] - phis[k] + np.pi) % (2 * np.pi) - np.pi
            spd = max(speeds[k], 0.5)
            sin_arg = np.clip(d_phi * L / (2.0 * spd * dt), -1.0, 1.0)
            delta_k = np.clip(np.arcsin(sin_arg),
                              -self._nlp_delta_max, self._nlp_delta_max)

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

        try:
            opti = ca.Opti()

            # Decision variables
            S = opti.variable(4, H + 1)   # [s; d; phi; v] at each step
            U = opti.variable(2, H)        # [a; delta] at each step

            # Parameters for obstacle positions (pre-allocated slots)
            N_obs = min(len(obstacles), self.N_OBS_MAX)

            # Goal
            s_goal = self._frenet.total_length
            v_goal = self._target_speed

            # --- Cost function ---
            cost = 0.0
            for k in range(H + 1):
                cost += self._nlp_w_x * (S[0, k] - s_goal)**2
                cost += self._nlp_w_v * (S[3, k] - v_goal)**2
                cost += self._nlp_w_y * S[1, k]**2
            for k in range(H):
                cost += self._nlp_w_a * U[0, k]**2
                cost += self._nlp_w_delta * U[1, k]**2
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

            # --- Road boundary constraints ---
            for k in range(H + 1):
                opti.subject_to(S[1, k] >= road_right[k])
                opti.subject_to(S[1, k] <= road_left[k])

            # --- Control bounds ---
            for k in range(H):
                opti.subject_to(opti.bounded(self._nlp_a_min, U[0, k], self._nlp_a_max))
                opti.subject_to(opti.bounded(-self._nlp_delta_max, U[1, k], self._nlp_delta_max))

            # --- Speed bounds ---
            for k in range(H + 1):
                opti.subject_to(opti.bounded(self._nlp_v_min, S[3, k], self._nlp_v_max))

            # --- Jerk constraints ---
            jerk_limit = self._nlp_jerk_max * dt
            for k in range(H - 1):
                opti.subject_to(opti.bounded(-jerk_limit,
                                             U[0, k + 1] - U[0, k],
                                             jerk_limit))

            # --- Elliptical collision avoidance (corner-based) ---
            # Check all four corners of the rotated ego vehicle against
            # an exclusion ellipse around each obstacle.
            half_L = self._ego_length / 2.0
            half_W = self._ego_width / 2.0

            for obs_idx in range(N_obs):
                obs = obstacles[obs_idx]

                # Project obstacle body-frame dimensions into Frenet (s, d)
                obs_half_L = obs['length'] / 2.0
                obs_half_W = obs['width'] / 2.0
                obs_s0 = float(obs['s'][0])
                _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
                dh = obs.get('heading', road_angle_obs) - road_angle_obs
                rx = abs(obs_half_L * np.cos(dh)) + abs(obs_half_W * np.sin(dh)) + self._collision_margin
                ry = abs(obs_half_L * np.sin(dh)) + abs(obs_half_W * np.cos(dh)) + self._collision_margin

                for k in range(1, H + 1):
                    s_obs = float(obs['s'][k]) if k < len(obs['s']) else float(obs['s'][-1])
                    d_obs = float(obs['d'][k]) if k < len(obs['d']) else float(obs['d'][-1])

                    cos_phi = ca.cos(S[2, k])
                    sin_phi = ca.sin(S[2, k])

                    for sl, sw in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        c_s = S[0, k] + sl * half_L * cos_phi - sw * half_W * sin_phi
                        c_d = S[1, k] + sl * half_L * sin_phi + sw * half_W * cos_phi

                        opti.subject_to(
                            ((c_s - s_obs) / rx)**2 +
                            ((c_d - d_obs) / ry)**2 >= 1.0)

            # --- Initial guess ---
            opti.set_initial(S, warm_states.T)
            opti.set_initial(U, warm_controls.T)

            # --- Solver options ---
            p_opts = {'expand': True}
            s_opts = {
                'max_iter': 200,
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
            return nlp_states, nlp_controls, True

        except Exception as e:
            logger.debug("NLP solver failed: %s", e)
            return warm_states, warm_controls, False
