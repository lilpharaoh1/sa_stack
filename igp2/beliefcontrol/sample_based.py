"""Sample-based trajectory planner with PID tracking.

Samples random action sequences, forward-simulates them through the
bicycle kinematic model, picks the candidate closest to the reference
path, and PID-tracks it.
"""

import logging
from typing import List

import numpy as np

from igp2.core.agentstate import AgentState, AgentMetadata
from igp2.core.vehicle import Action
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
        self._waypoint_margin = self.WAYPOINT_MARGIN * (20.0 / fps)

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

        if dists[-1] < self._waypoint_margin:
            target_idx = len(best_traj) - 1
        else:
            far = dists[closest_idx:]
            offset = np.argmax(far >= self._waypoint_margin)
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
