"""
Minimal belief-conditioned agent.

Extends Agent directly with:
- A beliefs dict for storing assumptions about other agents
- An update_beliefs() hook called each timestep
- A policy() hook that delegates to a configurable control policy
- A reference path to the goal computed via A* open-loop maneuvers
- Internal trajectory prediction for other agents (bicycle model forward sim)
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from igp2.agents.agent import Agent
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal
from igp2.core.vehicle import KinematicVehicle, Action, Observation, TrajectoryPrediction
from igp2.opendrive.map import Map
from igp2.opendrive.elements.road_lanes import Lane
from igp2.recognition.astar import AStar
from igp2.beliefcontrol.policy import SampleBased, TwoStageOPT
from igp2.beliefcontrol.plotting import BeliefPlotter, OptimisationPlotter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory forward-simulation utilities
# ---------------------------------------------------------------------------

def _bicycle_step(position: np.ndarray, heading: float, speed: float,
                  accel: float, steer: float, dt: float,
                  wheelbase: float) -> Tuple[np.ndarray, float, float]:
    """Single kinematic bicycle model step (matches CarlaSim._bicycle_step)."""
    x, y = position
    x_new = x + speed * np.cos(heading + steer) * dt
    y_new = y + speed * np.sin(heading + steer) * dt
    theta_new = heading + (2.0 * speed / wheelbase) * np.sin(steer) * dt
    v_new = max(0.0, speed + accel * dt)
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    return np.array([x_new, y_new]), theta_new, v_new


class _WaypointTracker:
    """Lightweight PID waypoint tracker for forward simulation.

    Replicates WaypointManeuver's PID control without modifying agent state.
    """

    LATERAL_KP = 1.0
    LATERAL_KI = 0.2
    LATERAL_KD = 0.05
    LONGITUDINAL_KP = 1.0
    LONGITUDINAL_KI = 0.05
    LONGITUDINAL_KD = 0.0

    def __init__(self, waypoints: np.ndarray, target_speed: float, dt: float):
        self.waypoints = waypoints
        self.target_speed = target_speed
        self.dt = dt
        self.waypoint_idx = 0
        self.waypoint_margin = max(1.0, dt * 20.0)
        self.lat_integral = 0.0
        self.lat_prev_error = 0.0
        self.lon_integral = 0.0
        self.lon_prev_error = 0.0

    def get_action(self, position: np.ndarray, heading: float, speed: float,
                   max_steer: float = 0.5, max_accel: float = 3.0
                   ) -> Tuple[float, float]:
        if self.waypoint_idx >= len(self.waypoints):
            return 0.0, 0.0

        while self.waypoint_idx < len(self.waypoints) - 1:
            if np.linalg.norm(self.waypoints[self.waypoint_idx] - position) < self.waypoint_margin:
                self.waypoint_idx += 1
            else:
                break

        target = self.waypoints[self.waypoint_idx]
        dx, dy = target[0] - position[0], target[1] - position[1]
        target_heading = np.arctan2(dy, dx)

        lat_error = (target_heading - heading + np.pi) % (2 * np.pi) - np.pi
        lon_error = self.target_speed - speed

        self.lat_integral += lat_error * self.dt
        lat_derivative = (lat_error - self.lat_prev_error) / self.dt if self.dt > 0 else 0.0
        steer = np.clip(
            self.LATERAL_KP * lat_error + self.LATERAL_KI * self.lat_integral + self.LATERAL_KD * lat_derivative,
            -max_steer, max_steer)
        self.lat_prev_error = lat_error

        self.lon_integral += lon_error * self.dt
        lon_derivative = (lon_error - self.lon_prev_error) / self.dt if self.dt > 0 else 0.0
        accel = np.clip(
            self.LONGITUDINAL_KP * lon_error + self.LONGITUDINAL_KI * self.lon_integral + self.LONGITUDINAL_KD * lon_derivative,
            -max_accel, max_accel)
        self.lon_prev_error = lon_error

        return accel, steer


def _get_agent_waypoints_and_speeds(agent) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract all future waypoints and velocities from an agent's macro actions."""
    if not hasattr(agent, 'macro_actions') or not agent.macro_actions:
        return None, None

    all_waypoints = []
    all_speeds = []
    for macro in agent.macro_actions:
        if not hasattr(macro, '_maneuvers'):
            continue
        for maneuver in macro._maneuvers:
            if hasattr(maneuver, 'trajectory') and maneuver.trajectory is not None:
                traj = maneuver.trajectory
                all_waypoints.append(traj.path)
                if traj.velocity is not None and len(traj.velocity) == len(traj.path):
                    vel = np.asarray(traj.velocity)
                    all_speeds.append(vel if vel.ndim == 1 else np.linalg.norm(vel, axis=1))
                else:
                    all_speeds.append(None)

    if not all_waypoints:
        return None, None

    waypoints = np.vstack(all_waypoints)
    speeds = np.concatenate(all_speeds) if all(s is not None for s in all_speeds) else None
    return waypoints, speeds


def _simulate_agent_trajectory(agent, frame: Dict[int, AgentState],
                               n_steps: int, dt: float) -> Optional[np.ndarray]:
    """Forward simulate one agent's trajectory using bicycle model + PID."""
    if not hasattr(agent, 'agent_id'):
        return None

    aid = agent.agent_id
    if aid not in frame:
        return None

    waypoints, speeds = _get_agent_waypoints_and_speeds(agent)
    if waypoints is None or len(waypoints) < 2:
        return None

    meta = agent.metadata if hasattr(agent, 'metadata') else None
    wheelbase = meta.wheelbase if meta is not None else 2.5
    max_steer = getattr(meta, 'max_steer', 0.5) if meta is not None else 0.5
    max_accel = getattr(meta, 'max_acceleration', 3.0) if meta is not None else 3.0

    state = frame[aid]
    position = np.array(state.position)
    heading = float(state.heading)
    speed = float(state.speed)

    dists = np.linalg.norm(waypoints - position, axis=1)
    closest_idx = int(np.argmin(dists))
    future_wp = waypoints[closest_idx:]
    future_sp = speeds[closest_idx:] if speeds is not None else None

    if len(future_wp) < 2:
        return None

    if future_sp is not None and len(future_sp) > 0:
        target_speed = float(np.mean(future_sp))
    elif speed > 0.5:
        target_speed = speed
    else:
        target_speed = 5.0

    tracker = _WaypointTracker(future_wp, target_speed, dt)

    trajectory = np.zeros((n_steps + 1, 2))
    trajectory[0] = position
    for k in range(n_steps):
        accel, steer = tracker.get_action(position, heading, speed, max_steer, max_accel)
        position, heading, speed = _bicycle_step(position, heading, speed, accel, steer, dt, wheelbase)
        trajectory[k + 1] = position

    return trajectory

# Mapping from string name to policy class for configuration
POLICY_TYPES = {
    "sample_based": SampleBased,
    "two_stage_opt": TwoStageOPT,
}


class BeliefAgent(Agent):
    """Agent whose control policy is conditioned on a belief state.

    At construction time, runs A* to compute an open-loop reference path
    to the goal (same as TrafficAgent) and stores it as a list of
    (Lane, waypoints) tuples.

    The simulation loop calls next_action() each step, which:
      1. Calls update_beliefs(observation) to revise beliefs
      2. Calls policy(observation) to produce an action via the
         configured control policy.

    The control policy is selected via ``policy_type``:

    * ``"sample_based"`` — :class:`SampleBased` (random sampling + PID).
    * ``"two_stage_opt"`` — :class:`TwoStageOPT`
      (NLP with drivable-area, speed, and smoothness constraints).

    Extra keyword arguments are forwarded to the policy constructor,
    so you can pass e.g. ``n_samples=100`` for SampleBased or
    ``w_ref=2.0`` for TwoStageOPT.

    Args:
        agent_id: ID of the agent.
        initial_state: Starting state.
        goal: Goal to navigate toward.
        fps: Simulation framerate.
        scenario_map: The road layout used for A* path planning.
        beliefs: Initial belief state dict. Defaults to empty.
        policy_type: Which control policy to use. One of
            ``"sample_based"`` or ``"two_stage_opt"``.
        plot_interval: Plot every N steps (0 to disable plotting).
        **policy_kwargs: Additional keyword arguments forwarded to the
            policy constructor (e.g. ``horizon``, ``n_samples``,
            ``target_speed``, ``w_ref``, etc.).
    """

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 goal: Goal = None,
                 fps: int = 20,
                 scenario_map: Map = None,
                 beliefs: Dict[str, Any] = None,
                 policy_type: str = "two_stage_opt",
                 plot_interval: int = 1,
                 **policy_kwargs):
        super().__init__(agent_id, initial_state, goal, fps)
        self._vehicle = KinematicVehicle(initial_state, self.metadata, fps)
        self._beliefs = beliefs if beliefs is not None else {}
        self._scenario_map = scenario_map
        self._policy_type = policy_type
        self._plot_interval = plot_interval
        self._step_count = 0
        self._other_agents: Dict[int, Any] = {}  # References to other agents in the scene
        self._agent_trajectories: Dict[int, np.ndarray] = {}  # Predicted trajectories

        # Compute reference path to goal via A* open-loop maneuvers.
        self._reference_path: List[Tuple[Lane, np.ndarray]] = []
        self._reference_waypoints: np.ndarray = np.empty((0, 2))
        if scenario_map is not None and goal is not None:
            self._compute_reference_path(agent_id, initial_state, goal, scenario_map)

        # Build the control policy
        self._policy_obj = self._build_policy(
            policy_type, fps, scenario_map, **policy_kwargs,
        )

        # Build the matching plotter
        self._plotter = None
        if scenario_map is not None and plot_interval > 0:
            self._plotter = self._build_plotter(
                policy_type, scenario_map, goal,
            )

    def _build_policy(self, policy_type, fps, scenario_map, **kwargs):
        """Instantiate the control policy based on ``policy_type``."""
        if policy_type not in POLICY_TYPES:
            raise ValueError(
                f"Unknown policy_type {policy_type!r}. "
                f"Choose from: {list(POLICY_TYPES.keys())}"
            )

        common = dict(
            fps=fps,
            metadata=self.metadata,
            reference_waypoints=self._reference_waypoints,
        )

        if policy_type == "two_stage_opt":
            common["scenario_map"] = scenario_map
            common.update(kwargs)
            return TwoStageOPT(**common)
        else:
            common.update(kwargs)
            return SampleBased(**common)

    def _build_plotter(self, policy_type, scenario_map, goal):
        """Instantiate the plotter matching ``policy_type``."""
        if policy_type == "two_stage_opt":
            return OptimisationPlotter(
                scenario_map, self._reference_waypoints,
                self.metadata, goal,
            )
        else:
            return BeliefPlotter(
                scenario_map, self._reference_waypoints, goal,
            )

    def _compute_reference_path(self, agent_id, initial_state, goal, scenario_map):
        """Run A* to find open-loop macro actions, then extract (Lane, waypoints)."""
        astar = AStar(max_iter=1000)
        frame = {agent_id: initial_state}
        _, solutions = astar.search(
            agent_id, frame, goal, scenario_map, open_loop=True,
        )
        if not solutions:
            logger.warning("BeliefAgent %d: A* found no path to goal %s", agent_id, goal)
            return

        macro_actions = solutions[0]
        all_waypoints = []
        for ma in macro_actions:
            for maneuver in ma._maneuvers:
                lane = maneuver.lane_sequence[0] if maneuver.lane_sequence else None
                waypoints = maneuver.trajectory.path.copy()
                self._reference_path.append((lane, waypoints))
                all_waypoints.append(waypoints)

        if all_waypoints:
            combined = [all_waypoints[0]]
            for wp in all_waypoints[1:]:
                combined.append(wp[1:])
            self._reference_waypoints = np.concatenate(combined, axis=0)

        logger.info(
            "BeliefAgent %d: reference path has %d segments (%d waypoints)",
            agent_id, len(self._reference_path), len(self._reference_waypoints),
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"BeliefAgent(ID={self.agent_id}, policy={self._policy_type})"

    @property
    def beliefs(self) -> Dict[str, Any]:
        """Current belief state."""
        return self._beliefs

    @property
    def reference_path(self) -> List[Tuple[Lane, np.ndarray]]:
        """Reference path as (Lane, waypoints) tuples, one per maneuver."""
        return self._reference_path

    def update_beliefs(self, observation: Observation):
        """Update beliefs from the current observation.

        Override this to implement belief tracking logic.

        Args:
            observation: Current observation with frame and scenario_map.
        """
        pass

    def set_agents(self, agents: Dict[int, Any]):
        """Register the other agents in the scene.

        The BeliefAgent will forward-simulate these agents each step to
        predict their trajectories for collision avoidance.  Later, beliefs
        will control which agents are visible and how far ahead to predict.

        Args:
            agents: Dict mapping agent_id -> Agent instance.  The ego agent
                (self) may be included; it will be skipped automatically.
        """
        self._other_agents = {aid: a for aid, a in agents.items()
                              if aid != self.agent_id}

    def _predict_agent_trajectories(self, frame: Dict[int, AgentState]) -> Dict[int, np.ndarray]:
        """Forward-simulate other agents to predict their trajectories.

        Uses the bicycle model + PID waypoint tracker to replicate what
        each agent's closed-loop controller will do.

        Args:
            frame: Current observation frame with all agent states.

        Returns:
            Dict mapping agent_id -> (T, 2) predicted position array.
        """
        dt = 1.0 / self._fps
        # Compute horizon from the policy if available
        policy = self._policy_obj
        if hasattr(policy, '_horizon') and hasattr(policy, '_dt'):
            horizon_time = policy._horizon * policy._dt + 1.0
        else:
            horizon_time = 10.0
        n_steps = int(horizon_time / dt)

        trajectories = {}
        for aid, agent in self._other_agents.items():
            traj = _simulate_agent_trajectory(agent, frame, n_steps, dt)
            if traj is not None:
                trajectories[aid] = traj
        return trajectories

    def policy(self, observation: Observation) -> Action:
        """Delegate to the configured control policy to produce an action.

        Args:
            observation: Current observation.

        Returns:
            Action to execute.
        """
        ego_state = observation.frame.get(self.agent_id)
        if ego_state is None or self._scenario_map is None:
            return Action(acceleration=0.0, steer_angle=0.0, target_speed=0.0)

        # Extract current lane info
        current_lane = self._scenario_map.best_lane_at(
            ego_state.position, ego_state.heading,
        )
        if current_lane is not None:
            lane_midline = current_lane.midline
            lane_boundary = current_lane.boundary

        # If no reference path, just stop
        if len(self._reference_waypoints) == 0:
            return Action(acceleration=0.0, steer_angle=0.0, target_speed=0.0)

        # Extract other agents for collision avoidance
        other_agents = {aid: s for aid, s in observation.frame.items()
                        if aid != self.agent_id}

        # Run the policy
        if isinstance(self._policy_obj, TwoStageOPT):
            action, candidates, best_idx = self._policy_obj.select_action(
                ego_state,
                other_agents=other_agents if other_agents else None,
                agent_trajectories=self._agent_trajectories if self._agent_trajectories else None)
        else:
            action, candidates, best_idx = self._policy_obj.select_action(ego_state)

        # Plot if enabled
        if (self._plotter is not None
                and self._plot_interval > 0
                and self._step_count % self._plot_interval == 0):
            self._plot(ego_state, candidates, best_idx)

        return action

    def _plot(self, ego_state, candidates, best_idx):
        """Dispatch to the correct plotter depending on policy type."""
        if isinstance(self._plotter, OptimisationPlotter):
            full_rollout = getattr(self._policy_obj, 'last_rollout', None)
            milp_rollout = getattr(self._policy_obj, 'last_milp_rollout', None)
            trajectory = candidates[best_idx] if candidates else None

            # Retrieve obstacle data for visualisation
            other_agents = getattr(self._policy_obj, 'last_other_agents', None)
            obstacles = getattr(self._policy_obj, 'last_obstacles', None)
            frenet = getattr(self._policy_obj, 'frenet_frame', None)
            collision_margin = getattr(self._policy_obj, 'collision_margin', 0.5)
            ego_length = getattr(self._policy_obj, 'ego_length', 4.5)
            ego_width = getattr(self._policy_obj, 'ego_width', 1.8)

            self._plotter.update(
                ego_state, trajectory, full_rollout,
                self._trajectory_cl, self.agent_id, self._step_count,
                milp_trajectory=milp_rollout,
                other_agents=other_agents,
                obstacles=obstacles,
                frenet=frenet,
                ego_length=ego_length,
                ego_width=ego_width,
                collision_margin=collision_margin,
            )
        else:
            self._plotter.update(
                ego_state, candidates, best_idx,
                self._trajectory_cl, self.agent_id, self._step_count,
            )

    def done(self, observation: Observation) -> bool:
        """Check whether the agent has finished its task."""
        return False

    def next_action(self, observation: Observation,
                    prediction: TrajectoryPrediction = None) -> Action:
        """Compute next action: predict other agents, update beliefs, then apply policy.

        Args:
            observation: Current observation.
            prediction: Unused, kept for interface compatibility.

        Returns:
            Belief-conditioned action.
        """
        # EMRAN TODO: Have reference path generated here when no path is found.
        if len(self.reference_path) == 0:
            pass

        self._step_count += 1

        # Predict other agents' trajectories for collision avoidance
        if self._other_agents:
            self._agent_trajectories = self._predict_agent_trajectories(
                observation.frame)

        self.update_beliefs(observation)
        return self.policy(observation)

    def reset(self):
        """Reset agent to initialisation defaults."""
        super().reset()
        self._vehicle = KinematicVehicle(self._initial_state, self.metadata, self._fps)
        self._beliefs = {}
        self._step_count = 0
        self._policy_obj.reset()
