"""
MCTS Belief Agent — drives using the MCTS trajectory planner directly.

Uses the same MCTS planner from ``igp2.beliefcontrol.mcts_planner`` to
generate candidate trajectories, then executes the first action of the
best trajectory at each planning step.

This agent is simpler than BeliefAgent: no dual human/true policy, no
belief inference loop.  It exists to test whether the MCTS planner
produces viable, collision-avoiding trajectories.
"""

import logging
import time as _time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from igp2.agents.agent import Agent
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal
from igp2.core.vehicle import KinematicVehicle, Action, Observation, TrajectoryPrediction
from igp2.opendrive.map import Map
from igp2.opendrive.elements.road_lanes import Lane
from igp2.recognition.astar import AStar
from igp2.beliefcontrol.frenet import FrenetFrame
from igp2.beliefcontrol.mcts_planner import MCTSPlanner
from igp2.beliefcontrol.planning_utils import sample_road_boundaries
from igp2.beliefcontrol.belief_inference import InferenceResult
from igp2.beliefcontrol.plotting import MCTSTrajectoryPlotter, MCTSTreePlotter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Forward-simulation helpers (duplicated from belief_agent to stay standalone)
# ---------------------------------------------------------------------------

def _bicycle_step(position, heading, speed, accel, steer, dt, wheelbase):
    x, y = position
    x_new = x + speed * np.cos(heading + steer) * dt
    y_new = y + speed * np.sin(heading + steer) * dt
    theta_new = heading + (2.0 * speed / wheelbase) * np.sin(steer) * dt
    v_new = max(0.0, speed + accel * dt)
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
    return np.array([x_new, y_new]), theta_new, v_new


class _WaypointTracker:
    LATERAL_KP = 1.0
    LATERAL_KI = 0.2
    LATERAL_KD = 0.05
    LONGITUDINAL_KP = 1.0
    LONGITUDINAL_KI = 0.05
    LONGITUDINAL_KD = 0.0

    def __init__(self, waypoints, target_speed, dt):
        self.waypoints = waypoints
        self.target_speed = target_speed
        self.dt = dt
        self.waypoint_idx = 0
        self.waypoint_margin = max(1.0, dt * 20.0)
        self.lat_integral = 0.0
        self.lat_prev_error = 0.0
        self.lon_integral = 0.0
        self.lon_prev_error = 0.0

    def get_action(self, position, heading, speed, max_steer=0.5, max_accel=3.0):
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


def _get_agent_waypoints_and_speeds(agent):
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


def _simulate_agent_trajectory(agent, frame, n_steps, dt):
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


# ---------------------------------------------------------------------------
# MCTSBeliefAgent
# ---------------------------------------------------------------------------

class MCTSBeliefAgent(Agent):
    """Agent that drives using the MCTS trajectory planner.

    Each step:
      1. Predicts other agent trajectories (forward sim with PID)
      2. Converts obstacles to Frenet frame
      3. Runs MCTS to get candidate trajectories
      4. Executes the first action of the best trajectory

    Args:
        agent_id: Agent ID.
        initial_state: Starting state.
        goal: Goal to navigate toward.
        fps: Simulation framerate.
        scenario_map: Road layout for A* reference path.
        target_speed: Desired cruising speed (m/s).
        horizon: Planning horizon in fine steps.
        plot: Enable live debug plots.
        **mcts_kwargs: Extra arguments forwarded to MCTSPlanner.
    """

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 goal: Goal = None,
                 fps: int = 20,
                 scenario_map: Map = None,
                 target_speed: float = 10.0,
                 horizon: int = 40,
                 plot: bool = True,
                 **mcts_kwargs):
        super().__init__(agent_id, initial_state, goal, fps)
        self._vehicle = KinematicVehicle(initial_state, self.metadata, fps)
        self._scenario_map = scenario_map
        self._target_speed = target_speed
        self._dt = 1.0 / fps
        self._horizon = horizon
        self._step_count = 0
        self._other_agents: Dict[int, Any] = {}
        self._prev_action = None  # scalar accel (1-D planner)

        # Compute reference path via A*
        self._reference_path: List[Tuple[Lane, np.ndarray]] = []
        self._reference_waypoints: np.ndarray = np.empty((0, 2))
        if scenario_map is not None and goal is not None:
            self._compute_reference_path(agent_id, initial_state, goal, scenario_map)

        # Build Frenet frame
        self._frenet: Optional[FrenetFrame] = None
        if len(self._reference_waypoints) >= 2:
            self._frenet = FrenetFrame(self._reference_waypoints)

        # Build MCTS planner
        self._mcts_planner: Optional[MCTSPlanner] = None
        if self._frenet is not None:
            self._mcts_planner = MCTSPlanner(
                horizon=horizon,
                dt=self._dt,
                ego_length=self.metadata.length,
                ego_width=self.metadata.width,
                wheelbase=self.metadata.wheelbase,
                collision_margin=mcts_kwargs.pop('collision_margin', 0.3),
                target_speed=target_speed,
                frenet=self._frenet,
                nlp_params=mcts_kwargs.pop('nlp_params', None),
                **mcts_kwargs,
            )

        # Plotters
        self._traj_plotter: Optional[MCTSTrajectoryPlotter] = None
        self._tree_plotter: Optional[MCTSTreePlotter] = None
        if plot and self._frenet is not None and self._mcts_planner is not None:
            self._traj_plotter = MCTSTrajectoryPlotter(
                scenario_map, self._reference_waypoints, self._frenet,
                dt=self._dt,
                ego_length=self.metadata.length,
                ego_width=self.metadata.width)
            self._tree_plotter = MCTSTreePlotter(
                self._frenet, self._mcts_planner._horizon, self._dt)

        # Per-step timing
        self.last_step_timing: Dict[str, float] = {}

    def __repr__(self) -> str:
        return f"MCTSBeliefAgent(ID={self.agent_id})"

    def _compute_reference_path(self, agent_id, initial_state, goal, scenario_map):
        astar = AStar(max_iter=1000)
        frame = {agent_id: initial_state}
        _, solutions = astar.search(agent_id, frame, goal, scenario_map, open_loop=True)
        if not solutions:
            logger.warning("MCTSBeliefAgent %d: A* found no path to goal", agent_id)
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
        logger.info("MCTSBeliefAgent %d: reference path %d segments (%d waypoints)",
                     agent_id, len(self._reference_path), len(self._reference_waypoints))

    def set_agents(self, agents: Dict[int, Any]):
        self._other_agents = {aid: a for aid, a in agents.items()
                              if aid != self.agent_id}

    def done(self, observation: Observation) -> bool:
        return self.goal is not None and self.goal.reached(self.state.position)

    def _predict_agent_trajectories(self, frame):
        dt = 1.0 / self._fps
        horizon_time = self._horizon * self._dt + 1.0
        n_steps = int(horizon_time / dt)
        trajectories = {}
        for aid, agent in self._other_agents.items():
            if not getattr(agent, 'alive', True):
                continue
            traj = _simulate_agent_trajectory(agent, frame, n_steps, dt)
            if traj is not None:
                trajectories[aid] = traj
        return trajectories

    def _state_to_frenet(self, state: AgentState) -> np.ndarray:
        f = self._frenet.world_to_frenet(
            float(state.position[0]), float(state.position[1]),
            heading=float(state.heading))
        return np.array([f['s'], f['d'], f['heading'], float(state.speed)])

    def _predict_obstacles(self, other_agents, agent_trajectories):
        """Convert other agents to obstacle dicts for the MCTS planner."""
        if not other_agents:
            return []

        obstacles = []
        H = self._horizon
        dt = self._dt

        for aid, agent_state in other_agents.items():
            meta = None
            if aid in self._other_agents and hasattr(self._other_agents[aid], 'metadata'):
                meta = self._other_agents[aid].metadata

            length = meta.length if meta else 4.5
            width = meta.width if meta else 1.8

            # Use predicted trajectory if available
            if agent_trajectories and aid in agent_trajectories:
                world_traj = agent_trajectories[aid]
                # Convert to Frenet
                frenet_traj = np.zeros((len(world_traj), 2))
                for k in range(len(world_traj)):
                    fk = self._frenet.world_to_frenet(
                        world_traj[k, 0], world_traj[k, 1])
                    frenet_traj[k] = [fk['s'], fk['d']]

                obstacles.append({
                    'agent_id': aid,
                    's': frenet_traj[:, 0],
                    'd': frenet_traj[:, 1],
                    'length': length,
                    'width': width,
                })
            else:
                # Constant velocity fallback
                f_obs = self._frenet.world_to_frenet(
                    agent_state.position[0], agent_state.position[1],
                    heading=float(agent_state.heading))
                s, d = f_obs['s'], f_obs['d']
                rel_heading = f_obs['heading']
                vs = agent_state.speed * np.cos(rel_heading)

                s_arr = np.array([s + vs * k * dt for k in range(H + 1)])
                d_arr = np.full(H + 1, d)

                obstacles.append({
                    'agent_id': aid,
                    's': s_arr,
                    'd': d_arr,
                    'length': length,
                    'width': width,
                })

        return obstacles

    def next_action(self, observation: Observation,
                    prediction: TrajectoryPrediction = None) -> Action:
        self._step_count += 1
        timing = {}

        if self._mcts_planner is None or self._frenet is None:
            return Action(acceleration=0.0, steer_angle=0.0,
                          target_speed=self._target_speed)

        ego_state = observation.frame.get(self.agent_id)
        if ego_state is None:
            return Action(acceleration=0.0, steer_angle=0.0,
                          target_speed=self._target_speed)

        # 1. Predict other agent trajectories
        t0 = _time.perf_counter()
        other_agents = {aid: s for aid, s in observation.frame.items()
                        if aid != self.agent_id}
        agent_trajectories = self._predict_agent_trajectories(observation.frame)
        timing['predict'] = _time.perf_counter() - t0

        # 2. Convert to Frenet state and obstacles
        frenet_state = self._state_to_frenet(ego_state)

        s_values = np.array([
            frenet_state[0] + frenet_state[3] *
            np.cos(frenet_state[2]) * k * self._dt
            for k in range(self._horizon + 1)])
        s_values = np.clip(s_values, 0.0, self._frenet.total_length)

        road_left, road_right = sample_road_boundaries(
            s_values, self._scenario_map, self._frenet)

        obstacles = self._predict_obstacles(other_agents, agent_trajectories)

        # 3. Run MCTS
        t0 = _time.perf_counter()
        trajectories, _ = self._mcts_planner.search(
            frenet_state, road_left, road_right, obstacles,
            prev_action=self._prev_action)
        timing['mcts'] = _time.perf_counter() - t0

        # 4. Pick best trajectory, execute first action
        if trajectories:
            best = max(trajectories, key=lambda t: t.mcts_reward)
            # First control from the fine-resolution trajectory
            a = float(best.controls[0, 0])
            delta = float(best.controls[0, 1])  # always 0 in 1-D mode
            self._prev_action = a  # scalar for 1-D planner

            action = Action(
                acceleration=a,
                steer_angle=delta,
                target_speed=self._target_speed,
            )

            logger.info("[Step %3d] MCTS action: a=%.3f  "
                         "reward=%.2f  n_traj=%d",
                         self._step_count, a, best.mcts_reward,
                         len(trajectories))
        else:
            logger.warning("[Step %3d] MCTS: no trajectories — stopping",
                            self._step_count)
            action = Action(acceleration=0.0, steer_angle=0.0,
                            target_speed=0.0)

        # 5. Update plots
        t0 = _time.perf_counter()
        if self._tree_plotter is not None and self._mcts_planner._last_root is not None:
            self._tree_plotter.update(
                self._mcts_planner._last_root,
                road_left, road_right,
                self._step_count)

        if self._traj_plotter is not None and trajectories:
            # Build lightweight result objects for the plotter
            results = []
            for tbp in trajectories:
                planned_sdvv = np.column_stack([
                    tbp.states[:, 0],
                    tbp.states[:, 1],
                    tbp.states[:, 3] * np.cos(tbp.states[:, 2]),
                    tbp.states[:, 3] * np.sin(tbp.states[:, 2]),
                ])
                results.append(InferenceResult(
                    config={},  # belief inference disabled
                    pos_cost=-tbp.mcts_reward,  # use neg reward as cost
                    vel_cost=0.0,
                    planned_sd=planned_sdvv[:, :2],
                    planned_vel=planned_sdvv[:, 2:4],
                    milp_ok=True,
                    nlp_ok=False,
                    nlp_states=tbp.states,
                    nlp_controls=tbp.controls,
                ))
            results.sort(key=lambda r: r.pos_cost)
            self._traj_plotter.update(
                results,
                np.array(ego_state.position),
                float(ego_state.heading),
                self._step_count,
                other_agent_states=other_agents)
        timing['plot'] = _time.perf_counter() - t0

        self.last_step_timing = timing
        return action

    def reset(self):
        super().reset()
        self._vehicle = KinematicVehicle(self._initial_state, self.metadata, self._fps)
        self._step_count = 0
        self._prev_action = None
        self.last_step_timing = {}
