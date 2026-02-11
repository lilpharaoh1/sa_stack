"""
BeliefAgent demo running in CARLA.

Uses the belief_agent_demo scenario config with one BeliefAgent (ego) and
TrafficAgents.  The BeliefAgent currently acts identically to a TrafficAgent;
override update_beliefs() and policy() to add belief-conditioned behaviour.

Run from the repo root:
    python scripts/debug/belief_agent_demo.py
    python scripts/debug/belief_agent_demo.py -m belief_agent_demo
    python scripts/debug/belief_agent_demo.py --carla_path /opt/carla-simulator
"""

import sys
import os
import logging
import argparse
import json
import time
import copy

import carla
import numpy as np
from shapely.geometry import Polygon
from typing import List, Tuple, Dict, Optional

# Ensure repo root is on the path so igp2 is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip
from igp2.core.agentstate import AgentState
from igp2.core.vehicle import Observation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bicycle Model Forward Simulation
# ---------------------------------------------------------------------------

def bicycle_step(position: np.ndarray, heading: float, speed: float,
                 accel: float, steer: float, dt: float, wheelbase: float) -> Tuple[np.ndarray, float, float]:
    """Single step of the kinematic bicycle model (matches CARLA's _bicycle_step).

    Equations:
        x_{k+1}     = x_k + v_k * cos(theta_k + delta_k) * dt
        y_{k+1}     = y_k + v_k * sin(theta_k + delta_k) * dt
        theta_{k+1} = theta_k + (2*v_k/L) * sin(delta_k) * dt
        v_{k+1}     = v_k + a_k * dt

    Args:
        position: [x, y] current position.
        heading: Current heading in radians.
        speed: Current speed (m/s).
        accel: Acceleration command (m/s^2).
        steer: Steering angle command (radians).
        dt: Timestep (seconds).
        wheelbase: Vehicle wheelbase (metres).

    Returns:
        (new_position, new_heading, new_speed)
    """
    x, y = position
    x_new = x + speed * np.cos(heading + steer) * dt
    y_new = y + speed * np.sin(heading + steer) * dt
    theta_new = heading + (2.0 * speed / wheelbase) * np.sin(steer) * dt
    v_new = max(0.0, speed + accel * dt)

    # Normalise heading to [-pi, pi]
    theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

    return np.array([x_new, y_new]), theta_new, v_new


class StandaloneWaypointTracker:
    """Standalone PID waypoint tracker that doesn't modify agent state.

    Replicates the behavior of WaypointManeuver's PID control for forward simulation.
    """

    # PID gains matching WaypointManeuver defaults
    LATERAL_KP = 1.95
    LATERAL_KI = 0.2
    LATERAL_KD = 0.0
    LONGITUDINAL_KP = 1.0
    LONGITUDINAL_KI = 0.05
    LONGITUDINAL_KD = 0.0

    def __init__(self, waypoints: np.ndarray, target_speed: float, dt: float):
        """
        Args:
            waypoints: (N, 2) array of waypoints to track.
            target_speed: Target speed (m/s).
            dt: Timestep (seconds).
        """
        self.waypoints = waypoints
        self.target_speed = target_speed
        self.dt = dt
        self.waypoint_idx = 0
        self.waypoint_margin = max(1.0, dt * 20.0)  # Scale with dt so look-ahead stays ~4 frames

        # PID state
        self.lat_integral = 0.0
        self.lat_prev_error = 0.0
        self.lon_integral = 0.0
        self.lon_prev_error = 0.0

    def get_action(self, position: np.ndarray, heading: float, speed: float,
                   max_steer: float = 0.5, max_accel: float = 3.0) -> Tuple[float, float]:
        """Compute control action to track waypoints.

        Args:
            position: Current [x, y] position.
            heading: Current heading (radians).
            speed: Current speed (m/s).
            max_steer: Maximum steering angle magnitude.
            max_accel: Maximum acceleration magnitude.

        Returns:
            (acceleration, steer_angle)
        """
        if self.waypoint_idx >= len(self.waypoints):
            return 0.0, 0.0

        # Advance waypoint if close enough
        while self.waypoint_idx < len(self.waypoints) - 1:
            dist_to_wp = np.linalg.norm(self.waypoints[self.waypoint_idx] - position)
            if dist_to_wp < self.waypoint_margin:
                self.waypoint_idx += 1
            else:
                break

        target = self.waypoints[self.waypoint_idx]

        # Compute errors
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        target_heading = np.arctan2(dy, dx)

        # Lateral error (heading difference)
        lat_error = target_heading - heading
        # Normalize to [-pi, pi]
        lat_error = (lat_error + np.pi) % (2 * np.pi) - np.pi

        # Longitudinal error (speed difference)
        lon_error = self.target_speed - speed

        # PID for lateral (steering)
        self.lat_integral += lat_error * self.dt
        lat_derivative = (lat_error - self.lat_prev_error) / self.dt if self.dt > 0 else 0.0
        steer = (self.LATERAL_KP * lat_error +
                 self.LATERAL_KI * self.lat_integral +
                 self.LATERAL_KD * lat_derivative)
        steer = np.clip(steer, -max_steer, max_steer)
        self.lat_prev_error = lat_error

        # PID for longitudinal (acceleration)
        self.lon_integral += lon_error * self.dt
        lon_derivative = (lon_error - self.lon_prev_error) / self.dt if self.dt > 0 else 0.0
        accel = (self.LONGITUDINAL_KP * lon_error +
                 self.LONGITUDINAL_KI * self.lon_integral +
                 self.LONGITUDINAL_KD * lon_derivative)
        accel = np.clip(accel, -max_accel, max_accel)
        self.lon_prev_error = lon_error

        return accel, steer


def get_agent_waypoints_and_speeds(agent) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract all future waypoints and velocities from an agent's macro actions.

    Args:
        agent: Agent with macro_actions attribute.

    Returns:
        Tuple of:
            - (N, 2) array of waypoints, or None if not available.
            - (N,) array of speeds at each waypoint, or None if not available.
    """
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
                # Get velocity magnitude at each point
                if traj.velocity is not None and len(traj.velocity) == len(traj.path):
                    # velocity can be scalar (speed) or 2D vector
                    vel = np.asarray(traj.velocity)
                    if vel.ndim == 1:
                        all_speeds.append(vel)
                    else:
                        all_speeds.append(np.linalg.norm(vel, axis=1))
                else:
                    # No velocity data - use None placeholder
                    all_speeds.append(None)

    if not all_waypoints:
        return None, None

    waypoints = np.vstack(all_waypoints)

    # Combine speeds if available
    if all(s is not None for s in all_speeds):
        speeds = np.concatenate(all_speeds)
    else:
        speeds = None

    return waypoints, speeds


def simulate_agent_trajectory(agent, frame: Dict, scenario_map,
                              n_steps: int, dt: float) -> Optional[np.ndarray]:
    """Forward simulate an agent's trajectory using bicycle model + PID tracking.

    This function simulates what the agent will actually do by:
    1. Extracting the agent's planned waypoints and target speeds
    2. Creating a standalone PID tracker (doesn't modify agent state)
    3. Forward propagating through the exact bicycle model CARLA uses

    This ensures the predicted trajectory matches what CARLA will execute.

    Args:
        agent: The agent to simulate (must have macro_actions with trajectories).
        frame: Current observation frame with all agent states.
        scenario_map: The scenario map (unused, kept for API compatibility).
        n_steps: Number of simulation steps to predict.
        dt: Simulation timestep (1/fps).

    Returns:
        (n_steps+1, 2) array of predicted positions, or None if simulation fails.
    """
    if not hasattr(agent, 'agent_id'):
        return None

    aid = agent.agent_id
    if aid not in frame:
        return None

    # Get all waypoints and speeds from the agent's plan
    waypoints, speeds = get_agent_waypoints_and_speeds(agent)
    if waypoints is None or len(waypoints) < 2:
        return None

    # Get agent metadata
    meta = agent.metadata if hasattr(agent, 'metadata') else None
    if meta is None:
        wheelbase = 2.5
        max_steer = 0.5
        max_accel = 3.0
    else:
        wheelbase = meta.wheelbase
        max_steer = getattr(meta, 'max_steer', 0.5)
        max_accel = getattr(meta, 'max_acceleration', 3.0)

    # Get initial state
    state = frame[aid]
    position = np.array(state.position)
    heading = float(state.heading)
    speed = float(state.speed)

    # Find closest waypoint to current position
    dists = np.linalg.norm(waypoints - position, axis=1)
    closest_idx = int(np.argmin(dists))
    future_waypoints = waypoints[closest_idx:]
    future_speeds = speeds[closest_idx:] if speeds is not None else None

    if len(future_waypoints) < 2:
        return None

    # Determine target speed
    if future_speeds is not None and len(future_speeds) > 0:
        # Use average of trajectory speeds
        target_speed = float(np.mean(future_speeds))
    elif speed > 0.5:
        # Use current speed
        target_speed = speed
    else:
        # Default
        target_speed = 5.0

    # Create standalone tracker
    tracker = StandaloneWaypointTracker(future_waypoints, target_speed, dt)

    # Simulate forward
    trajectory = np.zeros((n_steps + 1, 2))
    trajectory[0] = position

    for k in range(n_steps):
        # Get control action from PID tracker
        accel, steer = tracker.get_action(position, heading, speed, max_steer, max_accel)

        # Step the bicycle model (exact same equations as CARLA's _bicycle_step)
        position, heading, speed = bicycle_step(
            position, heading, speed, accel, steer, dt, wheelbase)
        trajectory[k + 1] = position

    return trajectory


def extract_agent_trajectories_simulated(agents: Dict, ego_id: int, frame: Dict,
                                         scenario_map, fps: int,
                                         horizon_time: float = 10.0) -> Dict[int, np.ndarray]:
    """Extract agent trajectories by forward simulating their controllers through bicycle model.

    This provides accurate predictions of where agents will be by simulating:
    1. The agent's control policy (maneuver-based waypoint tracking)
    2. The exact bicycle model dynamics used by CARLA

    Args:
        agents: Dict mapping agent_id -> Agent instance.
        ego_id: ID of the ego agent (excluded from output).
        frame: Current observation frame with agent states.
        scenario_map: The scenario map for observations.
        fps: Simulation framerate.
        horizon_time: How far ahead to simulate (seconds).

    Returns:
        Dict mapping agent_id -> (N, 2) array of predicted positions at simulation timestep.
    """
    dt = 1.0 / fps
    n_steps = int(horizon_time / dt)
    trajectories = {}

    for aid, agent in agents.items():
        if aid == ego_id:
            continue

        traj = simulate_agent_trajectory(agent, frame, scenario_map, n_steps, dt)
        if traj is not None:
            trajectories[aid] = traj

    return trajectories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BeliefAgent CARLA demo")
    parser.add_argument("--map", "-m", type=str, default="belief_agent_demo",
                        help="Scenario config name under scenarios/configs/")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of simulation steps")
    parser.add_argument("--carla_path", "-p", type=str,
                        default="/opt/carla-simulator",
                        help="Path to CARLA installation")
    parser.add_argument("--server", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    return parser.parse_args()


def generate_random_frame(ego: int,
                          layout: ip.Map,
                          spawn_vel_ranges: List[Tuple[ip.Box, Tuple[float, float]]]) -> Dict[int, ip.AgentState]:
    """Generate a new frame with randomised spawns and velocities."""
    ret = {}
    for i, (spawn, vel) in enumerate(spawn_vel_ranges, ego):
        poly = Polygon(spawn.boundary)
        best_lane = layout.best_lane_at(spawn.center, max_distance=500.0)

        intersections = list(best_lane.midline.intersection(poly).coords)
        start_d = best_lane.distance_at(intersections[0])
        end_d = best_lane.distance_at(intersections[1])
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        position_d = (end_d - start_d) * np.random.random() + start_d
        spawn_position = np.array(best_lane.point_at(position_d))

        speed = (vel[1] - vel[0]) * np.random.random() + vel[0]
        heading = best_lane.get_heading_at(position_d)
        ret[i] = ip.AgentState(time=0,
                               position=spawn_position,
                               velocity=speed * np.array([np.cos(heading), np.sin(heading)]),
                               acceleration=np.array([0.0, 0.0]),
                               heading=heading)
    return ret


def extract_agent_trajectories(agents: Dict, ego_id: int, frame: Dict,
                               fps: int = 20) -> Dict[int, np.ndarray]:
    """Extract planned trajectories from TrafficAgents for collision avoidance.

    Extracts the planned waypoints from each TrafficAgent's current macro action
    and resamples them to the simulation timestep (1/fps).

    IMPORTANT: The returned trajectories are sampled at the SIMULATION timestep
    (1/fps seconds apart), NOT the planning timestep. The policy's _predict_obstacles
    will resample these to the planning timestep.

    Args:
        agents: Dict mapping agent_id -> Agent instance.
        ego_id: ID of the ego agent (excluded from output).
        frame: Current observation frame with agent states.
        fps: Simulation framerate (trajectories sampled at 1/fps seconds).

    Returns:
        Dict mapping agent_id -> (N, 2) array of planned positions at simulation timestep.
    """
    trajectories = {}
    dt_sim = 1.0 / fps

    for aid, agent in agents.items():
        if aid == ego_id:
            continue

        # Only TrafficAgents have macro_actions with planned trajectories
        if not hasattr(agent, 'macro_actions') or not agent.macro_actions:
            continue

        # Collect all waypoints and times from current and future macro actions
        all_waypoints = []
        all_times = []
        cumulative_time = 0.0

        for macro in agent.macro_actions:
            for maneuver in macro._maneuvers:
                if hasattr(maneuver, 'trajectory') and maneuver.trajectory is not None:
                    traj = maneuver.trajectory
                    path = traj.path
                    times = traj.times

                    if times is not None and len(times) == len(path):
                        # Offset times by cumulative time from previous maneuvers
                        adjusted_times = times + cumulative_time
                        all_waypoints.append(path)
                        all_times.append(adjusted_times)
                        cumulative_time = adjusted_times[-1]
                    else:
                        # Fallback: estimate times from velocity if available
                        velocities = traj.velocity if traj.velocity is not None else None
                        if velocities is not None and len(velocities) == len(path):
                            # Compute time from distance and velocity
                            dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
                            avg_vels = (velocities[:-1] + velocities[1:]) / 2.0
                            avg_vels = np.maximum(avg_vels, 0.1)  # Avoid division by zero
                            dt_segments = dists / avg_vels
                            times_local = np.concatenate([[0], np.cumsum(dt_segments)])
                            adjusted_times = times_local + cumulative_time
                            all_waypoints.append(path)
                            all_times.append(adjusted_times)
                            cumulative_time = adjusted_times[-1]
                        else:
                            # Last resort: use arc length with constant velocity estimate
                            if aid in frame:
                                speed = np.linalg.norm(frame[aid].velocity)
                                speed = max(speed, 1.0)  # Minimum speed for estimation
                            else:
                                speed = 5.0  # Default speed
                            dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
                            dt_segments = dists / speed
                            times_local = np.concatenate([[0], np.cumsum(dt_segments)])
                            adjusted_times = times_local + cumulative_time
                            all_waypoints.append(path)
                            all_times.append(adjusted_times)
                            cumulative_time = adjusted_times[-1]

        if not all_waypoints:
            continue

        # Concatenate all waypoints and times
        combined_path = np.vstack(all_waypoints)
        combined_times = np.concatenate(all_times)

        # Get current position and find where we are in the trajectory
        if aid in frame:
            current_pos = np.array(frame[aid].position)
        else:
            current_pos = combined_path[0]

        # Find closest waypoint index
        dists = np.linalg.norm(combined_path - current_pos, axis=1)
        closest_idx = int(np.argmin(dists))

        # Extract future trajectory (from current position onwards)
        future_path = combined_path[closest_idx:]
        future_times = combined_times[closest_idx:] - combined_times[closest_idx]  # Reset time to 0

        if len(future_path) < 2:
            continue

        # Resample to simulation timestep using linear interpolation
        max_time = future_times[-1]
        n_samples = max(int(max_time / dt_sim) + 1, 2)
        sample_times = np.arange(n_samples) * dt_sim

        # Interpolate x and y separately
        resampled_path = np.empty((n_samples, 2))
        resampled_path[:, 0] = np.interp(sample_times, future_times, future_path[:, 0])
        resampled_path[:, 1] = np.interp(sample_times, future_times, future_path[:, 1])

        trajectories[aid] = resampled_path

    return trajectories


def create_agent(agent_config, frame, fps, scenario_map):
    """Create an agent from its config dict."""
    base = {
        "agent_id": agent_config["id"],
        "initial_state": frame[agent_config["id"]],
        "goal": ip.BoxGoal(ip.Box(**agent_config["goal"]["box"])),
        "fps": fps,
    }

    agent_type = agent_config["type"]

    if agent_type == "BeliefAgent":
        return ip.BeliefAgent(**base, scenario_map=scenario_map)
    elif agent_type == "TrafficAgent":
        open_loop = agent_config.get("open_loop", False)
        return ip.TrafficAgent(**base, open_loop=open_loop)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    ip.setup_logging(level=logging.INFO)
    np.random.seed(args.seed)
    np.seterr(divide="ignore")

    # Load scenario config
    config_path = os.path.join("scenarios", "configs", f"{args.map}.json")
    with open(config_path) as f:
        config = json.load(f)

    fps = config["scenario"].get("fps", 20)
    print("\n\n\n\n\nUsing fps:", fps)
    ip.Maneuver.MAX_SPEED = config["scenario"].get("max_speed", 10.0)

    scenario_xodr = config["scenario"]["map_path"]
    scenario_map = ip.Map.parse_from_opendrive(scenario_xodr)

    # Build spawn info and random initial frame
    ego_id = config["agents"][0]["id"]
    agent_spawns = []
    for agent_config in config["agents"]:
        spawn_box = ip.Box(
            np.array(agent_config["spawn"]["box"]["center"]),
            agent_config["spawn"]["box"]["length"],
            agent_config["spawn"]["box"]["width"],
            agent_config["spawn"]["box"]["heading"],
        )
        vel_range = agent_config["spawn"]["velocity"]
        agent_spawns.append((spawn_box, vel_range))

    frame = generate_random_frame(ego_id, scenario_map, agent_spawns)

    # Create CARLA simulation
    carla_sim = ip.carlasim.CarlaSim(
        map_name="Town01",
        xodr=scenario_xodr,
        carla_path=args.carla_path,
        server=args.server,
        port=args.port,
        fps=fps,
    )

    agents = {}
    for agent_config in config["agents"]:
        aid = agent_config["id"]
        agents[aid] = create_agent(agent_config, frame, fps, scenario_map)
        carla_sim.add_agent(agents[aid], "ego" if aid == ego_id else None)

    # Add static objects from config
    static_objs = config.get("static_objects", [])
    if static_objs:
        carla_sim.spawn_static_objects_from_config(static_objs)

    # Set up camera to follow the ego vehicle
    ego_wrapper = carla_sim.get_ego()
    if ego_wrapper is not None:
        camera_transform = carla.Transform(
            carla.Location(x=-10.0, z=6.0),
            carla.Rotation(pitch=-15.0),
        )
        carla_sim.attach_camera(ego_wrapper.actor, camera_transform)
        logger.info("Camera set to follow ego vehicle (Agent %d)", ego_id)

    logger.info("Starting CARLA simulation (%d steps, ego=%d as BeliefAgent)",
                args.steps, ego_id)

    ego_agent = agents.get(ego_id)

    # Get initial observation to have a starting frame
    current_frame = frame  # Use initial frame for first step

    # Compute horizon time from ego's policy (if available)
    horizon_time = 10.0  # default
    if ego_agent is not None and hasattr(ego_agent, '_policy_obj'):
        policy = ego_agent._policy_obj
        if hasattr(policy, '_horizon') and hasattr(policy, '_dt'):
            horizon_time = policy._horizon * policy._dt + 1.0  # Add buffer

    for t in range(args.steps):
        # Extract planned trajectories by forward simulating other agents
        # through the bicycle model to get exact predicted positions
        if ego_agent is not None and hasattr(ego_agent, 'set_agent_trajectories'):
            trajectories = extract_agent_trajectories_simulated(
                agents, ego_id, current_frame, scenario_map, fps, horizon_time)
            ego_agent.set_agent_trajectories(trajectories)

        obs, acts = carla_sim.step()

        # Update frame for next iteration
        if obs is not None:
            current_frame = obs.frame

        if ego_agent is not None and hasattr(ego_agent, "beliefs") and ego_agent.beliefs:
            if t % 20 == 0:
                logger.info("t=%d  beliefs=%s", t, ego_agent.beliefs)

        # time.sleep(0.05)

    logger.info("Done.")


if __name__ == "__main__":
    main()
