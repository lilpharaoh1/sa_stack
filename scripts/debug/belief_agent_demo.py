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

import carla
import numpy as np
from shapely.geometry import Polygon
from typing import List, Tuple, Dict

# Ensure repo root is on the path so igp2 is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip

logger = logging.getLogger(__name__)


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
                               horizon: int = 40, dt: float = 0.2) -> Dict[int, np.ndarray]:
    """Extract planned trajectories from TrafficAgents for collision avoidance.

    Extracts the planned waypoints from each TrafficAgent's current macro action
    and returns them as (T, 2) arrays of world positions.

    Args:
        agents: Dict mapping agent_id -> Agent instance.
        ego_id: ID of the ego agent (excluded from output).
        frame: Current observation frame with agent states.
        horizon: Number of planning steps (T).
        dt: Planning timestep in seconds.

    Returns:
        Dict mapping agent_id -> (T+1, 2) array of planned positions.
    """
    trajectories = {}

    for aid, agent in agents.items():
        if aid == ego_id:
            continue

        # Only TrafficAgents have macro_actions with planned trajectories
        if not hasattr(agent, 'macro_actions') or not agent.macro_actions:
            continue

        # Collect all waypoints from current and future macro actions
        all_waypoints = []
        for macro in agent.macro_actions:
            for maneuver in macro._maneuvers:
                if hasattr(maneuver, 'trajectory') and maneuver.trajectory is not None:
                    all_waypoints.append(maneuver.trajectory.path)

        if not all_waypoints:
            continue

        # Concatenate all waypoints
        combined = np.vstack(all_waypoints)

        # Get current position from frame
        if aid in frame:
            current_pos = np.array(frame[aid].position)
        else:
            # Fallback: use first waypoint
            current_pos = combined[0]

        # Find closest waypoint index
        dists = np.linalg.norm(combined - current_pos, axis=1)
        closest_idx = int(np.argmin(dists))

        # Extract trajectory from closest point onwards
        future_traj = combined[closest_idx:]

        # Resample to match planning timesteps if needed
        # For now, just use the waypoints directly (assuming similar spacing)
        trajectories[aid] = future_traj

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
        return ip.TrafficAgent(**base)
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

    # Get planning parameters from ego agent's policy
    ego_agent = agents.get(ego_id)
    planning_horizon = 40  # default
    planning_dt = 0.2      # default
    if ego_agent is not None and hasattr(ego_agent, '_policy_obj'):
        policy = ego_agent._policy_obj
        if hasattr(policy, '_horizon'):
            planning_horizon = policy._horizon
        if hasattr(policy, '_dt'):
            planning_dt = policy._dt

    # Get initial observation to have a starting frame
    current_frame = frame  # Use initial frame for first step

    for t in range(args.steps):
        # Extract planned trajectories from other agents and pass to ego
        if ego_agent is not None and hasattr(ego_agent, 'set_agent_trajectories'):
            trajectories = extract_agent_trajectories(
                agents, ego_id, current_frame, planning_horizon, planning_dt)
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
