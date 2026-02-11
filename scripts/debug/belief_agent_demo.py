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


class StepDiagnostics:
    """Collects per-timestep diagnostic data from the BeliefAgent and scene.

    Stores information that will later be used for analysis / belief-conditioned
    planning.  For now only the goal-reached check is acted upon; everything
    else is silently recorded.
    """

    def __init__(self, ego_agent, ego_goal):
        self.ego_agent = ego_agent
        self.ego_goal = ego_goal
        # Accumulated history (one entry per step)
        self.history: List[Dict] = []

    def check(self, step: int, frame: Dict) -> Dict:
        """Run all per-step checks and return the diagnostics dict.

        Args:
            step: Current simulation step index.
            frame: Current observation frame with all agent states.

        Returns:
            Dict with the diagnostics for this step. The same dict is also
            appended to ``self.history``.
        """
        ego_id = self.ego_agent.agent_id
        ego_state = frame.get(ego_id) if frame else None

        human_policy = getattr(self.ego_agent, '_human_policy', None)
        true_policy = getattr(self.ego_agent, '_true_policy', None)

        # --- Human (belief) policy outputs ---
        human_rollout = getattr(human_policy, 'last_rollout', None) if human_policy else None
        human_milp_rollout = getattr(human_policy, 'last_milp_rollout', None) if human_policy else None
        human_nlp_converged = None
        if human_policy is not None:
            human_nlp_converged = getattr(human_policy, '_prev_nlp_states', None) is not None
        human_obstacles = getattr(human_policy, 'last_obstacles', None) if human_policy else None
        human_other_agents = getattr(human_policy, 'last_other_agents', None) if human_policy else None

        # --- True (ground-truth) policy outputs ---
        true_rollout = getattr(true_policy, 'last_rollout', None) if true_policy else None
        true_milp_rollout = getattr(true_policy, 'last_milp_rollout', None) if true_policy else None
        true_nlp_converged = None
        if true_policy is not None:
            true_nlp_converged = getattr(true_policy, '_prev_nlp_states', None) is not None
        true_obstacles = getattr(true_policy, 'last_obstacles', None) if true_policy else None
        true_other_agents = getattr(true_policy, 'last_other_agents', None) if true_policy else None

        # --- Predicted trajectories ---
        human_trajectories = dict(self.ego_agent._human_agent_trajectories)
        true_trajectories = dict(self.ego_agent._true_agent_trajectories)

        # --- Dynamic agent states (from frame, excluding ego and static) ---
        dynamic_agents = {aid: s for aid, s in frame.items()
                          if aid != ego_id and aid >= 0} if frame else {}

        # --- Static obstacles (negative IDs in frame) ---
        static_obstacles = {aid: s for aid, s in frame.items()
                            if aid < 0} if frame else {}

        # --- Goal reached? Based on TRUE policy rollout ---
        goal_reached = False
        if self.ego_goal is not None and true_rollout is not None:
            for pt in true_rollout[:, :2]:
                if self.ego_goal.reached(pt):
                    goal_reached = True
                    break

        record = {
            "step": step,
            "ego_position": np.array(ego_state.position) if ego_state else None,
            "ego_speed": float(ego_state.speed) if ego_state else None,
            "ego_heading": float(ego_state.heading) if ego_state else None,
            # Human (belief) policy
            "human_rollout": human_rollout,
            "human_milp_rollout": human_milp_rollout,
            "human_nlp_converged": human_nlp_converged,
            "human_obstacles": human_obstacles,
            "human_other_agents": human_other_agents,
            "human_trajectories": human_trajectories,
            # True (ground-truth) policy
            "true_rollout": true_rollout,
            "true_milp_rollout": true_milp_rollout,
            "true_nlp_converged": true_nlp_converged,
            "true_obstacles": true_obstacles,
            "true_other_agents": true_other_agents,
            "true_trajectories": true_trajectories,
            # Scene
            "dynamic_agents": dynamic_agents,
            "static_obstacles": static_obstacles,
            "goal_reached": goal_reached,
        }
        self.history.append(record)
        return record


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
        agent_beliefs = agent_config.get("beliefs", None)
        logger.info("create_agent: BeliefAgent beliefs from config = %s", agent_beliefs)
        return ip.BeliefAgent(**base, scenario_map=scenario_map,
                              agent_beliefs=agent_beliefs)
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
    ego_goal = ego_agent.goal if ego_agent is not None else None

    # Tell the ego agent about the other agents so it can predict their trajectories
    if ego_agent is not None:
        ego_agent.set_agents(agents)

    # Per-step monitoring
    diagnostics = StepDiagnostics(ego_agent, ego_goal) if ego_agent is not None else None

    for t in range(args.steps):
        obs, acts = carla_sim.step()

        current_frame = obs.frame if obs is not None else None

        # Run diagnostics
        if diagnostics is not None and current_frame is not None:
            record = diagnostics.check(t, current_frame)

            if record["goal_reached"]:
                print(f"\n{'='*60}")
                print(f"  SCENARIO SOLVED at step {t}")
                print(f"  Ego position: {record['ego_position']}")
                print(f"  Goal: {ego_goal}")
                print(f"{'='*60}\n")

                # Remove all agents from CARLA
                for aid in list(carla_sim.agents.keys()):
                    if carla_sim.agents[aid] is not None:
                        carla_sim.remove_agent(aid)
                break

        if ego_agent is not None and hasattr(ego_agent, "agent_beliefs") and ego_agent.agent_beliefs:
            if t % 20 == 0:
                logger.info("t=%d  beliefs=%s", t,
                            {aid: (b.visible, b.velocity_error)
                             for aid, b in ego_agent.agent_beliefs.items()})

    else:
        # Loop completed without reaching goal
        print(f"\nScenario NOT solved within {args.steps} steps.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
