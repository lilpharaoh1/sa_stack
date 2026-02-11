"""
BeliefAgent experiment runner.

Loads a scenario config, runs the BeliefAgent in CARLA, collects per-step
diagnostics, and saves the results to a pickle file.

Usage:
    python scripts/experiments/belief_agent_experiment.py -m belief_agent_demo_parkedcars_dynamic
    python scripts/experiments/belief_agent_experiment.py -m belief_agent_demo_parkedcars_dynamic -o my_run --seed 42
    python scripts/experiments/belief_agent_experiment.py -m belief_agent_demo_parkedcars_dynamic --steps 300 --no-plot
"""

import sys
import os
import logging
import argparse
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import dill
import carla
import numpy as np
from shapely.geometry import Polygon

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Diagnostics captured at a single simulation step."""
    step: int
    wall_time: float  # seconds since experiment start

    # Ego state
    ego_position: Optional[np.ndarray]
    ego_speed: Optional[float]
    ego_heading: Optional[float]

    # Optimisation outputs
    ego_rollout: Optional[np.ndarray]        # (H+1, 4) [x, y, heading, speed]
    ego_milp_rollout: Optional[np.ndarray]   # (H+1, 2) [x, y]
    nlp_converged: Optional[bool]

    # Obstacle predictions used by the optimisation
    predicted_obstacles: Optional[List]
    other_agent_states: Optional[Dict]

    # Forward-simulated trajectories from BeliefAgent
    predicted_trajectories: Dict[int, np.ndarray]

    # Scene snapshot
    dynamic_agents: Dict[int, Any]   # non-ego, ID >= 0
    static_obstacles: Dict[int, Any] # ID < 0

    # Completion
    goal_reached: bool


@dataclass
class ExperimentResult:
    """Full result of a single experiment run."""
    # Metadata
    scenario_name: str
    config: Dict[str, Any]
    seed: int
    fps: int
    max_steps: int
    start_time: str       # ISO timestamp

    # Outcome
    solved: bool = False
    solved_step: Optional[int] = None
    total_steps: int = 0
    wall_time_seconds: float = 0.0

    # Per-step data
    steps: List[StepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BeliefAgent experiment runner")
    parser.add_argument("--map", "-m", type=str, required=True,
                        help="Scenario config name under scenarios/configs/")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output filename (without extension). "
                             "Defaults to '{map}_{seed}'.")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--steps", type=int, default=500,
                        help="Maximum number of simulation steps")
    parser.add_argument("--carla_path", "-p", type=str,
                        default="/opt/carla-simulator",
                        help="Path to CARLA installation")
    parser.add_argument("--server", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable the BeliefAgent plotter")
    return parser.parse_args()


def generate_random_frame(ego: int,
                          layout: ip.Map,
                          spawn_vel_ranges: List[Tuple[ip.Box, Tuple[float, float]]]
                          ) -> Dict[int, ip.AgentState]:
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


def create_agent(agent_config, frame, fps, scenario_map, plot_interval=1):
    """Create an agent from its config dict."""
    base = {
        "agent_id": agent_config["id"],
        "initial_state": frame[agent_config["id"]],
        "goal": ip.BoxGoal(ip.Box(**agent_config["goal"]["box"])),
        "fps": fps,
    }

    agent_type = agent_config["type"]

    if agent_type == "BeliefAgent":
        return ip.BeliefAgent(**base, scenario_map=scenario_map,
                              plot_interval=plot_interval)
    elif agent_type == "TrafficAgent":
        open_loop = agent_config.get("open_loop", False)
        return ip.TrafficAgent(**base, open_loop=open_loop)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def collect_step(step: int, t0: float, ego_agent, ego_goal, frame) -> StepRecord:
    """Collect diagnostics for one simulation step."""
    ego_id = ego_agent.agent_id
    ego_state = frame.get(ego_id) if frame else None
    policy = getattr(ego_agent, '_policy_obj', None)

    ego_rollout = getattr(policy, 'last_rollout', None) if policy else None
    ego_milp_rollout = getattr(policy, 'last_milp_rollout', None) if policy else None

    nlp_converged = None
    if policy is not None:
        nlp_converged = getattr(policy, '_prev_nlp_states', None) is not None

    predicted_obstacles = getattr(policy, 'last_obstacles', None) if policy else None
    other_agent_states = getattr(policy, 'last_other_agents', None) if policy else None
    predicted_trajectories = dict(ego_agent._agent_trajectories)

    dynamic_agents = {aid: s for aid, s in frame.items()
                      if aid != ego_id and aid >= 0} if frame else {}
    static_obstacles = {aid: s for aid, s in frame.items()
                        if aid < 0} if frame else {}

    # Goal reached: check if any point in the NLP rollout is at the goal
    goal_reached = False
    if ego_goal is not None and ego_rollout is not None:
        for pt in ego_rollout[:, :2]:
            if ego_goal.reached(pt):
                goal_reached = True
                break

    return StepRecord(
        step=step,
        wall_time=time.time() - t0,
        ego_position=np.array(ego_state.position) if ego_state else None,
        ego_speed=float(ego_state.speed) if ego_state else None,
        ego_heading=float(ego_state.heading) if ego_state else None,
        ego_rollout=ego_rollout,
        ego_milp_rollout=ego_milp_rollout,
        nlp_converged=nlp_converged,
        predicted_obstacles=predicted_obstacles,
        other_agent_states=other_agent_states,
        predicted_trajectories=predicted_trajectories,
        dynamic_agents=dynamic_agents,
        static_obstacles=static_obstacles,
        goal_reached=goal_reached,
    )


def dump_results(result: ExperimentResult, name: str):
    """Save experiment results to pickle."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, name + ".pkl")
    with open(filepath, 'wb') as f:
        dill.dump(result, f)
    logger.info("Results saved to %s", filepath)
    return filepath


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

    plot_interval = 0 if args.no_plot else 1
    agents = {}
    for agent_config in config["agents"]:
        aid = agent_config["id"]
        agents[aid] = create_agent(agent_config, frame, fps, scenario_map,
                                   plot_interval=plot_interval)
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

    ego_agent = agents.get(ego_id)
    ego_goal = ego_agent.goal if ego_agent is not None else None

    # Tell the ego agent about the other agents
    if ego_agent is not None:
        ego_agent.set_agents(agents)

    # Prepare result object
    result = ExperimentResult(
        scenario_name=args.map,
        config=config,
        seed=args.seed,
        fps=fps,
        max_steps=args.steps,
        start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    output_name = args.output if args.output else f"{args.map}_{args.seed}"

    print(f"\n{'='*60}")
    print(f"  Experiment: {args.map}")
    print(f"  Seed: {args.seed}  |  FPS: {fps}  |  Max steps: {args.steps}")
    print(f"  Output: {output_name}.pkl")
    print(f"{'='*60}\n")

    t0 = time.time()

    for t in range(args.steps):
        obs, acts = carla_sim.step()
        current_frame = obs.frame if obs is not None else None

        # Collect diagnostics
        if ego_agent is not None and current_frame is not None:
            record = collect_step(t, t0, ego_agent, ego_goal, current_frame)
            result.steps.append(record)
            result.total_steps = t + 1

            if record.goal_reached:
                result.solved = True
                result.solved_step = t
                result.wall_time_seconds = time.time() - t0

                print(f"\n{'='*60}")
                print(f"  SCENARIO SOLVED at step {t}")
                print(f"  Ego position: {record.ego_position}")
                print(f"  Goal: {ego_goal}")
                print(f"  Wall time: {result.wall_time_seconds:.1f}s")
                print(f"{'='*60}\n")

                # Remove all agents from CARLA
                for aid in list(carla_sim.agents.keys()):
                    if carla_sim.agents[aid] is not None:
                        carla_sim.remove_agent(aid)
                break
    else:
        result.wall_time_seconds = time.time() - t0
        print(f"\nScenario NOT solved within {args.steps} steps "
              f"({result.wall_time_seconds:.1f}s).")

    # Save results
    filepath = dump_results(result, output_name)
    print(f"\nResults saved: {filepath}")
    print(f"  solved={result.solved}  steps={result.total_steps}  "
          f"time={result.wall_time_seconds:.1f}s")

    logger.info("Done.")


if __name__ == "__main__":
    main()
