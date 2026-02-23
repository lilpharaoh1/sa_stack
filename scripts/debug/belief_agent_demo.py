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

import time as _time

import carla
import numpy as np

# Ensure repo root is on the path so igp2 is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Also add experiments dir for belief_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import igp2 as ip
from belief_utils import generate_random_frame, create_agent, collect_step

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
        agents[aid] = create_agent(agent_config, frame, fps, scenario_map,
                                   plot_interval=True)
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

    t0 = _time.time()

    for t in range(args.steps):
        t_step_start = _time.perf_counter()
        obs, acts = carla_sim.step()
        carla_step_total = _time.perf_counter() - t_step_start

        current_frame = obs.frame if obs is not None else None

        # Collect diagnostics using shared utility
        if ego_agent is not None and current_frame is not None:
            record = collect_step(t, t0, ego_agent, ego_goal, current_frame)

            if record.goal_reached:
                print(f"\n{'='*60}")
                print(f"  SCENARIO SOLVED at step {t}")
                print(f"  Ego position: {record.ego_position}")
                print(f"  Goal: {ego_goal}")
                print(f"{'='*60}\n")

                # Remove all agents from CARLA
                for aid in list(carla_sim.agents.keys()):
                    if carla_sim.agents[aid] is not None:
                        carla_sim.remove_agent(aid)
                break

        # Print per-step timing breakdown
        if ego_agent is not None and hasattr(ego_agent, 'last_step_timing'):
            st = ego_agent.last_step_timing
            if st:
                agent_total = sum(st.values())
                carla_overhead = carla_step_total - agent_total
                parts = "  ".join(f"{k}={v*1000:.1f}ms" for k, v in st.items())
                print(f"[t={t:3d}] total={carla_step_total*1000:.0f}ms  "
                      f"carla_overhead={carla_overhead*1000:.0f}ms  {parts}")

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
