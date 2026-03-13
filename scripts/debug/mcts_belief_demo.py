"""
MCTSBeliefAgent demo running in CARLA.

Drives the ego vehicle using the MCTS trajectory planner directly.
All other vehicles are treated as visible obstacles with hard collision
avoidance constraints.

Run from the repo root:
    python scripts/debug/mcts_belief_demo.py
    python scripts/debug/mcts_belief_demo.py -m belief_agent_demo_mcts
"""

import sys
import os
import logging
import argparse
import json
import time as _time

import carla
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import igp2 as ip
from belief_utils import generate_random_frame, collect_step

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="MCTSBeliefAgent CARLA demo")
    parser.add_argument("--map", "-m", type=str, default="belief_agent_demo_mcts",
                        help="Scenario config name under scenarios/configs/")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--carla_path", "-p", type=str,
                        default="/opt/carla-simulator")
    parser.add_argument("--server", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable live debug plots")
    parser.add_argument("--target_speed", type=float, default=None,
                        help="Override target speed (m/s)")
    parser.add_argument("--n_simulations", type=int, default=5000,
                        help="MCTS simulation count")
    return parser.parse_args()


def create_agent(agent_config, frame, fps, scenario_map, plot=True,
                 target_speed=None, mcts_kwargs=None):
    """Create an agent from config. MCTSBeliefAgent for ego, TrafficAgent for others."""
    aid = agent_config["id"]
    initial_state = frame[aid]
    goal = ip.BoxGoal(ip.Box(**agent_config["goal"]["box"]))

    agent_type = agent_config["type"]

    if agent_type == "MCTSBeliefAgent":
        kwargs = dict(
            agent_id=aid,
            initial_state=initial_state,
            goal=goal,
            fps=fps,
            scenario_map=scenario_map,
            target_speed=target_speed or 10.0,
            plot=plot,
        )
        if mcts_kwargs:
            kwargs.update(mcts_kwargs)
        return ip.MCTSBeliefAgent(**kwargs)

    elif agent_type == "TrafficAgent":
        open_loop = agent_config.get("open_loop", False)
        return ip.TrafficAgent(
            agent_id=aid, initial_state=initial_state,
            goal=goal, fps=fps, open_loop=open_loop)

    elif agent_type == "BeliefAgent":
        # Treat BeliefAgent configs as MCTSBeliefAgent for this demo
        kwargs = dict(
            agent_id=aid,
            initial_state=initial_state,
            goal=goal,
            fps=fps,
            scenario_map=scenario_map,
            target_speed=target_speed or 10.0,
            plot=plot,
        )
        if mcts_kwargs:
            kwargs.update(mcts_kwargs)
        return ip.MCTSBeliefAgent(**kwargs)

    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def main():
    args = parse_args()

    ip.setup_logging(level=logging.INFO)
    np.random.seed(args.seed)
    np.seterr(divide="ignore")

    config_path = os.path.join("scenarios", "configs", f"{args.map}.json")
    with open(config_path) as f:
        config = json.load(f)

    fps = config["scenario"].get("fps", 20)
    ip.Maneuver.MAX_SPEED = config["scenario"].get("max_speed", 10.0)
    target_speed = args.target_speed or config["scenario"].get("max_speed", 10.0)

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

    # MCTS kwargs from CLI
    mcts_kwargs = {
        'n_simulations': args.n_simulations,
    }

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
        if aid == ego_id:
            agents[aid] = create_agent(
                agent_config, frame, fps, scenario_map,
                plot=not args.no_plot,
                target_speed=target_speed,
                mcts_kwargs=mcts_kwargs)
        else:
            agents[aid] = create_agent(
                agent_config, frame, fps, scenario_map,
                plot=False)
        carla_sim.add_agent(agents[aid], "ego" if aid == ego_id else None)

    # Add static objects
    static_objs = config.get("static_objects", [])
    if static_objs:
        carla_sim.spawn_static_objects_from_config(static_objs)

    # Camera
    ego_wrapper = carla_sim.get_ego()
    if ego_wrapper is not None:
        camera_transform = carla.Transform(
            carla.Location(x=-10.0, z=6.0),
            carla.Rotation(pitch=-15.0),
        )
        carla_sim.attach_camera(ego_wrapper.actor, camera_transform)

    # Tell the ego about other agents
    ego_agent = agents.get(ego_id)
    if ego_agent is not None:
        ego_agent.set_agents(agents)

    logger.info("Starting CARLA simulation (%d steps, ego=%d as MCTSBeliefAgent, "
                "target_speed=%.1f, n_sim=%d)",
                args.steps, ego_id, target_speed, args.n_simulations)

    t0 = _time.time()

    for t in range(args.steps):
        t_step = _time.perf_counter()
        obs, acts = carla_sim.step()
        carla_total = _time.perf_counter() - t_step

        current_frame = obs.frame if obs is not None else None

        # Goal check
        if ego_agent is not None and current_frame is not None:
            ego_state = current_frame.get(ego_id)
            if ego_state is not None and ego_agent.goal is not None:
                if ego_agent.goal.reached(ego_state.position):
                    print(f"\n{'='*60}")
                    print(f"  GOAL REACHED at step {t}")
                    print(f"  Position: {ego_state.position}")
                    print(f"{'='*60}\n")
                    break

        # Timing
        if ego_agent is not None and hasattr(ego_agent, 'last_step_timing'):
            st = ego_agent.last_step_timing
            if st:
                parts = "  ".join(f"{k}={v*1000:.0f}ms" for k, v in st.items())
                print(f"[t={t:3d}] total={carla_total*1000:.0f}ms  {parts}")

    else:
        print(f"\nDid not reach goal in {args.steps} steps.")

    logger.info("Done. Total wall time: %.1fs", _time.time() - t0)


if __name__ == "__main__":
    main()
