"""
Simple example: ego (SimpleBeliefAgent) follows a lead (TrafficAgent).

The ego uses a basic proportional controller to maintain a target following
distance.  The ego has a velocity-error belief about the lead vehicle,
meaning it perceives the lead's speed as biased.  Inference and intervention
modules are stubs for now.

Run from the repo root:
    python scripts/debug/simple_example.py
    python scripts/debug/simple_example.py -m simpleexample
    python scripts/debug/simple_example.py --carla_path /opt/carla-simulator
"""

import sys
import os
import logging
import argparse
import json
import time as _time

import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip
from igp2.agents.simple_belief_agent import SimpleBeliefAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live probability plotter
# ---------------------------------------------------------------------------

class BeliefPlotter:
    """Live bar chart showing the velocity-error posterior each step."""

    def __init__(self, candidates: np.ndarray, true_vel_err: float = None):
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(8, 3))
        self._candidates = candidates
        self._true_vel_err = true_vel_err

        self._bars = self._ax.bar(candidates, np.ones(len(candidates)) / len(candidates),
                                  width=0.08, color="steelblue", edgecolor="white")
        if true_vel_err is not None:
            self._ax.axvline(true_vel_err, color="red", linestyle="--",
                             linewidth=1.5, label=f"true = {true_vel_err:.1f}")
            self._ax.legend(fontsize=9)

        self._ax.set_xlabel("velocity error")
        self._ax.set_ylabel("P(vel_err)")
        self._ax.set_title("Inferred velocity-error belief")
        self._ax.set_xlim(candidates[0] - 0.1, candidates[-1] + 0.1)
        self._ax.set_ylim(0, 1.0)
        self._fig.tight_layout()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def update(self, probabilities: np.ndarray, step: int = None):
        for bar, p in zip(self._bars, probabilities):
            bar.set_height(p)
        self._ax.set_ylim(0, max(0.2, probabilities.max() * 1.2))
        if step is not None:
            self._ax.set_title(f"Inferred velocity-error belief  (t={step})")
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
# Config parsing  (handles the ego + dynamic_groups format directly)
# ---------------------------------------------------------------------------

def parse_config(config: dict, scenario_map: ip.Map, fps: int, rng=None):
    """Parse a simpleexample-style config into agents and initial frame.

    Returns:
        agents: dict mapping agent_id -> Agent
        frame:  dict mapping agent_id -> AgentState
    """
    if rng is None:
        rng = np.random.RandomState()

    ego_cfg = config["ego"]
    groups = config.get("dynamic_groups", [])

    # --- Build spawn list and agent configs ---
    agent_configs = []  # list of (id, cfg_dict)

    # Ego is always id 0
    agent_configs.append((0, ego_cfg))

    # Each dynamic group entry is one agent (no count sampling for this
    # simple example)
    next_id = 1
    beliefs = {}
    for group in groups:
        agent_cfg = {
            "type": group.get("type", "TrafficAgent"),
            "spawn": group["spawn"],
            "goal": group["goal"],
            "open_loop": group.get("open_loop", False),
        }
        agent_configs.append((next_id, agent_cfg))
        if "belief" in group:
            beliefs[str(next_id)] = group["belief"]
        next_id += 1

    # --- Sample initial frame ---
    spawn_vel_ranges = []
    for aid, cfg in agent_configs:
        sb = cfg["spawn"]["box"]
        spawn_box = ip.Box(
            np.array(sb["center"]),
            sb["length"], sb["width"], sb.get("heading", 0.0),
        )
        vel_range = cfg["spawn"]["velocity"]
        spawn_vel_ranges.append((spawn_box, vel_range))

    frame = _generate_frame(0, scenario_map, spawn_vel_ranges, rng)

    # --- Create agents ---
    agents = {}
    for aid, cfg in agent_configs:
        initial_state = frame[aid]
        goal = ip.BoxGoal(ip.Box(**cfg["goal"]["box"]))
        agent_type = cfg.get("type", "TrafficAgent")

        if agent_type == "SimpleBeliefAgent":
            agents[aid] = SimpleBeliefAgent(
                agent_id=aid,
                initial_state=initial_state,
                goal=goal,
                fps=fps,
                scenario_map=scenario_map,
                agent_beliefs=beliefs,
                target_distance=cfg.get("target_distance", 20.0),
            )
        elif agent_type == "TrafficAgent":
            agents[aid] = ip.TrafficAgent(
                agent_id=aid,
                initial_state=initial_state,
                goal=goal,
                fps=fps,
                open_loop=cfg.get("open_loop", False),
            )
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

    return agents, frame


def _generate_frame(ego_id, layout, spawn_vel_ranges, rng):
    """Sample initial positions along lanes inside spawn boxes."""
    from shapely.geometry import Polygon

    frame = {}
    for i, (spawn, vel) in enumerate(spawn_vel_ranges, ego_id):
        poly = Polygon(spawn.boundary)
        best_lane = layout.best_lane_at(spawn.center, max_distance=500.0)
        intersections = list(best_lane.midline.intersection(poly).coords)
        start_d = best_lane.distance_at(intersections[0])
        end_d = best_lane.distance_at(intersections[1])
        if start_d > end_d:
            start_d, end_d = end_d, start_d
        position_d = (end_d - start_d) * rng.random() + start_d
        spawn_position = np.array(best_lane.point_at(position_d))

        speed = (vel[1] - vel[0]) * rng.random() + vel[0]
        heading = best_lane.get_heading_at(position_d)
        frame[i] = ip.AgentState(
            time=0,
            position=spawn_position,
            velocity=speed * np.array([np.cos(heading), np.sin(heading)]),
            acceleration=np.array([0.0, 0.0]),
            heading=heading,
        )
    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Simple belief-agent example")
    parser.add_argument("--map", "-m", type=str, default="simpleexample",
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
    max_steps = config["scenario"].get("max_steps", args.steps)
    ip.Maneuver.MAX_SPEED = config["scenario"].get("max_speed", 10.0)

    scenario_xodr = config["scenario"]["map_path"]
    scenario_map = ip.Map.parse_from_opendrive(scenario_xodr)

    rng = np.random.RandomState(args.seed)
    agents, frame = parse_config(config, scenario_map, fps, rng)

    ego_id = 0
    ego_agent = agents[ego_id]

    # Print scene summary
    print(f"\n{'='*60}")
    print(f"  Simple Example  (fps={fps}, max_steps={max_steps})")
    print(f"{'='*60}")
    for aid, state in frame.items():
        agent = agents[aid]
        role = "EGO" if aid == ego_id else "   "
        print(f"  {role} Agent {aid} ({type(agent).__name__})  "
              f"pos=({state.position[0]:.1f}, {state.position[1]:.1f})  "
              f"v={state.speed:.1f} m/s")
    if isinstance(ego_agent, SimpleBeliefAgent):
        print(f"  Target distance: {ego_agent.target_distance:.1f} m")
        print(f"  Velocity-error beliefs: "
              f"{ego_agent.agent_beliefs}")
    print(f"{'='*60}\n")

    # --- CARLA setup ---
    import carla

    map_name = config["scenario"].get("map_name", "Town01")
    carla_sim = ip.carlasim.CarlaSim(
        map_name=map_name,
        xodr=scenario_xodr,
        carla_path=args.carla_path,
        server=args.server,
        port=args.port,
        fps=fps,
    )

    for aid, agent in agents.items():
        carla_sim.add_agent(agent, "ego" if aid == ego_id else None)

    # Spawn static objects if any
    static_objs = config.get("static_objects", [])
    if static_objs:
        carla_sim.spawn_static_objects_from_config(static_objs)

    # Camera following ego
    ego_wrapper = carla_sim.get_ego()
    if ego_wrapper is not None:
        camera_transform = carla.Transform(
            carla.Location(x=-10.0, z=6.0),
            carla.Rotation(pitch=-15.0),
        )
        carla_sim.attach_camera(ego_wrapper.actor, camera_transform)

    # Tell ego about other agents
    if hasattr(ego_agent, "set_agents"):
        ego_agent.set_agents(agents)

    # Set up live belief plotter
    belief_plotter = None
    if isinstance(ego_agent, SimpleBeliefAgent):
        for aid, belief in ego_agent.inferred_belief.velocity_errors.items():
            true_ve = ego_agent._human_vel_errors.get(aid)
            belief_plotter = BeliefPlotter(belief.candidates, true_vel_err=true_ve)
            break  # only one lead vehicle for now

    dt = 1.0 / fps  # real-time step duration

    logger.info("Starting simulation (%d steps, ego=%d)", max_steps, ego_id)
    t0 = _time.time()

    # --- Main loop ---
    for t in range(max_steps):
        t0_step = _time.time()
        obs, acts = carla_sim.step()

        if obs is None:
            continue

        current_frame = obs.frame
        ego_state = current_frame.get(ego_id)

        # Print step diagnostics
        if isinstance(ego_agent, SimpleBeliefAgent) and ego_agent.last_step_info:
            info = ego_agent.last_step_info
            dist_str = (f"{info['distance']:.1f}" if info.get("distance")
                        else "N/A")
            inferred_mode = info.get('inferred_mode')
            inferred_mean = info.get('inferred_mean')
            inf_str = (f"mode={inferred_mode:.2f} mean={inferred_mean:.2f}"
                       if inferred_mode is not None else "N/A")
            print(f"[t={t:4d}]  dist={dist_str:>6s}m  "
                  f"v_perceived={info.get('v_perceived', 0) or 0:.2f}  "
                  f"v_desired={info.get('v_desired', 0) or 0:.2f}  "
                  f"accel={info.get('accel', 0):.2f}  "
                  f"inferred=[{inf_str}]")
            inferred_dist = info.get('inferred_dist')
            if inferred_dist is not None:
                probs_str = "  ".join(f"{k:+.1f}:{v:.4f}"
                                      for k, v in inferred_dist.items())
                print(f"         P(vel_err): {probs_str}")

            # Update live plot
            if belief_plotter is not None and inferred_dist is not None:
                probs = np.array(list(inferred_dist.values()))
                belief_plotter.update(probs, step=t)

        # Real-time pacing
        elapsed_step = _time.time() - t0_step
        sleep_time = dt - elapsed_step
        if sleep_time > 0:
            _time.sleep(sleep_time)

        # Check goal reached
        if ego_state is not None and ego_agent.goal is not None:
            if ego_agent.goal.reached(ego_state.position):
                print(f"\n{'='*60}")
                print(f"  GOAL REACHED at step {t}")
                print(f"  Position: ({ego_state.position[0]:.1f}, "
                      f"{ego_state.position[1]:.1f})")
                print(f"{'='*60}\n")
                for aid in list(carla_sim.agents.keys()):
                    if carla_sim.agents[aid] is not None:
                        carla_sim.remove_agent(aid)
                break
    else:
        elapsed = _time.time() - t0
        print(f"\nScenario NOT solved within {max_steps} steps "
              f"({elapsed:.1f}s wall time).")

    logger.info("Done.")


if __name__ == "__main__":
    main()
