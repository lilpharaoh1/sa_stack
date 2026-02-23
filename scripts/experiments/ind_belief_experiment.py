"""
BeliefAgent experiment runner.

Loads a scenario config, runs the BeliefAgent in CARLA, collects per-step
diagnostics, and saves the results to a pickle file.

Usage:
    python scripts/experiments/ind_belief_experiment.py -m belief_agent_demo_parkedcars_dynamic
    python scripts/experiments/ind_belief_experiment.py -m belief_agent_demo_parkedcars_dynamic -o my_run --seed 42
    python scripts/experiments/ind_belief_experiment.py -m belief_agent_demo_parkedcars_dynamic --steps 300 --no-plot
"""

import sys
import os
import logging
import argparse
import json
import time

import carla
import numpy as np

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip

from belief_utils import (
    ExperimentResult,
    StepRecord,
    RESULTS_DIR,
    generate_random_frame,
    create_agent,
    collect_step,
    dump_results,
    plot_spawn_preview,
    is_new_format,
    expand_new_config,
    expand_static_groups,
    check_viability,
    sample_viable_config,
    print_scene_summary,
    make_run_dir,
    build_run_metadata,
    save_experiment,
    build_summary,
    save_summary,
)

logger = logging.getLogger(__name__)


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
    parser.add_argument("--intervention-type", type=str, default="none",
                        choices=["none", "agency_only", "combined", "warmstart_only"],
                        help="Intervention scheme for the ego agent (default: none)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(config: dict,
                          frame,
                          scenario_map: 'ip.Map',
                          carla_sim: 'ip.carlasim.CarlaSim',
                          max_steps: int,
                          fps: int,
                          plot_interval: bool = True,
                          seed: int = 21,
                          scenario_name: str = "experiment",
                          intervention_type: str = "none",
                          ) -> ExperimentResult:
    """Run a single experiment episode.

    Creates agents, steps the simulation, collects diagnostics, and returns
    an :class:`ExperimentResult`.  Does **not** clean up CARLA -- the caller
    is responsible for removing agents and static objects afterwards.
    """
    ego_id = config["agents"][0]["id"]

    # Inject intervention_type into the ego agent config so create_agent picks it up
    config["agents"][0]["intervention_type"] = intervention_type

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
        scenario_name=scenario_name,
        config=config,
        seed=seed,
        fps=fps,
        max_steps=max_steps,
        start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    print_scene_summary(config, frame)

    t0 = time.time()
    prev_true_trajectories = None

    for t in range(max_steps):
        t_step_start = time.perf_counter()
        obs, acts = carla_sim.step()
        carla_step_total = time.perf_counter() - t_step_start

        current_frame = obs.frame if obs is not None else None

        # Print per-step timing breakdown
        if ego_agent is not None and hasattr(ego_agent, 'last_step_timing'):
            st = ego_agent.last_step_timing
            if st:
                step_num = getattr(ego_agent, '_step_count', t)
                agent_total = sum(st.values())
                carla_overhead = carla_step_total - agent_total
                parts = "  ".join(f"{k}={v*1000:.1f}ms" for k, v in st.items())
                print(f"[Step {step_num:4d}] total={carla_step_total*1000:.0f}ms  "
                      f"carla_overhead={carla_overhead*1000:.0f}ms  {parts}")

        # Collect diagnostics
        if ego_agent is not None and current_frame is not None:
            record = collect_step(t, t0, ego_agent, ego_goal, current_frame,
                                  prev_true_trajectories=prev_true_trajectories)
            prev_true_trajectories = dict(ego_agent._true_agent_trajectories)
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
                break

            # Stop if true policy optimisation failed
            if record.true_diag_nlp_ok is not None and not record.true_diag_nlp_ok:
                result.failed = True
                result.failure_step = t
                result.wall_time_seconds = time.time() - t0

                # Build failure reason -- one entry per violated constraint type
                reasons = []
                if record.true_diag_collision_violations > 0:
                    reasons.append("collision avoidance infeasible")
                if record.true_diag_road_violations > 0:
                    reasons.append("road boundary infeasible")
                if record.true_diag_velocity_violated:
                    reasons.append("velocity bounds infeasible")
                if record.true_diag_acceleration_violated:
                    reasons.append("acceleration bounds infeasible")
                if record.true_diag_steering_violated:
                    reasons.append("steering bounds infeasible")
                if record.true_diag_jerk_violated:
                    reasons.append("jerk limits infeasible")
                if record.true_diag_steer_rate_violated:
                    reasons.append("steering rate infeasible")
                result.failure_reason = "; ".join(reasons) if reasons else "NLP infeasible (unknown cause)"

                print(f"\n{'='*60}")
                print(f"  TRUE POLICY FAILED at step {t}")
                print(f"  Reason: {result.failure_reason}")
                print(f"  Ego position: {record.ego_position}")
                print(f"  Wall time: {result.wall_time_seconds:.1f}s")
                print(f"{'='*60}\n")
                break
    else:
        result.wall_time_seconds = time.time() - t0
        print(f"\nScenario NOT solved within {max_steps} steps "
              f"({result.wall_time_seconds:.1f}s).")

    # Close any matplotlib figures opened by agent plotters
    import matplotlib.pyplot as plt
    plt.close('all')

    return result


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
    map_name = config["scenario"].get("map_name", "Town01")

    rng = np.random.RandomState(args.seed)

    if is_new_format(config):
        # New format: expand dynamic groups and sample viable positions
        expanded, frame = sample_viable_config(
            config, scenario_map, seed=args.seed)
    else:
        # Old format: build spawn info and random initial frame
        expanded = config
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
        frame = generate_random_frame(ego_id, scenario_map, agent_spawns, rng=rng)

    plot_interval = False if args.no_plot else config["scenario"].get("plot_interval", True)

    # Show spawn preview before connecting to CARLA
    if plot_interval:
        plot_spawn_preview(scenario_map, expanded, frame,
                           title=f"Spawn Preview: {args.map}",
                           raw_config=config)

    # Create CARLA simulation
    carla_sim = ip.carlasim.CarlaSim(
        map_name=map_name,
        xodr=scenario_xodr,
        carla_path=args.carla_path,
        server=args.server,
        port=args.port,
        fps=fps,
    )

    result = run_single_experiment(
        config=expanded,
        frame=frame,
        scenario_map=scenario_map,
        carla_sim=carla_sim,
        max_steps=args.steps,
        fps=fps,
        plot_interval=plot_interval,
        seed=args.seed,
        scenario_name=args.map,
        intervention_type=args.intervention_type,
    )

    run_dir = make_run_dir(
        scenario_name=args.map,
        intervention_type=args.intervention_type,
        seed=args.seed,
        custom_name=args.output,
    )
    metadata = build_run_metadata(args, expanded)
    save_experiment(result, run_dir, metadata)

    summary = build_summary(result)
    save_summary(summary, run_dir)

    print(f"\n{'='*60}")
    print(f"  Experiment: {args.map}")
    print(f"  Seed: {args.seed}  |  FPS: {fps}  |  Max steps: {args.steps}")
    print(f"  Run directory: {run_dir}")
    print(f"{'='*60}\n")

    print(f"  solved={result.solved}  failed={result.failed}  "
          f"steps={result.total_steps}  time={result.wall_time_seconds:.1f}s")
    if result.failure_reason:
        print(f"  failure_reason: {result.failure_reason}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
