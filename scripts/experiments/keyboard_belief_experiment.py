"""
Interactive keyboard-controlled BeliefAgent experiment runner.

Like ind_belief_experiment.py but replaces the simulated human driver
(NLP-based TwoStagePolicy) with keyboard control (WASD/arrows via pygame).
The full inference + intervention pipeline stays intact.

Usage:
    python scripts/experiments/keyboard_belief_experiment.py \
        -m belief_agent_demo_mcts \
        --no-plot --live-speed \
        --intervention-type agency_only \
        --inference-type mcts_kalman
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
from igp2.agents.keyboard_belief_agent import KeyboardBeliefAgent

from belief_utils import (
    ExperimentResult,
    StepRecord,
    RESULTS_DIR,
    generate_random_frame,
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
    parser = argparse.ArgumentParser(
        description="Interactive keyboard-controlled BeliefAgent experiment")
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
    parser.add_argument("--preview", action="store_true",
                        help="Show spawn preview plot before running")
    parser.add_argument("--intervention-type", type=str, default="none",
                        choices=["none", "agency_only", "combined",
                                 "policy_only", "mcts", "always_policy"],
                        help="Intervention scheme for the ego agent (default: none)")
    parser.add_argument("--ref-controls", type=str, default="opt",
                        choices=["opt", "mcts-greedy", "mcts-qcbf"],
                        help="Source of reference controls for intervention (default: opt)")
    parser.add_argument("--inference-type", type=str, default="naive",
                        choices=["none", "naive", "mcts_naive",
                                 "mcts_resample", "mcts_kalman"],
                        help="Belief inference strategy (default: naive)")
    parser.add_argument("--relevance-method", type=str, default="naive",
                        choices=["corridor", "dual", "naive"],
                        help="Relevance detection method (default: naive)")
    parser.add_argument("--planning-mode", type=str, default="2d",
                        choices=["2d", "longitudinal"],
                        help="Planning mode (default: 2d)")
    parser.add_argument("--human-type", type=str, default="static",
                        choices=["static", "kalman"],
                        help="Human belief dynamics (default: static)")
    parser.add_argument("--live-awareness", action="store_true",
                        help="Show live awareness kernel plot during episode")
    parser.add_argument("--live-speed", action="store_true",
                        help="Show live reference vs. intervention speed plot")
    return parser.parse_args()


def create_keyboard_agent(agent_config, frame, fps, scenario_map,
                          plot_interval=True):
    """Create a KeyboardBeliefAgent from the ego agent config."""
    base = {
        "agent_id": agent_config["id"],
        "initial_state": frame[agent_config["id"]],
        "goal": ip.BoxGoal(ip.Box(**agent_config["goal"]["box"])),
        "fps": fps,
    }
    return KeyboardBeliefAgent(
        **base,
        scenario_map=scenario_map,
        plot_interval=plot_interval,
        agent_beliefs=agent_config.get("beliefs", None),
        human=agent_config.get("human", True),
        intervention_type=agent_config.get("intervention_type", "none"),
        inference_type=agent_config.get("inference_type", "naive"),
        relevance_method=agent_config.get("relevance_method", "dual"),
        planning_mode=agent_config.get("planning_mode", "2d"),
        ref_controls=agent_config.get("ref_controls", "opt"),
        human_type=agent_config.get("human_type", "static"),
    )


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
                          inference_type: str = "naive",
                          relevance_method: str = "dual",
                          planning_mode: str = "2d",
                          ref_controls: str = "opt",
                          human_type: str = "static",
                          live_awareness: bool = False,
                          live_speed: bool = False,
                          ) -> ExperimentResult:
    """Run a single keyboard-controlled experiment episode."""
    ego_id = config["agents"][0]["id"]

    # Inject settings into the ego agent config
    config["agents"][0]["intervention_type"] = intervention_type
    config["agents"][0]["inference_type"] = inference_type
    config["agents"][0]["relevance_method"] = relevance_method
    config["agents"][0]["planning_mode"] = planning_mode
    config["agents"][0]["ref_controls"] = ref_controls
    config["agents"][0]["human_type"] = human_type

    agents = {}
    for agent_config in config["agents"]:
        aid = agent_config["id"]
        if aid == ego_id:
            # Ego: use KeyboardBeliefAgent
            agents[aid] = create_keyboard_agent(
                agent_config, frame, fps, scenario_map,
                plot_interval=plot_interval)
        else:
            # Other agents: TrafficAgent
            from belief_utils import create_agent
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

    # Live awareness plotter
    awareness_plotter = None
    if live_awareness and ego_agent is not None:
        from live_awareness_plotter import LiveAwarenessPlotter
        awareness_plotter = LiveAwarenessPlotter(scenario_map, ego_agent)

    # Live speed plotter
    speed_plotter = None
    if live_speed:
        from live_speed_plotter import LiveSpeedPlotter
        speed_plotter = LiveSpeedPlotter()

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
    prev_action = None

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
                                  prev_true_trajectories=prev_true_trajectories,
                                  prev_action=prev_action)
            prev_true_trajectories = dict(ego_agent._true_agent_trajectories)
            if record.ego_acceleration is not None:
                prev_action = (record.ego_acceleration, record.ego_steer_angle)
            result.steps.append(record)
            result.total_steps = t + 1

            if awareness_plotter is not None:
                awareness_plotter.update(record)

            if speed_plotter is not None:
                speed_plotter.update(record)

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

            # Stop if ego collided
            if record.ego_collision:
                result.failed = True
                result.failure_step = t
                result.wall_time_seconds = time.time() - t0
                result.failure_reason = f"ego collision with {record.ego_collision_id}"

                print(f"\n{'='*60}")
                print(f"  EGO COLLISION at step {t}")
                print(f"  Collided with: {record.ego_collision_id}")
                print(f"  Ego position: {record.ego_position}")
                print(f"  Wall time: {result.wall_time_seconds:.1f}s")
                print(f"{'='*60}\n")
                break

            # NOTE: Human/true policy NLP/MILP failures are NOT fatal here
            # because the keyboard provides the actual control — the NLP
            # only runs for infrastructure (FrenetFrame, diagnostics).
            # Log warnings but keep going.
            if record.human_diag_milp_ok is not None and not record.human_diag_milp_ok:
                logger.warning("Step %d: human policy MILP infeasible (non-fatal, keyboard driving)", t)

            if record.human_diag_nlp_ok is not None and not record.human_diag_nlp_ok:
                logger.warning("Step %d: human policy NLP infeasible (non-fatal, keyboard driving)", t)

            if record.true_diag_milp_ok is not None and not record.true_diag_milp_ok:
                logger.warning("Step %d: true policy MILP infeasible (non-fatal)", t)

            if record.true_diag_nlp_ok is not None and not record.true_diag_nlp_ok:
                logger.warning("Step %d: true policy NLP infeasible (non-fatal)", t)

    else:
        result.wall_time_seconds = time.time() - t0
        print(f"\nScenario NOT solved within {max_steps} steps "
              f"({result.wall_time_seconds:.1f}s).")

    # Close matplotlib figures (except live plotters)
    import matplotlib.pyplot as plt
    keep_labels = {'live_awareness', 'live_speed'}
    for fig_num in list(plt.get_fignums()):
        fig = plt.figure(fig_num)
        if fig.get_label() in keep_labels:
            continue
        plt.close(fig)

    if awareness_plotter is not None:
        awareness_plotter.close()
    if speed_plotter is not None:
        speed_plotter.close()

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
        expanded, frame = sample_viable_config(
            config, scenario_map, seed=args.seed)
    else:
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

    if args.preview:
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
        inference_type=args.inference_type,
        relevance_method=args.relevance_method,
        planning_mode=args.planning_mode,
        ref_controls=args.ref_controls,
        human_type=args.human_type,
        live_awareness=args.live_awareness,
        live_speed=args.live_speed,
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
