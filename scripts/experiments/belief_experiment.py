"""
Batch runner for BeliefAgent experiments.

Runs N samples of a scenario config (new or old format), collects results,
and prints summary statistics.

Usage:
    python scripts/experiments/belief_experiment.py -m belief_experiment1 -n 10
    python scripts/experiments/belief_experiment.py -m belief_experiment1 -n 100 --seed 42 --steps 300
"""

import sys
import os
import logging
import argparse
import json
import time
from typing import List
from collections import Counter

import dill
import carla
import numpy as np

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip

from ind_belief_experiment import (
    ExperimentResult,
    is_new_format,
    sample_viable_config,
    expand_new_config,
    generate_random_frame,
    run_single_experiment,
    check_viability,
    dump_results,
    plot_spawn_preview,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch BeliefAgent experiment runner")
    parser.add_argument("--map", "-m", type=str, required=True,
                        help="Scenario config name under scenarios/configs/")
    parser.add_argument("-n", "--n-samples", type=int, default=100,
                        help="Number of samples to run (default: 100)")
    parser.add_argument("--seed", type=int, default=21,
                        help="Base seed; sample i uses seed + i")
    parser.add_argument("--steps", type=int, default=500,
                        help="Maximum number of simulation steps per episode")
    parser.add_argument("--carla_path", "-p", type=str,
                        default="/opt/carla-simulator",
                        help="Path to CARLA installation")
    parser.add_argument("--server", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--preview", action="store_true",
                        help="Show spawn preview plot for the first sample before running")
    return parser.parse_args()


def print_summary(scenario_name: str,
                  results: List[ExperimentResult],
                  n_viable: int,
                  n_nonviable: int,
                  save_path: str):
    """Print a formatted summary of batch experiment results."""
    n_total = n_viable + n_nonviable

    solved = [r for r in results if r.solved]
    failed = [r for r in results if r.failed]
    timed_out = [r for r in results if not r.solved and not r.failed]

    n_solved = len(solved)
    n_failed = len(failed)
    n_timed = len(timed_out)

    def pct(count, total):
        return f"{100 * count / total:.1f}" if total > 0 else "0.0"

    print(f"\n{'='*60}")
    print(f"  Batch Experiment Summary: {scenario_name}")
    print(f"  Samples: {n_total} attempted, {n_viable} viable, {n_nonviable} non-viable")
    print(f"{'='*60}")
    print(f"  Solved:    {n_solved:4d} ({pct(n_solved, n_viable)}%)")
    print(f"  Failed:    {n_failed:4d} ({pct(n_failed, n_viable)}%)")
    print(f"  Timed out: {n_timed:4d} ({pct(n_timed, n_viable)}%)")

    if failed:
        print(f"\n  Failure breakdown:")
        reason_counts = Counter()
        for r in failed:
            if r.failure_reason:
                for part in r.failure_reason.split("; "):
                    reason_counts[part] += 1
            else:
                reason_counts["NLP infeasible (unknown cause)"] += 1
        for reason, count in reason_counts.most_common():
            print(f"    {reason}: {count}")

    if solved:
        steps_arr = np.array([r.solved_step for r in solved])
        print(f"\n  Steps to solve ({n_solved} solved):")
        print(f"    min={steps_arr.min()}  max={steps_arr.max()}  "
              f"mean={steps_arr.mean():.1f}  std={steps_arr.std():.1f}")

        times_arr = np.array([r.wall_time_seconds for r in solved])
        print(f"\n  Wall time per episode ({n_solved} solved):")
        print(f"    min={times_arr.min():.1f}s  max={times_arr.max():.1f}s  "
              f"mean={times_arr.mean():.1f}s  std={times_arr.std():.1f}s")

        # Ego speed at goal (last step's ego speed for solved runs)
        speeds = []
        for r in solved:
            if r.steps and r.steps[-1].ego_speed is not None:
                speeds.append(r.steps[-1].ego_speed)
        if speeds:
            sp = np.array(speeds)
            print(f"\n  Ego speed at goal ({len(speeds)} solved):")
            print(f"    min={sp.min():.1f}  max={sp.max():.1f}  "
                  f"mean={sp.mean():.1f}  std={sp.std():.1f} m/s")

    print(f"\n  Results saved: {save_path}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    ip.setup_logging(level=logging.INFO)
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

    new_fmt = is_new_format(config)
    plot_interval = config["scenario"].get("plot_interval", False)

    # Connect to CARLA once
    carla_sim = ip.carlasim.CarlaSim(
        map_name=map_name,
        xodr=scenario_xodr,
        carla_path=args.carla_path,
        server=args.server,
        port=args.port,
        fps=fps,
    )

    results: List[ExperimentResult] = []
    n_viable = 0
    n_nonviable = 0

    print(f"\n{'='*60}")
    print(f"  Batch Experiment: {args.map}")
    print(f"  Samples: {args.n_samples}  |  Base seed: {args.seed}")
    print(f"  FPS: {fps}  |  Max steps: {args.steps}")
    print(f"{'='*60}\n")

    batch_t0 = time.time()

    next_seed = args.seed

    for i in range(args.n_samples):
        # Keep trying seeds until we get a viable configuration
        while True:
            sample_seed = next_seed
            next_seed += 1
            np.random.seed(sample_seed)
            rng = np.random.RandomState(sample_seed)

            if new_fmt:
                try:
                    expanded, frame = sample_viable_config(
                        config, scenario_map, seed=sample_seed)
                    break
                except RuntimeError:
                    n_nonviable += 1
                    print(f"  [seed={sample_seed}] non-viable, retrying...")
            else:
                expanded = config
                ego_id = config["agents"][0]["id"]
                agent_spawns = []
                for ac in config["agents"]:
                    spawn_box = ip.Box(
                        np.array(ac["spawn"]["box"]["center"]),
                        ac["spawn"]["box"]["length"],
                        ac["spawn"]["box"]["width"],
                        ac["spawn"]["box"]["heading"],
                    )
                    vel_range = ac["spawn"]["velocity"]
                    agent_spawns.append((spawn_box, vel_range))
                frame = generate_random_frame(ego_id, scenario_map, agent_spawns, rng=rng)

                if check_viability(frame, expanded.get("static_objects", [])):
                    break
                n_nonviable += 1
                print(f"  [seed={sample_seed}] non-viable, retrying...")

        n_viable += 1

        n_agents = len(expanded["agents"])
        print(f"\n[Sample {i+1:4d}/{args.n_samples}] seed={sample_seed}  "
              f"agents={n_agents}")

        if args.preview:
            plot_spawn_preview(scenario_map, expanded, frame,
                               title=f"Sample {i+1}/{args.n_samples}: {args.map} (seed={sample_seed})",
                               raw_config=config)

        result = run_single_experiment(
            config=expanded,
            frame=frame,
            scenario_map=scenario_map,
            carla_sim=carla_sim,
            max_steps=args.steps,
            fps=fps,
            plot_interval=plot_interval,
            seed=sample_seed,
            scenario_name=args.map,
        )
        results.append(result)

        status = "SOLVED" if result.solved else ("FAILED" if result.failed else "TIMEOUT")
        print(f"  >> {status}  steps={result.total_steps}  "
              f"time={result.wall_time_seconds:.1f}s")

        # Clean up CARLA for next sample
        for aid in list(carla_sim.agents.keys()):
            if carla_sim.agents[aid] is not None:
                carla_sim.remove_agent(aid)
        carla_sim.clear_static_objects()

    batch_time = time.time() - batch_t0

    # Save all results to a single pickle
    save_name = f"{args.map}_batch_{args.n_samples}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, save_name + ".pkl")
    with open(save_path, 'wb') as f:
        dill.dump({
            "results": results,
            "n_viable": n_viable,
            "n_nonviable": n_nonviable,
            "args": vars(args),
            "batch_wall_time": batch_time,
        }, f)

    print_summary(args.map, results, n_viable, n_nonviable, save_path)
    print(f"  Total batch time: {batch_time:.1f}s")


if __name__ == "__main__":
    main()
