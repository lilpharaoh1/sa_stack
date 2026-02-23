"""
Inspect saved experiment results.

Loads a .pkl file produced by ind_belief_experiment.py (single run) or
belief_experiment.py (batch run) and prints a summary of its contents.

Usage:
    python scripts/experiments/inspect_results.py results/belief_experiment1_batch_10.pkl
    python scripts/experiments/inspect_results.py results/belief_scenario1_21.pkl
"""

import sys
import os
import json
import argparse
from collections import Counter

import dill
import numpy as np

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from belief_utils import ExperimentResult, StepRecord


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect experiment results")
    parser.add_argument("path", type=str,
                        help="Path to .pkl results file or run directory")
    parser.add_argument("--steps", action="store_true",
                        help="Print per-step details for each episode")
    return parser.parse_args()


def print_episode(result: ExperimentResult, idx: int = None, show_steps: bool = False):
    """Print summary for a single experiment episode."""
    prefix = f"  Episode {idx}" if idx is not None else "  Episode"
    status = "SOLVED" if result.solved else ("FAILED" if result.failed else "TIMEOUT")

    print(f"{prefix}: {status}")
    print(f"    Scenario: {result.scenario_name}  |  Seed: {result.seed}")
    print(f"    FPS: {result.fps}  |  Steps: {result.total_steps}/{result.max_steps}")
    print(f"    Wall time: {result.wall_time_seconds:.1f}s")
    if result.solved:
        print(f"    Solved at step: {result.solved_step}")
    if result.failed:
        print(f"    Failed at step: {result.failure_step}")
        print(f"    Reason: {result.failure_reason}")

    # Agents in config
    agents = result.config.get("agents", [])
    print(f"    Agents: {len(agents)}")
    static = result.config.get("static_objects", [])
    print(f"    Static objects: {len(static)}")

    if not result.steps:
        print(f"    (no step records)")
        print()
        return

    # Timing summary
    timings = [s.ego_timing for s in result.steps if s.ego_timing]
    if timings:
        all_keys = set()
        for t in timings:
            all_keys.update(t.keys())

        print(f"    Timing ({len(timings)} steps with data):")
        # Total step time
        totals = [sum(t.values()) for t in timings]
        arr = np.array(totals) * 1000
        print(f"      total:          mean={arr.mean():.1f}ms  "
              f"std={arr.std():.1f}ms  min={arr.min():.1f}ms  max={arr.max():.1f}ms")

        for key in sorted(all_keys):
            vals = [t.get(key, 0.0) for t in timings]
            arr = np.array(vals) * 1000
            print(f"      {key + ':':<18s}mean={arr.mean():.1f}ms  "
                  f"std={arr.std():.1f}ms  min={arr.min():.1f}ms  max={arr.max():.1f}ms")

    # Prediction error summary
    pred_errors = [s.prediction_error for s in result.steps
                   if s.prediction_error is not None]
    if pred_errors:
        arr = np.array(pred_errors)
        print(f"    Prediction error ({len(pred_errors)} steps):")
        print(f"      mean={arr.mean():.3f}m  std={arr.std():.3f}m  "
              f"min={arr.min():.3f}m  max={arr.max():.3f}m")

    # NLP convergence
    nlp_flags = [s.true_diag_nlp_ok for s in result.steps
                 if s.true_diag_nlp_ok is not None]
    if nlp_flags:
        n_ok = sum(nlp_flags)
        print(f"    NLP convergence: {n_ok}/{len(nlp_flags)} "
              f"({100*n_ok/len(nlp_flags):.1f}%)")

    # Constraint violations
    road_viols = sum(s.true_diag_road_violations for s in result.steps)
    coll_viols = sum(s.true_diag_collision_violations for s in result.steps)
    if road_viols or coll_viols:
        print(f"    Constraint violations: road={road_viols}  collision={coll_viols}")

    # Intervention stats
    n_interv = sum(1 for s in result.steps if s.intervention_active)
    if n_interv > 0:
        print(f"    Intervention active: {n_interv}/{len(result.steps)} steps "
              f"({100*n_interv/len(result.steps):.1f}%)")
        dev_vals = [s.action_deviation for s in result.steps
                    if s.intervention_active and s.action_deviation is not None]
        if dev_vals:
            dev = np.array(dev_vals)
            print(f"    Action deviation (intervention steps only, n={len(dev_vals)}):")
            print(f"      mean={dev.mean():.4f}  std={dev.std():.4f}  "
                  f"min={dev.min():.4f}  max={dev.max():.4f}")

    # Ego speed at final step
    final = result.steps[-1]
    if final.ego_speed is not None:
        print(f"    Final ego speed: {final.ego_speed:.1f} m/s")
    if final.ego_position is not None:
        print(f"    Final ego position: ({final.ego_position[0]:.1f}, "
              f"{final.ego_position[1]:.1f})")

    if show_steps:
        print(f"    {'Step':>6s}  {'Time':>7s}  {'Speed':>6s}  "
              f"{'NLP':>4s}  {'PredErr':>8s}  {'Timing':s}")
        print(f"    {'-'*70}")
        for s in result.steps:
            spd = f"{s.ego_speed:.1f}" if s.ego_speed is not None else "?"
            nlp = "OK" if s.true_diag_nlp_ok else ("FAIL" if s.true_diag_nlp_ok is False else "?")
            pe = f"{s.prediction_error:.3f}" if s.prediction_error is not None else "-"
            if s.ego_timing:
                total_ms = sum(s.ego_timing.values()) * 1000
                timing_str = f"{total_ms:.0f}ms"
            else:
                timing_str = "-"
            print(f"    {s.step:6d}  {s.wall_time:7.2f}s  {spd:>6s}  "
                  f"{nlp:>4s}  {pe:>8s}  {timing_str}")

    print()


def inspect_batch(data: dict, show_steps: bool = False):
    """Print summary for a batch experiment."""
    results = data["results"]
    args = data.get("args", {})
    n_viable = data.get("n_viable", len(results))
    n_nonviable = data.get("n_nonviable", 0)
    batch_time = data.get("batch_wall_time", 0.0)

    print(f"{'='*70}")
    print(f"  Batch Results")
    print(f"  File contains {len(results)} episodes")
    print(f"  Viable: {n_viable}  |  Non-viable: {n_nonviable}")
    print(f"  Batch wall time: {batch_time:.1f}s")
    if args:
        print(f"  Args: map={args.get('map', '?')}  n_samples={args.get('n_samples', '?')}  "
              f"seed={args.get('seed', '?')}  steps={args.get('steps', '?')}")
    print(f"{'='*70}")
    print()

    if not results:
        print("  No results to inspect.")
        return

    # Aggregate outcomes
    solved = [r for r in results if r.solved]
    failed = [r for r in results if r.failed]
    timed_out = [r for r in results if not r.solved and not r.failed]

    def pct(n, total):
        return f"{100*n/total:.1f}%" if total > 0 else "0.0%"

    print(f"  Outcomes ({len(results)} episodes):")
    print(f"    Solved:    {len(solved):4d}  ({pct(len(solved), len(results))})")
    print(f"    Failed:    {len(failed):4d}  ({pct(len(failed), len(results))})")
    print(f"    Timed out: {len(timed_out):4d}  ({pct(len(timed_out), len(results))})")
    print()

    # Failure breakdown
    if failed:
        print(f"  Failure reasons:")
        reason_counts = Counter()
        for r in failed:
            if r.failure_reason:
                for part in r.failure_reason.split("; "):
                    reason_counts[part] += 1
            else:
                reason_counts["unknown"] += 1
        for reason, count in reason_counts.most_common():
            print(f"    {reason}: {count}")
        print()

    # Steps to solve
    if solved:
        steps_arr = np.array([r.solved_step for r in solved])
        print(f"  Steps to solve ({len(solved)} episodes):")
        print(f"    mean={steps_arr.mean():.1f}  std={steps_arr.std():.1f}  "
              f"min={steps_arr.min()}  max={steps_arr.max()}")

    # Wall time
    times = np.array([r.wall_time_seconds for r in results if r.wall_time_seconds > 0])
    if len(times) > 0:
        print(f"  Wall time per episode ({len(times)} episodes):")
        print(f"    mean={times.mean():.1f}s  std={times.std():.1f}s  "
              f"min={times.min():.1f}s  max={times.max():.1f}s")
    print()

    # Aggregate timing across all steps of all episodes
    all_timings = []
    for r in results:
        for s in r.steps:
            if s.ego_timing:
                all_timings.append(s.ego_timing)

    if all_timings:
        all_keys = set()
        for t in all_timings:
            all_keys.update(t.keys())

        print(f"  Ego timing (aggregated over {len(all_timings)} steps):")
        totals = [sum(t.values()) for t in all_timings]
        arr = np.array(totals) * 1000
        print(f"    total:          mean={arr.mean():.1f}ms  "
              f"std={arr.std():.1f}ms  min={arr.min():.1f}ms  max={arr.max():.1f}ms")
        for key in sorted(all_keys):
            vals = [t.get(key, 0.0) for t in all_timings]
            arr = np.array(vals) * 1000
            print(f"    {key + ':':<18s}mean={arr.mean():.1f}ms  "
                  f"std={arr.std():.1f}ms  min={arr.min():.1f}ms  max={arr.max():.1f}ms")
        print()

    # Aggregate prediction error
    all_pred_errors = []
    for r in results:
        for s in r.steps:
            if s.prediction_error is not None:
                all_pred_errors.append(s.prediction_error)

    if all_pred_errors:
        arr = np.array(all_pred_errors)
        print(f"  Prediction error (aggregated over {len(arr)} steps):")
        print(f"    mean={arr.mean():.3f}m  std={arr.std():.3f}m  "
              f"min={arr.min():.3f}m  max={arr.max():.3f}m")
        print()

    # Aggregate NLP convergence
    all_nlp = []
    for r in results:
        for s in r.steps:
            if s.true_diag_nlp_ok is not None:
                all_nlp.append(s.true_diag_nlp_ok)
    if all_nlp:
        n_ok = sum(all_nlp)
        print(f"  NLP convergence: {n_ok}/{len(all_nlp)} "
              f"({100*n_ok/len(all_nlp):.1f}%)")
        print()

    # Aggregate intervention stats
    all_steps = [s for r in results for s in r.steps]
    n_interv = sum(1 for s in all_steps if s.intervention_active)
    if n_interv > 0:
        print(f"  Intervention active: {n_interv}/{len(all_steps)} steps "
              f"({100*n_interv/len(all_steps):.1f}%)")
        dev_vals = [s.action_deviation for s in all_steps
                    if s.intervention_active and s.action_deviation is not None]
        if dev_vals:
            dev = np.array(dev_vals)
            print(f"  Action deviation (intervention steps only, n={len(dev_vals)}):")
            print(f"    mean={dev.mean():.4f}  std={dev.std():.4f}  "
                  f"min={dev.min():.4f}  max={dev.max():.4f}")
        print()

    # Agent count distribution
    agent_counts = [len(r.config.get("agents", [])) for r in results]
    if agent_counts:
        counts = Counter(agent_counts)
        print(f"  Agent count distribution:")
        for n, c in sorted(counts.items()):
            print(f"    {n} agents: {c} episodes")
        print()

    # Per-episode details
    if show_steps or len(results) <= 10:
        print(f"  {'='*60}")
        print(f"  Per-episode details:")
        print(f"  {'='*60}")
        for i, r in enumerate(results):
            print_episode(r, idx=i, show_steps=show_steps)


def main():
    args = parse_args()

    if not os.path.exists(args.path):
        print(f"Path not found: {args.path}")
        sys.exit(1)

    # Resolve the pickle path — support both directory and direct .pkl
    metadata = None
    if os.path.isdir(args.path):
        pkl_path = os.path.join(args.path, "results.pkl")
        if not os.path.exists(pkl_path):
            print(f"No results.pkl found in directory: {args.path}")
            sys.exit(1)
        meta_path = os.path.join(args.path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)
    else:
        pkl_path = args.path

    with open(pkl_path, 'rb') as f:
        data = dill.load(f)

    print(f"\nLoaded: {pkl_path}")
    print(f"Type: {type(data).__name__}")

    if metadata is not None:
        print(f"\n  Metadata:")
        print(f"    Scenario:          {metadata.get('scenario', '?')}")
        print(f"    Mode:              {metadata.get('mode', '?')}")
        print(f"    Seed:              {metadata.get('seed', '?')}")
        print(f"    Max steps:         {metadata.get('max_steps', '?')}")
        print(f"    Intervention type: {metadata.get('intervention_type', '?')}")
        n_samples = metadata.get('n_samples')
        if n_samples is not None:
            print(f"    N samples:         {n_samples}")
        print(f"    Timestamp:         {metadata.get('timestamp', '?')}")

    print()

    if isinstance(data, dict) and "results" in data:
        # Batch format
        inspect_batch(data, show_steps=args.steps)
    elif isinstance(data, ExperimentResult):
        # Single run format
        print(f"{'='*70}")
        print(f"  Single Episode Result")
        print(f"{'='*70}")
        print()
        print_episode(data, show_steps=args.steps)
    else:
        # Unknown format — dump keys/type info
        print(f"Unknown format. Top-level type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in data.items():
                print(f"  {k}: {type(v).__name__}", end="")
                if isinstance(v, list):
                    print(f" (len={len(v)})", end="")
                print()
        else:
            print(f"Attributes: {dir(data)}")


if __name__ == "__main__":
    main()
