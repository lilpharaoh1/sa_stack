"""Summary statistics computation for belief experiments."""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from .data import ExperimentResult


def _arr_stats(values, percentiles=False) -> dict:
    """Compute mean/std/min/max (and optionally percentiles) from a list of floats."""
    arr = np.array(values)
    d = {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }
    if percentiles and len(arr) > 0:
        for p in (50, 75, 90, 95, 99):
            d[f"p{p}"] = round(float(np.percentile(arr, p)), 4)
    return d


def _compute_violation_stats(steps: list) -> dict:
    """Compute ego violation and cost stats from a list of StepRecords."""
    n = len(steps)
    if n == 0:
        return {}

    # --- Ego violations (categorised) ---
    n_accel = sum(1 for s in steps if s.ego_accel_violated)
    n_steer = sum(1 for s in steps if s.ego_steering_violated)
    n_jerk = sum(1 for s in steps if s.ego_jerk_violated)
    n_steer_rate = sum(1 for s in steps if s.ego_steer_rate_violated)
    n_collision = sum(1 for s in steps if s.ego_collision)

    violations = {
        "control": {
            "acceleration": {"count": n_accel, "rate": round(n_accel / n, 4)},
            "steering": {"count": n_steer, "rate": round(n_steer / n, 4)},
        },
        "comfort": {
            "jerk": {"count": n_jerk, "rate": round(n_jerk / n, 4)},
            "steer_rate": {"count": n_steer_rate, "rate": round(n_steer_rate / n, 4)},
        },
        "collision": {"count": n_collision, "rate": round(n_collision / n, 4)},
        "total_steps": n,
    }

    # --- Per-step ego cost ---
    cost_vals = [s.ego_step_cost for s in steps if s.ego_step_cost is not None]
    ego_cost = _arr_stats(cost_vals) if cost_vals else None
    if ego_cost:
        ego_cost["n_steps"] = len(cost_vals)

    # --- Decomposed action deviation (with percentiles) ---
    dev_a = [s.action_deviation_accel for s in steps
             if s.action_deviation_accel is not None]
    dev_d = [s.action_deviation_steer for s in steps
             if s.action_deviation_steer is not None]
    action_deviation_detail = {}
    if dev_a:
        action_deviation_detail["acceleration"] = _arr_stats(dev_a, percentiles=True)
    if dev_d:
        action_deviation_detail["steering"] = _arr_stats(dev_d, percentiles=True)

    # --- Jerk and steer-rate magnitudes ---
    jerk_vals = [s.ego_jerk for s in steps if s.ego_jerk is not None]
    steer_rate_vals = [s.ego_steer_rate for s in steps if s.ego_steer_rate is not None]
    comfort = {}
    if jerk_vals:
        comfort["jerk"] = _arr_stats(jerk_vals, percentiles=True)
    if steer_rate_vals:
        comfort["steer_rate"] = _arr_stats(steer_rate_vals, percentiles=True)

    return {
        "violations": violations,
        "ego_cost": ego_cost,
        "action_deviation_detail": action_deviation_detail or None,
        "comfort_magnitudes": comfort or None,
    }


def _compute_timing_stats(steps: list) -> Optional[dict]:
    """Aggregate per-step ``ego_timing`` dicts into summary statistics.

    Returns a dict keyed by timing component (e.g. ``human_policy``,
    ``belief_inference``, etc.) with mean/std/min/max, plus a ``total``
    entry that sums all components per step.  Returns ``None`` if no
    timing data is available.
    """
    # Collect all timing dicts that are non-None
    timing_dicts = [s.ego_timing for s in steps if s.ego_timing]
    if not timing_dicts:
        return None

    # Gather all unique keys across steps
    all_keys = set()
    for td in timing_dicts:
        all_keys.update(td.keys())

    result = {}
    for key in sorted(all_keys):
        vals = [td[key] for td in timing_dicts if key in td]
        if vals:
            arr = np.array(vals)
            result[key] = {
                "mean_ms": round(float(arr.mean()) * 1000, 2),
                "std_ms": round(float(arr.std()) * 1000, 2),
                "min_ms": round(float(arr.min()) * 1000, 2),
                "max_ms": round(float(arr.max()) * 1000, 2),
                "n_steps": len(vals),
            }

    # Total per step (sum of all components)
    totals = [sum(td.values()) for td in timing_dicts]
    arr = np.array(totals)
    result["total"] = {
        "mean_ms": round(float(arr.mean()) * 1000, 2),
        "std_ms": round(float(arr.std()) * 1000, 2),
        "min_ms": round(float(arr.min()) * 1000, 2),
        "max_ms": round(float(arr.max()) * 1000, 2),
        "n_steps": len(totals),
    }

    return result


def build_summary(result: ExperimentResult) -> dict:
    """Compute summary stats for a single experiment run.

    Returns a JSON-serialisable dict.
    """
    # NLP convergence
    nlp_flags = [s.true_diag_nlp_ok for s in result.steps
                 if s.true_diag_nlp_ok is not None]
    n_ok = sum(1 for f in nlp_flags if f)
    n_total_nlp = len(nlp_flags)
    nlp_rate = n_ok / n_total_nlp if n_total_nlp > 0 else None

    # Intervention
    n_interv = sum(1 for s in result.steps if s.intervention_active)
    n_total_steps = len(result.steps)
    dev_vals = [s.action_deviation for s in result.steps
                if s.intervention_active and s.action_deviation is not None]
    if dev_vals:
        action_dev = _arr_stats(dev_vals)
    else:
        action_dev = None

    # Violations and cost
    perf = _compute_violation_stats(result.steps)

    # Timing
    timing = _compute_timing_stats(result.steps)

    summary = {
        "solved": result.solved,
        "failed": result.failed,
        "total_steps": result.total_steps,
        "wall_time_seconds": round(result.wall_time_seconds, 1),
        "failure_reason": result.failure_reason,
        "nlp_convergence": {
            "ok": n_ok,
            "total": n_total_nlp,
            "rate": round(nlp_rate, 3) if nlp_rate is not None else None,
        },
        "intervention": {
            "active_steps": n_interv,
            "total_steps": n_total_steps,
            "rate": round(n_interv / n_total_steps, 3) if n_total_steps > 0 else 0.0,
            "action_deviation": action_dev,
        },
        "timing": timing,
    }
    summary.update(perf)
    return summary


def build_batch_summary(results: List[ExperimentResult],
                        n_viable: int,
                        n_nonviable: int,
                        batch_wall_time: float) -> dict:
    """Compute summary stats for a batch experiment.

    Returns a JSON-serialisable dict.
    """
    n_episodes = len(results)
    solved = [r for r in results if r.solved]
    failed = [r for r in results if r.failed]
    timed_out = [r for r in results if not r.solved and not r.failed]

    def _pct(count, total):
        return round(100 * count / total, 1) if total > 0 else 0.0

    # Outcomes
    outcomes = {
        "solved": {"count": len(solved), "pct": _pct(len(solved), n_viable)},
        "failed": {"count": len(failed), "pct": _pct(len(failed), n_viable)},
        "timed_out": {"count": len(timed_out), "pct": _pct(len(timed_out), n_viable)},
    }

    # Steps to solve
    steps_to_solve = None
    if solved:
        steps_arr = np.array([r.solved_step for r in solved])
        steps_to_solve = {
            "mean": round(float(steps_arr.mean()), 1),
            "std": round(float(steps_arr.std()), 1),
            "min": int(steps_arr.min()),
            "max": int(steps_arr.max()),
        }

    # Wall time per episode
    times = np.array([r.wall_time_seconds for r in results
                      if r.wall_time_seconds > 0])
    wall_time_per_ep = None
    if len(times) > 0:
        wall_time_per_ep = {
            "mean": round(float(times.mean()), 1),
            "std": round(float(times.std()), 1),
            "min": round(float(times.min()), 1),
            "max": round(float(times.max()), 1),
        }

    # Belief accuracy
    all_steps = [s for r in results for s in r.steps]
    acc_vals = [s.belief_accuracy for s in all_steps
                if s.belief_accuracy is not None]
    belief_accuracy = None
    if acc_vals:
        acc = np.array(acc_vals)
        belief_accuracy = {
            "mean": round(float(acc.mean()), 3),
            "std": round(float(acc.std()), 3),
            "n_steps": len(acc_vals),
        }

    # Intervention
    n_interv = sum(1 for s in all_steps if s.intervention_active)
    n_total_steps = len(all_steps)
    dev_vals = [s.action_deviation for s in all_steps
                if s.intervention_active and s.action_deviation is not None]
    if dev_vals:
        dev_arr = np.array(dev_vals)
        action_dev = {
            "mean": round(float(dev_arr.mean()), 4),
            "std": round(float(dev_arr.std()), 4),
            "min": round(float(dev_arr.min()), 4),
            "max": round(float(dev_arr.max()), 4),
        }
    else:
        action_dev = None

    intervention = {
        "active_steps": n_interv,
        "total_steps": n_total_steps,
        "rate": round(n_interv / n_total_steps, 3) if n_total_steps > 0 else 0.0,
        "action_deviation": action_dev,
    }

    # Failure breakdown
    failure_breakdown = {}
    if failed:
        reason_counts = Counter()
        for r in failed:
            if r.failure_reason:
                for part in r.failure_reason.split("; "):
                    reason_counts[part] += 1
            else:
                reason_counts["NLP infeasible"] += 1
        failure_breakdown = dict(reason_counts)

    # Violations and cost (aggregated across all steps in all episodes)
    perf = _compute_violation_stats(all_steps)

    # Timing (aggregated across all steps in all episodes)
    timing = _compute_timing_stats(all_steps)

    # Per-episode wall time breakdown (mean time per episode for each component)
    per_episode_timing = None
    if timing:
        ep_timings = []
        for r in results:
            ep_td = [s.ego_timing for s in r.steps if s.ego_timing]
            if ep_td:
                ep_totals = {}
                for td in ep_td:
                    for k, v in td.items():
                        ep_totals[k] = ep_totals.get(k, 0.0) + v
                ep_totals["total"] = sum(ep_totals.values())
                ep_timings.append(ep_totals)
        if ep_timings:
            all_keys = set()
            for et in ep_timings:
                all_keys.update(et.keys())
            per_episode_timing = {}
            for key in sorted(all_keys):
                vals = [et.get(key, 0.0) for et in ep_timings]
                arr = np.array(vals)
                per_episode_timing[key] = {
                    "mean_s": round(float(arr.mean()), 3),
                    "std_s": round(float(arr.std()), 3),
                    "min_s": round(float(arr.min()), 3),
                    "max_s": round(float(arr.max()), 3),
                }

    summary = {
        "n_episodes": n_episodes,
        "n_viable": n_viable,
        "n_nonviable": n_nonviable,
        "batch_wall_time_seconds": round(batch_wall_time, 1),
        "outcomes": outcomes,
        "steps_to_solve": steps_to_solve,
        "wall_time_per_episode": wall_time_per_ep,
        "belief_accuracy": belief_accuracy,
        "intervention": intervention,
        "failure_breakdown": failure_breakdown,
        "timing_per_step": timing,
        "timing_per_episode": per_episode_timing,
    }
    summary.update(perf)
    return summary
