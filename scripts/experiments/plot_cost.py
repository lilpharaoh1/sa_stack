"""
Plot bar charts comparing average per-step second-stage cost across
intervention schemes.

The second-stage cost is evaluated from the ego's Frenet state at each
step using the NLP cost weights:

    J_state = w_d * d^2 + w_phi * phi^2 + w_v * (v - v_goal)^2

This gives a consistent, per-step metric of trajectory quality that is
comparable across all intervention types.

Usage:
    # Auto-discover all belief_experiment1 runs:
    python scripts/experiments/plot_cost.py -m belief_experiment1 --latest

    # Save figure:
    python scripts/experiments/plot_cost.py -m belief_experiment1 --latest -o cost.pdf
"""

import sys
import os
import json
import argparse
from collections import defaultdict

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _script_dir)

import dill
from belief_utils import ExperimentResult, StepRecord, RESULTS_DIR

# NLP second-stage cost weights (from SecondStagePlanner.DEFAULTS)
W_S = 0.1
W_D = 10.0
W_PHI = 2.0
W_V = 0.01
W_A = 1.0
W_DELTA = 2.0

# Nice labels and ordering for intervention types
INTERVENTION_ORDER = ['none', 'agency_only', 'combined', 'warmstart_only', 'policy_only']
INTERVENTION_LABELS = {
    'none': 'None',
    'agency_only': 'Agency Only',
    'combined': 'Combined',
    'warmstart_only': 'Warmstart Only',
    'policy_only': 'Policy Only',
}
INTERVENTION_COLOURS = {
    'none': '#7F7F7F',
    'agency_only': '#4C72B0',
    'combined': '#DD8452',
    'warmstart_only': '#55A868',
    'policy_only': '#C44E52',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-step second-stage cost by intervention type")
    parser.add_argument("dirs", nargs="*",
                        help="Run directories to include (optional)")
    parser.add_argument("-m", "--map", type=str, default=None,
                        help="Scenario name to auto-discover runs for")
    parser.add_argument("--latest", action="store_true",
                        help="Keep only the latest run per intervention type")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save figure to this path instead of showing")
    return parser.parse_args()


def discover_runs(scenario_name: str) -> list:
    if not os.path.isdir(RESULTS_DIR):
        return []
    runs = []
    for entry in sorted(os.listdir(RESULTS_DIR)):
        full = os.path.join(RESULTS_DIR, entry)
        if not os.path.isdir(full):
            continue
        if not entry.startswith(scenario_name + "_"):
            continue
        meta_path = os.path.join(full, "metadata.json")
        pkl_path = os.path.join(full, "results.pkl")
        if os.path.exists(meta_path) and os.path.exists(pkl_path):
            runs.append(full)
    return runs


def keep_latest(runs: list) -> list:
    by_type = defaultdict(list)
    for run_dir in runs:
        meta_path = os.path.join(run_dir, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        itype = meta.get("intervention_type", "none")
        by_type[itype].append(run_dir)
    result = []
    for itype, dirs in by_type.items():
        result.append(sorted(dirs)[-1])
    return result


def compute_horizon_cost(states: np.ndarray, controls: np.ndarray,
                         v_goal: float, dt: float) -> float:
    """Compute the full second-stage NLP cost over a horizon.

    Matches the objective in SecondStagePlanner:
        J = sum_k [ w_s*(s_k - ref_s_k)^2 + w_d*d_k^2
                   + w_phi*phi_k^2 + w_v*(v_k - v_goal)^2 ]
          + sum_k [ w_a*a_k^2 + w_delta*delta_k^2 ]

    Args:
        states: (H+1, 4) Frenet states [s, d, phi, v].
        controls: (H, 2) controls [a, delta].
        v_goal: Target cruising speed (m/s).
        dt: Timestep duration (seconds).

    Returns:
        Total cost over the horizon.
    """
    H = controls.shape[0]
    s0 = states[0, 0]

    cost = 0.0
    for k in range(H + 1):
        ref_s = s0 + v_goal * k * dt
        cost += W_S * (states[k, 0] - ref_s) ** 2
        cost += W_D * states[k, 1] ** 2
        cost += W_PHI * states[k, 2] ** 2
        cost += W_V * (states[k, 3] - v_goal) ** 2

    for k in range(H):
        cost += W_A * controls[k, 0] ** 2
        cost += W_DELTA * controls[k, 1] ** 2

    return cost


def compute_single_step_cost(frenet_state: np.ndarray, v_goal: float,
                             accel: float = 0.0,
                             steer: float = 0.0) -> float:
    """Compute an approximate single-step cost (fallback when no horizon data).

    Uses s_ref = s (no tracking error for single step) since we cannot
    reconstruct the reference trajectory without the initial s0.

    Args:
        frenet_state: [s, d, phi, v].
        v_goal: Target speed.
        accel: Estimated acceleration.
        steer: Estimated steering angle.

    Returns:
        Single-step cost.
    """
    _, d, phi, v = frenet_state
    cost = (W_D * d ** 2 + W_PHI * phi ** 2 + W_V * (v - v_goal) ** 2
            + W_A * accel ** 2 + W_DELTA * steer ** 2)
    return cost


def extract_cost_stats(run_dir: str) -> dict:
    """Load a run and compute per-step second-stage costs.

    When full horizon data is available (intervention_opt_states/controls),
    computes the exact NLP cost.  Otherwise falls back to a single-step
    approximation from ego_frenet_state + estimated acceleration.

    Returns dict with cost statistics or None on failure.
    """
    meta_path = os.path.join(run_dir, "metadata.json")
    pkl_path = os.path.join(run_dir, "results.pkl")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)

    itype = meta.get("intervention_type", "none")

    if isinstance(data, ExperimentResult):
        episodes = [data]
    elif isinstance(data, dict) and "results" in data:
        episodes = data["results"]
    else:
        print(f"  Warning: unknown format in {run_dir}")
        return None

    all_costs = []
    n_episodes = len(episodes)
    n_solved = 0
    n_failed = 0
    n_steps = 0
    n_skipped = 0
    n_horizon = 0
    n_fallback = 0

    for ep in episodes:
        if ep.solved:
            n_solved += 1
        if ep.failed:
            n_failed += 1

        v_goal = ep.config.get("scenario", {}).get("max_speed", 10.0)
        fps = ep.fps
        dt = 1.0 / fps

        prev_speed = None
        for sr in ep.steps:
            n_steps += 1

            # Prefer full horizon cost from intervention data
            if (sr.intervention_opt_states is not None
                    and sr.intervention_opt_controls is not None):
                c = compute_horizon_cost(
                    sr.intervention_opt_states,
                    sr.intervention_opt_controls,
                    v_goal, dt)
                all_costs.append(c)
                n_horizon += 1
            elif sr.ego_frenet_state is not None:
                # Fallback: single-step cost with estimated acceleration
                cur_speed = sr.ego_speed if sr.ego_speed is not None else sr.ego_frenet_state[3]
                if prev_speed is not None:
                    accel_est = (cur_speed - prev_speed) / dt
                else:
                    accel_est = 0.0
                c = compute_single_step_cost(
                    sr.ego_frenet_state, v_goal,
                    accel=accel_est)
                all_costs.append(c)
                n_fallback += 1
            else:
                n_skipped += 1

            prev_speed = sr.ego_speed if sr.ego_speed is not None else prev_speed

    if n_skipped > 0:
        print(f"  Warning: {n_skipped}/{n_steps} steps missing frenet state")
    print(f"    Cost source: {n_horizon} horizon, {n_fallback} single-step fallback")

    return {
        'intervention_type': itype,
        'costs': all_costs,
        'n_episodes': n_episodes,
        'n_steps': n_steps,
        'n_solved': n_solved,
        'n_failed': n_failed,
    }


def plot_bars(stats_by_type: dict, output: str = None):
    import matplotlib.pyplot as plt

    types = [t for t in INTERVENTION_ORDER if t in stats_by_type]
    if not types:
        print("No data to plot.")
        return

    labels = [INTERVENTION_LABELS.get(t, t) for t in types]
    colours = [INTERVENTION_COLOURS.get(t, '#999999') for t in types]

    means = []
    stds = []
    for t in types:
        c = np.array(stats_by_type[t]['costs']) if stats_by_type[t]['costs'] else np.array([0.0])
        means.append(c.mean())
        stds.append(c.std())

    x = np.arange(len(types))
    width = 0.6

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean per-step cost')
    ax.set_title('Second-Stage State Cost by Intervention Type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)

    for bar, val in zip(bars, means):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Subtitle with episode counts
    subtitle_parts = []
    for t in types:
        s = stats_by_type[t]
        lbl = INTERVENTION_LABELS.get(t, t)
        subtitle_parts.append(
            f"{lbl}: {s['n_episodes']}ep, "
            f"{s['n_solved']}ok/{s['n_failed']}fail")
    fig.text(0.5, -0.02, "  |  ".join(subtitle_parts),
             ha='center', fontsize=7, color='gray')

    # Cost formula annotation
    ax.annotate(
        r"$J = w_d \, d^2 + w_\phi \, \phi^2 + w_v \, (v - v_{goal})^2$"
        f"\n($w_d$={W_D}, $w_\\phi$={W_PHI}, $w_v$={W_V})",
        xy=(0.98, 0.95), xycoords='axes fraction',
        ha='right', va='top', fontsize=7, color='gray',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output}")
    else:
        plt.show(block=True)


def main():
    args = parse_args()

    run_dirs = list(args.dirs)
    if args.map:
        discovered = discover_runs(args.map)
        if not discovered:
            print(f"No runs found for scenario '{args.map}' in {RESULTS_DIR}")
            if not run_dirs:
                sys.exit(1)
        run_dirs.extend(discovered)

    if not run_dirs:
        print("No run directories specified. Use -m <scenario> or pass dirs.")
        sys.exit(1)

    if args.latest:
        run_dirs = keep_latest(run_dirs)

    seen = set()
    unique = []
    for d in run_dirs:
        d = os.path.abspath(d)
        if d not in seen:
            seen.add(d)
            unique.append(d)
    run_dirs = unique

    print(f"Loading {len(run_dirs)} run(s)...")

    merged = defaultdict(lambda: {
        'costs': [],
        'n_episodes': 0, 'n_steps': 0,
        'n_solved': 0, 'n_failed': 0,
    })

    for d in run_dirs:
        print(f"  {os.path.basename(d)}")
        stats = extract_cost_stats(d)
        if stats is None:
            continue

        itype = stats['intervention_type']
        m = merged[itype]
        m['costs'].extend(stats['costs'])
        m['n_episodes'] += stats['n_episodes']
        m['n_steps'] += stats['n_steps']
        m['n_solved'] += stats['n_solved']
        m['n_failed'] += stats['n_failed']
        m['intervention_type'] = itype

    if not merged:
        print("No valid data found.")
        sys.exit(1)

    # Print text summary
    print(f"\n{'=' * 60}")
    print(f"  Per-Step Second-Stage Cost Comparison")
    print(f"{'=' * 60}")
    for itype in INTERVENTION_ORDER:
        if itype not in merged:
            continue
        s = merged[itype]
        c = np.array(s['costs']) if s['costs'] else np.array([0.0])
        label = INTERVENTION_LABELS.get(itype, itype)
        print(f"\n  {label} ({s['n_episodes']} episodes, "
              f"{s['n_solved']} solved, {s['n_failed']} failed):")
        print(f"    Steps: {s['n_steps']} total")
        print(f"    Cost:  mean={c.mean():.6f}  std={c.std():.6f}  "
              f"median={np.median(c):.6f}  max={c.max():.6f}")
    print()

    plot_bars(merged, output=args.output)


if __name__ == "__main__":
    main()
