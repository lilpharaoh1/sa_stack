"""
Plot bar charts comparing intervention magnitude across intervention types.

Scans result directories, groups by intervention type, and plots:
  - Mean |acceleration intervention| (m/s^2)
  - Mean |steering intervention| (rad)
  - Mean input modification (L2 norm of [da, ddelta])

Usage:
    # Auto-discover all belief_experiment1 runs:
    python scripts/experiments/plot_intervention_bars.py -m belief_experiment1

    # Specific run directories:
    python scripts/experiments/plot_intervention_bars.py \
        data/results/belief_experiment1_combined_seed21_n1_... \
        data/results/belief_experiment1_policy_only_seed21_n1_...

    # Use latest run per intervention type (skip older duplicates):
    python scripts/experiments/plot_intervention_bars.py -m belief_experiment1 --latest

    # Save figure instead of showing:
    python scripts/experiments/plot_intervention_bars.py -m belief_experiment1 -o bars.pdf
"""

import sys
import os
import json
import argparse
from collections import defaultdict

import numpy as np

# Ensure repo root and scripts/experiments are on the path
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _script_dir)

import dill
from belief_utils import ExperimentResult, StepRecord, RESULTS_DIR


# Nice labels and ordering for intervention types
INTERVENTION_ORDER = ['none', 'agency_only', 'combined', 'policy_only']
INTERVENTION_LABELS = {
    'none': 'None',
    'agency_only': 'Agency Only',
    'combined': 'Combined',
    'policy_only': 'Policy Only',
}

# Consistent colour per intervention type (used across all subplots)
INTERVENTION_COLOURS = {
    'none': '#7F7F7F',          # grey
    'agency_only': '#4C72B0',   # blue
    'combined': '#DD8452',      # orange
    'policy_only': '#C44E52',   # red
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot intervention magnitude bar charts")
    parser.add_argument("dirs", nargs="*",
                        help="Run directories to include (optional)")
    parser.add_argument("-m", "--map", type=str, default=None,
                        help="Scenario name to auto-discover runs for "
                             "(e.g. belief_experiment1)")
    parser.add_argument("--latest", action="store_true",
                        help="Keep only the latest run per intervention type")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save figure to this path instead of showing")
    parser.add_argument("--all-steps", action="store_true",
                        help="Compute mean over ALL steps (not just "
                             "intervention-active steps)")
    return parser.parse_args()


def discover_runs(scenario_name: str) -> list:
    """Find all run directories matching a scenario name."""
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
    """Keep only the latest run directory per intervention type."""
    by_type = defaultdict(list)
    for run_dir in runs:
        meta_path = os.path.join(run_dir, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        itype = meta.get("intervention_type", "none")
        by_type[itype].append(run_dir)

    # Directories are sorted alphabetically; last = newest timestamp
    result = []
    for itype, dirs in by_type.items():
        result.append(sorted(dirs)[-1])
    return result


def extract_intervention_stats(run_dir: str) -> dict:
    """Load a run and compute intervention statistics.

    Returns dict with:
        intervention_type, n_episodes, n_steps, n_active,
        da_values, dd_values (lists of first-step |da|, |ddelta|),
        action_dev_values (list of combined L2 norms),
        scenario, solved, failed
    """
    meta_path = os.path.join(run_dir, "metadata.json")
    pkl_path = os.path.join(run_dir, "results.pkl")

    with open(meta_path) as f:
        meta = json.load(f)

    with open(pkl_path, 'rb') as f:
        data = dill.load(f)

    itype = meta.get("intervention_type", "none")
    scenario = meta.get("scenario", "?")

    # Handle both single and batch formats
    if isinstance(data, ExperimentResult):
        episodes = [data]
    elif isinstance(data, dict) and "results" in data:
        episodes = data["results"]
    else:
        print(f"  Warning: unknown format in {run_dir}")
        return None

    da_vals = []       # |acceleration intervention| at applied step
    dd_vals = []       # |steering intervention| at applied step
    im_vals = []       # L2 norm of [da, ddelta] (input modification)
    dev_vals = []      # combined L2 action deviation
    n_steps = 0
    n_active = 0
    n_solved = 0
    n_failed = 0

    for ep in episodes:
        if ep.solved:
            n_solved += 1
        if ep.failed:
            n_failed += 1

        for s in ep.steps:
            n_steps += 1
            if s.intervention_active:
                n_active += 1
                ic = s.intervention_controls
                if ic is not None and ic.ndim == 2 and ic.shape[1] >= 2:
                    da = float(ic[0, 0])
                    dd = float(ic[0, 1])
                    da_vals.append(abs(da))
                    dd_vals.append(abs(dd))
                    im_vals.append(np.sqrt(da**2 + dd**2))
                elif ic is not None and ic.ndim == 2 and ic.shape[1] == 1:
                    # Longitudinal-only: only acceleration
                    da = float(ic[0, 0])
                    da_vals.append(abs(da))
                    dd_vals.append(0.0)
                    im_vals.append(abs(da))
                if s.action_deviation is not None:
                    dev_vals.append(float(s.action_deviation))

    return {
        'intervention_type': itype,
        'scenario': scenario,
        'run_dir': run_dir,
        'n_episodes': len(episodes),
        'n_steps': n_steps,
        'n_active': n_active,
        'n_solved': n_solved,
        'n_failed': n_failed,
        'da_values': da_vals,
        'dd_values': dd_vals,
        'input_mod_values': im_vals,
        'action_dev_values': dev_vals,
    }


def plot_bars(stats_by_type: dict, output: str = None,
              all_steps: bool = False):
    """Plot grouped bar charts for intervention magnitude.

    Args:
        stats_by_type: {intervention_type: stats_dict}
        output: If given, save figure to this path.
        all_steps: If True, denominator is all steps (not just active).
    """
    import matplotlib.pyplot as plt

    # Order types
    types = [t for t in INTERVENTION_ORDER if t in stats_by_type]
    if not types:
        print("No data to plot.")
        return

    labels = [INTERVENTION_LABELS.get(t, t) for t in types]

    # Compute means and stds
    mean_da = []
    std_da = []
    mean_dd = []
    std_dd = []
    mean_im = []
    std_im = []

    for t in types:
        s = stats_by_type[t]
        da = np.array(s['da_values']) if s['da_values'] else np.array([0.0])
        dd = np.array(s['dd_values']) if s['dd_values'] else np.array([0.0])
        im = np.array(s['input_mod_values']) if s['input_mod_values'] else np.array([0.0])

        if all_steps and s['n_steps'] > 0:
            # Pad with zeros for non-intervention steps
            n_zeros = s['n_steps'] - s['n_active']
            da = np.concatenate([da, np.zeros(n_zeros)])
            dd = np.concatenate([dd, np.zeros(n_zeros)])
            im = np.concatenate([im, np.zeros(n_zeros)])

        mean_da.append(da.mean())
        std_da.append(da.std())
        mean_dd.append(dd.mean())
        std_dd.append(dd.std())
        mean_im.append(im.mean())
        std_im.append(im.std())

    x = np.arange(len(types))
    width = 0.6
    colours = [INTERVENTION_COLOURS.get(t, '#999999') for t in types]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # --- Bar 1: Mean |da| (acceleration) ---
    ax = axes[0]
    bars = ax.bar(x, mean_da, width, yerr=std_da, capsize=4,
                  color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean |da| (m/s²)')
    ax.set_title('Acceleration Intervention')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    # Value labels
    for bar, val in zip(bars, mean_da):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # --- Bar 2: Mean |ddelta| (steering) ---
    ax = axes[1]
    bars = ax.bar(x, mean_dd, width, yerr=std_dd, capsize=4,
                  color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean |dδ| (rad)')
    ax.set_title('Steering Intervention')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, mean_dd):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # --- Bar 3: Input Modification (L2 norm of [da, ddelta]) ---
    ax = axes[2]
    bars = ax.bar(x, mean_im, width, yerr=std_im, capsize=4,
                  color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean ||[da, dδ]||₂')
    ax.set_title('Input Modification')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, mean_im):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Episode counts as subtitle
    subtitle_parts = []
    for t in types:
        s = stats_by_type[t]
        lbl = INTERVENTION_LABELS.get(t, t)
        subtitle_parts.append(
            f"{lbl}: {s['n_episodes']}ep, "
            f"{s['n_solved']}ok/{s['n_failed']}fail, "
            f"{s['n_active']}/{s['n_steps']} active steps")
    fig.suptitle("Intervention Magnitude by Type", fontsize=13, y=1.02)
    fig.text(0.5, -0.02, "  |  ".join(subtitle_parts),
             ha='center', fontsize=7, color='gray')

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output}")
    else:
        plt.show(block=True)


def main():
    args = parse_args()

    # Collect run directories
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

    # Remove duplicates preserving order
    seen = set()
    unique = []
    for d in run_dirs:
        d = os.path.abspath(d)
        if d not in seen:
            seen.add(d)
            unique.append(d)
    run_dirs = unique

    print(f"Loading {len(run_dirs)} run(s)...")

    # Group stats by intervention type (merge multiple runs of same type)
    merged = defaultdict(lambda: {
        'da_values': [], 'dd_values': [], 'input_mod_values': [],
        'action_dev_values': [],
        'n_episodes': 0, 'n_steps': 0, 'n_active': 0,
        'n_solved': 0, 'n_failed': 0,
    })

    for d in run_dirs:
        print(f"  {os.path.basename(d)}")
        stats = extract_intervention_stats(d)
        if stats is None:
            continue

        itype = stats['intervention_type']
        m = merged[itype]
        m['da_values'].extend(stats['da_values'])
        m['dd_values'].extend(stats['dd_values'])
        m['input_mod_values'].extend(stats['input_mod_values'])
        m['action_dev_values'].extend(stats['action_dev_values'])
        m['n_episodes'] += stats['n_episodes']
        m['n_steps'] += stats['n_steps']
        m['n_active'] += stats['n_active']
        m['n_solved'] += stats['n_solved']
        m['n_failed'] += stats['n_failed']
        m['intervention_type'] = itype

    if not merged:
        print("No valid data found.")
        sys.exit(1)

    # Print text summary
    print(f"\n{'='*60}")
    print(f"  Intervention Comparison Summary")
    print(f"{'='*60}")
    for itype in INTERVENTION_ORDER:
        if itype not in merged:
            continue
        s = merged[itype]
        da = np.array(s['da_values']) if s['da_values'] else np.array([0.0])
        dd = np.array(s['dd_values']) if s['dd_values'] else np.array([0.0])
        im = np.array(s['input_mod_values']) if s['input_mod_values'] else np.array([0.0])
        rate = s['n_active'] / s['n_steps'] if s['n_steps'] > 0 else 0.0
        label = INTERVENTION_LABELS.get(itype, itype)
        print(f"\n  {label} ({s['n_episodes']} episodes, "
              f"{s['n_solved']} solved, {s['n_failed']} failed):")
        print(f"    Steps: {s['n_steps']} total, "
              f"{s['n_active']} active ({rate*100:.1f}%)")
        if s['n_active'] > 0:
            print(f"    |da|:     mean={da.mean():.4f}  "
                  f"std={da.std():.4f}  max={da.max():.4f} m/s²")
            print(f"    |ddelta|: mean={dd.mean():.4f}  "
                  f"std={dd.std():.4f}  max={dd.max():.4f} rad "
                  f"({np.degrees(dd.mean()):.2f}°)")
            print(f"    Input mod (||[da,dδ]||₂): mean={im.mean():.4f}  "
                  f"std={im.std():.4f}  max={im.max():.4f}")
    print()

    plot_bars(merged, output=args.output, all_steps=args.all_steps)


if __name__ == "__main__":
    main()
