"""
Visualise how belief estimates evolve through episodes.

X-axis: episode progress (0 to 1).
Y-axis: average P(hidden) across tracked agents.

One line per run (inference method), with ground-truth visibility shown
as horizontal reference lines.  Supports comparing across runs.

Usage:
    python scripts/experiments/plot_belief_trajectory.py -m belief_experiment4 --latest
    python scripts/experiments/plot_belief_trajectory.py -m belief_experiment4 --latest -o beliefs.pdf
    python scripts/experiments/plot_belief_trajectory.py -m belief_experiment4 --latest --per-agent
"""

import sys
import os
import json
import argparse
from collections import defaultdict

import dill
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", ".."))
sys.path.insert(0, _script_dir)

from belief_utils import ExperimentResult, RESULTS_DIR
from plot_comparison import (
    _composite_key, _sort_keys, _group_label,
)

# ── Colours ──────────────────────────────────────────────────────────
LINE_COLOURS = [
    '#4C72B0', '#DD8452', '#C44E52', '#55A868',
    '#8172B2', '#CCB974', '#64B5CD', '#E377C2',
    '#17BECF', '#7F7F7F',
]
LINE_STYLES = ['-', '--', '-.', ':']


# ── Discovery & loading ──────────────────────────────────────────────

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
        key = _composite_key(meta)
        by_type[key].append(run_dir)
    return [sorted(dirs)[-1] for dirs in by_type.values()]


def load_run(run_dir: str, episode: int = 0):
    """Return (meta, result) for the selected episode."""
    meta_path = os.path.join(run_dir, "metadata.json")
    pkl_path = os.path.join(run_dir, "results.pkl")
    with open(meta_path) as f:
        meta = json.load(f)
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)
    if isinstance(data, ExperimentResult):
        return meta, data
    elif isinstance(data, dict) and "results" in data:
        results = data["results"]
        if episode < len(results):
            return meta, results[episode]
        elif results:
            return meta, results[0]
    return meta, None


# ── Belief extraction ────────────────────────────────────────────────

def extract_beliefs(steps):
    """Extract per-agent P(hidden) traces and ground truth.

    Returns:
        agent_ids: sorted list of tracked agent ids
        progress: (N,) array in [0, 1]
        p_hidden: {aid: (N,) array of P(hidden)} — NaN where not available
        gt_visible: {aid: bool} ground truth (from first step that has it)
    """
    # Discover all agent ids across steps
    all_aids = set()
    for sr in steps:
        if sr.belief_marginals:
            all_aids.update(sr.belief_marginals.keys())
    agent_ids = sorted(all_aids)

    if not agent_ids:
        return agent_ids, np.array([]), {}, {}

    n = len(steps)
    progress = np.linspace(0, 1, n)
    p_hidden = {aid: np.full(n, np.nan) for aid in agent_ids}

    for i, sr in enumerate(steps):
        if sr.belief_marginals:
            for aid in agent_ids:
                if aid in sr.belief_marginals:
                    p_hidden[aid][i] = sr.belief_marginals[aid]

    # Ground truth from first step that has it
    gt_visible = {}
    for sr in steps:
        if sr.belief_ground_truth:
            gt_visible = dict(sr.belief_ground_truth)
            break

    return agent_ids, progress, p_hidden, gt_visible


# ── Plotting ─────────────────────────────────────────────────────────

MAX_COLS = 4


def plot_average(ax, run_data):
    """Plot average P(hidden) across all agents for each run."""
    for i, rd in enumerate(run_data):
        progress = rd['progress']
        p_hidden = rd['p_hidden']
        if len(progress) == 0:
            continue

        # Average across agents at each step
        aids = list(p_hidden.keys())
        stacked = np.array([p_hidden[aid] for aid in aids])
        # nanmean handles steps where some agents don't have marginals yet
        avg = np.nanmean(stacked, axis=0)

        colour = LINE_COLOURS[i % len(LINE_COLOURS)]
        ax.plot(progress, avg, '-', color=colour, linewidth=1.8,
                alpha=0.85, label=rd['label'])

    # Ground truth reference (from first run that has it)
    for rd in run_data:
        gt = rd['gt_visible']
        if gt:
            # Average ground truth P(hidden)
            gt_p_hidden = [0.0 if vis else 1.0 for vis in gt.values()]
            avg_gt = np.mean(gt_p_hidden)
            ax.axhline(avg_gt, color='black', linestyle=':', linewidth=1.0,
                        alpha=0.5, label=f'Ground truth (avg)')
            break

    ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.6, alpha=0.4)
    ax.set_xlabel('Episode progress')
    ax.set_ylabel('Average P(hidden)')
    ax.set_title('Belief Evolution (averaged across agents)')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.2)


def plot_per_agent(fig, run_data):
    """One subplot per agent, all runs overlaid."""
    # Collect all agent ids across runs
    all_aids = set()
    for rd in run_data:
        all_aids.update(rd['p_hidden'].keys())
    agent_ids = sorted(all_aids)

    if not agent_ids:
        return

    n_agents = len(agent_ids)
    n_cols = min(n_agents, MAX_COLS)
    n_rows = (n_agents + n_cols - 1) // n_cols

    axes = fig.subplots(n_rows, n_cols, squeeze=False)

    for ai, aid in enumerate(agent_ids):
        r, c = divmod(ai, n_cols)
        ax = axes[r, c]

        for i, rd in enumerate(run_data):
            progress = rd['progress']
            if aid not in rd['p_hidden'] or len(progress) == 0:
                continue
            vals = rd['p_hidden'][aid]
            colour = LINE_COLOURS[i % len(LINE_COLOURS)]
            ax.plot(progress, vals, '-', color=colour, linewidth=1.5,
                    alpha=0.85, label=rd['label'])

        # Ground truth for this agent
        for rd in run_data:
            gt = rd['gt_visible']
            if aid in gt:
                gt_val = 0.0 if gt[aid] else 1.0
                gt_label = 'visible' if gt[aid] else 'hidden'
                ax.axhline(gt_val, color='black', linestyle=':',
                            linewidth=1.0, alpha=0.5)
                ax.text(0.02, gt_val + 0.05, f'GT: {gt_label}',
                        fontsize=6, color='black', alpha=0.6,
                        transform=ax.get_yaxis_transform())
                break

        ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.6, alpha=0.4)
        ax.set_title(f'Agent {aid}', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)

        if r == n_rows - 1:
            ax.set_xlabel('Episode progress')
        if c == 0:
            ax.set_ylabel('P(hidden)')
        if ai == 0:
            ax.legend(fontsize=6, loc='best')

    # Hide unused axes
    for ai in range(n_agents, n_rows * n_cols):
        r, c = divmod(ai, n_cols)
        axes[r, c].set_visible(False)


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise belief evolution across episode progress")
    p.add_argument("dirs", nargs="*",
                   help="Run directories to include (optional)")
    p.add_argument("-m", "--map", type=str, default=None,
                   help="Scenario name to auto-discover runs")
    p.add_argument("--latest", action="store_true",
                   help="Keep only the latest run per type")
    p.add_argument("-e", "--episode", type=int, default=0,
                   help="Episode index for batch results (default: 0)")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Save figure to file instead of showing")
    p.add_argument("--per-agent", action="store_true",
                   help="Show separate subplot per agent instead of average")
    return p.parse_args()


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

    # Deduplicate
    seen = set()
    unique = []
    for d in run_dirs:
        d = os.path.abspath(d)
        if d not in seen:
            seen.add(d)
            unique.append(d)
    run_dirs = unique

    print(f"Loading {len(run_dirs)} run(s)...")

    run_data = []
    for d in run_dirs:
        print(f"  {os.path.basename(d)}")
        meta, result = load_run(d, episode=args.episode)
        if result is None or not result.steps:
            print(f"    Warning: no steps in {d}")
            continue

        agent_ids, progress, p_hidden, gt_visible = extract_beliefs(result.steps)
        key = _composite_key(meta)
        label = _group_label(key)

        run_data.append({
            'meta': meta,
            'key': key,
            'label': label,
            'agent_ids': agent_ids,
            'progress': progress,
            'p_hidden': p_hidden,
            'gt_visible': gt_visible,
        })

        if agent_ids:
            # Count steps with valid marginals
            any_valid = sum(1 for i in range(len(progress))
                           if any(not np.isnan(p_hidden[a][i])
                                  for a in agent_ids))
            print(f"    {label}: {len(progress)} steps, "
                  f"{any_valid} with marginals, "
                  f"agents={agent_ids}")
        else:
            print(f"    {label}: no belief marginals found")

    if not run_data:
        print("No valid data found.")
        sys.exit(1)

    # Sort runs in same order as plot_comparison
    sorted_keys = _sort_keys([rd['key'] for rd in run_data])
    key_order = {k: i for i, k in enumerate(sorted_keys)}
    run_data.sort(key=lambda rd: key_order.get(rd['key'], 99))

    # Plot
    import matplotlib.pyplot as plt

    if args.per_agent:
        # Determine grid size from agent count
        all_aids = set()
        for rd in run_data:
            all_aids.update(rd['p_hidden'].keys())
        n_agents = len(all_aids)
        n_cols = min(n_agents, MAX_COLS)
        n_rows = max(1, (n_agents + n_cols - 1) // n_cols)

        fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))
        plot_per_agent(fig, run_data)
        fig.suptitle('Belief Evolution per Agent', fontsize=13, y=1.01)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plot_average(ax, run_data)
        fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.output}")
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
