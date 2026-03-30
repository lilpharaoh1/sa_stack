"""
Plot how the human's inferred beliefs evolve over time.

For each tracked vehicle, plots P(hidden) marginals across simulation steps.
Ground-truth visibility is shown as a shaded band so you can see how quickly
(or whether) the inference converges to the correct answer.

Supports multiple runs side-by-side for comparison.

Usage:
    python scripts/experiments/plot_belief_evolution.py results/my_run/
    python scripts/experiments/plot_belief_evolution.py run1/ run2/ --labels "Naive" "MCTS-K"
    python scripts/experiments/plot_belief_evolution.py results/my_run/ -o beliefs.png
    python scripts/experiments/plot_belief_evolution.py results/my_run/ --episode 0 --episode 2
"""

import sys
import os
import argparse
from typing import List, Dict, Optional, Tuple

import dill
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from belief_utils import ExperimentResult, StepRecord, RESULTS_DIR

# Distinguishable colours for different agents
AGENT_COLOURS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]


def _load_run(run_dir: str) -> List[ExperimentResult]:
    """Load all episodes from a run directory."""
    pkl_path = os.path.join(run_dir, "results.pkl")
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(RESULTS_DIR, run_dir, "results.pkl")
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    elif isinstance(data, ExperimentResult):
        return [data]
    return []


def _extract_belief_traces(
        episode: ExperimentResult,
) -> Tuple[Dict[int, List[float]], Dict[int, List[int]], Dict[int, bool]]:
    """Extract per-agent P(hidden) traces from one episode.

    Returns:
        (marginals_by_aid, steps_by_aid, ground_truth)
        where marginals_by_aid[aid] = list of P(hidden) values,
        steps_by_aid[aid] = list of step indices,
        ground_truth[aid] = True if agent is visible (from config).
    """
    marginals_by_aid: Dict[int, List[float]] = {}
    steps_by_aid: Dict[int, List[int]] = {}
    ground_truth: Dict[int, bool] = {}

    for sr in episode.steps:
        if sr.belief_marginals:
            for aid, p_hidden in sr.belief_marginals.items():
                if aid not in marginals_by_aid:
                    marginals_by_aid[aid] = []
                    steps_by_aid[aid] = []
                marginals_by_aid[aid].append(p_hidden)
                steps_by_aid[aid].append(sr.step)

        if sr.belief_ground_truth and not ground_truth:
            ground_truth = {
                aid: visible for aid, visible in sr.belief_ground_truth.items()
            }

    return marginals_by_aid, steps_by_aid, ground_truth


def plot_belief_evolution(
        episodes: List[ExperimentResult],
        episode_indices: Optional[List[int]] = None,
        figsize: Tuple[float, float] = (12, 4),
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
        alpha_band: float = 0.15,
        alpha_line: float = 0.4,
        show_mean: bool = True,
        hidden_threshold: float = 0.6,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot P(hidden) traces for all agents across episodes.

    Args:
        episodes: List of ExperimentResult from one run.
        episode_indices: Which episodes to include (None = all).
        figsize: Figure size (only used if ax is None).
        title: Optional plot title.
        ax: Optional existing axes.
        alpha_band: Transparency for per-episode traces.
        alpha_line: Transparency for individual episode lines.
        show_mean: If True, overlay the mean trace across episodes.
        hidden_threshold: Threshold line for P(hidden) classification.

    Returns:
        (fig, ax) tuple.
    """
    if episode_indices is not None:
        episodes = [episodes[i] for i in episode_indices
                    if i < len(episodes)]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Collect traces across all episodes
    # {aid: list of (steps_array, marginals_array)} across episodes
    all_traces: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    ground_truth: Dict[int, bool] = {}

    for ep in episodes:
        marginals_by_aid, steps_by_aid, gt = _extract_belief_traces(ep)
        if gt and not ground_truth:
            ground_truth = gt
        for aid in marginals_by_aid:
            if aid not in all_traces:
                all_traces[aid] = []
            all_traces[aid].append((
                np.array(steps_by_aid[aid]),
                np.array(marginals_by_aid[aid]),
            ))

    if not all_traces:
        ax.text(0.5, 0.5, "No belief data available",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        if title:
            ax.set_title(title, fontsize=11)
        return fig, ax

    # Sort agents by ID for consistent ordering
    sorted_aids = sorted(all_traces.keys())

    for idx, aid in enumerate(sorted_aids):
        colour = AGENT_COLOURS[idx % len(AGENT_COLOURS)]
        traces = all_traces[aid]

        gt_visible = ground_truth.get(aid, True)
        gt_label = "visible" if gt_visible else "hidden"

        # Plot individual episode traces
        for i, (steps, margs) in enumerate(traces):
            label = f"Agent {aid} ({gt_label})" if i == 0 else None
            ax.plot(steps, margs, color=colour, alpha=alpha_line,
                    linewidth=0.8, label=label)

        # Compute and plot mean trace
        if show_mean and len(traces) > 1:
            # Interpolate all traces onto a common step grid
            all_steps = np.unique(np.concatenate([t[0] for t in traces]))
            interp_vals = np.full((len(traces), len(all_steps)), np.nan)
            for i, (steps, margs) in enumerate(traces):
                interp_vals[i] = np.interp(all_steps, steps, margs,
                                           left=np.nan, right=np.nan)
            mean_vals = np.nanmean(interp_vals, axis=0)
            valid = ~np.isnan(mean_vals)
            ax.plot(all_steps[valid], mean_vals[valid], color=colour,
                    linewidth=2.0, alpha=0.9)

            # Std band
            std_vals = np.nanstd(interp_vals, axis=0)
            ax.fill_between(all_steps[valid],
                            np.clip(mean_vals[valid] - std_vals[valid], 0, 1),
                            np.clip(mean_vals[valid] + std_vals[valid], 0, 1),
                            color=colour, alpha=alpha_band)
        elif len(traces) == 1:
            # Single episode — make the line more visible
            steps, margs = traces[0]
            ax.plot(steps, margs, color=colour, linewidth=2.0, alpha=0.9)

        # Ground-truth band
        gt_y = 1.0 if not gt_visible else 0.0
        ax.axhline(y=gt_y, color=colour, linestyle=':', linewidth=0.8,
                   alpha=0.4)

    # Threshold line
    ax.axhline(y=hidden_threshold, color='gray', linestyle='--',
               linewidth=1.0, alpha=0.5, label=f'Threshold ({hidden_threshold})')

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('P(hidden)', fontsize=10)
    ax.set_xlabel('Step', fontsize=10)
    ax.legend(fontsize=8, loc='best', framealpha=0.8)

    if title:
        ax.set_title(title, fontsize=11)

    # Stats annotation
    n_eps = len(episodes)
    ax.text(0.98, 0.02, f"{n_eps} episode{'s' if n_eps != 1 else ''}",
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    return fig, ax


def main():
    args = parse_args()

    run_dirs = args.dirs
    resolved = []
    for d in run_dirs:
        if os.path.isdir(d):
            resolved.append(d)
        elif os.path.isdir(os.path.join(RESULTS_DIR, d)):
            resolved.append(os.path.join(RESULTS_DIR, d))
        else:
            print(f"Warning: directory not found: {d}")
    run_dirs = resolved

    if not run_dirs:
        print("No valid run directories.")
        sys.exit(1)

    all_runs = []
    all_labels = []
    for i, run_dir in enumerate(run_dirs):
        episodes = _load_run(run_dir)
        if not episodes:
            print(f"Warning: no episodes in {run_dir}")
            continue
        all_runs.append(episodes)
        if args.labels and i < len(args.labels):
            all_labels.append(args.labels[i])
        else:
            all_labels.append(os.path.basename(run_dir.rstrip('/')))

    if not all_runs:
        print("No data to plot.")
        sys.exit(1)

    # Parse episode indices
    ep_indices = args.episode if args.episode else None

    n_runs = len(all_runs)
    if n_runs == 1:
        fig, ax = plot_belief_evolution(
            all_runs[0],
            episode_indices=ep_indices,
            figsize=tuple(args.figsize),
            title=all_labels[0],
            hidden_threshold=args.threshold,
        )
    else:
        fig, axes = plt.subplots(n_runs, 1,
                                 figsize=(args.figsize[0],
                                          args.figsize[1] * n_runs),
                                 sharex=True)
        if n_runs == 1:
            axes = [axes]
        for i, (episodes, label) in enumerate(zip(all_runs, all_labels)):
            plot_belief_evolution(
                episodes,
                episode_indices=ep_indices,
                title=label,
                ax=axes[i],
                hidden_threshold=args.threshold,
            )
        fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot belief evolution (P(hidden) per agent over time)")
    parser.add_argument("dirs", nargs="+",
                        help="Run directories (or names under results/)")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each run directory")
    parser.add_argument("--episode", "-e", type=int, action="append",
                        default=None,
                        help="Episode index to include (repeatable; default: all)")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Save figure to file instead of showing")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--figsize", type=float, nargs=2, default=[12, 4],
                        metavar=("W", "H"),
                        help="Figure size per panel (default: 12 4)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Hidden threshold for classification (default: 0.6)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
