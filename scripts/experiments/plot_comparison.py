"""
Compare experiment runs across intervention types.

Produces a multi-panel figure and text summary covering:
  1. Outcome rates (solved / failed / timeout)
  2. Ego violations breakdown (collision, control, comfort)
  3. Per-step ego cost
  4. Action deviation (acceleration + steering, decomposed)
  5. Steps to solve (for solved episodes)

Auto-discovers run directories by scenario name, groups by intervention
type, and (optionally) by inference type.

Usage:
    # Auto-discover and compare latest run per intervention type:
    python scripts/experiments/plot_comparison.py -m belief_experiment4 --latest

    # Save figure:
    python scripts/experiments/plot_comparison.py -m belief_experiment4 --latest -o comparison.pdf

    # Include specific directories:
    python scripts/experiments/plot_comparison.py dir1/ dir2/ dir3/

    # Export LaTeX table:
    python scripts/experiments/plot_comparison.py -m belief_experiment4 --latest --latex
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
from belief_utils import ExperimentResult, RESULTS_DIR


# ── Labels & colours ─────────────────────────────────────────────────

# Readable short labels for each component
INFERENCE_LABELS = {
    'naive': 'Naive',
    'mcts_naive': 'MCTS',
    'mcts_resample': 'MCTS-R',
}
INTERVENTION_LABELS = {
    'none': 'None',
    'agency_only': 'Agency',
    'combined': 'Combined',
    'policy_only': 'Policy',
    'mcts': 'MCTS-int',
}
REF_CONTROLS_LABELS = {
    'opt': 'OPT',
    'mcts-greedy': 'Greedy',
    'mcts-qcbf': 'QCBF',
}

# Colour palette — assigned dynamically to composite keys as they appear
_COLOUR_PALETTE = [
    '#7F7F7F', '#4C72B0', '#DD8452', '#C44E52', '#55A868',
    '#8172B2', '#CCB974', '#64B5CD', '#E377C2', '#17BECF',
]


def _parse_composite_key(key: str):
    """Parse composite key into (inference, intervention, ref_controls|None)."""
    # Try matching known inference prefixes (longest first)
    inf = None
    remainder = key
    for inf_key in sorted(INFERENCE_LABELS.keys(), key=len, reverse=True):
        if key.startswith(inf_key + '_'):
            inf = inf_key
            remainder = key[len(inf_key) + 1:]
            break
    if inf is None:
        parts = key.split('_', 1)
        inf = parts[0]
        remainder = parts[1] if len(parts) > 1 else ''

    # Check if remainder ends with a known ref_controls suffix
    ref = None
    interv = remainder
    for ref_key in sorted(REF_CONTROLS_LABELS.keys(), key=len, reverse=True):
        suffix = '_' + ref_key
        if remainder.endswith(suffix):
            ref = ref_key
            interv = remainder[:-len(suffix)]
            break

    return inf, interv, ref


def _group_label(key: str) -> str:
    """Convert a composite key to a readable label."""
    inf, interv, ref = _parse_composite_key(key)
    inf_label = INFERENCE_LABELS.get(inf, inf)
    interv_label = INTERVENTION_LABELS.get(interv, interv)
    if ref is not None:
        ref_label = REF_CONTROLS_LABELS.get(ref, ref)
        return f"{inf_label} / {interv_label} / {ref_label}"
    return f"{inf_label} / {interv_label}"


def _group_colour(key: str, all_keys: list) -> str:
    """Assign a colour to a composite key based on its position."""
    idx = all_keys.index(key) if key in all_keys else 0
    return _COLOUR_PALETTE[idx % len(_COLOUR_PALETTE)]


def _sort_keys(keys) -> list:
    """Sort composite group keys in a sensible order."""
    inference_order = ['naive', 'mcts_naive', 'mcts_resample']
    intervention_order = ['none', 'agency_only', 'combined', 'policy_only', 'mcts']
    ref_order = [None, 'opt', 'mcts-greedy', 'mcts-qcbf']

    def _sort_key(k):
        inf, interv, ref = _parse_composite_key(k)
        inf_idx = inference_order.index(inf) if inf in inference_order else 99
        interv_idx = (intervention_order.index(interv)
                      if interv in intervention_order else 99)
        ref_idx = ref_order.index(ref) if ref in ref_order else 99
        return (inf_idx, interv_idx, ref_idx)

    return sorted(keys, key=_sort_key)


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare experiment metrics across intervention types")
    p.add_argument("dirs", nargs="*",
                   help="Run directories to include (optional)")
    p.add_argument("-m", "--map", type=str, default=None,
                   help="Scenario name to auto-discover runs")
    p.add_argument("--latest", action="store_true",
                   help="Keep only the latest run per intervention type")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Save figure to path instead of showing")
    p.add_argument("--latex", action="store_true",
                   help="Print a LaTeX table to stdout")
    return p.parse_args()


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


def _composite_key(meta: dict) -> str:
    """Build composite grouping key: inference_intervention[_refcontrols]."""
    inf = meta.get("inference_type", "naive")
    interv = meta.get("intervention_type", "none")
    ref = meta.get("ref_controls", "opt")
    if interv != "none" and ref != "opt":
        return f"{inf}_{interv}_{ref}"
    return f"{inf}_{interv}"


def keep_latest(runs: list) -> list:
    by_type = defaultdict(list)
    for run_dir in runs:
        meta_path = os.path.join(run_dir, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        key = _composite_key(meta)
        by_type[key].append(run_dir)
    return [sorted(dirs)[-1] for dirs in by_type.values()]


def load_run(run_dir: str):
    """Load a run directory, return (composite_key, episodes_list)."""
    meta_path = os.path.join(run_dir, "metadata.json")
    pkl_path = os.path.join(run_dir, "results.pkl")
    with open(meta_path) as f:
        meta = json.load(f)
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)

    key = _composite_key(meta)
    if isinstance(data, ExperimentResult):
        episodes = [data]
    elif isinstance(data, dict) and "results" in data:
        episodes = data["results"]
    else:
        return None, None
    return key, episodes


# ── Metric extraction ────────────────────────────────────────────────

def extract_metrics(episodes: list) -> dict:
    """Extract all comparison metrics from a list of episodes."""
    n_ep = len(episodes)
    n_solved = sum(1 for r in episodes if r.solved)
    n_failed = sum(1 for r in episodes if r.failed)
    n_timeout = sum(1 for r in episodes if not r.solved and not r.failed)

    all_steps = [s for r in episodes for s in r.steps]
    n_steps = len(all_steps)

    # Violations
    n_collision = sum(1 for s in all_steps if s.ego_collision)
    n_accel_v = sum(1 for s in all_steps if s.ego_accel_violated)
    n_steer_v = sum(1 for s in all_steps if s.ego_steering_violated)
    n_jerk_v = sum(1 for s in all_steps if s.ego_jerk_violated)
    n_srate_v = sum(1 for s in all_steps if s.ego_steer_rate_violated)

    # Per-step cost
    cost_vals = [s.ego_step_cost for s in all_steps
                 if s.ego_step_cost is not None]

    # Action deviation (all steps, not just intervention)
    dev_a = [s.action_deviation_accel for s in all_steps
             if s.action_deviation_accel is not None]
    dev_d = [s.action_deviation_steer for s in all_steps
             if s.action_deviation_steer is not None]
    dev_l2 = [s.action_deviation for s in all_steps
              if s.action_deviation is not None]

    # Intervention rate
    n_interv = sum(1 for s in all_steps if s.intervention_active)

    # Steps to solve
    solve_steps = [r.solved_step for r in episodes if r.solved]

    # Failure breakdown
    failure_reasons = defaultdict(int)
    for r in episodes:
        if r.failed and r.failure_reason:
            for part in r.failure_reason.split("; "):
                failure_reasons[part] += 1

    # Wall time per episode
    wall_times = [r.wall_time_seconds for r in episodes
                  if r.wall_time_seconds and r.wall_time_seconds > 0]

    # Per-step timing breakdown
    timing_dicts = [s.ego_timing for s in all_steps if s.ego_timing]
    timing_components = {}
    if timing_dicts:
        all_keys = set()
        for td in timing_dicts:
            all_keys.update(td.keys())
        for key in sorted(all_keys):
            vals_ms = [td[key] * 1000 for td in timing_dicts if key in td]
            if vals_ms:
                a = np.array(vals_ms)
                timing_components[key] = {
                    "mean": float(a.mean()), "std": float(a.std()),
                    "min": float(a.min()), "max": float(a.max()),
                    "n": len(vals_ms),
                }
        # Total per step
        totals_ms = [sum(td.values()) * 1000 for td in timing_dicts]
        a = np.array(totals_ms)
        timing_components["total"] = {
            "mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "n": len(totals_ms),
        }

    # Per-episode total time by component
    per_episode_timing = {}
    if timing_dicts:
        for r in episodes:
            ep_td = [s.ego_timing for s in r.steps if s.ego_timing]
            if not ep_td:
                continue
            ep_sums = {}
            for td in ep_td:
                for k, v in td.items():
                    ep_sums[k] = ep_sums.get(k, 0.0) + v
            for k, v in ep_sums.items():
                per_episode_timing.setdefault(k, []).append(v)
        # Compute stats
        for k in list(per_episode_timing.keys()):
            a = np.array(per_episode_timing[k])
            per_episode_timing[k] = {
                "mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
            }

    def _stats(vals):
        if not vals:
            return None
        a = np.array(vals)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max()),
                "median": float(np.median(a)), "n": len(vals)}

    return {
        "n_episodes": n_ep,
        "n_solved": n_solved,
        "n_failed": n_failed,
        "n_timeout": n_timeout,
        "n_steps": n_steps,
        "n_interv": n_interv,
        # Violation counts
        "n_collision": n_collision,
        "n_accel_violated": n_accel_v,
        "n_steer_violated": n_steer_v,
        "n_jerk_violated": n_jerk_v,
        "n_srate_violated": n_srate_v,
        # Stats
        "cost": _stats(cost_vals),
        "dev_accel": _stats(dev_a),
        "dev_steer": _stats(dev_d),
        "dev_l2": _stats(dev_l2),
        "solve_steps": _stats(solve_steps),
        "wall_time": _stats(wall_times),
        "failure_reasons": dict(failure_reasons),
        # Timing
        "timing_per_step": timing_components,
        "timing_per_episode": per_episode_timing,
        # Raw values for statistical tests
        "raw_cost": cost_vals,
        "raw_dev_accel": dev_a,
        "raw_dev_steer": dev_d,
        "raw_dev_l2": dev_l2,
        "raw_solve_steps": solve_steps,
        "raw_wall_time": wall_times,
    }


# ── Text summary ─────────────────────────────────────────────────────

def print_text_summary(metrics_by_type: dict):
    def pct(n, total):
        return f"{100*n/total:.1f}%" if total > 0 else "-"

    print(f"\n{'='*72}")
    print(f"  Experiment Comparison Summary")
    print(f"{'='*72}")

    for itype in _sort_keys(metrics_by_type.keys()):
        m = metrics_by_type[itype]
        label = _group_label(itype)
        n = m["n_steps"]

        print(f"\n  {label}  ({m['n_episodes']} episodes, {n} steps)")
        print(f"  {'─'*60}")

        # Outcomes
        print(f"    Outcomes:  solved={m['n_solved']}  failed={m['n_failed']}  "
              f"timeout={m['n_timeout']}")
        if m["solve_steps"]:
            ss = m["solve_steps"]
            print(f"    Steps to solve: mean={ss['mean']:.1f}  "
                  f"std={ss['std']:.1f}  "
                  f"min={ss['min']:.0f}  max={ss['max']:.0f}")

        # Violations
        print(f"    Violations ({n} steps):")
        print(f"      Collision:      {m['n_collision']:5d}  ({pct(m['n_collision'], n)})")
        print(f"      Accel bounds:   {m['n_accel_violated']:5d}  ({pct(m['n_accel_violated'], n)})")
        print(f"      Steer bounds:   {m['n_steer_violated']:5d}  ({pct(m['n_steer_violated'], n)})")
        print(f"      Jerk:           {m['n_jerk_violated']:5d}  ({pct(m['n_jerk_violated'], n)})")
        print(f"      Steer rate:     {m['n_srate_violated']:5d}  ({pct(m['n_srate_violated'], n)})")

        # Intervention rate
        print(f"    Intervention: {m['n_interv']}/{n} steps "
              f"({pct(m['n_interv'], n)})")

        # Cost
        if m["cost"]:
            c = m["cost"]
            print(f"    Ego step cost: mean={c['mean']:.4f}  "
                  f"std={c['std']:.4f}  median={c['median']:.4f}")

        # Action deviation
        if m["dev_l2"]:
            d = m["dev_l2"]
            print(f"    Action deviation (L2): mean={d['mean']:.4f}  "
                  f"std={d['std']:.4f}  max={d['max']:.4f}")
        if m["dev_accel"]:
            da = m["dev_accel"]
            print(f"      accel: mean={da['mean']:.4f}  max={da['max']:.4f}")
        if m["dev_steer"]:
            dd = m["dev_steer"]
            print(f"      steer: mean={dd['mean']:.4f}  max={dd['max']:.4f}")

        # Wall time per episode
        if m["wall_time"]:
            wt = m["wall_time"]
            print(f"    Wall time/episode: mean={wt['mean']:.1f}s  "
                  f"std={wt['std']:.1f}s  "
                  f"min={wt['min']:.1f}s  max={wt['max']:.1f}s")

        # Per-step timing breakdown
        if m["timing_per_step"]:
            print(f"    Per-step timing:")
            for key, st in sorted(m["timing_per_step"].items()):
                print(f"      {key:20s}  mean={st['mean']:7.1f}ms  "
                      f"std={st['std']:7.1f}ms  max={st['max']:7.1f}ms")

        # Per-episode timing
        if m["timing_per_episode"]:
            print(f"    Per-episode cumulative timing:")
            for key, st in sorted(m["timing_per_episode"].items()):
                print(f"      {key:20s}  mean={st['mean']:6.2f}s  "
                      f"std={st['std']:6.2f}s")

        # Failure breakdown
        if m["failure_reasons"]:
            print(f"    Failure breakdown:")
            for reason, count in sorted(m["failure_reasons"].items(),
                                         key=lambda x: -x[1]):
                print(f"      {reason}: {count}")

    print(f"\n{'='*72}\n")


# ── LaTeX table ──────────────────────────────────────────────────────

def print_latex_table(metrics_by_type: dict):
    types = _sort_keys(metrics_by_type.keys())
    if not types:
        return

    def pct(n, total):
        return f"{100*n/total:.1f}" if total > 0 else "--"

    def val_or_dash(stats, key, fmt=".4f"):
        if stats is None:
            return "--"
        return f"{stats[key]:{fmt}}"

    print()
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Experiment comparison across intervention types.}")
    print(r"\label{tab:comparison}")
    cols = "l" + "c" * len(types)
    print(r"\begin{tabular}{" + cols + r"}")
    print(r"\toprule")

    headers = " & ".join(_group_label(t) for t in types)
    print(r"Metric & " + headers + r" \\")
    print(r"\midrule")

    # Outcomes
    for label, key in [("Solved (\\%)", "n_solved"),
                       ("Failed (\\%)", "n_failed")]:
        vals = []
        for t in types:
            m = metrics_by_type[t]
            vals.append(pct(m[key], m["n_episodes"]))
        print(f"{label} & " + " & ".join(vals) + r" \\")

    # Steps to solve
    vals = [val_or_dash(metrics_by_type[t]["solve_steps"], "mean", ".1f")
            for t in types]
    print(r"Steps to solve (mean) & " + " & ".join(vals) + r" \\")

    print(r"\midrule")

    # Violations
    for label, key in [("Collision (\\%)", "n_collision"),
                       ("Accel violation (\\%)", "n_accel_violated"),
                       ("Steer violation (\\%)", "n_steer_violated"),
                       ("Jerk violation (\\%)", "n_jerk_violated"),
                       ("Steer rate violation (\\%)", "n_srate_violated")]:
        vals = []
        for t in types:
            m = metrics_by_type[t]
            vals.append(pct(m[key], m["n_steps"]))
        print(f"{label} & " + " & ".join(vals) + r" \\")

    print(r"\midrule")

    # Cost
    vals = [val_or_dash(metrics_by_type[t]["cost"], "mean")
            for t in types]
    print(r"Ego cost (mean) & " + " & ".join(vals) + r" \\")

    # Action deviation
    vals = [val_or_dash(metrics_by_type[t]["dev_l2"], "mean")
            for t in types]
    print(r"Action deviation $L_2$ (mean) & " + " & ".join(vals) + r" \\")

    vals = [val_or_dash(metrics_by_type[t]["dev_accel"], "mean")
            for t in types]
    print(r"$\Delta a$ (mean) & " + " & ".join(vals) + r" \\")

    vals = [val_or_dash(metrics_by_type[t]["dev_steer"], "mean")
            for t in types]
    print(r"$\Delta \delta$ (mean) & " + " & ".join(vals) + r" \\")

    print(r"\midrule")

    # Wall time
    vals = [val_or_dash(metrics_by_type[t]["wall_time"], "mean", ".1f")
            for t in types]
    print(r"Wall time/ep (s) & " + " & ".join(vals) + r" \\")

    # Per-step total timing
    vals = []
    for t in types:
        tc = metrics_by_type[t].get("timing_per_step", {})
        if "total" in tc:
            vals.append(f"{tc['total']['mean']:.1f}")
        else:
            vals.append("--")
    print(r"Step time (ms, mean) & " + " & ".join(vals) + r" \\")

    print(r"\midrule")

    # Belief inference timing breakdown
    bi_keys = ['bi_relevance', 'bi_inference', 'bi_intervention', 'bi_plotting']
    bi_latex = [r'BI: Relevance (ms)', r'BI: Inference (ms)',
                r'BI: Intervention (ms)', r'BI: Plotting (ms)']
    for bk, bl in zip(bi_keys, bi_latex):
        vals = []
        for t in types:
            tc = metrics_by_type[t].get("timing_per_step", {})
            st = tc.get(bk)
            if st:
                vals.append(f"{st['mean']:.1f}")
            else:
                vals.append("--")
        print(f"{bl} & " + " & ".join(vals) + r" \\")

    # BI total
    vals = []
    for t in types:
        tc = metrics_by_type[t].get("timing_per_step", {})
        total = sum(tc.get(bk, {}).get("mean", 0) for bk in bi_keys)
        vals.append(f"{total:.1f}" if total > 0 else "--")
    print(r"BI: Total (ms) & " + " & ".join(vals) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# ── Significance testing ──────────────────────────────────────────────

def _significance_stars(p: float) -> str:
    """Convert p-value to star notation."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def _add_significance_brackets(ax, x_positions, raw_data_by_type, types):
    """Add significance brackets between all pairs of bars.

    Uses Mann-Whitney U test (non-parametric, no normality assumption).

    Args:
        ax: Matplotlib axes.
        x_positions: Array of x positions for each bar.
        raw_data_by_type: Dict mapping type key -> list of raw values.
        types: Ordered list of type keys.
    """
    from scipy.stats import mannwhitneyu

    # Collect pairs that have enough data
    pairs = []
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            a = raw_data_by_type.get(types[i], [])
            b = raw_data_by_type.get(types[j], [])
            if len(a) >= 3 and len(b) >= 3:
                pairs.append((i, j, a, b))

    if not pairs:
        return

    # Get current y-axis upper limit
    y_max = ax.get_ylim()[1]
    bracket_height = y_max * 0.05
    y_offset = y_max * 0.02

    for level, (i, j, a, b) in enumerate(pairs):
        try:
            _, p = mannwhitneyu(a, b, alternative='two-sided')
        except ValueError:
            continue

        stars = _significance_stars(p)

        # Position bracket above bars, stacking for multiple pairs
        y_bar = y_max + y_offset + level * (bracket_height + y_offset)
        x1, x2 = x_positions[i], x_positions[j]

        ax.plot([x1, x1, x2, x2],
                [y_bar, y_bar + bracket_height, y_bar + bracket_height, y_bar],
                color='black', linewidth=0.8)

        label = f'{stars}\np={p:.3f}' if p >= 0.001 else f'{stars}\np={p:.1e}'
        ax.text((x1 + x2) / 2, y_bar + bracket_height,
                label, ha='center', va='bottom', fontsize=6)

    # Expand y-axis to fit brackets
    n_levels = len(pairs)
    new_top = y_max + y_offset + n_levels * (bracket_height + y_offset) + y_max * 0.08
    ax.set_ylim(top=new_top)


# ── Plotting ─────────────────────────────────────────────────────────

def plot_comparison(metrics_by_type: dict, output: str = None):
    import matplotlib.pyplot as plt

    types = _sort_keys(metrics_by_type.keys())
    if not types:
        print("No data to plot.")
        return

    labels = [_group_label(t) for t in types]
    colours = [_group_colour(t, types) for t in types]
    x = np.arange(len(types))
    width = 0.6

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    # ── Panel 1: Outcome rates (stacked bar) ──
    ax = axes[0, 0]
    solved_pct = [100 * metrics_by_type[t]["n_solved"] /
                  max(metrics_by_type[t]["n_episodes"], 1) for t in types]
    failed_pct = [100 * metrics_by_type[t]["n_failed"] /
                  max(metrics_by_type[t]["n_episodes"], 1) for t in types]
    timeout_pct = [100 * metrics_by_type[t]["n_timeout"] /
                   max(metrics_by_type[t]["n_episodes"], 1) for t in types]

    ax.bar(x, solved_pct, width, label='Solved', color='#55A868')
    ax.bar(x, failed_pct, width, bottom=solved_pct, label='Failed',
           color='#C44E52')
    ax.bar(x, timeout_pct, width,
           bottom=[s + f for s, f in zip(solved_pct, failed_pct)],
           label='Timeout', color='#CCCCCC')
    ax.set_ylabel('Episodes (%)')
    ax.set_title('Outcome Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc='upper right')

    # ── Panel 2: Violation rates ──
    ax = axes[0, 1]
    viol_labels = ['Collision', 'Accel', 'Steer', 'Jerk', 'Steer\nrate']
    viol_keys = ['n_collision', 'n_accel_violated', 'n_steer_violated',
                 'n_jerk_violated', 'n_srate_violated']
    bar_w = width / len(types)
    for i, t in enumerate(types):
        m = metrics_by_type[t]
        n = max(m["n_steps"], 1)
        rates = [100 * m[k] / n for k in viol_keys]
        offset = (i - len(types) / 2 + 0.5) * bar_w
        ax.bar(np.arange(len(viol_labels)) + offset, rates, bar_w,
               label=_group_label(t),
               color=_group_colour(t, types))
    ax.set_ylabel('Steps with violation (%)')
    ax.set_title('Ego Violations')
    ax.set_xticks(np.arange(len(viol_labels)))
    ax.set_xticklabels(viol_labels, fontsize=8)
    ax.legend(fontsize=7, loc='upper right')

    # ── Panel 3: Per-step ego cost ──
    ax = axes[0, 2]
    means = [metrics_by_type[t]["cost"]["mean"]
             if metrics_by_type[t]["cost"] else 0 for t in types]
    stds = [metrics_by_type[t]["cost"]["std"]
            if metrics_by_type[t]["cost"] else 0 for t in types]
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean step cost')
    ax.set_title('Ego Step Cost')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, means):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    # _add_significance_brackets(ax, x, {t: metrics_by_type[t]["raw_cost"] for t in types}, types)

    # ── Panel 4: Action deviation (accel) ──
    ax = axes[1, 0]
    means = [metrics_by_type[t]["dev_accel"]["mean"]
             if metrics_by_type[t]["dev_accel"] else 0 for t in types]
    stds = [metrics_by_type[t]["dev_accel"]["std"]
            if metrics_by_type[t]["dev_accel"] else 0 for t in types]
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel(r'Mean $|\Delta a|$ (m/s²)')
    ax.set_title('Acceleration Deviation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, means):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    # _add_significance_brackets(ax, x, {t: metrics_by_type[t]["raw_dev_accel"] for t in types}, types)

    # ── Panel 5: Action deviation (steering) ──
    ax = axes[1, 1]
    means = [metrics_by_type[t]["dev_steer"]["mean"]
             if metrics_by_type[t]["dev_steer"] else 0 for t in types]
    stds = [metrics_by_type[t]["dev_steer"]["std"]
            if metrics_by_type[t]["dev_steer"] else 0 for t in types]
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel(r'Mean $|\Delta \delta|$ (rad)')
    ax.set_title('Steering Deviation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, means):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    # _add_significance_brackets(ax, x, {t: metrics_by_type[t]["raw_dev_steer"] for t in types}, types)

    # ── Panel 6: Steps to solve ──
    ax = axes[1, 2]
    means = [metrics_by_type[t]["solve_steps"]["mean"]
             if metrics_by_type[t]["solve_steps"] else 0 for t in types]
    stds = [metrics_by_type[t]["solve_steps"]["std"]
            if metrics_by_type[t]["solve_steps"] else 0 for t in types]
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Steps')
    ax.set_title('Steps to Solve (solved episodes)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, means):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    # _add_significance_brackets(ax, x, {t: metrics_by_type[t]["raw_solve_steps"] for t in types}, types)

    # ── Panel 7: Wall time per episode ──
    ax = axes[2, 0]
    means = [metrics_by_type[t]["wall_time"]["mean"]
             if metrics_by_type[t]["wall_time"] else 0 for t in types]
    stds = [metrics_by_type[t]["wall_time"]["std"]
            if metrics_by_type[t]["wall_time"] else 0 for t in types]
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Time (s)')
    ax.set_title('Wall Time per Episode')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, means):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1f}s', ha='center', va='bottom', fontsize=7)
    # _add_significance_brackets(ax, x, {t: metrics_by_type[t]["raw_wall_time"] for t in types}, types)

    # ── Panel 8: Per-step total timing ──
    ax = axes[2, 1]
    means = []
    stds = []
    for t in types:
        tc = metrics_by_type[t].get("timing_per_step", {})
        if "total" in tc:
            means.append(tc["total"]["mean"])
            stds.append(tc["total"]["std"])
        else:
            means.append(0)
            stds.append(0)
    bars = ax.bar(x, means, width, yerr=stds, capsize=4,
                  color=colours, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Time (ms)')
    ax.set_title('Mean Step Time')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    for bar, val in zip(bars, means):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.0f}ms', ha='center', va='bottom', fontsize=7)

    # ── Panel 9: Belief inference timing breakdown (stacked bar) ──
    ax = axes[2, 2]
    bi_keys = ['bi_relevance', 'bi_inference', 'bi_intervention', 'bi_plotting']
    bi_labels = ['Relevance', 'Inference', 'Intervention', 'Plotting']
    bi_colours = ['#4C72B0', '#DD8452', '#C44E52', '#CCCCCC']

    bottoms = np.zeros(len(types))
    for bk, bl, bc in zip(bi_keys, bi_labels, bi_colours):
        vals = []
        for t in types:
            tc = metrics_by_type[t].get("timing_per_step", {})
            vals.append(tc.get(bk, {}).get("mean", 0))
        vals = np.array(vals)
        ax.bar(x, vals, width, bottom=bottoms, label=bl, color=bc)
        bottoms += vals

    # Add total label on top
    for i, total in enumerate(bottoms):
        if total > 0:
            ax.text(x[i], total, f'{total:.0f}ms', ha='center',
                    va='bottom', fontsize=7)

    ax.set_ylabel('Time (ms)')
    ax.set_title('Belief Inference Timing')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, loc='upper right')

    fig.suptitle("Experiment Comparison (Inference / Intervention)",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output}")
    else:
        import matplotlib
        if matplotlib.get_backend().lower() != 'agg':
            plt.show(block=True)
        else:
            fig.savefig("comparison.pdf", dpi=150, bbox_inches='tight')
            print("Figure saved to comparison.pdf (non-interactive backend)")


# ── Main ─────────────────────────────────────────────────────────────

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

    # Group by inference_intervention composite key
    merged_episodes = defaultdict(list)
    for d in run_dirs:
        print(f"  {os.path.basename(d)}")
        key, episodes = load_run(d)
        if episodes is None:
            print(f"    Warning: could not load {d}")
            continue
        merged_episodes[key].extend(episodes)

    if not merged_episodes:
        print("No valid data found.")
        sys.exit(1)

    # Compute metrics
    metrics_by_type = {
        key: extract_metrics(eps)
        for key, eps in merged_episodes.items()
    }

    # Output
    print_text_summary(metrics_by_type)

    if args.latex:
        print_latex_table(metrics_by_type)

    plot_comparison(metrics_by_type, output=args.output)


if __name__ == "__main__":
    main()
