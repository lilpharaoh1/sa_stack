"""
Agency and comfort evaluation across intervention methods.

Focuses on **peak / worst-case** intervention magnitude (not just averages)
and ride comfort (jerk, steering rate).  Produces:

  1. Peak intervention magnitude (max, p95, p99 bar chart)
  2. Intervention magnitude CDF  (what fraction exceed threshold X?)
  3. Decomposed intervention peaks (accel vs steer)
  4. Jerk distribution (box / violin)
  5. Steering-rate distribution (box / violin)
  6. Per-episode worst-case summary
  7. Text summary with percentile-based statistics

Uses the same run-discovery and grouping infrastructure as plot_comparison.py.

Usage:
    python scripts/experiments/analysis/plot_agency.py -m belief_experiment4 --latest
    python scripts/experiments/analysis/plot_agency.py -m belief_experiment4 --latest -o agency.pdf
    python scripts/experiments/analysis/plot_agency.py dir1/ dir2/
"""

import sys
import os
import argparse
from collections import defaultdict

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", ".."))
sys.path.insert(0, _EXPERIMENTS_DIR)

# Reuse discovery / loading / labelling from plot_comparison
from analysis.plot_comparison import (
    discover_runs, keep_latest, load_run,
    _sort_keys, _group_label, _group_colour,
    _COLOUR_PALETTE,
)


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Agency & comfort evaluation across methods")
    p.add_argument("dirs", nargs="*",
                   help="Run directories to include (optional)")
    p.add_argument("-m", "--map", type=str, default=None,
                   help="Scenario name to auto-discover runs")
    p.add_argument("--latest", action="store_true",
                   help="Keep only the latest run per method")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Save figure to path instead of showing")
    p.add_argument("--latex", action="store_true",
                   help="Print a LaTeX table to stdout")
    p.add_argument("--bold", action="store_true",
                   help="Bold the best value in each row of the LaTeX table")
    return p.parse_args()


# ── Metric extraction ────────────────────────────────────────────────

_PERCENTILES = (50, 75, 90, 95, 99)


def _pstats(vals):
    """Compute mean/std/min/max + percentiles from a list of floats."""
    if not vals:
        return None
    a = np.array(vals)
    d = {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": len(vals),
    }
    for p in _PERCENTILES:
        d[f"p{p}"] = float(np.percentile(a, p))
    return d


def _recompute_comfort(episodes: list):
    """Recompute jerk and steer-rate from consecutive ego controls.

    Works even on older pickle files that lack the ``ego_jerk`` /
    ``ego_steer_rate`` fields by differencing ``ego_acceleration`` and
    ``ego_steer_angle`` across consecutive steps.
    """
    jerk_vals = []
    srate_vals = []
    ep_max_jerk = []
    ep_max_srate = []

    for r in episodes:
        dt = 1.0 / r.fps if r.fps > 0 else 0.05
        ep_jerks = []
        ep_srates = []
        prev_a = None
        prev_d = None
        for s in r.steps:
            # Try stored field first
            j = getattr(s, 'ego_jerk', None)
            sr = getattr(s, 'ego_steer_rate', None)

            # Recompute from raw controls if not stored
            a_now = getattr(s, 'ego_acceleration', None)
            d_now = getattr(s, 'ego_steer_angle', None)
            if j is None and a_now is not None and prev_a is not None:
                j = abs(a_now - prev_a) / dt
            if sr is None and d_now is not None and prev_d is not None:
                sr = abs(d_now - prev_d) / dt

            if j is not None:
                jerk_vals.append(j)
                ep_jerks.append(j)
            if sr is not None:
                srate_vals.append(sr)
                ep_srates.append(sr)

            prev_a = a_now
            prev_d = d_now

        if ep_jerks:
            ep_max_jerk.append(max(ep_jerks))
        if ep_srates:
            ep_max_srate.append(max(ep_srates))

    return jerk_vals, srate_vals, ep_max_jerk, ep_max_srate


def extract_agency_metrics(episodes: list) -> dict:
    """Extract agency and comfort metrics from a list of episodes."""
    all_steps = [s for r in episodes for s in r.steps]

    # -- Action deviation (all timesteps) --
    dev_l2 = [s.action_deviation for s in all_steps
              if s.action_deviation is not None]
    dev_accel = [s.action_deviation_accel for s in all_steps
                 if s.action_deviation_accel is not None]
    dev_steer = [s.action_deviation_steer for s in all_steps
                 if s.action_deviation_steer is not None]

    # -- Comfort: jerk and steering rate (with fallback recompute) --
    jerk_vals, srate_vals, ep_max_jerk, ep_max_srate = \
        _recompute_comfort(episodes)

    # -- Per-episode worst-case deviation --
    ep_max_dev = []
    for r in episodes:
        devs = [s.action_deviation for s in r.steps
                if s.action_deviation is not None]
        if devs:
            ep_max_dev.append(max(devs))

    # Intervention rate
    n_interv = sum(1 for s in all_steps if s.intervention_active)
    n_total = len(all_steps)

    return {
        "n_episodes": len(episodes),
        "n_steps": n_total,
        "n_interv": n_interv,
        "interv_rate": n_interv / n_total if n_total > 0 else 0.0,
        # All steps
        "dev_l2": _pstats(dev_l2),
        "dev_accel": _pstats(dev_accel),
        "dev_steer": _pstats(dev_steer),
        # Comfort
        "jerk": _pstats(jerk_vals),
        "steer_rate": _pstats(srate_vals),
        # Per-episode worst case
        "ep_max_dev": _pstats(ep_max_dev),
        "ep_max_jerk": _pstats(ep_max_jerk),
        "ep_max_srate": _pstats(ep_max_srate),
        # Raw arrays for CDF / box plots
        "raw_dev_l2": dev_l2,
        "raw_dev_accel": dev_accel,
        "raw_dev_steer": dev_steer,
        "raw_jerk": jerk_vals,
        "raw_srate": srate_vals,
        "raw_ep_max_dev": ep_max_dev,
        "raw_ep_max_jerk": ep_max_jerk,
        "raw_ep_max_srate": ep_max_srate,
    }


# ── Text summary ─────────────────────────────────────────────────────

def _fmt(stats, key, fmt_str=".4f"):
    if stats is None:
        return "--"
    return f"{stats[key]:{fmt_str}}"


def print_text_summary(metrics_by_type: dict):
    print(f"\n{'='*78}")
    print(f"  Agency & Comfort Evaluation")
    print(f"{'='*78}")

    for key in _sort_keys(metrics_by_type.keys()):
        m = metrics_by_type[key]
        label = _group_label(key)
        print(f"\n  {label}  ({m['n_episodes']} episodes, {m['n_steps']} steps)")
        print(f"  {'─'*68}")

        # Intervention rate
        print(f"    Intervention rate: {m['n_interv']}/{m['n_steps']} "
              f"({100*m['interv_rate']:.1f}%)")

        # Action deviation (all timesteps)
        for name, skey in [("L2", "dev_l2"),
                           ("Accel |da|", "dev_accel"),
                           ("Steer |dd|", "dev_steer")]:
            s = m[skey]
            if s is None:
                continue
            print(f"    {name} (n={s['n']}):")
            print(f"      mean={s['mean']:.4f}  std={s['std']:.4f}  "
                  f"max={s['max']:.4f}")
            pstr = "  ".join(f"p{p}={s[f'p{p}']:.4f}" for p in _PERCENTILES)
            print(f"      {pstr}")

        # Per-episode worst case
        s = m["ep_max_dev"]
        if s is not None:
            print(f"    Per-episode max deviation (n={s['n']} episodes):")
            print(f"      mean-of-max={s['mean']:.4f}  "
                  f"worst-case={s['max']:.4f}  "
                  f"p95={s['p95']:.4f}")

        # Comfort: jerk
        s = m["jerk"]
        if s is not None:
            print(f"    Jerk |da/dt| (m/s^3, n={s['n']}):")
            print(f"      mean={s['mean']:.2f}  std={s['std']:.2f}  "
                  f"max={s['max']:.2f}")
            pstr = "  ".join(f"p{p}={s[f'p{p}']:.2f}" for p in _PERCENTILES)
            print(f"      {pstr}")

        # Comfort: steer rate
        s = m["steer_rate"]
        if s is not None:
            print(f"    Steer rate |dd/dt| (rad/s, n={s['n']}):")
            print(f"      mean={s['mean']:.3f}  std={s['std']:.3f}  "
                  f"max={s['max']:.3f}")
            pstr = "  ".join(f"p{p}={s[f'p{p}']:.3f}" for p in _PERCENTILES)
            print(f"      {pstr}")

    print(f"\n{'='*78}\n")


# ── LaTeX table ──────────────────────────────────────────────────────

def print_latex_table(metrics_by_type: dict, bold: bool = False):
    types = _sort_keys(metrics_by_type.keys())
    if not types:
        return

    _none_mask = [t.startswith("none_none") for t in types]

    def _bold_best(vals, lower_is_better=True):
        nums = []
        for v in vals:
            try:
                nums.append(float(v))
            except (ValueError, TypeError):
                nums.append(None)
        valid = [n for n, is_none in zip(nums, _none_mask)
                 if n is not None and not is_none]
        if not valid:
            return vals
        best = min(valid) if lower_is_better else max(valid)
        out = []
        for v, n, is_none in zip(vals, nums, _none_mask):
            if not is_none and n is not None and abs(n - best) < 1e-9:
                out.append(r"\textbf{" + v + "}")
            else:
                out.append(v)
        return out

    def _print_row(label, vals, lower_is_better=True):
        if bold:
            vals = _bold_best(vals, lower_is_better)
        print(f"{label} & " + " & ".join(vals) + r" \\")

    headers = " & ".join(_group_label(t) for t in types)
    cols = "l" + "c" * len(types)

    print()
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Agency and comfort metrics across methods.}")
    print(r"\label{tab:agency}")
    print(r"\begin{tabular}{" + cols + r"}")
    print(r"\toprule")
    print(r"Metric & " + headers + r" \\")
    print(r"\midrule")

    # Intervention rate
    vals = [f"{100*metrics_by_type[t]['interv_rate']:.1f}"
            for t in types]
    _print_row(r"Intervention rate (\%)", vals, lower_is_better=True)

    print(r"\midrule")

    # Deviation stats (all timesteps)
    for label, key, fmt in [
        (r"$\|\Delta u\|_2$ mean", "dev_l2", ".4f"),
        (r"$\|\Delta u\|_2$ p95", "dev_l2", ".4f"),
        (r"$\|\Delta u\|_2$ max", "dev_l2", ".4f"),
        (r"$|\Delta a|$ mean", "dev_accel", ".4f"),
        (r"$|\Delta a|$ p95", "dev_accel", ".4f"),
        (r"$|\Delta a|$ max", "dev_accel", ".4f"),
        (r"$|\Delta \delta|$ mean", "dev_steer", ".4f"),
        (r"$|\Delta \delta|$ p95", "dev_steer", ".4f"),
        (r"$|\Delta \delta|$ max", "dev_steer", ".4f"),
    ]:
        stat_key = "mean"
        if "p95" in label:
            stat_key = "p95"
        elif "max" in label:
            stat_key = "max"
        vals = [_fmt(metrics_by_type[t][key], stat_key, fmt) for t in types]
        _print_row(label, vals, lower_is_better=True)

    print(r"\midrule")

    # Comfort
    for label, key, stat_key, fmt in [
        (r"Jerk mean (m/s$^3$)", "jerk", "mean", ".2f"),
        (r"Jerk p95", "jerk", "p95", ".2f"),
        (r"Jerk max", "jerk", "max", ".2f"),
        (r"Steer rate mean (rad/s)", "steer_rate", "mean", ".3f"),
        (r"Steer rate p95", "steer_rate", "p95", ".3f"),
        (r"Steer rate max", "steer_rate", "max", ".3f"),
    ]:
        vals = [_fmt(metrics_by_type[t][key], stat_key, fmt) for t in types]
        _print_row(label, vals, lower_is_better=True)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()


# ── Plotting ─────────────────────────────────────────────────────────

def plot_agency(metrics_by_type: dict, output: str = None):
    import matplotlib
    import matplotlib.pyplot as plt

    types = _sort_keys(metrics_by_type.keys())
    if not types:
        print("No data to plot.")
        return

    labels = [_group_label(t) for t in types]
    colours = [_group_colour(t, types) for t in types]
    x = np.arange(len(types))
    width = 0.6

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # ── Panel 1: Peak action deviation (bar chart) ──
    ax = axes[0, 0]
    bar_w = width / 4
    for j, (stat_key, stat_label) in enumerate([
        ("mean", "mean"), ("p95", "p95"), ("p99", "p99"), ("max", "max")
    ]):
        vals = []
        for t in types:
            s = metrics_by_type[t]["dev_l2"]
            vals.append(s[stat_key] if s else 0)
        offset = (j - 1.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=stat_label,
                      color=plt.cm.Blues(0.3 + 0.175 * j), edgecolor='black',
                      linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(), f'{val:.3f}',
                        ha='center', va='bottom', fontsize=6)
    ax.set_ylabel(r'$\|\Delta u\|_2$')
    ax.set_title('Peak Action Deviation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, title='Statistic', title_fontsize=7)

    # ── Panel 2: Action deviation CDF ──
    ax = axes[0, 1]
    for i, t in enumerate(types):
        vals = metrics_by_type[t]["raw_dev_l2"]
        if not vals:
            continue
        sorted_v = np.sort(vals)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.step(sorted_v, cdf, where='post', label=_group_label(t),
                color=colours[i], linewidth=1.5)
    ax.set_xlabel(r'$\|\Delta u\|_2$')
    ax.set_ylabel('CDF')
    ax.set_title('Action Deviation CDF')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Decomposed peak deviation (accel vs steer) ──
    ax = axes[0, 2]
    bar_w = width / len(types)
    metric_labels = [r'$|\Delta a|$' + '\np95',
                     r'$|\Delta a|$' + '\nmax',
                     r'$|\Delta\delta|$' + '\np95',
                     r'$|\Delta\delta|$' + '\nmax']
    metric_keys = [("dev_accel", "p95"),
                   ("dev_accel", "max"),
                   ("dev_steer", "p95"),
                   ("dev_steer", "max")]
    x_m = np.arange(len(metric_labels))
    for i, t in enumerate(types):
        vals = []
        for mkey, skey in metric_keys:
            s = metrics_by_type[t][mkey]
            vals.append(s[skey] if s else 0)
        offset = (i - len(types) / 2 + 0.5) * bar_w
        ax.bar(x_m + offset, vals, bar_w, label=_group_label(t),
               color=colours[i], edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Magnitude')
    ax.set_title('Decomposed Deviation Peaks\n(accel vs steering)')
    ax.set_xticks(x_m)
    ax.set_xticklabels(metric_labels, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, loc='upper right')

    # ── Panel 4: Jerk distribution (box + strip) ──
    ax = axes[1, 0]
    jerk_data = []
    jerk_labels_plot = []
    jerk_colours_plot = []
    for i, t in enumerate(types):
        vals = metrics_by_type[t]["raw_jerk"]
        if vals:
            jerk_data.append(vals)
            jerk_labels_plot.append(_group_label(t))
            jerk_colours_plot.append(colours[i])
    if jerk_data:
        bp = ax.boxplot(jerk_data, patch_artist=True, showfliers=False,
                        widths=0.5, medianprops=dict(color='black', linewidth=1.5))
        for patch, c in zip(bp['boxes'], jerk_colours_plot):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticklabels(jerk_labels_plot, rotation=30, ha='right',
                           fontsize=8)
    ax.set_ylabel(r'Jerk $|da/dt|$ (m/s$^3$)')
    ax.set_title('Jerk Distribution')

    # ── Panel 5: Steering rate distribution (box + strip) ──
    ax = axes[1, 1]
    srate_data = []
    srate_labels_plot = []
    srate_colours_plot = []
    for i, t in enumerate(types):
        vals = metrics_by_type[t]["raw_srate"]
        if vals:
            srate_data.append(vals)
            srate_labels_plot.append(_group_label(t))
            srate_colours_plot.append(colours[i])
    if srate_data:
        bp = ax.boxplot(srate_data, patch_artist=True, showfliers=False,
                        widths=0.5, medianprops=dict(color='black', linewidth=1.5))
        for patch, c in zip(bp['boxes'], srate_colours_plot):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticklabels(srate_labels_plot, rotation=30, ha='right',
                           fontsize=8)
    ax.set_ylabel(r'Steer rate $|d\delta/dt|$ (rad/s)')
    ax.set_title('Steering Rate Distribution')

    # ── Panel 6: Per-episode worst-case deviation ──
    ax = axes[1, 2]
    ep_data = []
    ep_labels_plot = []
    ep_colours_plot = []
    for i, t in enumerate(types):
        vals = metrics_by_type[t]["raw_ep_max_dev"]
        if vals:
            ep_data.append(vals)
            ep_labels_plot.append(_group_label(t))
            ep_colours_plot.append(colours[i])
    if ep_data:
        bp = ax.boxplot(ep_data, patch_artist=True, showfliers=True,
                        widths=0.5, medianprops=dict(color='black', linewidth=1.5))
        for patch, c in zip(bp['boxes'], ep_colours_plot):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        # Label medians and p95
        for i, vals in enumerate(ep_data):
            a = np.array(vals)
            med = np.median(a)
            p95 = np.percentile(a, 95)
            ax.text(i + 1, med, f'med={med:.3f}', ha='left', va='bottom',
                    fontsize=6, color='black')
            ax.text(i + 1, p95, f'p95={p95:.3f}', ha='left', va='bottom',
                    fontsize=6, color='red')
        ax.set_xticklabels(ep_labels_plot, rotation=30, ha='right',
                           fontsize=8)
    ax.set_ylabel(r'Max $\|\Delta u\|_2$ per episode')
    ax.set_title('Per-Episode Worst-Case Intervention')

    fig.suptitle("Agency & Comfort Evaluation",
                 fontsize=14, y=1.01)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output}")
    else:
        if matplotlib.get_backend().lower() != 'agg':
            plt.show(block=True)
        else:
            fig.savefig("agency_comfort.pdf", dpi=150, bbox_inches='tight')
            print("Figure saved to agency_comfort.pdf (non-interactive backend)")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    from utils import RESULTS_DIR

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

    metrics_by_type = {
        key: extract_agency_metrics(eps)
        for key, eps in merged_episodes.items()
    }

    print_text_summary(metrics_by_type)

    if args.latex:
        print_latex_table(metrics_by_type, bold=args.bold)

    plot_agency(metrics_by_type, output=args.output)


if __name__ == "__main__":
    main()
