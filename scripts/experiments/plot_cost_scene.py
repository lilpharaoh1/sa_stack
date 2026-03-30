"""
Visualise per-step ego cost components on the road layout.

Produces two views:
  1. **Scene panels** — one per run, ego trajectory on the road coloured by
     the selected cost component magnitude.
  2. **Along-track panel** — cost component vs arc-length s for all runs
     overlaid, showing where each method accumulates cost.

Supports comparing across runs (auto-discovery or explicit dirs).

Usage:
    # Compare latest runs, coloured by total cost:
    python scripts/experiments/plot_cost_scene.py -m belief_experiment4 --latest

    # Colour by acceleration cost only:
    python scripts/experiments/plot_cost_scene.py -m belief_experiment4 --latest -c accel

    # Show all components in separate rows:
    python scripts/experiments/plot_cost_scene.py -m belief_experiment4 --latest --all-components

    # Specific directories:
    python scripts/experiments/plot_cost_scene.py dir1/ dir2/ -c speed
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

import igp2 as ip
from igp2.opendrive.plot_map import plot_map
from igp2.core.util import calculate_multiple_bboxes
from belief_utils import ExperimentResult, RESULTS_DIR
from plot_comparison import (
    _composite_key, _sort_keys, _group_label,
    INFERENCE_LABELS, INTERVENTION_LABELS,
)

# ── NLP cost weights (SecondStagePlanner.DEFAULTS) ────────────────────
_W = {'w_d': 10.0, 'w_v': 0.01, 'w_a': 1.0, 'w_delta': 2.0,
      'w_phi': 2.0, 'v_target': 10.0}

# ── Component definitions ────────────────────────────────────────────
COMPONENTS = {
    'total':    {'label': 'Total cost',               'cmap': 'inferno'},
    'lateral':  {'label': r'Lateral ($w_d \cdot d^2$)',  'cmap': 'YlOrRd'},
    'speed':    {'label': r'Speed ($w_v \cdot \Delta v^2$)', 'cmap': 'YlGnBu'},
    'accel':    {'label': r'Accel ($w_a \cdot a^2$)',    'cmap': 'Oranges'},
    'steering': {'label': r'Steering ($w_\delta \cdot \delta^2$)', 'cmap': 'Purples'},
    'heading':  {'label': r'Heading ($w_\phi \cdot \phi^2$)',  'cmap': 'Greens'},
}
COMPONENT_ORDER = ['total', 'lateral', 'speed', 'accel', 'steering', 'heading']



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


def load_run(run_dir: str):
    """Return (meta, result) for the first episode."""
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
        if results:
            return meta, results[0]
    return meta, None


# ── Cost computation ─────────────────────────────────────────────────

def compute_step_costs(steps):
    """Return dict of arrays: {component: (N,)} plus 'xy' and 's'.

    Recomputes from raw state so it works on old pickle files.
    """
    xy = []
    s_vals = []
    costs = {k: [] for k in COMPONENT_ORDER}

    for sr in steps:
        if sr.ego_position is None or sr.ego_frenet_state is None:
            continue
        s, d, phi, v = sr.ego_frenet_state[:4]
        a = sr.ego_acceleration if sr.ego_acceleration is not None else 0.0
        delta = sr.ego_steer_angle if sr.ego_steer_angle is not None else 0.0

        c_lat = _W['w_d'] * d ** 2
        c_spd = _W['w_v'] * (v - _W['v_target']) ** 2
        c_acc = _W['w_a'] * a ** 2
        c_str = _W['w_delta'] * delta ** 2
        c_hea = _W['w_phi'] * phi ** 2

        costs['lateral'].append(c_lat)
        costs['speed'].append(c_spd)
        costs['accel'].append(c_acc)
        costs['steering'].append(c_str)
        costs['heading'].append(c_hea)
        costs['total'].append(c_lat + c_spd + c_acc + c_str + c_hea)

        xy.append(sr.ego_position[:2])
        s_vals.append(s)

    out = {k: np.array(v) for k, v in costs.items()}
    out['xy'] = np.array(xy) if xy else np.empty((0, 2))
    out['s'] = np.array(s_vals)
    return out


# ── Scene rendering ──────────────────────────────────────────────────

def _draw_static_obstacles(ax, steps):
    """Draw static obstacles from the last step."""
    last_sr = steps[-1]
    static_obs = getattr(last_sr, 'static_obstacles', {})
    for aid, state in static_obs.items():
        meta = getattr(state, 'metadata', None)
        vl = meta.length if meta else 4.5
        vw = meta.width if meta else 1.8
        corners = calculate_multiple_bboxes(
            [state.position[0]], [state.position[1]],
            vl, vw, state.heading)[0]
        from matplotlib.patches import Polygon as MplPolygon
        poly = MplPolygon(corners, closed=True,
                          facecolor=(0.6, 0.6, 0.6, 0.5),
                          edgecolor=(0.6, 0.6, 0.6, 0.9),
                          linewidth=1.0, zorder=4)
        ax.add_patch(poly)


def _draw_dynamic_agents(ax, steps, point_size):
    """Draw non-ego agent trajectories in muted gold."""
    agent_xy = defaultdict(list)
    for sr in steps:
        for aid, state in sr.dynamic_agents.items():
            agent_xy[aid].append(state.position[:2])
    for aid, pts in agent_xy.items():
        pts = np.array(pts)
        ax.plot(pts[:, 0], pts[:, 1], 'o',
                color=(0.85, 0.75, 0.3, 0.4),
                markersize=point_size * 0.5, zorder=3)


def plot_scene_panel(ax, scenario_map, steps, cost_vals, xy,
                     cmap_name, vmin, vmax, point_size, title):
    """Render one scene panel: road + ego trajectory coloured by cost."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    plot_map(scenario_map, ax=ax, markings=True,
             junction_color=(0, 0, 0, 0))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    _draw_static_obstacles(ax, steps)
    _draw_dynamic_agents(ax, steps, point_size)

    if len(xy) > 0:
        cmap = plt.get_cmap(cmap_name)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colours = cmap(norm(cost_vals))
        ax.scatter(xy[:, 0], xy[:, 1], c=colours,
                   s=point_size ** 2, zorder=5, edgecolors='none')

    ax.set_title(title, fontsize=9)

    # Auto-fit view
    if len(xy) > 0:
        pad = 15.0
        ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)


# ── Main figure builders ─────────────────────────────────────────────

MAX_COLS = 4


def plot_single_component(scenario_map, run_data, component, output,
                          point_size, figsize_per_panel):
    """Scene panels for a single cost component, max 4 per row."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    n_runs = len(run_data)
    n_cols = min(n_runs, MAX_COLS)
    n_rows = (n_runs + n_cols - 1) // n_cols

    fig_w = figsize_per_panel[0] * n_cols + 1.5
    fig_h = figsize_per_panel[1] * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             squeeze=False)

    info = COMPONENTS[component]

    # Shared colour scale across all runs
    all_vals = np.concatenate([d['costs'][component]
                               for d in run_data
                               if len(d['costs'][component]) > 0])
    if len(all_vals) == 0:
        print("No cost data found.")
        return
    vmax = np.percentile(all_vals, 98)
    vmin = 0.0

    used_axes = []
    for i, rd in enumerate(run_data):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        used_axes.append(ax)
        plot_scene_panel(
            ax, scenario_map, rd['steps'],
            rd['costs'][component], rd['costs']['xy'],
            info['cmap'], vmin, vmax, point_size,
            title=rd['label'])

    # Hide unused axes
    for i in range(n_runs, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].set_visible(False)

    # Colourbar
    cmap = plt.get_cmap(info['cmap'])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8,
                        pad=0.02, aspect=30)
    cbar.set_label(info['label'], fontsize=8)

    fig.suptitle(f'Ego Cost: {info["label"]}', fontsize=12, y=1.01)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved to {output}")
    else:
        plt.show(block=True)


def plot_all_components(scenario_map, run_data, output,
                        point_size, figsize_per_panel):
    """Grid: rows = components * ceil(n_runs/4), columns = min(n_runs, 4)."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    n_runs = len(run_data)
    n_cols = min(n_runs, MAX_COLS)
    run_rows = (n_runs + n_cols - 1) // n_cols  # rows per component
    n_comp = len(COMPONENT_ORDER)
    total_rows = n_comp * run_rows

    fig_w = figsize_per_panel[0] * n_cols + 1.5
    fig_h = figsize_per_panel[1] * total_rows

    fig, axes = plt.subplots(total_rows, n_cols,
                             figsize=(fig_w, fig_h), squeeze=False)

    for ci, comp in enumerate(COMPONENT_ORDER):
        info = COMPONENTS[comp]

        # Shared colour scale for this component
        all_vals = np.concatenate([d['costs'][comp]
                                   for d in run_data
                                   if len(d['costs'][comp]) > 0])
        if len(all_vals) == 0:
            continue
        vmax = np.percentile(all_vals, 98)
        vmin = 0.0

        base_row = ci * run_rows
        comp_axes = []
        for j, rd in enumerate(run_data):
            r, c = divmod(j, n_cols)
            ax = axes[base_row + r, c]
            comp_axes.append(ax)
            # Title on first row of first component only
            title = rd['label'] if ci == 0 and r == 0 else ''
            plot_scene_panel(
                ax, scenario_map, rd['steps'],
                rd['costs'][comp], rd['costs']['xy'],
                info['cmap'], vmin, vmax, point_size,
                title=title)
            if c == 0 and r == 0:
                ax.set_ylabel(info['label'], fontsize=7, rotation=90,
                              labelpad=8)

        # Hide unused axes in this component's rows
        for j in range(n_runs, run_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[base_row + r, c].set_visible(False)

        # Colourbar per component
        cmap = plt.get_cmap(info['cmap'])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        row_axes = [axes[base_row + r, c]
                    for r in range(run_rows) for c in range(n_cols)]
        fig.colorbar(sm, ax=row_axes, shrink=0.7, pad=0.02, aspect=20)

    fig.suptitle('Ego Cost Component Breakdown', fontsize=13, y=1.005)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved to {output}")
    else:
        plt.show(block=True)


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise ego cost components on the road layout")
    p.add_argument("dirs", nargs="*",
                   help="Run directories to include (optional)")
    p.add_argument("-m", "--map", type=str, default=None,
                   help="Scenario name to auto-discover runs")
    p.add_argument("--latest", action="store_true",
                   help="Keep only the latest run per type")
    p.add_argument("-c", "--component", type=str, default="total",
                   choices=COMPONENT_ORDER,
                   help="Cost component to visualise (default: total)")
    p.add_argument("--all-components", action="store_true",
                   help="Show all components in a grid")
    p.add_argument("-e", "--episode", type=int, default=0,
                   help="Episode index for batch results (default: 0)")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Save figure to file instead of showing")
    p.add_argument("--point-size", type=float, default=6.0,
                   help="Marker size for trajectory points (default: 6)")
    p.add_argument("--panel-size", type=float, nargs=2, default=[6, 5],
                   metavar=("W", "H"),
                   help="Size per scene panel in inches (default: 6 5)")
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

    # Load runs
    run_data = []
    scenario_map = None
    for d in run_dirs:
        print(f"  {os.path.basename(d)}")
        meta, result = load_run(d)
        if result is None or not result.steps:
            print(f"    Warning: no steps in {d}")
            continue

        # Use first available map
        if scenario_map is None:
            map_path = result.config.get("scenario", {}).get("map_path")
            if map_path is not None:
                scenario_map = ip.Map.parse_from_opendrive(map_path)

        costs = compute_step_costs(result.steps)
        key = _composite_key(meta)
        label = _group_label(key)

        run_data.append({
            'meta': meta,
            'key': key,
            'label': label,
            'steps': result.steps,
            'costs': costs,
            'result': result,
        })

        n = len(costs['total'])
        if n > 0:
            print(f"    {label}: {n} steps, "
                  f"total_cost mean={costs['total'].mean():.4f}")

    if not run_data:
        print("No valid data found.")
        sys.exit(1)

    # Sort runs in same order as plot_comparison
    sorted_keys = _sort_keys([rd['key'] for rd in run_data])
    key_order = {k: i for i, k in enumerate(sorted_keys)}
    run_data.sort(key=lambda rd: key_order.get(rd['key'], 99))

    if scenario_map is None:
        print("No map_path found in any result config.")
        sys.exit(1)

    # Print text summary
    print(f"\n{'='*65}")
    print(f"  Cost Component Summary")
    print(f"{'='*65}")
    for rd in run_data:
        c = rd['costs']
        n = len(c['total'])
        if n == 0:
            continue
        print(f"\n  {rd['label']}  ({n} steps)")
        print(f"  {'─'*55}")
        total_mean = c['total'].mean()
        for comp in COMPONENT_ORDER:
            vals = c[comp]
            mean = vals.mean()
            frac = 100.0 * mean / total_mean if total_mean > 0 else 0.0
            info = COMPONENTS[comp]
            print(f"    {info['label']:40s}  "
                  f"mean={mean:.4f}  ({frac:5.1f}%)")
    print()

    # Plot
    if args.all_components:
        plot_all_components(scenario_map, run_data, args.output,
                            args.point_size, tuple(args.panel_size))
    else:
        plot_single_component(scenario_map, run_data, args.component,
                              args.output, args.point_size,
                              tuple(args.panel_size))


if __name__ == "__main__":
    main()
