"""
Visualise the dual-RBF awareness kernel on a road layout.

For a given ego position and heading, computes the world-frame feature value
at every point in a grid.  Overlays the result as a heatmap on the road map.

The feature used matches what the Kalman awareness filter actually computes:
  - RBF 1: omnidirectional proximity (narrow spread sigma_1)
  - RBF 2: forward field-of-view gated (wider spread sigma_2)

Usage:
    # Single ego pose on a scenario map:
    python scripts/experiments/plot_attention_kernel.py -m belief_experiment4 \
        --ego-x 50 --ego-y -2 --ego-heading 0

    # From a results file (uses ego pose at a given step):
    python scripts/experiments/plot_attention_kernel.py --from-result results/my_run/ \
        --step 10

    # Adjust kernel parameters:
    python scripts/experiments/plot_attention_kernel.py -m belief_experiment4 \
        --ego-x 50 --ego-y -2 --ego-heading 0 --sigma1 10 --sigma2 30
"""

import sys
import os
import argparse
import json
import math

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import igp2 as ip
from igp2.opendrive.plot_map import plot_map
from igp2.beliefcontrol.kalman_awareness import compute_feature_world


def plot_attention_kernel(
        scenario_map,
        ego_x: float,
        ego_y: float,
        ego_heading: float = 0.0,
        sigma_1: float = 15.0,
        sigma_2: float = 25.0,
        fov_half_angle: float = math.radians(30),
        w1: float = 0.7,
        w2: float = 0.3,
        grid_res: float = 0.5,
        extent: float = 50.0,
        figsize=(18, 6),
        participant_positions=None,
):
    """Plot the world-frame awareness kernel as a heatmap on the road map.

    Args:
        scenario_map: Parsed road map.
        ego_x, ego_y: Ego world position.
        ego_heading: Ego heading (radians).
        sigma_1: Narrow RBF spread (omnidirectional).
        sigma_2: Wide RBF spread (forward-gated).
        fov_half_angle: Half FOV angle (radians).
        w1: Weight for omnidirectional RBF.
        w2: Weight for forward RBF.
        grid_res: Grid resolution in metres.
        extent: How far from ego to evaluate (metres).
        figsize: Figure size.
        participant_positions: Optional list of (x, y) to mark on the plot.

    Returns:
        (fig, axes) tuple.
    """
    ego_xy = np.array([ego_x, ego_y], dtype=float)

    # Build evaluation grid (world coordinates)
    xs = np.arange(ego_x - extent, ego_x + extent + grid_res, grid_res)
    ys = np.arange(ego_y - extent, ego_y + extent + grid_res, grid_res)
    XX, YY = np.meshgrid(xs, ys)

    # Compute feature components at each grid point
    F_total = np.zeros_like(XX)
    F_rbf1 = np.zeros_like(XX)
    F_rbf2 = np.zeros_like(XX)

    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            p_xy = np.array([XX[i, j], YY[i, j]], dtype=float)
            diff = p_xy - ego_xy
            dist_sq = float(np.dot(diff, diff))

            rbf1 = math.exp(-dist_sq / (2.0 * sigma_1 ** 2))

            rbf2 = math.exp(-dist_sq / (2.0 * sigma_2 ** 2))
            angle_to = math.atan2(diff[1], diff[0])
            rel_angle = (angle_to - ego_heading + math.pi) % (2.0 * math.pi) - math.pi
            if abs(rel_angle) > fov_half_angle:
                rbf2 = 0.0

            F_rbf1[i, j] = w1 * rbf1
            F_rbf2[i, j] = w2 * rbf2
            F_total[i, j] = w1 * rbf1 + w2 * rbf2

    # --- Figure: 3 panels ---
    fig, axes = plt.subplots(1, 3, figsize=(figsize[0], figsize[1]))

    fov_deg = math.degrees(fov_half_angle)
    panels = [
        (F_rbf1,
         f'w\u2081\u00b7RBF\u2081: Omnidirectional '
         f'(w\u2081={w1:.1f}, \u03c3\u2081={sigma_1:.1f}m)'),
        (F_rbf2,
         f'w\u2082\u00b7RBF\u2082: Forward FOV \u00b1{fov_deg:.0f}\u00b0 '
         f'(w\u2082={w2:.1f}, \u03c3\u2082={sigma_2:.1f}m)'),
        (F_total,
         f'Combined f(s) = {w1:.1f}\u00b7RBF\u2081 + {w2:.1f}\u00b7RBF\u2082'),
    ]

    # FOV cone boundary lines in world coordinates
    cone_len = extent * 0.7
    cone_lines = []
    for sign in [-1, 1]:
        angle = ego_heading + sign * fov_half_angle
        ex = ego_x + cone_len * math.cos(angle)
        ey = ego_y + cone_len * math.sin(angle)
        cone_lines.append(([ego_x, ex], [ego_y, ey]))

    for ax, (F, title) in zip(axes, panels):
        plot_map(scenario_map, ax=ax, markings=True,
                 junction_color=(0, 0, 0, 0))
        ax.set_aspect('equal')

        # Heatmap
        vmax = max(F.max(), 1e-6)
        im = ax.pcolormesh(XX, YY, F, cmap='YlOrRd', alpha=0.6,
                           shading='auto', vmin=0, vmax=vmax, zorder=3)
        fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

        # Ego marker + heading arrow
        ax.plot(ego_x, ego_y, 'ko', markersize=8, zorder=10)
        arrow_len = 4.0
        dx = arrow_len * math.cos(ego_heading)
        dy = arrow_len * math.sin(ego_heading)
        ax.annotate('', xy=(ego_x + dx, ego_y + dy),
                    xytext=(ego_x, ego_y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2),
                    zorder=10)

        # FOV cone edges
        for ci, (xs_c, ys_c) in enumerate(cone_lines):
            ax.plot(xs_c, ys_c,
                    'k--', linewidth=1.0, alpha=0.5, zorder=9,
                    label=f'FOV \u00b1{fov_deg:.0f}\u00b0' if ci == 0 else None)

        # Participant markers
        if participant_positions:
            for px, py in participant_positions:
                p_xy = np.array([px, py], dtype=float)
                f_val = compute_feature_world(
                    ego_xy, ego_heading, p_xy,
                    sigma_1, sigma_2, fov_half_angle, w1, w2)
                ax.plot(px, py, 's', color='blue', markersize=10,
                        markeredgecolor='black', zorder=11)
                ax.text(px + 1.5, py + 1.5, f'f={f_val:.3f}',
                        fontsize=7, zorder=11,
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.8))

        # View bounds
        pad = 5.0
        ax.set_xlim(ego_x - extent - pad, ego_x + extent + pad)
        ax.set_ylim(ego_y - extent - pad, ego_y + extent + pad)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)

    fig.suptitle(
        f'Awareness Kernel (World)  |  '
        f'ego=({ego_x:.1f}, {ego_y:.1f}), '
        f'heading={math.degrees(ego_heading):.1f}\u00b0',
        fontsize=12)
    fig.tight_layout()
    return fig, axes


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise world-frame dual-RBF awareness kernel on road layout")
    p.add_argument("-m", "--map", type=str, default=None,
                   help="Scenario config name under scenarios/configs/")
    p.add_argument("--from-result", type=str, default=None,
                   help="Load ego pose from a results directory")
    p.add_argument("--step", type=int, default=0,
                   help="Step index when loading from result (default: 0)")
    p.add_argument("--episode", "-e", type=int, default=0,
                   help="Episode index for batch results (default: 0)")
    p.add_argument("--ego-x", type=float, default=None)
    p.add_argument("--ego-y", type=float, default=None)
    p.add_argument("--ego-heading", type=float, default=None,
                   help="Ego heading in degrees (default: 0)")
    p.add_argument("--sigma1", type=float, default=15.0,
                   help="Narrow RBF spread (default: 15.0)")
    p.add_argument("--sigma2", type=float, default=25.0,
                   help="Wide RBF spread (default: 25.0)")
    p.add_argument("--fov", type=float, default=30.0,
                   help="Half FOV angle in degrees (default: 30)")
    p.add_argument("--w1", type=float, default=0.7,
                   help="Weight for omnidirectional RBF (default: 0.7)")
    p.add_argument("--w2", type=float, default=0.3,
                   help="Weight for forward RBF (default: 0.3)")
    p.add_argument("--extent", type=float, default=50.0,
                   help="Grid extent from ego in metres (default: 50)")
    p.add_argument("--grid-res", type=float, default=0.5,
                   help="Grid resolution in metres (default: 0.5)")
    p.add_argument("--figsize", type=float, nargs=2, default=[18, 6],
                   metavar=("W", "H"))
    p.add_argument("-o", "--out", type=str, default=None,
                   help="Save figure to file")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main():
    args = parse_args()

    scenario_map = None
    ego_x, ego_y = args.ego_x, args.ego_y
    ego_heading = math.radians(args.ego_heading) if args.ego_heading is not None else 0.0
    participant_positions = []

    if args.from_result:
        import dill
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from belief_utils import ExperimentResult, RESULTS_DIR

        run_dir = args.from_result
        if not os.path.isdir(run_dir):
            run_dir = os.path.join(RESULTS_DIR, args.from_result)

        pkl_path = os.path.join(run_dir, "results.pkl")
        with open(pkl_path, 'rb') as f:
            data = dill.load(f)

        if isinstance(data, dict) and "results" in data:
            result = data["results"][args.episode]
        else:
            result = data

        sr = result.steps[args.step]
        if sr.ego_position is not None:
            ego_x, ego_y = float(sr.ego_position[0]), float(sr.ego_position[1])
        if sr.ego_heading is not None:
            ego_heading = float(sr.ego_heading)

        # Participant positions at this step
        for aid, state in sr.dynamic_agents.items():
            participant_positions.append(
                (float(state.position[0]), float(state.position[1])))

        map_path = result.config.get("scenario", {}).get("map_path")
        if map_path:
            scenario_map = ip.Map.parse_from_opendrive(map_path)

        print(f"Loaded step {args.step} from episode {args.episode}: "
              f"ego=({ego_x:.1f}, {ego_y:.1f}), "
              f"heading={math.degrees(ego_heading):.1f}\u00b0, "
              f"{len(participant_positions)} participants")

    if scenario_map is None and args.map:
        config_path = os.path.join("scenarios", "configs", f"{args.map}.json")
        with open(config_path) as f:
            config = json.load(f)
        map_path = config["scenario"]["map_path"]
        scenario_map = ip.Map.parse_from_opendrive(map_path)

    if scenario_map is None:
        print("Error: provide --map or --from-result")
        sys.exit(1)

    if ego_x is None or ego_y is None:
        print("Error: provide --ego-x/--ego-y or --from-result")
        sys.exit(1)

    fig, axes = plot_attention_kernel(
        scenario_map,
        ego_x=ego_x,
        ego_y=ego_y,
        ego_heading=ego_heading,
        sigma_1=args.sigma1,
        sigma_2=args.sigma2,
        fov_half_angle=math.radians(args.fov),
        w1=args.w1,
        w2=args.w2,
        grid_res=args.grid_res,
        extent=args.extent,
        figsize=tuple(args.figsize),
        participant_positions=participant_positions or None,
    )

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
