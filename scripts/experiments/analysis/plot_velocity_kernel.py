"""
Visualise the velocity attention kernel on a road layout.

For a given ego position, heading, and awareness (phi), computes the
velocity feature f_kappa at every point in a grid.  Overlays the result
as a heatmap on the road map.

The feature is:  f_kappa = RBF(sigma) * FOV_gate * phi_i

Uses a single FOV-gated RBF (no omnidirectional component), reflecting
that velocity estimation requires direct visual observation.

Two panels are shown:
  1. Spatial kernel (dual-RBF) — ignoring phi gating
  2. Full kernel (dual-RBF * phi) — what actually drives kappa updates

Usage:
    # Single ego pose on a scenario map:
    python scripts/experiments/analysis/plot_velocity_kernel.py -m belief_experiment5 \
        --ego-x 50 --ego-y -2 --ego-heading 0

    # From a results file (uses ego pose at a given step):
    python scripts/experiments/analysis/plot_velocity_kernel.py --from-result results/my_run/ \
        --step 10

    # Adjust kernel parameters:
    python scripts/experiments/analysis/plot_velocity_kernel.py -m belief_experiment5 \
        --ego-x 50 --ego-y -2 --ego-heading 0 --sigma1 15 --sigma2 25 --fov 30
"""

import sys
import os
import argparse
import json
import math

import numpy as np
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", ".."))
sys.path.insert(0, _EXPERIMENTS_DIR)

import igp2 as ip
from igp2.opendrive.plot_map import plot_map
from igp2.beliefcontrol.velocity_particles import compute_velocity_feature


def plot_velocity_kernel(
        scenario_map,
        ego_x: float,
        ego_y: float,
        ego_heading: float = 0.0,
        sigma: float = 25.0,
        fov_half_angle: float = math.radians(30),
        grid_res: float = 0.5,
        extent: float = 50.0,
        figsize=(12, 6),
        participant_positions=None,
        participant_phis=None,
        participant_kappas=None,
):
    """Plot the velocity feature kernel as a heatmap on the road map.

    FOV-gated RBF kernel:
      f_kappa = RBF(sigma) * FOV_gate * phi_i

    Args:
        scenario_map: Parsed road map.
        ego_x, ego_y: Ego world position.
        ego_heading: Ego heading (radians).
        sigma: RBF spread (m).
        fov_half_angle: Half FOV angle (radians).
        grid_res: Grid resolution in metres.
        extent: How far from ego to evaluate (metres).
        figsize: Figure size.
        participant_positions: Optional list of (x, y).
        participant_phis: Optional list of phi values (same order).
        participant_kappas: Optional list of kappa values (same order).

    Returns:
        (fig, axes) tuple.
    """
    ego_xy = np.array([ego_x, ego_y], dtype=float)

    # Build evaluation grid
    xs = np.arange(ego_x - extent, ego_x + extent + grid_res, grid_res)
    ys = np.arange(ego_y - extent, ego_y + extent + grid_res, grid_res)
    XX, YY = np.meshgrid(xs, ys)

    # Compute spatial kernel (phi=1)
    F_spatial = np.zeros_like(XX)

    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            p_xy = np.array([XX[i, j], YY[i, j]], dtype=float)
            diff = p_xy - ego_xy
            dist_sq = float(np.dot(diff, diff))

            rbf = math.exp(-dist_sq / (2.0 * sigma ** 2))
            angle_to = math.atan2(diff[1], diff[0])
            rel_angle = (angle_to - ego_heading + math.pi) % (2.0 * math.pi) - math.pi
            if abs(rel_angle) > fov_half_angle:
                rbf = 0.0

            F_spatial[i, j] = rbf

    # --- Figure: 2 panels ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    fov_deg = math.degrees(fov_half_angle)
    panels = [
        (F_spatial,
         f'Spatial: RBF(\u03c3={sigma:.0f}m) \u00b7 FOV\u00b1{fov_deg:.0f}\u00b0'),
        (F_spatial,
         f'Full: RBF \u00b7 FOV \u00b7 \u03c6\u1d62 '
         f'(per-agent \u03c6 shown at markers)'),
    ]

    # FOV cone boundary lines
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
        im = ax.pcolormesh(XX, YY, F, cmap='PuBu', alpha=0.6,
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
            phis = participant_phis or [1.0] * len(participant_positions)
            kappas = participant_kappas or [None] * len(participant_positions)
            for idx, (px, py) in enumerate(participant_positions):
                phi_i = phis[idx] if idx < len(phis) else 1.0
                kappa_i = kappas[idx] if idx < len(kappas) else None

                p_xy = np.array([px, py], dtype=float)
                f_val = compute_velocity_feature(
                    ego_xy, ego_heading, p_xy,
                    phi_i=phi_i,
                    sigma=sigma,
                    fov_half_angle=fov_half_angle)
                f_spatial = compute_velocity_feature(
                    ego_xy, ego_heading, p_xy,
                    phi_i=1.0,
                    sigma=sigma,
                    fov_half_angle=fov_half_angle)

                ax.plot(px, py, 's', color='blue', markersize=10,
                        markeredgecolor='black', zorder=11)
                parts = [f'f_\u03ba={f_val:.3f}',
                         f'\u03c6={phi_i:.2f}']
                if kappa_i is not None:
                    parts.append(f'\u03ba={kappa_i:.2f}')
                ax.text(px + 1.5, py + 1.5, '\n'.join(parts),
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
        f'Velocity Kernel (World)  |  '
        f'ego=({ego_x:.1f}, {ego_y:.1f}), '
        f'heading={math.degrees(ego_heading):.1f}\u00b0',
        fontsize=12)
    fig.tight_layout()
    return fig, axes


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise velocity attention kernel on road layout")
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
    p.add_argument("--sigma", type=float, default=25.0,
                   help="RBF spread (default: 25.0)")
    p.add_argument("--fov", type=float, default=30.0,
                   help="Half FOV angle in degrees (default: 30)")
    p.add_argument("--extent", type=float, default=50.0,
                   help="Grid extent from ego in metres (default: 50)")
    p.add_argument("--grid-res", type=float, default=0.5,
                   help="Grid resolution in metres (default: 0.5)")
    p.add_argument("--figsize", type=float, nargs=2, default=[12, 6],
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
    participant_phis = []
    participant_kappas = []

    if args.from_result:
        import dill
        from utils import ExperimentResult, RESULTS_DIR

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

        # Participant positions, awareness, kappa at this step
        awareness = sr.human_awareness or {}
        gt_kappa = sr.velocity_kappa_gt or {}
        for aid, state in sr.dynamic_agents.items():
            participant_positions.append(
                (float(state.position[0]), float(state.position[1])))
            participant_phis.append(awareness.get(aid, 1.0))
            participant_kappas.append(gt_kappa.get(aid, 1.0))

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

    fig, axes = plot_velocity_kernel(
        scenario_map,
        ego_x=ego_x,
        ego_y=ego_y,
        ego_heading=ego_heading,
        sigma=args.sigma,
        fov_half_angle=math.radians(args.fov),
        grid_res=args.grid_res,
        extent=args.extent,
        figsize=tuple(args.figsize),
        participant_positions=participant_positions or None,
        participant_phis=participant_phis or None,
        participant_kappas=participant_kappas or None,
    )

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved to {args.out}")
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
