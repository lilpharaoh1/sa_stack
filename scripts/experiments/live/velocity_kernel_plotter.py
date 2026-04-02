"""
Live velocity attention kernel plotter.

Shows the velocity feature kernel heatmap (RBF * FOV * awareness),
ego/participant positions, and per-agent kappa values updating each step.

Mirrors ``LiveAwarenessPlotter`` but visualises the velocity estimation
kernel instead of the awareness kernel.

Usage:
    from live.velocity_kernel_plotter import LiveVelocityKernelPlotter
    plotter = LiveVelocityKernelPlotter(scenario_map, ego_agent)

    # Inside the step loop, after collect_step:
    plotter.update(record)
"""

import sys
import os
import math
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", ".."))
sys.path.insert(0, _EXPERIMENTS_DIR)

from igp2.beliefcontrol.velocity_particles import compute_velocity_feature


def _compute_velocity_kernel(ego_xy, ego_heading, query_xy,
                             phi_i, sigma, fov_half_angle):
    """Evaluate the velocity feature at a query point (spatial only, no phi).

    FOV-gated RBF kernel with phi_i factored out of the heatmap so the
    spatial kernel is visible even when phi is low.
    """
    diff = query_xy - ego_xy
    dist_sq = float(np.dot(diff, diff))

    rbf = math.exp(-dist_sq / (2.0 * sigma ** 2))
    angle_to = math.atan2(diff[1], diff[0])
    rel_angle = (angle_to - ego_heading + math.pi) % (2.0 * math.pi) - math.pi
    if abs(rel_angle) > fov_half_angle:
        rbf = 0.0

    return rbf


class LiveVelocityKernelPlotter:
    """Live matplotlib figure showing velocity attention kernel.

    Left panel:  velocity feature heatmap on the road with ego + participants.
    The heatmap shows the spatial RBF*FOV kernel. Per-agent annotations
    show the full f_kappa value (including phi gating) and current kappa.
    """

    def __init__(self,
                 scenario_map,
                 ego_agent,
                 sigma: float = 25.0,
                 fov_half_angle: float = math.radians(30),
                 grid_res: float = 1.0,
                 extent: float = 40.0,
                 update_interval: int = 1):
        """
        Args:
            scenario_map: Parsed road map (igp2.Map).
            ego_agent: The ego BeliefAgent instance.
            sigma: RBF spread (m).
            fov_half_angle: Half-angle of FOV cone (radians).
            grid_res: Grid resolution in metres.
            extent: How far from ego to evaluate (metres).
            update_interval: Update plot every N steps.
        """
        self._map = scenario_map
        self._ego_agent = ego_agent
        self._sigma = sigma
        self._fov = fov_half_angle
        self._grid_res = grid_res
        self._extent = extent
        self._update_interval = update_interval

        self._step_count = 0

        self._agent_colours = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        ]

        # Set up figure with static map
        plt.ion()
        self._fig, self._ax_map = plt.subplots(
            1, 1, figsize=(10, 8),
            num='live_velocity_kernel')
        self._fig.suptitle('Live Velocity Kernel', fontsize=13)

        from igp2.opendrive.plot_map import plot_map
        plot_map(self._map, ax=self._ax_map, markings=True,
                 junction_color=(0, 0, 0, 0))
        self._ax_map.set_aspect('equal')

        self._fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
        plt.pause(0.01)

        self._dynamic_artists: List = []

    def _draw_vehicle(self, ax, x, y, heading, length, width,
                       facecolor='black', edgecolor='black',
                       alpha=1.0, zorder=10):
        """Draw a rotated vehicle rectangle centred at (x, y)."""
        rect = plt.Rectangle(
            (-length / 2, -width / 2), length, width,
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=1.0, alpha=alpha, zorder=zorder)
        t = Affine2D().rotate(heading).translate(x, y) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        return rect

    def _remove_dynamic(self):
        for artist in self._dynamic_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._dynamic_artists.clear()

    def _add(self, artist):
        self._dynamic_artists.append(artist)
        return artist

    def update(self, record):
        """Update the live plot with data from the current step.

        Args:
            record: StepRecord from collect_step().
        """
        self._step_count += 1
        if self._step_count % self._update_interval != 0:
            return

        if not plt.fignum_exists(self._fig.number):
            return

        if record.ego_position is None:
            return

        ego_x = float(record.ego_position[0])
        ego_y = float(record.ego_position[1])
        ego_heading = float(record.ego_heading) if record.ego_heading is not None else None

        # Participant info
        participants = {}
        for aid, state in record.dynamic_agents.items():
            px, py = float(state.position[0]), float(state.position[1])
            heading = float(state.heading) if hasattr(state, 'heading') else None
            length = float(state.metadata.length) if hasattr(state, 'metadata') else 4.5
            width = float(state.metadata.width) if hasattr(state, 'metadata') else 1.8
            participants[aid] = (px, py, heading, length, width)

        # Human awareness (phi) for each agent
        human_awareness = record.human_awareness or {}

        # Human kappa beliefs
        gt_kappa = record.velocity_kappa_gt or {}

        self._remove_dynamic()

        ax = self._ax_map

        # --- Heatmap: spatial velocity kernel (RBF * FOV, no phi gating) ---
        ext = self._extent
        res = self._grid_res
        xs = np.arange(ego_x - ext, ego_x + ext + res, res)
        ys = np.arange(ego_y - ext, ego_y + ext + res, res)
        XX, YY = np.meshgrid(xs, ys)
        F = np.zeros_like(XX)

        ego_xy = np.array([ego_x, ego_y], dtype=float)
        if ego_heading is not None:
            for i in range(XX.shape[0]):
                for j in range(XX.shape[1]):
                    q_xy = np.array([XX[i, j], YY[i, j]], dtype=float)
                    F[i, j] = _compute_velocity_kernel(
                        ego_xy, ego_heading, q_xy,
                        phi_i=1.0,  # show full spatial kernel
                        sigma=self._sigma,
                        fov_half_angle=self._fov)

        vmax = max(F.max(), 1e-6)
        mesh = ax.pcolormesh(XX, YY, F, cmap='PuBu', alpha=0.5,
                             shading='auto', vmin=0, vmax=vmax, zorder=3)
        self._add(mesh)

        # Ego vehicle
        ego_len = getattr(self._ego_agent, '_ego_length', 4.5)
        ego_wid = getattr(self._ego_agent, '_ego_width', 1.8)
        if ego_heading is not None:
            ego_rect = self._draw_vehicle(
                ax, ego_x, ego_y, ego_heading, ego_len, ego_wid,
                facecolor='black', edgecolor='black', alpha=0.8, zorder=10)
            self._add(ego_rect)
            dx = (ego_len / 2 + 2.0) * math.cos(ego_heading)
            dy = (ego_len / 2 + 2.0) * math.sin(ego_heading)
            ann = ax.annotate(
                '', xy=(ego_x + dx, ego_y + dy), xytext=(ego_x, ego_y),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                zorder=11)
            self._add(ann)
        else:
            ego_dot, = ax.plot(ego_x, ego_y, 'ko', markersize=10, zorder=10)
            self._add(ego_dot)

        # Participant vehicles with f_kappa and kappa annotations
        for idx, (aid, (px, py, p_heading, p_len, p_wid)) in enumerate(participants.items()):
            colour = self._agent_colours[idx % len(self._agent_colours)]
            if p_heading is not None:
                veh_rect = self._draw_vehicle(
                    ax, px, py, p_heading, p_len, p_wid,
                    facecolor=colour, edgecolor='black', alpha=0.7, zorder=11)
                self._add(veh_rect)
            else:
                marker, = ax.plot(px, py, 's', color=colour, markersize=12,
                                  markeredgecolor='black', linewidth=0.5, zorder=11)
                self._add(marker)

            label_parts = [f'Agent {aid}']

            phi_i = human_awareness.get(aid, 1.0)
            kappa = gt_kappa.get(aid, 1.0)

            if ego_heading is not None:
                p_xy = np.array([px, py], dtype=float)
                f_val = compute_velocity_feature(
                    ego_xy, ego_heading, p_xy,
                    phi_i=phi_i)
                label_parts.append(f'f_\u03ba={f_val:.2f}')

            label_parts.append(f'\u03ba={kappa:.2f}')
            label_parts.append(f'\u03c6={phi_i:.2f}')

            txt = ax.text(
                px + 2, py + 2, '\n'.join(label_parts),
                fontsize=7, zorder=12,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.85))
            self._add(txt)

        ax.set_title(
            f'Velocity Kernel  (step {self._step_count})', fontsize=10)

        pad = 5.0
        ax.set_xlim(ego_x - ext - pad, ego_x + ext + pad)
        ax.set_ylim(ego_y - ext - pad, ego_y + ext + pad)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Close the figure."""
        plt.ioff()
        plt.close(self._fig)
