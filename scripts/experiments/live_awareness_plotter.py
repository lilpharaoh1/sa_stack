"""
Live awareness plotter for the belief experiment.

Shows the Frenet awareness kernel heatmap, ego/participant positions,
and per-agent Kalman state (phi, psi) updating each simulation step.

Usage:
    Instantiate ``LiveAwarenessPlotter`` and call ``update()`` each step
    from the experiment loop.  See ``hook_into_experiment()`` for
    automatic integration.

    # In ind_belief_experiment.py, after agent creation:
    from live_awareness_plotter import LiveAwarenessPlotter
    live_plotter = LiveAwarenessPlotter(scenario_map, ego_agent)

    # Inside the step loop, after collect_step:
    live_plotter.update(record)
"""

import sys
import os
import math
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Affine2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from igp2.beliefcontrol.kalman_awareness import compute_feature_world


class LiveAwarenessPlotter:
    """Live matplotlib figure showing awareness kernel during an episode.

    Left panel:  awareness heatmap on the road with ego + participants.
    Right panel: per-agent phi (awareness probability) time series.

    The static road map is drawn once; only dynamic content is redrawn
    each step to avoid flicker.
    """

    def __init__(self,
                 scenario_map,
                 ego_agent,
                 sigma_1: float = 15.0,
                 sigma_2: float = 25.0,
                 fov_half_angle: float = math.radians(30),
                 w1: float = 0.7,
                 w2: float = 0.3,
                 grid_res: float = 1.0,
                 extent: float = 40.0,
                 update_interval: int = 1):
        """
        Args:
            scenario_map: Parsed road map (igp2.Map).
            ego_agent: The ego BeliefAgent instance.
            sigma_1, sigma_2, fov_half_angle, w1, w2: Kernel parameters.
            grid_res: Grid resolution in metres (coarser = faster).
            extent: How far from ego to evaluate (metres).
            update_interval: Update plot every N steps (1 = every step).
        """
        self._map = scenario_map
        self._ego_agent = ego_agent
        self._sigma_1 = sigma_1
        self._sigma_2 = sigma_2
        self._fov = fov_half_angle
        self._w1 = w1
        self._w2 = w2
        self._grid_res = grid_res
        self._extent = extent
        self._update_interval = update_interval

        self._step_count = 0

        # Colours for agents
        self._agent_colours = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        ]

        # Set up figure with static map
        plt.ion()
        self._fig, self._ax_map = plt.subplots(
            1, 1, figsize=(10, 8),
            num='live_awareness')  # named figure to avoid conflicts
        self._fig.suptitle('Live Awareness', fontsize=13)

        # Draw static map once
        from igp2.opendrive.plot_map import plot_map
        plot_map(self._map, ax=self._ax_map, markings=True,
                 junction_color=(0, 0, 0, 0))
        self._ax_map.set_aspect('equal')

        # Store map xlim/ylim for reference
        self._map_xlim = self._ax_map.get_xlim()
        self._map_ylim = self._ax_map.get_ylim()

        self._fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
        plt.pause(0.01)

        # Track dynamic artists so we can remove them without clearing
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
        """Remove all dynamic artists from previous frame."""
        for artist in self._dynamic_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._dynamic_artists.clear()

    def _add(self, artist):
        """Track a dynamic artist for removal next frame."""
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

        # Check figure still exists
        if not plt.fignum_exists(self._fig.number):
            return

        if record.ego_position is None:
            return

        ego_x = float(record.ego_position[0])
        ego_y = float(record.ego_position[1])
        ego_heading = float(record.ego_heading) if record.ego_heading is not None else None

        # Participant positions, headings, dimensions (world frame)
        participants = {}
        for aid, state in record.dynamic_agents.items():
            px, py = float(state.position[0]), float(state.position[1])
            heading = float(state.heading) if hasattr(state, 'heading') else None
            length = float(state.metadata.length) if hasattr(state, 'metadata') else 4.5
            width = float(state.metadata.width) if hasattr(state, 'metadata') else 1.8
            participants[aid] = (px, py, heading, length, width)

        # --- Remove previous dynamic artists ---
        self._remove_dynamic()

        # --- Left panel: awareness heatmap (dynamic overlay on static map) ---
        ax = self._ax_map

        # Compute heatmap (world-frame features)
        ext = self._extent
        res = self._grid_res
        xs = np.arange(ego_x - ext, ego_x + ext + res, res)
        ys = np.arange(ego_y - ext, ego_y + ext + res, res)
        XX, YY = np.meshgrid(xs, ys)
        F = np.zeros_like(XX)

        ego_xy = np.array([ego_x, ego_y], dtype=float)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                p_xy = np.array([XX[i, j], YY[i, j]], dtype=float)
                if ego_heading is not None:
                    F[i, j] = compute_feature_world(
                        ego_xy, ego_heading, p_xy,
                        self._sigma_1, self._sigma_2, self._fov,
                        self._w1, self._w2)
                else:
                    # No heading — omnidirectional only
                    diff = p_xy - ego_xy
                    dist_sq = float(np.dot(diff, diff))
                    F[i, j] = self._w1 * math.exp(
                        -dist_sq / (2.0 * self._sigma_1 ** 2))

        vmax = max(F.max(), 1e-6)
        mesh = ax.pcolormesh(XX, YY, F, cmap='YlOrRd', alpha=0.5,
                             shading='auto', vmin=0, vmax=vmax, zorder=3)
        self._add(mesh)

        # Ego vehicle footprint
        ego_len = getattr(self._ego_agent, '_ego_length', 4.5)
        ego_wid = getattr(self._ego_agent, '_ego_width', 1.8)
        if ego_heading is not None:
            ego_rect = self._draw_vehicle(
                ax, ego_x, ego_y, ego_heading, ego_len, ego_wid,
                facecolor='black', edgecolor='black', alpha=0.8, zorder=10)
            self._add(ego_rect)
            # Heading arrow
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

        # Participant vehicle footprints with feature values
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
            if ego_heading is not None:
                f_val = compute_feature_world(
                    ego_xy, ego_heading,
                    np.array([px, py], dtype=float),
                    self._sigma_1, self._sigma_2, self._fov,
                    self._w1, self._w2)
                label_parts.append(f'f={f_val:.2f}')

            txt = ax.text(
                px + 2, py + 2, '\n'.join(label_parts),
                fontsize=7, zorder=12,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.85))
            self._add(txt)

        # Step label
        step_txt = ax.set_title(
            f'Awareness Kernel  (step {self._step_count})', fontsize=10)

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
