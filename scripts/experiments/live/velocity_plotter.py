"""Live velocity belief plotter for the belief experiment.

Shows the velocity-estimation feature kernel on the road map (analogous to
the awareness plotter), per-agent particle distributions, and kappa
time series with ground-truth reference.

Layout (3 columns):
    Left:   velocity feature heatmap on the road, ego + participant vehicles
            annotated with estimated kappa and awareness phi.
    Middle: per-agent particle bar charts (weights vs kappa).
    Right:  per-agent kappa time series with +/- 1 std band and ground truth.

Usage:
    from live.velocity_plotter import LiveVelocityPlotter
    plotter = LiveVelocityPlotter(scenario_map, ego_agent)

    # Inside the step loop, after collect_step:
    plotter.update(record)
"""

import sys
import os
import math
from typing import Dict, List, Optional

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


class LiveVelocityPlotter:
    """Live matplotlib figure showing velocity beliefs during an episode.

    Left panel:  velocity feature heatmap on the road with vehicles annotated.
    Middle column: per-agent particle bar charts.
    Right column:  per-agent kappa time series.
    """

    MAX_AGENTS = 3  # max rows for particle / time-series panels

    def __init__(self,
                 scenario_map,
                 ego_agent,
                 grid_res: float = 1.5,
                 extent: float = 40.0,
                 update_interval: int = 1):
        """
        Args:
            scenario_map: Parsed road map (igp2.Map).
            ego_agent: The ego BeliefAgent instance.
            grid_res: Heatmap grid resolution in metres.
            extent: How far from ego to evaluate (metres).
            update_interval: Update plot every N steps.
        """
        self._map = scenario_map
        self._ego_agent = ego_agent
        self._grid_res = grid_res
        self._extent = extent
        self._update_interval = update_interval

        self._step_count = 0
        self._initialised = False

        # History for time-series
        self._steps: List[int] = []
        self._mean_history: Dict[int, List[float]] = {}
        self._std_history: Dict[int, List[float]] = {}
        self._ess_history: Dict[int, List[float]] = {}

        # Colours for agents
        self._agent_colours = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        ]

    def _init_figure(self, n_agents):
        """Create the figure once we know how many agents exist."""
        n_rows = max(1, min(n_agents, self.MAX_AGENTS))

        plt.ion()
        self._fig = plt.figure(
            figsize=(16, max(5, 3 * n_rows)),
            num='live_velocity')
        self._fig.suptitle('Velocity Belief', fontsize=13)

        # GridSpec: left half = map, right half = n_rows x 2 (bars + timeseries)
        gs = self._fig.add_gridspec(n_rows, 3, width_ratios=[2, 1, 1],
                                     hspace=0.4, wspace=0.35)
        self._ax_map = self._fig.add_subplot(gs[:, 0])

        # Draw static map once
        from igp2.opendrive.plot_map import plot_map
        plot_map(self._map, ax=self._ax_map, markings=True,
                 junction_color=(0, 0, 0, 0))
        self._ax_map.set_aspect('equal')

        self._ax_bars = []
        self._ax_ts = []
        for row in range(n_rows):
            self._ax_bars.append(self._fig.add_subplot(gs[row, 1]))
            self._ax_ts.append(self._fig.add_subplot(gs[row, 2]))

        self._fig.tight_layout(rect=[0, 0, 1, 0.94])
        plt.show(block=False)
        plt.pause(0.01)

        self._dynamic_artists: List = []
        self._n_rows = n_rows
        self._initialised = True

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
        """Update the live plot with velocity data from the current step.

        Args:
            record: StepRecord from collect_step().
        """
        self._step_count += 1
        if self._step_count % self._update_interval != 0:
            return

        if record.ego_position is None:
            return

        vp_data = getattr(record, 'velocity_particles', None) or {}
        gt_kappa = getattr(record, 'velocity_kappa_gt', None) or {}

        ego_x = float(record.ego_position[0])
        ego_y = float(record.ego_position[1])
        ego_heading = float(record.ego_heading) if record.ego_heading is not None else None

        # Participant positions
        participants = {}
        for aid, state in record.dynamic_agents.items():
            px, py = float(state.position[0]), float(state.position[1])
            heading = float(state.heading) if hasattr(state, 'heading') else None
            length = float(state.metadata.length) if hasattr(state, 'metadata') else 4.5
            width = float(state.metadata.width) if hasattr(state, 'metadata') else 1.8
            participants[aid] = (px, py, heading, length, width)

        agent_ids = sorted(participants.keys())

        # Initialise figure on first call
        if not self._initialised:
            self._init_figure(len(agent_ids))
            self._agent_order = agent_ids[:self._n_rows]
            for aid in agent_ids:
                self._mean_history[aid] = []
                self._std_history[aid] = []
                self._ess_history[aid] = []

        if not plt.fignum_exists(self._fig.number):
            return

        self._steps.append(self._step_count)

        # Get vehicle awareness for phi values (needed for velocity feature)
        vehicle_awareness = getattr(record, 'vehicle_awareness', None) or {}

        # =================================================================
        # Left panel: velocity feature heatmap on road
        # =================================================================
        self._remove_dynamic()
        ax = self._ax_map

        ext = self._extent
        res = self._grid_res
        ego_xy = np.array([ego_x, ego_y], dtype=float)

        # Use an average phi for the heatmap (the feature is phi-gated)
        avg_phi = np.mean(list(vehicle_awareness.values())) if vehicle_awareness else 0.5

        xs = np.arange(ego_x - ext, ego_x + ext + res, res)
        ys = np.arange(ego_y - ext, ego_y + ext + res, res)
        XX, YY = np.meshgrid(xs, ys)
        F = np.zeros_like(XX)

        if ego_heading is not None:
            for i in range(XX.shape[0]):
                for j in range(XX.shape[1]):
                    p_xy = np.array([XX[i, j], YY[i, j]], dtype=float)
                    F[i, j] = compute_velocity_feature(
                        ego_xy, ego_heading, p_xy,
                        phi_i=avg_phi)

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

        # Participant vehicles with velocity annotations
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

            # Build annotation label
            label_parts = [f'Agent {aid}']
            phi_i = vehicle_awareness.get(aid, 0.5)
            label_parts.append(f'\u03c6={phi_i:.2f}')

            if aid in vp_data:
                mean_k = vp_data[aid]['mean']
                label_parts.append(f'\u03ba={mean_k:.2f}')
            gt_k = gt_kappa.get(aid)
            if gt_k is not None:
                label_parts.append(f'gt={gt_k:.2f}')

            # Velocity feature value at this participant
            if ego_heading is not None:
                f_vel = compute_velocity_feature(
                    ego_xy, ego_heading,
                    np.array([px, py], dtype=float),
                    phi_i=phi_i)
                label_parts.append(f'f\u03ba={f_vel:.2f}')

            txt = ax.text(
                px + 2, py + 2, '\n'.join(label_parts),
                fontsize=7, zorder=12,
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', alpha=0.85))
            self._add(txt)

        ax.set_title(f'Velocity Feature  (step {self._step_count})', fontsize=10)
        pad = 5.0
        ax.set_xlim(ego_x - ext - pad, ego_x + ext + pad)
        ax.set_ylim(ego_y - ext - pad, ego_y + ext + pad)

        # =================================================================
        # Middle + Right panels: per-agent particle bars and time series
        # =================================================================
        for row, aid in enumerate(self._agent_order):
            colour = self._agent_colours[row % len(self._agent_colours)]
            gt_k = gt_kappa.get(aid)

            if aid in vp_data:
                d = vp_data[aid]
                kappas = d['kappa_values']
                weights = d['weights']
                mean_k = d['mean']
                std_k = d['std']
                ess = d['ess']
            else:
                kappas = []
                weights = []
                mean_k = 1.0
                std_k = 0.0
                ess = 0.0

            self._mean_history.setdefault(aid, []).append(mean_k)
            self._std_history.setdefault(aid, []).append(std_k)
            self._ess_history.setdefault(aid, []).append(ess)

            # --- Particle bar chart ---
            ax_bar = self._ax_bars[row]
            ax_bar.cla()
            title = f'Agent {aid}'
            if gt_k is not None:
                title += f'  (gt={gt_k:.2f})'
            ax_bar.set_title(title, fontsize=9, color=colour)

            if kappas:
                ax_bar.bar(kappas, weights, width=0.05, color=colour,
                           edgecolor='black', alpha=0.8)
                ax_bar.axvline(mean_k, color='red', linestyle='--',
                               linewidth=1.5, label=f'\u03ba={mean_k:.2f}')
                if gt_k is not None:
                    ax_bar.axvline(gt_k, color='green', linestyle='-',
                                   linewidth=2.0, alpha=0.7)
                ax_bar.set_ylim(0, max(max(weights) * 1.3, 0.3))
            else:
                ax_bar.text(0.5, 0.5, 'no particles', transform=ax_bar.transAxes,
                            ha='center', va='center', fontsize=8, alpha=0.5)

            ax_bar.set_xlim(0, 1.1)
            ax_bar.set_xlabel('\u03ba', fontsize=8)
            ax_bar.set_ylabel('weight', fontsize=8)
            ax_bar.tick_params(labelsize=7)
            if kappas:
                ax_bar.legend(fontsize=7, loc='upper left')

            # --- Kappa time series ---
            ax_ts = self._ax_ts[row]
            ax_ts.cla()
            steps = self._steps
            means = self._mean_history[aid]
            stds = self._std_history[aid]

            ax_ts.plot(steps, means, '-', color=colour, linewidth=1.5,
                       label=f'\u03ba est')
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax_ts.fill_between(steps,
                               np.clip(means_arr - stds_arr, 0, 1),
                               np.clip(means_arr + stds_arr, 0, 1),
                               alpha=0.15, color=colour)

            if gt_k is not None:
                ax_ts.axhline(gt_k, color='green', linestyle='-',
                              linewidth=1.5, alpha=0.7, label=f'gt={gt_k:.2f}')
            else:
                ax_ts.axhline(1.0, color='green', linestyle=':',
                              linewidth=0.8, alpha=0.5)

            ax_ts.set_ylim(0, 1.15)
            ax_ts.set_xlabel('step', fontsize=8)
            ax_ts.set_ylabel('\u03ba', fontsize=8)
            ax_ts.tick_params(labelsize=7)
            ax_ts.legend(fontsize=7, loc='upper right')

            # ESS on secondary axis
            esses = self._ess_history[aid]
            ax_ess = ax_ts.twinx()
            ax_ess.plot(steps, esses, 'b--', linewidth=0.8, alpha=0.4)
            ax_ess.set_ylabel('ESS', color='blue', fontsize=7)
            ax_ess.tick_params(axis='y', labelcolor='blue', labelsize=6)
            K = len(kappas) if kappas else 4
            ax_ess.set_ylim(0, K + 0.5)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Close the figure."""
        plt.ioff()
        plt.close(self._fig)
