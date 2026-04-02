"""Live velocity particle plotter for the belief experiment.

Shows the per-agent velocity particle distribution (kappa values and
weights) updating each simulation step, along with the weighted mean
and ESS over time.

Usage:
    from live.velocity_particle_plotter import LiveVelocityParticlePlotter
    vp_plotter = LiveVelocityParticlePlotter()

    # Inside the step loop, after collect_step:
    vp_plotter.update(record)
"""

import math
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class LiveVelocityParticlePlotter:
    """Live matplotlib figure showing velocity particle evolution.

    Top row:    per-agent particle distribution (bar chart of weights vs kappa).
    Bottom row: per-agent time series of weighted mean kappa and ESS.
    """

    MAX_AGENTS = 4  # max columns

    def __init__(self, update_interval: int = 1):
        self._update_interval = update_interval
        self._step_count = 0
        self._initialised = False

        # History for time-series
        self._steps: List[int] = []
        self._mean_history: Dict[int, List[float]] = {}   # aid -> [mean_kappa]
        self._ess_history: Dict[int, List[float]] = {}    # aid -> [ESS]
        self._std_history: Dict[int, List[float]] = {}    # aid -> [std]

    def _init_figure(self, agent_ids):
        n = min(len(agent_ids), self.MAX_AGENTS)
        plt.ion()
        self._fig, self._axes = plt.subplots(
            2, n, figsize=(4 * n, 6), squeeze=False,
            num='live_velocity_particles')
        self._fig.suptitle('Velocity Particles (kappa)', fontsize=13)
        self._agent_order = agent_ids[:n]
        for col, aid in enumerate(self._agent_order):
            self._axes[0, col].set_title(f'Agent {aid}', fontsize=9)
            self._axes[0, col].set_xlabel('kappa')
            self._axes[0, col].set_ylabel('weight')
            self._axes[0, col].set_xlim(0, 1.1)
            self._axes[0, col].set_ylim(0, 1.05)
            self._axes[1, col].set_xlabel('step')
            self._axes[1, col].set_ylabel('kappa / ESS')
        self._fig.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show(block=False)
        plt.pause(0.01)
        self._initialised = True

    def update(self, record):
        """Update the live plot with velocity particle data from the current step.

        Args:
            record: StepRecord from collect_step().
        """
        self._step_count += 1
        if self._step_count % self._update_interval != 0:
            return

        vp = getattr(record, 'velocity_particles', None)
        if vp is None or not vp:
            return

        agent_ids = sorted(vp.keys())

        if not self._initialised:
            self._init_figure(agent_ids)
            for aid in agent_ids:
                self._mean_history[aid] = []
                self._ess_history[aid] = []
                self._std_history[aid] = []

        if not plt.fignum_exists(self._fig.number):
            return

        self._steps.append(self._step_count)

        # Ground-truth kappa values (if available)
        gt_kappa = getattr(record, 'velocity_kappa_gt', None) or {}

        for col, aid in enumerate(self._agent_order):
            if aid not in vp:
                continue
            d = vp[aid]
            kappas = d['kappa_values']
            weights = d['weights']
            mean_k = d['mean']
            std_k = d['std']
            ess = d['ess']
            gt_k = gt_kappa.get(aid)

            self._mean_history[aid].append(mean_k)
            self._ess_history[aid].append(ess)
            self._std_history[aid].append(std_k)

            # --- Top: bar chart of particles ---
            ax_bar = self._axes[0, col]
            ax_bar.cla()
            title_str = f'Agent {aid}  (mean={mean_k:.3f}'
            if gt_k is not None:
                title_str += f', gt={gt_k:.2f}'
            title_str += ')'
            ax_bar.set_title(title_str, fontsize=9)
            ax_bar.bar(kappas, weights, width=0.05, color='steelblue',
                       edgecolor='black', alpha=0.8)
            ax_bar.axvline(mean_k, color='red', linestyle='--', linewidth=1.5,
                           label=f'mean={mean_k:.2f}')
            if gt_k is not None:
                ax_bar.axvline(gt_k, color='green', linestyle='-', linewidth=2.0,
                               alpha=0.7, label=f'gt={gt_k:.2f}')
            ax_bar.set_xlim(0, 1.1)
            ax_bar.set_ylim(0, max(max(weights) * 1.2, 0.3))
            ax_bar.set_xlabel('kappa')
            ax_bar.set_ylabel('weight')
            ax_bar.legend(fontsize=7, loc='upper left')

            # --- Bottom: time series ---
            ax_ts = self._axes[1, col]
            ax_ts.cla()
            steps = self._steps
            means = self._mean_history[aid]
            stds = self._std_history[aid]
            esses = self._ess_history[aid]

            ax_ts.plot(steps, means, 'r-', linewidth=1.5, label='mean kappa')
            means_arr = np.array(means)
            stds_arr = np.array(stds)
            ax_ts.fill_between(steps,
                               np.clip(means_arr - stds_arr, 0, 1),
                               np.clip(means_arr + stds_arr, 0, 1),
                               alpha=0.2, color='red')
            if gt_k is not None:
                ax_ts.axhline(gt_k, color='green', linestyle='-', linewidth=1.5,
                              alpha=0.7, label=f'gt kappa={gt_k:.2f}')
            else:
                ax_ts.axhline(1.0, color='green', linestyle=':', linewidth=0.8,
                              alpha=0.5, label='kappa=1')

            # ESS on secondary axis
            ax_ess = ax_ts.twinx()
            ax_ess.plot(steps, esses, 'b--', linewidth=1.0, alpha=0.6,
                        label='ESS')
            ax_ess.set_ylabel('ESS', color='blue', fontsize=8)
            ax_ess.tick_params(axis='y', labelcolor='blue')

            ax_ts.set_xlabel('step')
            ax_ts.set_ylabel('kappa', color='red', fontsize=8)
            ax_ts.tick_params(axis='y', labelcolor='red')
            ax_ts.set_ylim(0, 1.15)
            ax_ts.legend(fontsize=7, loc='upper left')

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Close the figure."""
        plt.ioff()
        plt.close(self._fig)
