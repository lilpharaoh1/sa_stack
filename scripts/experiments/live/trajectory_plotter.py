"""
Live MCTS traversal comparison: greedy belief vs QCBF intervention.

Left panel:  Greedy MCTS traversal under predicted beliefs (green).
             The acceleration the human would take at each tree depth
             if we don't intervene.
Right panel: QCBF traversal from the same tree. Steps that follow the
             belief-optimal action are green; steps where QCBF overrides
             due to regret are blue.

Automatically enabled when inference is mcts_* and ref_controls is mcts-*.
"""

import sys
import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENTS_DIR = os.path.dirname(_DIR)
sys.path.insert(0, os.path.join(_EXPERIMENTS_DIR, "..", ".."))
sys.path.insert(0, _EXPERIMENTS_DIR)


_GREEN = '#00CC44'
_ORANGE = '#E67300'


class LiveTrajectoryPlotter:
    """Live plot: MCTS greedy traversal vs QCBF-filtered traversal."""

    def __init__(self, scenario_map, ego_agent, update_interval: int = 1):
        self._ego_agent = ego_agent
        self._update_interval = update_interval
        self._step_count = 0

        # Get coarse dt from the MCTS planner
        bi = getattr(ego_agent, '_belief_inference', None)
        mcts = getattr(bi, '_mcts_planner', None) if bi else None
        self._dt_coarse = mcts._dt if mcts else 0.5

        # Set up figure
        plt.ion()
        self._fig, (self._ax_greedy, self._ax_qcbf) = plt.subplots(
            1, 2, figsize=(14, 5),
            num='live_trajectory')
        self._fig.suptitle('MCTS Traversal: Belief-Greedy vs QCBF', fontsize=12)

        for ax in (self._ax_greedy, self._ax_qcbf):
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Acceleration (m/s²)', fontsize=9)
            ax.grid(True, alpha=0.3)

        self._fig.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show(block=False)
        plt.pause(0.01)

    def update(self, record):
        """Update with data from the current step."""
        self._step_count += 1
        if self._step_count % self._update_interval != 0:
            return

        if not plt.fignum_exists(self._fig.number):
            return

        bi = getattr(self._ego_agent, '_belief_inference', None)
        if bi is None:
            return

        # --- Get greedy trajectory (belief-optimal, no intervention) ---
        greedy_controls = None
        greedy_traj = None
        mcts = getattr(bi, '_mcts_planner', None)
        if mcts is not None:
            greedy_traj = getattr(mcts, '_last_greedy_trajectory', None)
        if greedy_traj is not None and greedy_traj.controls is not None:
            greedy_controls = greedy_traj.controls[:, 0]  # acceleration

        # --- Get QCBF trajectory (with intervention flags) ---
        qcbf_traj = getattr(bi, '_last_qcbf_trajectory', None)
        qcbf_controls = None
        qcbf_flags = None
        if qcbf_traj is not None and qcbf_traj.controls is not None:
            qcbf_controls = qcbf_traj.controls[:, 0]  # acceleration
            qcbf_flags = qcbf_traj.interventions or [False] * len(qcbf_controls)

        dt = self._dt_coarse

        # === Left panel: greedy traversal under beliefs ===
        ax = self._ax_greedy
        ax.cla()
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Acceleration (m/s²)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='grey', linewidth=0.5, alpha=0.5)

        ax.set_ylim(-3.0, 3.0)

        if greedy_controls is not None:
            # Only show tree-depth steps (not heuristic padding)
            tree_k = len(greedy_controls)
            if hasattr(greedy_traj, 'tree_depth') and greedy_traj.tree_depth is not None:
                tree_k = min(greedy_traj.tree_depth, len(greedy_controls))
            a_tree = greedy_controls[:tree_k]
            t = np.arange(tree_k) * dt
            ax.plot(t, a_tree, 'o-', color=_GREEN, linewidth=2.5,
                    markersize=9, markeredgecolor='black',
                    markeredgewidth=0.5, alpha=0.9)
            # Show padding as faded
            if tree_k < len(greedy_controls):
                a_pad = greedy_controls[tree_k - 1:]  # overlap last tree point
                t_pad = np.arange(tree_k - 1, len(greedy_controls)) * dt
                ax.plot(t_pad, a_pad, 'o--', color=_GREEN, linewidth=1.0,
                        markersize=5, alpha=0.3)
                ax.axvline(x=(tree_k - 1) * dt, color='grey', linestyle=':',
                           linewidth=1.0, alpha=0.5)

        ax.set_title(f'Greedy under belief (step {self._step_count})', fontsize=10)

        # === Right panel: QCBF traversal with intervention markers ===
        ax2 = self._ax_qcbf
        ax2.cla()
        ax2.set_xlabel('Time (s)', fontsize=9)
        ax2.set_ylabel('Acceleration (m/s²)', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='grey', linewidth=0.5, alpha=0.5)

        ax2.set_ylim(-3.0, 3.0)

        if qcbf_controls is not None and qcbf_flags is not None:
            # Only show tree-depth steps
            K_total = len(qcbf_controls)
            tree_k = K_total
            if qcbf_traj is not None and qcbf_traj.tree_depth is not None:
                tree_k = min(qcbf_traj.tree_depth, K_total)
            t = np.arange(K_total) * dt

            # Draw tree-depth segments — each line colored by the dot it leads to
            for i in range(min(tree_k, K_total) - 1):
                color = _ORANGE if qcbf_flags[i + 1] else _GREEN
                ax2.plot(t[i:i+2], qcbf_controls[i:i+2], '-',
                         color=color, linewidth=2.5, alpha=0.9)

            # Draw tree-depth points — stars for overrides, dots for belief-optimal
            for i in range(tree_k):
                if qcbf_flags[i]:
                    ax2.plot(t[i], qcbf_controls[i], '*', color=_ORANGE,
                             markersize=14, markeredgecolor='black',
                             markeredgewidth=0.5, zorder=5)
                else:
                    ax2.plot(t[i], qcbf_controls[i], 'o', color=_GREEN,
                             markersize=9, markeredgecolor='black',
                             markeredgewidth=0.5, zorder=5)

            # Show padding as faded
            if tree_k < K_total:
                a_pad = qcbf_controls[tree_k - 1:]
                t_pad = np.arange(tree_k - 1, K_total) * dt
                ax2.plot(t_pad, a_pad, 'o--', color='grey', linewidth=1.0,
                         markersize=5, alpha=0.3)
                ax2.axvline(x=(tree_k - 1) * dt, color='grey', linestyle=':',
                            linewidth=1.0, alpha=0.5)

            n_interv = sum(qcbf_flags[:tree_k])
            title = f'QCBF ({n_interv}/{tree_k} overridden, step {self._step_count})'
        else:
            title = f'QCBF (no data, step {self._step_count})'

        # Legend proxy
        ax2.plot([], [], 'o-', color=_GREEN, linewidth=2.5,
                 markersize=9, label='Belief-optimal')
        ax2.plot([], [], '*-', color=_ORANGE, linewidth=2.5,
                 markersize=14, label='QCBF override')
        ax2.legend(fontsize=7, loc='best', framealpha=0.8)
        ax2.set_title(title, fontsize=10)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Close the figure."""
        plt.ioff()
        plt.close(self._fig)
