"""
Live speed profile plotter for the belief experiment.

Shows the estimated (reference) vs. intervening speed profiles over the
planning horizon, updating each simulation step when an intervention is
active.

Usage:
    from live_speed_plotter import LiveSpeedPlotter
    plotter = LiveSpeedPlotter()

    # Inside the step loop, after collect_step:
    plotter.update(record)
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


class LiveSpeedPlotter:
    """Live matplotlib figure showing reference vs. intervention speed profiles.

    One subplot showing speed (m/s) vs. horizon step:
      - **Reference (blue)**: what the human would do (belief-based plan)
      - **Intervention (green)**: the optimized intervention speed profile
      - **Current speed (red dot)**: ego speed at step 0

    Only draws when an intervention is active (both arrays non-None).
    """

    def __init__(self):
        plt.ion()
        self._fig, self._ax = plt.subplots(
            1, 1, figsize=(7, 4), num='live_speed')
        self._fig.suptitle('Speed Profile (Intervention)', fontsize=12)
        self._ax.set_xlabel('Horizon step')
        self._ax.set_ylabel('Speed (m/s)')
        self._ax.grid(True, alpha=0.3)

        self._line_ref, = self._ax.plot([], [], 'b-o', markersize=3,
                                         label='Reference (human)')
        self._line_opt, = self._ax.plot([], [], 'g-s', markersize=3,
                                         label='Intervention (opt)')
        self._dot_cur, = self._ax.plot([], [], 'ro', markersize=8,
                                        label='Current speed')
        self._ax.legend(loc='upper right', fontsize=8)
        self._fig.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show(block=False)
        plt.pause(0.01)

    def update(self, record):
        """Update the live plot with data from the current step.

        Args:
            record: StepRecord from collect_step().
        """
        if not plt.fignum_exists(self._fig.number):
            return

        ref = record.intervention_ref_states
        opt = record.intervention_opt_states

        if ref is None or opt is None:
            # No intervention active — clear the plot
            self._line_ref.set_data([], [])
            self._line_opt.set_data([], [])
            self._dot_cur.set_data([], [])
            self._ax.set_title('No intervention active', fontsize=10)
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            plt.pause(0.001)
            return

        # Speed is column index 3 in the Frenet state [s, d, phi, v]
        ref_speed = ref[:, 3]
        opt_speed = opt[:, 3]

        # ref and opt can have different horizon lengths
        ref_steps = np.arange(len(ref_speed))
        opt_steps = np.arange(len(opt_speed))

        self._line_ref.set_data(ref_steps, ref_speed)
        self._line_opt.set_data(opt_steps, opt_speed)

        # Current ego speed at step 0
        ego_speed = record.ego_speed
        if ego_speed is not None:
            self._dot_cur.set_data([0], [ego_speed])
        else:
            self._dot_cur.set_data([], [])

        # Adjust axes
        all_speeds = np.concatenate([ref_speed, opt_speed])
        if ego_speed is not None:
            all_speeds = np.append(all_speeds, ego_speed)
        ymin = max(0, all_speeds.min() - 1.0)
        ymax = all_speeds.max() + 1.0
        max_steps = max(len(ref_speed), len(opt_speed))
        self._ax.set_xlim(-0.5, max_steps - 0.5)
        self._ax.set_ylim(ymin, ymax)
        self._ax.set_title(f'Intervention active  (step {record.step})',
                           fontsize=10)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        """Close the figure."""
        plt.ioff()
        plt.close(self._fig)
