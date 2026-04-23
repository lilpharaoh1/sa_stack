"""
Config-driven car-following experiment runner with live diagnostic plots.

Mirrors the experiment loop from scripts/debug/car_follow_example.py but
runs on the CommonRoad simulation harness (no CARLA / SUMO needed).

Usage (from repo root, carla-igp2 env):

    # Run with live plots (belief posterior, kappa evolution, actions)
    python experiments/commonroad/run_carfollow.py -e exp1_simple_acc

    # Override inference / intervention from CLI
    python experiments/commonroad/run_carfollow.py -e exp1_simple_acc \
        --inference boltzmann_reactive --intervention cbf_lookahead

    # Save gif + results, headless
    python experiments/commonroad/run_carfollow.py -e exp2_merge_in_front \
        --save --headless

    # Static beliefs (no RBF evolution)
    python experiments/commonroad/run_carfollow.py -e exp1_simple_acc --human static
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime
from typing import Dict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.transforms import Affine2D

# Ensure this directory is importable
sys.path.insert(0, os.path.dirname(__file__))

from run_experiment import (
    Simulation, SimVehicle, VehicleState, Action, Observation,
    bicycle_step, constant_velocity_controller,
    lane_change_controller_factory, accelerating_controller_factory,
    LANE_WIDTH, N_LANES, VEH_LENGTH, VEH_WIDTH,
    OBSTACLE_COLORS, EGO_COLOR, SCENARIO_DIR, FIG_DIR,
)
from carfollow_controller import CarFollowController, VelocityErrorBelief

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
#  Live diagnostic plotters (matching car_follow_example.py)
# ---------------------------------------------------------------------------

class BeliefPlotter:
    """Live bar chart of velocity-error posteriors for all vehicles."""

    def __init__(self, candidates: np.ndarray,
                 vehicle_ids: list, true_vel_errors: Dict[int, float]):
        n = len(vehicle_ids)
        self._fig, self._axes = plt.subplots(
            1, n, figsize=(6 * n, 3), squeeze=False)
        self._axes = self._axes[0]
        self._candidates = candidates
        self._vehicle_ids = vehicle_ids
        self._bars = {}
        colours = ["steelblue", "#e74c3c", "#2ecc71", "#f39c12"]

        for i, vid in enumerate(vehicle_ids):
            ax = self._axes[i]
            colour = colours[i % len(colours)]
            bars = ax.bar(candidates,
                          np.ones(len(candidates)) / len(candidates),
                          width=0.08, color=colour, edgecolor="white")
            self._bars[vid] = bars
            true_ve = true_vel_errors.get(vid)
            if true_ve is not None:
                ax.axvline(true_ve, color="red", linestyle="--",
                           linewidth=1.5,
                           label=f"true $\\kappa$ = {true_ve:.2f}")
                ax.legend(fontsize=8)
            ax.set_xlabel("$\\kappa$")
            ax.set_ylabel("$P(\\kappa)$")
            ax.set_title(f"Vehicle {vid}")
            ax.set_xlim(candidates[0] - 0.1, candidates[-1] + 0.1)
            ax.set_ylim(0, 1.0)
        self._fig.suptitle("Inferred velocity-error beliefs", fontsize=11)
        self._fig.tight_layout()

    def update(self, per_vehicle_probs: Dict[int, np.ndarray],
               step: int = None):
        for vid, bars in self._bars.items():
            if vid in per_vehicle_probs:
                probs = per_vehicle_probs[vid]
                for bar, p in zip(bars, probs):
                    bar.set_height(p)
        for ax in self._axes:
            ax.set_ylim(0, 1.05)
        if step is not None:
            self._fig.suptitle(
                f"Inferred velocity-error beliefs  (t={step})", fontsize=11)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


class HumanBeliefPlotter:
    """Live time-series of kappa evolution + inferred estimate for all vehicles."""

    def __init__(self, vehicle_ids: list,
                 initial_vel_errors: Dict[int, float], window: int = 500):
        n = len(vehicle_ids)
        self._fig, self._axes = plt.subplots(
            n, 1, figsize=(8, 3 * n), squeeze=False, sharex=True)
        self._axes = [ax for ax in self._axes[:, 0]]
        self._window = window
        self._vehicle_ids = vehicle_ids

        colours = ["steelblue", "#e74c3c", "#2ecc71", "#f39c12"]
        self._data = {}  # vid → dict of lists
        self._lines = {}  # vid → (line_true, line_inf)
        self._fills = {}  # vid → fill artist

        for i, vid in enumerate(vehicle_ids):
            ax = self._axes[i]
            colour = colours[i % len(colours)]
            init_ve = initial_vel_errors.get(vid, 0.0)

            line_true, = ax.plot([], [], color=colour, linewidth=1.5,
                                 label=f"true $\\kappa_{{{vid}}}$")
            line_inf, = ax.plot([], [], color="orange", linewidth=1.5,
                                label="inferred")
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
            ax.axhline(init_ve, color="red", linewidth=0.8,
                       linestyle="--", alpha=0.5,
                       label=f"$\\kappa_0$ = {init_ve:.2f}")
            ax.set_ylabel(f"Veh {vid}  $\\kappa$")
            ax.legend(fontsize=7, loc="upper right")

            self._lines[vid] = (line_true, line_inf)
            self._fills[vid] = None
            self._data[vid] = {
                "steps": [], "vel_errs": [],
                "inf_means": [], "inf_stds": [],
            }

        self._axes[-1].set_xlabel("step")
        self._fig.suptitle("Human velocity-error evolution", fontsize=11)
        self._fig.tight_layout()

    def update(self, step: int, per_vehicle: Dict[int, dict]):
        """Update with per-vehicle diagnostics.

        per_vehicle: {vid: {"human_vel_err", "inferred_mean", "inferred_std",
                            "kf_kappa", "kf_P"}}
        """
        for vid in self._vehicle_ids:
            info = per_vehicle.get(vid, {})
            d = self._data[vid]
            hve = info.get("human_vel_err")
            if hve is None:
                continue

            d["steps"].append(step)
            d["vel_errs"].append(hve)

            est = info.get("kf_kappa")
            if est is None:
                est = info.get("inferred_mean")
            # Prefer continuous Kalman std when available;
            # fall back to discrete posterior std otherwise.
            std = info.get("inferred_std_continuous")
            if std is None:
                std = info.get("inferred_std")

            d["inf_means"].append(est)
            d["inf_stds"].append(std)

            # Draw
            ax = self._axes[self._vehicle_ids.index(vid)]
            line_true, line_inf = self._lines[vid]

            lo = max(0, len(d["steps"]) - self._window)
            s = d["steps"][lo:]
            line_true.set_data(s, d["vel_errs"][lo:])

            means = d["inf_means"][lo:]
            stds = d["inf_stds"][lo:]
            if any(m is not None for m in means):
                s_inf = [si for si, m in zip(s, means) if m is not None]
                v_inf = [m for m in means if m is not None]
                line_inf.set_data(s_inf, v_inf)

                if self._fills[vid] is not None:
                    self._fills[vid].remove()
                    self._fills[vid] = None
                std_inf = [sd for sd, m in zip(stds, means)
                           if m is not None and sd is not None]
                if std_inf and len(std_inf) == len(s_inf):
                    upper = [m + 2 * sd for m, sd in zip(v_inf, std_inf)]
                    lower = [m - 2 * sd for m, sd in zip(v_inf, std_inf)]
                    self._fills[vid] = ax.fill_between(
                        s_inf, lower, upper, color="orange", alpha=0.15)

            ax.set_xlim(s[0] if s else 0, (s[-1] if s else 1) + 1)
            all_v = d["vel_errs"][lo:] + [m for m in means if m is not None]
            for m, sd in zip(means, stds):
                if m is not None and sd is not None:
                    all_v.extend([m - 2 * sd, m + 2 * sd])
            if all_v:
                ax.set_ylim(min(all_v) - 0.05, max(all_v) + 0.05)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


class ActionPlotter:
    """Live time-series: human accel vs executed accel + safety bound."""

    def __init__(self, window: int = 200):
        self._fig, (self._ax_act, self._ax_safe) = plt.subplots(
            2, 1, figsize=(9, 5), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]})
        self._window = window
        self._steps = []
        self._human_accel, self._exec_accel, self._a_safe = [], [], []

        self._line_human, = self._ax_act.plot(
            [], [], color="steelblue", linewidth=1.5, label="human accel")
        self._line_exec, = self._ax_act.plot(
            [], [], color="orange", linewidth=1.5, label="executed accel")
        self._ax_act.axhline(0, color="grey", linewidth=0.5)
        self._ax_act.set_ylabel("acceleration (m/s\u00b2)")
        self._ax_act.set_title("Actions & safety constraint")
        self._ax_act.legend(loc="upper right", fontsize=9)

        self._line_safe, = self._ax_safe.plot(
            [], [], color="red", linewidth=1.5,
            label="$a_{\\mathrm{max}}^{\\mathrm{safe}}$")
        self._ax_safe.axhline(0, color="grey", linewidth=0.5)
        self._ax_safe.set_xlabel("step")
        self._ax_safe.set_ylabel("$a_{\\mathrm{max}}^{\\mathrm{safe}}$")
        self._ax_safe.legend(loc="upper right", fontsize=9)
        self._fig.tight_layout()

    def update(self, step, human_accel, exec_accel, a_safe=None):
        self._steps.append(step)
        self._human_accel.append(human_accel)
        self._exec_accel.append(exec_accel)
        self._a_safe.append(a_safe)

        lo = max(0, len(self._steps) - self._window)
        s = self._steps[lo:]
        self._line_human.set_data(s, self._human_accel[lo:])
        self._line_exec.set_data(s, self._exec_accel[lo:])

        vals = self._human_accel[lo:] + self._exec_accel[lo:]
        if vals:
            self._ax_act.set_ylim(min(vals) - 1, max(vals) + 1)

        safe_v = self._a_safe[lo:]
        if any(v is not None for v in safe_v):
            safe_s = [si for si, v in zip(s, safe_v) if v is not None]
            safe_vals = [v for v in safe_v if v is not None]
            self._line_safe.set_data(safe_s, safe_vals)
            self._ax_safe.set_ylim(min(safe_vals) - 1, max(safe_vals) + 1)

        self._ax_safe.set_xlim(s[0], s[-1] + 1)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


class RiccatiPlotter:
    """Live plot of Kalman covariance P vs Riccati steady-state.

    Top:    P (or sqrt(P)) over time with the algebraic Riccati solution.
    Bottom: Inferred estimate +/- 2*sqrt(P) with true value.
    """

    def __init__(self, Q: float, R: float, C: float = 1.0,
                 A: float = 1.0,
                 window: int = 500, sqrt_scale: bool = True):
        self._fig, (self._ax_p, self._ax_est) = plt.subplots(
            2, 1, figsize=(9, 5), sharex=True,
            gridspec_kw={"height_ratios": [1, 1]})
        self._window = window
        self._sqrt = sqrt_scale
        self._C = C

        # Algebraic Riccati steady-state with process gain A:
        # P = R*(A²P+Q)/(C²*(A²P+Q)+R)
        # (C²A²)P² + (QC²+R(1-A²))P - RQ = 0
        a2c2 = (A * C) ** 2
        b = Q * C ** 2 + R * (1.0 - A ** 2)
        c = -R * Q
        disc = b ** 2 - 4 * a2c2 * c
        self._P_ss = (-b + np.sqrt(disc)) / (2 * a2c2)

        ss_label = (f"$\\sqrt{{P_{{ss}}}}$ = {np.sqrt(self._P_ss):.4f}"
                    if sqrt_scale
                    else f"$P_{{ss}}$ = {self._P_ss:.6f}")
        self._ax_p.axhline(
            np.sqrt(self._P_ss) if sqrt_scale else self._P_ss,
            color="red", linestyle="--", linewidth=1.5,
            label=ss_label)
        self._line_p, = self._ax_p.plot(
            [], [], color="steelblue", linewidth=1.5,
            label="$\\sqrt{P}$" if sqrt_scale else "$P$")
        ylabel = "$\\sqrt{P_{\\varepsilon}}$ (m/s)" if sqrt_scale else "$P_{\\varepsilon}$"
        self._ax_p.set_ylabel(ylabel)
        self._ax_p.set_title(
            f"Kalman covariance convergence   (Q={Q}, R={R}, C={C})",
            fontsize=10)
        self._ax_p.legend(fontsize=8, loc="upper right")

        # Estimate + uncertainty band
        self._line_true, = self._ax_est.plot(
            [], [], color="steelblue", linewidth=1.5,
            label="true $\\kappa$")
        self._line_inf, = self._ax_est.plot(
            [], [], color="orange", linewidth=1.5,
            label="inferred $\\kappa$")
        self._fill = None
        self._ax_est.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        self._ax_est.set_xlabel("step")
        self._ax_est.set_ylabel("$\\kappa$")
        self._ax_est.legend(fontsize=8, loc="upper right")
        self._fig.tight_layout()

        self._steps = []
        self._P_vals = []
        self._true_vals = []
        self._inf_vals = []
        self._std_vals = []

    def update(self, step: int, kf_P: float,
               true_kappa: float = None,
               inferred_kappa: float = None,
               inferred_std: float = None):
        self._steps.append(step)
        self._P_vals.append(np.sqrt(kf_P) if self._sqrt else kf_P)
        self._true_vals.append(true_kappa)
        self._inf_vals.append(inferred_kappa)
        self._std_vals.append(inferred_std)

        lo = max(0, len(self._steps) - self._window)
        s = self._steps[lo:]

        # --- Top: P convergence ---
        self._line_p.set_data(s, self._P_vals[lo:])
        p_vals = self._P_vals[lo:]
        ss_v = np.sqrt(self._P_ss) if self._sqrt else self._P_ss
        y_hi = max(max(p_vals), ss_v) * 1.1
        self._ax_p.set_ylim(0, max(y_hi, 1e-6))
        self._ax_p.set_xlim(s[0], s[-1] + 1)

        # --- Bottom: estimate + band ---
        true_s = [si for si, v in zip(s, self._true_vals[lo:]) if v is not None]
        true_v = [v for v in self._true_vals[lo:] if v is not None]
        self._line_true.set_data(true_s, true_v)

        inf_s = [si for si, v in zip(s, self._inf_vals[lo:]) if v is not None]
        inf_v = [v for v in self._inf_vals[lo:] if v is not None]
        self._line_inf.set_data(inf_s, inf_v)

        if self._fill is not None:
            self._fill.remove()
            self._fill = None
        band = [(si, m, sd) for si, m, sd in
                zip(s, self._inf_vals[lo:], self._std_vals[lo:])
                if m is not None and sd is not None]
        if band:
            bs, bm, bsd = zip(*band)
            upper = [m + 2 * sd for m, sd in zip(bm, bsd)]
            lower = [m - 2 * sd for m, sd in zip(bm, bsd)]
            self._fill = self._ax_est.fill_between(
                bs, lower, upper, color="orange", alpha=0.15)

        all_v = true_v + inf_v
        if all_v:
            self._ax_est.set_ylim(min(all_v) - 0.1, max(all_v) + 0.1)
        self._ax_est.set_xlim(s[0], s[-1] + 1)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


# ---------------------------------------------------------------------------
#  Config loading
# ---------------------------------------------------------------------------

def load_config(name: str) -> dict:
    path = os.path.join(CONFIG_DIR, f"{name}.json")
    with open(path) as f:
        return json.load(f)


def _make_ego_lane_change_controller(ego_ctrl: CarFollowController,
                                     lc_cfg: dict):
    """Wrap CarFollowController with lane-change steering.

    The CarFollowController handles longitudinal acceleration; this wrapper
    adds the lateral steering needed for the ego to change lanes.
    """
    y_from = lc_cfg["y_from"]
    y_to = lc_cfg["y_to"]
    t_start = lc_cfg["t_start"]
    t_dur = lc_cfg["t_dur"]
    dy = y_to - y_from
    t_lc_end = t_start + t_dur

    def controller(obs: Observation) -> Action:
        # Longitudinal: delegate to CarFollowController
        accel_action = ego_ctrl(obs)
        accel = accel_action.acceleration

        # Lateral: lane-change steering (same profile as non-ego lane change)
        t = obs.time_step * obs.dt
        ego = obs.ego
        v_fwd = max(ego.velocity, 1.0)  # avoid division by zero

        if t < t_start:
            vy_desired = 0.0
            y_target = y_from
        elif t < t_lc_end:
            s = (t - t_start) / t_dur
            vy_desired = dy / t_dur * (1 - np.cos(2 * np.pi * s))
            y_target = y_from + dy * (s - np.sin(2 * np.pi * s) / (2 * np.pi))
        else:
            vy_desired = 0.0
            y_target = y_to

        heading_ff = np.arctan2(vy_desired, v_fwd)
        y_err = y_target - ego.y
        heading_fb = np.clip(y_err * 0.3, -0.05, 0.05)
        heading_desired = heading_ff + heading_fb
        heading_err = heading_desired - ego.heading
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
        steer = np.clip(heading_err * 2.0, -0.3, 0.3)

        return Action(accel, steer)

    return controller


def build_simulation(cfg: dict, ego_ctrl) -> Simulation:
    """Build a Simulation from config, with vehicle overrides."""
    scenario_name = cfg["scenario"]["xml"]
    vehicles_cfg = cfg.get("vehicles", {})
    overrides = {}

    for vid_str, vcfg in vehicles_cfg.items():
        vid = int(vid_str)
        ctrl_type = vcfg.get("controller", "constant_velocity")

        if ctrl_type == "constant_velocity":
            overrides[vid] = constant_velocity_controller
        elif ctrl_type == "lane_change":
            lc = vcfg["lane_change"]
            overrides[vid] = lane_change_controller_factory(**lc)
        elif ctrl_type == "accelerating":
            ac = vcfg.get("accelerating", {})
            overrides[vid] = accelerating_controller_factory(**ac)
        # else: fall back to playback from XML

    return Simulation(
        scenario_name,
        ego_controller=ego_ctrl,
        vehicle_overrides=overrides,
    )


# ---------------------------------------------------------------------------
#  Scene renderer (same as run_experiment.py SimulationRenderer)
# ---------------------------------------------------------------------------

class SceneRenderer:
    def __init__(self, sim: Simulation, title: str = "",
                 view_half_x: float = 60.0):
        self.sim = sim
        self.title = title
        self.view_half_x = view_half_x
        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.y_lo = -2.0
        self.y_hi = N_LANES * LANE_WIDTH + 2.0

    def _draw_vehicle(self, ax, cx, cy, orient, color,
                      alpha=0.7, lw=1.5, label=None):
        rect = mpatches.FancyBboxPatch(
            (-VEH_LENGTH / 2, -VEH_WIDTH / 2), VEH_LENGTH, VEH_WIDTH,
            boxstyle="round,pad=0.15",
            facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw)
        tr = Affine2D().rotate(orient).translate(cx, cy) + ax.transData
        rect.set_transform(tr)
        ax.add_patch(rect)
        if label:
            ax.annotate(label, xy=(cx, cy), fontsize=7, fontweight="bold",
                        ha="center", va="center", color="white", zorder=20)

    def draw_frame(self, t: int, ego_ctrl: CarFollowController = None):
        ax = self.ax
        ax.clear()

        for i in range(N_LANES + 1):
            y = i * LANE_WIDTH
            is_edge = (i == 0 or i == N_LANES)
            ax.axhline(y, color="gray", linewidth=2 if is_edge else 1,
                       linestyle="-" if is_edge else "--", zorder=1)
        for i in range(N_LANES):
            ax.axhspan(i * LANE_WIDTH, (i + 1) * LANE_WIDTH,
                       color="#f5f5f5", zorder=0)

        # Goal region
        for pp in self.sim.pps.planning_problem_dict.values():
            for gs in pp.goal.state_list:
                if hasattr(gs, "position") and gs.position is not None:
                    shape = gs.position
                    if hasattr(shape, "center") and hasattr(shape, "length"):
                        cx, cy = shape.center
                        rect = mpatches.Rectangle(
                            (cx - shape.length / 2, cy - shape.width / 2),
                            shape.length, shape.width,
                            linewidth=2, edgecolor=EGO_COLOR,
                            facecolor=EGO_COLOR, alpha=0.10,
                            linestyle="--", zorder=2)
                        ax.add_patch(rect)

        # Non-ego vehicles
        sorted_ids = sorted(self.sim.vehicles.keys())
        for idx, vid in enumerate(sorted_ids):
            veh = self.sim.vehicles[vid]
            s = veh.history[min(t, len(veh.history) - 1)]
            color = OBSTACLE_COLORS[idx % len(OBSTACLE_COLORS)]
            self._draw_vehicle(ax, s.x, s.y, s.heading, color)
            ax.annotate(f"{vid}", xy=(s.x, s.y), fontsize=7,
                        fontweight="bold", ha="center", va="center",
                        color="white", zorder=20)
            ax.annotate(f"{s.velocity:.1f} m/s",
                        xy=(s.x, s.y + VEH_WIDTH),
                        fontsize=6, ha="center", color=color, zorder=15)
            trail_start = max(0, t - 30)
            trail = veh.history[trail_start:min(t + 1, len(veh.history))]
            if len(trail) > 1:
                ax.plot([h.x for h in trail], [h.y for h in trail],
                        "-", color=color, alpha=0.25, linewidth=2, zorder=1)

        # Ego
        ego_s = self.sim.ego.history[min(t, len(self.sim.ego.history) - 1)]
        self._draw_vehicle(ax, ego_s.x, ego_s.y, ego_s.heading,
                           EGO_COLOR, alpha=0.85, label="EGO")
        ax.annotate(f"{ego_s.velocity:.1f} m/s",
                    xy=(ego_s.x, ego_s.y + VEH_WIDTH),
                    fontsize=6, ha="center", color=EGO_COLOR, zorder=15)
        trail_start = max(0, t - 30)
        trail = self.sim.ego.history[trail_start:min(t + 1, len(self.sim.ego.history))]
        if len(trail) > 1:
            ax.plot([h.x for h in trail], [h.y for h in trail],
                    "-", color=EGO_COLOR, alpha=0.4, linewidth=2, zorder=1)

        ax.set_xlim(ego_s.x - self.view_half_x, ego_s.x + self.view_half_x)
        ax.set_ylim(self.y_lo, self.y_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        # Title with diagnostics
        time_s = t * self.sim.dt
        info_str = f"t = {time_s:.1f} s   v = {ego_s.velocity:.1f} m/s"
        if ego_ctrl and ego_ctrl.last_step_info:
            info = ego_ctrl.last_step_info
            d = info.get("distance")
            if d is not None:
                info_str += f"   d = {d:.1f} m"
            if info.get("intervened"):
                info_str += "   [INTERVENED]"
        ax.set_title(f"{self.title}    {info_str}", fontsize=11)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

METHODS_DIR = os.path.join(CONFIG_DIR, "methods")
HUMANS_DIR = os.path.join(CONFIG_DIR, "humans")


def parse_args():
    p = argparse.ArgumentParser(
        description="Car-following experiment (CommonRoad)")
    p.add_argument("--experiment", "-e", required=True,
                   help="Config name (e.g. exp1_simple_acc)")
    p.add_argument("--method", "-m", type=str, default=None,
                   help="Method config from configs/methods/ "
                        "(e.g. kalman_cbf_kalman). Overrides "
                        "inference/intervention in the "
                        "experiment config.")
    p.add_argument("--human-config", "-hc", type=str, default=None,
                   help="Human config from configs/humans/ "
                        "(e.g. walk_noisy). Sets human type "
                        "and parameters.")
    p.add_argument("--inference", type=str, default=None,
                   help="Override inference method")
    p.add_argument("--intervention", type=str, default=None,
                   help="Override intervention method")
    p.add_argument("--human", type=str, default=None,
                   help="Override human type (static/gaussian/walk)")
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--save", action="store_true", help="Save scene gif")
    p.add_argument("--no-save", action="store_true",
                   help="Don't save episode/metadata to results/")
    p.add_argument("--headless", action="store_true", help="No live window")
    p.add_argument("--interval", type=int, default=50,
                   help="Animation interval ms")
    p.add_argument("--seed", type=int, default=21)
    return p.parse_args()


TITLES = {
    "exp1_simple_acc": "Exp 1: ACC with Belief Inference",
    "exp2_merge_in_front": "Exp 2: Merge in Front + Belief",
    "exp3_ego_merge": "Exp 3: Ego Merge Behind Vehicle + Belief",
}


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.headless:
        matplotlib.use("Agg")
    else:
        plt.ion()

    # Load config
    cfg = load_config(args.experiment)
    ego_cfg = cfg["ego"]
    vehicles_cfg = cfg.get("vehicles", {})

    # Apply method config (overrides experiment defaults for filter params)
    if args.method is not None:
        method_path = os.path.join(METHODS_DIR, f"{args.method}.json")
        with open(method_path) as f:
            method_cfg = json.load(f)
        for key, val in method_cfg.items():
            ego_cfg[key] = val

    # Apply human config (overrides experiment defaults for human params)
    if args.human_config is not None:
        human_path = os.path.join(HUMANS_DIR, f"{args.human_config}.json")
        with open(human_path) as f:
            human_cfg = json.load(f)
        # "type" in human config maps to "human" in ego config
        if "type" in human_cfg:
            ego_cfg["human"] = human_cfg.pop("type")
        for key, val in human_cfg.items():
            ego_cfg[key] = val

    # Apply CLI overrides (highest priority)
    if args.inference is not None:
        ego_cfg["inference"] = args.inference
    if args.intervention is not None:
        ego_cfg["intervention"] = args.intervention
    if args.human is not None:
        ego_cfg["human"] = args.human
    if args.gamma is not None:
        ego_cfg["gamma"] = args.gamma
    if args.beta is not None:
        ego_cfg["beta"] = args.beta

    # Build velocity error map
    vel_errors = {}
    for vid_str, vcfg in vehicles_cfg.items():
        ve = vcfg.get("velocity_error", 0.0)
        vel_errors[int(vid_str)] = ve

    # Create the ego controller
    ego_ctrl = CarFollowController(
        velocity_errors=vel_errors,
        target_distance=ego_cfg.get("target_distance", 20.0),
        d_safe=ego_cfg.get("d_safe", 15.0),
        desired_speed=ego_cfg.get("desired_speed", 15.0),
        beta=ego_cfg.get("beta", 1.0),
        gamma=ego_cfg.get("gamma", 0.99),
        inference=ego_cfg.get("inference", "none"),
        intervention=ego_cfg.get("intervention", "none"),
        human=ego_cfg.get("human", "static"),
        b_kappa=ego_cfg.get("b_kappa", 0.01),
        sigma_kappa=ego_cfg.get("sigma_kappa", 25.0),
        kf_Q=ego_cfg.get("kf_Q", 0.001),
        kf_R=ego_cfg.get("kf_R", 0.01),
        kf_alpha=ego_cfg.get("kf_alpha", 0.0),
        kf_epsilon_init=ego_cfg.get("kf_epsilon_init", 0.0),
        kf_revert_alpha=ego_cfg.get("kf_revert_alpha", 0.9),
        kf_lingap_beta=ego_cfg.get("kf_lingap_beta", 0.1),
        human_sigma=ego_cfg.get("human_sigma"),
        human_mu=ego_cfg.get("human_mu"),
        human_walk_Q=ego_cfg.get("human_walk_Q"),
        human_revert_alpha=ego_cfg.get("human_revert_alpha", 0.9),
        human_lingap_beta=ego_cfg.get("human_lingap_beta", 0.1),
        action_noise_std=ego_cfg.get("action_noise_std", 0.0),
    )

    # Wrap ego controller with lane-change steering if configured
    lc_cfg = ego_cfg.get("ego_lane_change")
    if lc_cfg:
        ego_sim_ctrl = _make_ego_lane_change_controller(ego_ctrl, lc_cfg)
    else:
        ego_sim_ctrl = ego_ctrl

    # Build simulation
    sim = build_simulation(cfg, ego_sim_ctrl)

    title = TITLES.get(args.experiment, args.experiment)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  inference={ego_cfg['inference']}  "
          f"intervention={ego_cfg['intervention']}  "
          f"human={ego_cfg['human']}")
    print(f"  velocity errors: {vel_errors}")
    print(f"{'='*60}\n")

    # --- Set up live plotters ---
    belief_plotter = None
    human_plotter = None
    action_plotter = None
    riccati_plotter = None
    scene_renderer = SceneRenderer(sim, title=title)

    if not args.headless:
        vehicle_ids = sorted(vel_errors.keys())
        if vehicle_ids:
            # Use candidates from first vehicle (same grid for all)
            cands = ego_ctrl.beliefs[vehicle_ids[0]].candidates
            belief_plotter = BeliefPlotter(
                cands, vehicle_ids, vel_errors)
            human_plotter = HumanBeliefPlotter(
                vehicle_ids, vel_errors)
        action_plotter = ActionPlotter()

        # Riccati convergence plot for Kalman-based inference
        if ego_cfg["inference"] in ("boltzmann_kalman", "iidkalman", "walkkalman", "revertkalman", "lingapkalman", "lingapkalman"):
            C = 1.0 if ego_cfg["inference"] in ("iidkalman", "walkkalman", "revertkalman", "lingapkalman") else 15.0
            inf = ego_cfg["inference"]
            if inf in ("revertkalman", "lingapkalman"):
                A_proc = ego_ctrl._kf_revert_alpha
            elif inf == "iidkalman":
                A_proc = 0.0
            else:
                A_proc = 1.0
            riccati_plotter = RiccatiPlotter(
                Q=ego_ctrl._kf_Q, R=ego_ctrl._kf_R, C=C, A=A_proc)

    # --- Episode recorder (collects all per-step data) ---
    recorder = {
        "steps": [],
        "human_accel": [],
        "executed_accel": [],
        "accel_deviation": [],
        "jerk": [],
        "distance": [],
        "ego_speed": [],
        "lead_speed": [],
        "intervened": [],
        "human_vel_err": [],
        "kf_vhat": [],
        "kf_epsilon": [],
        "kf_kappa": [],
        "kf_P": [],
        "norm_innovation": [],
        "inferred_mean": [],
        "inferred_std": [],
        "belief_dists": [],
        "kl_divergence": [],
        "ego_positions": [],
        "vehicle_positions": {},  # vid → list of (x, y, heading, v)
    }
    # Initialise vehicle position lists
    for vid in sim.vehicles:
        recorder["vehicle_positions"][vid] = []

    # Compute oracle distribution for KL divergence
    first_vid = next(iter(vel_errors), None)
    oracle_dist = None
    if first_vid is not None:
        belief = ego_ctrl.beliefs.get(first_vid)
        if belief is not None:
            oracle_dist = np.zeros(len(belief.candidates))
            idx = np.argmin(np.abs(
                belief.candidates - vel_errors[first_vid]))
            oracle_dist[idx] = 1.0

    max_steps = cfg["scenario"].get("max_steps", 200)
    prev_exec_accel = 0.0

    # --- Main simulation loop ---
    for t in range(max_steps):
        if not sim.step():
            break

        info = ego_ctrl.last_step_info

        # --- Record step data ---
        human_a = info.get("accel", 0.0)
        exec_a = info.get("executed_accel", human_a)
        recorder["steps"].append(t)
        recorder["human_accel"].append(human_a)
        recorder["executed_accel"].append(exec_a)
        recorder["accel_deviation"].append(human_a - exec_a)
        recorder["jerk"].append(exec_a - prev_exec_accel)
        prev_exec_accel = exec_a
        recorder["distance"].append(info.get("distance"))
        recorder["ego_speed"].append(info.get("ego_speed", 0.0))
        recorder["lead_speed"].append(info.get("lead_speed"))
        recorder["intervened"].append(info.get("intervened", False))
        recorder["human_vel_err"].append(info.get("human_vel_err"))
        recorder["kf_vhat"].append(info.get("kf_vhat"))
        recorder["kf_epsilon"].append(info.get("kf_epsilon"))
        recorder["kf_kappa"].append(info.get("kf_kappa"))
        recorder["kf_P"].append(info.get("kf_P"))
        recorder["norm_innovation"].append(info.get("norm_innovation"))
        recorder["inferred_mean"].append(info.get("inferred_mean"))
        recorder["inferred_std"].append(info.get("inferred_std"))

        # Belief distribution
        inferred_dist = info.get("inferred_dist")
        if inferred_dist is not None:
            probs = list(inferred_dist.values())
            recorder["belief_dists"].append(probs)
            if oracle_dist is not None:
                eps = 1e-12
                p = np.array(probs)
                kl = float(np.sum(
                    oracle_dist * np.log((oracle_dist + eps) / (p + eps))))
                recorder["kl_divergence"].append(kl)
            else:
                recorder["kl_divergence"].append(None)
        else:
            recorder["belief_dists"].append(None)
            recorder["kl_divergence"].append(None)

        # Positions
        ego_s = sim.ego.state
        recorder["ego_positions"].append(
            [ego_s.x, ego_s.y, ego_s.heading, ego_s.velocity])
        for vid, veh in sim.vehicles.items():
            s = veh.state
            recorder["vehicle_positions"][vid].append(
                [s.x, s.y, s.heading, s.velocity])

        # --- Update live plots ---
        if not args.headless:
            # Scene window
            scene_renderer.draw_frame(t + 1, ego_ctrl)
            scene_renderer.fig.canvas.draw_idle()
            scene_renderer.fig.canvas.flush_events()

            if info:
                # Update belief bar charts for all vehicles
                per_veh = info.get("per_vehicle", {})
                if belief_plotter and per_veh:
                    per_veh_probs = {}
                    for vid, vdiag in per_veh.items():
                        dist = vdiag.get("inferred_dist")
                        if dist is not None:
                            per_veh_probs[vid] = np.array(
                                list(dist.values()))
                    if per_veh_probs:
                        belief_plotter.update(per_veh_probs, step=t)

                # Update kappa evolution for all vehicles
                if human_plotter and per_veh:
                    human_plotter.update(t, per_veh)

                if action_plotter:
                    action_plotter.update(
                        step=t, human_accel=human_a,
                        exec_accel=exec_a,
                        a_safe=info.get("a_max_safe"))

                if riccati_plotter:
                    kf_p = info.get("kf_P")
                    if kf_p is not None:
                        # For idkalman, kf_P is in epsilon space;
                        # inferred_std_continuous is already sqrt(P_kappa)
                        riccati_plotter.update(
                            step=t, kf_P=kf_p,
                            true_kappa=info.get("human_vel_err"),
                            inferred_kappa=info.get("kf_kappa",
                                                    info.get("inferred_mean")),
                            inferred_std=info.get("inferred_std_continuous",
                                                  info.get("inferred_std")))

            plt.pause(0.001)

        # Console output every 20 steps
        if t % 20 == 0 and info:
            d = info.get("distance")
            d_str = f"{d:.1f}" if d else "N/A"
            intv = " [INTERVENED]" if info.get("intervened") else ""
            kappa = info.get("human_vel_err")
            k_str = f"{kappa:.3f}" if kappa is not None else "N/A"
            print(f"[t={t:4d}]  d={d_str:>6s}m  "
                  f"v_ego={info.get('ego_speed', 0):.1f}  "
                  f"a_human={human_a:+.2f}  "
                  f"a_exec={exec_a:+.2f}  "
                  f"kappa={k_str}{intv}")

    # --- Episode complete: save everything, close windows ---

    # Print summary
    n_intv = sum(1 for v in recorder["intervened"] if v)
    n_steps = len(recorder["steps"])
    dev = np.array(recorder["accel_deviation"])
    jerk = np.array(recorder["jerk"])
    kl_vals = [v for v in recorder["kl_divergence"] if v is not None]

    print(f"\n{'='*60}")
    print(f"  Episode Summary  ({n_steps} steps)")
    print(f"{'='*60}")
    print(f"  Accel deviation  |  mean={np.mean(dev):.4f}  "
          f"std={np.std(dev):.4f}  max={np.max(np.abs(dev)):.4f}")
    print(f"  Jerk             |  mean={np.mean(jerk):.4f}  "
          f"std={np.std(jerk):.4f}  max={np.max(np.abs(jerk)):.4f}")
    if kl_vals:
        print(f"  KL(oracle||inf)  |  mean={np.mean(kl_vals):.4f}  "
              f"final={kl_vals[-1]:.4f}")
    print(f"  Interventions    |  {n_intv}/{n_steps} steps "
          f"({100*n_intv/n_steps:.1f}%)")
    print(f"  Final ego speed  |  {sim.ego.state.velocity:.1f} m/s")
    kappa_final = recorder["human_vel_err"][-1] if recorder["human_vel_err"] else None
    if kappa_final is not None:
        print(f"  Final kappa      |  {kappa_final:.4f}")
    print(f"{'='*60}")

    # Save episode data to JSON
    run_dir = None
    if not args.no_save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        hc_tag = f"_hc_{args.human_config}" if args.human_config else ""
        run_name = (f"{args.experiment}_inf_{ego_cfg['inference']}"
                    f"_int_{ego_cfg['intervention']}"
                    f"_human_{ego_cfg['human']}{hc_tag}"
                    f"_seed{args.seed}_{ts}")
        run_dir = os.path.join(RESULTS_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Convert vehicle_positions keys to strings for JSON
        recorder_json = dict(recorder)
        recorder_json["vehicle_positions"] = {
            str(k): v for k, v in recorder["vehicle_positions"].items()}
        # Add belief candidates
        first_vid = next(iter(vel_errors), None)
        if first_vid is not None:
            belief = ego_ctrl.beliefs.get(first_vid)
            if belief is not None:
                recorder_json["belief_candidates"] = belief.candidates.tolist()
        recorder_json["true_vel_errors"] = {
            str(k): v for k, v in vel_errors.items()}

        episode_path = os.path.join(run_dir, "episode.json")
        with open(episode_path, "w") as f:
            json.dump(recorder_json, f, indent=2)

        # Save run metadata
        meta = {
            "experiment": args.experiment,
            "inference": ego_cfg["inference"],
            "intervention": ego_cfg["intervention"],
            "human": ego_cfg["human"],
            "gamma": ego_cfg.get("gamma"),
            "beta": ego_cfg.get("beta"),
            "target_distance": ego_cfg.get("target_distance"),
            "d_safe": ego_cfg.get("d_safe"),
            "b_kappa": ego_cfg.get("b_kappa"),
            "sigma_kappa": ego_cfg.get("sigma_kappa"),
            "seed": args.seed,
            "n_steps": n_steps,
            "dt": sim.dt,
            "n_interventions": n_intv,
            "velocity_errors": {str(k): v for k, v in vel_errors.items()},
            "kf_Q": ego_ctrl._kf_Q,
            "kf_R": ego_ctrl._kf_R,
            "kf_alpha": ego_ctrl._kf_alpha,
            "kf_revert_alpha": ego_ctrl._kf_revert_alpha,
            "kf_lingap_beta": ego_ctrl._kf_lingap_beta,
            "human_sigma": ego_ctrl._human_sigma,
            "human_mu": ego_ctrl._human_mu,
            "human_walk_Q": ego_ctrl._human_walk_Q,
            "action_noise_std": ego_ctrl.action_noise_std,
        }
        meta_path = os.path.join(run_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n  Results saved to: {run_dir}")
    else:
        print(f"\n  Results not saved (--no-save)")

    # Save scene animation gif
    if args.save:
        n_frames = len(sim.ego.history)
        save_path = os.path.join(run_dir, "scene.gif")
        print(f"  Saving scene animation ({n_frames} frames) ...")

        save_fig, save_ax = plt.subplots(figsize=(14, 5))
        save_renderer = SceneRenderer(sim, title=title)
        save_renderer.fig = save_fig
        save_renderer.ax = save_ax

        def update(frame):
            save_renderer.draw_frame(frame, ego_ctrl)

        anim = animation.FuncAnimation(
            save_fig, update, frames=n_frames,
            interval=args.interval, repeat=False)
        writer = animation.PillowWriter(fps=int(1000 / args.interval))
        anim.save(save_path, writer=writer)
        plt.close(save_fig)
        print(f"  Saved → {save_path}")

    # Close all windows
    plt.close("all")


if __name__ == "__main__":
    main()
