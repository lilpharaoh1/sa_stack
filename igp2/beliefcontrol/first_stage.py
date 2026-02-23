"""First-stage planner: smooth MILP-like optimisation over a point-mass model.

Solves a CasADi/IPOPT smooth approximation of MILP over a point-mass
model ``[s, d, vs, vd]`` in Frenet coordinates.  Minimises L2 tracking
error subject to ZOH dynamics, kinematic feasibility, road boundaries,
and smooth penalty-based collision avoidance.
"""

import logging
from typing import List, Optional, Dict

import numpy as np
import casadi as ca

from igp2.beliefcontrol.frenet import FrenetFrame

logger = logging.getLogger(__name__)


class FirstStagePlanner:
    """First-stage Frenet-frame trajectory optimisation (point-mass model).

    Based on "A Two-Stage Optimization-based Motion Planner for Safe Urban
    Driving" (Eiras et al.), using smooth softplus approximation of max()
    instead of true MILP formulation.

    The ego vehicle is treated as a **point mass** and obstacles as
    **rectangles** enlarged by ego dimensions (Minkowski sum).

    Args:
        horizon: Number of planning steps.
        dt: Planning timestep in seconds.
        ego_length: Ego vehicle length (m).
        ego_width: Ego vehicle width (m).
        collision_margin: Extra safety margin around obstacles (m).
        target_speed: Desired cruising speed (m/s).
        frenet: FrenetFrame for coordinate transforms.
        params: Dict of MILP-specific parameters (see DEFAULTS).
        n_obs_max: Maximum number of obstacles to consider.
    """

    DEFAULTS = {
        'a_s_min': -3.0,
        'a_s_max': 3.0,
        'a_d_min': -3.0,
        'a_d_max': 3.0,
        'jerk_s_max': 1.0,
        'jerk_d_max': 1.0,
        'vs_min': 0.0,
        'vs_max': 10.0,
        'vd_min': -10.0,
        'vd_max': 10.0,
        'rho': 1.5,
        'w_s': 0.9,
        'w_d': 10.0,
        'w_v': 0.01,
        'w_a_s': 0.5,
        'w_a_d': 0.5,
    }

    def __init__(self,
                 horizon: int,
                 dt: float,
                 ego_length: float,
                 ego_width: float,
                 collision_margin: float,
                 target_speed: float,
                 frenet: Optional[FrenetFrame],
                 params: Optional[Dict] = None,
                 n_obs_max: int = 10):
        self._horizon = horizon
        self._dt = dt
        self._ego_length = ego_length
        self._ego_width = ego_width
        self._collision_margin = collision_margin
        self._target_speed = target_speed
        self._frenet = frenet
        self._n_obs_max = n_obs_max

        self._params = dict(self.DEFAULTS)
        if params is not None:
            self._params.update(params)

        # MPC state
        self._prev_milp_states: Optional[np.ndarray] = None

    @property
    def params(self) -> Dict:
        return dict(self._params)

    @property
    def frenet(self) -> Optional[FrenetFrame]:
        return self._frenet

    @frenet.setter
    def frenet(self, value: FrenetFrame):
        self._frenet = value

    def reset(self):
        self._prev_milp_states = None

    def solve(self, frenet_state: np.ndarray,
              road_left: np.ndarray,
              road_right: np.ndarray,
              obstacles: List[Dict]) -> Optional[np.ndarray]:
        """Solve the first-stage optimisation.

        Args:
            frenet_state: [s, d, phi, v] current state in Frenet.
            road_left: (H+1,) left boundary offsets (positive).
            road_right: (H+1,) right boundary offsets (negative).
            obstacles: List of obstacle dicts from predict_obstacles.

        Returns:
            (H+1, 4) array of states [s, d, vs, vd] in Frenet,
            or None if solver failed.
        """
        H = self._horizon
        dt = self._dt
        M = 1e4
        beta = 10.0

        p = self._params
        rho = p['rho']
        a_s_min, a_s_max = p['a_s_min'], p['a_s_max']
        a_d_min, a_d_max = p['a_d_min'], p['a_d_max']
        jerk_s_max, jerk_d_max = p['jerk_s_max'], p['jerk_d_max']
        vs_min, vs_max = p['vs_min'], p['vs_max']
        vd_min, vd_max = p['vd_min'], p['vd_max']
        w_s, w_d, w_v = p['w_s'], p['w_d'], p['w_v']
        w_a_s, w_a_d = p['w_a_s'], p['w_a_d']

        s0, d0, phi0, v0 = frenet_state
        vs0 = v0 * np.cos(phi0)
        vd0 = v0 * np.sin(phi0)

        N_obs = min(len(obstacles), self._n_obs_max)

        v_goal = self._target_speed
        s_goal = min(s0 + v_goal * H * dt, self._frenet.total_length)

        dx = self._ego_length / 2.0
        dy = self._ego_width / 2.0

        def softplus(a):
            return ca.fmax(a, 0) + (1.0 / beta) * ca.log(1.0 + ca.exp(-beta * ca.fabs(a)))

        try:
            opti = ca.Opti()

            S = opti.variable(4, H + 1)
            s = S[0, :]
            d = S[1, :]
            vs = S[2, :]
            vd = S[3, :]

            U = opti.variable(2, H)
            a_s = U[0, :]
            a_d = U[1, :]

            # Initial state
            opti.subject_to(s[0] == s0)
            opti.subject_to(d[0] == d0)
            opti.subject_to(vs[0] == vs0)
            opti.subject_to(vd[0] == vd0)

            # Dynamics (ZOH)
            for k in range(H):
                opti.subject_to(s[k + 1] == s[k] + vs[k] * dt)
                opti.subject_to(d[k + 1] == d[k] + vd[k] * dt)
                opti.subject_to(vs[k + 1] == vs[k] + a_s[k] * dt)
                opti.subject_to(vd[k + 1] == vd[k] + a_d[k] * dt)

            # Kinematic feasibility
            for k in range(H + 1):
                opti.subject_to(vs[k] >= rho * vd[k])
                opti.subject_to(vs[k] >= -rho * vd[k])

            # State bounds
            for k in range(H + 1):
                opti.subject_to(vs[k] >= vs_min)
                opti.subject_to(vs[k] <= vs_max)
                opti.subject_to(vd[k] >= vd_min)
                opti.subject_to(vd[k] <= vd_max)

            # Control bounds
            for k in range(H):
                opti.subject_to(opti.bounded(a_s_min, a_s[k], a_s_max))
                opti.subject_to(opti.bounded(a_d_min, a_d[k], a_d_max))

            # Jerk constraints
            jerk_s_limit = jerk_s_max * dt
            jerk_d_limit = jerk_d_max * dt
            for k in range(H - 1):
                opti.subject_to(opti.bounded(-jerk_s_limit,
                                             a_s[k + 1] - a_s[k],
                                             jerk_s_limit))
                opti.subject_to(opti.bounded(-jerk_d_limit,
                                             a_d[k + 1] - a_d[k],
                                             jerk_d_limit))

            # Road boundaries
            for k in range(H + 1):
                opti.subject_to(d[k] >= road_right[k])
                opti.subject_to(d[k] <= road_left[k])

            # Cost function
            cost = 0.0

            for k in range(H + 1):
                ref_s = min(s0 + v_goal * k * dt, s_goal)
                ref_d = 0.0
                cost += w_s * (s[k] - ref_s) ** 2
                cost += w_d * (d[k] - ref_d) ** 2
                cost += w_v * (vs[k] - v_goal) ** 2

            for k in range(H):
                cost += w_a_s * a_s[k] ** 2
                cost += w_a_d * a_d[k] ** 2

            # Collision avoidance (smooth penalty)
            def smooth_max4(a, b, c, e):
                max_val = ca.fmax(ca.fmax(a, b), ca.fmax(c, e))
                return max_val + (1.0 / beta) * ca.log(
                    ca.exp(beta * (a - max_val)) +
                    ca.exp(beta * (b - max_val)) +
                    ca.exp(beta * (c - max_val)) +
                    ca.exp(beta * (e - max_val))
                )

            collision_penalty_weight = 10000.0
            safety_margin = 0.1

            for obs_idx in range(N_obs):
                obs = obstacles[obs_idx]

                obs_half_L = obs['length'] / 2.0
                obs_half_W = obs['width'] / 2.0
                obs_s0 = float(obs['s'][0])
                _, _, _, road_angle = self._frenet._interpolate(obs_s0)
                dh = obs.get('heading', road_angle) - road_angle

                obs_a = abs(obs_half_L * np.cos(dh)) + abs(obs_half_W * np.sin(dh)) + self._collision_margin
                obs_b = abs(obs_half_L * np.sin(dh)) + abs(obs_half_W * np.cos(dh)) + self._collision_margin

                for k in range(1, H + 1):
                    s_obs = float(obs['s'][k] if k < len(obs['s']) else obs['s'][-1])
                    d_obs = float(obs['d'][k] if k < len(obs['d']) else obs['d'][-1])

                    s_min_val = s_obs - obs_a - dx
                    s_max_val = s_obs + obs_a + dx
                    d_min_val = d_obs - obs_b - dy
                    d_max_val = d_obs + obs_b + dy

                    escape_behind = s_min_val - s[k]
                    escape_ahead = s[k] - s_max_val
                    escape_right = d_min_val - d[k]
                    escape_left = d[k] - d_max_val

                    max_escape = smooth_max4(escape_behind, escape_ahead, escape_right, escape_left)
                    violation = softplus(safety_margin - max_escape)
                    cost += collision_penalty_weight * violation ** 2

            opti.minimize(cost)

            # Solver options
            p_opts = {'expand': True, 'print_time': False}
            s_opts = {
                'max_iter': 1000,
                'constr_viol_tol': 1e-3,
                'acceptable_constr_viol_tol': 1e-3,
                'acceptable_tol': 1e-3,
                'acceptable_iter': 5,
                'tol': 1e-3,
                'print_level': 0,
                'sb': 'yes',
            }
            opti.solver('ipopt', p_opts, s_opts)

            # Initial guess
            for k in range(H + 1):
                opti.set_initial(s[k], s0 + 0.5 * v_goal * k * dt)
                opti.set_initial(d[k], d0)
                opti.set_initial(vs[k], 0.5 * v_goal)
                opti.set_initial(vd[k], 0)
            for k in range(H):
                opti.set_initial(a_s[k], 0)
                opti.set_initial(a_d[k], 0)

            sol = opti.solve()

            s_val = sol.value(s).flatten()
            d_val = sol.value(d).flatten()
            vs_val = sol.value(vs).flatten()
            vd_val = sol.value(vd).flatten()

            milp_states = np.column_stack([s_val, d_val, vs_val, vd_val])
            self._prev_milp_states = milp_states.copy()
            return milp_states

        except Exception as e:
            logger.warning("Stage1 (MILP) failed: %s", e)

            logger.debug("MILP FAILURE DIAGNOSIS:")
            try:
                s_dbg = opti.debug.value(s).flatten()
                d_dbg = opti.debug.value(d).flatten()
                vs_dbg = opti.debug.value(vs).flatten()
                vd_dbg = opti.debug.value(vd).flatten()
                a_s_dbg = opti.debug.value(a_s).flatten()
                a_d_dbg = opti.debug.value(a_d).flatten()

                violations = []

                if abs(s_dbg[0] - s0) > 1e-3:
                    violations.append(f"Initial s: {s_dbg[0]:.3f} != {s0:.3f}")
                if abs(d_dbg[0] - d0) > 1e-3:
                    violations.append(f"Initial d: {d_dbg[0]:.3f} != {d0:.3f}")
                if abs(vs_dbg[0] - vs0) > 1e-3:
                    violations.append(f"Initial vs: {vs_dbg[0]:.3f} != {vs0:.3f}")
                if abs(vd_dbg[0] - vd0) > 1e-3:
                    violations.append(f"Initial vd: {vd_dbg[0]:.3f} != {vd0:.3f}")

                for k in range(H + 1):
                    if vs_dbg[k] < vs_min - 1e-3:
                        violations.append(f"vs[{k}]={vs_dbg[k]:.3f} < vs_min={vs_min}")
                    if vs_dbg[k] > vs_max + 1e-3:
                        violations.append(f"vs[{k}]={vs_dbg[k]:.3f} > vs_max={vs_max}")
                    if vd_dbg[k] < vd_min - 1e-3:
                        violations.append(f"vd[{k}]={vd_dbg[k]:.3f} < vd_min={vd_min}")
                    if vd_dbg[k] > vd_max + 1e-3:
                        violations.append(f"vd[{k}]={vd_dbg[k]:.3f} > vd_max={vd_max}")

                for k in range(H + 1):
                    if vs_dbg[k] < rho * abs(vd_dbg[k]) - 1e-3:
                        violations.append(f"Kinematic[{k}]: vs={vs_dbg[k]:.3f} < rho*|vd|={rho * abs(vd_dbg[k]):.3f}")

                for k in range(H + 1):
                    if d_dbg[k] < road_right[k] - 1e-3:
                        violations.append(f"Road right[{k}]: d={d_dbg[k]:.3f} < {road_right[k]:.3f}")
                    if d_dbg[k] > road_left[k] + 1e-3:
                        violations.append(f"Road left[{k}]: d={d_dbg[k]:.3f} > {road_left[k]:.3f}")

                for k in range(H):
                    if a_s_dbg[k] < a_s_min - 1e-3:
                        violations.append(f"a_s[{k}]={a_s_dbg[k]:.3f} < a_s_min={a_s_min}")
                    if a_s_dbg[k] > a_s_max + 1e-3:
                        violations.append(f"a_s[{k}]={a_s_dbg[k]:.3f} > a_s_max={a_s_max}")
                    if a_d_dbg[k] < a_d_min - 1e-3:
                        violations.append(f"a_d[{k}]={a_d_dbg[k]:.3f} < a_d_min={a_d_min}")
                    if a_d_dbg[k] > a_d_max + 1e-3:
                        violations.append(f"a_d[{k}]={a_d_dbg[k]:.3f} > a_d_max={a_d_max}")

                for k in range(H - 1):
                    jerk_s = a_s_dbg[k + 1] - a_s_dbg[k]
                    jerk_d = a_d_dbg[k + 1] - a_d_dbg[k]
                    if abs(jerk_s) > jerk_s_limit + 1e-3:
                        violations.append(f"Jerk_s[{k}]={jerk_s:.3f} > limit={jerk_s_limit:.3f}")
                    if abs(jerk_d) > jerk_d_limit + 1e-3:
                        violations.append(f"Jerk_d[{k}]={jerk_d:.3f} > limit={jerk_d_limit:.3f}")

                if violations:
                    logger.debug("Violated constraints (%d total):", len(violations))
                    for v in violations[:10]:
                        logger.debug("  - %s", v)
                    if len(violations) > 10:
                        logger.debug("  ... and %d more", len(violations) - 10)
                else:
                    logger.debug("No obvious constraint violations found in debug values")
                    logger.debug("Initial state: s0=%.2f, d0=%.2f, vs0=%.2f, vd0=%.2f",
                                 s0, d0, vs0, vd0)
                    logger.debug("Road bounds[0]: left=%.2f, right=%.2f",
                                 road_left[0], road_right[0])

            except Exception as debug_e:
                logger.debug("Could not extract debug values: %s", debug_e)

            return None
