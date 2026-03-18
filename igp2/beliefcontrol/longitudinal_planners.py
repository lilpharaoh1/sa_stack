"""Longitudinal-only MILP and NLP planners.

Mirrors ``first_stage.py`` (MILP, point-mass) and ``second_stage.py``
(NLP, bicycle model) but restricts motion to the reference path (d=0).
Acceleration is the sole free control; steering is an output computed
by the optimizer to satisfy the bicycle dynamics with d=0.

The interfaces are identical to :class:`FirstStagePlanner` and
:class:`SecondStagePlanner` so these can be swapped in wherever the
originals are used.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import casadi as ca

from igp2.beliefcontrol.frenet import FrenetFrame

logger = logging.getLogger(__name__)


# ======================================================================
# Longitudinal First Stage (point-mass, s-only)
# ======================================================================

class LongitudinalFirstStage:
    """First-stage MILP planner restricted to the reference path (d=0).

    State is ``[s, d=0, vs, vd=0]`` for output compatibility with the
    two-stage pipeline.  Internally only ``s`` and ``vs`` are optimised.
    Collision avoidance is longitudinal only (behind / ahead penalties).

    Constructor arguments match :class:`FirstStagePlanner`.
    """

    DEFAULTS = {
        'a_s_min': -3.0,
        'a_s_max': 3.0,
        'jerk_s_max': 1.0,
        'vs_min': 0.0,
        'vs_max': 10.0,
        'w_s': 0.9,
        'w_v': 0.01,
        'w_a_s': 0.5,
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
        """Solve the longitudinal first-stage optimisation.

        Returns (H+1, 4) states ``[s, 0, vs, 0]`` or None on failure.
        """
        H = self._horizon
        dt = self._dt
        beta = 10.0

        p = self._params
        a_s_min, a_s_max = p['a_s_min'], p['a_s_max']
        jerk_s_max = p['jerk_s_max']
        vs_min, vs_max = p['vs_min'], p['vs_max']
        w_s, w_v, w_a_s = p['w_s'], p['w_v'], p['w_a_s']

        s0 = float(frenet_state[0])
        v0 = float(frenet_state[3]) if len(frenet_state) > 3 else float(frenet_state[1])
        vs0 = v0 * np.cos(frenet_state[2]) if len(frenet_state) > 3 else v0

        N_obs = min(len(obstacles), self._n_obs_max)
        v_goal = self._target_speed
        s_goal = min(s0 + v_goal * H * dt, self._frenet.total_length)
        dx = self._ego_length / 2.0

        def softplus(a):
            return ca.fmax(a, 0) + (1.0 / beta) * ca.log(
                1.0 + ca.exp(-beta * ca.fabs(a)))

        try:
            opti = ca.Opti()

            # Decision variables: s and vs only
            s = opti.variable(1, H + 1)
            vs = opti.variable(1, H + 1)
            a_s = opti.variable(1, H)

            # Initial state
            opti.subject_to(s[0] == s0)
            opti.subject_to(vs[0] == vs0)

            # Dynamics (1-D)
            for k in range(H):
                opti.subject_to(s[k + 1] == s[k] + vs[k] * dt)
                opti.subject_to(vs[k + 1] == vs[k] + a_s[k] * dt)

            # State bounds
            for k in range(H + 1):
                opti.subject_to(vs[k] >= vs_min)
                opti.subject_to(vs[k] <= vs_max)

            # Control bounds
            for k in range(H):
                opti.subject_to(opti.bounded(a_s_min, a_s[k], a_s_max))

            # Jerk constraints
            jerk_s_limit = jerk_s_max * dt
            for k in range(H - 1):
                opti.subject_to(opti.bounded(-jerk_s_limit,
                                             a_s[k + 1] - a_s[k],
                                             jerk_s_limit))

            # Cost function
            cost = 0.0
            for k in range(H + 1):
                ref_s = min(s0 + v_goal * k * dt, s_goal)
                cost += w_s * (s[k] - ref_s) ** 2
                cost += w_v * (vs[k] - v_goal) ** 2
            for k in range(H):
                cost += w_a_s * a_s[k] ** 2

            # Collision avoidance (longitudinal only, smooth penalty)
            collision_penalty_weight = 10000.0
            safety_margin = 0.1

            for obs_idx in range(N_obs):
                obs = obstacles[obs_idx]
                obs_half_L = obs['length'] / 2.0

                for k in range(1, H + 1):
                    s_obs = float(obs['s'][k] if k < len(obs['s'])
                                  else obs['s'][-1])

                    gap = dx + obs_half_L + self._collision_margin
                    escape_behind = (s_obs - gap) - s[k]
                    escape_ahead = s[k] - (s_obs + gap)

                    max_escape = ca.fmax(escape_behind, escape_ahead)
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
                opti.set_initial(vs[k], 0.5 * v_goal)
            for k in range(H):
                opti.set_initial(a_s[k], 0)

            sol = opti.solve()

            s_val = sol.value(s).flatten()
            vs_val = sol.value(vs).flatten()

            # Pad to [s, d=0, vs, vd=0] for compatibility
            milp_states = np.column_stack([
                s_val,
                np.zeros(H + 1),
                vs_val,
                np.zeros(H + 1),
            ])
            self._prev_milp_states = milp_states.copy()
            return milp_states

        except Exception as e:
            logger.warning("LongitudinalFirstStage failed: %s", e)
            return None


# ======================================================================
# Longitudinal Second Stage (bicycle model, d=0 enforced)
# ======================================================================

class LongitudinalSecondStage:
    """Second-stage NLP planner with bicycle dynamics constrained to d=0.

    The full bicycle model dynamics are retained:

        s_{k+1}   = s_k   + v_k * cos(phi_k + delta_k) * dt
        d_{k+1}   = d_k   + v_k * sin(phi_k + delta_k) * dt
        phi_{k+1} = phi_k + (2*v_k / L) * sin(delta_k) * dt
        v_{k+1}   = v_k   + a_k * dt

    A hard constraint ``d[k] = 0`` is added for all k.  The optimizer
    therefore finds the steering angle delta that satisfies the bicycle
    dynamics while keeping the vehicle on the reference path.
    Acceleration ``a`` is the sole *free* control variable.

    All other constraints from :class:`SecondStagePlanner` are preserved:
    speed bounds, acceleration bounds, steering bounds, jerk limits,
    steering-rate limits, and elliptical corner-based collision avoidance.

    Constructor arguments and :meth:`solve` signature match
    :class:`SecondStagePlanner`.
    """

    DEFAULTS = {
        'a_min': -3.0,
        'a_max': 3.0,
        'delta_max': 0.65,
        'delta_rate_max': 2.0,
        'jerk_max': 1.0,
        'v_min': 0.0,
        'v_max': 10.0,
        'w_s': 0.1,
        'w_d': 10.0,
        'w_v': 0.01,
        'w_a': 1.0,
        'w_delta': 2.0,
        'w_phi': 2.0,
    }

    def __init__(self,
                 horizon: int,
                 dt: float,
                 ego_length: float,
                 ego_width: float,
                 wheelbase: float,
                 collision_margin: float,
                 target_speed: float,
                 frenet: Optional[FrenetFrame],
                 params: Optional[Dict] = None,
                 n_obs_max: int = 10):
        self._horizon = horizon
        self._dt = dt
        self._ego_length = ego_length
        self._ego_width = ego_width
        self._wheelbase = wheelbase
        self._collision_margin = collision_margin
        self._target_speed = target_speed
        self._frenet = frenet
        self._n_obs_max = n_obs_max

        self._params = dict(self.DEFAULTS)
        if params is not None:
            self._params.update(params)

        self._step_count: int = 0

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
        self._step_count = 0

    def solve(self, frenet_state, warm_states, warm_controls,
              road_left, road_right, obstacles,
              analyse_duals: bool = False, step_label: int = 0,
              ref_controls: Optional[np.ndarray] = None,
              w_agency: float = 1.0,
              agency_only: bool = False):
        """Solve the longitudinal NLP.

        Same signature and return format as
        :meth:`SecondStagePlanner.solve`:
        ``(nlp_states, nlp_controls, success, debug_info)``
        where nlp_states is (H+1, 4) ``[s, d, phi, v]`` and
        nlp_controls is (H, 2) ``[a, delta]``.
        """
        H = self._horizon
        dt = self._dt
        L = self._wheelbase

        nlp = self._params
        a_min, a_max = nlp['a_min'], nlp['a_max']
        delta_max = nlp['delta_max']
        delta_rate_max = nlp['delta_rate_max']
        jerk_max = nlp['jerk_max']
        v_min, v_max = nlp['v_min'], nlp['v_max']
        w_s, w_v = nlp['w_s'], nlp['w_v']
        w_a = nlp['w_a']

        try:
            opti = ca.Opti()

            # Decision variables — full bicycle state + controls
            S = opti.variable(4, H + 1)   # [s; d; phi; v]
            U = opti.variable(2, H)        # [a; delta]

            N_obs = min(len(obstacles), self._n_obs_max)

            s0 = frenet_state[0]
            v_goal = self._target_speed
            s_max = self._frenet.total_length

            # --- Cost function ---
            cost = 0.0
            if not agency_only:
                for k in range(H + 1):
                    ref_s = min(s0 + v_goal * k * dt, s_max)
                    cost += w_s * (S[0, k] - ref_s)**2
                    cost += w_v * (S[3, k] - v_goal)**2
                for k in range(H):
                    cost += w_a * U[0, k]**2

            # Agency-preserving term (acceleration only)
            if ref_controls is not None:
                for k in range(H):
                    cost += w_agency * (U[0, k] - ref_controls[k, 0])**2

            opti.minimize(cost)

            # --- Initial state ---
            opti.subject_to(S[0, 0] == frenet_state[0])
            opti.subject_to(S[1, 0] == frenet_state[1])
            opti.subject_to(S[2, 0] == frenet_state[2])
            opti.subject_to(S[3, 0] == frenet_state[3])

            # --- Bicycle dynamics ---
            for k in range(H):
                opti.subject_to(
                    S[0, k + 1] == S[0, k]
                    + S[3, k] * ca.cos(S[2, k] + U[1, k]) * dt)
                opti.subject_to(
                    S[1, k + 1] == S[1, k]
                    + S[3, k] * ca.sin(S[2, k] + U[1, k]) * dt)
                opti.subject_to(
                    S[2, k + 1] == S[2, k]
                    + (2.0 * S[3, k] / L) * ca.sin(U[1, k]) * dt)
                opti.subject_to(
                    S[3, k + 1] == S[3, k] + U[0, k] * dt)

            # --- Longitudinal constraint: d = 0 for all k ---
            for k in range(H + 1):
                opti.subject_to(S[1, k] == 0.0)

            # --- Road boundary constraints (corner-based) ---
            half_L = self._ego_length / 2.0
            half_W = self._ego_width / 2.0

            for k in range(H + 1):
                cos_phi = ca.cos(S[2, k])
                sin_phi = ca.sin(S[2, k])
                for sl, sw in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    c_d = S[1, k] + sl * half_L * sin_phi + sw * half_W * cos_phi
                    opti.subject_to(c_d >= road_right[k])
                    opti.subject_to(c_d <= road_left[k])

            # --- Control bounds ---
            for k in range(H):
                opti.subject_to(opti.bounded(a_min, U[0, k], a_max))
                opti.subject_to(opti.bounded(-delta_max, U[1, k], delta_max))

            # --- Speed bounds ---
            for k in range(H + 1):
                opti.subject_to(opti.bounded(v_min, S[3, k], v_max))

            # --- Jerk constraints ---
            jerk_limit = jerk_max * dt
            for k in range(H - 1):
                opti.subject_to(opti.bounded(-jerk_limit,
                                             U[0, k + 1] - U[0, k],
                                             jerk_limit))

            # --- Steering rate constraints ---
            delta_rate_limit = delta_rate_max * dt
            for k in range(H - 1):
                opti.subject_to(opti.bounded(-delta_rate_limit,
                                             U[1, k + 1] - U[1, k],
                                             delta_rate_limit))

            # --- Elliptical collision avoidance (corner-based) ---
            half_L = self._ego_length / 2.0
            half_W = self._ego_width / 2.0

            _collision_lam_start = opti.ng
            _collision_meta = []

            for obs_idx in range(N_obs):
                obs = obstacles[obs_idx]
                a_i = obs['length'] / 2.0 + self._collision_margin
                b_i = obs['width'] / 2.0 + self._collision_margin

                obs_s0 = float(obs['s'][0])
                _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
                phi_i = obs.get('heading', road_angle_obs) - road_angle_obs
                cos_phi_i = np.cos(phi_i)
                sin_phi_i = np.sin(phi_i)

                for k in range(1, H + 1):
                    s_obs_k = float(obs['s'][k] if k < len(obs['s'])
                                    else obs['s'][-1])
                    d_obs_k = float(obs['d'][k] if k < len(obs['d'])
                                    else obs['d'][-1])

                    cos_phi_k = ca.cos(S[2, k])
                    sin_phi_k = ca.sin(S[2, k])

                    for alpha_l, alpha_w in [(1, 1), (1, -1),
                                             (-1, 1), (-1, -1)]:
                        c_s = (S[0, k]
                               + alpha_l * half_L * cos_phi_k
                               - alpha_w * half_W * sin_phi_k)
                        c_d = (S[1, k]
                               + alpha_l * half_L * sin_phi_k
                               + alpha_w * half_W * cos_phi_k)

                        d_s = c_s - s_obs_k
                        d_d = c_d - d_obs_k

                        d_body_x = cos_phi_i * d_s + sin_phi_i * d_d
                        d_body_y = -sin_phi_i * d_s + cos_phi_i * d_d

                        g = (d_body_x / a_i)**2 + (d_body_y / b_i)**2
                        opti.subject_to(g >= 1.0)
                        corner = ('FL' if (alpha_l, alpha_w) == (1, 1)
                                  else 'FR' if (alpha_l, alpha_w) == (1, -1)
                                  else 'RL' if (alpha_l, alpha_w) == (-1, 1)
                                  else 'RR')
                        _collision_meta.append((obs_idx, k, corner))

            # --- Initial guess ---
            # TEMPORARY: zero init instead of MILP warmstart
            init_s = np.zeros((H + 1, 4))
            init_c = np.zeros((H, 2))
            opti.set_initial(S, init_s.T)
            opti.set_initial(U, init_c.T)

            # --- Solver options ---
            p_opts = {'expand': True, 'print_time': False}
            s_opts = {
                'max_iter': 10000,
                'warm_start_init_point': 'yes',
                'constr_viol_tol': 1e-1,
                'acceptable_constr_viol_tol': 1e-1,
                'acceptable_tol': 1e-1,
                'acceptable_iter': 5,
                'tol': 1e-1,
                'print_level': 0,
                'sb': 'yes',
            }
            opti.solver('ipopt', p_opts, s_opts)

            # --- Solve ---
            sol = opti.solve()

            nlp_states = sol.value(S).T     # (H+1, 4)
            nlp_controls = sol.value(U).T   # (H, 2)

            # --- Dual analysis ---
            dual_info = None
            if analyse_duals and _collision_meta and N_obs > 0:
                try:
                    lam_g = np.array(sol.value(opti.lam_g)).flatten()
                    n_coll = len(_collision_meta)
                    coll_lam = lam_g[_collision_lam_start:
                                     _collision_lam_start + n_coll]

                    obs_influence = {}
                    for i, (oi, k, corner) in enumerate(_collision_meta):
                        lam_abs = abs(float(coll_lam[i]))
                        obs_influence.setdefault(oi, 0.0)
                        obs_influence[oi] += lam_abs

                    logger.info("[Step %4d] LONGITUDINAL NLP DUAL ANALYSIS "
                                "(%d obstacles):", step_label, N_obs)
                    for oi in sorted(obs_influence.keys()):
                        aid = obstacles[oi].get('agent_id', '?')
                        total = obs_influence[oi]
                        active = "ACTIVE" if total > 1e-4 else "inactive"
                        logger.info("  Agent %s: Σ|λ|=%.4f  [%s]",
                                    aid, total, active)

                    dual_info = {}
                    for oi in obs_influence:
                        aid = obstacles[oi].get('agent_id')
                        if aid is not None:
                            dual_info[aid] = obs_influence[oi]

                except Exception as dual_e:
                    logger.debug("Could not extract duals: %s", dual_e)

            return nlp_states, nlp_controls, True, dual_info

        except Exception as e:
            logger.warning("LongitudinalSecondStage failed: %s", e)

            # --- Diagnostic dump using opti.debug ---
            try:
                S_dbg = opti.debug.value(S).T   # (H+1, 4)
                U_dbg = opti.debug.value(U).T    # (H, 2)

                s_d = S_dbg[:, 0]
                d_d = S_dbg[:, 1]
                phi_d = S_dbg[:, 2]
                v_d = S_dbg[:, 3]
                a_d_arr = U_dbg[:, 0]
                delta_d = U_dbg[:, 1]

                logger.warning("  --- LongSecondStage DIAGNOSTICS (step=%d) ---",
                               step_label)
                logger.warning("  Initial frenet_state: s=%.3f d=%.3f phi=%.4f v=%.3f",
                               frenet_state[0], frenet_state[1],
                               frenet_state[2], frenet_state[3])
                logger.warning("  State ranges: s=[%.2f,%.2f]  d=[%.4f,%.4f]  "
                               "phi=[%.4f,%.4f]  v=[%.2f,%.2f]",
                               s_d.min(), s_d.max(), d_d.min(), d_d.max(),
                               phi_d.min(), phi_d.max(), v_d.min(), v_d.max())
                logger.warning("  Control ranges: a=[%.3f,%.3f]  delta=[%.4f,%.4f]",
                               a_d_arr.min(), a_d_arr.max(),
                               delta_d.min(), delta_d.max())

                # Check d=0 constraint violation
                max_d_viol = float(np.max(np.abs(d_d)))
                if max_d_viol > 1e-3:
                    worst_k = int(np.argmax(np.abs(d_d)))
                    logger.warning("  d=0 VIOLATED: max|d|=%.6f at k=%d", max_d_viol, worst_k)

                # Check velocity bounds
                if v_d.min() < nlp['v_min'] - 1e-3 or v_d.max() > nlp['v_max'] + 1e-3:
                    logger.warning("  Velocity bounds violated: v=[%.3f,%.3f] vs bounds [%.1f,%.1f]",
                                   v_d.min(), v_d.max(), nlp['v_min'], nlp['v_max'])

                # Check acceleration bounds
                if a_d_arr.min() < nlp['a_min'] - 1e-3 or a_d_arr.max() > nlp['a_max'] + 1e-3:
                    logger.warning("  Accel bounds violated: a=[%.3f,%.3f] vs bounds [%.1f,%.1f]",
                                   a_d_arr.min(), a_d_arr.max(), nlp['a_min'], nlp['a_max'])

                # Check jerk
                if len(a_d_arr) > 1:
                    jerk_d = np.diff(a_d_arr) / dt
                    if np.max(np.abs(jerk_d)) > nlp['jerk_max'] + 1e-3:
                        logger.warning("  Jerk violated: |jerk|_max=%.4f vs limit %.4f",
                                       np.max(np.abs(jerk_d)), nlp['jerk_max'])

                # Check road boundaries
                road_viols = 0
                for k in range(min(H + 1, len(S_dbg))):
                    cos_phi = np.cos(phi_d[k])
                    sin_phi = np.sin(phi_d[k])
                    for sl, sw in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        c_d_val = d_d[k] + sl * half_L * sin_phi + sw * half_W * cos_phi
                        if c_d_val > road_left[k] + 1e-3 or c_d_val < road_right[k] - 1e-3:
                            road_viols += 1
                if road_viols > 0:
                    logger.warning("  Road boundary violations: %d corner-steps out of bounds",
                                   road_viols)

                # Check collision avoidance (per-obstacle)
                total_coll_viols = 0
                for obs_idx in range(N_obs):
                    obs = obstacles[obs_idx]
                    a_i = obs['length'] / 2.0 + self._collision_margin
                    b_i = obs['width'] / 2.0 + self._collision_margin
                    obs_s0 = float(obs['s'][0])
                    _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
                    phi_i_obs = obs.get('heading', road_angle_obs) - road_angle_obs
                    cos_phi_i_obs = np.cos(phi_i_obs)
                    sin_phi_i_obs = np.sin(phi_i_obs)

                    obs_viols = 0
                    obs_worst_g = float('inf')
                    obs_worst_k = 0
                    for k in range(1, min(H + 1, len(S_dbg))):
                        s_obs_k = float(obs['s'][k] if k < len(obs['s']) else obs['s'][-1])
                        d_obs_k = float(obs['d'][k] if k < len(obs['d']) else obs['d'][-1])
                        cos_phi_k = np.cos(phi_d[k])
                        sin_phi_k = np.sin(phi_d[k])

                        for al, aw in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            c_s = s_d[k] + al * half_L * cos_phi_k - aw * half_W * sin_phi_k
                            c_d_v = d_d[k] + al * half_L * sin_phi_k + aw * half_W * cos_phi_k
                            d_s_v = c_s - s_obs_k
                            d_d_v = c_d_v - d_obs_k
                            d_bx = cos_phi_i_obs * d_s_v + sin_phi_i_obs * d_d_v
                            d_by = -sin_phi_i_obs * d_s_v + cos_phi_i_obs * d_d_v
                            g_val = (d_bx / a_i)**2 + (d_by / b_i)**2
                            if g_val < 1.0 - 1e-3:
                                obs_viols += 1
                                if g_val < obs_worst_g:
                                    obs_worst_g = g_val
                                    obs_worst_k = k

                    if obs_viols > 0:
                        total_coll_viols += obs_viols
                        wk = obs_worst_k
                        s_obs_wk = float(obs['s'][wk] if wk < len(obs['s']) else obs['s'][-1])
                        d_obs_wk = float(obs['d'][wk] if wk < len(obs['d']) else obs['d'][-1])
                        # World positions if available
                        wp = obs.get('world_positions')
                        wp_str = ""
                        if wp is not None and wk < len(wp):
                            wp_str = f"  obs_world=[{wp[wk,0]:.2f},{wp[wk,1]:.2f}]"
                        logger.warning("  Collision: obs_idx=%d agent=%s  viols=%d  "
                                       "worst g=%.4f at k=%d",
                                       obs_idx, obs.get('agent_id', '?'),
                                       obs_viols, obs_worst_g, wk)
                        logger.warning("    obs s=%.3f d=%.3f  ego s=%.3f d=%.4f  "
                                       "Δs=%.3f Δd=%.3f  ellipse a=%.2f b=%.2f  "
                                       "phi_i=%.4f%s",
                                       s_obs_wk, d_obs_wk,
                                       s_d[wk] if wk < len(s_d) else s_d[-1],
                                       d_d[wk] if wk < len(d_d) else d_d[-1],
                                       s_obs_wk - (s_d[wk] if wk < len(s_d) else s_d[-1]),
                                       d_obs_wk - (d_d[wk] if wk < len(d_d) else d_d[-1]),
                                       a_i, b_i, phi_i_obs, wp_str)
                if total_coll_viols > 0:
                    logger.warning("  Total collision violations: %d across %d obstacles",
                                   total_coll_viols, N_obs)

                # Check if dynamics are consistent (d=0 + bicycle → what steering is needed?)
                if frenet_state[3] > 0.5 and abs(frenet_state[2]) > 0.01:
                    # With d=0 and phi≠0, the dynamics d_{k+1} = d_k + v*sin(phi+delta)*dt = 0
                    # requires sin(phi+delta) = 0, i.e. delta = -phi (mod pi)
                    needed_delta = -frenet_state[2]
                    if abs(needed_delta) > nlp['delta_max']:
                        logger.warning("  INFEASIBLE: initial phi=%.4f requires delta=%.4f "
                                       "to maintain d=0, but delta_max=%.4f",
                                       frenet_state[2], needed_delta, nlp['delta_max'])

                logger.warning("  --- END DIAGNOSTICS ---")

            except Exception as diag_e:
                logger.warning("  Could not extract debug values: %s", diag_e)
                S_dbg, U_dbg = None, None

            return warm_states, warm_controls, False, (S_dbg, U_dbg)

    def analyse_constraints(self, final_states, final_controls,
                            road_left, road_right, obstacles,
                            *, milp_ok, nlp_ok, nlp_status,
                            t_milp, t_nlp) -> Dict:
        """Analyse constraint satisfaction — same as SecondStagePlanner."""
        H = len(final_states) - 1
        dt = self._dt
        nlp = self._params
        half_L = self._ego_length / 2.0
        half_W = self._ego_width / 2.0

        s_traj = final_states[:, 0]
        d_traj = final_states[:, 1]
        phi_traj = final_states[:, 2]
        v_traj = final_states[:, 3]
        a_traj = final_controls[:, 0]
        delta_traj = final_controls[:, 1]

        jerk = np.diff(a_traj) / dt
        delta_rate = np.diff(delta_traj) / dt

        a_min, a_max = nlp['a_min'], nlp['a_max']
        delta_max = nlp['delta_max']
        delta_rate_max = nlp['delta_rate_max']
        jerk_max = nlp['jerk_max']
        v_min, v_max = nlp['v_min'], nlp['v_max']

        vel_violated = bool(np.min(v_traj) < v_min - 1e-3
                            or np.max(v_traj) > v_max + 1e-3)
        accel_violated = bool(np.min(a_traj) < a_min - 1e-3
                              or np.max(a_traj) > a_max + 1e-3)
        steer_violated = bool(np.max(np.abs(delta_traj)) > delta_max + 1e-3)
        jerk_violated = (bool(np.max(np.abs(jerk)) > jerk_max + 1e-3)
                         if len(jerk) > 0 else False)
        steer_rate_violated = (
            bool(np.max(np.abs(delta_rate)) > delta_rate_max + 1e-3)
            if len(delta_rate) > 0 else False)

        # Road boundary violations (corner-based)
        road_violations = []
        for k in range(H + 1):
            cos_phi = np.cos(phi_traj[k])
            sin_phi = np.sin(phi_traj[k])
            for alpha_l, alpha_w, name in [(1, 1, 'FL'), (1, -1, 'FR'),
                                           (-1, 1, 'RL'), (-1, -1, 'RR')]:
                c_d = (d_traj[k] + alpha_l * half_L * sin_phi
                       + alpha_w * half_W * cos_phi)
                margin_left = road_left[k] - c_d
                margin_right = c_d - road_right[k]
                if margin_left < -1e-3 or margin_right < -1e-3:
                    road_violations.append({
                        'k': k, 'corner': name, 'd': float(c_d),
                        'margin_left': float(margin_left),
                        'margin_right': float(margin_right),
                    })

        # Collision violations (ellipse)
        collision_violations = []
        if obstacles:
            for obs_idx, obs in enumerate(obstacles):
                a_i = obs['length'] / 2.0 + self._collision_margin
                b_i = obs['width'] / 2.0 + self._collision_margin

                obs_s0 = float(obs['s'][0])
                _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
                phi_i = obs.get('heading', road_angle_obs) - road_angle_obs
                cos_phi_i = np.cos(phi_i)
                sin_phi_i = np.sin(phi_i)

                for k in range(1, H + 1):
                    s_obs_k = float(obs['s'][k] if k < len(obs['s'])
                                    else obs['s'][-1])
                    d_obs_k = float(obs['d'][k] if k < len(obs['d'])
                                    else obs['d'][-1])
                    cos_phi_k = np.cos(phi_traj[k])
                    sin_phi_k = np.sin(phi_traj[k])

                    for alpha_l, alpha_w, name in [(1, 1, 'FL'), (1, -1, 'FR'),
                                                   (-1, 1, 'RL'),
                                                   (-1, -1, 'RR')]:
                        c_s = (s_traj[k]
                               + alpha_l * half_L * cos_phi_k
                               - alpha_w * half_W * sin_phi_k)
                        c_d = (d_traj[k]
                               + alpha_l * half_L * sin_phi_k
                               + alpha_w * half_W * cos_phi_k)
                        d_s = c_s - s_obs_k
                        d_d = c_d - d_obs_k
                        d_body_x = cos_phi_i * d_s + sin_phi_i * d_d
                        d_body_y = -sin_phi_i * d_s + cos_phi_i * d_d
                        g = (d_body_x / a_i)**2 + (d_body_y / b_i)**2
                        if g < 1.0 - 1e-3:
                            collision_violations.append({
                                'obs_idx': obs_idx, 'k': k,
                                'corner': name, 'g': float(g),
                            })

        any_violated = (not nlp_ok or vel_violated or accel_violated
                        or steer_violated or jerk_violated
                        or steer_rate_violated
                        or len(road_violations) > 0
                        or len(collision_violations) > 0)

        return {
            'step': self._step_count,
            'milp_ok': milp_ok,
            'nlp_ok': nlp_ok,
            'nlp_status': nlp_status,
            't_milp': t_milp,
            't_nlp': t_nlp,
            'velocity_violated': vel_violated,
            'acceleration_violated': accel_violated,
            'steering_violated': steer_violated,
            'jerk_violated': jerk_violated,
            'steer_rate_violated': steer_rate_violated,
            'road_boundary_violations': road_violations,
            'collision_violations': collision_violations,
            'any_violated': any_violated,
            'v_range': (float(np.min(v_traj)), float(np.max(v_traj))),
            'a_range': (float(np.min(a_traj)), float(np.max(a_traj))),
        }
