"""Minimum-intervention NLP optimizer.

Given a reference trajectory (states + controls) from the believed
configuration, finds a feasible control sequence subject to safety
constraints with ALL obstacles visible.

Supports multiple intervention modes (objective functions):

* ``'agency_only'`` — min Σ_k [ (a_k - ref_a_k)² + (δ_k - ref_δ_k)² ]
  Pure agency preservation: smallest control deviation from the believed
  trajectory that satisfies true-world constraints.

* ``'combined'`` — second-stage tracking cost + w_agency × agency term
  Balances driving quality (speed tracking, lane centring, comfort) with
  agency preservation (staying close to the believed controls).

* ``'warmstart_only'`` — pure second-stage tracking cost
  Uses the believed trajectory only as a warm-start; the objective is the
  original SecondStagePlanner cost with no agency-preserving term.

All constraints are identical to SecondStagePlanner: bicycle dynamics,
road boundaries, control/state bounds, jerk/steering-rate smoothness,
and elliptical collision avoidance.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import casadi as ca

from igp2.beliefcontrol.frenet import FrenetFrame
from igp2.beliefcontrol.second_stage import SecondStagePlanner

logger = logging.getLogger(__name__)

# Valid intervention mode strings
INTERVENTION_MODES = ('none', 'agency_only', 'combined', 'warmstart_only')


class SecondStageIntervention:
    """Minimum-deviation NLP: correct believed controls for true safety.

    Finds a feasible control sequence that satisfies all constraints with
    the true obstacle set (all agents visible).  The objective function
    depends on ``mode``:

    * ``'agency_only'``: minimise L2 deviation from believed controls.
    * ``'combined'``: second-stage tracking cost + agency-preserving term.
    * ``'warmstart_only'``: pure second-stage tracking cost (believed
      trajectory used only as warm-start, no agency term).

    Args:
        horizon: Number of planning steps.
        dt: Planning timestep in seconds.
        ego_length: Ego vehicle length (m).
        ego_width: Ego vehicle width (m).
        wheelbase: Wheelbase of the bicycle model (m).
        collision_margin: Extra safety margin around obstacles (m).
        frenet: FrenetFrame for coordinate transforms.
        params: Dict of NLP-specific parameters (uses SecondStagePlanner.DEFAULTS).
        n_obs_max: Maximum number of obstacles to consider.
        target_speed: Desired cruising speed (m/s).  Required for
            ``'combined'`` mode.
        mode: ``'agency_only'`` or ``'combined'``.
        w_agency: Weight on the agency-preserving term in ``'combined'``
            mode.  Ignored for ``'agency_only'``.
    """

    def __init__(self,
                 horizon: int,
                 dt: float,
                 ego_length: float,
                 ego_width: float,
                 wheelbase: float,
                 collision_margin: float,
                 frenet: Optional[FrenetFrame],
                 params: Optional[Dict] = None,
                 n_obs_max: int = 10,
                 target_speed: float = 10.0,
                 mode: str = 'combined',
                 w_agency: float = 1.0):
        if mode not in INTERVENTION_MODES:
            raise ValueError(
                f"Unknown intervention mode {mode!r}; "
                f"expected one of {INTERVENTION_MODES}")

        self._horizon = horizon
        self._dt = dt
        self._ego_length = ego_length
        self._ego_width = ego_width
        self._wheelbase = wheelbase
        self._collision_margin = collision_margin
        self._frenet = frenet
        self._n_obs_max = n_obs_max
        self._target_speed = target_speed
        self._mode = mode
        self._w_agency = w_agency

        self._params = dict(SecondStagePlanner.DEFAULTS)
        if params is not None:
            self._params.update(params)

    @property
    def params(self) -> Dict:
        return dict(self._params)

    def solve(self, frenet_state: np.ndarray,
              ref_states: np.ndarray,
              ref_controls: np.ndarray,
              road_left: np.ndarray,
              road_right: np.ndarray,
              obstacles: list
              ) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        """Solve the minimum-intervention NLP.

        Args:
            frenet_state: [s, d, phi, v] initial state.
            ref_states: (H+1, 4) reference states from believed config.
            ref_controls: (H, 2) reference controls [a, delta] from believed config.
            road_left: (H+1,) left road boundary in Frenet d.
            road_right: (H+1,) right road boundary in Frenet d.
            obstacles: List of obstacle dicts (ALL agents, true visibility).

        Returns:
            (opt_states, opt_controls, success, intervention) where
            intervention = opt_controls - ref_controls.  On failure,
            returns (ref_states, ref_controls, False, zeros).
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

        try:
            opti = ca.Opti()

            # Decision variables
            S = opti.variable(4, H + 1)   # [s; d; phi; v]
            U = opti.variable(2, H)        # [a; delta]

            N_obs = min(len(obstacles), self._n_obs_max)

            # --- Cost function (depends on mode) ---
            cost = 0.0

            if self._mode in ('combined', 'warmstart_only'):
                # Second-stage tracking terms (identical to SecondStagePlanner)
                nlp = self._params
                w_s, w_d, w_v = nlp['w_s'], nlp['w_d'], nlp['w_v']
                w_a, w_delta = nlp['w_a'], nlp['w_delta']
                w_phi = nlp['w_phi']
                v_goal = self._target_speed
                s_max = self._frenet.total_length
                s0 = frenet_state[0]

                for k in range(H + 1):
                    ref_s = min(s0 + v_goal * k * dt, s_max)
                    cost += w_s * (S[0, k] - ref_s)**2
                    cost += w_d * S[1, k]**2
                    cost += w_phi * S[2, k]**2
                    cost += w_v * (S[3, k] - v_goal)**2
                for k in range(H):
                    cost += w_a * U[0, k]**2
                    cost += w_delta * U[1, k]**2

            # Agency-preserving term (not used in warmstart_only)
            if self._mode != 'warmstart_only':
                w_ag = self._w_agency if self._mode == 'combined' else 1.0
                for k in range(H):
                    cost += w_ag * (U[0, k] - ref_controls[k, 0])**2
                    cost += w_ag * (U[1, k] - ref_controls[k, 1])**2

            opti.minimize(cost)

            # --- Initial state constraint ---
            opti.subject_to(S[0, 0] == frenet_state[0])
            opti.subject_to(S[1, 0] == frenet_state[1])
            opti.subject_to(S[2, 0] == frenet_state[2])
            opti.subject_to(S[3, 0] == frenet_state[3])

            # --- Dynamics constraints (bicycle model) ---
            for k in range(H):
                opti.subject_to(
                    S[0, k + 1] == S[0, k] + S[3, k] * ca.cos(S[2, k] + U[1, k]) * dt)
                opti.subject_to(
                    S[1, k + 1] == S[1, k] + S[3, k] * ca.sin(S[2, k] + U[1, k]) * dt)
                opti.subject_to(
                    S[2, k + 1] == S[2, k] + (2.0 * S[3, k] / L) * ca.sin(U[1, k]) * dt)
                opti.subject_to(
                    S[3, k + 1] == S[3, k] + U[0, k] * dt)

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
                    s_obs_k = float(obs['s'][k]) if k < len(obs['s']) else float(obs['s'][-1])
                    d_obs_k = float(obs['d'][k]) if k < len(obs['d']) else float(obs['d'][-1])

                    cos_phi_k = ca.cos(S[2, k])
                    sin_phi_k = ca.sin(S[2, k])

                    for alpha_l, alpha_w in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        c_s = S[0, k] + alpha_l * half_L * cos_phi_k - alpha_w * half_W * sin_phi_k
                        c_d = S[1, k] + alpha_l * half_L * sin_phi_k + alpha_w * half_W * cos_phi_k

                        d_s = c_s - s_obs_k
                        d_d = c_d - d_obs_k

                        d_body_x = cos_phi_i * d_s + sin_phi_i * d_d
                        d_body_y = -sin_phi_i * d_s + cos_phi_i * d_d

                        g = (d_body_x / a_i)**2 + (d_body_y / b_i)**2
                        opti.subject_to(g >= 1.0)

            # --- Warm-start from reference ---
            opti.set_initial(S, ref_states.T)
            opti.set_initial(U, ref_controls.T)

            # --- Solver options (same as SecondStagePlanner) ---
            p_opts = {'expand': True, 'print_time': False}
            s_opts = {
                'max_iter': 10000,
                'warm_start_init_point': 'yes',
                'constr_viol_tol': 1e-3,
                'acceptable_constr_viol_tol': 1e-3,
                'acceptable_tol': 1e-3,
                'acceptable_iter': 5,
                'tol': 1e-3,
                'print_level': 0,
                'sb': 'yes',
            }
            opti.solver('ipopt', p_opts, s_opts)

            # --- Solve ---
            sol = opti.solve()

            opt_states = sol.value(S).T    # (H+1, 4)
            opt_controls = sol.value(U).T  # (H, 2)
            intervention = opt_controls - ref_controls

            return opt_states, opt_controls, True, intervention

        except Exception as e:
            logger.warning("Intervention NLP failed: %s", e)

            # Diagnose constraint violations (same pattern as SecondStagePlanner)
            logger.debug("INTERVENTION NLP FAILURE DIAGNOSIS:")
            S_dbg, U_dbg = None, None
            try:
                S_dbg = opti.debug.value(S).T  # (H+1, 4)
                U_dbg = opti.debug.value(U).T  # (H, 2)

                s_dbg = S_dbg[:, 0]
                d_dbg = S_dbg[:, 1]
                phi_dbg = S_dbg[:, 2]
                v_dbg = S_dbg[:, 3]
                a_dbg = U_dbg[:, 0]
                delta_dbg = U_dbg[:, 1]

                violations = []

                # Check initial state constraints
                if abs(s_dbg[0] - frenet_state[0]) > 1e-3:
                    violations.append(f"Initial s: {s_dbg[0]:.3f} != {frenet_state[0]:.3f}")
                if abs(d_dbg[0] - frenet_state[1]) > 1e-3:
                    violations.append(f"Initial d: {d_dbg[0]:.3f} != {frenet_state[1]:.3f}")
                if abs(phi_dbg[0] - frenet_state[2]) > 1e-3:
                    violations.append(f"Initial phi: {phi_dbg[0]:.3f} != {frenet_state[2]:.3f}")
                if abs(v_dbg[0] - frenet_state[3]) > 1e-3:
                    violations.append(f"Initial v: {v_dbg[0]:.3f} != {frenet_state[3]:.3f}")

                # Check speed bounds
                for k in range(H + 1):
                    if v_dbg[k] < v_min - 1e-3:
                        violations.append(f"v[{k}]={v_dbg[k]:.3f} < v_min={v_min}")
                    if v_dbg[k] > v_max + 1e-3:
                        violations.append(f"v[{k}]={v_dbg[k]:.3f} > v_max={v_max}")

                # Check control bounds
                for k in range(H):
                    if a_dbg[k] < a_min - 1e-3:
                        violations.append(f"a[{k}]={a_dbg[k]:.3f} < a_min={a_min}")
                    if a_dbg[k] > a_max + 1e-3:
                        violations.append(f"a[{k}]={a_dbg[k]:.3f} > a_max={a_max}")
                    if abs(delta_dbg[k]) > delta_max + 1e-3:
                        violations.append(f"|delta[{k}]|={abs(delta_dbg[k]):.3f} > delta_max={delta_max}")

                # Check jerk constraints
                jerk_limit = jerk_max * dt
                for k in range(H - 1):
                    jerk = abs(a_dbg[k + 1] - a_dbg[k])
                    if jerk > jerk_limit + 1e-3:
                        violations.append(f"Jerk[{k}]={jerk:.3f} > limit={jerk_limit:.3f}")

                # Check steering rate constraints
                delta_rate_limit = delta_rate_max * dt
                for k in range(H - 1):
                    delta_rate = abs(delta_dbg[k + 1] - delta_dbg[k])
                    if delta_rate > delta_rate_limit + 1e-3:
                        violations.append(f"DeltaRate[{k}]={delta_rate:.3f} > limit={delta_rate_limit:.3f}")

                # Check road boundary constraints (corner-based)
                half_L = self._ego_length / 2.0
                half_W = self._ego_width / 2.0
                for k in range(H + 1):
                    cos_phi = np.cos(phi_dbg[k])
                    sin_phi = np.sin(phi_dbg[k])
                    for sl, sw in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        c_d = d_dbg[k] + sl * half_L * sin_phi + sw * half_W * cos_phi
                        if c_d < road_right[k] - 1e-3:
                            violations.append(f"Road right[{k}] corner({sl},{sw}): c_d={c_d:.3f} < {road_right[k]:.3f}")
                        if c_d > road_left[k] + 1e-3:
                            violations.append(f"Road left[{k}] corner({sl},{sw}): c_d={c_d:.3f} > {road_left[k]:.3f}")

                # Check collision constraints (ellipse)
                for obs_idx in range(N_obs):
                    obs = obstacles[obs_idx]
                    obs_half_L = obs['length'] / 2.0
                    obs_half_W = obs['width'] / 2.0
                    rx = obs_half_L + self._collision_margin
                    ry = obs_half_W + self._collision_margin

                    obs_s0 = float(obs['s'][0])
                    _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
                    obs_heading_frenet = obs.get('heading', road_angle_obs) - road_angle_obs
                    cos_theta = np.cos(obs_heading_frenet)
                    sin_theta = np.sin(obs_heading_frenet)

                    for k in range(1, H + 1):
                        s_obs = float(obs['s'][k] if k < len(obs['s']) else obs['s'][-1])
                        d_obs = float(obs['d'][k] if k < len(obs['d']) else obs['d'][-1])

                        cos_phi = np.cos(phi_dbg[k])
                        sin_phi = np.sin(phi_dbg[k])

                        for sl, sw in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                            c_s = s_dbg[k] + sl * half_L * cos_phi - sw * half_W * sin_phi
                            c_d = d_dbg[k] + sl * half_L * sin_phi + sw * half_W * cos_phi
                            ds = c_s - s_obs
                            dd = c_d - d_obs
                            x_body = cos_theta * ds + sin_theta * dd
                            y_body = -sin_theta * ds + cos_theta * dd
                            ellipse_val = (x_body / rx)**2 + (y_body / ry)**2
                            if ellipse_val < 1.0 - 1e-3:
                                violations.append(f"Collision obs{obs_idx}[{k}] corner({sl},{sw}): ellipse={ellipse_val:.3f} < 1.0")

                # Log constraint value summaries
                logger.debug("--- Debug State Constraint Values ---")
                logger.debug("Acceleration: min=%.3f, max=%.3f | bounds=[%.2f, %.2f]",
                             np.min(a_dbg), np.max(a_dbg), a_min, a_max)
                logger.debug("Steering (deg): min=%.2f, max=%.2f | bound=+-%.2f",
                             np.degrees(np.min(delta_dbg)), np.degrees(np.max(delta_dbg)), np.degrees(delta_max))
                logger.debug("Velocity: min=%.3f, max=%.3f | bounds=[%.2f, %.2f]",
                             np.min(v_dbg), np.max(v_dbg), v_min, v_max)

                # Jerk and steering rate
                if len(a_dbg) > 1:
                    jerk_vals = np.diff(a_dbg) / dt
                    logger.debug("Jerk: min=%.3f, max=%.3f | bound=+-%.2f",
                                 np.min(jerk_vals), np.max(jerk_vals), jerk_max)
                if len(delta_dbg) > 1:
                    delta_rate_vals = np.diff(delta_dbg) / dt
                    logger.debug("Steer rate (deg/s): min=%.2f, max=%.2f | bound=+-%.2f",
                                 np.degrees(np.min(delta_rate_vals)), np.degrees(np.max(delta_rate_vals)), np.degrees(delta_rate_max))

                # Find minimum g value across all corners and obstacles
                if N_obs > 0:
                    min_g_overall = float('inf')
                    min_g_info = ""
                    for obs_idx in range(N_obs):
                        obs = obstacles[obs_idx]
                        rx = obs['length'] / 2.0 + self._collision_margin
                        ry = obs['width'] / 2.0 + self._collision_margin
                        obs_s0 = float(obs['s'][0])
                        _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
                        obs_heading_frenet = obs.get('heading', road_angle_obs) - road_angle_obs
                        cos_theta = np.cos(obs_heading_frenet)
                        sin_theta = np.sin(obs_heading_frenet)

                        for k in range(1, H + 1):
                            s_obs = float(obs['s'][k] if k < len(obs['s']) else obs['s'][-1])
                            d_obs = float(obs['d'][k] if k < len(obs['d']) else obs['d'][-1])
                            cos_phi = np.cos(phi_dbg[k])
                            sin_phi = np.sin(phi_dbg[k])

                            for sl, sw, name in [(1, 1, 'FL'), (1, -1, 'FR'), (-1, 1, 'RL'), (-1, -1, 'RR')]:
                                c_s = s_dbg[k] + sl * half_L * cos_phi - sw * half_W * sin_phi
                                c_d = d_dbg[k] + sl * half_L * sin_phi + sw * half_W * cos_phi
                                ds = c_s - s_obs
                                dd = c_d - d_obs
                                x_body = cos_theta * ds + sin_theta * dd
                                y_body = -sin_theta * ds + cos_theta * dd
                                g_val = (x_body / rx)**2 + (y_body / ry)**2
                                if g_val < min_g_overall:
                                    min_g_overall = g_val
                                    min_g_info = f"Obs{obs_idx} k={k} {name}: g={g_val}, corner=({c_s:.2f},{c_d:.2f}), obs=({s_obs:.2f},{d_obs:.2f})"

                    status = "VIOLATED" if min_g_overall < 1.0 - 1e-3 else "OK"
                    logger.debug("Collision (min_g): %s (%s) | %s",
                                 min_g_overall, status, min_g_info)

                if violations:
                    logger.debug("Violated constraints (%d total):", len(violations))
                    for v in violations[:10]:
                        logger.debug("  - %s", v)
                    if len(violations) > 10:
                        logger.debug("  ... and %d more", len(violations) - 10)
                else:
                    logger.debug("No obvious constraint violations found in debug values")
                    logger.debug("Initial: s=%.2f, d=%.2f, phi=%.2f, v=%.2f",
                                 frenet_state[0], frenet_state[1], frenet_state[2], frenet_state[3])
                    logger.debug("Ego dimensions: L=%.2f, W=%.2f", 2*half_L, 2*half_W)
                    logger.debug("Road bounds[0]: left=%.2f, right=%.2f, width=%.2f",
                                 road_left[0], road_right[0], road_left[0]-road_right[0])

                    # Show obstacle info and gap analysis
                    if N_obs > 0:
                        logger.debug("Obstacles (%d):", N_obs)
                        for obs_idx in range(min(N_obs, 3)):
                            obs = obstacles[obs_idx]
                            obs_d0 = float(obs['d'][0])
                            obs_half_W = obs['width'] / 2.0
                            obs_left = obs_d0 + obs_half_W + self._collision_margin
                            obs_right = obs_d0 - obs_half_W - self._collision_margin
                            gap_left = road_left[0] - obs_left
                            gap_right = obs_right - road_right[0]
                            logger.debug("  Obs%d: d=%.2f, W=%.2f, gaps: left=%.2f, right=%.2f",
                                         obs_idx, obs_d0, obs['width'], gap_left, gap_right)
                            ego_needed = 2 * half_W + 0.1
                            if gap_left < ego_needed and gap_right < ego_needed:
                                logger.warning("Ego needs %.2fm but gaps are too small!", ego_needed)

            except Exception as debug_e:
                logger.debug("Could not extract debug values: %s", debug_e)

            zeros = np.zeros_like(ref_controls)
            return ref_states, ref_controls, False, zeros
