"""Second-stage planner: NLP (bicycle model) using CasADi + IPOPT.

Solves a nonlinear program over a bicycle model in Frenet coordinates
``[s, d, phi, v]`` with controls ``[a, delta]``.  Minimises tracking
error subject to bicycle dynamics, road boundaries, control/state
bounds, jerk/steering-rate smoothness, and elliptical collision
avoidance constraints.

Based on "A Two-Stage Optimization-based Motion Planner for Safe Urban
Driving" (Eiras et al.).
"""

import logging
from typing import List, Optional, Dict, Tuple

import numpy as np
import casadi as ca

from igp2.beliefcontrol.frenet import FrenetFrame

logger = logging.getLogger(__name__)


class SecondStagePlanner:
    """Second-stage NLP planner (bicycle model) using CasADi + IPOPT.

    Solves a constrained nonlinear program over a bicycle kinematic model
    in Frenet coordinates.  The state is ``[s, d, phi, v]`` (arc-length,
    lateral offset, heading relative to road, speed) and the control is
    ``[a, delta]`` (acceleration, steering angle).

    Dynamics (Frenet bicycle model):
        s_{k+1}   = s_k   + v_k * cos(phi_k + delta_k) * dt
        d_{k+1}   = d_k   + v_k * sin(phi_k + delta_k) * dt
        phi_{k+1} = phi_k + (2*v_k / L) * sin(delta_k) * dt
        v_{k+1}   = v_k   + a_k * dt

    Collision avoidance uses corner-based elliptical constraints per
    Eiras et al. Section III-3, Equations 9-10.

    Args:
        horizon: Number of planning steps.
        dt: Planning timestep in seconds.
        ego_length: Ego vehicle length (m).
        ego_width: Ego vehicle width (m).
        wheelbase: Wheelbase of the bicycle model (m).
        collision_margin: Extra safety margin around obstacles (m).
        target_speed: Desired cruising speed (m/s).
        frenet: FrenetFrame for coordinate transforms.
        params: Dict of NLP-specific parameters (see DEFAULTS).
        n_obs_max: Maximum number of obstacles to consider.
    """

    DEFAULTS = {
        'a_min': -3.0,        # Acceleration min (m/s^2)
        'a_max': 3.0,         # Acceleration max (m/s^2)
        'delta_max': 0.45,    # Max steering angle magnitude (rad)
        'delta_rate_max': 0.18,  # Max steering rate (rad/s)
        'jerk_max': 1.0,      # Max jerk (m/s^3)
        'v_min': 0.0,         # Velocity min (m/s)
        'v_max': 10.0,        # Velocity max (m/s)
        'w_s': 0.1,           # Weight for longitudinal position tracking
        'w_d': 10.0,           # Weight for lateral position tracking
        'w_v': 0.01,           # Weight for velocity tracking
        'w_a': 1.0,           # Weight for acceleration norm
        'w_delta': 2.0,       # Weight for steering angle norm
        'w_phi': 2.0,         # Weight for heading alignment with road
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

        # Step counter for diagnostics
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
              road_left, road_right, obstacles):
        """Solve the NLP using CasADi + IPOPT.

        Bicycle model in Frenet frame:
            s_{k+1} = s_k + v_k * cos(phi_k + delta_k) * dt
            d_{k+1} = d_k + v_k * sin(phi_k + delta_k) * dt
            phi_{k+1} = phi_k + (2*v_k/L) * sin(delta_k) * dt
            v_{k+1} = v_k + a_k * dt

        Args:
            frenet_state: [s, d, phi, v] initial state.
            warm_states: (H+1, 4) initial guess for states.
            warm_controls: (H, 2) initial guess for controls [a, delta].
            road_left: (H+1,) left boundary.
            road_right: (H+1,) right boundary.
            obstacles: List of obstacle dicts.

        Returns:
            (nlp_states, nlp_controls, success, debug_info) --
            (H+1, 4), (H, 2), bool, optional debug tuple.
        """
        H = self._horizon
        dt = self._dt
        L = self._wheelbase

        # NLP-specific parameters
        nlp = self._params
        a_min, a_max = nlp['a_min'], nlp['a_max']
        delta_max = nlp['delta_max']
        delta_rate_max = nlp['delta_rate_max']
        jerk_max = nlp['jerk_max']
        v_min, v_max = nlp['v_min'], nlp['v_max']
        w_s, w_d, w_v = nlp['w_s'], nlp['w_d'], nlp['w_v']
        w_a, w_delta = nlp['w_a'], nlp['w_delta']
        w_phi = nlp['w_phi']

        try:
            opti = ca.Opti()

            # Decision variables
            S = opti.variable(4, H + 1)   # [s; d; phi; v] at each step
            U = opti.variable(2, H)        # [a; delta] at each step

            # Parameters for obstacle positions (pre-allocated slots)
            N_obs = min(len(obstacles), self._n_obs_max)

            # Goal: point ahead if driving at target speed for horizon
            s0 = frenet_state[0]
            v_goal = self._target_speed
            s_max = self._frenet.total_length

            # --- Cost function ---
            cost = 0.0
            for k in range(H + 1):
                # Reference s: where vehicle would be at timestep k driving at target speed
                ref_s = min(s0 + v_goal * k * dt, s_max)
                cost += w_s * (S[0, k] - ref_s)**2
                cost += w_d * S[1, k]**2
                cost += w_phi * S[2, k]**2
                cost += w_v * (S[3, k] - v_goal)**2
            for k in range(H):
                cost += w_a * U[0, k]**2
                cost += w_delta * U[1, k]**2
            opti.minimize(cost)

            # --- Initial state constraint ---
            opti.subject_to(S[0, 0] == frenet_state[0])
            opti.subject_to(S[1, 0] == frenet_state[1])
            opti.subject_to(S[2, 0] == frenet_state[2])
            opti.subject_to(S[3, 0] == frenet_state[3])

            # --- Dynamics constraints ---
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
            # Check all four corners of the rotated ego vehicle stay within road
            half_L = self._ego_length / 2.0
            half_W = self._ego_width / 2.0

            for k in range(H + 1):
                cos_phi = ca.cos(S[2, k])
                sin_phi = ca.sin(S[2, k])

                # Check all four corners: (sl, sw) in {(1,1), (1,-1), (-1,1), (-1,-1)}
                for sl, sw in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    # Corner lateral position in Frenet frame
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
            # Based on Eiras et al. Section III-3, Equations 9-10
            #
            # OBSTACLE REPRESENTATION (Eq. 9):
            #   Each obstacle i at timestep k is an ellipse with:
            #   - Center: (s^i_k, d^i_k) in Frenet coordinates
            #   - Orientation: φ^i_k (heading relative to road)
            #   - Semi-axes: (a^i_k, b^i_k) where a is along length, b along width
            #
            # EGO VEHICLE CORNERS (Eq. 6):
            #   c^α(z_k) = R(φ_k) × [α_l * l/2, α_w * w/2]^T + [s_k, d_k]^T
            #   where α = (α_l, α_w) ∈ {(±1, ±1)} for the 4 corners
            #   R(φ) = [[cos(φ), -sin(φ)], [sin(φ), cos(φ)]]
            #
            # COLLISION CONSTRAINT (Eq. 10): g^{i,α}(z_k) >= 1
            #   g = d^T × R(φ^i)^T × S × R(φ^i) × d
            #   where d = corner - obstacle_center, S = diag(1/a^2, 1/b^2)
            #
            # Equivalently in obstacle body frame:
            #   d_body = R(-φ^i) × d
            #   g = (d_body_x / a)^2 + (d_body_y / b)^2 >= 1
            #
            # Interpretation: g > 1 means corner is OUTSIDE ellipse (safe)

            half_L = self._ego_length / 2.0  # l/2
            half_W = self._ego_width / 2.0   # w/2

            for obs_idx in range(N_obs):
                obs = obstacles[obs_idx]

                # Ellipse semi-axes (Eq. 9): a^i_k, b^i_k
                # a = along obstacle length, b = along obstacle width
                # collision_margin acts as safety buffer (like uncertainty in paper)
                a_i = obs['length'] / 2.0 + self._collision_margin
                b_i = obs['width'] / 2.0 + self._collision_margin

                # Obstacle heading φ^i relative to Frenet s-axis
                obs_s0 = float(obs['s'][0])
                _, _, _, road_angle_obs = self._frenet._interpolate(obs_s0)
                phi_i = obs.get('heading', road_angle_obs) - road_angle_obs

                # Precompute rotation to obstacle body frame: R(-φ^i)
                # R(-φ) = [[cos(φ), sin(φ)], [-sin(φ), cos(φ)]]
                cos_phi_i = np.cos(phi_i)
                sin_phi_i = np.sin(phi_i)

                for k in range(1, H + 1):
                    # Obstacle center (s^i_k, d^i_k) at timestep k
                    s_obs_k = float(obs['s'][k]) if k < len(obs['s']) else float(obs['s'][-1])
                    d_obs_k = float(obs['d'][k]) if k < len(obs['d']) else float(obs['d'][-1])

                    # Ego heading φ_k at timestep k
                    cos_phi_k = ca.cos(S[2, k])
                    sin_phi_k = ca.sin(S[2, k])

                    # Check all 4 corners: α = (α_l, α_w) ∈ {(±1, ±1)}
                    for alpha_l, alpha_w in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        # Corner position c^α (Eq. 6):
                        # c = R(φ_k) × [α_l * l/2, α_w * w/2]^T + [s_k, d_k]^T
                        c_s = S[0, k] + alpha_l * half_L * cos_phi_k - alpha_w * half_W * sin_phi_k
                        c_d = S[1, k] + alpha_l * half_L * sin_phi_k + alpha_w * half_W * cos_phi_k

                        # Vector from obstacle center to corner: d = c - (s^i_k, d^i_k)
                        d_s = c_s - s_obs_k
                        d_d = c_d - d_obs_k

                        # Rotate to obstacle body frame: d_body = R(-φ^i) × d
                        # R(-φ) = [[cos(φ), sin(φ)], [-sin(φ), cos(φ)]]
                        d_body_x = cos_phi_i * d_s + sin_phi_i * d_d
                        d_body_y = -sin_phi_i * d_s + cos_phi_i * d_d

                        # Ellipse containment function g (Eq. 10):
                        # g = (d_body_x / a)^2 + (d_body_y / b)^2
                        # Constraint: g >= 1 (corner must be outside ellipse)
                        g = (d_body_x / a_i)**2 + (d_body_y / b_i)**2
                        opti.subject_to(g >= 1.0)

            # --- Initial guess ---
            opti.set_initial(S, warm_states.T)
            opti.set_initial(U, warm_controls.T)

            # --- Solver options ---
            p_opts = {'expand': True, 'print_time': False}
            s_opts = {
                'max_iter': 10000,
                'warm_start_init_point': 'yes',
                # Relax constraint tolerance
                'constr_viol_tol': 1e-3,           # Allow small violations
                'acceptable_constr_viol_tol': 1e-3, # Accept slightly larger violations
                'acceptable_tol': 1e-3,
                'acceptable_iter': 5,
                # Optional: also relax optimality tolerance
                'tol': 1e-3,
                'print_level': 0,
                'sb': 'yes',
            }
            opti.solver('ipopt', p_opts, s_opts)

            # --- Solve ---
            sol = opti.solve()

            nlp_states = sol.value(S).T   # (H+1, 4)
            nlp_controls = sol.value(U).T  # (H, 2)

            return nlp_states, nlp_controls, True, None

        except Exception as e:
            print("NLP solver failed: %s", e)
            logger.debug("NLP solver failed: %s", e)

            # Diagnose constraint violations
            print(f"  NLP FAILURE DIAGNOSIS:")
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
                    # R(-φ) = [[cos(φ), sin(φ)], [-sin(φ), cos(φ)]]
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
                            # R(-φ) × d
                            x_body = cos_theta * ds + sin_theta * dd
                            y_body = -sin_theta * ds + cos_theta * dd
                            ellipse_val = (x_body / rx)**2 + (y_body / ry)**2
                            if ellipse_val < 1.0 - 1e-3:
                                violations.append(f"Collision obs{obs_idx}[{k}] corner({sl},{sw}): ellipse={ellipse_val:.3f} < 1.0")

                # Always print constraint value summaries
                print(f"    --- Debug State Constraint Values ---")
                print(f"    Acceleration: min={np.min(a_dbg):.3f}, max={np.max(a_dbg):.3f} | bounds=[{a_min:.2f}, {a_max:.2f}]")
                print(f"    Steering (deg): min={np.degrees(np.min(delta_dbg)):.2f}, max={np.degrees(np.max(delta_dbg)):.2f} | bound=\u00b1{np.degrees(delta_max):.2f}")
                print(f"    Velocity: min={np.min(v_dbg):.3f}, max={np.max(v_dbg):.3f} | bounds=[{v_min:.2f}, {v_max:.2f}]")

                # Jerk and steering rate
                if len(a_dbg) > 1:
                    jerk_vals = np.diff(a_dbg) / dt
                    print(f"    Jerk: min={np.min(jerk_vals):.3f}, max={np.max(jerk_vals):.3f} | bound=\u00b1{jerk_max:.2f}")
                if len(delta_dbg) > 1:
                    delta_rate_vals = np.diff(delta_dbg) / dt
                    print(f"    Steer rate (deg/s): min={np.degrees(np.min(delta_rate_vals)):.2f}, max={np.degrees(np.max(delta_rate_vals)):.2f} | bound=\u00b1{np.degrees(delta_rate_max):.2f}")

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
                    print(f"    Collision (min_g): {min_g_overall} ({status}) | {min_g_info}")

                if violations:
                    print(f"    Violated constraints ({len(violations)} total):")
                    for v in violations[:10]:  # Show first 10
                        print(f"      - {v}")
                    if len(violations) > 10:
                        print(f"      ... and {len(violations) - 10} more")
                else:
                    print(f"    No obvious constraint violations found in debug values")
                    print(f"    Initial: s={frenet_state[0]:.2f}, d={frenet_state[1]:.2f}, phi={frenet_state[2]:.2f}, v={frenet_state[3]:.2f}")
                    print(f"    Ego dimensions: L={2*half_L:.2f}, W={2*half_W:.2f}")
                    print(f"    Road bounds[0]: left={road_left[0]:.2f}, right={road_right[0]:.2f}, width={road_left[0]-road_right[0]:.2f}")

                    # Show obstacle info and gap analysis
                    if N_obs > 0:
                        print(f"    Obstacles ({N_obs}):")
                        for obs_idx in range(min(N_obs, 3)):
                            obs = obstacles[obs_idx]
                            obs_d0 = float(obs['d'][0])
                            obs_half_W = obs['width'] / 2.0
                            # Obstacle occupies [obs_d0 - obs_half_W, obs_d0 + obs_half_W]
                            obs_left = obs_d0 + obs_half_W + self._collision_margin
                            obs_right = obs_d0 - obs_half_W - self._collision_margin
                            gap_left = road_left[0] - obs_left  # Gap between obstacle and left road edge
                            gap_right = obs_right - road_right[0]  # Gap between obstacle and right road edge
                            print(f"      Obs{obs_idx}: d={obs_d0:.2f}, W={obs['width']:.2f}, gaps: left={gap_left:.2f}, right={gap_right:.2f}")
                            # Check if ego can fit in either gap
                            ego_needed = 2 * half_W + 0.1  # Width needed + small margin
                            if gap_left < ego_needed and gap_right < ego_needed:
                                print(f"      WARNING: Ego needs {ego_needed:.2f}m but gaps are too small!")

            except Exception as debug_e:
                print(f"    (Could not extract debug values: {debug_e})")
                S_dbg, U_dbg = None, None

            return warm_states, warm_controls, False, (S_dbg, U_dbg)

    def analyse_constraints(self, final_states, final_controls,
                            road_left, road_right, obstacles,
                            *, milp_ok, nlp_ok, nlp_status,
                            t_milp, t_nlp) -> Dict:
        """Analyse constraint satisfaction and return structured diagnostics.

        Returns a dict with grouped constraint violation info.
        """
        H = len(final_states) - 1
        dt = self._dt
        nlp = self._params
        half_L = self._ego_length / 2.0
        half_W = self._ego_width / 2.0

        # Extract state and control trajectories
        s_traj = final_states[:, 0]
        d_traj = final_states[:, 1]
        phi_traj = final_states[:, 2]
        v_traj = final_states[:, 3]
        a_traj = final_controls[:, 0]
        delta_traj = final_controls[:, 1]

        # Derived quantities
        jerk = np.diff(a_traj) / dt   # (H-1,)
        delta_rate = np.diff(delta_traj) / dt  # (H-1,)

        # Constraint bounds
        a_min, a_max = nlp['a_min'], nlp['a_max']
        delta_max = nlp['delta_max']
        delta_rate_max = nlp['delta_rate_max']
        jerk_max = nlp['jerk_max']
        v_min, v_max = nlp['v_min'], nlp['v_max']

        # -- Velocity bounds --
        vel_violated = bool(np.min(v_traj) < v_min - 1e-3
                            or np.max(v_traj) > v_max + 1e-3)

        # -- Acceleration bounds --
        accel_violated = bool(np.min(a_traj) < a_min - 1e-3
                              or np.max(a_traj) > a_max + 1e-3)

        # -- Steering bounds --
        steer_violated = bool(np.max(np.abs(delta_traj)) > delta_max + 1e-3)

        # -- Jerk bounds --
        jerk_violated = (bool(np.max(np.abs(jerk)) > jerk_max + 1e-3)
                         if len(jerk) > 0 else False)

        # -- Steering rate bounds --
        steer_rate_violated = (bool(np.max(np.abs(delta_rate)) > delta_rate_max + 1e-3)
                               if len(delta_rate) > 0 else False)

        # -- Road boundary violations (corner-based) --
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

        # -- Collision constraint violations (ellipse g-value) --
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
                    s_obs_k = float(obs['s'][k]) if k < len(obs['s']) else float(obs['s'][-1])
                    d_obs_k = float(obs['d'][k]) if k < len(obs['d']) else float(obs['d'][-1])
                    cos_phi_k = np.cos(phi_traj[k])
                    sin_phi_k = np.sin(phi_traj[k])

                    for alpha_l, alpha_w, name in [(1, 1, 'FL'), (1, -1, 'FR'),
                                                   (-1, 1, 'RL'), (-1, -1, 'RR')]:
                        c_s = (s_traj[k] + alpha_l * half_L * cos_phi_k
                               - alpha_w * half_W * sin_phi_k)
                        c_d = (d_traj[k] + alpha_l * half_L * sin_phi_k
                               + alpha_w * half_W * cos_phi_k)
                        d_s = c_s - s_obs_k
                        d_d = c_d - d_obs_k
                        d_body_x = cos_phi_i * d_s + sin_phi_i * d_d
                        d_body_y = -sin_phi_i * d_s + cos_phi_i * d_d
                        g = (d_body_x / a_i)**2 + (d_body_y / b_i)**2

                        if g < 1.0 - 1e-3:
                            collision_violations.append({
                                'obs_idx': obs_idx, 'k': k, 'corner': name,
                                'g': float(g),
                            })

        any_violated = (not nlp_ok or vel_violated or accel_violated
                        or steer_violated or jerk_violated
                        or steer_rate_violated
                        or len(road_violations) > 0
                        or len(collision_violations) > 0)

        return {
            # Solver status
            'step': self._step_count,
            'milp_ok': milp_ok,
            'nlp_ok': nlp_ok,
            'nlp_status': nlp_status,
            't_milp': t_milp,
            't_nlp': t_nlp,

            # Input / state bound violations
            'velocity_violated': vel_violated,
            'acceleration_violated': accel_violated,
            'steering_violated': steer_violated,
            'jerk_violated': jerk_violated,
            'steer_rate_violated': steer_rate_violated,

            # Spatial violations
            'road_boundary_violations': road_violations,
            'collision_violations': collision_violations,

            # Summary
            'any_violated': any_violated,

            # Ranges (for logging)
            'v_range': (float(np.min(v_traj)), float(np.max(v_traj))),
            'a_range': (float(np.min(a_traj)), float(np.max(a_traj))),
        }
