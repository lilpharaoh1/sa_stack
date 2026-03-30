"""Kalman-filtered awareness dynamics for per-belief MCTS.

Tracks the robot's estimate of the human driver's awareness of each
traffic participant using a diagonal Kalman filter in logit space.

The continuous awareness state psi_i in R maps to awareness probability
phi_i in [0, 1] via the sigmoid: phi = 1 / (1 + exp(-psi)).

Feature computation uses a dual radial-basis-function (RBF) kernel:
  - RBF 1: omnidirectional proximity (narrow spread sigma_1)
  - RBF 2: forward field-of-view gated, wider spread (sigma_2)
"""

import math
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm as _norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_feature_world(ego_xy: np.ndarray,
                          ego_heading: float,
                          participant_xy: np.ndarray,
                          sigma_1: float = 15.0,
                          sigma_2: float = 25.0,
                          fov_half_angle: float = math.radians(30),
                          w1: float = 0.7,
                          w2: float = 0.3,
                          ) -> float:
    """Dual-RBF awareness feature in world coordinates.

    Args:
        ego_xy: (2,) ego position [x, y].
        ego_heading: Ego heading in radians.
        participant_xy: (2,) participant position [x, y].
        sigma_1: Narrow RBF spread (omnidirectional proximity).
        sigma_2: Wide RBF spread (forward FOV detection).
        fov_half_angle: Half-angle of forward field of view (radians).
        w1: Weight for omnidirectional RBF.
        w2: Weight for forward FOV RBF.

    Returns:
        Scalar feature value >= 0.
    """
    diff = participant_xy - ego_xy
    dist_sq = float(np.dot(diff, diff))

    # RBF 1: omnidirectional proximity
    rbf_1 = math.exp(-dist_sq / (2.0 * sigma_1 ** 2))

    # RBF 2: forward FOV gated
    rbf_2 = math.exp(-dist_sq / (2.0 * sigma_2 ** 2))

    angle_to = math.atan2(diff[1], diff[0])
    rel_angle = (angle_to - ego_heading + math.pi) % (2.0 * math.pi) - math.pi
    if abs(rel_angle) > fov_half_angle:
        rbf_2 = 0.0

    return w1 * rbf_1 + w2 * rbf_2


def compute_feature_frenet(s_ego: float,
                           d_ego: float,
                           s_obs: float,
                           d_obs: float,
                           sigma_1: float = 15.0,
                           sigma_2: float = 25.0,
                           w1: float = 0.7,
                           w2: float = 0.3,
                           fov_half_angle: float = math.radians(45),
                           ) -> float:
    """Dual-RBF awareness feature in Frenet coordinates.

    Uses 2-D Frenet distance sqrt((ds)^2 + (dd)^2).  The forward RBF is
    gated by a cone in Frenet space: the participant must be ahead
    (ds > 0) and within ``fov_half_angle`` of the forward (s) direction,
    i.e. ds >= cot(fov_half_angle) * |dd|.

    Args:
        s_ego: Ego arc-length position.
        d_ego: Ego lateral offset (typically 0 in the MCTS planner).
        s_obs: Participant arc-length position.
        d_obs: Participant lateral offset.
        sigma_1: Narrow RBF spread (omnidirectional).
        sigma_2: Wide RBF spread (forward).
        w1: Weight for omnidirectional RBF.
        w2: Weight for forward FOV RBF.
        fov_half_angle: Half-angle of the forward cone in Frenet space
            (radians).  Default 45° (ds >= |dd|).

    Returns:
        Scalar feature value >= 0.
    """
    ds = s_obs - s_ego   # positive = participant ahead
    dd = d_ego - d_obs
    dist_sq = ds * ds + dd * dd

    # RBF 1: omnidirectional
    rbf_1 = math.exp(-dist_sq / (2.0 * sigma_1 ** 2))

    # RBF 2: forward cone gated in Frenet
    rbf_2 = math.exp(-dist_sq / (2.0 * sigma_2 ** 2))
    if ds <= 0:
        rbf_2 = 0.0
    else:
        alpha = 1.0 / math.tan(fov_half_angle) if fov_half_angle < math.pi / 2 else 0.0
        if ds < alpha * abs(dd):
            rbf_2 = 0.0

    return w1 * rbf_1 + w2 * rbf_2


# ---------------------------------------------------------------------------
# Kalman awareness tracker
# ---------------------------------------------------------------------------

class KalmanAwareness:
    """Diagonal Kalman filter tracking awareness in logit space.

    State per participant: psi_i in R  (logit of awareness probability).
    Dynamics: psi_{k+1} = A * psi_k + b * f_i(s_k) + w,  w ~ N(0, q)
    Observation: y_i = log-likelihood ratio from Boltzmann action model.

    Parameters:
        agent_ids: Sorted list of tracked participant IDs.
        A: State transition (persistence). 1.0 = no decay.
        b: Feature weight. Larger = faster awareness shift.
        q: Process noise variance.
        R: Observation noise variance.
        phi_th: Awareness threshold for discretisation.
        sigma_1: Narrow RBF spread.
        sigma_2: Wide RBF spread.
        fov_half_angle: Half FOV cone angle in Frenet space (radians).
        psi_0: Initial logit mean per participant.
        P_0: Initial variance per participant.
    """

    def __init__(self,
                 agent_ids: List[int],
                 A: float = 1.0,
                 b: float = 1.0,
                 q: float = 0.05,
                 R: float = 1.0,
                 phi_th: float = 0.5,
                 sigma_1: float = 15.0,
                 sigma_2: float = 25.0,
                 fov_half_angle: float = math.radians(30),
                 w1: float = 0.7,
                 w2: float = 0.3,
                 psi_0: float = 0.0,
                 P_0: float = 1.0,
                 initial_visibility: Optional[Dict[int, bool]] = None):
        self._agent_ids = sorted(agent_ids)
        self._n = len(self._agent_ids)
        self._A = A
        self._b = b
        self._q = q
        self._R = R
        self._phi_th = phi_th
        self._psi_th = math.log(phi_th / (1.0 - phi_th))  # logit of threshold
        self._sigma_1 = sigma_1
        self._sigma_2 = sigma_2
        self._fov_half_angle = fov_half_angle
        self._w1 = w1
        self._w2 = w2

        # State — initialise per-agent from ground-truth visibility if given
        if initial_visibility is not None:
            self.psi_hat = np.empty(self._n, dtype=float)
            for idx, aid in enumerate(self._agent_ids):
                visible = initial_visibility.get(aid, True)
                # visible → high awareness (psi = +7 → phi ≈ 0.999)
                # hidden  → low awareness  (psi = -7 → phi ≈ 0.001)
                self.psi_hat[idx] = 7.0 if visible else -7.0
        else:
            self.psi_hat = np.full(self._n, psi_0, dtype=float)
        self.P_diag = np.full(self._n, P_0, dtype=float)

        # Debug: last feature values from predict step
        self.last_features = np.zeros(self._n, dtype=float)

    @property
    def agent_ids(self) -> List[int]:
        return self._agent_ids

    @property
    def n(self) -> int:
        return self._n

    @property
    def phi(self) -> np.ndarray:
        """Current awareness probabilities (sigmoid of psi_hat)."""
        return 1.0 / (1.0 + np.exp(-self.psi_hat))

    def copy(self) -> 'KalmanAwareness':
        """Return a deep copy (for independent propagation in MCTS sims)."""
        c = KalmanAwareness.__new__(KalmanAwareness)
        c._agent_ids = self._agent_ids
        c._n = self._n
        c._A = self._A
        c._b = self._b
        c._q = self._q
        c._R = self._R
        c._phi_th = self._phi_th
        c._psi_th = self._psi_th
        c._sigma_1 = self._sigma_1
        c._sigma_2 = self._sigma_2
        c._fov_half_angle = self._fov_half_angle
        c._w1 = self._w1
        c._w2 = self._w2
        c.psi_hat = self.psi_hat.copy()
        c.P_diag = self.P_diag.copy()
        c.last_features = self.last_features.copy()
        return c

    def reset(self, agent_ids: Optional[List[int]] = None,
              psi_0: float = 0.0, P_0: float = 1.0):
        """Reinitialise to neutral prior."""
        if agent_ids is not None:
            self._agent_ids = sorted(agent_ids)
            self._n = len(self._agent_ids)
        self.psi_hat = np.full(self._n, psi_0, dtype=float)
        self.P_diag = np.full(self._n, P_0, dtype=float)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self,
                ego_xy: np.ndarray,
                ego_heading: float,
                participant_xys: Dict[int, np.ndarray]):
        """Kalman prediction step using world-frame features.

        Args:
            ego_xy: (2,) ego world position.
            ego_heading: Ego heading (radians).
            participant_xys: {agent_id: (2,) world position}.
        """
        for idx, aid in enumerate(self._agent_ids):
            p_xy = participant_xys.get(aid)
            if p_xy is not None:
                f_i = compute_feature_world(
                    ego_xy, ego_heading, p_xy,
                    self._sigma_1, self._sigma_2, self._fov_half_angle,
                    self._w1, self._w2)
            else:
                f_i = 0.0
            self.last_features[idx] = f_i
            self.psi_hat[idx] = self._A * self.psi_hat[idx] + self._b * f_i
            self.P_diag[idx] = self._A ** 2 * self.P_diag[idx] + self._q

    def predict_from_obstacles(self,
                               ego_xy: np.ndarray,
                               ego_heading: float,
                               obstacles: list,
                               fine_k: int):
        """Kalman prediction from obstacle dicts (MCTS simulation convenience).

        Extracts world positions from obstacle dicts and delegates to
        :meth:`predict`.

        Args:
            ego_xy: (2,) ego world position.
            ego_heading: Ego heading (radians).
            obstacles: List of obstacle dicts with 'world_positions' and
                'agent_id' keys.
            fine_k: Fine timestep index for position lookup.
        """
        participant_xys = {}
        for obs in obstacles:
            aid = obs.get('agent_id')
            if aid is not None and aid in self._agent_ids:
                wp = obs.get('world_positions')
                if wp is not None:
                    k = min(fine_k, len(wp) - 1)
                    participant_xys[aid] = np.array(wp[k], dtype=float)
        self.predict(ego_xy, ego_heading, participant_xys)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, y: np.ndarray, min_informativeness: float = 0.1):
        """Gated Kalman observation update.

        Skips the update for participant i when |y_i| < min_informativeness,
        preventing uninformative observations (y ≈ 0) from pulling the
        estimate toward zero.

        Args:
            y: (n,) log-likelihood ratio observations per participant.
            min_informativeness: Threshold below which the observation is
                considered uninformative and the update is skipped.
        """
        self.last_K = np.zeros(self._n, dtype=float)
        self.last_P_pre = self.P_diag.copy()
        self.last_gated = np.zeros(self._n, dtype=bool)

        for i in range(self._n):
            if abs(y[i]) < min_informativeness:
                # Uninformative observation — skip update
                self.last_gated[i] = True
                continue
            K = self.P_diag[i] / (self.P_diag[i] + self._R)
            self.last_K[i] = K
            self.psi_hat[i] = self.psi_hat[i] + K * (y[i] - self.psi_hat[i])
            self.P_diag[i] = (1.0 - K) * self.P_diag[i]

    # ------------------------------------------------------------------
    # Belief derivation
    # ------------------------------------------------------------------

    def compute_b_theta(self, configs: List[tuple]) -> Dict[tuple, float]:
        """Compute b(theta) for each discrete configuration.

        Uses the Gaussian CDF in logit space:
          P(d_i=1) = 1 - Phi((psi_th - psi_hat_i) / sqrt(P_ii))

        Args:
            configs: List of theta tuples, e.g. [(0,0), (0,1), (1,0), (1,1)].

        Returns:
            Dict mapping config -> probability.
        """
        p_aware = np.empty(self._n)
        for i in range(self._n):
            std = math.sqrt(max(self.P_diag[i], 1e-12))
            z = (self._psi_th - self.psi_hat[i]) / std
            p_aware[i] = 1.0 - _norm.cdf(z)

        b_theta = {}
        for cfg in configs:
            prob = 1.0
            for i in range(self._n):
                if cfg[i] == 1:
                    prob *= p_aware[i]
                else:
                    prob *= (1.0 - p_aware[i])
            b_theta[cfg] = prob
        return b_theta

    def compute_observation(self,
                            likelihoods: Dict[tuple, float],
                            configs: List[tuple]) -> np.ndarray:
        """Convert per-config likelihoods to per-participant log-LR observations.

        Args:
            likelihoods: {theta: P(u_H | theta)} from Boltzmann model.
            configs: All theta configurations.

        Returns:
            (n,) array of log(L_aware / L_unaware) per participant.
        """
        y = np.zeros(self._n)
        for i in range(self._n):
            L_aware = 0.0
            L_unaware = 0.0
            for cfg in configs:
                lk = likelihoods.get(cfg, 0.0)
                if cfg[i] == 1:
                    L_aware += lk
                else:
                    L_unaware += lk
            L_aware = max(L_aware, 1e-10)
            L_unaware = max(L_unaware, 1e-10)
            y[i] = math.log(L_aware / L_unaware)
        return y

    def marginals(self) -> Dict[int, float]:
        """Return per-agent P(hidden) marginals from the Kalman state.

        Returns dict {aid: P(hidden)} where P(hidden) = 1 - phi_i.
        """
        phi = self.phi
        return {aid: float(1.0 - phi[i])
                for i, aid in enumerate(self._agent_ids)}

    def phi_bounds(self) -> Dict[int, Tuple[float, float]]:
        """Return per-agent ±1σ confidence bounds in awareness (φ) space.

        Maps (ψ ± sqrt(P)) through the sigmoid to get (φ_lower, φ_upper).

        Returns:
            {aid: (phi_lower, phi_upper)}
        """
        def _sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))

        bounds = {}
        for i, aid in enumerate(self._agent_ids):
            std = math.sqrt(max(self.P_diag[i], 1e-12))
            bounds[aid] = (
                _sigmoid(self.psi_hat[i] - std),
                _sigmoid(self.psi_hat[i] + std),
            )
        return bounds
