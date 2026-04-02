"""Particle-based velocity scaling factor estimation.

Maintains K weighted particles representing the human's velocity scaling
factor κ ∈ [κ_min, 1] for each traffic participant.  κ = 1 means the
human has a correct velocity estimate; κ < 1 means they underestimate
the speed.

The particles are:
- **Propagated** each planning cycle (drift toward 1.0, modulated by
  perceptual features and awareness).
- **Reweighted** after observing the human's action (Boltzmann likelihood
  from MCTS Q values).
- **Resampled** when the effective sample size drops below K/2.

See ``velocity_particles_spec.md`` for the full specification.
"""

import math
import random
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VelocityParticles:
    """Weighted particle set for velocity scaling factor κ.

    Args:
        K: Number of particles.
        kappa_min: Lower bound of the κ range.
        resample_noise: Std of Gaussian noise added after resampling.
        neff_threshold_frac: Resample when ESS < K * this fraction.
    """

    def __init__(self,
                 K: int = 4,
                 kappa_min: float = 0.3,
                 resample_noise: float = 0.05,
                 neff_threshold_frac: float = 0.5):
        self.K = K
        self.kappa_min = kappa_min
        self._resample_noise = resample_noise
        self._neff_threshold = K * neff_threshold_frac

        # Initialise particles uniformly across [kappa_min, 1.0]
        if K <= 1:
            self.kappa_values = [1.0]
        else:
            self.kappa_values = [
                kappa_min + j * (1.0 - kappa_min) / (K - 1)
                for j in range(K)
            ]
        self.weights = [1.0 / K] * K

    def weighted_mean(self) -> float:
        """Weighted mean of κ particles."""
        return sum(w * k for w, k in zip(self.weights, self.kappa_values))

    def weighted_std(self) -> float:
        """Weighted standard deviation of κ particles."""
        mu = self.weighted_mean()
        var = sum(w * (k - mu) ** 2 for w, k in zip(self.weights, self.kappa_values))
        return math.sqrt(max(var, 0.0))

    def effective_sample_size(self) -> float:
        """Effective sample size: 1 / Σ w_j²."""
        return 1.0 / sum(w ** 2 for w in self.weights)

    def map_estimate(self) -> float:
        """Maximum a posteriori: particle with highest weight."""
        idx = max(range(self.K), key=lambda j: self.weights[j])
        return self.kappa_values[idx]

    def copy(self) -> 'VelocityParticles':
        """Return a deep copy."""
        c = VelocityParticles.__new__(VelocityParticles)
        c.K = self.K
        c.kappa_min = self.kappa_min
        c._resample_noise = self._resample_noise
        c._neff_threshold = self._neff_threshold
        c.kappa_values = list(self.kappa_values)
        c.weights = list(self.weights)
        return c

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(self,
                  f_kappa: float,
                  b_kappa: float = 0.1,
                  q_kappa: float = 0.001):
        """Propagate each particle's κ toward 1.0.

        κ_new = κ + b_κ · f_κ · (1 - κ) + noise

        Args:
            f_kappa: Velocity feature value (from RBF + FOV + awareness).
            b_kappa: Drift rate toward κ = 1.
            q_kappa: Process noise variance.
        """
        noise_std = math.sqrt(max(q_kappa, 0.0))
        for j in range(self.K):
            kappa = self.kappa_values[j]
            kappa_new = kappa + b_kappa * f_kappa * (1.0 - kappa)
            kappa_new += random.gauss(0.0, noise_std)
            kappa_new = max(self.kappa_min, min(1.0, kappa_new))
            self.kappa_values[j] = kappa_new

    # ------------------------------------------------------------------
    # Observation update (reweighting)
    # ------------------------------------------------------------------

    def reweight(self,
                 likelihoods: List[float]):
        """Bayesian reweight: w_new ∝ w_old · P(u_H | κ_j).

        Standard particle filter update — multiplies prior weights
        by observation likelihoods and renormalises.

        Args:
            likelihoods: Length-K list of P(u_H | κ_j) values.
        """
        for j in range(self.K):
            self.weights[j] *= likelihoods[j]

        total = sum(self.weights)
        if total > 1e-30:
            self.weights = [w / total for w in self.weights]
        else:
            self.weights = [1.0 / self.K] * self.K

        # Check ESS and resample if needed
        if self.effective_sample_size() < self._neff_threshold:
            self.resample()

    # ------------------------------------------------------------------
    # Systematic resampling
    # ------------------------------------------------------------------

    def resample(self):
        """Systematic resampling with jittering to prevent collapse."""
        K = self.K
        cumulative = []
        running = 0.0
        for w in self.weights:
            running += w
            cumulative.append(running)

        new_kappas = []
        u = random.uniform(0.0, 1.0 / K)
        idx = 0
        for j in range(K):
            threshold = u + j / K
            while cumulative[idx] < threshold and idx < K - 1:
                idx += 1
            new_kappas.append(self.kappa_values[idx])

        # Add small noise to prevent particle collapse
        self.kappa_values = [
            max(self.kappa_min, min(1.0,
                k + random.gauss(0.0, self._resample_noise)))
            for k in new_kappas
        ]
        self.weights = [1.0 / K] * K


# ---------------------------------------------------------------------------
# Velocity feature computation
# ---------------------------------------------------------------------------

def compute_velocity_feature(ego_xy: np.ndarray,
                             ego_heading: float,
                             participant_xy: np.ndarray,
                             phi_i: float,
                             sigma: float = 25.0,
                             fov_half_angle: float = math.radians(30),
                             ) -> float:
    """FOV-gated RBF feature for velocity estimation, gated by awareness.

    f_κ = RBF(σ) · FOV_gate · φ_i

    The velocity kernel uses only the forward FOV-gated RBF (no
    omnidirectional component), reflecting that velocity estimation
    requires direct visual observation.

    Args:
        ego_xy: (2,) ego world position.
        ego_heading: Ego heading (radians).
        participant_xy: (2,) participant world position.
        phi_i: Current awareness probability for this participant.
        sigma: RBF spread (m).
        fov_half_angle: Half-angle of forward field of view (radians).

    Returns:
        Scalar feature value in [0, 1].
    """
    diff = participant_xy - ego_xy
    dist_sq = float(np.dot(diff, diff))

    rbf = math.exp(-dist_sq / (2.0 * sigma ** 2))

    # FOV gating
    angle_to = math.atan2(diff[1], diff[0])
    rel_angle = (angle_to - ego_heading + math.pi) % (2.0 * math.pi) - math.pi
    if abs(rel_angle) > fov_half_angle:
        rbf = 0.0

    return rbf * phi_i
