"""Belief data structures for the car-following experiment."""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class VelocityErrorBelief:
    """Discrete posterior over velocity-error candidates for one agent.

    Attributes:
        candidates:    Array of velocity-error values, e.g. [-0.5, ..., 0.5].
        probabilities: Posterior probabilities (sums to 1).
    """
    candidates: np.ndarray = field(
        default_factory=lambda: np.arange(-0.5, 0.55, 0.1))
    probabilities: np.ndarray = None

    def __post_init__(self):
        if self.probabilities is None:
            self.probabilities = np.ones(len(self.candidates)) / len(self.candidates)

    @property
    def mode(self) -> float:
        return float(self.candidates[np.argmax(self.probabilities)])

    @property
    def mean(self) -> float:
        return float(np.sum(self.candidates * self.probabilities))


@dataclass
class CarFollowBeliefState:
    """Belief state the assistive system maintains about the human driver."""
    velocity_errors: Dict[int, VelocityErrorBelief] = field(default_factory=dict)
