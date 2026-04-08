"""Car-following experiment package.

Provides inference, CBF safety filters, human models, and belief
tracking for adaptive cruise control experiments.
"""

from .beliefs import VelocityErrorBelief, CarFollowBeliefState
from .human_model import compute_accel, evolve_vel_err, get_lead_vehicle, rbf_feature
from .recorder import EpisodeRecorder
