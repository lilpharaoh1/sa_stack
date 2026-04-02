"""Result dataclasses for belief experiments."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np


@dataclass
class StepRecord:
    """Diagnostics captured at a single simulation step."""
    step: int
    wall_time: float  # seconds since experiment start

    # Ego state (world frame)
    ego_position: Optional[np.ndarray]   # [x, y]
    ego_speed: Optional[float]
    ego_heading: Optional[float]

    # Ego state (Frenet frame) from the true policy
    # [s, d, phi, v]: arc-length, lateral offset, heading rel. road, speed
    ego_frenet_state: Optional[np.ndarray]

    # Human (belief) policy outputs
    human_rollout: Optional[np.ndarray]        # (H+1, 4) [x, y, heading, speed]
    human_milp_rollout: Optional[np.ndarray]   # (H+1, 2) [x, y]
    human_nlp_converged: Optional[bool]
    human_obstacles: Optional[List]
    human_other_agents: Optional[Dict]
    human_trajectories: Dict[int, np.ndarray]

    # True (ground-truth) policy outputs
    true_rollout: Optional[np.ndarray]
    true_milp_rollout: Optional[np.ndarray]
    true_nlp_converged: Optional[bool]
    true_obstacles: Optional[List]
    true_other_agents: Optional[Dict]
    true_trajectories: Dict[int, np.ndarray]

    # Scene snapshot
    dynamic_agents: Dict[int, Any]   # non-ego, ID >= 0
    static_obstacles: Dict[int, Any] # ID < 0

    # Completion (based on true policy)
    goal_reached: bool

    # True policy constraint diagnostics (from TwoStageOPT._analyse_constraints)
    #   - nlp_ok: whether the NLP solver converged
    #   - velocity_violated: speed outside [v_min, v_max]
    #   - acceleration_violated: acceleration outside [a_min, a_max]
    #   - steering_violated: steering angle exceeds delta_max
    #   - jerk_violated: jerk exceeds jerk_max
    #   - steer_rate_violated: steering rate exceeds delta_rate_max
    #   - road_boundary_violations: number of corner-timestep road violations
    #   - collision_violations: number of corner-timestep collision violations
    # Human (belief) policy constraint diagnostics
    human_diag_milp_ok: Optional[bool] = None
    human_diag_nlp_ok: Optional[bool] = None

    # True policy constraint diagnostics
    true_diag_milp_ok: Optional[bool] = None
    true_diag_nlp_ok: Optional[bool] = None
    true_diag_velocity_violated: Optional[bool] = None
    true_diag_acceleration_violated: Optional[bool] = None
    true_diag_steering_violated: Optional[bool] = None
    true_diag_jerk_violated: Optional[bool] = None
    true_diag_steer_rate_violated: Optional[bool] = None
    true_diag_road_violations: int = 0
    true_diag_collision_violations: int = 0

    # Per-step ego timing breakdown (seconds).
    # Keys match BeliefAgent.last_step_timing, e.g.:
    #   human_predict, true_predict, human_policy, true_policy, plotting
    ego_timing: Optional[Dict[str, float]] = None

    # Prediction error: mean L2 distance (metres) between the 1-step-ahead
    # predicted positions from the *previous* step and the actual positions
    # observed at *this* step, averaged over all predicted agents.
    # NOTE: Currently uses the TRUE (ground-truth) predicted trajectories.
    # When belief inference is implemented, a separate belief_prediction_error
    # field should be added for the human-policy predictions.
    prediction_error: Optional[float] = None

    # --- Belief inference & intervention metrics ---

    # Belief accuracy: fraction of relevant agents whose visibility was
    # correctly classified by the inference module.
    # None when inference has not run yet (warmup period).
    belief_accuracy: Optional[float] = None

    # Raw marginal posteriors P(hidden | tau_obs) per agent from belief
    # inference.  Empty dict when inference hasn't run.
    belief_marginals: Optional[Dict[int, float]] = None

    # Ground-truth visibility per agent from the ego's configured beliefs
    # (True = visible, False = hidden).
    belief_ground_truth: Optional[Dict[int, bool]] = None

    # Human's Kalman awareness per agent: {aid: phi} where phi in [0,1].
    # phi > phi_th means the human sees the agent.  None when Kalman is
    # not running.
    human_awareness: Optional[Dict[int, float]] = None

    # Vehicle's Kalman estimate of human awareness: {aid: phi}.
    vehicle_awareness: Optional[Dict[int, float]] = None

    # Vehicle's Kalman uncertainty bounds: {aid: (phi_lo, phi_hi)}.
    vehicle_awareness_bounds: Optional[Dict[int, tuple]] = None

    # Per-configuration inference detail.  Each entry is a dict with:
    #   'config': {aid: visible_bool}
    #   'pos_cost': float
    #   'vel_cost': float
    #   'energy': float   (pos_cost + w_vel * vel_cost)
    #   'prob': float     (Boltzmann posterior P(b | tau_obs))
    #   'milp_ok': bool
    #   'nlp_ok': bool
    # Empty list when inference hasn't run yet.
    belief_config_results: Optional[List[Dict]] = None

    # Believed visibility vector after thresholding marginals.
    # {aid: visible_bool}.  None when inference hasn't run.
    believed_config: Optional[Dict[int, bool]] = None

    # L2 norm of the difference between the human-policy action and the
    # actually executed action: sqrt((da)^2 + (ddelta)^2).
    # 0.0 when no intervention is active.
    action_deviation: Optional[float] = None

    # Whether intervention was active at this timestep (i.e., the
    # intervention NLP succeeded and overrode the human-policy action).
    intervention_active: bool = False

    # --- Intervention detail ---

    # Raw control intervention: (H, 2) array of [da, ddelta] corrections
    # applied to the believed trajectory.  None when no intervention.
    intervention_controls: Optional[np.ndarray] = None

    # Intervention-corrected optimal states: (H+1, 4) [s, d, phi, v].
    # None when no intervention.
    intervention_opt_states: Optional[np.ndarray] = None

    # Intervention-corrected optimal controls: (H, 2) [a, delta].
    # None when no intervention.
    intervention_opt_controls: Optional[np.ndarray] = None

    # Reference (believed) states fed to the intervention NLP: (H+1, 4).
    # None when no intervention.
    intervention_ref_states: Optional[np.ndarray] = None

    # Reference (believed) controls fed to the intervention NLP: (H, 2).
    # None when no intervention.
    intervention_ref_controls: Optional[np.ndarray] = None

    # Whether the intervention NLP converged.
    intervention_success: Optional[bool] = None

    # Reference waypoints (N, 2) from the human policy's FrenetFrame.
    # Used to convert Frenet intervention states to world for rendering.
    reference_waypoints: Optional[np.ndarray] = None

    # Actual ego collision: True if the ego's bounding box overlaps any
    # other agent or static obstacle at this timestep.
    ego_collision: bool = False
    # ID of the agent/object the ego collided with (first detected).
    ego_collision_id: Optional[int] = None

    # --- Per-step ego violations (actual, from executed controls) ---
    # These are computed from the ego's actual state and actions, independent
    # of whether the NLP solver converged.

    # Control violations: executed control exceeds hard bounds
    ego_accel_violated: bool = False        # |a| > a_max
    ego_steering_violated: bool = False     # |delta| > delta_max

    # Comfort violations: rate-of-change exceeds smooth-driving bounds
    ego_jerk_violated: bool = False         # |a_k - a_{k-1}| / dt > jerk_max
    ego_steer_rate_violated: bool = False   # |delta_k - delta_{k-1}| / dt > delta_rate_max

    # Actual comfort magnitudes (m/s^3 and rad/s)
    ego_jerk: Optional[float] = None            # |a_k - a_{k-1}| / dt
    ego_steer_rate: Optional[float] = None      # |delta_k - delta_{k-1}| / dt

    # Actual executed acceleration and steering angle
    ego_acceleration: Optional[float] = None
    ego_steer_angle: Optional[float] = None

    # Decomposed action deviation (human action vs executed action)
    action_deviation_accel: Optional[float] = None   # |a_exec - a_human|
    action_deviation_steer: Optional[float] = None   # |delta_exec - delta_human|

    # --- Velocity particle state ---
    # Per-agent velocity particle diagnostics: {aid: {kappa_values, weights, mean, std, ess}}
    velocity_particles: Optional[Dict[int, Dict]] = None
    # Ground-truth kappa per agent: {aid: true κ value}
    velocity_kappa_gt: Optional[Dict[int, float]] = None

    # --- Per-step ego cost ---
    # Instantaneous cost incurred by the ego vehicle at this step, computed
    # from the NLP cost function weights and the ego's actual Frenet state.
    # Cost = w_s*(s-s_ref)^2 + w_d*d^2 + w_v*(v-v_tgt)^2
    #      + w_a*a^2 + w_delta*delta^2 + w_phi*phi^2
    ego_step_cost: Optional[float] = None
    ego_cost_lateral: Optional[float] = None      # w_d * d^2
    ego_cost_speed: Optional[float] = None        # w_v * (v - v_tgt)^2
    ego_cost_accel: Optional[float] = None        # w_a * a^2
    ego_cost_steering: Optional[float] = None     # w_delta * delta^2
    ego_cost_heading: Optional[float] = None      # w_phi * phi^2


@dataclass
class ExperimentResult:
    """Full result of a single experiment run."""
    # Metadata
    scenario_name: str
    config: Dict[str, Any]
    seed: int
    fps: int
    max_steps: int
    start_time: str       # ISO timestamp

    # Outcome
    solved: bool = False
    solved_step: Optional[int] = None
    total_steps: int = 0
    wall_time_seconds: float = 0.0

    # Failure outcome (human policy optimisation failure)
    failed: bool = False
    failure_step: Optional[int] = None
    failure_reason: Optional[str] = None

    # Per-step data
    steps: List[StepRecord] = field(default_factory=list)
