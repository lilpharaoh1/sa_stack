"""
Intervention Optimizer Module

Designs minimal adjustments to future trajectory waypoints such that the resulting
trajectory would be recognized as following the MCTS plan rather than the predicted
(incorrect) plan.

The optimization problem:
    minimize ||adjusted - predicted||_2^2  (minimal intervention)
    subject to: cost_diff(optimal, observed + adjusted) <= cost_diff(optimal, observed + mcts) - margin

This ensures the adjusted trajectory is "closer" to optimal than the MCTS trajectory,
so recognition would select it as the best explanation for driver behavior.

Usage:
    optimizer = InterventionOptimizer(cost=cost_instance)
    result = optimizer.optimize(
        observed_trajectory=obs_traj,
        predicted_trajectory=pred_traj,
        mcts_trajectory=mcts_traj,
        optimal_trajectory=opt_traj,
        goal=goal
    )
    adjusted_trajectory = result.trajectory
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from scipy.optimize import minimize, NonlinearConstraint, BFGS
from scipy.interpolate import splprep, splev

from igp2.core.trajectory import VelocityTrajectory, Trajectory
from igp2.core.cost import Cost
from igp2.core.goal import Goal

logger = logging.getLogger(__name__)


@dataclass
class InterventionResult:
    """Result of the intervention optimization."""
    success: bool
    trajectory: Optional[VelocityTrajectory]  # The adjusted trajectory
    predicted_trajectory: VelocityTrajectory  # Original predicted trajectory
    mcts_trajectory: VelocityTrajectory       # Target MCTS trajectory

    # Optimization metrics
    position_adjustment: float   # L2 norm of position changes
    velocity_adjustment: float   # L2 norm of velocity changes
    total_adjustment: float      # Combined adjustment magnitude

    # Cost metrics (for verification)
    original_cost_diff: float    # cost_diff(optimal, observed + predicted)
    adjusted_cost_diff: float    # cost_diff(optimal, observed + adjusted)
    mcts_cost_diff: float        # cost_diff(optimal, observed + mcts)

    message: str                 # Description of result


@dataclass
class OptimizerConfig:
    """Configuration for the intervention optimizer."""
    # Weighting between position and velocity adjustments
    position_weight: float = 1.0
    velocity_weight: float = 0.5

    # Cost margin: adjusted must be this much better than MCTS
    cost_margin: float = 0.01

    # Soft constraint weight (for soft-constrained formulation)
    lambda_cost: float = 10.0

    # Use soft constraints (weighted objective) vs hard constraints
    use_soft_constraints: bool = True

    # Maximum adjustment magnitudes (regularization)
    max_position_change: float = 5.0   # meters
    max_velocity_change: float = 3.0   # m/s

    # Optimizer settings
    max_iterations: int = 200
    tolerance: float = 1e-6

    # Resampling for cost comparison
    n_resample_points: int = 50


class InterventionOptimizer:
    """Optimizes trajectory adjustments to change recognition outcome.

    Given:
    - observed_trajectory: What the driver did so far
    - predicted_trajectory: Future trajectory following the (wrong) predicted plan
    - mcts_trajectory: Future trajectory following the optimal MCTS plan
    - optimal_trajectory: The optimal benchmark from initial position

    Finds the minimal adjustment to predicted_trajectory such that recognition
    would select the MCTS plan instead of the predicted plan.
    """

    def __init__(self,
                 cost: Cost = None,
                 config: OptimizerConfig = None):
        """Initialize the intervention optimizer.

        Args:
            cost: Cost instance for trajectory comparison. If None, uses default.
            config: Optimizer configuration. If None, uses defaults.
        """
        self._cost = cost if cost is not None else Cost()
        self._config = config if config is not None else OptimizerConfig()

    def optimize(self,
                 observed_trajectory: Trajectory,
                 predicted_trajectory: VelocityTrajectory,
                 mcts_trajectory: VelocityTrajectory,
                 optimal_trajectory: VelocityTrajectory,
                 goal: Goal) -> InterventionResult:
        """Optimize the trajectory adjustment.

        Args:
            observed_trajectory: What the driver did so far (past)
            predicted_trajectory: Future trajectory following predicted plan
            mcts_trajectory: Future trajectory following MCTS plan
            optimal_trajectory: Optimal benchmark from initial position
            goal: The goal to reach

        Returns:
            InterventionResult with the adjusted trajectory and metrics
        """
        # Validate inputs
        if len(predicted_trajectory.path) < 2:
            return self._failure_result(
                predicted_trajectory, mcts_trajectory,
                "Predicted trajectory too short for optimization"
            )

        if len(mcts_trajectory.path) < 2:
            return self._failure_result(
                predicted_trajectory, mcts_trajectory,
                "MCTS trajectory too short for optimization"
            )

        # Compute baseline costs
        try:
            original_cost = self._compute_combined_cost(
                observed_trajectory, predicted_trajectory, optimal_trajectory, goal
            )
            mcts_cost = self._compute_combined_cost(
                observed_trajectory, mcts_trajectory, optimal_trajectory, goal
            )
        except Exception as e:
            logger.warning(f"Failed to compute baseline costs: {e}")
            return self._failure_result(
                predicted_trajectory, mcts_trajectory,
                f"Cost computation failed: {e}"
            )

        # If predicted is already better than MCTS, no intervention needed
        if original_cost <= mcts_cost:
            logger.debug("Predicted trajectory already optimal, no intervention needed")
            return InterventionResult(
                success=True,
                trajectory=predicted_trajectory,
                predicted_trajectory=predicted_trajectory,
                mcts_trajectory=mcts_trajectory,
                position_adjustment=0.0,
                velocity_adjustment=0.0,
                total_adjustment=0.0,
                original_cost_diff=original_cost,
                adjusted_cost_diff=original_cost,
                mcts_cost_diff=mcts_cost,
                message="No intervention needed - predicted plan is already optimal"
            )

        # Resample trajectories to common length for optimization
        n_points = min(
            len(predicted_trajectory.path),
            len(mcts_trajectory.path),
            self._config.n_resample_points
        )

        pred_resampled = self._resample_trajectory(predicted_trajectory, n_points)
        mcts_resampled = self._resample_trajectory(mcts_trajectory, n_points)

        # Initial guess: predicted trajectory waypoints
        x0 = self._trajectory_to_vector(pred_resampled)

        # Bounds for the optimization
        bounds = self._compute_bounds(pred_resampled, n_points)

        # Run optimization
        if self._config.use_soft_constraints:
            result = self._optimize_soft(
                x0, pred_resampled, mcts_resampled,
                observed_trajectory, optimal_trajectory, goal,
                bounds, n_points
            )
        else:
            result = self._optimize_constrained(
                x0, pred_resampled, mcts_resampled,
                observed_trajectory, optimal_trajectory, goal,
                bounds, n_points, mcts_cost
            )

        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            # Return interpolation between predicted and MCTS as fallback
            return self._interpolation_fallback(
                predicted_trajectory, mcts_trajectory,
                observed_trajectory, optimal_trajectory, goal,
                original_cost, mcts_cost
            )

        # Extract optimized trajectory
        adjusted_traj = self._vector_to_trajectory(result.x, n_points)

        # Compute final metrics
        try:
            adjusted_cost = self._compute_combined_cost(
                observed_trajectory, adjusted_traj, optimal_trajectory, goal
            )
        except Exception:
            adjusted_cost = float('inf')

        # Compute adjustment magnitudes
        pos_adj = np.linalg.norm(adjusted_traj.path - pred_resampled.path)
        vel_adj = np.linalg.norm(adjusted_traj.velocity - pred_resampled.velocity)
        total_adj = np.sqrt(
            self._config.position_weight * pos_adj**2 +
            self._config.velocity_weight * vel_adj**2
        )

        return InterventionResult(
            success=True,
            trajectory=adjusted_traj,
            predicted_trajectory=predicted_trajectory,
            mcts_trajectory=mcts_trajectory,
            position_adjustment=pos_adj,
            velocity_adjustment=vel_adj,
            total_adjustment=total_adj,
            original_cost_diff=original_cost,
            adjusted_cost_diff=adjusted_cost,
            mcts_cost_diff=mcts_cost,
            message=f"Optimization converged in {result.nit} iterations"
        )

    def _optimize_soft(self, x0, pred_resampled, mcts_resampled,
                       observed_trajectory, optimal_trajectory, goal,
                       bounds, n_points):
        """Optimize using soft constraints (weighted objective).

        minimize: ||x - x_pred||^2 + lambda * cost_diff(optimal, observed + x)
        """
        def objective(x):
            # Deviation from predicted trajectory
            deviation = self._compute_deviation(x, pred_resampled, n_points)

            # Cost difference with optimal
            traj = self._vector_to_trajectory(x, n_points)
            try:
                cost_diff = self._compute_combined_cost(
                    observed_trajectory, traj, optimal_trajectory, goal
                )
            except Exception:
                cost_diff = 1e6  # Large penalty for invalid trajectories

            return deviation + self._config.lambda_cost * cost_diff

        # Use L-BFGS-B for bounded optimization
        result = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self._config.max_iterations,
                'ftol': self._config.tolerance
            }
        )

        return result

    def _optimize_constrained(self, x0, pred_resampled, mcts_resampled,
                               observed_trajectory, optimal_trajectory, goal,
                               bounds, n_points, mcts_cost):
        """Optimize using hard constraints.

        minimize: ||x - x_pred||^2
        subject to: cost_diff(optimal, observed + x) <= mcts_cost - margin
        """
        def objective(x):
            return self._compute_deviation(x, pred_resampled, n_points)

        def constraint_func(x):
            traj = self._vector_to_trajectory(x, n_points)
            try:
                cost_diff = self._compute_combined_cost(
                    observed_trajectory, traj, optimal_trajectory, goal
                )
            except Exception:
                return 1e6
            # Constraint: cost_diff <= mcts_cost - margin
            # In scipy: constraint_func(x) >= 0
            # So: mcts_cost - margin - cost_diff >= 0
            return mcts_cost - self._config.cost_margin - cost_diff

        # Define nonlinear constraint
        constraint = NonlinearConstraint(
            constraint_func,
            0.0,  # lb
            np.inf,  # ub
        )

        # Use SLSQP for constrained optimization
        result = minimize(
            objective, x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraint_func},
            options={
                'maxiter': self._config.max_iterations,
                'ftol': self._config.tolerance
            }
        )

        return result

    def _compute_deviation(self, x, pred_resampled, n_points):
        """Compute weighted deviation from predicted trajectory."""
        # Extract positions and velocities from optimization vector
        positions = x[:n_points * 2].reshape(n_points, 2)
        velocities = x[n_points * 2:]

        # Compute weighted L2 deviations
        pos_dev = np.sum((positions - pred_resampled.path)**2)
        vel_dev = np.sum((velocities - pred_resampled.velocity)**2)

        return (self._config.position_weight * pos_dev +
                self._config.velocity_weight * vel_dev)

    def _compute_combined_cost(self, observed, future, optimal, goal):
        """Compute cost_difference_resampled for combined trajectory."""
        # Combine observed + future
        combined = self._combine_trajectories(observed, future)

        # Use the cost function's resampled comparison
        return self._cost.cost_difference_resampled(optimal, combined, goal)

    def _combine_trajectories(self, observed: Trajectory,
                               future: VelocityTrajectory) -> VelocityTrajectory:
        """Combine observed trajectory with future trajectory."""
        if observed is None or len(observed.path) == 0:
            return future

        # Create a copy of future and insert observed at the beginning
        combined_path = np.concatenate([observed.path, future.path[1:]], axis=0)
        combined_velocity = np.concatenate([observed.velocity, future.velocity[1:]])

        # Handle heading
        if hasattr(observed, 'heading') and observed.heading is not None:
            combined_heading = np.concatenate([observed.heading, future.heading[1:]])
        else:
            combined_heading = None

        return VelocityTrajectory(combined_path, combined_velocity, combined_heading)

    def _trajectory_to_vector(self, traj: VelocityTrajectory) -> np.ndarray:
        """Flatten trajectory to optimization vector [positions, velocities]."""
        positions_flat = traj.path.flatten()  # (n*2,)
        velocities = traj.velocity  # (n,)
        return np.concatenate([positions_flat, velocities])

    def _vector_to_trajectory(self, x: np.ndarray, n_points: int) -> VelocityTrajectory:
        """Convert optimization vector back to VelocityTrajectory."""
        positions = x[:n_points * 2].reshape(n_points, 2)
        velocities = x[n_points * 2:]

        # Ensure velocities are positive
        velocities = np.maximum(velocities, 0.1)

        return VelocityTrajectory(positions, velocities)

    def _compute_bounds(self, pred_resampled: VelocityTrajectory,
                        n_points: int) -> List[Tuple[float, float]]:
        """Compute bounds for optimization variables."""
        bounds = []

        # Position bounds: predicted ± max_position_change
        for i in range(n_points):
            for j in range(2):  # x, y
                center = pred_resampled.path[i, j]
                bounds.append((
                    center - self._config.max_position_change,
                    center + self._config.max_position_change
                ))

        # Velocity bounds: predicted ± max_velocity_change, minimum 0.1
        for i in range(n_points):
            center = pred_resampled.velocity[i]
            bounds.append((
                max(0.1, center - self._config.max_velocity_change),
                center + self._config.max_velocity_change
            ))

        return bounds

    def _resample_trajectory(self, traj: VelocityTrajectory,
                             n_points: int) -> VelocityTrajectory:
        """Resample trajectory to n_points using spline interpolation."""
        if len(traj.path) <= 2:
            # Not enough points for spline, use linear interpolation
            t_orig = np.linspace(0, 1, len(traj.path))
            t_new = np.linspace(0, 1, n_points)

            new_path = np.zeros((n_points, 2))
            new_path[:, 0] = np.interp(t_new, t_orig, traj.path[:, 0])
            new_path[:, 1] = np.interp(t_new, t_orig, traj.path[:, 1])
            new_velocity = np.interp(t_new, t_orig, traj.velocity)

            return VelocityTrajectory(new_path, new_velocity)

        # Use spline interpolation for smoother resampling
        try:
            u = traj.pathlength / traj.pathlength[-1]  # Normalize to [0, 1]
            k = min(3, len(traj.path) - 1)  # Spline degree

            tck, _ = splprep(
                [traj.path[:, 0], traj.path[:, 1], traj.velocity],
                u=u, k=k, s=0
            )

            u_new = np.linspace(0, 1, n_points)
            new_x, new_y, new_v = splev(u_new, tck)

            new_path = np.column_stack([new_x, new_y])
            new_velocity = np.maximum(new_v, 0.1)  # Ensure positive

            return VelocityTrajectory(new_path, new_velocity)

        except Exception as e:
            logger.debug(f"Spline resampling failed, using linear: {e}")
            # Fallback to linear interpolation
            t_orig = np.linspace(0, 1, len(traj.path))
            t_new = np.linspace(0, 1, n_points)

            new_path = np.zeros((n_points, 2))
            new_path[:, 0] = np.interp(t_new, t_orig, traj.path[:, 0])
            new_path[:, 1] = np.interp(t_new, t_orig, traj.path[:, 1])
            new_velocity = np.interp(t_new, t_orig, traj.velocity)

            return VelocityTrajectory(new_path, np.maximum(new_velocity, 0.1))

    def _interpolation_fallback(self, predicted, mcts, observed, optimal, goal,
                                 original_cost, mcts_cost) -> InterventionResult:
        """Fallback: interpolate between predicted and MCTS trajectories."""
        # Find interpolation factor that makes cost similar to MCTS
        best_alpha = 0.5
        best_cost = float('inf')

        for alpha in np.linspace(0.1, 0.9, 9):
            try:
                interp_traj = self._interpolate_trajectories(predicted, mcts, alpha)
                cost = self._compute_combined_cost(observed, interp_traj, optimal, goal)
                if cost < best_cost:
                    best_cost = cost
                    best_alpha = alpha
            except Exception:
                continue

        # Create final interpolated trajectory
        adjusted = self._interpolate_trajectories(predicted, mcts, best_alpha)

        # Compute metrics
        n_points = min(len(predicted.path), len(mcts.path))
        pred_resamp = self._resample_trajectory(predicted, n_points)
        adj_resamp = self._resample_trajectory(adjusted, n_points)

        pos_adj = np.linalg.norm(adj_resamp.path - pred_resamp.path)
        vel_adj = np.linalg.norm(adj_resamp.velocity - pred_resamp.velocity)

        return InterventionResult(
            success=True,
            trajectory=adjusted,
            predicted_trajectory=predicted,
            mcts_trajectory=mcts,
            position_adjustment=pos_adj,
            velocity_adjustment=vel_adj,
            total_adjustment=np.sqrt(pos_adj**2 + vel_adj**2),
            original_cost_diff=original_cost,
            adjusted_cost_diff=best_cost,
            mcts_cost_diff=mcts_cost,
            message=f"Fallback interpolation (alpha={best_alpha:.2f})"
        )

    def _interpolate_trajectories(self, traj1: VelocityTrajectory,
                                   traj2: VelocityTrajectory,
                                   alpha: float) -> VelocityTrajectory:
        """Interpolate between two trajectories.

        Args:
            traj1: First trajectory
            traj2: Second trajectory
            alpha: Interpolation factor (0 = traj1, 1 = traj2)
        """
        n_points = min(len(traj1.path), len(traj2.path))

        t1 = self._resample_trajectory(traj1, n_points)
        t2 = self._resample_trajectory(traj2, n_points)

        interp_path = (1 - alpha) * t1.path + alpha * t2.path
        interp_velocity = (1 - alpha) * t1.velocity + alpha * t2.velocity

        return VelocityTrajectory(interp_path, np.maximum(interp_velocity, 0.1))

    def _failure_result(self, predicted, mcts, message) -> InterventionResult:
        """Create a failure result."""
        return InterventionResult(
            success=False,
            trajectory=None,
            predicted_trajectory=predicted,
            mcts_trajectory=mcts,
            position_adjustment=0.0,
            velocity_adjustment=0.0,
            total_adjustment=0.0,
            original_cost_diff=float('inf'),
            adjusted_cost_diff=float('inf'),
            mcts_cost_diff=float('inf'),
            message=message
        )


class GradientInterventionOptimizer(InterventionOptimizer):
    """Intervention optimizer using gradient-based trajectory blending.

    This optimizer uses a simpler approach:
    1. Compute the direction from predicted to MCTS trajectory
    2. Move along this direction until the cost constraint is satisfied
    3. Use binary search to find the minimal step size

    This is faster and more robust than general-purpose optimization.
    """

    def __init__(self, cost: Cost = None, config: OptimizerConfig = None):
        super().__init__(cost, config)
        self._blend_steps = 20  # Number of steps for binary search

    def optimize(self, observed_trajectory, predicted_trajectory,
                 mcts_trajectory, optimal_trajectory, goal) -> InterventionResult:
        """Optimize using gradient-based blending."""
        # Validate inputs
        if len(predicted_trajectory.path) < 2 or len(mcts_trajectory.path) < 2:
            return self._failure_result(
                predicted_trajectory, mcts_trajectory,
                "Trajectories too short"
            )

        # Compute baseline costs
        try:
            original_cost = self._compute_combined_cost(
                observed_trajectory, predicted_trajectory, optimal_trajectory, goal
            )
            mcts_cost = self._compute_combined_cost(
                observed_trajectory, mcts_trajectory, optimal_trajectory, goal
            )
        except Exception as e:
            return self._failure_result(
                predicted_trajectory, mcts_trajectory,
                f"Cost computation failed: {e}"
            )

        # No intervention needed if predicted is already better
        if original_cost <= mcts_cost:
            return InterventionResult(
                success=True,
                trajectory=predicted_trajectory,
                predicted_trajectory=predicted_trajectory,
                mcts_trajectory=mcts_trajectory,
                position_adjustment=0.0,
                velocity_adjustment=0.0,
                total_adjustment=0.0,
                original_cost_diff=original_cost,
                adjusted_cost_diff=original_cost,
                mcts_cost_diff=mcts_cost,
                message="No intervention needed"
            )

        # Binary search for minimal blend factor
        alpha_low, alpha_high = 0.0, 1.0
        target_cost = mcts_cost - self._config.cost_margin

        best_alpha = 1.0
        best_traj = mcts_trajectory
        best_cost = mcts_cost

        for _ in range(self._blend_steps):
            alpha_mid = (alpha_low + alpha_high) / 2

            try:
                blended = self._interpolate_trajectories(
                    predicted_trajectory, mcts_trajectory, alpha_mid
                )
                cost = self._compute_combined_cost(
                    observed_trajectory, blended, optimal_trajectory, goal
                )

                if cost <= target_cost:
                    # This alpha works, try smaller
                    if alpha_mid < best_alpha:
                        best_alpha = alpha_mid
                        best_traj = blended
                        best_cost = cost
                    alpha_high = alpha_mid
                else:
                    # Need more blending toward MCTS
                    alpha_low = alpha_mid

            except Exception:
                # Invalid trajectory, need more blending toward MCTS
                alpha_low = alpha_mid

        # Compute adjustment metrics
        n_points = min(len(predicted_trajectory.path), len(best_traj.path))
        pred_resamp = self._resample_trajectory(predicted_trajectory, n_points)
        adj_resamp = self._resample_trajectory(best_traj, n_points)

        pos_adj = np.linalg.norm(adj_resamp.path - pred_resamp.path)
        vel_adj = np.linalg.norm(adj_resamp.velocity - pred_resamp.velocity)

        return InterventionResult(
            success=True,
            trajectory=best_traj,
            predicted_trajectory=predicted_trajectory,
            mcts_trajectory=mcts_trajectory,
            position_adjustment=pos_adj,
            velocity_adjustment=vel_adj,
            total_adjustment=np.sqrt(pos_adj**2 + vel_adj**2),
            original_cost_diff=original_cost,
            adjusted_cost_diff=best_cost,
            mcts_cost_diff=mcts_cost,
            message=f"Gradient blend (alpha={best_alpha:.3f})"
        )


def compute_intervention(observed_trajectory: Trajectory,
                         predicted_trajectory: VelocityTrajectory,
                         mcts_trajectory: VelocityTrajectory,
                         optimal_trajectory: VelocityTrajectory,
                         goal: Goal,
                         cost: Cost = None,
                         use_gradient: bool = True) -> InterventionResult:
    """Convenience function to compute trajectory intervention.

    Args:
        observed_trajectory: What the driver did so far (past)
        predicted_trajectory: Future trajectory following predicted plan
        mcts_trajectory: Future trajectory following MCTS plan
        optimal_trajectory: Optimal benchmark from initial position
        goal: The goal to reach
        cost: Cost instance (uses default if None)
        use_gradient: Use gradient-based optimizer (faster) or scipy optimizer

    Returns:
        InterventionResult with the adjusted trajectory
    """
    if use_gradient:
        optimizer = GradientInterventionOptimizer(cost=cost)
    else:
        optimizer = InterventionOptimizer(cost=cost)

    return optimizer.optimize(
        observed_trajectory=observed_trajectory,
        predicted_trajectory=predicted_trajectory,
        mcts_trajectory=mcts_trajectory,
        optimal_trajectory=optimal_trajectory,
        goal=goal
    )
