"""
Macro-guided maneuver recognition using inverse planning.

Uses MacroGuidedSequenceGenerator to generate candidate maneuver sequences,
then applies inverse planning (like standard IGP2 GoalRecognition) to determine
which plan the agent is most likely executing.

This approach:
1. Generates OPTIMAL trajectory from initial position (what agent SHOULD do)
2. Generates candidate trajectories from current position (possible futures)
3. Prepends observed trajectory to each candidate (what agent DID + future)
4. Compares using cost functions to compute likelihood
5. Best-matching candidate indicates which maneuvers the agent is following
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

from igp2.core.trajectory import Trajectory, VelocityTrajectory
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal, StoppingGoal
from igp2.core.velocitysmoother import VelocitySmoother
from igp2.core.cost import Cost
from igp2.core.util import Circle
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver, GiveWay
from igp2.epistemic.macro_guided_generator import MacroGuidedSequenceGenerator
from igp2.epistemic.maneuver_probabilities import ManeuverProbabilities

logger = logging.getLogger(__name__)


class MacroGuidedRecognition:
    """Goal recognition using macro-guided maneuver sequence generation with inverse planning.

    This class uses the same inverse planning approach as standard IGP2 GoalRecognition:
    1. Generate optimal trajectory from initial position (benchmark)
    2. Generate candidate trajectories from current position
    3. Prepend observed trajectory to candidates
    4. Compare candidates against optimal using cost functions
    5. Compute likelihood based on how close observed behavior is to optimal

    The key insight is that macro-actions provide robust structure for
    determining what maneuvers are applicable, while the inverse planning
    comparison determines which maneuver variation the agent is following.
    """

    def __init__(self,
                 scenario_map: Map,
                 smoother: VelocitySmoother = None,
                 cost: Cost = None,
                 n_trajectories: int = 5,
                 beta: float = 1.0,
                 gamma: float = 1.0,
                 reward_as_difference: bool = True,
                 generate_variations: bool = True):
        """Initialize macro-guided recognition.

        Args:
            scenario_map: The road network map
            smoother: Velocity smoother for trajectory smoothing
            cost: Cost function for trajectory evaluation
            n_trajectories: Maximum number of candidate trajectories per goal
            beta: Temperature for goal likelihood computation
            gamma: Temperature for trajectory probability computation
            reward_as_difference: If True, use cost_difference_resampled for comparison
            generate_variations: Whether to generate with/without GiveWay variations
        """
        self._scenario_map = scenario_map
        self._smoother = smoother or VelocitySmoother(
            vmin_m_s=1, vmax_m_s=10, n=10, amax_m_s2=5, lambda_acc=10
        )
        self._cost = cost or Cost()
        self._n_trajectories = n_trajectories
        self._beta = beta
        self._gamma = gamma
        self._reward_as_difference = reward_as_difference

        # Create the macro-guided generator
        self._generator = MacroGuidedSequenceGenerator(
            scenario_map=scenario_map,
            max_depth=4,
            max_candidates=n_trajectories * 2,  # Extra for variations
            max_iterations=100,
            generate_variations=generate_variations
        )

    def update_goals_probabilities(self,
                                   goals_probabilities: ManeuverProbabilities,
                                   observed_trajectory: Trajectory,
                                   agent_id: int,
                                   frame_ini: Dict[int, AgentState],
                                   frame: Dict[int, AgentState],
                                   visible_region: Circle = None,
                                   open_loop: bool = True,
                                   debug: bool = False,
                                   optimum_trajectories: Dict = None) -> ManeuverProbabilities:
        """Update goal probabilities using inverse planning.

        For each goal:
        1. Generate optimal trajectory from frame_ini (what agent SHOULD have done)
           - OR use a pre-computed optimal (e.g. from MCTS) if provided
        2. Generate candidate trajectories from frame (possible futures)
        3. Prepend observed trajectory to each candidate
        4. Compute likelihood by comparing against optimal
        5. Update goal probabilities

        Args:
            goals_probabilities: ManeuverProbabilities to update
            observed_trajectory: The observed trajectory so far
            agent_id: Agent ID
            frame_ini: Initial frame (start of observation)
            frame: Current frame
            visible_region: Visible region constraint
            open_loop: Whether to use open-loop planning
            debug: Enable debug output
            optimum_trajectories: Optional dict mapping (goal, None) -> (trajectory, plan).
                When provided and an entry exists for the current goal, that trajectory/plan
                is used as the optimal benchmark instead of generating from frame_ini.
                Goals without an entry fall back to the default generator-based optimal.

        Returns:
            Updated ManeuverProbabilities
        """
        norm_factor = 0.0

        logger.info(f"Agent {agent_id} macro-guided recognition (inverse planning):")

        for goal_and_type, prob in goals_probabilities.goals_probabilities.items():
            goal = goal_and_type[0]

            try:
                logger.debug(f"  Processing goal: {goal}")

                # Skip if already at goal
                if goal.reached(frame_ini[agent_id].position) and not isinstance(goal, StoppingGoal):
                    raise RuntimeError(f"Agent {agent_id} already at goal")

                # === STEP 1: Generate OPTIMAL trajectory (benchmark) ===
                # Check if a pre-computed optimal (e.g. from MCTS) is available for this goal
                mcts_opt = None
                if optimum_trajectories is not None:
                    mcts_opt = optimum_trajectories.get(goal_and_type)

                if goals_probabilities.optimum_trajectory[goal_and_type] is None:
                    if mcts_opt is not None:
                        # Use pre-computed MCTS trajectory as the optimal benchmark
                        opt_traj, opt_plan = mcts_opt
                        goals_probabilities.optimum_trajectory[goal_and_type] = opt_traj
                        goals_probabilities.optimum_plan[goal_and_type] = opt_plan
                        opt_maneuvers = [type(m).__name__ for m in opt_plan]
                        logger.info(f"    [OPTIMAL from MCTS] Using MCTS trajectory as benchmark: {opt_maneuvers}")
                    else:
                        # Fall back to generating from initial position
                        logger.info(f"    [OPTIMAL from INITIAL pos] Generating benchmark trajectory...")
                        opt_trajectories, opt_plans = self._generator.generate(
                            agent_id, frame_ini, goal, visible_region
                        )

                        if len(opt_trajectories) == 0:
                            raise RuntimeError(f"No optimal paths found to {goal}")

                        # Log the optimal plan
                        opt_maneuvers = [type(m).__name__ for m in opt_plans[0]]
                        logger.info(f"    [OPTIMAL from INITIAL pos] Result: {opt_maneuvers}")

                        # Smooth the optimal trajectory
                        opt_traj = opt_trajectories[0]
                        opt_traj.velocity[0] = frame_ini[agent_id].speed
                        self._smoother.load_trajectory(opt_traj)
                        opt_traj.velocity = self._smoother.split_smooth()

                        goals_probabilities.optimum_trajectory[goal_and_type] = opt_traj
                        goals_probabilities.optimum_plan[goal_and_type] = opt_plans[0]
                else:
                    opt_maneuvers = [type(m).__name__ for m in goals_probabilities.optimum_plan[goal_and_type]]
                    logger.debug(f"    [OPTIMAL] Using cached: {opt_maneuvers}")

                opt_trajectory = goals_probabilities.optimum_trajectory[goal_and_type]

                # === STEP 2: Generate candidate trajectories from CURRENT position ===
                # These represent possible futures (with/without GiveWay, etc.)
                logger.info(f"    [CANDIDATES from CURRENT pos] Generating possible futures...")
                all_trajectories, all_plans = self._generator.generate(
                    agent_id, frame, goal, visible_region
                )

                if len(all_trajectories) == 0:
                    raise RuntimeError(f"No paths found from current position to {goal}")

                # Smooth trajectories and set initial velocity from observed trajectory
                for traj in all_trajectories:
                    if observed_trajectory is not None and len(observed_trajectory.velocity) > 0:
                        traj.velocity[0] = observed_trajectory.velocity[-1]
                    else:
                        traj.velocity[0] = frame[agent_id].speed
                    self._smoother.load_trajectory(traj)
                    traj.velocity = self._smoother.split_smooth()

                # Log candidates from current position
                logger.info(f"    [CANDIDATES from CURRENT pos] Found {len(all_plans)} candidate(s):")
                for i, plan in enumerate(all_plans[:5]):
                    maneuver_names = [type(m).__name__ for m in plan]
                    has_gw = any(isinstance(m, GiveWay) for m in plan)
                    logger.info(f"      Candidate {i}: {maneuver_names} (GiveWay: {has_gw})")

                # === STEP 3: Prepend observed trajectory to each candidate ===
                # This creates: what agent DID + what they would optimally do from here
                combined_trajectories = []
                for traj in all_trajectories:
                    # Create a copy and insert observed trajectory at the beginning
                    combined = VelocityTrajectory(
                        traj.path.copy(),
                        traj.velocity.copy()
                    )
                    if observed_trajectory is not None and len(observed_trajectory.path) > 0:
                        combined.insert(observed_trajectory)
                    combined_trajectories.append(combined)

                # === STEP 4: Compute rewards and likelihoods ===
                # Calculate optimum reward (benchmark)
                opt_reward = self._reward(opt_trajectory, goal)
                goals_probabilities.optimum_reward[goal_and_type] = opt_reward
                logger.debug(f"    [COMPARISON] Optimal (from initial) reward: {opt_reward:.3f}")

                # Calculate reward for each candidate (observed + future)
                all_rewards = []
                all_reward_diffs = []
                for i, combined_traj in enumerate(combined_trajectories):
                    reward = self._reward(combined_traj, goal)
                    all_rewards.append(reward)

                    reward_diff = self._reward_difference(opt_trajectory, combined_traj, goal)
                    all_reward_diffs.append(reward_diff)
                    logger.debug(f"    [COMPARISON] Candidate {i} (observed+future) reward: {reward:.3f}, diff from optimal: {reward_diff:.3f}")

                # === STEP 5: Compute likelihood and select best candidate ===
                # Calculate trajectory probabilities based on reward DIFFERENCES (closeness to optimal)
                # This is the key for inverse planning: the plan closest to optimal is most likely
                traj_probs = self._trajectory_probabilities(all_reward_diffs)

                # Find best trajectory based on probability (closest to optimal)
                best_idx = np.argmax(traj_probs)
                best_plan = all_plans[best_idx]
                has_give_way = any(isinstance(m, GiveWay) for m in best_plan)

                # Use the best candidate for goal likelihood (not necessarily the first)
                likelihood = self._likelihood(opt_trajectory, combined_trajectories[best_idx], goal)

                logger.info(f"    [RESULT] Best match: {[type(m).__name__ for m in best_plan]} "
                           f"(prob: {traj_probs[best_idx]:.3f}, GiveWay: {has_give_way})")

                # Store results
                goals_probabilities.all_trajectories[goal_and_type] = all_trajectories
                goals_probabilities.all_plans[goal_and_type] = all_plans
                goals_probabilities.current_trajectory[goal_and_type] = all_trajectories[best_idx]

                # Store rewards
                goals_probabilities.all_rewards[goal_and_type] = all_rewards
                goals_probabilities.current_reward[goal_and_type] = all_rewards[best_idx]

                # Store trajectory probabilities
                goals_probabilities.trajectories_probabilities[goal_and_type] = traj_probs

            except Exception as e:
                logger.debug(f"    Failed: {e}")
                likelihood = 0.0
                goals_probabilities.current_trajectory[goal_and_type] = None

            # Update goal probability
            goals_probabilities.goals_probabilities[goal_and_type] = prob * likelihood
            norm_factor += goals_probabilities.goals_probabilities[goal_and_type]

        # Normalize probabilities
        if norm_factor > 0:
            for key in goals_probabilities.goals_probabilities:
                goals_probabilities.goals_probabilities[key] /= norm_factor

        return goals_probabilities

    def _reward(self, trajectory: Trajectory, goal: Goal) -> float:
        """Calculate the reward (negative cost) for a trajectory."""
        try:
            return -self._cost.trajectory_cost(trajectory, goal)
        except (IndexError, ValueError) as e:
            logger.debug(f"Failed to compute reward: {e}")
            return -np.inf

    def _reward_difference(self, opt_trajectory: Trajectory,
                          current_trajectory: Trajectory, goal: Goal) -> float:
        """Calculate reward difference between optimal and current trajectory.

        If reward_as_difference is True, uses cost_difference_resampled which
        compares trajectory attributes at sampled points. Otherwise, just
        computes the difference of individual rewards.
        """
        try:
            if self._reward_as_difference:
                return -self._cost.cost_difference_resampled(opt_trajectory, current_trajectory, goal)
            else:
                return self._reward(current_trajectory, goal) - self._reward(opt_trajectory, goal)
        except (IndexError, ValueError) as e:
            logger.debug(f"Failed to compute reward difference: {e}")
            return -np.inf

    def _likelihood(self, opt_trajectory: Trajectory,
                   current_trajectory: Trajectory, goal: Goal) -> float:
        """Calculate likelihood using inverse planning.

        likelihood = exp(beta * reward_difference)

        Higher reward_difference (closer to optimal) = higher likelihood.
        """
        difference = self._reward_difference(opt_trajectory, current_trajectory, goal)
        return float(np.clip(np.exp(self._beta * difference), 1e-305, 1e305))

    def _trajectory_probabilities(self, rewards: List[float]) -> List[float]:
        """Calculate probabilities for each trajectory using Boltzmann distribution.

        P(trajectory_i) = exp(gamma * reward_i) / sum(exp(gamma * reward_j))
        """
        if not rewards:
            return []

        rewards = np.array(rewards)

        # Handle -inf values
        valid_mask = np.isfinite(rewards)
        if not np.any(valid_mask):
            return [1.0 / len(rewards)] * len(rewards)

        # Softmax with numerical stability
        max_reward = np.max(rewards[valid_mask])
        exp_rewards = np.zeros_like(rewards)
        exp_rewards[valid_mask] = np.exp(self._gamma * (rewards[valid_mask] - max_reward))

        total = np.sum(exp_rewards)
        if total > 0:
            return (exp_rewards / total).tolist()
        else:
            return [1.0 / len(rewards)] * len(rewards)

    def get_give_way_analysis(self, goals_probabilities: ManeuverProbabilities) -> Dict:
        """Analyze whether the agent appears to be executing GiveWay.

        Examines the best-matching plans for each goal and determines
        whether they include GiveWay maneuvers.

        Args:
            goals_probabilities: The updated probabilities

        Returns:
            Dict with:
            - 'best_plan_has_give_way': bool
            - 'give_way_confidence': float
            - 'analysis': str description
        """
        # Get most likely goal
        best_goal = None
        best_prob = 0.0
        for (goal, _), prob in goals_probabilities.goals_probabilities.items():
            if prob > best_prob:
                best_prob = prob
                best_goal = goal

        if best_goal is None:
            return {
                'best_plan_has_give_way': False,
                'give_way_confidence': 0.0,
                'analysis': "No goals available"
            }

        # Get plans and probabilities for this goal
        all_plans = goals_probabilities.all_plans.get((best_goal, None), [])
        traj_probs = goals_probabilities.trajectories_probabilities.get((best_goal, None), [])

        if not all_plans:
            return {
                'best_plan_has_give_way': False,
                'give_way_confidence': 0.0,
                'analysis': "No plan available"
            }

        # Find the most likely plan
        if traj_probs:
            best_idx = np.argmax(traj_probs)
            best_plan = all_plans[best_idx]
        else:
            best_plan = all_plans[0]

        # Check if best plan has GiveWay
        has_give_way = any(isinstance(m, GiveWay) for m in best_plan)

        # Calculate confidence by comparing probabilities of plans with/without GiveWay
        with_gw_prob = 0.0
        without_gw_prob = 0.0

        for plan, prob in zip(all_plans, traj_probs if traj_probs else [1.0/len(all_plans)]*len(all_plans)):
            plan_has_gw = any(isinstance(m, GiveWay) for m in plan)
            if plan_has_gw:
                with_gw_prob += prob
            else:
                without_gw_prob += prob

        # Compute confidence based on probability difference
        total_prob = with_gw_prob + without_gw_prob
        if total_prob > 0:
            if has_give_way:
                confidence = with_gw_prob / total_prob
            else:
                confidence = without_gw_prob / total_prob
        else:
            confidence = 0.5

        if has_give_way:
            analysis = f"Driver appears to be executing GiveWay (confidence: {confidence:.2f})"
        else:
            analysis = f"Driver appears to be skipping GiveWay (confidence: {confidence:.2f})"

        return {
            'best_plan_has_give_way': has_give_way,
            'give_way_confidence': confidence,
            'analysis': analysis,
            'best_plan_maneuvers': [type(m).__name__ for m in best_plan]
        }
