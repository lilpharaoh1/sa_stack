"""
Shared Autonomy Agent for human-in-the-loop driving with MCTS predictions and safety interventions.

This agent combines:
- Keyboard control for manual driving (like KeyboardAgent)
- Goal recognition and predictions for ALL agents including ego
- MCTS planning to generate an "optimal" safe plan
- Safety analysis comparing predicted driver intent vs optimal plan
- Interventions when the predicted plan is unsafe

The system:
1. Predicts what the driver intends to do (goal recognition on ego)
2. Generates what the driver SHOULD do (MCTS optimal plan)
3. Compares the two to identify safety issues
4. Applies interventions (slow down, stop) when needed
5. Communicates changes to the driver

Controls:
    W / UP      : Accelerate
    S / DOWN    : Brake / Reverse
    A / LEFT    : Steer left
    D / RIGHT   : Steer right
"""

import logging
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Set SDL to use dummy video driver if no display is available
if os.environ.get('DISPLAY') is None and os.environ.get('SDL_VIDEODRIVER') is None:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
from pygame.locals import K_w, K_s, K_a, K_d, K_UP, K_DOWN, K_LEFT, K_RIGHT
from shapely.geometry import Point

from igp2.agents.agent import Agent
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal, PointGoal, StoppingGoal, PointCollectionGoal
from igp2.core.vehicle import Action, Observation, TrajectoryPrediction, KinematicVehicle
from igp2.core.trajectory import Trajectory, StateTrajectory, VelocityTrajectory
from igp2.core.cost import Cost
from igp2.core.velocitysmoother import VelocitySmoother
from igp2.core.util import Circle
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver
from igp2.planlibrary.macro_action import MacroAction, MacroActionConfig, MacroActionFactory
from igp2.planning.reward import Reward
from igp2.planning.mcts import MCTS
from igp2.recognition.astar import AStar
from igp2.recognition.goalrecognition import GoalRecognition
from igp2.recognition.goalprobabilities import GoalsProbabilities

# Epistemic module for maneuver-level recognition
from igp2.epistemic.maneuver_astar import ManeuverAStar
from igp2.epistemic.maneuver_recognition import ManeuverRecognition
from igp2.epistemic.maneuver_probabilities import ManeuverProbabilities

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of safety interventions the system can apply."""
    NONE = "none"
    SLOW_DOWN = "slow_down"
    STOP = "stop"
    YIELD = "yield"


@dataclass
class SequenceDifference:
    """Represents a difference between predicted and optimal plans at a specific position."""
    position: int              # 0-indexed position in the plan
    predicted_action: str      # What the predicted plan has (or None if shorter)
    optimal_action: str        # What the MCTS plan has (or None if shorter)
    is_safety_critical: bool   # True if optimal action is safety-critical (StopMA, etc.)
    description: str           # Human-readable description


@dataclass
class SafetyAnalysis:
    """Results of comparing predicted plan vs MCTS optimal plan.

    Comparison happens at two levels:
    - Macro-actions: High-level actions like Continue, Exit, ChangeLane, StopMA
    - Maneuvers: Low-level actions like FollowLane, Turn, GiveWay, Stop

    Sequence comparison:
    - sequence_differences: List of differences at each position where plans diverge
    - first_divergence: Position where plans first differ (or -1 if identical)
    """
    is_safe: bool
    intervention_type: InterventionType
    message: str
    # Macro-action level
    predicted_actions: List[str]  # MacroAction types in predicted plan
    optimal_actions: List[str]    # MacroAction types in MCTS plan
    missing_actions: List[str]    # MacroActions in optimal but not in predicted (set-based)
    # Sequence-level comparison
    sequence_differences: List[SequenceDifference]  # Differences at each position
    first_divergence: int         # Position of first difference (-1 if identical)
    # Maneuver level
    predicted_maneuvers: List[str]  # Maneuver types in predicted plan
    missing_maneuvers: List[str]    # Safety-critical maneuvers missing from predicted
    risk_score: float               # 0.0 = safe, 1.0 = very unsafe


@dataclass
class DriverMessage:
    """Message to communicate to the driver."""
    text: str
    severity: str  # "info", "warning", "critical"
    action_hint: str  # What the driver should do


class SharedAutonomyAgent(Agent):
    """Agent that combines keyboard control with MCTS predictions and safety interventions.

    This agent is controlled by a human via keyboard, but also:
    - Performs goal recognition for ALL agents including itself (ego)
    - Runs MCTS to generate an "optimal" safe plan
    - Compares predicted intent vs optimal plan for safety
    - Applies interventions when the predicted plan is unsafe
    - Communicates safety issues to the driver

    Attributes:
        goal_probabilities: Dict mapping agent_id -> GoalsProbabilities for all agents
        observations: Dict mapping agent_id -> (trajectory, initial_frame) for all agents
        possible_goals: List of possible goals in the current view
        mcts_plan: The optimal plan generated by MCTS
        safety_analysis: Results of comparing predicted vs optimal plan
        current_message: Current message to display to driver
    """

    # Control parameters
    MAX_ACCELERATION = 5.0  # m/s^2
    MAX_BRAKE = 8.0  # m/s^2
    MAX_STEER = 0.7  # radians
    STEER_SPEED = 0.05

    # Intervention parameters
    INTERVENTION_DECEL = 3.0  # m/s^2 deceleration when slowing down
    STOP_DECEL = 6.0  # m/s^2 deceleration when stopping

    # Class-level pygame tracking
    _pygame_initialized = False

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 t_update: float,
                 scenario_map: Map,
                 goal: Goal = None,
                 view_radius: float = 50.0,
                 fps: int = 20,
                 n_simulations: int = 5,
                 max_depth: int = 5,
                 store_results: str = 'final',
                 trajectory_agents: bool = True,
                 cost_factors: Dict[str, float] = None,
                 reward_factors: Dict[str, float] = None,
                 default_rewards: Dict[str, float] = None,
                 velocity_smoother: dict = None,
                 goal_recognition: dict = None,
                 stop_goals: bool = False,
                 predict_ego: bool = True,
                 ego_goal_mode: str = "true_goal",
                 enable_interventions: bool = False, # True, 
                 prediction_level: str = "maneuver"):
        """Initialize a shared autonomy agent.

        Args:
            agent_id: ID of the agent
            initial_state: Starting state of the agent
            t_update: Time interval between prediction/planning updates (seconds)
            scenario_map: The road network map
            goal: Final goal of the agent
            view_radius: Radius within which other agents are visible
            fps: Execution rate of the environment simulation
            n_simulations: Number of MCTS simulations
            max_depth: Maximum MCTS search depth
            store_results: Whether to save MCTS rollout traces
            trajectory_agents: Whether to use trajectories for non-egos in MCTS
            cost_factors: Cost factors for trajectory evaluation
            reward_factors: Reward factors for MCTS rollouts
            default_rewards: Default rewards for MCTS
            velocity_smoother: Velocity smoother parameters
            goal_recognition: Goal recognition parameters
            stop_goals: Whether to include stopping goals
            predict_ego: Whether to include ego in goal recognition (default True)
            ego_goal_mode: How to determine ego's goal for prediction:
                - "goal_recognition": Predict goal from observed trajectory (default)
                - "true_goal": Use the actual ego goal, run A* to find path
            enable_interventions: Whether to apply safety interventions
            prediction_level: Level of prediction granularity:
                - "macro": Use macro-action level (Continue, Exit, etc.) via IGP2 GoalRecognition
                - "maneuver": Use maneuver level (FollowLane, GiveWay, Turn, etc.) via epistemic module
        """
        super().__init__(agent_id, initial_state, goal, fps)

        # Vehicle for state tracking
        self._vehicle = KinematicVehicle(initial_state, self.metadata, fps)

        # Keyboard control state
        self._reverse = False
        self._current_steer = 0.0
        self._pygame_available = self._ensure_pygame_initialized()

        # Prediction system
        self._view_radius = view_radius
        self._scenario_map = scenario_map
        self._predict_ego = predict_ego
        self._stop_goals = stop_goals
        self._ego_goal_mode = ego_goal_mode  # "goal_recognition" or "true_goal"

        # Update timing
        self._k = 0
        self._kmax = t_update * fps

        # Prediction level configuration
        if prediction_level not in ("macro", "maneuver"):
            raise ValueError(f"Invalid prediction_level: {prediction_level}. Must be 'macro' or 'maneuver'.")
        self._prediction_level = prediction_level

        # Goal recognition setup
        self._cost = Cost(factors=cost_factors) if cost_factors is not None else Cost()

        if velocity_smoother is None:
            velocity_smoother = {"vmin_m_s": 1, "vmax_m_s": 10, "n": 10, "amax_m_s2": 5, "lambda_acc": 10}
        self._smoother = VelocitySmoother(**velocity_smoother)

        if goal_recognition is None:
            goal_recognition = {"reward_as_difference": False, "n_trajectories": 2}

        # Create macro-level (IGP2) recognition
        self._astar = AStar(next_lane_offset=0.1)
        self._goal_recognition = GoalRecognition(
            astar=self._astar,
            smoother=self._smoother,
            scenario_map=scenario_map,
            cost=self._cost,
            **goal_recognition
        )

        # Create maneuver-level (epistemic) recognition
        self._maneuver_astar = ManeuverAStar(next_lane_offset=0.1)
        self._maneuver_recognition = ManeuverRecognition(
            astar=self._maneuver_astar,
            smoother=self._smoother,
            scenario_map=scenario_map,
            cost=self._cost,
            **goal_recognition
        )

        # MCTS setup
        self._reward = Reward(factors=reward_factors, default_rewards=default_rewards) \
            if reward_factors is not None else Reward()
        self._mcts = MCTS(
            scenario_map=scenario_map,
            reward=self._reward,
            n_simulations=n_simulations,
            max_depth=max_depth,
            store_results=store_results,
            trajectory_agents=trajectory_agents
        )

        # Storage for predictions and plans
        # For macro-level (IGP2) predictions
        self._goal_probabilities: Dict[int, GoalsProbabilities] = {}
        # For maneuver-level (epistemic) predictions
        self._maneuver_probabilities: Dict[int, ManeuverProbabilities] = {}
        # Shared storage
        self._observations: Dict[int, Tuple[StateTrajectory, Dict[int, AgentState]]] = {}
        self._goals: List[Goal] = []

        # MCTS plan storage
        self._mcts_plan: List[MacroAction] = []  # MCTSAction wrappers from search
        self._mcts_macro_actions: List[MacroAction] = []  # Instantiated MacroActions
        self._mcts_maneuvers: List[List] = []  # Maneuvers per macro-action
        self._mcts_trajectory: Optional[VelocityTrajectory] = None

        # Safety analysis
        self._enable_interventions = enable_interventions
        self._safety_analysis: Optional[SafetyAnalysis] = None
        self._current_message: Optional[DriverMessage] = None
        self._intervention_active = False

        # For compatibility
        self._current_macro = None
        self._pgp_control = False
        self._pgp_drive = False

        if self._pygame_available:
            logger.info(f"SharedAutonomyAgent {agent_id} initialized. "
                       f"predict_ego={predict_ego}, ego_goal_mode={ego_goal_mode}, "
                       f"prediction_level={prediction_level}, "
                       f"interventions={enable_interventions}")
        else:
            logger.warning(f"SharedAutonomyAgent {agent_id}: pygame unavailable, zero controls.")

    @classmethod
    def _ensure_pygame_initialized(cls) -> bool:
        """Ensure pygame is initialized for keyboard input."""
        if cls._pygame_initialized:
            return True

        try:
            pygame.init()
            if not pygame.display.get_init():
                pygame.display.init()
            if pygame.display.get_surface() is None:
                pygame.display.set_mode((100, 100), pygame.NOFRAME)
            pygame.event.pump()
            pygame.key.get_pressed()
            cls._pygame_initialized = True
            return True
        except Exception as e:
            logger.warning(f"Could not initialize pygame: {e}")
            cls._pygame_initialized = True
            return False

    def done(self, observation: Observation) -> bool:
        """Keyboard-controlled agent is never 'done' automatically."""
        if self.goal is not None:
            return self.goal.reached(observation.frame[self.agent_id].position)
        return False

    def next_action(self, observation: Observation, prediction: TrajectoryPrediction = None) -> Action:
        """Get next action from keyboard with safety interventions.

        This method:
        1. Updates observations for all agents
        2. Periodically runs goal recognition for all agents (including ego)
        3. Runs MCTS to generate optimal plan
        4. Analyzes safety by comparing predicted vs optimal
        5. Applies interventions if needed
        6. Returns (possibly modified) keyboard action

        Args:
            observation: Current observation of the environment
            prediction: Optional external prediction (unused)

        Returns:
            Action from keyboard input, possibly modified by interventions
        """
        # Update observations for all agents
        self._update_observations(observation)

        # Periodically update predictions and MCTS plan
        self._k += 1
        if self._k >= self._kmax:
            self._goals = self._get_goals(observation)
            if self._goals:
                self._update_predictions(observation)
                self._update_mcts_plan(observation)
                self._analyze_safety()
            self._k = 0

        # Get keyboard action
        action = self._get_keyboard_action(observation)

        # Apply interventions if enabled and needed
        if self._enable_interventions and self._safety_analysis is not None:
            action = self._apply_intervention(action, observation)

        return action

    def _get_keyboard_action(self, observation: Observation) -> Action:
        """Read keyboard input and return corresponding action."""
        current_state = observation.frame.get(self.agent_id)
        current_speed = current_state.speed if current_state else 0.0

        if not self._pygame_available:
            return Action(0.0, 0.0, target_speed=current_speed)

        try:
            pygame.event.pump()
            if not pygame.key.get_focused():
                return Action(0.0, 0.0, target_speed=current_speed)
            keys = pygame.key.get_pressed()
        except pygame.error:
            return Action(0.0, 0.0, target_speed=current_speed)

        # Calculate acceleration
        acceleration = 0.0
        if keys[K_w] or keys[K_UP]:
            acceleration = -self.MAX_BRAKE if self._reverse else self.MAX_ACCELERATION
        elif keys[K_s] or keys[K_DOWN]:
            acceleration = self.MAX_ACCELERATION if self._reverse else -self.MAX_BRAKE

        # Calculate steering
        target_steer = 0.0
        if keys[K_a] or keys[K_LEFT]:
            target_steer = -self.MAX_STEER
        elif keys[K_d] or keys[K_RIGHT]:
            target_steer = self.MAX_STEER

        # Smooth steering
        steer_diff = target_steer - self._current_steer
        if abs(steer_diff) > self.STEER_SPEED:
            self._current_steer += self.STEER_SPEED if steer_diff > 0 else -self.STEER_SPEED
        else:
            self._current_steer = target_steer

        dt = 1.0 / self._fps
        target_speed = max(0.0, current_speed + acceleration * dt)

        return Action(acceleration, self._current_steer, target_speed=target_speed)

    def _apply_intervention(self, action: Action, observation: Observation) -> Action:
        """Apply safety intervention to the action if needed."""
        if self._safety_analysis is None or self._safety_analysis.is_safe:
            self._intervention_active = False
            return action

        current_state = observation.frame.get(self.agent_id)
        current_speed = current_state.speed if current_state else 0.0
        dt = 1.0 / self._fps

        intervention = self._safety_analysis.intervention_type

        if intervention == InterventionType.STOP:
            # Force stop
            self._intervention_active = True
            new_accel = -self.STOP_DECEL
            new_target_speed = max(0.0, current_speed + new_accel * dt)
            return Action(new_accel, action.steer_angle, target_speed=new_target_speed)

        elif intervention == InterventionType.SLOW_DOWN:
            # Limit acceleration to slow down
            self._intervention_active = True
            new_accel = min(action.acceleration, -self.INTERVENTION_DECEL)
            new_target_speed = max(0.0, current_speed + new_accel * dt)
            return Action(new_accel, action.steer_angle, target_speed=new_target_speed)

        elif intervention == InterventionType.YIELD:
            # Reduce speed moderately
            self._intervention_active = True
            max_allowed_accel = -self.INTERVENTION_DECEL * 0.5
            new_accel = min(action.acceleration, max_allowed_accel)
            new_target_speed = max(0.0, current_speed + new_accel * dt)
            return Action(new_accel, action.steer_angle, target_speed=new_target_speed)

        return action

    def _update_observations(self, observation: Observation):
        """Update trajectory observations for all visible agents including ego."""
        frame = observation.frame

        for aid, agent_state in frame.items():
            try:
                self._observations[aid][0].add_state(agent_state)
            except KeyError:
                self._observations[aid] = (
                    StateTrajectory(fps=self._fps, states=[agent_state]),
                    frame.copy()
                )

        # Remove agents no longer in frame
        for aid in list(self._observations.keys()):
            if aid not in frame:
                self._observations.pop(aid)

    def _update_predictions(self, observation: Observation):
        """Run goal recognition for all agents including ego (if enabled).

        The prediction level can be configured via prediction_level:
        - "macro": Use IGP2 GoalRecognition (macro-action level: Continue, Exit, etc.)
        - "maneuver": Use epistemic ManeuverRecognition (maneuver level: FollowLane, GiveWay, etc.)

        The ego agent's prediction can be done in two modes (set via ego_goal_mode):
        - "goal_recognition": Predict goal from observed trajectory
        - "true_goal": Use the actual ego goal, just run A* to find path to it

        Note: MCTS always requires macro-level predictions (GoalsProbabilities) for non-ego
        agents. When prediction_level="maneuver", we still run macro-level recognition for
        non-ego agents to feed MCTS, while using maneuver-level for ego.
        """
        frame = observation.frame

        if not self._goals:
            logger.debug("No goals found in view, skipping prediction update")
            return

        # Determine which agents to predict
        agents_to_predict = list(frame.keys())
        if not self._predict_ego:
            agents_to_predict = [aid for aid in agents_to_predict if aid != self.agent_id]

        # Separate ego and non-ego agents
        non_ego_agents = [aid for aid in agents_to_predict if aid != self.agent_id]

        # Always initialize goal_probabilities for non-ego agents (MCTS needs them)
        # When prediction_level="maneuver", we still need macro-level predictions for MCTS
        self._goal_probabilities = {
            aid: GoalsProbabilities(self._goals) for aid in non_ego_agents
        }

        # Initialize maneuver probabilities when using maneuver-level prediction
        if self._prediction_level == "maneuver":
            self._maneuver_probabilities = {
                aid: ManeuverProbabilities(self._goals) for aid in agents_to_predict
            }
        else:
            # For macro level, also add ego to goal_probabilities
            if self.agent_id in agents_to_predict:
                self._goal_probabilities[self.agent_id] = GoalsProbabilities(self._goals)

        visible_region = Circle(frame[self.agent_id].position, self._view_radius)

        # Run goal recognition for each agent
        for aid in agents_to_predict:
            if aid not in self._observations:
                continue

            # Check if this is ego and we should use true goal mode
            use_true_goal = (aid == self.agent_id and
                            self._ego_goal_mode == "true_goal" and
                            self.goal is not None)

            if use_true_goal:
                # Use actual ego goal - run A* to find path to true goal
                self._update_ego_with_true_goal(observation, frame, visible_region)
            elif aid == self.agent_id:
                # Ego prediction based on prediction_level
                if self._prediction_level == "macro":
                    self._run_macro_recognition(aid, frame, visible_region)
                else:
                    self._run_maneuver_recognition(aid, frame, visible_region)
            else:
                # Non-ego: always use macro-level recognition (MCTS needs GoalsProbabilities)
                self._run_macro_recognition(aid, frame, visible_region)
                # # Additionally run maneuver recognition if prediction_level is maneuver
                # if self._prediction_level == "maneuver":
                #     self._run_maneuver_recognition(aid, frame, visible_region)

    def _run_macro_recognition(self, aid: int, frame: Dict, visible_region: Circle):
        """Run macro-action level goal recognition (IGP2).

        This uses GoalRecognition which searches over macro-actions
        like Continue, Exit, ChangeLane, StopMA.
        """
        try:
            self._goal_recognition.update_goals_probabilities(
                goals_probabilities=self._goal_probabilities[aid],
                observed_trajectory=self._observations[aid][0],
                agent_id=aid,
                frame_ini=self._observations[aid][1],
                frame=frame,
                visible_region=visible_region,
                open_loop=(aid != self.agent_id)
            )

            if aid == self.agent_id:
                logger.debug(f"Ego macro-level recognition updated")
            else:
                logger.debug(f"Agent {aid} macro-level probabilities updated")

        except Exception as e:
            logger.debug(f"Macro-level recognition failed for agent {aid}: {e}")

    def _run_maneuver_recognition(self, aid: int, frame: Dict, visible_region: Circle):
        """Run maneuver-level goal recognition (epistemic).

        This uses ManeuverRecognition which searches over maneuvers
        directly: FollowLane, GiveWay, Turn, Stop, SwitchLane, etc.
        """
        try:
            self._maneuver_recognition.update_goals_probabilities(
                goals_probabilities=self._maneuver_probabilities[aid],
                observed_trajectory=self._observations[aid][0],
                agent_id=aid,
                frame_ini=self._observations[aid][1],
                frame=frame,
                visible_region=visible_region,
                open_loop=(aid != self.agent_id)
            )

            if aid == self.agent_id:
                # Log the predicted maneuver sequence
                maneuvers = self._maneuver_probabilities[aid].get_maneuver_sequence()
                logger.debug(f"Ego maneuver-level recognition updated: {maneuvers}")
            else:
                logger.debug(f"Agent {aid} maneuver-level probabilities updated")

        except Exception as e:
            logger.debug(f"Maneuver-level recognition failed for agent {aid}: {e}")

    def _update_ego_with_true_goal(self, observation: Observation, frame: Dict,
                                    visible_region: Circle):
        """Update ego prediction using the true/actual goal instead of goal recognition.

        This bypasses goal prediction but still computes the path using A*.
        Useful when you want to analyze the path to a known destination
        without the uncertainty of goal recognition.

        Works with both macro-level and maneuver-level prediction.
        """
        if self.goal is None or self.agent_id not in self._observations:
            return

        try:
            if self._prediction_level == "macro":
                # Macro-level: Use GoalsProbabilities and GoalRecognition
                ego_goals = GoalsProbabilities([self.goal])

                self._goal_recognition.update_goals_probabilities(
                    goals_probabilities=ego_goals,
                    observed_trajectory=self._observations[self.agent_id][0],
                    agent_id=self.agent_id,
                    frame_ini=self._observations[self.agent_id][1],
                    frame=frame,
                    visible_region=visible_region,
                    open_loop=False
                )

                # Set probability to 1.0 for the true goal
                for key in ego_goals.goals_probabilities:
                    ego_goals.goals_probabilities[key] = 1.0

                self._goal_probabilities[self.agent_id] = ego_goals
                logger.debug(f"Ego macro-level prediction updated with true goal")

            else:  # maneuver level
                # Maneuver-level: Use ManeuverProbabilities and ManeuverRecognition
                ego_maneuver_probs = ManeuverProbabilities([self.goal])

                self._maneuver_recognition.update_goals_probabilities(
                    goals_probabilities=ego_maneuver_probs,
                    observed_trajectory=self._observations[self.agent_id][0],
                    agent_id=self.agent_id,
                    frame_ini=self._observations[self.agent_id][1],
                    frame=frame,
                    visible_region=visible_region,
                    open_loop=False
                )

                # Set probability to 1.0 for the true goal
                for key in ego_maneuver_probs.goals_probabilities:
                    ego_maneuver_probs.goals_probabilities[key] = 1.0

                self._maneuver_probabilities[self.agent_id] = ego_maneuver_probs

                # Log predicted maneuver sequence
                maneuvers = ego_maneuver_probs.get_maneuver_sequence()
                logger.debug(f"Ego maneuver-level prediction updated with true goal: {maneuvers}")

        except Exception as e:
            logger.debug(f"Failed to update ego with true goal: {e}")

    def _update_mcts_plan(self, observation: Observation):
        """Run MCTS to generate optimal plan for ego."""
        if self.goal is None:
            logger.debug("No goal set for ego, skipping MCTS planning")
            return

        frame = observation.frame
        agents_metadata = {aid: state.metadata for aid, state in frame.items()}

        # Get goal probabilities for non-ego agents (for MCTS predictions)
        non_ego_predictions = {
            aid: gp for aid, gp in self._goal_probabilities.items()
            if aid != self.agent_id
        }

        try:
            self._mcts_plan, _ = self._mcts.search(
                agent_id=self.agent_id,
                goal=self.goal,
                frame=frame,
                meta=agents_metadata,
                predictions=non_ego_predictions
            )

            if self._mcts_plan:
                logger.debug(f"MCTS plan: {[str(ma) for ma in self._mcts_plan]}")

                # Instantiate MacroActions from MCTSAction wrappers to get maneuvers
                self._mcts_macro_actions, self._mcts_maneuvers = \
                    self._instantiate_mcts_plan(observation)

        except Exception as e:
            logger.warning(f"MCTS planning failed: {e}")
            self._mcts_plan = []
            self._mcts_macro_actions = []
            self._mcts_maneuvers = []
            self._mcts_trajectory = None

    # Mapping from class names to MacroActionFactory registered names
    _MA_CLASS_TO_FACTORY_NAME = {
        "Continue": "Continue",
        "ChangeLaneLeft": "ChangeLaneLeft",
        "ChangeLaneRight": "ChangeLaneRight",
        "Exit": "Exit",
        "StopMA": "Stop",  # StopMA class is registered as "Stop"
    }

    def _instantiate_mcts_plan(self, observation: Observation) -> Tuple[List[MacroAction], List[List]]:
        """Instantiate MacroActions from MCTSAction wrappers to extract maneuvers.

        MCTSAction contains macro_action_type and ma_args, which we use to create
        actual MacroAction instances that have maneuvers.

        Returns:
            Tuple of (list of MacroActions, list of maneuver lists per macro-action)
        """
        macro_actions = []
        maneuvers_per_ma = []
        current_frame = observation.frame.copy()

        for mcts_action in self._mcts_plan:
            if not hasattr(mcts_action, 'macro_action_type'):
                continue

            try:
                # Create MacroActionConfig from MCTSAction
                ma_type = mcts_action.macro_action_type
                ma_class_name = ma_type.__name__

                # Map class name to factory registered name
                factory_name = self._MA_CLASS_TO_FACTORY_NAME.get(ma_class_name, ma_class_name)

                ma_args = mcts_action.ma_args.copy() if mcts_action.ma_args else {}
                ma_args['type'] = factory_name
                ma_args['open_loop'] = True

                config = MacroActionConfig(ma_args)

                # Create the actual MacroAction
                macro_action = MacroActionFactory.create(
                    config, self.agent_id, current_frame, self._scenario_map
                )

                if macro_action is not None:
                    macro_actions.append(macro_action)

                    # Extract maneuvers
                    if hasattr(macro_action, '_maneuvers') and macro_action._maneuvers:
                        maneuvers_per_ma.append(macro_action._maneuvers)
                    else:
                        maneuvers_per_ma.append([])

                    # Update frame for next macro-action (play forward)
                    try:
                        current_frame = MacroAction.play_forward_macro_action(
                            self.agent_id, self._scenario_map, current_frame, macro_action
                        )
                    except Exception:
                        pass  # Continue with same frame if play forward fails

            except Exception as e:
                logger.debug(f"Failed to instantiate MCTSAction {mcts_action}: {e}")
                continue

        return macro_actions, maneuvers_per_ma

    def _analyze_safety(self):
        """Compare predicted ego plan vs MCTS optimal plan to assess safety.

        Analysis happens at multiple levels:
        - Sequence comparison: Position-by-position comparison of action sequences
        - Macro-actions: High-level (Continue, Exit, ChangeLane, StopMA)
        - Maneuvers: Low-level (FollowLane, Turn, GiveWay, Stop, etc.)

        When prediction_level is "maneuver", the predicted plan is directly a maneuver sequence.
        Safety-critical maneuvers like GiveWay and Stop are checked specifically.
        """
        # Get predicted ego plan based on prediction level
        if self._prediction_level == "macro":
            ego_probs = self._goal_probabilities.get(self.agent_id)
        else:
            ego_probs = self._maneuver_probabilities.get(self.agent_id)

        if ego_probs is None:
            self._safety_analysis = SafetyAnalysis(
                is_safe=True,
                intervention_type=InterventionType.NONE,
                message="No prediction available",
                predicted_actions=[],
                optimal_actions=[],
                missing_actions=[],
                sequence_differences=[],
                first_divergence=-1,
                predicted_maneuvers=[],
                missing_maneuvers=[],
                risk_score=0.0
            )
            self._current_message = None
            return

        # Get most likely predicted goal and plan
        predicted_goal, predicted_prob = self._get_most_likely_goal(ego_probs)
        all_plans = ego_probs.all_plans.get((predicted_goal, None), [])
        predicted_plan = all_plans[0] if all_plans and len(all_plans) > 0 else []

        # Extract action/maneuver names based on prediction level
        if self._prediction_level == "macro":
            # all_plans is List[List[MacroAction]]
            predicted_actions = [type(ma).__name__ for ma in predicted_plan] if predicted_plan else []
            # Extract maneuvers from macro-actions
            predicted_maneuvers = self._extract_maneuvers_from_plan(predicted_plan)
        else:
            # all_plans is List[List[Maneuver]] at maneuver level
            # No macro-actions, just maneuvers
            predicted_actions = []  # No macro-actions in maneuver-level prediction
            predicted_maneuvers = [type(m).__name__ for m in predicted_plan] if predicted_plan else []
            logger.debug(f"Maneuver-level predicted sequence: {predicted_maneuvers}")

        # Get MCTS optimal plan macro-action names (MCTSAction wrappers)
        optimal_actions = []
        for ma in self._mcts_plan:
            if hasattr(ma, 'macro_action_type'):
                optimal_actions.append(ma.macro_action_type.__name__)
            else:
                optimal_actions.append(type(ma).__name__)

        # Analyze differences at macro-action level (set-based)
        missing_actions = self._find_missing_macro_actions(predicted_actions, optimal_actions)

        # Analyze sequence differences (position-based)
        sequence_differences, first_divergence = self._compare_action_sequences(
            predicted_actions, optimal_actions
        )

        # Extract maneuvers from MCTS plan
        mcts_maneuvers = self._extract_maneuvers_from_list(self._mcts_maneuvers)

        # Analyze differences at maneuver level (safety-critical maneuvers)
        missing_maneuvers = self._find_missing_safety_maneuvers(predicted_maneuvers, mcts_maneuvers)

        # Calculate risk score based on all levels including sequence differences
        risk_score = self._calculate_risk_score(
            predicted_actions, optimal_actions, missing_actions, missing_maneuvers,
            sequence_differences
        )

        # Determine intervention based on sequence differences
        intervention_type, message = self._determine_intervention(
            missing_actions, missing_maneuvers, risk_score, predicted_prob,
            sequence_differences
        )

        self._safety_analysis = SafetyAnalysis(
            is_safe=(intervention_type == InterventionType.NONE),
            intervention_type=intervention_type,
            message=message,
            predicted_actions=predicted_actions,
            optimal_actions=optimal_actions,
            missing_actions=missing_actions,
            sequence_differences=sequence_differences,
            first_divergence=first_divergence,
            predicted_maneuvers=predicted_maneuvers,
            missing_maneuvers=missing_maneuvers,
            risk_score=risk_score
        )

        # Create driver message
        if intervention_type != InterventionType.NONE:
            self._current_message = DriverMessage(
                text=message,
                severity="critical" if intervention_type == InterventionType.STOP else "warning",
                action_hint=self._get_action_hint(
                    intervention_type, missing_actions, missing_maneuvers, sequence_differences
                )
            )
        else:
            self._current_message = None

    def _extract_maneuvers_from_plan(self, plan: List[MacroAction]) -> List[str]:
        """Extract all maneuver type names from a plan of macro-actions."""
        maneuvers = []
        for ma in plan:
            if hasattr(ma, '_maneuvers') and ma._maneuvers:
                for m in ma._maneuvers:
                    maneuvers.append(type(m).__name__)
        return maneuvers

    def _find_missing_macro_actions(self, predicted: List[str], optimal: List[str]) -> List[str]:
        """Find macro-actions in optimal plan that are missing from predicted.

        Note: StopMA is a safety-critical macro-action.
        """
        safety_macro_actions = {"StopMA"}

        predicted_set = set(predicted)

        missing = []
        for action in optimal:
            if action in safety_macro_actions and action not in predicted_set:
                missing.append(action)

        return missing

    def _compare_action_sequences(self, predicted: List[str], optimal: List[str]) \
            -> Tuple[List[SequenceDifference], int]:
        """Compare predicted and optimal action sequences position by position.

        This detects differences like:
        - MCTS: [Exit, StopMA] vs Predicted: [Exit, Continue] -> difference at position 1
        - MCTS: [Exit, StopMA, Continue] vs Predicted: [Exit] -> missing actions at positions 1, 2

        Args:
            predicted: List of predicted macro-action names
            optimal: List of optimal (MCTS) macro-action names

        Returns:
            Tuple of (list of SequenceDifference, first_divergence_position)
            first_divergence_position is -1 if sequences are identical
        """
        safety_critical_actions = {"StopMA", "GiveWay", "Stop"}
        differences = []
        first_divergence = -1
        max_len = max(len(predicted), len(optimal))

        for i in range(max_len):
            pred_action = predicted[i] if i < len(predicted) else None
            opt_action = optimal[i] if i < len(optimal) else None

            if pred_action != opt_action:
                if first_divergence == -1:
                    first_divergence = i

                # Determine if this is a safety-critical difference
                is_critical = opt_action in safety_critical_actions if opt_action else False

                # Create description
                if pred_action is None:
                    desc = f"Position {i+1}: MCTS suggests {opt_action}, predicted plan is shorter"
                elif opt_action is None:
                    desc = f"Position {i+1}: Predicted has {pred_action}, MCTS plan is shorter"
                else:
                    desc = f"Position {i+1}: MCTS suggests {opt_action}, predicted has {pred_action}"

                if is_critical:
                    desc += " [SAFETY-CRITICAL]"

                differences.append(SequenceDifference(
                    position=i,
                    predicted_action=pred_action,
                    optimal_action=opt_action,
                    is_safety_critical=is_critical,
                    description=desc
                ))

        return differences, first_divergence

    def _extract_maneuvers_from_list(self, maneuvers_per_ma: List[List]) -> List[str]:
        """Flatten a list of maneuver lists into a single list of maneuver names."""
        maneuvers = []
        for ma_maneuvers in maneuvers_per_ma:
            for m in ma_maneuvers:
                maneuvers.append(type(m).__name__)
        return maneuvers

    def _find_missing_safety_maneuvers(self, predicted_maneuvers: List[str],
                                        mcts_maneuvers: List[str]) -> List[str]:
        """Find safety-critical maneuvers in MCTS plan that are missing from predicted.

        Safety-critical maneuvers:
        - GiveWay / GiveWayCL: Yielding to other traffic
        - Stop / StopCL: Coming to a complete stop

        Args:
            predicted_maneuvers: Maneuver names from predicted plan
            mcts_maneuvers: Maneuver names from MCTS plan

        Returns:
            List of safety-critical maneuvers in MCTS but not in predicted
        """
        safety_maneuvers = {"GiveWay", "GiveWayCL", "Stop", "StopCL"}

        predicted_set = set(predicted_maneuvers)

        # Find safety maneuvers in MCTS plan that are missing from predicted
        missing = []
        for maneuver in mcts_maneuvers:
            if maneuver in safety_maneuvers and maneuver not in predicted_set:
                if maneuver not in missing:  # Avoid duplicates
                    missing.append(maneuver)

        return missing

    def _calculate_risk_score(self, predicted: List[str], optimal: List[str],
                              missing_actions: List[str], missing_maneuvers: List[str],
                              sequence_differences: List[SequenceDifference] = None) -> float:
        """Calculate a risk score based on plan differences at both levels.

        Args:
            predicted: Predicted macro-action names
            optimal: Optimal (MCTS) macro-action names
            missing_actions: Missing macro-actions (e.g., StopMA)
            missing_maneuvers: Missing safety-critical maneuvers (e.g., GiveWay, Stop)
            sequence_differences: Position-by-position differences in action sequences
        """
        if not optimal and not predicted:
            return 0.0

        risk = 0.0

        # Risk from missing macro-actions (set-based)
        macro_weights = {"StopMA": 0.5}
        risk += sum(macro_weights.get(action, 0.2) for action in missing_actions)

        # Risk from missing safety maneuvers
        maneuver_weights = {"GiveWay": 0.4, "GiveWayCL": 0.4, "Stop": 0.5, "StopCL": 0.5}
        risk += sum(maneuver_weights.get(m, 0.2) for m in missing_maneuvers)

        # Risk from sequence differences (position-based)
        if sequence_differences:
            for diff in sequence_differences:
                # Safety-critical differences are weighted more heavily
                if diff.is_safety_critical:
                    risk += 0.3
                # Early divergences (first few actions) are more concerning
                elif diff.position == 0:
                    risk += 0.25  # First action differs
                elif diff.position == 1:
                    risk += 0.15  # Second action differs
                else:
                    risk += 0.1  # Later differences

        return min(1.0, risk)

    def _determine_intervention(self, missing_actions: List[str], missing_maneuvers: List[str],
                                risk_score: float, predicted_prob: float,
                                sequence_differences: List[SequenceDifference] = None) -> Tuple[InterventionType, str]:
        """Determine what intervention to apply based on safety analysis.

        Checks macro-action level, maneuver level, and sequence-level differences.
        Provides detailed reasoning about sequence differences for the driver.

        Args:
            missing_actions: Missing macro-actions (set-based comparison)
            missing_maneuvers: Missing safety-critical maneuvers
            risk_score: Calculated risk score (0.0 to 1.0)
            predicted_prob: Confidence in the predicted goal
            sequence_differences: Position-by-position differences in plans
        """
        # Low confidence prediction - be cautious
        if predicted_prob < 0.3:
            return InterventionType.NONE, "Low prediction confidence"

        # Check for safety-critical sequence differences first
        if sequence_differences:
            critical_diffs = [d for d in sequence_differences if d.is_safety_critical]
            if critical_diffs:
                # Find the first safety-critical difference
                first_critical = critical_diffs[0]
                if first_critical.optimal_action in ("StopMA", "Stop"):
                    msg = f"STOP REQUIRED - {first_critical.description}"
                    return InterventionType.STOP, msg
                elif first_critical.optimal_action in ("GiveWay", "GiveWayCL"):
                    msg = f"YIELD REQUIRED - {first_critical.description}"
                    return InterventionType.YIELD, msg

        # Check for critical missing macro-actions (set-based)
        if "StopMA" in missing_actions:
            return InterventionType.STOP, "STOP REQUIRED - Missing stop macro-action!"

        # Check for critical missing maneuvers
        if any(m in missing_maneuvers for m in ["Stop", "StopCL"]):
            return InterventionType.STOP, "STOP REQUIRED - Missing stop maneuver!"

        if any(m in missing_maneuvers for m in ["GiveWay", "GiveWayCL"]):
            return InterventionType.YIELD, "YIELD REQUIRED - Give way to other traffic!"

        # Check for early sequence divergences (non-critical)
        if sequence_differences:
            first_diff = sequence_differences[0] if sequence_differences else None
            if first_diff and first_diff.position == 0:
                # Plans differ from the very first action
                msg = f"CAUTION - Plans diverge immediately: {first_diff.description}"
                return InterventionType.SLOW_DOWN, msg
            elif first_diff and first_diff.position == 1:
                # Plans differ at second action
                msg = f"CAUTION - Plans differ at step 2: {first_diff.description}"
                if risk_score > 0.4:
                    return InterventionType.SLOW_DOWN, msg
                else:
                    return InterventionType.YIELD, msg

        # High risk score
        if risk_score > 0.6:
            return InterventionType.SLOW_DOWN, "CAUTION - Predicted plan may be unsafe!"

        if risk_score > 0.3:
            return InterventionType.YIELD, "Slow down - approaching potential conflict"

        return InterventionType.NONE, "Plan appears safe"

    def _get_action_hint(self, intervention: InterventionType, missing_actions: List[str],
                         missing_maneuvers: List[str],
                         sequence_differences: List[SequenceDifference] = None) -> str:
        """Get a hint for what the driver should do based on plan differences.

        Args:
            intervention: The type of intervention being applied
            missing_actions: Missing macro-actions
            missing_maneuvers: Missing safety-critical maneuvers
            sequence_differences: Position-by-position differences in plans
        """
        # Build a specific hint based on sequence differences when available
        if sequence_differences:
            critical_diffs = [d for d in sequence_differences if d.is_safety_critical]
            if critical_diffs:
                first_critical = critical_diffs[0]
                if first_critical.optimal_action in ("StopMA", "Stop"):
                    return f"Apply brakes - MCTS recommends stop at step {first_critical.position + 1}"
                elif first_critical.optimal_action in ("GiveWay", "GiveWayCL"):
                    return f"Yield to traffic - MCTS recommends giving way at step {first_critical.position + 1}"

        # Generic hints based on intervention type
        if intervention == InterventionType.STOP:
            return "Apply brakes and stop before proceeding"
        elif intervention == InterventionType.YIELD:
            return "Reduce speed and yield to crossing traffic"
        elif intervention == InterventionType.SLOW_DOWN:
            return "Reduce speed and assess the situation"
        return ""

    def _get_most_likely_goal(self, goal_probs: GoalsProbabilities) -> Tuple[Optional[Goal], float]:
        """Get the most likely goal from goal probabilities."""
        best_goal = None
        best_prob = 0.0
        for (goal, _), prob in goal_probs.goals_probabilities.items():
            if prob > best_prob:
                best_prob = prob
                best_goal = goal
        return best_goal, best_prob

    def _get_goals(self, observation: Observation, threshold: float = 2.0) -> List[Goal]:
        """Get all possible goals reachable from current position."""
        scenario_map = observation.scenario_map
        frame = observation.frame
        state = frame[self.agent_id]
        view_circle = Point(*state.position).buffer(self._view_radius)

        possible_goals = []

        for road in scenario_map.roads.values():
            if not road.boundary.intersects(view_circle):
                continue

            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.all_lanes:
                    if lane.id == 0 or lane.type != "driving":
                        continue

                    new_point = None

                    intersection = lane.midline.intersection(view_circle.boundary)
                    if not intersection.is_empty:
                        if hasattr(intersection, "geoms"):
                            max_distance = np.inf
                            for point in intersection.geoms:
                                if lane.distance_at(point) < max_distance:
                                    new_point = point
                        else:
                            new_point = intersection
                    elif view_circle.contains(lane.boundary):
                        if lane.link.successor is None:
                            new_point = Point(lane.midline.coords[-1])
                        else:
                            continue
                    else:
                        continue

                    new_point = np.array(new_point.coords[0])
                    if not any([np.allclose(new_point, g.center, atol=threshold) for _, g in possible_goals]):
                        new_goal = PointGoal(new_point, threshold=threshold)
                        possible_goals.append((lane, new_goal))

        # Add stopping goals
        stopping_goals = []
        if self._stop_goals:
            for aid, s in frame.items():
                if aid == self.agent_id:
                    continue
                if s.speed < Trajectory.VELOCITY_STOP:
                    stopping_goals.append(StoppingGoal(s.position, threshold=threshold))

        # Group neighboring lane goals
        goals = []
        used = []
        for lane, goal in possible_goals:
            if goal in used:
                continue

            neighbouring_goals = [goal]
            for other_lane, other_goal in possible_goals:
                if goal == other_goal:
                    continue
                if lane.parent_road == other_lane.parent_road and np.abs(lane.id - other_lane.id) == 1:
                    neighbouring_goals.append(other_goal)
                    used.append(other_goal)

            if len(neighbouring_goals) > 1:
                goals.append(PointCollectionGoal(neighbouring_goals))
            else:
                goals.append(goal)

        return goals + stopping_goals

    def next_state(self, observation: Observation, return_action: bool = False) -> AgentState:
        """Get next state by executing action through attached vehicle."""
        action = self.next_action(observation)
        if self._vehicle is not None:
            self.vehicle.execute_action(action, observation.frame[self.agent_id])
            next_state = self.vehicle.get_state(observation.frame[self.agent_id].time + 1)
        else:
            next_state = observation.frame[self.agent_id]

        if return_action:
            return next_state, action
        return next_state

    def reset(self):
        """Reset agent to initial state."""
        super().reset()
        self._vehicle = KinematicVehicle(self._initial_state, self.metadata, self._fps)
        self._reverse = False
        self._current_steer = 0.0
        self._observations = {}
        self._goal_probabilities = {}
        self._goals = []
        self._mcts_plan = []
        self._mcts_macro_actions = []
        self._mcts_maneuvers = []
        self._mcts_trajectory = None
        self._safety_analysis = None
        self._current_message = None
        self._intervention_active = False
        self._k = 0

    # ==================== Public API for Visualization ====================

    def get_ego_predictions(self) -> Optional[GoalsProbabilities]:
        """Get goal probabilities for the ego agent."""
        return self._goal_probabilities.get(self.agent_id)

    def get_agent_predictions(self, agent_id: int) -> Optional[GoalsProbabilities]:
        """Get goal probabilities for a specific agent."""
        return self._goal_probabilities.get(agent_id)

    def get_most_likely_ego_goal(self) -> Tuple[Optional[Goal], float]:
        """Get the most likely goal for the ego agent."""
        ego_probs = self.get_ego_predictions()
        if ego_probs is None:
            return None, 0.0
        return self._get_most_likely_goal(ego_probs)

    @property
    def view_radius(self) -> float:
        """The view radius of the agent."""
        return self._view_radius

    @property
    def observations(self) -> Dict[int, Tuple[StateTrajectory, Dict[int, AgentState]]]:
        """Observed trajectories for all agents."""
        return self._observations

    @property
    def possible_goals(self) -> List[Goal]:
        """Current list of possible goals."""
        return self._goals

    @property
    def goal_probabilities(self) -> Dict[int, GoalsProbabilities]:
        """Goal probabilities for all predicted agents."""
        return self._goal_probabilities

    @property
    def mcts_plan(self) -> List[MacroAction]:
        """The MCTS-generated optimal plan for ego."""
        return self._mcts_plan

    @property
    def mcts_trajectory(self) -> Optional[VelocityTrajectory]:
        """Trajectory from the MCTS optimal plan."""
        return self._mcts_trajectory

    @property
    def mcts_macro_actions(self) -> List[MacroAction]:
        """The instantiated MacroActions from the MCTS plan."""
        return self._mcts_macro_actions

    @property
    def mcts_maneuvers(self) -> List[List]:
        """Maneuvers for each macro-action in the MCTS plan.

        Returns a list of maneuver lists, one per macro-action.
        Maneuvers are low-level actions: FollowLane, Turn, GiveWay, Stop, etc.
        """
        return self._mcts_maneuvers

    @property
    def safety_analysis(self) -> Optional[SafetyAnalysis]:
        """Current safety analysis results."""
        return self._safety_analysis

    @property
    def current_message(self) -> Optional[DriverMessage]:
        """Current message for the driver."""
        return self._current_message

    @property
    def intervention_active(self) -> bool:
        """Whether an intervention is currently being applied."""
        return self._intervention_active

    @property
    def current_macro(self):
        """SharedAutonomyAgent doesn't use macro actions for control."""
        return self._current_macro

    @property
    def predict_ego(self) -> bool:
        """Whether ego is included in predictions."""
        return self._predict_ego

    @property
    def ego_goal_mode(self) -> str:
        """How ego's goal is determined for prediction.

        Returns:
            "goal_recognition": Predict goal from observed trajectory
            "true_goal": Use the actual ego goal, run A* to find path
        """
        return self._ego_goal_mode

    @ego_goal_mode.setter
    def ego_goal_mode(self, mode: str):
        """Set the ego goal prediction mode.

        Args:
            mode: Either "goal_recognition" or "true_goal"
        """
        if mode not in ("goal_recognition", "true_goal"):
            raise ValueError(f"Invalid ego_goal_mode: {mode}. "
                           f"Must be 'goal_recognition' or 'true_goal'")
        self._ego_goal_mode = mode

    @property
    def prediction_level(self) -> str:
        """Level of prediction granularity.

        Returns:
            "macro": Uses IGP2 GoalRecognition (macro-action level)
            "maneuver": Uses epistemic ManeuverRecognition (maneuver level)
        """
        return self._prediction_level

    @prediction_level.setter
    def prediction_level(self, level: str):
        """Set the prediction level.

        Args:
            level: Either "macro" or "maneuver"
        """
        if level not in ("macro", "maneuver"):
            raise ValueError(f"Invalid prediction_level: {level}. "
                           f"Must be 'macro' or 'maneuver'")
        self._prediction_level = level

    @property
    def maneuver_probabilities(self) -> Dict[int, ManeuverProbabilities]:
        """Get maneuver-level probabilities for all agents (when prediction_level='maneuver').

        This contains the maneuver sequences predicted for each agent.
        """
        return self._maneuver_probabilities

    @property
    def predicted_maneuver_sequence(self) -> List[str]:
        """Get the predicted maneuver sequence for ego (when prediction_level='maneuver').

        Returns a list of maneuver type names like ['FollowLane', 'GiveWay', 'Turn'].
        Returns empty list if prediction_level='macro' or no prediction available.
        """
        if self._prediction_level != "maneuver":
            return []

        ego_probs = self._maneuver_probabilities.get(self.agent_id)
        if ego_probs is None:
            return []

        return ego_probs.get_maneuver_sequence()

    @property
    def predicted_plan(self) -> List[MacroAction]:
        """Get the predicted plan (macro-actions) for ego from goal recognition.

        This is the first/best plan corresponding to the most likely goal.
        MacroActions are high-level actions: Continue, Exit, ChangeLane, StopMA.
        Each MacroAction contains a sequence of Maneuvers.
        """
        ego_probs = self._goal_probabilities.get(self.agent_id)
        if ego_probs is None:
            return []

        # Get most likely goal
        best_goal, _ = self._get_most_likely_goal(ego_probs)
        if best_goal is None:
            return []

        # all_plans returns List[List[MacroAction]] - get the first (best) plan
        plans = ego_probs.all_plans.get((best_goal, None), [])
        if plans and len(plans) > 0:
            return plans[0]  # Return the first/best plan
        return []

    @property
    def predicted_maneuvers(self) -> List[List]:
        """Get the maneuvers for each macro-action in the predicted plan.

        Returns a list of maneuver lists, one per macro-action.
        Maneuvers are low-level actions: FollowLane, Turn, GiveWay, Stop, etc.
        """
        plan = self.predicted_plan
        if not plan:
            return []

        maneuvers_per_ma = []
        for ma in plan:
            if hasattr(ma, '_maneuvers') and ma._maneuvers:
                maneuvers_per_ma.append(ma._maneuvers)
            elif hasattr(ma, 'get_maneuvers'):
                try:
                    maneuvers_per_ma.append(ma.get_maneuvers())
                except:
                    maneuvers_per_ma.append([])
            else:
                maneuvers_per_ma.append([])
        return maneuvers_per_ma

    @property
    def predicted_trajectory(self) -> Optional[VelocityTrajectory]:
        """Get the predicted trajectory for ego from goal recognition.

        This is the optimum trajectory for the most likely goal.
        """
        ego_probs = self._goal_probabilities.get(self.agent_id)
        if ego_probs is None:
            return None

        # Get most likely goal
        best_goal, _ = self._get_most_likely_goal(ego_probs)
        if best_goal is None:
            return None

        # Get the optimum trajectory for this goal
        return ego_probs.optimum_trajectory.get((best_goal, None))
