"""
Shared Autonomy Agent for human-in-the-loop driving with MCTS predictions.

This agent combines:
- Keyboard control for manual driving (like KeyboardAgent)
- Goal recognition and predictions for ALL agents including ego (like MCTSAgent)

The predictions can be used for:
- Visualizing predicted trajectories
- Shared autonomy interventions
- Human intent inference
- Safety monitoring

Controls:
    W / UP      : Accelerate
    S / DOWN    : Brake / Reverse
    A / LEFT    : Steer left
    D / RIGHT   : Steer right
"""

import logging
import os
import numpy as np
from typing import List, Dict, Tuple

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
from igp2.core.trajectory import Trajectory, StateTrajectory
from igp2.core.cost import Cost
from igp2.core.velocitysmoother import VelocitySmoother
from igp2.core.util import Circle, find_lane_sequence
from igp2.opendrive.map import Map
from igp2.planlibrary.maneuver import Maneuver
from igp2.planning.reward import Reward
from igp2.planning.mcts import MCTS
from igp2.recognition.astar import AStar
from igp2.recognition.goalrecognition import GoalRecognition
from igp2.recognition.goalprobabilities import GoalsProbabilities

logger = logging.getLogger(__name__)


class SharedAutonomyAgent(Agent):
    """Agent that combines keyboard control with MCTS predictions for all agents.

    This agent is controlled by a human via keyboard, but also performs goal
    recognition and trajectory prediction for all visible agents INCLUDING itself.
    This enables shared autonomy applications where the system predicts human intent.

    Key differences from MCTSAgent:
    - Control comes from keyboard, not MCTS planning
    - Goal recognition includes the ego agent (self)
    - Predictions are stored but not used for control

    Attributes:
        goal_probabilities: Dict mapping agent_id -> GoalsProbabilities for all agents
        observations: Dict mapping agent_id -> (trajectory, initial_frame) for all agents
        possible_goals: List of possible goals in the current view
    """

    # Control parameters (same as KeyboardAgent)
    MAX_ACCELERATION = 5.0  # m/s^2
    MAX_BRAKE = 8.0  # m/s^2
    MAX_STEER = 0.7  # radians
    STEER_SPEED = 0.05

    # Class-level pygame tracking
    _pygame_initialized = False

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 scenario_map: Map,
                 goal: Goal = None,
                 view_radius: float = 50.0,
                 fps: int = 20,
                 t_update: float = 1.0,
                 cost_factors: Dict[str, float] = None,
                 velocity_smoother: dict = None,
                 goal_recognition: dict = None,
                 stop_goals: bool = False,
                 predict_ego: bool = True):
        """Initialize a shared autonomy agent.

        Args:
            agent_id: ID of the agent
            initial_state: Starting state of the agent
            scenario_map: The road network map
            goal: Optional final goal of the agent (for ego prediction)
            view_radius: Radius within which other agents are visible
            fps: Execution rate of the environment simulation
            t_update: Time interval between prediction updates (seconds)
            cost_factors: Cost factors for trajectory evaluation
            velocity_smoother: Velocity smoother parameters
            goal_recognition: Goal recognition parameters
            stop_goals: Whether to include stopping goals
            predict_ego: Whether to include ego in goal recognition (default True)
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

        # Update timing
        self._k = 0
        self._kmax = t_update * fps

        # Goal recognition setup
        self._cost = Cost(factors=cost_factors) if cost_factors is not None else Cost()
        self._astar = AStar(next_lane_offset=0.1)

        if velocity_smoother is None:
            velocity_smoother = {"vmin_m_s": 1, "vmax_m_s": 10, "n": 10, "amax_m_s2": 5, "lambda_acc": 10}
        self._smoother = VelocitySmoother(**velocity_smoother)

        if goal_recognition is None:
            goal_recognition = {"reward_as_difference": False, "n_trajectories": 2}
        self._goal_recognition = GoalRecognition(
            astar=self._astar,
            smoother=self._smoother,
            scenario_map=scenario_map,
            cost=self._cost,
            **goal_recognition
        )

        # Storage for predictions
        self._goal_probabilities: Dict[int, GoalsProbabilities] = {}
        self._observations: Dict[int, Tuple[StateTrajectory, Dict[int, AgentState]]] = {}
        self._goals: List[Goal] = []

        # For compatibility with systems expecting current_macro
        self._current_macro = None
        self._pgp_control = False
        self._pgp_drive = False

        if self._pygame_available:
            logger.info(f"SharedAutonomyAgent {agent_id} initialized with prediction. "
                       f"predict_ego={predict_ego}")
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
        """Get next action from keyboard and update predictions.

        This method:
        1. Updates observations for all agents
        2. Periodically runs goal recognition for all agents (including ego if enabled)
        3. Returns keyboard-controlled action

        Args:
            observation: Current observation of the environment
            prediction: Optional external prediction (unused)

        Returns:
            Action from keyboard input
        """
        # Update observations for all agents
        self._update_observations(observation)

        # Periodically update predictions
        self._k += 1
        if self._k >= self._kmax:
            self._update_predictions(observation)
            self._k = 0

        # Get keyboard action
        return self._get_keyboard_action(observation)

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
        """Run goal recognition for all agents including ego (if enabled)."""
        frame = observation.frame

        # Get possible goals from ego's perspective
        self._goals = self._get_goals(observation)

        if not self._goals:
            logger.debug("No goals found in view, skipping prediction update")
            return

        # Initialize goal probabilities for all agents
        agents_to_predict = list(frame.keys())
        if not self._predict_ego:
            agents_to_predict = [aid for aid in agents_to_predict if aid != self.agent_id]

        self._goal_probabilities = {
            aid: GoalsProbabilities(self._goals) for aid in agents_to_predict
        }

        visible_region = Circle(frame[self.agent_id].position, self._view_radius)

        # Run goal recognition for each agent
        for aid in agents_to_predict:
            if aid not in self._observations:
                continue

            try:
                self._goal_recognition.update_goals_probabilities(
                    goals_probabilities=self._goal_probabilities[aid],
                    observed_trajectory=self._observations[aid][0],
                    agent_id=aid,
                    frame_ini=self._observations[aid][1],
                    frame=frame,
                    visible_region=visible_region
                )

                if aid == self.agent_id:
                    logger.info(f"Ego agent goal probabilities:")
                else:
                    logger.info(f"Agent {aid} goal probabilities:")
                self._goal_probabilities[aid].log(logger)

            except Exception as e:
                logger.debug(f"Goal recognition failed for agent {aid}: {e}")

    def _get_goals(self, observation: Observation, threshold: float = 2.0) -> List[Goal]:
        """Get all possible goals reachable from current position.

        This is similar to MCTSAgent.get_goals() but operates from ego's position.
        """
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
        self._k = 0

    def get_ego_predictions(self) -> GoalsProbabilities:
        """Get goal probabilities for the ego agent.

        Returns:
            GoalsProbabilities for ego, or None if not available
        """
        return self._goal_probabilities.get(self.agent_id)

    def get_agent_predictions(self, agent_id: int) -> GoalsProbabilities:
        """Get goal probabilities for a specific agent.

        Args:
            agent_id: ID of the agent

        Returns:
            GoalsProbabilities for the agent, or None if not available
        """
        return self._goal_probabilities.get(agent_id)

    def get_most_likely_ego_goal(self) -> Tuple[Goal, float]:
        """Get the most likely goal for the ego agent.

        Returns:
            Tuple of (goal, probability), or (None, 0) if not available
        """
        ego_probs = self.get_ego_predictions()
        if ego_probs is None:
            return None, 0.0

        best_goal = None
        best_prob = 0.0
        for (goal, _), prob in ego_probs.goals_probabilities.items():
            if prob > best_prob:
                best_prob = prob
                best_goal = goal

        return best_goal, best_prob

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
    def current_macro(self):
        """SharedAutonomyAgent doesn't use macro actions."""
        return self._current_macro

    @property
    def predict_ego(self) -> bool:
        """Whether ego is included in predictions."""
        return self._predict_ego
