"""
Keyboard-controlled agent for manual driving in CARLA simulations.

Controls:
    W / UP      : Accelerate
    S / DOWN    : Brake / Reverse
    A / LEFT    : Steer left
    D / RIGHT   : Steer right
    SPACE       : Hand brake
    Q           : Toggle reverse mode
"""

import logging
import os

# Set SDL to use dummy video driver if no display is available
# This must be done before importing pygame
if os.environ.get('DISPLAY') is None and os.environ.get('SDL_VIDEODRIVER') is None:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
from pygame.locals import K_w, K_s, K_a, K_d, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_SPACE, K_q

from igp2.agents.agent import Agent
from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal
from igp2.core.vehicle import Action, Observation, TrajectoryPrediction, KinematicVehicle

logger = logging.getLogger(__name__)


class KeyboardAgent(Agent):
    """Agent that can be controlled via keyboard input using pygame.

    This agent reads keyboard state each frame and converts it to vehicle controls.
    Initializes pygame automatically if not already initialized.
    """

    # Control parameters
    MAX_ACCELERATION = 5.0  # m/s^2
    MAX_BRAKE = 8.0  # m/s^2 (braking deceleration)
    MAX_STEER = 0.7  # radians (about 40 degrees)
    STEER_SPEED = 0.05  # How quickly steering responds

    # Class-level flag to track if we initialized pygame
    _pygame_initialized = False

    def __init__(self,
                 agent_id: int,
                 initial_state: AgentState,
                 goal: Goal = None,
                 fps: int = 20):
        """Initialize a keyboard-controlled agent.

        Args:
            agent_id: ID of the agent
            initial_state: Starting state of the agent
            goal: Optional final goal of the agent
            fps: Execution rate of the environment simulation
        """
        super().__init__(agent_id, initial_state, goal, fps)
        self._vehicle = KinematicVehicle(initial_state, self.metadata, fps)
        self._reverse = False
        self._reverse_cooldown = 0.0
        self._current_steer = 0.0
        self._hand_brake = False
        self._pgp_control = False  # KeyboardAgent doesn't use PGP control
        self._pgp_drive = False  # KeyboardAgent doesn't use PGP drive
        self._current_macro = None  # KeyboardAgent doesn't use macro actions
        self._pygame_available = self._ensure_pygame_initialized()

        if self._pygame_available:
            logger.info(f"KeyboardAgent {agent_id} initialized. Use WASD or arrow keys to drive.")
        else:
            logger.warning(f"KeyboardAgent {agent_id} initialized but pygame unavailable. "
                          f"Agent will output zero controls.")

    @classmethod
    def _ensure_pygame_initialized(cls) -> bool:
        """Ensure pygame is initialized. Returns True if successful."""
        if cls._pygame_initialized:
            return True

        try:
            # Try to initialize pygame
            pygame.init()

            # For keyboard input, we need the display module
            if not pygame.display.get_init():
                pygame.display.init()

            # Try to set a display mode (required for keyboard input)
            if pygame.display.get_surface() is None:
                # Create a minimal window
                pygame.display.set_mode((100, 100), pygame.NOFRAME)

            # Test that keyboard reading actually works
            pygame.event.pump()
            pygame.key.get_pressed()

            cls._pygame_initialized = True
            logger.debug("Pygame initialized successfully for KeyboardAgent")
            return True

        except Exception as e:
            logger.warning(f"Could not initialize pygame for keyboard input: {e}")
            cls._pygame_initialized = True  # Mark as attempted so we don't retry
            return False

    def done(self, observation: Observation) -> bool:
        """Keyboard agent is never 'done' - it continues until simulation ends."""
        return False

    def next_action(self, observation: Observation, prediction: TrajectoryPrediction = None) -> Action:
        """Read keyboard input and return corresponding action.

        Args:
            observation: Current observation of the environment
            prediction: Optional trajectory prediction (unused)

        Returns:
            Action with acceleration and steering based on keyboard input
        """
        current_state = observation.frame.get(self.agent_id)
        current_speed = current_state.speed if current_state else 0.0

        # If pygame isn't available, return zero action maintaining current speed
        if not self._pygame_available:
            return Action(0.0, 0.0, target_speed=current_speed)

        try:
            # Process any pending events (required for keyboard state to update)
            pygame.event.pump()

            # Check if our window has focus - if not, don't respond to keys
            if not pygame.key.get_focused():
                return Action(0.0, 0.0, target_speed=current_speed)

            # Get current keyboard state
            keys = pygame.key.get_pressed()
        except pygame.error as e:
            # If pygame fails at runtime, log and return zero action
            logger.debug(f"Pygame error reading keyboard: {e}")
            return Action(0.0, 0.0, target_speed=current_speed)

        # # Get Reverse
        # if keys[K_q] and self._reverse_cooldown == 0.0:
        #     self._reverse = not self._reverse
        #     logger.info(f"Reverse mode: {'ON' if self._reverse else 'OFF'}")
        # self._reverse_cooldown = max(self._reverse_cooldown - 1.0, 0.0)

        # Calculate acceleration
        acceleration = 0.0
        if keys[K_w] or keys[K_UP]:
            if self._reverse:
                acceleration = -self.MAX_BRAKE  # Reverse acceleration
            else:
                acceleration = self.MAX_ACCELERATION
        elif keys[K_s] or keys[K_DOWN]:
            if self._reverse:
                acceleration = self.MAX_ACCELERATION  # Brake in reverse = accelerate forward
            else:
                acceleration = -self.MAX_BRAKE

        # Calculate steering (with smooth interpolation)
        target_steer = 0.0
        if keys[K_a] or keys[K_LEFT]:
            target_steer = -self.MAX_STEER
        elif keys[K_d] or keys[K_RIGHT]:
            target_steer = self.MAX_STEER

        # Smooth steering transition
        steer_diff = target_steer - self._current_steer
        if abs(steer_diff) > self.STEER_SPEED:
            self._current_steer += self.STEER_SPEED if steer_diff > 0 else -self.STEER_SPEED
        else:
            self._current_steer = target_steer

        # # Handle hand brake
        # if keys[K_SPACE]:
        #     self._hand_brake = True
        #     acceleration = -self.MAX_BRAKE  # Apply maximum braking
        # else:
        #     self._hand_brake = False

        # Toggle reverse mode
        # Note: This should be handled in parse_events for proper key up detection
        # Here we just use current state

        # Compute target speed from current speed and acceleration
        dt = 1.0 / self._fps
        target_speed = max(0.0, current_speed + acceleration * dt)

        return Action(acceleration, self._current_steer, target_speed=target_speed)

    def next_state(self, observation: Observation, return_action: bool = False) -> AgentState:
        """Get next state by executing action through attached vehicle.

        Args:
            observation: Current observation
            return_action: If True, also return the action taken

        Returns:
            Next agent state (and optionally the action)
        """
        action = self.next_action(observation)
        if self._vehicle is not None:
            self.vehicle.execute_action(action, observation.frame[self.agent_id])
            next_state = self.vehicle.get_state(observation.frame[self.agent_id].time + 1)
        else:
            next_state = observation.frame[self.agent_id]

        if return_action:
            return next_state, action
        return next_state

    def parse_events(self):
        """Parse pygame events for toggle keys like reverse mode.

        Call this once per frame to handle key press/release events.
        """
        # for event in pygame.event.get():
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == K_q:
        #             self._reverse = not self._reverse
        #             logger.info(f"Reverse mode: {'ON' if self._reverse else 'OFF'}")
        pass

    def reset(self):
        """Reset agent to initial state."""
        super().reset()
        self._vehicle = KinematicVehicle(self._initial_state, self.metadata, self._fps)
        self._reverse = False
        self._current_steer = 0.0
        self._hand_brake = False

    @property
    def reverse(self) -> bool:
        """Whether the agent is in reverse mode."""
        return self._reverse

    @property
    def hand_brake(self) -> bool:
        """Whether the hand brake is engaged."""
        return self._hand_brake

    @property
    def current_macro(self):
        """KeyboardAgent doesn't use macro actions, always returns None."""
        return self._current_macro
