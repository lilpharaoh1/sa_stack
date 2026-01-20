"""
Prediction Visualizer for SharedAutonomyAgent.

A pygame-based top-down visualization window that displays:
- Agent locations with IDs
- Predicted goals with probabilities
- Predicted trajectories/paths
- View radius circle

Updated in real-time as predictions are made.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

# Set SDL environment before importing pygame
if os.environ.get('DISPLAY') is None and os.environ.get('SDL_VIDEODRIVER') is None:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
from pygame import gfxdraw

from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal
from igp2.recognition.goalprobabilities import GoalsProbabilities

logger = logging.getLogger(__name__)


class PredictionVisualizer:
    """Top-down visualization of agent predictions from SharedAutonomyAgent.

    This visualizer shows a 2D bird's-eye view of:
    - All agent positions (colored by type)
    - View radius around ego agent
    - Possible goals with probability labels
    - Predicted trajectories to goals
    - Failure zones (if provided)
    """

    # Colors (RGB)
    COLOR_BG = (40, 40, 40)
    COLOR_ROAD = (80, 80, 80)
    COLOR_EGO = (0, 200, 100)
    COLOR_TRAFFIC = (200, 100, 50)
    COLOR_GOAL = (100, 200, 255)
    COLOR_GOAL_BEST = (50, 255, 100)
    COLOR_TRAJECTORY = (255, 200, 100)
    COLOR_TRAJECTORY_OPT = (100, 255, 150)
    COLOR_VIEW_RADIUS = (100, 100, 150)
    COLOR_FAILURE_ZONE = (255, 50, 50)
    COLOR_TEXT = (255, 255, 255)
    COLOR_GRID = (60, 60, 60)

    def __init__(self,
                 width: int = 800,
                 height: int = 600,
                 title: str = "Prediction Visualizer",
                 scale: float = 3.0,
                 center: Tuple[float, float] = None):
        """Initialize the prediction visualizer.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            title: Window title
            scale: Pixels per meter (zoom level)
            center: Initial world center position (x, y). If None, centers on ego.
        """
        self.width = width
        self.height = height
        self.title = title
        self.scale = scale
        self.center = center  # World coordinates to center view on
        self._follow_ego = center is None

        # Pygame state
        self._display: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None
        self._initialized = False

        # Data storage
        self._frame: Dict[int, AgentState] = {}
        self._ego_id: int = 0
        self._goal_probabilities: Dict[int, GoalsProbabilities] = {}
        self._possible_goals: List[Goal] = []
        self._view_radius: float = 50.0
        self._failure_zones: List[dict] = []
        self._current_timestep: int = 0
        self._in_failure_zone: bool = False

    def initialize(self) -> bool:
        """Initialize pygame and create the window.

        Returns:
            True if successful, False otherwise.
        """
        try:
            pygame.init()
            pygame.font.init()

            self._display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption(self.title)

            self._clock = pygame.time.Clock()
            self._font = pygame.font.SysFont('monospace', 14)
            self._font_small = pygame.font.SysFont('monospace', 11)

            self._initialized = True
            logger.info(f"PredictionVisualizer initialized: {self.width}x{self.height}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PredictionVisualizer: {e}")
            return False

    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates.

        Args:
            world_pos: World position (x, y) in meters

        Returns:
            Screen position (x, y) in pixels
        """
        if self.center is None:
            center = np.array([0.0, 0.0])
        else:
            center = np.array(self.center)

        # Relative position from center
        rel_pos = world_pos - center

        # Scale and flip Y axis (screen Y is inverted)
        screen_x = int(self.width / 2 + rel_pos[0] * self.scale)
        screen_y = int(self.height / 2 - rel_pos[1] * self.scale)

        return (screen_x, screen_y)

    def update(self,
               frame: Dict[int, AgentState],
               ego_id: int,
               goal_probabilities: Dict[int, GoalsProbabilities] = None,
               possible_goals: List[Goal] = None,
               view_radius: float = 50.0,
               failure_zones: List[dict] = None,
               timestep: int = 0) -> bool:
        """Update the visualization with new data.

        Args:
            frame: Current frame with agent states
            ego_id: ID of the ego agent
            goal_probabilities: Goal probabilities for each agent
            possible_goals: List of possible goals
            view_radius: View radius of ego agent
            failure_zones: List of failure zone dicts with 'box' and 'frames' keys
            timestep: Current simulation timestep

        Returns:
            True if should continue, False if window was closed.
        """
        if not self._initialized:
            if not self.initialize():
                return False

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.scale *= 1.2
                elif event.key == pygame.K_MINUS:
                    self.scale /= 1.2
                elif event.key == pygame.K_f:
                    self._follow_ego = not self._follow_ego

        # Store data
        self._frame = frame
        self._ego_id = ego_id
        self._goal_probabilities = goal_probabilities or {}
        self._possible_goals = possible_goals or []
        self._view_radius = view_radius
        self._failure_zones = failure_zones or []
        self._current_timestep = timestep

        # Update center if following ego
        if self._follow_ego and ego_id in frame:
            self.center = frame[ego_id].position

        # Check if ego is in a failure zone
        self._in_failure_zone = False
        if ego_id in frame:
            ego_pos = frame[ego_id].position
            for fz in self._failure_zones:
                frames = fz["frames"]
                if frames["start"] <= timestep <= frames["end"]:
                    if fz["box"].inside(ego_pos):
                        self._in_failure_zone = True
                        break

        # Render
        self._render()

        pygame.display.flip()
        self._clock.tick(60)

        return True

    def _render(self):
        """Render all visualization elements."""
        self._display.fill(self.COLOR_BG)

        # Draw grid
        self._draw_grid()

        # Draw failure zones
        self._draw_failure_zones()

        # Draw view radius
        if self._ego_id in self._frame:
            self._draw_view_radius()

        # Draw goals with probabilities
        self._draw_goals()

        # Draw predicted trajectories
        self._draw_trajectories()

        # Draw agents
        self._draw_agents()

        # Draw HUD
        self._draw_hud()

    def _draw_grid(self):
        """Draw a reference grid."""
        grid_spacing = 20  # meters

        if self.center is None:
            return

        # Calculate grid bounds
        half_width = (self.width / 2) / self.scale
        half_height = (self.height / 2) / self.scale

        min_x = int((self.center[0] - half_width) / grid_spacing) * grid_spacing
        max_x = int((self.center[0] + half_width) / grid_spacing + 1) * grid_spacing
        min_y = int((self.center[1] - half_height) / grid_spacing) * grid_spacing
        max_y = int((self.center[1] + half_height) / grid_spacing + 1) * grid_spacing

        # Draw vertical lines
        for x in range(int(min_x), int(max_x) + 1, grid_spacing):
            start = self.world_to_screen(np.array([x, min_y]))
            end = self.world_to_screen(np.array([x, max_y]))
            pygame.draw.line(self._display, self.COLOR_GRID, start, end, 1)

        # Draw horizontal lines
        for y in range(int(min_y), int(max_y) + 1, grid_spacing):
            start = self.world_to_screen(np.array([min_x, y]))
            end = self.world_to_screen(np.array([max_x, y]))
            pygame.draw.line(self._display, self.COLOR_GRID, start, end, 1)

    def _draw_view_radius(self):
        """Draw the ego's view radius circle."""
        ego_state = self._frame[self._ego_id]
        center = self.world_to_screen(ego_state.position)
        radius = int(self._view_radius * self.scale)

        # Draw filled semi-transparent circle
        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_VIEW_RADIUS, 30), (radius, radius), radius)
        self._display.blit(s, (center[0] - radius, center[1] - radius))

        # Draw circle outline
        pygame.draw.circle(self._display, self.COLOR_VIEW_RADIUS, center, radius, 2)

    def _draw_failure_zones(self):
        """Draw failure zones with time-based highlighting."""
        for fz in self._failure_zones:
            box = fz["box"]
            frames = fz["frames"]

            # Check if zone is active
            is_active = frames["start"] <= self._current_timestep <= frames["end"]

            # Get box corners
            corners = box.boundary
            screen_corners = [self.world_to_screen(np.array(c)) for c in corners]

            if is_active:
                # Draw filled zone
                s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.polygon(s, (*self.COLOR_FAILURE_ZONE, 50), screen_corners)
                self._display.blit(s, (0, 0))
                # Draw outline
                pygame.draw.polygon(self._display, self.COLOR_FAILURE_ZONE, screen_corners, 3)
            else:
                # Draw outline only (dimmed)
                color = tuple(c // 2 for c in self.COLOR_FAILURE_ZONE)
                pygame.draw.polygon(self._display, color, screen_corners, 1)

            # Draw label
            center = self.world_to_screen(box.center)
            label = f"{frames['start']}-{frames['end']}"
            text = self._font_small.render(label, True, self.COLOR_FAILURE_ZONE if is_active else color)
            self._display.blit(text, (center[0] - text.get_width() // 2, center[1]))

    def _draw_goals(self):
        """Draw possible goals with probability labels."""
        if not self._possible_goals:
            return

        # Get ego's goal probabilities if available
        ego_probs = self._goal_probabilities.get(self._ego_id)

        # Find best goal
        best_goal = None
        best_prob = 0.0
        if ego_probs:
            for (goal, _), prob in ego_probs.goals_probabilities.items():
                if prob > best_prob:
                    best_prob = prob
                    best_goal = goal

        for goal in self._possible_goals:
            pos = self.world_to_screen(goal.center)

            # Determine color
            is_best = goal == best_goal
            color = self.COLOR_GOAL_BEST if is_best else self.COLOR_GOAL

            # Draw goal marker (diamond shape)
            size = 8 if is_best else 6
            points = [
                (pos[0], pos[1] - size),
                (pos[0] + size, pos[1]),
                (pos[0], pos[1] + size),
                (pos[0] - size, pos[1])
            ]
            pygame.draw.polygon(self._display, color, points)
            pygame.draw.polygon(self._display, self.COLOR_TEXT, points, 1)

            # Draw probability label if available
            if ego_probs:
                for (g, _), prob in ego_probs.goals_probabilities.items():
                    if np.allclose(g.center, goal.center, atol=2.0):
                        if prob > 0.01:
                            label = f"{prob:.2f}"
                            text = self._font_small.render(label, True, color)
                            self._display.blit(text, (pos[0] + 10, pos[1] - 8))
                        break

    def _draw_trajectories(self):
        """Draw predicted trajectories for all agents."""
        for aid, goal_probs in self._goal_probabilities.items():
            if goal_probs is None:
                continue

            color = self.COLOR_EGO if aid == self._ego_id else self.COLOR_TRAFFIC

            # Draw trajectories to each goal
            for (goal, _), trajs in goal_probs.all_trajectories.items():
                prob = goal_probs.goals_probabilities.get((goal, None), 0.0)
                if prob < 0.01:
                    continue

                for traj in trajs[:2]:  # Limit to 2 trajectories per goal
                    if traj is None:
                        continue

                    path = traj.path
                    if len(path) < 2:
                        continue

                    # Draw path with alpha based on probability
                    alpha = int(255 * min(1.0, prob * 2))
                    traj_color = (*color[:3], alpha)

                    screen_points = [self.world_to_screen(p) for p in path[::3]]  # Subsample
                    if len(screen_points) >= 2:
                        # Create surface for alpha
                        s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                        pygame.draw.lines(s, traj_color, False, screen_points, 2)
                        self._display.blit(s, (0, 0))

            # Draw optimal trajectory (brighter)
            for (goal, _), opt_traj in goal_probs.optimum_trajectory.items():
                if opt_traj is None:
                    continue

                prob = goal_probs.goals_probabilities.get((goal, None), 0.0)
                if prob < 0.1:
                    continue

                path = opt_traj.path
                if len(path) < 2:
                    continue

                screen_points = [self.world_to_screen(p) for p in path[::3]]
                if len(screen_points) >= 2:
                    opt_color = self.COLOR_TRAJECTORY_OPT if aid == self._ego_id else self.COLOR_TRAJECTORY
                    pygame.draw.lines(self._display, opt_color, False, screen_points, 2)

    def _draw_agents(self):
        """Draw all agents as triangles indicating heading."""
        for aid, state in self._frame.items():
            pos = self.world_to_screen(state.position)
            heading = state.heading

            # Determine color
            is_ego = aid == self._ego_id
            color = self.COLOR_EGO if is_ego else self.COLOR_TRAFFIC

            # Draw vehicle as triangle pointing in heading direction
            size = 12 if is_ego else 8

            # Triangle points
            front = np.array([np.cos(heading), np.sin(heading)]) * size
            back_left = np.array([np.cos(heading + 2.5), np.sin(heading + 2.5)]) * size * 0.6
            back_right = np.array([np.cos(heading - 2.5), np.sin(heading - 2.5)]) * size * 0.6

            points = [
                (pos[0] + front[0], pos[1] - front[1]),
                (pos[0] + back_left[0], pos[1] - back_left[1]),
                (pos[0] + back_right[0], pos[1] - back_right[1])
            ]

            pygame.draw.polygon(self._display, color, points)
            pygame.draw.polygon(self._display, self.COLOR_TEXT, points, 1)

            # Draw agent ID label
            label = f"{aid}"
            text = self._font_small.render(label, True, color)
            self._display.blit(text, (pos[0] + 12, pos[1] - 6))

            # Draw speed label
            speed_label = f"{state.speed:.1f}m/s"
            speed_text = self._font_small.render(speed_label, True, (180, 180, 180))
            self._display.blit(speed_text, (pos[0] + 12, pos[1] + 6))

    def _draw_hud(self):
        """Draw heads-up display with info."""
        y_offset = 10
        line_height = 18

        # Failure zone status banner
        if self._in_failure_zone:
            banner_rect = pygame.Rect(0, 0, self.width, 35)
            pygame.draw.rect(self._display, self.COLOR_FAILURE_ZONE, banner_rect)
            status_text = self._font.render("FAILURE ZONE ACTIVE", True, self.COLOR_TEXT)
            self._display.blit(status_text, (self.width // 2 - status_text.get_width() // 2, 8))
            y_offset = 45

        # Title
        text = self._font.render(f"Prediction Visualizer - Frame {self._current_timestep}", True, self.COLOR_TEXT)
        self._display.blit(text, (10, y_offset))
        y_offset += line_height + 5

        # Controls
        controls = [
            "+/- : Zoom",
            "F   : Toggle follow ego",
            "ESC : Close"
        ]
        for ctrl in controls:
            text = self._font_small.render(ctrl, True, (150, 150, 150))
            self._display.blit(text, (10, y_offset))
            y_offset += line_height - 4

        y_offset += 10

        # Scale info
        text = self._font_small.render(f"Scale: {self.scale:.1f} px/m", True, (150, 150, 150))
        self._display.blit(text, (10, y_offset))
        y_offset += line_height

        # Follow mode
        mode = "Following ego" if self._follow_ego else "Fixed view"
        text = self._font_small.render(f"Mode: {mode}", True, (150, 150, 150))
        self._display.blit(text, (10, y_offset))
        y_offset += line_height + 10

        # Goal probabilities panel (right side)
        self._draw_goal_panel()

    def _draw_goal_panel(self):
        """Draw goal probabilities panel on the right side."""
        panel_width = 200
        panel_x = self.width - panel_width - 10
        y_offset = 10
        line_height = 16

        # Panel background
        panel_rect = pygame.Rect(panel_x - 5, 5, panel_width + 10, self.height - 10)
        s = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        s.fill((30, 30, 30, 180))
        self._display.blit(s, (panel_rect.x, panel_rect.y))

        # Title
        text = self._font.render("Goal Probabilities", True, self.COLOR_TEXT)
        self._display.blit(text, (panel_x, y_offset))
        y_offset += line_height + 10

        # For each agent with predictions
        for aid in sorted(self._goal_probabilities.keys()):
            goal_probs = self._goal_probabilities[aid]
            if goal_probs is None:
                continue

            # Agent header
            is_ego = aid == self._ego_id
            color = self.COLOR_EGO if is_ego else self.COLOR_TRAFFIC
            agent_label = f"Agent {aid}" + (" (EGO)" if is_ego else "")
            text = self._font_small.render(agent_label, True, color)
            self._display.blit(text, (panel_x, y_offset))
            y_offset += line_height

            # Goal probabilities
            sorted_goals = sorted(
                goal_probs.goals_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for i, ((goal, _), prob) in enumerate(sorted_goals):
                if prob < 0.01:
                    continue
                if i >= 5:  # Limit to top 5
                    break

                # Draw probability bar
                bar_width = int(prob * 100)
                bar_rect = pygame.Rect(panel_x + 10, y_offset + 2, bar_width, line_height - 4)
                bar_color = self.COLOR_GOAL_BEST if i == 0 else self.COLOR_GOAL
                pygame.draw.rect(self._display, (*bar_color, 100), bar_rect)

                # Draw label
                goal_pos = goal.center if hasattr(goal, 'center') else np.array([0, 0])
                label = f"{prob:.2f} ({goal_pos[0]:.0f},{goal_pos[1]:.0f})"
                text = self._font_small.render(label, True, (200, 200, 200))
                self._display.blit(text, (panel_x + 10, y_offset))
                y_offset += line_height

            y_offset += 5

            if y_offset > self.height - 50:
                break

    def close(self):
        """Close the visualizer and clean up."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
            logger.info("PredictionVisualizer closed")
