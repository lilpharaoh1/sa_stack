"""
Prediction Visualizer for SharedAutonomyAgent.

A pygame-based top-down visualization window that displays:
- Agent locations with IDs
- Predicted goals with probabilities
- Predicted trajectories/paths
- MCTS optimal plan trajectory
- Safety analysis comparing predicted vs optimal plan
- Driver messages and intervention status
- View radius circle
- Failure zones

Updated in real-time as predictions are made.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Set SDL environment before importing pygame
if os.environ.get('DISPLAY') is None and os.environ.get('SDL_VIDEODRIVER') is None:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

import pygame
from pygame import gfxdraw

from igp2.core.agentstate import AgentState
from igp2.core.goal import Goal
from igp2.core.trajectory import VelocityTrajectory
from igp2.recognition.goalprobabilities import GoalsProbabilities

logger = logging.getLogger(__name__)


class PredictionVisualizer:
    """Top-down visualization of agent predictions from SharedAutonomyAgent.

    This visualizer shows a 2D bird's-eye view of:
    - All agent positions (colored by type)
    - View radius around ego agent
    - Possible goals with probability labels
    - Predicted trajectories to goals
    - MCTS optimal plan trajectory
    - Safety analysis panel comparing predicted vs optimal
    - Driver messages and intervention status
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
    COLOR_MCTS_TRAJECTORY = (255, 100, 255)  # Magenta for MCTS plan
    COLOR_VIEW_RADIUS = (100, 100, 150)
    COLOR_FAILURE_ZONE = (255, 50, 50)
    COLOR_TEXT = (255, 255, 255)
    COLOR_GRID = (60, 60, 60)
    COLOR_WARNING = (255, 200, 50)
    COLOR_CRITICAL = (255, 50, 50)
    COLOR_SAFE = (50, 200, 100)
    COLOR_INTERVENTION = (255, 150, 50)

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
        self.center = center
        self._follow_ego = center is None

        # Pygame state
        self._display: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._font: Optional[pygame.font.Font] = None
        self._font_small: Optional[pygame.font.Font] = None
        self._font_large: Optional[pygame.font.Font] = None
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

        # MCTS and safety data
        self._mcts_trajectory: Optional[VelocityTrajectory] = None
        self._mcts_plan: List[Any] = []  # List of MCTSAction (macro-actions)
        self._mcts_maneuvers: List[List[Any]] = []  # Maneuvers per MCTS macro-action
        self._predicted_plan: List[Any] = []  # List of MacroAction from goal recognition
        self._predicted_maneuvers: List[List[Any]] = []  # Maneuvers per predicted macro-action
        self._predicted_trajectory: Optional[VelocityTrajectory] = None
        self._safety_analysis: Optional[Any] = None
        self._driver_message: Optional[Any] = None
        self._intervention_active: bool = False

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
            self._font_large = pygame.font.SysFont('monospace', 18, bold=True)

            self._initialized = True
            logger.info(f"PredictionVisualizer initialized: {self.width}x{self.height}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PredictionVisualizer: {e}")
            return False

    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        if self.center is None:
            center = np.array([0.0, 0.0])
        else:
            center = np.array(self.center)

        rel_pos = world_pos - center
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
               timestep: int = 0,
               mcts_trajectory: VelocityTrajectory = None,
               mcts_plan: List[Any] = None,
               mcts_maneuvers: List[List[Any]] = None,
               predicted_plan: List[Any] = None,
               predicted_maneuvers: List[List[Any]] = None,
               predicted_trajectory: VelocityTrajectory = None,
               safety_analysis: Any = None,
               driver_message: Any = None,
               intervention_active: bool = False) -> bool:
        """Update the visualization with new data.

        Args:
            frame: Current frame with agent states
            ego_id: ID of the ego agent
            goal_probabilities: Goal probabilities for each agent
            possible_goals: List of possible goals
            view_radius: View radius of ego agent
            failure_zones: List of failure zone dicts
            timestep: Current simulation timestep
            mcts_trajectory: MCTS optimal plan trajectory
            mcts_plan: List of MCTS macro actions
            predicted_plan: List of predicted macro actions from goal recognition
            predicted_trajectory: Predicted trajectory from goal recognition
            safety_analysis: SafetyAnalysis object from SharedAutonomyAgent
            driver_message: DriverMessage object from SharedAutonomyAgent
            intervention_active: Whether an intervention is currently active

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
        self._mcts_plan = mcts_plan or []
        self._mcts_maneuvers = mcts_maneuvers or []
        self._mcts_trajectory = mcts_trajectory
        self._predicted_plan = predicted_plan or []
        self._predicted_trajectory = predicted_trajectory
        self._predicted_maneuvers = predicted_maneuvers or []
        self._safety_analysis = safety_analysis
        self._driver_message = driver_message
        self._intervention_active = intervention_active

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

        # Draw MCTS trajectory
        self._draw_mcts_trajectory()

        # Draw agents
        self._draw_agents()

        # Draw HUD
        self._draw_hud()

        # Draw driver message banner
        self._draw_driver_message()

    def _draw_grid(self):
        """Draw a reference grid."""
        grid_spacing = 20

        if self.center is None:
            return

        half_width = (self.width / 2) / self.scale
        half_height = (self.height / 2) / self.scale

        min_x = int((self.center[0] - half_width) / grid_spacing) * grid_spacing
        max_x = int((self.center[0] + half_width) / grid_spacing + 1) * grid_spacing
        min_y = int((self.center[1] - half_height) / grid_spacing) * grid_spacing
        max_y = int((self.center[1] + half_height) / grid_spacing + 1) * grid_spacing

        for x in range(int(min_x), int(max_x) + 1, grid_spacing):
            start = self.world_to_screen(np.array([x, min_y]))
            end = self.world_to_screen(np.array([x, max_y]))
            pygame.draw.line(self._display, self.COLOR_GRID, start, end, 1)

        for y in range(int(min_y), int(max_y) + 1, grid_spacing):
            start = self.world_to_screen(np.array([min_x, y]))
            end = self.world_to_screen(np.array([max_x, y]))
            pygame.draw.line(self._display, self.COLOR_GRID, start, end, 1)

    def _draw_view_radius(self):
        """Draw the ego's view radius circle."""
        ego_state = self._frame[self._ego_id]
        center = self.world_to_screen(ego_state.position)
        radius = int(self._view_radius * self.scale)

        s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_VIEW_RADIUS, 30), (radius, radius), radius)
        self._display.blit(s, (center[0] - radius, center[1] - radius))

        pygame.draw.circle(self._display, self.COLOR_VIEW_RADIUS, center, radius, 2)

    def _draw_failure_zones(self):
        """Draw failure zones with time-based highlighting."""
        for fz in self._failure_zones:
            box = fz["box"]
            frames = fz["frames"]

            is_active = frames["start"] <= self._current_timestep <= frames["end"]
            corners = box.boundary
            screen_corners = [self.world_to_screen(np.array(c)) for c in corners]

            if is_active:
                s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.polygon(s, (*self.COLOR_FAILURE_ZONE, 50), screen_corners)
                self._display.blit(s, (0, 0))
                pygame.draw.polygon(self._display, self.COLOR_FAILURE_ZONE, screen_corners, 3)
            else:
                color = tuple(c // 2 for c in self.COLOR_FAILURE_ZONE)
                pygame.draw.polygon(self._display, color, screen_corners, 1)

            center = self.world_to_screen(box.center)
            label = f"{frames['start']}-{frames['end']}"
            text = self._font_small.render(label, True, self.COLOR_FAILURE_ZONE if is_active else color)
            self._display.blit(text, (center[0] - text.get_width() // 2, center[1]))

    def _draw_goals(self):
        """Draw possible goals with probability labels."""
        if not self._possible_goals:
            return

        ego_probs = self._goal_probabilities.get(self._ego_id)

        best_goal = None
        best_prob = 0.0
        if ego_probs:
            for (goal, _), prob in ego_probs.goals_probabilities.items():
                if prob > best_prob:
                    best_prob = prob
                    best_goal = goal

        for goal in self._possible_goals:
            pos = self.world_to_screen(goal.center)

            is_best = goal == best_goal
            color = self.COLOR_GOAL_BEST if is_best else self.COLOR_GOAL

            size = 8 if is_best else 6
            points = [
                (pos[0], pos[1] - size),
                (pos[0] + size, pos[1]),
                (pos[0], pos[1] + size),
                (pos[0] - size, pos[1])
            ]
            pygame.draw.polygon(self._display, color, points)
            pygame.draw.polygon(self._display, self.COLOR_TEXT, points, 1)

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

            for (goal, _), trajs in goal_probs.all_trajectories.items():
                prob = goal_probs.goals_probabilities.get((goal, None), 0.0)
                if prob < 0.01:
                    continue

                for traj in trajs[:2]:
                    if traj is None:
                        continue

                    path = traj.path
                    if len(path) < 2:
                        continue

                    alpha = int(255 * min(1.0, prob * 2))
                    traj_color = (*color[:3], alpha)

                    screen_points = [self.world_to_screen(p) for p in path[::3]]
                    if len(screen_points) >= 2:
                        s = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                        pygame.draw.lines(s, traj_color, False, screen_points, 2)
                        self._display.blit(s, (0, 0))

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

    def _draw_mcts_trajectory(self):
        """Draw the MCTS optimal plan trajectory."""
        if self._mcts_trajectory is None:
            return

        try:
            path = self._mcts_trajectory.path
            if len(path) < 2:
                return

            screen_points = [self.world_to_screen(p) for p in path[::2]]
            if len(screen_points) >= 2:
                # Draw thicker line for MCTS trajectory
                pygame.draw.lines(self._display, self.COLOR_MCTS_TRAJECTORY, False, screen_points, 3)

                # Draw dashed effect
                for i in range(0, len(screen_points) - 1, 2):
                    if i + 1 < len(screen_points):
                        pygame.draw.line(self._display, (255, 255, 255),
                                       screen_points[i], screen_points[i + 1], 1)
        except Exception:
            pass

    def _draw_agents(self):
        """Draw all agents as triangles indicating heading."""
        for aid, state in self._frame.items():
            pos = self.world_to_screen(state.position)
            heading = state.heading

            is_ego = aid == self._ego_id
            color = self.COLOR_EGO if is_ego else self.COLOR_TRAFFIC

            # Highlight ego if intervention is active
            if is_ego and self._intervention_active:
                color = self.COLOR_INTERVENTION

            size = 12 if is_ego else 8

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

            label = f"{aid}"
            text = self._font_small.render(label, True, color)
            self._display.blit(text, (pos[0] + 12, pos[1] - 6))

            speed_label = f"{state.speed:.1f}m/s"
            speed_text = self._font_small.render(speed_label, True, (180, 180, 180))
            self._display.blit(speed_text, (pos[0] + 12, pos[1] + 6))

    def _draw_driver_message(self):
        """Draw driver message banner at top of screen."""
        if self._driver_message is None:
            return

        # Choose color based on severity
        if hasattr(self._driver_message, 'severity'):
            if self._driver_message.severity == "critical":
                bg_color = self.COLOR_CRITICAL
            elif self._driver_message.severity == "warning":
                bg_color = self.COLOR_WARNING
            else:
                bg_color = self.COLOR_SAFE
        else:
            bg_color = self.COLOR_WARNING

        # Draw banner
        banner_height = 60
        y_start = 0 if not self._in_failure_zone else 35

        banner_rect = pygame.Rect(0, y_start, self.width, banner_height)
        pygame.draw.rect(self._display, bg_color, banner_rect)

        # Draw message text
        if hasattr(self._driver_message, 'text'):
            text = self._font_large.render(self._driver_message.text, True, self.COLOR_TEXT)
            self._display.blit(text, (self.width // 2 - text.get_width() // 2, y_start + 8))

        # Draw action hint
        if hasattr(self._driver_message, 'action_hint') and self._driver_message.action_hint:
            hint = self._font_small.render(self._driver_message.action_hint, True, self.COLOR_TEXT)
            self._display.blit(hint, (self.width // 2 - hint.get_width() // 2, y_start + 35))

    def _draw_hud(self):
        """Draw heads-up display with info."""
        y_offset = 10

        # Adjust for banners
        if self._driver_message is not None:
            y_offset = 70 if not self._in_failure_zone else 105
        elif self._in_failure_zone:
            banner_rect = pygame.Rect(0, 0, self.width, 35)
            pygame.draw.rect(self._display, self.COLOR_FAILURE_ZONE, banner_rect)
            status_text = self._font.render("FAILURE ZONE ACTIVE", True, self.COLOR_TEXT)
            self._display.blit(status_text, (self.width // 2 - status_text.get_width() // 2, 8))
            y_offset = 45

        line_height = 18

        # Title
        text = self._font.render(f"SOA Visualizer - Frame {self._current_timestep}", True, self.COLOR_TEXT)
        self._display.blit(text, (10, y_offset))
        y_offset += line_height + 5

        # Controls
        controls = ["+/- : Zoom", "F : Follow", "ESC : Close"]
        for ctrl in controls:
            text = self._font_small.render(ctrl, True, (150, 150, 150))
            self._display.blit(text, (10, y_offset))
            y_offset += line_height - 4

        y_offset += 10

        # Scale info
        text = self._font_small.render(f"Scale: {self.scale:.1f} px/m", True, (150, 150, 150))
        self._display.blit(text, (10, y_offset))
        y_offset += line_height

        # Intervention status
        if self._intervention_active:
            text = self._font_small.render("INTERVENTION ACTIVE", True, self.COLOR_INTERVENTION)
        else:
            text = self._font_small.render("No intervention", True, self.COLOR_SAFE)
        self._display.blit(text, (10, y_offset))
        y_offset += line_height + 10

        # Right side panels
        self._draw_goal_panel()
        self._draw_plan_comparison_panel()

    def _draw_goal_panel(self):
        """Draw goal probabilities panel on the right side."""
        panel_width = 200
        panel_x = self.width - panel_width - 10
        y_offset = 10
        line_height = 16

        # Panel background
        panel_rect = pygame.Rect(panel_x - 5, 5, panel_width + 10, 200)
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

            is_ego = aid == self._ego_id
            color = self.COLOR_EGO if is_ego else self.COLOR_TRAFFIC
            agent_label = f"Agent {aid}" + (" (EGO)" if is_ego else "")
            text = self._font_small.render(agent_label, True, color)
            self._display.blit(text, (panel_x, y_offset))
            y_offset += line_height

            sorted_goals = sorted(
                goal_probs.goals_probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for i, ((goal, _), prob) in enumerate(sorted_goals):
                if prob < 0.01 or i >= 3:
                    break

                bar_width = int(prob * 100)
                bar_rect = pygame.Rect(panel_x + 10, y_offset + 2, bar_width, line_height - 4)
                bar_color = self.COLOR_GOAL_BEST if i == 0 else self.COLOR_GOAL
                pygame.draw.rect(self._display, (*bar_color, 100), bar_rect)

                goal_pos = goal.center if hasattr(goal, 'center') else np.array([0, 0])
                label = f"{prob:.2f}"
                text = self._font_small.render(label, True, (200, 200, 200))
                self._display.blit(text, (panel_x + 10, y_offset))
                y_offset += line_height

            y_offset += 5

            if y_offset > 190:
                break

    def _draw_plan_comparison_panel(self):
        """Draw panel comparing predicted plan vs MCTS optimal plan.

        Shows both macro-actions (Continue, Exit, ChangeLane, StopMA) and their
        constituent maneuvers (FollowLane, Turn, GiveWay, Stop, etc.).
        """
        panel_width = 240
        panel_x = self.width - panel_width - 10
        y_offset = 220
        line_height = 14

        # Panel background
        panel_rect = pygame.Rect(panel_x - 5, y_offset - 5, panel_width + 10, self.height - y_offset - 5)
        s = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        s.fill((30, 30, 30, 180))
        self._display.blit(s, (panel_rect.x, panel_rect.y))

        # Title
        text = self._font.render("Plan Comparison", True, self.COLOR_TEXT)
        self._display.blit(text, (panel_x, y_offset))
        y_offset += line_height + 6

        # Safety status
        if self._safety_analysis is not None and hasattr(self._safety_analysis, 'is_safe'):
            if self._safety_analysis.is_safe:
                status_text = "SAFE"
                status_color = self.COLOR_SAFE
            else:
                status_text = "UNSAFE"
                status_color = self.COLOR_CRITICAL
            text = self._font.render(status_text, True, status_color)
            self._display.blit(text, (panel_x, y_offset))

            if hasattr(self._safety_analysis, 'risk_score'):
                risk = self._safety_analysis.risk_score
                risk_text = self._font_small.render(f"Risk: {risk:.2f}", True, (180, 180, 180))
                self._display.blit(risk_text, (panel_x + 80, y_offset + 2))
            y_offset += line_height + 6

        # --- Predicted Plan Section (with maneuvers) ---
        text = self._font.render("Predicted Plan:", True, self.COLOR_TRAJECTORY_OPT)
        self._display.blit(text, (panel_x, y_offset))
        y_offset += line_height + 2

        if self._predicted_plan:
            for i, ma in enumerate(self._predicted_plan[:4]):  # Limit to 4 macro-actions
                ma_name = self._get_macro_action_name(ma)
                label = f"{i+1}. {ma_name}"
                text = self._font_small.render(label, True, self.COLOR_TRAJECTORY_OPT)
                self._display.blit(text, (panel_x + 3, y_offset))
                y_offset += line_height - 2

                # Show maneuvers for this macro-action
                maneuvers = self._get_maneuvers_from_ma(ma, i)
                if maneuvers:
                    maneuver_names = [type(m).__name__ for m in maneuvers[:3]]  # Limit to 3
                    maneuver_str = " > ".join(maneuver_names)
                    if len(maneuver_str) > 30:
                        maneuver_str = maneuver_str[:27] + "..."
                    man_text = self._font_small.render(f"   [{maneuver_str}]", True, (120, 160, 120))
                    self._display.blit(man_text, (panel_x + 3, y_offset))
                    y_offset += line_height - 2
        else:
            text = self._font_small.render("  (no prediction)", True, (120, 120, 120))
            self._display.blit(text, (panel_x, y_offset))
            y_offset += line_height

        y_offset += 6

        # --- MCTS Plan Section ---
        text = self._font.render("MCTS Plan:", True, self.COLOR_MCTS_TRAJECTORY)
        self._display.blit(text, (panel_x, y_offset))
        y_offset += line_height + 2

        if self._mcts_plan:
            for i, ma in enumerate(self._mcts_plan[:4]):  # Limit to 4 macro-actions
                ma_name = self._get_macro_action_name(ma)

                # Check if this action type is missing from predicted
                is_missing = False
                if self._safety_analysis and hasattr(self._safety_analysis, 'missing_actions'):
                    if ma_name in self._safety_analysis.missing_actions:
                        is_missing = True

                color = self.COLOR_CRITICAL if is_missing else self.COLOR_MCTS_TRAJECTORY
                label = f"{i+1}. {ma_name}" + (" !" if is_missing else "")
                text = self._font_small.render(label, True, color)
                self._display.blit(text, (panel_x + 3, y_offset))
                y_offset += line_height - 2

                # Show maneuvers for this MCTS macro-action (if available)
                mcts_maneuvers = self._get_mcts_maneuvers(i)
                if mcts_maneuvers:
                    maneuver_names = [type(m).__name__ for m in mcts_maneuvers[:3]]
                    maneuver_str = " > ".join(maneuver_names)
                    if len(maneuver_str) > 30:
                        maneuver_str = maneuver_str[:27] + "..."
                    man_text = self._font_small.render(f"   [{maneuver_str}]", True, (160, 120, 180))
                    self._display.blit(man_text, (panel_x + 3, y_offset))
                    y_offset += line_height - 2
        else:
            text = self._font_small.render("  (no MCTS plan)", True, (120, 120, 120))
            self._display.blit(text, (panel_x, y_offset))
            y_offset += line_height

        y_offset += 6

        # Missing maneuvers summary (for safety-critical maneuvers)
        if self._safety_analysis and hasattr(self._safety_analysis, 'missing_maneuvers') and self._safety_analysis.missing_maneuvers:
            text = self._font_small.render("Missing maneuvers:", True, self.COLOR_CRITICAL)
            self._display.blit(text, (panel_x, y_offset))
            y_offset += line_height
            for maneuver in self._safety_analysis.missing_maneuvers[:3]:
                text = self._font_small.render(f"  ! {maneuver}", True, self.COLOR_CRITICAL)
                self._display.blit(text, (panel_x, y_offset))
                y_offset += line_height - 2

    def _get_maneuvers_from_ma(self, ma, index: int = 0) -> List[Any]:
        """Get maneuvers from a macro-action or from stored predicted_maneuvers."""
        # First try to get from the stored predicted_maneuvers list
        if self._predicted_maneuvers and index < len(self._predicted_maneuvers):
            return self._predicted_maneuvers[index]

        # Otherwise try to get directly from the macro-action
        if hasattr(ma, '_maneuvers') and ma._maneuvers:
            return ma._maneuvers

        return []

    def _get_mcts_maneuvers(self, index: int = 0) -> List[Any]:
        """Get maneuvers from the MCTS plan at the given index."""
        if self._mcts_maneuvers and index < len(self._mcts_maneuvers):
            return self._mcts_maneuvers[index]
        return []

    def _get_macro_action_name(self, ma) -> str:
        """Get the name of a macro action, handling both MacroAction and MCTSAction."""
        # MCTSAction has macro_action_type attribute
        if hasattr(ma, 'macro_action_type'):
            return ma.macro_action_type.__name__
        # Regular MacroAction - use type name directly
        return type(ma).__name__

    def _get_macro_action_details(self, ma) -> str:
        """Extract readable details from a macro action or MCTSAction."""
        try:
            ma_type = self._get_macro_action_name(ma)

            # For MCTSAction, try to get details from ma_args
            if hasattr(ma, 'ma_args'):
                args = ma.ma_args
                if ma_type == "Exit":
                    turn = args.get('turn')
                    if turn:
                        return f"turn={turn}"
                elif ma_type in ("ChangeLane", "SwitchLane"):
                    left = args.get('left')
                    if left is not None:
                        direction = "left" if left else "right"
                        return f"dir={direction}"
                elif ma_type == "Stop":
                    dur = args.get('stop_duration')
                    if dur is not None:
                        return f"dur={dur:.1f}s"
                elif ma_type == "GiveWay":
                    return "yield to traffic"
                elif ma_type == "Continue":
                    return ""
                return ""

            # For actual MacroAction instances
            if ma_type == "Continue":
                return ""
            elif ma_type == "Exit":
                if hasattr(ma, 'turn'):
                    return f"turn={ma.turn}"
            elif ma_type == "ChangeLane":
                if hasattr(ma, 'left'):
                    direction = "left" if ma.left else "right"
                    return f"dir={direction}"
            elif ma_type == "GiveWay":
                return "yield to traffic"
            elif ma_type == "Stop":
                if hasattr(ma, 'stop_duration'):
                    return f"dur={ma.stop_duration:.1f}s"
            elif ma_type == "SwitchLane":
                if hasattr(ma, 'left'):
                    direction = "left" if ma.left else "right"
                    return f"dir={direction}"

            # Try to get target info
            if hasattr(ma, 'target_sequence') and ma.target_sequence:
                first_lane = ma.target_sequence[0]
                if hasattr(first_lane, 'id'):
                    return f"lane={first_lane.id}"
        except:
            pass
        return ""

    def close(self):
        """Close the visualizer and clean up."""
        if self._initialized:
            pygame.quit()
            self._initialized = False
            logger.info("PredictionVisualizer closed")
