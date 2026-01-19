"""
Status window for displaying simulation state using pygame.
"""

import pygame
from typing import Tuple, Optional

# Initialize pygame if not already done
if not pygame.get_init():
    pygame.init()


class StatusWindow:
    """A simple pygame window for displaying status information."""

    # Default colors
    COLOR_GREEN = (50, 205, 50)    # Safe/Normal state
    COLOR_RED = (220, 50, 50)      # Failure/Alert state
    COLOR_YELLOW = (255, 200, 50)  # Warning state
    COLOR_BLUE = (50, 100, 200)    # Info state
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)

    def __init__(self,
                 width: int = 300,
                 height: int = 150,
                 title: str = "Status",
                 position: Optional[Tuple[int, int]] = None):
        """Initialize the status window.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            title: Window title
            position: Optional (x, y) position for the window
        """
        self._width = width
        self._height = height
        self._title = title

        # Set window position if specified
        if position is not None:
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{position[0]},{position[1]}"

        # Create the window
        self._screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption(title)

        # Font for text rendering
        self._font_large = pygame.font.Font(None, 48)
        self._font_small = pygame.font.Font(None, 32)

        # Current state
        self._bg_color = self.COLOR_GREEN
        self._text = "SAFE"
        self._subtext = ""

        # Draw initial state
        self._draw()

    def set_state(self,
                  color: Tuple[int, int, int],
                  text: str = "",
                  subtext: str = ""):
        """Set the window state.

        Args:
            color: Background color as RGB tuple
            text: Main text to display
            subtext: Secondary text to display below main text
        """
        self._bg_color = color
        self._text = text
        self._subtext = subtext
        self._draw()

    def set_safe(self, subtext: str = ""):
        """Set window to safe/normal state (green)."""
        self.set_state(self.COLOR_GREEN, "SAFE", subtext)

    def set_failure(self, subtext: str = ""):
        """Set window to failure/alert state (red)."""
        self.set_state(self.COLOR_RED, "FAILURE", subtext)

    def set_warning(self, subtext: str = ""):
        """Set window to warning state (yellow)."""
        self.set_state(self.COLOR_YELLOW, "WARNING", subtext)

    def set_info(self, text: str, subtext: str = ""):
        """Set window to info state (blue) with custom text."""
        self.set_state(self.COLOR_BLUE, text, subtext)

    def _draw(self):
        """Redraw the window."""
        # Fill background
        self._screen.fill(self._bg_color)

        # Draw main text (centered)
        if self._text:
            text_surface = self._font_large.render(self._text, True, self.COLOR_WHITE)
            text_rect = text_surface.get_rect(center=(self._width // 2, self._height // 2 - 15))
            self._screen.blit(text_surface, text_rect)

        # Draw subtext (centered, below main text)
        if self._subtext:
            subtext_surface = self._font_small.render(self._subtext, True, self.COLOR_WHITE)
            subtext_rect = subtext_surface.get_rect(center=(self._width // 2, self._height // 2 + 25))
            self._screen.blit(subtext_surface, subtext_rect)

        # Update display
        pygame.display.flip()

    def update(self):
        """Process pygame events to keep window responsive."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.VIDEORESIZE:
                self._width = event.w
                self._height = event.h
                self._screen = pygame.display.set_mode((self._width, self._height), pygame.RESIZABLE)
                self._draw()
        return True

    def close(self):
        """Close the window."""
        pygame.quit()

    @property
    def screen(self):
        """The pygame screen surface."""
        return self._screen
