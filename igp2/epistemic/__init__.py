"""
Epistemic module for maneuver-level goal recognition.

This module provides the same functionality as the recognition module,
but operates at the MANEUVER level instead of the MACRO-ACTION level.

Where recognition uses:
    Goal -> A*(MacroActions) -> Maneuvers -> Trajectory

Epistemic uses:
    Goal -> A*(Maneuvers) -> Trajectory
    or
    Goal -> BFS(Maneuvers) -> Trajectory  (similarity mode)

This provides finer-grained recognition of driver intent by working
directly with primitive maneuvers like FollowLane, Turn, GiveWay, Stop, etc.

Recognition Modes:
- "cost": Original A*-based approach with cost comparison (time-optimal)
- "similarity": BFS-based approach with trajectory similarity (interpretable)
"""

from igp2.epistemic.maneuver_factory import ManeuverFactory
from igp2.epistemic.maneuver_astar import ManeuverAStar
from igp2.epistemic.maneuver_probabilities import ManeuverProbabilities
from igp2.epistemic.maneuver_recognition import ManeuverRecognition
from igp2.epistemic.sequence_generator import ManeuverSequenceGenerator
from igp2.epistemic.trajectory_similarity import (
    path_similarity,
    velocity_similarity,
    combined_similarity,
    trajectory_overlap_similarity
)

__all__ = [
    'ManeuverFactory',
    'ManeuverAStar',
    'ManeuverProbabilities',
    'ManeuverRecognition',
    'ManeuverSequenceGenerator',
    'path_similarity',
    'velocity_similarity',
    'combined_similarity',
    'trajectory_overlap_similarity'
]
