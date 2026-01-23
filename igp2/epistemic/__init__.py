"""
Epistemic module for maneuver-level goal recognition.

This module provides the same functionality as the recognition module,
but operates at the MANEUVER level instead of the MACRO-ACTION level.

Where recognition uses:
    Goal -> A*(MacroActions) -> Maneuvers -> Trajectory

Epistemic uses:
    Goal -> A*(Maneuvers) -> Trajectory

This provides finer-grained recognition of driver intent by working
directly with primitive maneuvers like FollowLane, Turn, GiveWay, Stop, etc.
"""

from igp2.epistemic.maneuver_factory import ManeuverFactory
from igp2.epistemic.maneuver_astar import ManeuverAStar
from igp2.epistemic.maneuver_probabilities import ManeuverProbabilities
from igp2.epistemic.maneuver_recognition import ManeuverRecognition

__all__ = [
    'ManeuverFactory',
    'ManeuverAStar',
    'ManeuverProbabilities',
    'ManeuverRecognition'
]
