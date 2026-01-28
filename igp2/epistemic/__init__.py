"""
Epistemic module for maneuver-level goal recognition.

This module provides maneuver-level goal recognition with two approaches:

1. Direct maneuver search (ManeuverAStar, ManeuverSequenceGenerator):
   Goal -> A*/BFS(Maneuvers) -> Trajectory
   - Searches directly over maneuvers
   - Issues: maneuver applicability can be tricky

2. Macro-guided maneuver generation (MacroGuidedSequenceGenerator):
   Goal -> BFS(MacroActions) -> Expand to Maneuvers -> Trajectory
   - Uses macro-action structure to guide search
   - Expands each macro-action into its maneuvers
   - More robust: uses well-tested macro-action applicability
   - Generates variations (with/without GiveWay)

The macro-guided approach is recommended as it:
- Uses macro-action applicability (robust, well-tested)
- Provides maneuver-level granularity for comparison
- Supports generating plan variations for reasoning
"""

from igp2.epistemic.maneuver_factory import ManeuverFactory
from igp2.epistemic.maneuver_astar import ManeuverAStar
from igp2.epistemic.maneuver_probabilities import ManeuverProbabilities
from igp2.epistemic.maneuver_recognition import ManeuverRecognition
from igp2.epistemic.sequence_generator import ManeuverSequenceGenerator
from igp2.epistemic.macro_guided_generator import MacroGuidedSequenceGenerator
from igp2.epistemic.macro_guided_recognition import MacroGuidedRecognition
from igp2.epistemic.trajectory_similarity import (
    path_similarity,
    velocity_similarity,
    combined_similarity,
    trajectory_overlap_similarity
)
from igp2.epistemic.intervention_optimizer import (
    InterventionOptimizer,
    GradientInterventionOptimizer,
    InterventionResult,
    OptimizerConfig,
    compute_intervention
)
from igp2.epistemic.plot_intervention import (
    plot_intervention_result,
    plot_intervention_comparison,
    plot_velocity_colored_trajectory,
    create_intervention_animation_frames
)

__all__ = [
    'ManeuverFactory',
    'ManeuverAStar',
    'ManeuverProbabilities',
    'ManeuverRecognition',
    'ManeuverSequenceGenerator',
    'MacroGuidedSequenceGenerator',
    'MacroGuidedRecognition',
    'path_similarity',
    'velocity_similarity',
    'combined_similarity',
    'trajectory_overlap_similarity',
    'InterventionOptimizer',
    'GradientInterventionOptimizer',
    'InterventionResult',
    'OptimizerConfig',
    'compute_intervention',
    'plot_intervention_result',
    'plot_intervention_comparison',
    'plot_velocity_colored_trajectory',
    'create_intervention_animation_frames'
]
