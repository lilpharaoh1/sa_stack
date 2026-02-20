"""Backward-compatible re-exports from the refactored beliefcontrol modules.

The original monolithic policy.py has been split into:
  - sample_based.py  (SampleBased)
  - frenet.py         (FrenetFrame)
  - first_stage.py    (FirstStagePlanner)
  - second_stage.py   (SecondStagePlanner)
  - two_stage_policy.py (TwoStagePolicy)

This file re-exports the old names so existing imports continue to work.
"""

from igp2.beliefcontrol.sample_based import SampleBased  # noqa: F401
from igp2.beliefcontrol.two_stage_policy import TwoStagePolicy as TwoStageOPT  # noqa: F401
