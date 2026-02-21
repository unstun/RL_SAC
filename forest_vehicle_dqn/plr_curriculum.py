"""Syllabus PLR curriculum wrapper for forest navigation."""
from __future__ import annotations
from typing import List, Tuple

import numpy as np

try:
    from syllabus.task_space import DiscreteTaskSpace
    from syllabus.curricula import PrioritizedLevelReplay
    import gymnasium as gym
    HAS_SYLLABUS = True
except ImportError:
    HAS_SYLLABUS = False


class ForestPLRCurriculum:
    """Maps distance levels to Syllabus PLR tasks."""

    def __init__(self, levels_m: List[float],
                 staleness_coef: float = 0.3,
                 seed: int = 0):
        assert HAS_SYLLABUS, "pip install syllabus-rl"
        self.levels_m = sorted(levels_m)
        n = len(self.levels_m)
        task_space = DiscreteTaskSpace(n)
        obs_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)
        self.plr = PrioritizedLevelReplay(
            task_space=task_space,
            observation_space=obs_space,
            num_steps=256,
            num_processes=1,
            suppress_usage_warnings=True,
        )
        self._rng = np.random.default_rng(seed)

    def sample_level(self) -> int:
        """Return next level index."""
        return int(self.plr.sample()[0])

    def level_to_dist_range(self, idx: int) -> Tuple[float, float]:
        """Convert level index to (min_dist, max_dist).

        max_dist=0.0 for the last level means 'no upper bound'.
        """
        lo = self.levels_m[idx]
        hi = (self.levels_m[idx + 1]
              if idx + 1 < len(self.levels_m) else 0.0)
        return float(lo), float(hi)

    def report(self, level_idx: int, episode_return: float,
               length: int = 1):
        """Report episode result to PLR."""
        self.plr.update_on_episode(
            episode_return=episode_return,
            length=length,
            task=[level_idx],
            progress=0.0,
        )
