"""Multi-scene track regeneration callback.

CRL paper III-C-3: split parallel environments into groups, each group
gets a different track configuration regenerated each rollout.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

try:
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def assign_scene_groups(n_envs: int, n_scenes: int) -> list[list[int]]:
    """Split env indices into n_scenes groups."""
    groups: list[list[int]] = [[] for _ in range(n_scenes)]
    for i in range(n_envs):
        groups[i % n_scenes].append(i)
    return groups


if _SB3_AVAILABLE:

    class MultiSceneCallback(BaseCallback):
        """Regenerate tracks per scene group at each rollout start."""

        def __init__(self, env: Any, n_scenes: int = 10, verbose: int = 0) -> None:
            super().__init__(verbose)
            self.env = env
            self.n_scenes = n_scenes
            self.groups = assign_scene_groups(env.n_envs, n_scenes)

        def _on_step(self) -> bool:
            return True

        def _on_rollout_start(self) -> None:
            if self.env.track_generator is None:
                return
            rng = self.env.np_random
            for group in self.groups:
                track = self.env.track_generator.generate(rng)
                for idx in group:
                    self.env._tracks[idx] = track
                    if hasattr(self.env, "_splines") and self.env._splines:
                        from sim.spline import GateSpline
                        positions = np.array([g.position for g in track.gates])
                        self.env._splines[idx] = (
                            GateSpline(positions) if len(positions) >= 2 else None
                        )
