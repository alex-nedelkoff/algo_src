"""Curriculum learning callback for SB3 PPO.

Supports both timestep-based and performance-based stage transitions.
Each stage defines reward weights, v_max, and optionally a performance
trigger that can advance the stage early.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

try:
    from stable_baselines3.common.callbacks import BaseCallback
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


class CurriculumCallback:
    """Manage curriculum stage transitions.

    Stages advance when EITHER:
    1. The timestep threshold for the next stage is reached, OR
    2. A performance metric exceeds the trigger threshold
    Whichever comes first.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = config.get("enabled", False)
        self.stages = config.get("stages", [])
        self.current_stage = 0

    def should_advance(self, timestep: int, metrics: dict[str, float]) -> bool:
        if not self.enabled:
            return False
        next_stage = self.current_stage + 1
        if next_stage >= len(self.stages):
            return False

        next_cfg = self.stages[next_stage]

        if timestep >= next_cfg.get("timestep", float("inf")):
            return True

        trigger = next_cfg.get("trigger")
        if trigger is not None:
            metric_name = trigger["metric"]
            threshold = trigger["threshold"]
            if metric_name in metrics and metrics[metric_name] >= threshold:
                return True

        return False

    def advance(self) -> None:
        if self.current_stage + 1 < len(self.stages):
            self.current_stage += 1
            log.info("Curriculum: advanced to stage %d", self.current_stage)

    def get_current_stage_config(self) -> dict[str, Any]:
        return self.stages[self.current_stage]


if _SB3_AVAILABLE:

    class CurriculumSB3Callback(BaseCallback):
        """SB3 callback wrapper for CurriculumCallback.

        On each rollout end, checks whether to advance the curriculum stage
        and applies the new stage's reward weights and v_max to the environment.
        """

        def __init__(self, curriculum: CurriculumCallback, env: Any, verbose: int = 1) -> None:
            super().__init__(verbose)
            self.curriculum = curriculum
            self.env = env
            self._applied_stage = -1

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if not self.curriculum.enabled:
                return
            if self._applied_stage < 0:
                self._apply_stage()

            metrics: dict[str, float] = {}
            if hasattr(self.logger, "name_to_value"):
                metrics = dict(self.logger.name_to_value)

            if self.curriculum.should_advance(self.num_timesteps, metrics):
                self.curriculum.advance()
                self._apply_stage()

        def _apply_stage(self) -> None:
            stage = self.curriculum.get_current_stage_config()
            self._applied_stage = self.curriculum.current_stage

            rw = stage.get("reward_weights")
            if rw is not None:
                self.env.set_reward_weights(rw)

            v_max = stage.get("v_max")
            if v_max is not None:
                self.env.set_v_max(v_max)

            # Entropy coefficient annealing
            ent_coef = stage.get("ent_coef")
            model = self.model if hasattr(self, "model") and self.model is not None else None
            if ent_coef is not None and model is not None:
                model.ent_coef = ent_coef
                log.info("Curriculum: set ent_coef=%.4f", ent_coef)

            # Progressive difficulty — env parameters
            arena_bounds = stage.get("arena_bounds")
            if arena_bounds is not None and hasattr(self.env, "set_arena_bounds"):
                self.env.set_arena_bounds(float(arena_bounds))
                log.info("Curriculum: set arena_bounds=%.1f", arena_bounds)

            gate_radius = stage.get("gate_passage_radius")
            if gate_radius is not None and hasattr(self.env, "set_gate_passage_radius"):
                self.env.set_gate_passage_radius(float(gate_radius))
                log.info("Curriculum: set gate_passage_radius=%.2f", gate_radius)

            if self.verbose:
                log.info(
                    "Curriculum stage %d applied: v_max=%s, ent_coef=%s, arena=%s, gate_r=%s",
                    self.curriculum.current_stage,
                    stage.get("v_max", "?"),
                    stage.get("ent_coef", "unchanged"),
                    stage.get("arena_bounds", "unchanged"),
                    stage.get("gate_passage_radius", "unchanged"),
                )
