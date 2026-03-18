"""Tests for curriculum learning callback."""
import numpy as np
import pytest

from training.curriculum_callback import CurriculumCallback


class TestCurriculumCallback:
    def _make_config(self):
        return {
            "enabled": True,
            "stages": [
                {
                    "timestep": 0,
                    "reward_weights": {"gate_passage": 1.5, "gate_progress": 1.0},
                    "v_max": 10.0,
                    "trigger": None,
                },
                {
                    "timestep": 1_000_000,
                    "reward_weights": {"gate_passage": 10.0, "gate_progress": 0.5},
                    "v_max": 20.0,
                    "trigger": {"metric": "racing/gate_passage_rate", "threshold": 0.8, "window": 50},
                },
                {
                    "timestep": 5_000_000,
                    "reward_weights": {"gate_passage": 30.0, "gate_progress": 0.0},
                    "v_max": 30.0,
                    "trigger": {"metric": "racing/lap_completion_rate", "threshold": 0.5, "window": 50},
                },
            ],
        }

    def test_starts_at_stage_0(self):
        cb = CurriculumCallback(self._make_config())
        assert cb.current_stage == 0

    def test_timestep_trigger(self):
        cb = CurriculumCallback(self._make_config())
        assert cb.should_advance(timestep=500_000, metrics={}) is False
        assert cb.should_advance(timestep=1_000_001, metrics={}) is True

    def test_performance_trigger_advances_early(self):
        cb = CurriculumCallback(self._make_config())
        metrics = {"racing/gate_passage_rate": 0.85}
        assert cb.should_advance(timestep=500_000, metrics=metrics) is True

    def test_get_stage_config(self):
        cb = CurriculumCallback(self._make_config())
        stage = cb.get_current_stage_config()
        assert stage["v_max"] == 10.0
        cb.advance()
        stage = cb.get_current_stage_config()
        assert stage["v_max"] == 20.0

    def test_no_advance_past_last_stage(self):
        cb = CurriculumCallback(self._make_config())
        cb.advance()
        cb.advance()
        assert cb.current_stage == 2
        assert cb.should_advance(timestep=999_999_999, metrics={}) is False
