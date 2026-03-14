"""Tests for the metrics contract module."""
from __future__ import annotations

import numpy as np
import pytest

from metrics.contract import (
    ContractViolation,
    validate_episode_metrics,
    validate_trajectory_state,
)


class TestValidateEpisodeMetrics:
    """validate_episode_metrics should accept valid dicts and reject violations."""

    @staticmethod
    def _make_valid() -> dict:
        """Return a minimal valid episode dict."""
        return {
            "r": 1.5,
            "l": 100,
            "effective_dt": 0.01,
            "gates_passed": 3,
            "laps_completed": 0,
            "termination": "timeout",
            "success": True,
            "success_criterion": "survived_full_episode",
            "avg_speed": 5.0,
            "first_gate_step": 42,
            "reward_components": {"progress": 1.0, "crash": -0.5},
        }

    def test_valid_dict_passes(self) -> None:
        validate_episode_metrics(self._make_valid(), "TestEnv")

    def test_missing_key_raises(self) -> None:
        d = self._make_valid()
        del d["termination"]
        with pytest.raises(ContractViolation, match="missing required episode key 'termination'"):
            validate_episode_metrics(d, "TestEnv")

    def test_wrong_type_raises(self) -> None:
        d = self._make_valid()
        d["termination"] = 8  # should be str, not int
        with pytest.raises(ContractViolation, match="termination has type"):
            validate_episode_metrics(d, "TestEnv")

    def test_numpy_types_accepted(self) -> None:
        d = self._make_valid()
        d["r"] = np.float64(1.5)
        d["l"] = np.int64(100)
        d["success"] = np.bool_(True)
        d["gates_passed"] = np.int32(3)
        validate_episode_metrics(d, "TestEnv")


class TestValidateTrajectoryState:
    """validate_trajectory_state should accept valid state dicts."""

    def test_valid_state_passes(self) -> None:
        state = {
            "position": np.zeros(3),
            "quaternion": np.array([1, 0, 0, 0.0]),
            "velocity": np.zeros(3),
            "body_rates": np.zeros(3),
            "motor_rpms": np.zeros(4),
        }
        validate_trajectory_state(state, "TestEnv")

    def test_missing_key_raises(self) -> None:
        state = {
            "position": np.zeros(3),
            "quaternion": np.array([1, 0, 0, 0.0]),
        }
        with pytest.raises(ContractViolation, match="missing required keys"):
            validate_trajectory_state(state, "TestEnv")
