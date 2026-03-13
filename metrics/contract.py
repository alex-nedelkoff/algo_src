"""Typed episode info schema and trajectory provider protocol.

Every environment must populate info["episode"] with fields matching
the EpisodeMetrics TypedDict.  Consumers (GateMetricsCallback,
TrajectoryRecorderCallback, VecEnvAdapter) read only these keys.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable

import numpy as np


class EpisodeMetrics(TypedDict):
    """Mandatory episode metrics contract.

    Every environment must populate all of these fields in info["episode"]
    at the end of each episode.  The GateMetricsCallback and VecEnvAdapter
    read only these keys — no env-specific assumptions.

    Non-racing environments (e.g., HoverEnv) should set racing-specific
    fields to their zero values: gates_passed=0, laps_completed=0,
    first_gate_step=-1.
    """

    r: float
    l: int
    effective_dt: float
    gates_passed: int
    laps_completed: int
    termination: str
    success: bool
    success_criterion: str
    avg_speed: float
    first_gate_step: int
    reward_components: dict[str, float]


@runtime_checkable
class TrajectoryProvider(Protocol):
    """Interface for trajectory recording.

    Envs implement these methods so the TrajectoryRecorderCallback
    can extract per-step state without reaching into env internals.
    """

    def get_state(self, env_idx: int) -> dict[str, np.ndarray]:
        """Return current state for one environment.

        Returns dict with keys:
            position: (3,) float64
            quaternion: (4,) float64 — w, x, y, z
            velocity: (3,) float64
            body_rates: (3,) float64
            motor_rpms: (4,) float64
        """
        ...

    def get_gate_geometry(self) -> dict[str, np.ndarray]:
        """Return gate geometry for the track (shared across all envs).

        Returns dict with keys:
            positions: (n_gates, 3) float64
            orientations: (n_gates, 4) float64 — quaternions (w, x, y, z)
            half_extents: (n_gates, 2) float64 — (half_width, half_height)
        """
        ...

    def get_step_reward_components(self, env_idx: int) -> tuple[list[str], np.ndarray]:
        """Return per-step reward component breakdown.

        Returns:
            names: list of component name strings
            values: (n_components,) float64 array
        """
        ...


# ---- Runtime Validation ----

REQUIRED_EPISODE_KEYS: dict[str, tuple[type, ...]] = {
    "r": (int, float, np.floating),
    "l": (int, np.integer),
    "effective_dt": (int, float, np.floating),
    "gates_passed": (int, np.integer),
    "laps_completed": (int, np.integer),
    "termination": (str,),
    "success": (bool, np.bool_),
    "success_criterion": (str,),
    "avg_speed": (int, float, np.floating),
    "first_gate_step": (int, np.integer),
    "reward_components": (dict,),
}


class ContractViolation(Exception):
    """Raised when an environment violates the metrics contract."""

    pass


def validate_episode_metrics(episode: dict[str, Any], env_name: str) -> None:
    """Validate episode info dict satisfies the metrics contract.

    Called once on first episode completion in VecEnvAdapter.
    Uses explicit raises (not assert) so validation cannot be
    disabled with python -O.
    """
    for key, types in REQUIRED_EPISODE_KEYS.items():
        if key not in episode:
            raise ContractViolation(
                f"{env_name} missing required episode key '{key}'. "
                f"See metrics/contract.py for the full contract."
            )
        if not isinstance(episode[key], types):
            raise ContractViolation(
                f"{env_name}.{key} has type {type(episode[key])}, "
                f"expected one of {types}."
            )


_REQUIRED_STATE_KEYS = {"position", "quaternion", "velocity", "body_rates", "motor_rpms"}


def validate_trajectory_state(state: dict[str, Any], env_name: str) -> None:
    """Validate get_state() return dict has the required keys.

    Called once on first trajectory extraction, then never again.
    """
    missing = _REQUIRED_STATE_KEYS - set(state.keys())
    if missing:
        raise ContractViolation(
            f"{env_name}.get_state() missing required keys: {missing}. "
            f"See TrajectoryProvider protocol in metrics/contract.py."
        )
