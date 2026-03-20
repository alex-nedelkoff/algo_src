"""Load and parse .npz trajectory files (schema_version=2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrajectoryData:
    """Parsed trajectory from a .npz file."""

    schema_version: int
    positions: NDArray[np.float64]       # (T, 3)
    quaternions: NDArray[np.float64]     # (T, 4)
    velocities: NDArray[np.float64]     # (T, 3)
    body_rates: NDArray[np.float64]     # (T, 3)
    motor_rpms: NDArray[np.float64]     # (T, 4)
    actions: NDArray[np.float64]        # (T, 4)
    rewards: NDArray[np.float64]        # (T,)
    reward_components: NDArray[np.float64]  # (T, n_components)
    reward_component_names: list[str]       # (n_components,)
    gate_events: NDArray[np.int64]      # (N_events, 2)
    gate_positions: NDArray[np.float64] # (n_gates, 3)
    gate_orientations: NDArray[np.float64]  # (n_gates, 4)
    gate_half_extents: NDArray[np.float64]  # (n_gates, 2)
    dt: float

    @property
    def n_timesteps(self) -> int:
        return self.positions.shape[0]

    @property
    def n_gates(self) -> int:
        return self.gate_positions.shape[0]


def load_trajectory(path: Path | str) -> TrajectoryData:
    """Load a .npz trajectory file and return parsed TrajectoryData."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    return TrajectoryData(
        schema_version=int(data["schema_version"]),
        positions=data["positions"].astype(np.float64),
        quaternions=data["quaternions"].astype(np.float64),
        velocities=data["velocities"].astype(np.float64),
        body_rates=data["body_rates"].astype(np.float64),
        motor_rpms=data["motor_rpms"].astype(np.float64),
        actions=data["actions"].astype(np.float64),
        rewards=data["rewards"].astype(np.float64),
        reward_components=data["reward_components"].astype(np.float64),
        reward_component_names=list(data["reward_component_names"]),
        gate_events=data["gate_events"].astype(np.int64),
        gate_positions=data["gate_positions"].astype(np.float64),
        gate_orientations=data["gate_orientations"].astype(np.float64),
        gate_half_extents=data["gate_half_extents"].astype(np.float64),
        dt=float(data["dt"]),
    )
