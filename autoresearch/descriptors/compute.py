"""Orchestrate all behavioral descriptor computations."""

from __future__ import annotations

from dataclasses import dataclass

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.descriptors.actuator import actuator_utilization
from autoresearch.descriptors.smoothness import control_smoothness
from autoresearch.descriptors.aero_regime import aero_regime_index


@dataclass(frozen=True)
class DescriptorVector:
    """Three-axis behavioral descriptor for MAP-Elites archive placement."""

    actuator_utilization: float  # [0, 1]
    control_smoothness: float   # [0, 1]
    aero_regime: float          # [0, v_max]

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.actuator_utilization, self.control_smoothness, self.aero_regime)


def compute_descriptors(
    traj: TrajectoryData,
    max_rpm: float,
    control_freq: float,
) -> DescriptorVector:
    """Compute all three behavioral descriptors from trajectory data."""
    return DescriptorVector(
        actuator_utilization=actuator_utilization(traj.motor_rpms, max_rpm),
        control_smoothness=control_smoothness(traj.motor_rpms, max_rpm, control_freq),
        aero_regime=aero_regime_index(traj.velocities, traj.quaternions),
    )
