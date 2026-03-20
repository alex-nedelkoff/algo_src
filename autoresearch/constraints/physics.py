"""Physics plausibility constraint checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from autoresearch.analysis.trajectory import TrajectoryData

G = 9.81


@dataclass
class ConstraintCheck:
    passed: bool
    reason: str


def check_physics(
    traj: TrajectoryData,
    max_rpm: float,
    max_accel_g: float = 4.0,
) -> ConstraintCheck:
    z = traj.positions[:, 2]
    underground = np.mean(z <= 0.0)
    if underground > 0.01:
        return ConstraintCheck(False, f"Ground violation: {underground:.1%} of timesteps below ground")

    above_ceiling = np.mean(z > 10.0)
    if above_ceiling > 0.01:
        return ConstraintCheck(False, f"Ceiling violation: {above_ceiling:.1%} of timesteps above ceiling")

    max_motor = np.max(traj.motor_rpms)
    if max_motor > max_rpm * 1.05:
        return ConstraintCheck(False, f"Motor limit exceeded: {max_motor:.0f} > {max_rpm * 1.05:.0f} RPM")

    if traj.velocities.shape[0] >= 2:
        accel = np.diff(traj.velocities, axis=0) / traj.dt
        accel_mag = np.linalg.norm(accel, axis=1)
        max_accel = np.max(accel_mag)
        if max_accel > max_accel_g * G:
            return ConstraintCheck(False, f"Acceleration exceeded: {max_accel / G:.1f}g > {max_accel_g}g limit")

    quat_norms = np.linalg.norm(traj.quaternions, axis=1)
    max_deviation = np.max(np.abs(quat_norms - 1.0))
    if max_deviation > 0.01:
        return ConstraintCheck(False, f"Quaternion norm deviation: {max_deviation:.4f} > 0.01")

    return ConstraintCheck(True, "Physics plausibility: all checks passed")
