"""Gate passage quality and behavioral sanity constraint checks."""

from __future__ import annotations

import numpy as np

from autoresearch.analysis.trajectory import TrajectoryData
from autoresearch.constraints.physics import ConstraintCheck


def check_gate_passage(traj: TrajectoryData, max_offset_ratio: float = 0.8) -> ConstraintCheck:
    if traj.gate_events.shape[0] == 0:
        return ConstraintCheck(True, "No gate events to validate")

    gate_indices = traj.gate_events[:, 1]
    for i in range(1, len(gate_indices)):
        expected_next = (gate_indices[i - 1] + 1) % traj.n_gates
        if gate_indices[i] != expected_next:
            return ConstraintCheck(
                False,
                f"Gate skip detected: gate {gate_indices[i-1]} -> {gate_indices[i]}, expected {expected_next}",
            )

    offsets = []
    for event_idx in range(traj.gate_events.shape[0]):
        timestep = traj.gate_events[event_idx, 0]
        gate_idx = traj.gate_events[event_idx, 1]
        if timestep >= traj.n_timesteps:
            continue
        drone_pos = traj.positions[timestep]
        gate_pos = traj.gate_positions[gate_idx]
        half_ext = traj.gate_half_extents[gate_idx]
        max_half = np.max(half_ext)
        offset_dist = np.linalg.norm(drone_pos - gate_pos)
        offsets.append(offset_dist / max_half if max_half > 0 else 0.0)

    if offsets:
        mean_ratio = np.mean(offsets)
        max_ratio = np.max(offsets)
        if mean_ratio > max_offset_ratio:
            return ConstraintCheck(False, f"Gate offset too large: mean ratio {mean_ratio:.2f} > {max_offset_ratio}")
        if max_ratio > 0.95:
            return ConstraintCheck(False, f"Gate clearance too tight: max ratio {max_ratio:.2f} > 0.95")

    return ConstraintCheck(True, "Gate passage quality: all checks passed")


def check_behavioral_sanity(traj: TrajectoryData, min_avg_speed: float = 2.0, dr_active: bool = True) -> ConstraintCheck:
    speeds = np.linalg.norm(traj.velocities, axis=1)
    avg_speed = float(np.mean(speeds))
    if avg_speed < min_avg_speed:
        return ConstraintCheck(False, f"Average speed too low: {avg_speed:.2f} m/s < {min_avg_speed} m/s")

    window_steps = int(3.0 / traj.dt)
    if traj.gate_events.shape[0] > 0 and traj.n_timesteps > window_steps:
        for start in range(0, traj.n_timesteps - window_steps):
            end = start + window_steps
            displacement = np.linalg.norm(traj.positions[end] - traj.positions[start])
            if displacement < 0.1:
                return ConstraintCheck(False, f"No forward progress: displacement {displacement:.2f}m over 3s window at step {start}")

    return ConstraintCheck(True, "Behavioral sanity: all checks passed")
