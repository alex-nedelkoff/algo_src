"""Extract failure patterns and behavioral signals from trajectory data."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from autoresearch.analysis.trajectory import TrajectoryData


@dataclass
class GateAnalysis:
    gate_idx: int
    timestep: int
    passage_time: float
    approach_speed: float
    offset_distance: float


@dataclass
class TerminationAnalysis:
    final_position: tuple[float, float, float]
    final_speed: float
    total_time: float


@dataclass
class TrajectoryInsights:
    mean_speed: float
    max_speed: float
    min_speed: float
    mean_motor_utilization: float
    max_motor_utilization: float
    max_body_rate: float
    mean_body_rate: float
    gate_analyses: list[GateAnalysis] = field(default_factory=list)
    gates_passed: int = 0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    dominant_reward_component: str = ""
    termination: TerminationAnalysis | None = None

    def to_summary(self) -> str:
        lines = [
            f"Speed: mean={self.mean_speed:.1f} max={self.max_speed:.1f} m/s",
            f"Motor utilization: mean={self.mean_motor_utilization:.2f} max={self.max_motor_utilization:.2f}",
            f"Body rates: max={self.max_body_rate:.1f} rad/s",
            f"Gates passed: {self.gates_passed}",
        ]
        if self.gate_analyses:
            times = [ga.passage_time for ga in self.gate_analyses]
            speeds = [ga.approach_speed for ga in self.gate_analyses]
            lines.append(f"Gate times: {[f'{t:.2f}s' for t in times]}")
            lines.append(f"Gate approach speeds: {[f'{s:.1f}' for s in speeds]}")
        if self.reward_breakdown:
            top = sorted(self.reward_breakdown.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            lines.append(f"Top reward components: {[(k, f'{v:.2f}') for k, v in top]}")
            lines.append(f"Dominant: {self.dominant_reward_component}")
        return "\n".join(lines)


def analyze_trajectory(traj: TrajectoryData, max_rpm: float = 31470.0) -> TrajectoryInsights:
    speeds = np.linalg.norm(traj.velocities, axis=1)
    motor_norms = np.linalg.norm(traj.motor_rpms, axis=1)
    max_motor_norm = np.linalg.norm(np.full(4, max_rpm))
    utilization = motor_norms / max_motor_norm
    rate_mags = np.linalg.norm(traj.body_rates, axis=1)

    gate_analyses = []
    for i in range(traj.gate_events.shape[0]):
        ts = int(traj.gate_events[i, 0])
        gi = int(traj.gate_events[i, 1])
        if ts >= traj.n_timesteps:
            continue
        gate_analyses.append(GateAnalysis(
            gate_idx=gi, timestep=ts, passage_time=ts * traj.dt,
            approach_speed=float(speeds[ts]),
            offset_distance=float(np.linalg.norm(traj.positions[ts] - traj.gate_positions[gi])),
        ))

    reward_breakdown = {}
    if traj.reward_components.shape[1] > 0 and len(traj.reward_component_names) > 0:
        for j, name in enumerate(traj.reward_component_names):
            reward_breakdown[name] = float(np.sum(traj.reward_components[:, j]))
    dominant = max(reward_breakdown, key=lambda k: abs(reward_breakdown[k])) if reward_breakdown else ""

    final_pos = tuple(float(x) for x in traj.positions[-1])
    termination = TerminationAnalysis(
        final_position=final_pos, final_speed=float(speeds[-1]),
        total_time=traj.n_timesteps * traj.dt,
    )

    return TrajectoryInsights(
        mean_speed=float(np.mean(speeds)), max_speed=float(np.max(speeds)),
        min_speed=float(np.min(speeds)),
        mean_motor_utilization=float(np.mean(utilization)),
        max_motor_utilization=float(np.max(utilization)),
        max_body_rate=float(np.max(rate_mags)),
        mean_body_rate=float(np.mean(rate_mags)),
        gate_analyses=gate_analyses, gates_passed=len(gate_analyses),
        reward_breakdown=reward_breakdown,
        dominant_reward_component=dominant,
        termination=termination,
    )
