"""Reward function with M16 (fastest) and M23 (safest) presets.

Implements the MAVLab MonoRace reward formulation:
    r = r_prog + r_gate - p_rate - p_offset - p_perc - p_delta_u - p_crash
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class RewardPreset:
    lambda_gate: float
    lambda_offset: float
    lambda_perc: float
    lambda_prog: float
    lambda_rate: float
    lambda_delta_u: float
    lambda_crash: float
    lambda_alive: float  # survival bonus per step
    v_max: float


PRESETS: Dict[str, RewardPreset] = {
    # Baseline: matches MAVLab open-source training reward with survival bonus.
    # Use this for initial training before switching to M16/M23 for fine-tuning.
    "baseline": RewardPreset(
        lambda_gate=10.0,
        lambda_offset=0.0,
        lambda_perc=0.0,
        lambda_prog=1.0,
        lambda_rate=0.001,
        lambda_delta_u=0.001,
        lambda_crash=10.0,
        lambda_alive=0.0,
        v_max=0.0,  # 0 = no clamping, raw distance change
    ),
    "M16": RewardPreset(
        lambda_gate=30.0,
        lambda_offset=2.0,
        lambda_perc=0.1,
        lambda_prog=0.0,
        lambda_rate=0.001,
        lambda_delta_u=0.001,
        lambda_crash=1.0,
        lambda_alive=0.0,
        v_max=30.0,
    ),
    "M23": RewardPreset(
        lambda_gate=1.5,
        lambda_offset=1.5,
        lambda_perc=0.0,
        lambda_prog=1.0,
        lambda_rate=0.001,
        lambda_delta_u=0.001,
        lambda_crash=10.0,
        lambda_alive=0.0,
        v_max=10.0,
    ),
}


def compute_reward(
    *,
    d2g_old: np.ndarray | float,
    d2g_new: np.ndarray | float,
    omega: np.ndarray,
    delta_action: np.ndarray,
    gate_passed: np.ndarray | bool,
    crashed: np.ndarray | bool,
    offset: np.ndarray | float,
    theta_cam: np.ndarray | float,
    preset: str = "M23",
    dt: float = 0.005,
) -> np.ndarray | float:
    """Compute reward following the MonoRace formulation.

    All inputs can be scalars or arrays (vectorized).

    Parameters
    ----------
    d2g_old : distance to next gate at previous step
    d2g_new : distance to next gate at current step
    omega : angular velocity vector(s), shape (..., 3)
    delta_action : change in action, shape (..., 4)
    gate_passed : whether a gate was passed this step
    crashed : whether the drone crashed this step
    offset : lateral offset when passing gate (0 if no pass)
    theta_cam : angle between camera axis and gate direction
    preset : reward preset name ("M16" or "M23")
    dt : physics timestep

    Returns
    -------
    Scalar or array of reward values.
    """
    p = PRESETS[preset]

    d2g_old = np.asarray(d2g_old, dtype=np.float64)
    d2g_new = np.asarray(d2g_new, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    delta_action = np.asarray(delta_action, dtype=np.float64)
    gate_passed = np.asarray(gate_passed, dtype=np.float64)
    crashed = np.asarray(crashed, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    theta_cam = np.asarray(theta_cam, dtype=np.float64)

    # r_prog: progress toward next gate, optionally clamped by v_max * dt
    raw_progress = d2g_old - d2g_new
    if p.v_max > 0:
        raw_progress = np.minimum(raw_progress, p.v_max * dt)
    r_prog = p.lambda_prog * raw_progress

    # r_gate: bonus for passing a gate
    r_gate = p.lambda_gate * gate_passed

    # p_rate: angular rate penalty (||omega||)
    p_rate = p.lambda_rate * np.sqrt(np.sum(omega ** 2, axis=-1))

    # p_offset: lateral offset penalty (only when gate passed)
    p_offset = p.lambda_offset * offset * gate_passed

    # p_perc: perception penalty when camera angle exceeds pi/3
    p_perc = p.lambda_perc * theta_cam * (theta_cam > np.pi / 3)

    # p_delta_u: action smoothness penalty
    p_delta_u = p.lambda_delta_u * np.sum(np.abs(delta_action), axis=-1)

    # p_crash: crash penalty
    p_crash = p.lambda_crash * crashed

    # r_alive: survival bonus (only when not crashed)
    r_alive = p.lambda_alive * (1.0 - crashed)

    reward = r_prog + r_gate + r_alive - p_rate - p_offset - p_perc - p_delta_u - p_crash

    return reward


# Component names for structured logging (matches compute_reward_components column order)
MAVLAB_REWARD_COMPONENT_NAMES = [
    "progress", "gate", "alive", "rate", "offset", "perc", "delta_u", "crash",
]


def compute_reward_components(
    *,
    d2g_old: np.ndarray,
    d2g_new: np.ndarray,
    omega: np.ndarray,
    delta_action: np.ndarray,
    gate_passed: np.ndarray,
    crashed: np.ndarray,
    offset: np.ndarray,
    theta_cam: np.ndarray,
    preset: str = "M23",
    dt: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reward and per-component breakdown.

    Returns
    -------
    rewards : (n_envs,) total reward
    components : (n_envs, 8) per-component values in MAVLAB_REWARD_COMPONENT_NAMES order
    """
    p = PRESETS[preset]

    d2g_old = np.asarray(d2g_old, dtype=np.float64)
    d2g_new = np.asarray(d2g_new, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    delta_action = np.asarray(delta_action, dtype=np.float64)
    gate_passed = np.asarray(gate_passed, dtype=np.float64)
    crashed = np.asarray(crashed, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    theta_cam = np.asarray(theta_cam, dtype=np.float64)

    raw_progress = d2g_old - d2g_new
    if p.v_max > 0:
        raw_progress = np.minimum(raw_progress, p.v_max * dt)

    r_prog = p.lambda_prog * raw_progress
    r_gate = p.lambda_gate * gate_passed
    r_alive = p.lambda_alive * (1.0 - crashed)
    p_rate = p.lambda_rate * np.sqrt(np.sum(omega ** 2, axis=-1))
    p_offset = p.lambda_offset * offset * gate_passed
    p_perc = p.lambda_perc * theta_cam * (theta_cam > np.pi / 3)
    p_delta_u = p.lambda_delta_u * np.sum(np.abs(delta_action), axis=-1)
    p_crash = p.lambda_crash * crashed

    reward = r_prog + r_gate + r_alive - p_rate - p_offset - p_perc - p_delta_u - p_crash

    n = d2g_old.shape[0] if d2g_old.ndim > 0 else 1
    components = np.column_stack([
        r_prog, r_gate, r_alive,
        -p_rate, -p_offset, -p_perc, -p_delta_u, -p_crash,
    ]).reshape(n, 8)

    return reward, components
