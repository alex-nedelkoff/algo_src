"""PyTorch port of NumpyQuadDynamics for grey-box system identification.

Single-environment, single-step forward Euler integration of a quadrotor
in X-configuration. Mirrors ``sim/dynamics/numpy_quad.py`` exactly so that
when ``TorchVehicleParams`` is initialised from a ``VehicleParams`` instance,
the two implementations produce identical state trajectories given identical
inputs.

State vector (17):
    [0:3]   position (world frame)
    [3:6]   velocity (world frame)
    [6:10]  quaternion [w, x, y, z] (body-to-world)
    [10:13] angular velocity (body frame, rad/s)
    [13:17] motor speeds (rad/s)

Action: motor speed commands (4,) in rad/s, clipped to [0, max_omega].
"""
from __future__ import annotations

import math

import torch

from sim.dynamics.torch_params import TorchVehicleParams

GRAVITY = 9.81
_SQRT2_INV = 1.0 / math.sqrt(2.0)


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Quaternion [w, x, y, z] (1D, length 4) to rotation matrix (3, 3)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            torch.stack([1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)]),
            torch.stack([2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)]),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)]),
        ]
    )


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two [w, x, y, z] quaternions (1D, length 4)."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def step(
    state: torch.Tensor,
    action: torch.Tensor,
    params: TorchVehicleParams,
    dt: float,
) -> torch.Tensor:
    """Advance one quadrotor by ``dt`` seconds.

    Args:
        state: Shape (17,), float64. Layout per module docstring.
        action: Shape (4,), float64. Motor speed commands in rad/s.
        params: TorchVehicleParams (learnable scalars + max_omega buffer).
        dt: Integration timestep in seconds.

    Returns:
        Next state, shape (17,).
    """
    pos = state[0:3]
    vel = state[3:6]
    quat = state[6:10]
    omega = state[10:13]
    motor_w = state[13:17]

    # Motor command clip
    cmd_w = action.clamp(min=0.0, max=params.max_omega.item())

    # First-order motor lag
    motor_w_new = motor_w + dt * (cmd_w - motor_w) / params.tau_motor
    motor_w_new = motor_w_new.clamp(min=0.0, max=params.max_omega.item())

    # Use mid-step motor speed for force/torque computation (matches numpy_quad)
    motor_w_avg = 0.5 * (motor_w + motor_w_new)
    w_sq = motor_w_avg * motor_w_avg

    # Per-motor thrust → total body-frame thrust along +Z body
    thrust_per_motor = params.k_thrust * w_sq
    total_thrust = thrust_per_motor.sum()
    force_body = torch.stack(
        [
            torch.zeros_like(total_thrust),
            torch.zeros_like(total_thrust),
            total_thrust,
        ]
    )

    # Body torques from motor thrust differential (X-config)
    arm_k = params.arm_length * params.k_thrust * _SQRT2_INV
    tau_roll = arm_k * (-w_sq[0] + w_sq[1] + w_sq[2] - w_sq[3])
    tau_pitch = arm_k * (-w_sq[0] - w_sq[1] + w_sq[2] + w_sq[3])
    tau_yaw = params.k_torque * (w_sq[0] - w_sq[1] + w_sq[2] - w_sq[3])
    torques = torch.stack([tau_roll, tau_pitch, tau_yaw])

    # Gyroscopic torque: omega × (J ⋅ omega) with diagonal J
    J_diag = torch.stack([params.Ixx, params.Iyy, params.Izz])
    J_omega = omega * J_diag
    gyro_torque = torch.linalg.cross(omega, J_omega)

    # Angular acceleration: alpha = J^{-1} (tau - omega × J omega)
    net_torque = torques - gyro_torque
    alpha = net_torque / J_diag

    # World-frame force + linear acceleration
    R = _quat_to_rotmat(quat)
    force_world = R @ force_body
    gravity_vec = torch.tensor([0.0, 0.0, -GRAVITY], dtype=state.dtype)
    accel = force_world / params.mass + gravity_vec

    # Quaternion derivative (body rates as pure quaternion)
    omega_quat = torch.stack([torch.zeros_like(omega[0]), omega[0], omega[1], omega[2]])
    dquat = 0.5 * _quat_multiply(quat, omega_quat)

    # Forward Euler integration with quaternion renormalisation
    new_pos = pos + dt * vel
    new_vel = vel + dt * accel
    new_quat_unnorm = quat + dt * dquat
    new_quat = new_quat_unnorm / new_quat_unnorm.norm().clamp(min=1e-10)
    new_omega = omega + dt * alpha

    return torch.cat([new_pos, new_vel, new_quat, new_omega, motor_w_new])
