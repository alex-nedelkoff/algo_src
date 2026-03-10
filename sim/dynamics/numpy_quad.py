"""Vectorized NumPy quadrotor dynamics.

Implements forward Euler integration for N parallel quadrotor environments.
State vector per environment: 17 elements
    [pos(3), vel(3), quat(4), omega(3), motor_speed(4)]

Quaternion convention: [w, x, y, z] (scalar first, Hamilton).
Motor layout: X-configuration
    Motor 0: front-right (CW)
    Motor 1: rear-right  (CCW)
    Motor 2: rear-left   (CW)
    Motor 3: front-left  (CCW)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sim.dynamics.params import VehicleParams

# Gravity constant (m/s^2)
GRAVITY = 9.81

# State vector indices
POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)
MOTOR = slice(13, 17)

# Motor spin directions: +1 for CW, -1 for CCW
# Motors: FR(CW), RR(CCW), RL(CW), FL(CCW)
MOTOR_DIRS = np.array([1.0, -1.0, 1.0, -1.0])

# Arm angles for X-config (45, -45, -135, 135 degrees from forward/x-axis)
# Used for computing roll and pitch torques
_SQRT2_INV = 1.0 / np.sqrt(2.0)


def quat_to_rotmat_batch(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert quaternion [w, x, y, z] to rotation matrix (body-to-world).

    Args:
        q: Quaternions of shape (N, 4).

    Returns:
        Rotation matrices of shape (N, 3, 3).
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Pre-compute products
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    n = q.shape[0]
    R = np.empty((n, 3, 3), dtype=np.float64)

    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)

    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)

    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)

    return R


def quat_multiply_batch(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiply two quaternion arrays element-wise.

    Args:
        q1: First quaternions (N, 4) in [w, x, y, z].
        q2: Second quaternions (N, 4) in [w, x, y, z].

    Returns:
        Product quaternions (N, 4).
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    result = np.empty_like(q1)
    result[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    result[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    result[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    result[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return result


class NumpyQuadDynamics:
    """Vectorized quadrotor dynamics using NumPy.

    Simulates N parallel quadrotor environments using batched array operations.
    Uses forward Euler integration.

    State layout per env (17 elements):
        [0:3]   position (world frame)
        [3:6]   velocity (world frame)
        [6:10]  quaternion [w, x, y, z] (body-to-world)
        [10:13] angular velocity (body frame, rad/s)
        [13:17] motor speeds (rad/s)
    """

    def __init__(self, params: VehicleParams | None = None, dt: float = 0.01) -> None:
        """Initialize dynamics with vehicle parameters.

        Args:
            params: Vehicle parameters. Uses CrazyFlie 2.1 defaults if None.
            dt: Default integration timestep in seconds.
        """
        self.params = params if params is not None else VehicleParams()
        self.dt = dt

        # Pre-compute inverse inertia for efficiency
        self._J = self.params.inertia
        self._J_inv = np.linalg.inv(self._J)
        self._J_diag = np.diag(self._J)  # [Jxx, Jyy, Jzz]

    def reset(self, n_envs: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        """Reset all environments to hover initial condition.

        Args:
            n_envs: Number of parallel environments.
            rng: Optional random generator for randomized initial states.

        Returns:
            Initial states array of shape (n_envs, 17).
        """
        states = np.zeros((n_envs, 17), dtype=np.float64)

        # Position: start at z=1.0 (1 meter above ground)
        states[:, 2] = 1.0

        # Quaternion: identity [w=1, x=0, y=0, z=0]
        states[:, 6] = 1.0

        # Motor speeds: hover equilibrium (T_per_motor = mg/4)
        hover_omega = np.sqrt(
            self.params.mass * GRAVITY / (4.0 * self.params.k_thrust)
        )
        states[:, 13:17] = hover_omega

        return states

    def step(
        self,
        states: NDArray[np.float64],
        actions: NDArray[np.float64],
        dt: float | None = None,
    ) -> NDArray[np.float64]:
        """Advance all environments by one timestep.

        Args:
            states: Current states (N, 17).
            actions: Motor speed commands in rad/s (N, 4). Clipped to [0, max_omega].
            dt: Timestep override. Uses self.dt if None.

        Returns:
            Next states (N, 17).
        """
        if dt is None:
            dt = self.dt

        p = self.params
        n = states.shape[0]

        # Extract state components
        pos = states[:, POS]          # (N, 3)
        vel = states[:, VEL]          # (N, 3)
        quat = states[:, QUAT]        # (N, 4)
        omega = states[:, OMEGA]      # (N, 3)
        motor_w = states[:, MOTOR]    # (N, 4)

        # Clip actions to valid range
        max_w = p.max_omega
        cmd_w = np.clip(actions, 0.0, max_w)  # (N, 4)

        # --- Motor dynamics: first-order lag ---
        # dw/dt = (cmd - w) / tau
        motor_w_new = motor_w + dt * (cmd_w - motor_w) / p.tau_motor  # (N, 4)
        motor_w_new = np.clip(motor_w_new, 0.0, max_w)

        # Use average of old and new motor speeds for force computation
        motor_w_avg = 0.5 * (motor_w + motor_w_new)

        # --- Compute per-motor thrust ---
        w_sq = motor_w_avg ** 2  # (N, 4)
        thrust_per_motor = p.k_thrust * w_sq  # (N, 4)
        total_thrust = np.sum(thrust_per_motor, axis=1)  # (N,)

        # --- Body-frame force ---
        # Thrust along body z-axis (upward in body frame)
        force_body = np.zeros((n, 3), dtype=np.float64)
        force_body[:, 2] = total_thrust

        # Body drag (quadratic, if drag_coeff nonzero)
        if np.any(p.drag_coeff != 0.0):
            # Transform velocity to body frame
            R = quat_to_rotmat_batch(quat)  # (N, 3, 3)
            vel_body = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), vel)
            drag_force = -p.drag_coeff * vel_body * np.abs(vel_body)
            force_body += drag_force

        # --- Torques ---
        # Roll torque: arm_length * k_thrust * (-w0^2 + w1^2 + w2^2 - w3^2) / sqrt(2)
        tau_roll = p.arm_length * p.k_thrust * _SQRT2_INV * (
            -w_sq[:, 0] + w_sq[:, 1] + w_sq[:, 2] - w_sq[:, 3]
        )
        # Pitch torque: arm_length * k_thrust * (-w0^2 - w1^2 + w2^2 + w3^2) / sqrt(2)
        tau_pitch = p.arm_length * p.k_thrust * _SQRT2_INV * (
            -w_sq[:, 0] - w_sq[:, 1] + w_sq[:, 2] + w_sq[:, 3]
        )
        # Yaw torque: k_torque * (w0^2 - w1^2 + w2^2 - w3^2)
        # CW motors (+) produce negative yaw torque, CCW (-) produce positive
        tau_yaw = p.k_torque * (
            w_sq[:, 0] - w_sq[:, 1] + w_sq[:, 2] - w_sq[:, 3]
        )

        torques = np.stack([tau_roll, tau_pitch, tau_yaw], axis=1)  # (N, 3)

        # --- Gyroscopic torque: omega x (J * omega) ---
        J_omega = omega * self._J_diag  # (N, 3) assuming diagonal inertia
        gyro_torque = np.cross(omega, J_omega)  # (N, 3)

        # --- Angular acceleration ---
        # J * alpha = torques - omega x (J * omega)
        net_torque = torques - gyro_torque
        alpha = np.einsum("ij,nj->ni", self._J_inv, net_torque)  # (N, 3)

        # --- Transform thrust to world frame ---
        R = quat_to_rotmat_batch(quat)  # (N, 3, 3)
        force_world = np.einsum("nij,nj->ni", R, force_body)  # (N, 3)

        # --- Translational acceleration ---
        gravity_vec = np.array([0.0, 0.0, -GRAVITY])
        accel = force_world / p.mass + gravity_vec  # (N, 3)

        # --- Quaternion derivative ---
        # dq/dt = 0.5 * q * [0, omega]
        omega_quat = np.zeros((n, 4), dtype=np.float64)
        omega_quat[:, 1:4] = omega
        dquat = 0.5 * quat_multiply_batch(quat, omega_quat)

        # --- Forward Euler integration ---
        new_states = np.empty_like(states)

        # Position
        new_states[:, POS] = pos + dt * vel
        # Velocity
        new_states[:, VEL] = vel + dt * accel
        # Quaternion (with renormalization)
        new_quat = quat + dt * dquat
        quat_norms = np.linalg.norm(new_quat, axis=1, keepdims=True)
        quat_norms = np.maximum(quat_norms, 1e-10)  # avoid division by zero
        new_states[:, QUAT] = new_quat / quat_norms
        # Angular velocity
        new_states[:, OMEGA] = omega + dt * alpha
        # Motor speeds (already computed above)
        new_states[:, MOTOR] = motor_w_new

        return new_states

    def hover_omega(self) -> float:
        """Compute the per-motor angular velocity for hover equilibrium.

        Returns:
            Motor speed in rad/s such that 4 * k_thrust * omega^2 = mass * g.
        """
        return float(np.sqrt(
            self.params.mass * GRAVITY / (4.0 * self.params.k_thrust)
        ))
