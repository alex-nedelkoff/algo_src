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
        self._n_envs = 0  # set by reset()

        # Per-env arrays initialized lazily in reset()
        self._mass: NDArray[np.float64] = np.empty(0)
        self._k_thrust: NDArray[np.float64] = np.empty(0)
        self._k_torque: NDArray[np.float64] = np.empty(0)
        self._arm_length: NDArray[np.float64] = np.empty(0)
        self._tau_motor: NDArray[np.float64] = np.empty(0)
        self._max_omega: NDArray[np.float64] = np.empty(0)
        self._drag_coeff: NDArray[np.float64] = np.empty(0)  # (N, 3)
        self._J_inv: NDArray[np.float64] = np.empty(0)       # (N, 3, 3)
        self._J_diag: NDArray[np.float64] = np.empty(0)      # (N, 3)

    def reset(self, n_envs: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        """Reset all environments to hover initial condition.

        Args:
            n_envs: Number of parallel environments.
            rng: Optional random generator for randomized initial states.

        Returns:
            Initial states array of shape (n_envs, 17).
        """
        self._n_envs = n_envs
        p = self.params

        # Broadcast nominal params to per-env arrays
        self._mass = np.full(n_envs, p.mass)
        self._k_thrust = np.full(n_envs, p.k_thrust)
        self._k_torque = np.full(n_envs, p.k_torque)
        self._arm_length = np.full(n_envs, p.arm_length)
        self._tau_motor = np.full(n_envs, p.tau_motor)
        self._max_omega = np.full(n_envs, p.max_omega)
        self._drag_coeff = np.tile(p.drag_coeff, (n_envs, 1))           # (N, 3)
        self._J_inv = np.tile(np.linalg.inv(p.inertia), (n_envs, 1, 1)) # (N, 3, 3)
        self._J_diag = np.tile(np.diag(p.inertia), (n_envs, 1))         # (N, 3)

        return self.make_reset_states(n_envs)

    def make_reset_states(
        self, n_envs: int, env_indices: NDArray[np.intp] | None = None
    ) -> NDArray[np.float64]:
        """Generate initial state vectors without touching param arrays.

        Uses per-env mass/k_thrust for hover omega when env_indices are
        provided (auto-reset path). Falls back to nominal params otherwise.

        Args:
            n_envs: Number of state vectors to generate.
            env_indices: If provided, use per-env params at these indices
                for hover omega computation. Shape (n_envs,).

        Returns:
            Initial states (n_envs, 17).
        """
        states = np.zeros((n_envs, 17), dtype=np.float64)
        states[:, 2] = 1.0   # z = 1m
        states[:, 6] = 1.0   # quat w = 1

        if env_indices is not None and self._n_envs > 0:
            # Use per-env randomized mass/k_thrust for correct hover
            hover_omega = np.sqrt(
                self._mass[env_indices] * GRAVITY
                / (4.0 * self._k_thrust[env_indices])
            )
        else:
            hover_omega = np.full(n_envs, np.sqrt(
                self.params.mass * GRAVITY / (4.0 * self.params.k_thrust)
            ))

        states[:, 13:17] = hover_omega[:, None]
        return states

    def update_params(
        self,
        env_indices: NDArray[np.intp],
        mass: NDArray[np.float64] | None = None,
        inertia: NDArray[np.float64] | None = None,
        k_thrust: NDArray[np.float64] | None = None,
        k_torque: NDArray[np.float64] | None = None,
        arm_length: NDArray[np.float64] | None = None,
        tau_motor: NDArray[np.float64] | None = None,
        max_rpm: NDArray[np.float64] | None = None,
        drag_coeff: NDArray[np.float64] | None = None,
    ) -> None:
        """Update physics params for specific environments.

        Args:
            env_indices: Indices of envs to update, shape (K,).
            mass: New masses, shape (K,).
            inertia: New inertia tensors, shape (K, 3, 3).
            k_thrust: shape (K,). k_torque: shape (K,).
            arm_length: shape (K,). tau_motor: shape (K,).
            max_rpm: shape (K,). drag_coeff: shape (K, 3).
        """
        if mass is not None:
            self._mass[env_indices] = mass
        if k_thrust is not None:
            self._k_thrust[env_indices] = k_thrust
        if k_torque is not None:
            self._k_torque[env_indices] = k_torque
        if arm_length is not None:
            self._arm_length[env_indices] = arm_length
        if tau_motor is not None:
            self._tau_motor[env_indices] = tau_motor
        if max_rpm is not None:
            self._max_omega[env_indices] = max_rpm * 2.0 * np.pi / 60.0
        if drag_coeff is not None:
            self._drag_coeff[env_indices] = drag_coeff
        if inertia is not None:
            # Batched inversion: np.linalg.inv supports (K, 3, 3) input
            self._J_inv[env_indices] = np.linalg.inv(inertia)
            self._J_diag[env_indices] = inertia[:, [0, 1, 2], [0, 1, 2]]

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

        n = states.shape[0]

        # Extract state components
        pos = states[:, POS]          # (N, 3)
        vel = states[:, VEL]          # (N, 3)
        quat = states[:, QUAT]        # (N, 4)
        omega = states[:, OMEGA]      # (N, 3)
        motor_w = states[:, MOTOR]    # (N, 4)

        # Clip actions to valid range (per-env max_omega)
        max_w = self._max_omega[:n]                     # (N,)
        cmd_w = np.clip(actions, 0.0, max_w[:, None])   # (N, 4)

        # --- Motor dynamics: first-order lag ---
        tau = self._tau_motor[:n, None]                  # (N, 1) for broadcast
        motor_w_new = motor_w + dt * (cmd_w - motor_w) / tau  # (N, 4)
        motor_w_new = np.clip(motor_w_new, 0.0, max_w[:, None])

        # Use average of old and new motor speeds for force computation
        motor_w_avg = 0.5 * (motor_w + motor_w_new)

        # --- Compute per-motor thrust ---
        w_sq = motor_w_avg ** 2                          # (N, 4)
        k_t = self._k_thrust[:n, None]                   # (N, 1)
        thrust_per_motor = k_t * w_sq                    # (N, 4)
        total_thrust = np.sum(thrust_per_motor, axis=1)  # (N,)

        # --- Body-frame force ---
        force_body = np.zeros((n, 3), dtype=np.float64)
        force_body[:, 2] = total_thrust

        # Body drag (quadratic, if drag_coeff nonzero)
        if np.any(self._drag_coeff[:n] != 0.0):
            R = quat_to_rotmat_batch(quat)  # (N, 3, 3)
            vel_body = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), vel)
            drag_force = -self._drag_coeff[:n] * vel_body * np.abs(vel_body)
            force_body += drag_force

        # --- Torques (per-env arm_length, k_thrust, k_torque) ---
        arm_k = self._arm_length[:n] * self._k_thrust[:n] * _SQRT2_INV  # (N,)
        tau_roll = arm_k * (
            -w_sq[:, 0] + w_sq[:, 1] + w_sq[:, 2] - w_sq[:, 3]
        )
        tau_pitch = arm_k * (
            -w_sq[:, 0] - w_sq[:, 1] + w_sq[:, 2] + w_sq[:, 3]
        )
        k_q = self._k_torque[:n]                        # (N,)
        tau_yaw = k_q * (
            w_sq[:, 0] - w_sq[:, 1] + w_sq[:, 2] - w_sq[:, 3]
        )

        torques = np.stack([tau_roll, tau_pitch, tau_yaw], axis=1)  # (N, 3)

        # --- Gyroscopic torque: omega x (J * omega) ---
        J_omega = omega * self._J_diag[:n]               # (N, 3)
        gyro_torque = np.cross(omega, J_omega)            # (N, 3)

        # --- Angular acceleration (per-env inertia) ---
        net_torque = torques - gyro_torque
        alpha = np.einsum("nij,nj->ni", self._J_inv[:n], net_torque)  # (N, 3)

        # --- Transform thrust to world frame ---
        R = quat_to_rotmat_batch(quat)  # (N, 3, 3)
        force_world = np.einsum("nij,nj->ni", R, force_body)  # (N, 3)

        # --- Translational acceleration (per-env mass) ---
        gravity_vec = np.array([0.0, 0.0, -GRAVITY])
        accel = force_world / self._mass[:n, None] + gravity_vec  # (N, 3)

        # --- Quaternion derivative ---
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
        quat_norms = np.maximum(quat_norms, 1e-10)
        new_states[:, QUAT] = new_quat / quat_norms
        # Angular velocity
        new_states[:, OMEGA] = omega + dt * alpha
        # Motor speeds
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

    def hover_omega_per_env(self) -> NDArray[np.float64]:
        """Per-env hover omega, shape (N,). Uses current per-env params."""
        return np.sqrt(self._mass * GRAVITY / (4.0 * self._k_thrust))
