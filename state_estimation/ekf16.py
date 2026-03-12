"""16-state Extended Kalman Filter for quadrotor state estimation.

State vector: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4]

Prediction uses the same dynamics model as the simulator.
Update fuses gate corner pixel measurements from the camera.
"""
from __future__ import annotations

import numpy as np

from sim.dynamics.sympy_quad import NOMINAL_PARAMS, build_dynamics_fn


class EKF16:
    """Vectorized 16-state EKF across n_envs."""

    # Maximum allowed diagonal covariance values per state group
    _P_MAX = np.array([
        100, 100, 100,          # position
        50, 50, 50,             # velocity
        10, 10, 10,             # euler angles
        10, 10, 10,             # angular rates
        500, 500, 500, 500,     # motor speeds
    ])

    # Physical state bounds: [min, max] per state
    _STATE_LO = np.array([
        -50, -50, -50,          # position (m)
        -30, -30, -30,          # velocity (m/s)
        -np.pi, -np.pi/2, -np.pi,  # euler angles
        -50, -50, -50,          # angular rates (rad/s)
        0, 0, 0, 0,            # motor speeds (rpm >= 0)
    ])
    _STATE_HI = np.array([
        50, 50, 50,
        30, 30, 30,
        np.pi, np.pi/2, np.pi,
        50, 50, 50,
        3000, 3000, 3000, 3000,
    ])

    def __init__(
        self,
        n_envs: int,
        dt: float = 0.01,
    ) -> None:
        self.n_envs = n_envs
        self.dt = dt
        self._dynamics_fn = build_dynamics_fn()

        # State estimate: (n_envs, 16)
        self.x = np.zeros((n_envs, 16))

        # Covariance: (n_envs, 16, 16)
        p0_diag = np.array([
            1.0, 1.0, 1.0,       # position
            0.5, 0.5, 0.5,       # velocity
            0.1, 0.1, 0.5,       # euler angles
            0.5, 0.5, 0.5,       # angular rates
            100, 100, 100, 100,  # motor speeds
        ])
        self.P = np.zeros((n_envs, 16, 16))
        for i in range(n_envs):
            self.P[i] = np.diag(p0_diag)
        self._p0_diag = p0_diag

        # Process noise
        q_diag = np.array([
            0.001, 0.001, 0.001,    # position
            0.01, 0.01, 0.01,       # velocity
            0.005, 0.005, 0.005,    # euler
            0.1, 0.1, 0.1,          # angular rates
            1.0, 1.0, 1.0, 1.0,    # motor speeds (reduced from 10.0)
        ])
        self.Q = np.diag(q_diag) * dt

        # Nominal params (no DR in EKF)
        self._params = {k: np.full(n_envs, v) for k, v in NOMINAL_PARAMS.items()}

    def set_state(self, state: np.ndarray, env_mask: np.ndarray) -> None:
        """Initialize EKF state for masked envs."""
        self.x[env_mask] = state
        self.P[env_mask] = np.diag(self._p0_diag)

    def predict(self) -> None:
        """EKF predict step using simple kinematic model.

        Uses linear kinematics instead of full nonlinear dynamics to avoid
        divergence from dynamics model mismatch during aggressive flight.
        The IMU update provides angular rates and motor speeds directly,
        so the prediction only needs to propagate position and orientation.
        """
        # Kinematic prediction:
        # pos += vel * dt
        self.x[:, 0:3] += self.x[:, 3:6] * self.dt

        # Proper euler rate equations from body rates (p, q, r):
        # phi_dot = p + (q*sin(phi) + r*cos(phi)) * tan(theta)
        # theta_dot = q*cos(phi) - r*sin(phi)
        # psi_dot = (q*sin(phi) + r*cos(phi)) / cos(theta)
        phi = self.x[:, 6]
        theta = self.x[:, 7]
        p = self.x[:, 9]
        q = self.x[:, 10]
        r = self.x[:, 11]
        sp, cp = np.sin(phi), np.cos(phi)
        tt = np.tan(np.clip(theta, -1.4, 1.4))  # clip to avoid tan singularity
        ct = np.cos(np.clip(theta, -1.4, 1.4))
        ct = np.maximum(ct, 0.1)  # avoid division by zero

        phi_dot = p + (q * sp + r * cp) * tt
        theta_dot = q * cp - r * sp
        psi_dot = (q * sp + r * cp) / ct

        self.x[:, 6] += phi_dot * self.dt
        self.x[:, 7] += theta_dot * self.dt
        self.x[:, 8] += psi_dot * self.dt

        # Wrap euler angles
        self.x[:, 6] = np.arctan2(np.sin(self.x[:, 6]), np.cos(self.x[:, 6]))
        self.x[:, 8] = np.arctan2(np.sin(self.x[:, 8]), np.cos(self.x[:, 8]))
        self.x[:, 7] = np.clip(self.x[:, 7], -np.pi/2, np.pi/2)

        # Clamp state to physical bounds
        self.x = np.clip(self.x, self._STATE_LO, self._STATE_HI)

        # Kinematic Jacobian — per-env since it depends on euler angles
        for i in range(self.n_envs):
            F = np.eye(16)
            F[0:3, 3:6] = np.eye(3) * self.dt   # pos depends on vel

            # Euler rate Jacobian (d_euler_dot / d_state)
            phi_i = self.x[i, 6]
            theta_i = self.x[i, 7]
            p_i, q_i, r_i = self.x[i, 9], self.x[i, 10], self.x[i, 11]
            sp, cp = np.sin(phi_i), np.cos(phi_i)
            ct = max(np.cos(np.clip(theta_i, -1.4, 1.4)), 0.1)
            tt = np.tan(np.clip(theta_i, -1.4, 1.4))

            # d(phi_dot)/d(phi) = (q*cos(phi) - r*sin(phi)) * tan(theta)
            F[6, 6] += (q_i * cp - r_i * sp) * tt * self.dt
            # d(phi_dot)/d(p,q,r)
            F[6, 9] += 1.0 * self.dt
            F[6, 10] += sp * tt * self.dt
            F[6, 11] += cp * tt * self.dt

            # d(theta_dot)/d(phi) = -q*sin(phi) - r*cos(phi)
            F[7, 6] += (-q_i * sp - r_i * cp) * self.dt
            F[7, 10] += cp * self.dt
            F[7, 11] += -sp * self.dt

            # d(psi_dot)/d(phi) = (q*cos(phi) - r*sin(phi)) / cos(theta)
            F[8, 6] += (q_i * cp - r_i * sp) / ct * self.dt
            F[8, 10] += sp / ct * self.dt
            F[8, 11] += cp / ct * self.dt

            self.P[i] = F @ self.P[i] @ F.T + self.Q

        # Clamp covariance diagonals to prevent explosion
        self._clamp_covariance()

    def update_imu(
        self,
        velocity: np.ndarray,
        euler: np.ndarray,
        angular_rates: np.ndarray,
        motor_speeds: np.ndarray,
        vel_noise_std: float = 0.1,
        euler_noise_std: float = 0.05,
        rate_noise_std: float = 0.05,
        motor_noise_std: float = 20.0,
    ) -> None:
        """EKF update using IMU/AHRS direct state measurements.

        Fuses velocity, euler angles (from AHRS), angular rates (gyro),
        and motor speed readings as direct observations.

        Args:
            velocity: (n_envs, 3) measured vx, vy, vz.
            euler: (n_envs, 3) measured phi, theta, psi (from AHRS).
            angular_rates: (n_envs, 3) measured p, q, r.
            motor_speeds: (n_envs, 4) measured motor speeds.
        """
        # Measurement: z = [vx,vy,vz, phi,theta,psi, p,q,r, w1,w2,w3,w4] (13D)
        z_obs = np.concatenate([velocity, euler, angular_rates, motor_speeds], axis=1)
        z_pred = np.concatenate([
            self.x[:, 3:6], self.x[:, 6:9], self.x[:, 9:12], self.x[:, 12:16]
        ], axis=1)
        innovation = z_obs - z_pred

        # H maps observed quantities to state indices
        H = np.zeros((13, 16))
        H[0:3, 3:6] = np.eye(3)     # velocity
        H[3:6, 6:9] = np.eye(3)     # euler angles
        H[6:9, 9:12] = np.eye(3)    # angular rates
        H[9:13, 12:16] = np.eye(4)  # motor speeds

        # Measurement noise
        r_diag = np.array([
            vel_noise_std**2, vel_noise_std**2, vel_noise_std**2,
            euler_noise_std**2, euler_noise_std**2, euler_noise_std**2,
            rate_noise_std**2, rate_noise_std**2, rate_noise_std**2,
            motor_noise_std**2, motor_noise_std**2, motor_noise_std**2, motor_noise_std**2,
        ])
        R = np.diag(r_diag)

        for i in range(self.n_envs):
            P = self.P[i]
            S = H @ P @ H.T + R
            try:
                K = P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(K)):
                continue

            self.x[i] += K @ innovation[i]

            I_KH = np.eye(16) - K @ H
            self.P[i] = I_KH @ P @ I_KH.T + K @ R @ K.T

        self.x = np.clip(self.x, self._STATE_LO, self._STATE_HI)
        self._clamp_covariance()

    def update(
        self,
        pixels: np.ndarray,
        visible: np.ndarray,
        gate_pos: np.ndarray,
        gate_yaw: np.ndarray,
        camera: "PinholeCamera",
    ) -> None:
        """EKF measurement update using gate corner pixel observations.

        Args:
            pixels: (n_envs, 4, 2) observed pixel coordinates.
            visible: (n_envs, 4) bool mask of detected corners.
            gate_pos: (n_envs, 3) gate center positions.
            gate_yaw: (n_envs,) gate yaw angles.
            camera: PinholeCamera instance.
        """
        from perception.camera import compute_gate_corners, world_to_camera

        for env_idx in range(self.n_envs):
            vis = visible[env_idx]
            n_vis = int(vis.sum())
            if n_vis == 0:
                continue

            # Predicted measurement
            corners_w = compute_gate_corners(
                gate_pos[env_idx:env_idx+1],
                gate_yaw[env_idx:env_idx+1],
                0.55, 0.55,
            )
            cam_pts = world_to_camera(
                corners_w,
                self.x[env_idx:env_idx+1, :3],
                self.x[env_idx:env_idx+1, 6:9],
            )
            pred_px, _ = camera.project(cam_pts)

            z_obs = pixels[env_idx, vis].flatten()
            z_pred = pred_px[0, vis].flatten()
            innovation = z_obs - z_pred

            # Measurement Jacobian via finite differences (adaptive eps)
            m_dim = n_vis * 2
            H = np.zeros((m_dim, 16))

            # Only compute Jacobian for states that affect pixel projection:
            # positions (0-2), euler angles (6-8). Other states have zero H columns.
            proj_states = [0, 1, 2, 6, 7, 8]
            for j in proj_states:
                eps_j = max(abs(self.x[env_idx, j]) * 1e-5, 1e-6)
                x_pert = self.x[env_idx].copy()
                x_pert[j] += eps_j
                cam_pts_pert = world_to_camera(
                    corners_w,
                    x_pert[:3].reshape(1, 3),
                    x_pert[6:9].reshape(1, 3),
                )
                pred_pert, _ = camera.project(cam_pts_pert)
                z_pert = pred_pert[0, vis].flatten()
                H[:, j] = (z_pert - z_pred) / eps_j

            # Measurement noise — bounded to prevent near-zero R
            d = float(np.linalg.norm(gate_pos[env_idx] - self.x[env_idx, :3]))
            r_var = max(2.0 * d + 1.0, 4.0) ** 2
            R = np.eye(m_dim) * r_var

            # Skip update if Jacobian has NaN/Inf (numerical issues)
            if not np.all(np.isfinite(H)):
                continue

            # Kalman gain
            P = self.P[env_idx]
            S = H @ P @ H.T + R
            try:
                K = P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue

            # Skip if gain is non-finite
            if not np.all(np.isfinite(K)):
                continue

            # State update
            self.x[env_idx] += K @ innovation

            # Covariance update (Joseph form)
            I_KH = np.eye(16) - K @ H
            self.P[env_idx] = I_KH @ P @ I_KH.T + K @ R @ K.T

        # Clamp state after update
        self.x = np.clip(self.x, self._STATE_LO, self._STATE_HI)
        self._clamp_covariance()

    def _clamp_covariance(self) -> None:
        """Clamp covariance diagonals and enforce symmetry + PSD."""
        for i in range(self.n_envs):
            # Enforce symmetry
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T)

            # Clamp diagonal elements
            diag = np.diag(self.P[i]).copy()
            diag = np.clip(diag, 1e-8, self._P_MAX)

            # Scale off-diagonals proportionally if diag was clamped
            old_diag = np.diag(self.P[i])
            scale = np.where(old_diag > 0, np.sqrt(diag / np.maximum(old_diag, 1e-12)), 1.0)
            self.P[i] = self.P[i] * np.outer(scale, scale)

    def _compute_jacobian_fd(self) -> np.ndarray:
        """Compute state transition Jacobian via finite differences.

        Uses adaptive eps: max(|x_j| * 1e-5, 1e-6) for numerical stability
        when state values span many orders of magnitude.

        Returns: (n_envs, 16, 16) Jacobian matrices.
        """
        F = np.zeros((self.n_envs, 16, 16))
        dW = np.zeros((self.n_envs, 4))
        dW_T = dW.T

        state_T = self.x.T
        f0 = self._dynamics_fn(state_T, dW_T, self._params)

        for j in range(16):
            # Adaptive eps based on state magnitude
            x_abs_max = np.max(np.abs(self.x[:, j]))
            eps_j = max(x_abs_max * 1e-5, 1e-6)

            x_pert = self.x.copy()
            x_pert[:, j] += eps_j
            state_T_pert = x_pert.T
            f_pert = self._dynamics_fn(state_T_pert, dW_T, self._params)

            dfdx = (f_pert - f0) / eps_j  # (16, n_envs)

            for i in range(self.n_envs):
                F[i, :, j] = dfdx[:, i] * self.dt

        # F = I + df/dx * dt
        for i in range(self.n_envs):
            F[i] += np.eye(16)

        return F
