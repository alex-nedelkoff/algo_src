"""Pinhole camera model for synthetic gate corner projection.

Camera convention (matches drone body frame):
  - Optical axis: +X (forward)
  - Right: +Y
  - Down: +Z (NED)

Image convention:
  - u: horizontal (0 = left, width-1 = right)
  - v: vertical (0 = top, height-1 = bottom)
"""
from __future__ import annotations

import numpy as np


class PinholeCamera:
    """Vectorized pinhole camera for projecting 3D points to pixel coords."""

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        hfov_deg: float = 90.0,
    ) -> None:
        self.width = width
        self.height = height
        self.fx = width / (2 * np.tan(np.radians(hfov_deg / 2)))
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0

    def project(
        self, pts_cam: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project 3D points in camera frame to pixel coordinates.

        Args:
            pts_cam: (n_envs, n_points, 3) points in camera frame.

        Returns:
            pixels: (n_envs, n_points, 2) pixel coordinates [u, v].
            visible: (n_envs, n_points) bool mask.
        """
        x = pts_cam[..., 0]
        y = pts_cam[..., 1]
        z = pts_cam[..., 2]

        in_front = x > 0.01
        x_safe = np.where(in_front, x, 1.0)
        u = self.fx * (y / x_safe) + self.cx
        v = self.fy * (z / x_safe) + self.cy

        in_frame = (
            (u >= 0) & (u < self.width) &
            (v >= 0) & (v < self.height)
        )

        visible = in_front & in_frame
        pixels = np.stack([u, v], axis=-1)
        return pixels, visible


def compute_gate_corners(
    gate_pos: np.ndarray,
    gate_yaw: np.ndarray,
    width: float = 0.55,
    height: float = 0.55,
) -> np.ndarray:
    """Compute 3D world coordinates of 4 gate corners.

    Args:
        gate_pos: (n_envs, 3) gate center positions.
        gate_yaw: (n_envs,) gate yaw angles.
        width: gate opening width in meters.
        height: gate opening height in meters.

    Returns:
        corners: (n_envs, 4, 3) corner positions in world frame.
            Order: top-left, top-right, bottom-right, bottom-left
            (as seen from the approach side).
    """
    n = gate_pos.shape[0]
    hw, hh = width / 2, height / 2

    right_x = -np.sin(gate_yaw)
    right_y = np.cos(gate_yaw)

    corners = np.zeros((n, 4, 3))
    for i, (dy_sign, dz_sign) in enumerate([
        (-1, -1),  # top-left (NED: -z is up)
        (+1, -1),  # top-right
        (+1, +1),  # bottom-right
        (-1, +1),  # bottom-left
    ]):
        corners[:, i, 0] = gate_pos[:, 0] + dy_sign * hw * right_x
        corners[:, i, 1] = gate_pos[:, 1] + dy_sign * hw * right_y
        corners[:, i, 2] = gate_pos[:, 2] + dz_sign * hh

    return corners


def _euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to rotation matrices.

    Args:
        euler: (n, 3) array of [phi, theta, psi].

    Returns:
        R: (n, 3, 3) rotation matrices (body -> world).
    """
    phi = euler[:, 0]
    theta = euler[:, 1]
    psi = euler[:, 2]

    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    n = euler.shape[0]
    R = np.zeros((n, 3, 3))
    R[:, 0, 0] = cpsi * cth
    R[:, 0, 1] = cpsi * sth * sphi - spsi * cphi
    R[:, 0, 2] = cpsi * sth * cphi + spsi * sphi
    R[:, 1, 0] = spsi * cth
    R[:, 1, 1] = spsi * sth * sphi + cpsi * cphi
    R[:, 1, 2] = spsi * sth * cphi - cpsi * sphi
    R[:, 2, 0] = -sth
    R[:, 2, 1] = cth * sphi
    R[:, 2, 2] = cth * cphi
    return R


def world_to_camera(
    pts_world: np.ndarray,
    drone_pos: np.ndarray,
    drone_euler: np.ndarray,
) -> np.ndarray:
    """Transform world points to camera (body) frame.

    Camera is body-fixed: +X forward, +Y right, +Z down (NED).

    Args:
        pts_world: (n_envs, n_points, 3) world coordinates.
        drone_pos: (n_envs, 3) drone position.
        drone_euler: (n_envs, 3) drone euler angles [phi, theta, psi].

    Returns:
        pts_cam: (n_envs, n_points, 3) points in camera frame.
    """
    rel = pts_world - drone_pos[:, np.newaxis, :]
    R = _euler_to_rotation_matrix(drone_euler)
    # world -> body = R^T @ rel  =>  einsum 'nji,npj->npi'
    pts_cam = np.einsum('nji,npj->npi', R, rel)
    return pts_cam


class CornerDetector:
    """Synthetic corner detection with noise and HMM dropout.

    Wraps PinholeCamera projection with:
    - Distance-scaled pixel noise: sigma = noise_k * d2g
    - HMM dropout (same model as PerceptionNoise)
    """

    def __init__(
        self,
        camera: PinholeCamera,
        n_envs: int,
        seed: int,
        noise_k: float = 2.0,
        dropout_onset: float | None = None,
        dropout_continuation: float = 0.7,
    ) -> None:
        self.camera = camera
        self.n_envs = n_envs
        self.noise_k = noise_k
        self.dropout_continuation = dropout_continuation
        self.dropout_onset_fixed = dropout_onset

        self._rng = np.random.default_rng(seed)
        self._hmm_state = np.zeros(n_envs, dtype=bool)
        self._p_onset = self._sample_onset(n_envs)
        self._initialized = np.zeros(n_envs, dtype=bool)

    def detect(
        self,
        corners_cam: np.ndarray,
        d2g: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project corners, add noise, apply dropout.

        Args:
            corners_cam: (n_envs, 4, 3) gate corners in camera frame.
            d2g: (n_envs,) distance to gate.

        Returns:
            pixels: (n_envs, 4, 2) noisy pixel coordinates.
            visible: (n_envs, 4) bool mask.
        """
        pixels, geo_visible = self.camera.project(corners_cam)

        sigma = self.noise_k * d2g
        noise = self._rng.normal(
            scale=sigma[:, np.newaxis, np.newaxis],
            size=pixels.shape,
        )
        pixels = pixels + noise

        first_step = ~self._initialized
        self._initialized |= True

        drop = self._step_hmm()
        # Never drop on the first step after init/reset
        drop = drop & ~first_step
        visible = geo_visible.copy()
        visible[drop] = False

        return pixels, visible

    def reset(self, env_mask: np.ndarray) -> None:
        self._hmm_state[env_mask] = False
        self._initialized[env_mask] = False
        n_reset = int(np.sum(env_mask))
        if n_reset > 0:
            self._p_onset[env_mask] = self._sample_onset(n_reset)

    def _sample_onset(self, n: int) -> np.ndarray:
        if self.dropout_onset_fixed is not None:
            return np.full(n, self.dropout_onset_fixed)
        return self._rng.uniform(0.05, 0.50, size=n)

    def _step_hmm(self) -> np.ndarray:
        u = self._rng.uniform(size=self.n_envs)
        threshold = np.where(
            self._hmm_state,
            self.dropout_continuation,
            self._p_onset,
        )
        self._hmm_state = u < threshold
        return self._hmm_state
