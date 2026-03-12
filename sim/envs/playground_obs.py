"""24-dim gate-relative observation transform.

Converts world-frame drone state into the current target gate's reference frame.
Works for both single env (state shape (16,)) and batched (state shape (N, 16)).

State layout (16):
  [0:3]   position (x, y, z)
  [3:6]   velocity (vx, vy, vz)
  [6:9]   euler angles (roll, pitch, yaw)
  [9:12]  angular velocity (p, q, r)
  [12:16] motor speeds

Observation layout (24):
  [0:3]   position in gate frame
  [3:6]   velocity in gate frame
  [6:9]   euler angles in gate frame (yaw relative to gate heading)
  [9:12]  angular velocity (world frame)
  [12:15] angular velocity (body frame, same as world for p,q,r)
  [15:18] next gate relative position (in current gate frame)
  [18]    next gate relative yaw
  [19:23] normalized motor speeds (mapped to [0, 1])
  [23]    placeholder (zero)
"""

from __future__ import annotations

import numpy as np


def _rot2d(yaw: np.ndarray) -> np.ndarray:
    """Return 2x2 rotation matrix (or batch) that rotates by *yaw*.

    For a single scalar yaw, returns shape (2, 2).
    For a batch of shape (N,), returns shape (N, 2, 2).
    """
    yaw = np.asarray(yaw, dtype=np.float64)
    c = np.cos(yaw)
    s = np.sin(yaw)
    if yaw.ndim == 0:
        return np.array([[c, -s], [s, c]])
    # Batched: (N, 2, 2)
    R = np.empty((*yaw.shape, 2, 2), dtype=np.float64)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    return R


def _rotate_xy(R: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply 2D rotation R to xy vectors.

    Single: R (2,2), xy (2,) -> (2,)
    Batch:  R (N,2,2), xy (N,2) -> (N,2)
    """
    if R.ndim == 2:
        return R @ xy
    # (N, 2, 2) @ (N, 2, 1) -> (N, 2, 1) -> (N, 2)
    return np.einsum("nij,nj->ni", R, xy)


def gate_relative_obs(
    state: np.ndarray,
    gate_pos: np.ndarray,
    gate_yaw: float | np.ndarray,
    next_gate_pos: np.ndarray,
    next_gate_yaw: float | np.ndarray,
) -> np.ndarray:
    """Compute 24-dim gate-relative observation.

    Parameters
    ----------
    state : (16,) or (N, 16)
    gate_pos : (3,) or (N, 3)
    gate_yaw : scalar or (N,)
    next_gate_pos : (3,) or (N, 3)
    next_gate_yaw : scalar or (N,)

    Returns
    -------
    obs : (24,) or (N, 24)
    """
    state = np.asarray(state, dtype=np.float64)
    gate_pos = np.asarray(gate_pos, dtype=np.float64)
    gate_yaw = np.asarray(gate_yaw, dtype=np.float64)
    next_gate_pos = np.asarray(next_gate_pos, dtype=np.float64)
    next_gate_yaw = np.asarray(next_gate_yaw, dtype=np.float64)

    single = state.ndim == 1
    if single:
        state = state[np.newaxis, :]
        gate_pos = gate_pos[np.newaxis, :]
        gate_yaw = gate_yaw[np.newaxis]
        next_gate_pos = next_gate_pos[np.newaxis, :]
        next_gate_yaw = next_gate_yaw[np.newaxis]

    n = state.shape[0]

    # Rotation matrix: world -> gate frame (rotate by -gate_yaw)
    R = _rot2d(-gate_yaw)  # (N, 2, 2)

    # --- Position in gate frame [0:3] ---
    dp = state[:, 0:3] - gate_pos  # (N, 3)
    pos_gate = np.empty((n, 3), dtype=np.float64)
    pos_gate[:, :2] = _rotate_xy(R, dp[:, :2])
    pos_gate[:, 2] = dp[:, 2]

    # --- Velocity in gate frame [3:6] ---
    vel = state[:, 3:6]  # (N, 3)
    vel_gate = np.empty((n, 3), dtype=np.float64)
    vel_gate[:, :2] = _rotate_xy(R, vel[:, :2])
    vel_gate[:, 2] = vel[:, 2]

    # --- Euler angles in gate frame [6:9] ---
    euler_gate = state[:, 6:9].copy()  # roll, pitch, yaw
    euler_gate[:, 2] = euler_gate[:, 2] - gate_yaw  # relative yaw

    # --- Angular velocity (world = body) [9:12] and [12:15] ---
    omega = state[:, 9:12]  # (N, 3)

    # --- Next gate relative position in current gate frame [15:18] ---
    dnp = next_gate_pos - gate_pos  # (N, 3)
    next_pos_gate = np.empty((n, 3), dtype=np.float64)
    next_pos_gate[:, :2] = _rotate_xy(R, dnp[:, :2])
    next_pos_gate[:, 2] = dnp[:, 2]

    # --- Next gate relative yaw [18] ---
    next_rel_yaw = (next_gate_yaw - gate_yaw)[:, np.newaxis]  # (N, 1)

    # --- Normalized motor speeds [19:23] ---
    motors = state[:, 12:16]  # raw motor speeds
    w_min, w_max = 238.49, 3295.50
    motors_norm = (motors - w_min) / (w_max - w_min)

    # --- Placeholder [23] ---
    placeholder = np.zeros((n, 1), dtype=np.float64)

    # Assemble observation (N, 24)
    obs = np.concatenate(
        [
            pos_gate,        # 0:3
            vel_gate,        # 3:6
            euler_gate,      # 6:9
            omega,           # 9:12
            omega,           # 12:15  (body frame = world frame for p,q,r)
            next_pos_gate,   # 15:18
            next_rel_yaw,    # 18:19
            motors_norm,     # 19:23
            placeholder,     # 23:24
        ],
        axis=1,
    )

    if single:
        return obs[0]
    return obs
