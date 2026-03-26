"""PD waypoint tracker — classical base policy for α-RPO.

Uses privileged gate positions (available in sim) to compute TRPY commands.
Only used during training; fully attenuated by end of α-RPO training.
Outputs: [thrust_N, ωx_cmd, ωy_cmd, ωz_cmd] in physical units.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

GRAVITY = 9.81


class PDWaypointTracker:
    def __init__(
        self,
        gate_positions: NDArray[np.float64],
        mass: float = 0.752,
        kp_pos: NDArray[np.float64] | None = None,
        kd_pos: NDArray[np.float64] | None = None,
        kp_att: float = 8.0,
    ) -> None:
        self.gate_positions = np.asarray(gate_positions, dtype=np.float64)
        self.mass = mass
        self.kp_pos = np.asarray(kp_pos if kp_pos is not None else [6.0, 6.0, 6.0])
        self.kd_pos = np.asarray(kd_pos if kd_pos is not None else [4.0, 4.0, 4.0])
        self.kp_att = kp_att

    def get_action(self, pos, vel, quat, omega, gate_idx):
        target = self.gate_positions[gate_idx % len(self.gate_positions)]
        pos_error = target - pos

        # Target velocity toward gate (not zero — better bootstrap)
        dist = np.linalg.norm(pos_error)
        if dist > 0.1:
            direction = pos_error / dist
            desired_vel = direction * min(dist * 2.0, 5.0)
        else:
            desired_vel = np.zeros(3)
        vel_error = desired_vel - vel
        acc_des = self.kp_pos * pos_error + self.kd_pos * vel_error

        acc_total = acc_des + np.array([0.0, 0.0, GRAVITY])
        thrust = self.mass * np.linalg.norm(acc_total)

        z_des = acc_total / max(np.linalg.norm(acc_total), 1e-6)
        R = _quat_to_rotmat(quat)
        z_cur = R[:, 2]
        att_error_world = np.cross(z_cur, z_des)
        att_error_body = R.T @ att_error_world
        rate_cmd = self.kp_att * att_error_body - 2.0 * omega

        return np.array([thrust, rate_cmd[0], rate_cmd[1], rate_cmd[2]])

    def get_action_batch(self, pos, vel, quat, omega, gate_idx):
        n = pos.shape[0]
        actions = np.empty((n, 4), dtype=np.float64)
        for i in range(n):
            actions[i] = self.get_action(pos[i], vel[i], quat[i], omega[i], int(gate_idx[i]))
        return actions


def _quat_to_rotmat(q):
    """Quaternion [w, x, y, z] to 3x3 rotation matrix (body-to-world)."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
