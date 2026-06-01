"""Camera model + NED/body/camera transforms. Pure numpy."""
from __future__ import annotations

import numpy as np

IMG_W, IMG_H = 640, 360
# Given intrinsics: fx=fy=320, cx=320, cy=180 (HFoV 90 deg; VFoV ~58.7 deg).
K = np.array([[320.0, 0.0, 320.0],
              [0.0, 320.0, 180.0],
              [0.0, 0.0, 1.0]])

# Body frame is FRD (x-forward, y-right, z-down). Camera optical frame is
# (x-right, y-down, z-forward), looking along body +x:
#   optical_x(right)   = body_y
#   optical_y(down)    = body_z
#   optical_z(forward) = body_x
R_BODY_TO_CAM = np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]])


def quat_to_R(q_wxyz) -> np.ndarray:
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix (body-to-world)."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def world_to_camera(p_world_ned, drone_pos_ned, drone_quat_wxyz) -> np.ndarray:
    """Point in world NED -> camera optical frame."""
    R_bw = quat_to_R(drone_quat_wxyz)         # body -> world
    p_body = R_bw.T @ (np.asarray(p_world_ned, float) - np.asarray(drone_pos_ned, float))
    return R_BODY_TO_CAM @ p_body


def project(p_cam, intrinsics: np.ndarray = K):
    """Camera-optical point -> (u, v) pixels, or None if behind the camera."""
    z = p_cam[2]
    if z <= 1e-6:
        return None
    u = intrinsics[0, 0] * p_cam[0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * p_cam[1] / z + intrinsics[1, 2]
    return float(u), float(v)


def in_frame(u: float, v: float) -> bool:
    return 0.0 <= u < IMG_W and 0.0 <= v < IMG_H
