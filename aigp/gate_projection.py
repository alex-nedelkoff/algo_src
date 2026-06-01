"""Project a Gate (NED pose + size) into the camera image."""
from __future__ import annotations

import numpy as np

from .geometry import quat_to_R, world_to_camera, project, in_frame
from .protocol import Gate


def gate_corners_world(gate: Gate) -> np.ndarray:
    """4 gate-opening corners in world NED, shape (4,3).
    Opening spans the gate-local y-z plane (local x = through-gate normal)."""
    w, h = gate.width, gate.height
    local = np.array([
        [0.0, -w / 2, -h / 2],
        [0.0, +w / 2, -h / 2],
        [0.0, +w / 2, +h / 2],
        [0.0, -w / 2, +h / 2],
    ])
    R = quat_to_R(gate.quat_ned_wxyz)   # gate-local -> world
    return gate.pos_ned[None, :] + local @ R.T


def project_gate(gate: Gate, drone_pos_ned, drone_quat_wxyz) -> dict:
    """Return center pixel, bbox, in-frame flag, and range for a gate."""
    c_cam = world_to_camera(gate.pos_ned, drone_pos_ned, drone_quat_wxyz)
    rng = float(np.linalg.norm(np.asarray(gate.pos_ned) - np.asarray(drone_pos_ned)))
    center = project(c_cam)

    pts = []
    for corner in gate_corners_world(gate):
        p = project(world_to_camera(corner, drone_pos_ned, drone_quat_wxyz))
        if p is not None:
            pts.append(p)

    if center is None or not pts:
        return {"center_px": None, "bbox_px": None, "in_frame": False, "range_m": rng}

    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    bbox = (min(us), min(vs), max(us), max(vs))
    visible = in_frame(*center)
    return {"center_px": center, "bbox_px": bbox, "in_frame": visible, "range_m": rng}
