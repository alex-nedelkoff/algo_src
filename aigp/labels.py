"""Compute a JSON-serializable training label from drone state + gate pose."""
from __future__ import annotations

import numpy as np

from .gate_projection import project_gate
from .geometry import world_to_camera
from .protocol import Gate
from .state import DroneState


def compute_label(frame_idx: int, t_sim_ns: int, drone: DroneState, gate: Gate) -> dict:
    proj = project_gate(gate, drone.pos_ned, drone.quat_wxyz)
    rel_cam = world_to_camera(gate.pos_ned, drone.pos_ned, drone.quat_wxyz)
    return {
        "frame": int(frame_idx),
        "t_sim_ns": int(t_sim_ns),
        "drone_pos_ned": [float(x) for x in drone.pos_ned],
        "drone_quat_wxyz": [float(x) for x in drone.quat_wxyz],
        "gate_id": int(gate.id),
        "gate_pos_ned": [float(x) for x in gate.pos_ned],
        "gate_quat_wxyz": [float(x) for x in gate.quat_ned_wxyz],
        "gate_rel_cam": [float(x) for x in rel_cam],
        "range_m": float(proj["range_m"]),
        "in_frame": bool(proj["in_frame"]),
        "center_px": list(proj["center_px"]) if proj["center_px"] else None,
        "bbox_px": list(proj["bbox_px"]) if proj["bbox_px"] else None,
    }
