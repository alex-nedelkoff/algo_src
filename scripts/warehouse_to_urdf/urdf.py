"""URDF and gates_enu.json writers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sim.pybullet.coords import ned_to_enu_position, ned_to_enu_quaternion

# ---------------------------------------------------------------------------
# Empirical gate→warehouse alignment correction (2026-04-27).
#
# After fixing the ue_to_ned handedness flip + cyclic ned_to_enu permutation,
# warehouse-mesh vertices and gate positions for the same UE point should
# coincide in PyBullet world. They DON'T — gates are still displaced by
# ~3.25m in X and Z, plus the constellation needs M_x ∘ Ry(+90) to align
# orientation-wise with the FBX-embedded gate geometry.
#
# Root cause is most likely either (a) the FAB BP_RaceGate prefab having
# its actor pivot offset from the torus center, or (b) Megascans-style
# UE asset internal transforms not surfacing through File→Export FBX.
# Tracking-down is left as future work; for now we bake the empirical
# transform into write_gates_enu_json so the published gate positions match
# the warehouse mesh visually. Gate orientations are similarity-transformed
# (M @ R @ M^T) to stay valid rotations.
#
# Transform: gate_world = M @ pos_enu_cyclic + offset
#   where M = M_x ∘ Ry(+90) = [[0,0,-1], [0,1,0], [-1,0,0]]
#   and offset = (-3.25, 0, -3.25).
# ---------------------------------------------------------------------------
_GATE_ALIGN_M = np.array([[0.0, 0.0, -1.0],
                          [0.0, 1.0,  0.0],
                          [-1.0, 0.0, 0.0]], dtype=np.float64)
_GATE_ALIGN_OFFSET = np.array([-3.25, 0.0, -3.25], dtype=np.float64)


def _align_gate_pose(pos_enu: np.ndarray, quat_enu_wxyz: np.ndarray) -> tuple[list, list]:
    """Apply the empirical gate→warehouse alignment transform.

    Position: M @ pos + offset.
    Orientation: similarity transform M @ R @ M^T (preserves det = +1).
    Returns (pos_world list, quat_wxyz list).
    """
    pos_world = (_GATE_ALIGN_M @ np.asarray(pos_enu, dtype=np.float64) + _GATE_ALIGN_OFFSET)

    # scipy uses (x, y, z, w) order; gates_enu.json uses (w, x, y, z)
    q_in_wxyz = np.asarray(quat_enu_wxyz, dtype=np.float64)
    q_xyzw = np.array([q_in_wxyz[1], q_in_wxyz[2], q_in_wxyz[3], q_in_wxyz[0]])
    R_in = Rotation.from_quat(q_xyzw).as_matrix()
    R_out = _GATE_ALIGN_M @ R_in @ _GATE_ALIGN_M  # M = M^T for our specific matrix
    q_out_xyzw = Rotation.from_matrix(R_out).as_quat()
    q_out_wxyz = [float(q_out_xyzw[3]), float(q_out_xyzw[0]),
                  float(q_out_xyzw[1]), float(q_out_xyzw[2])]
    return pos_world.tolist(), q_out_wxyz

# URDF rotations are identity because the pipeline produces meshes and
# gate poses in PyBullet-compatible right-handed Z-up ENU directly.
# Earlier versions baked rpy="0 ±1.5708 0" rotations to compensate for a
# bug in `ue_to_ned_mesh` / `ue_to_ned_position` (incorrect double sign
# flip → no actual handedness change, mesh ended up axis-swapped vs
# gate positions). Fixed at the pipeline level on 2026-04-27 — the
# basis flip matrix S is now diag(1, 1, -1), a true RH↔LH handedness
# flip, and no URDF compensation is needed.

_WAREHOUSE_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="warehouse">
  <link name="walls">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}"/></geometry>
      <material name="grey"><color rgba="0.6 0.6 0.6 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}"/></geometry>
    </collision>
    <inertial>
      <mass value="0.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>
</robot>
"""

# Gate URDF applies +90° pitch (Y axis) so the procedural torus mesh
# (which is generated with axis = +Z, lying flat) stands upright with
# axis = +X. The gate's loadURDF-time orientation then rotates this
# upright torus around its base Z by the gate's yaw, putting the hole
# in the right direction for the drone to fly through.
_GATE_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="gate">
  <link name="ring">
    <visual>
      <origin xyz="0 0 0" rpy="0 1.5708 0"/>
      <geometry><mesh filename="{mesh_filename}"/></geometry>
      <material name="orange"><color rgba="1.0 0.5 0.1 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 1.5708 0"/>
      <geometry><mesh filename="{mesh_filename}"/></geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
</robot>
"""


def write_warehouse_urdf(path: Path, mesh_filename: str) -> None:
    Path(path).write_text(_WAREHOUSE_URDF_TEMPLATE.format(mesh_filename=mesh_filename))


def write_gate_urdf(path: Path, mesh_filename: str) -> None:
    Path(path).write_text(_GATE_URDF_TEMPLATE.format(mesh_filename=mesh_filename))


def write_gates_enu_json(path: Path, gates_ned: dict) -> None:
    """Convert each gate's pose NED → PyBullet world, apply gate-alignment
    correction, and write to JSON."""
    out: dict = {}
    for name, g in gates_ned.items():
        pos_enu_cyclic = ned_to_enu_position(g["position_ned"])
        quat_enu_cyclic = ned_to_enu_quaternion(g["orientation_wxyz"])
        pos_world, quat_world = _align_gate_pose(pos_enu_cyclic, quat_enu_cyclic)
        out[name] = {
            "position_enu": pos_world,
            "orientation_enu_wxyz": quat_world,
            "inner_radius_m": g["inner_radius_m"],
            "outer_radius_m": g["outer_radius_m"],
        }
    Path(path).write_text(json.dumps(out, indent=2))
