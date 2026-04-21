"""URDF and gates_enu.json writers."""
from __future__ import annotations

import json
from pathlib import Path

from sim.pybullet.coords import ned_to_enu_position, ned_to_enu_quaternion

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

_GATE_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="gate">
  <link name="ring">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_filename}"/></geometry>
      <material name="orange"><color rgba="1.0 0.5 0.1 1.0"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
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
    """Convert each gate's pose from NED to ENU, write to JSON."""
    out: dict = {}
    for name, g in gates_ned.items():
        pos_enu = ned_to_enu_position(g["position_ned"]).tolist()
        quat_enu = ned_to_enu_quaternion(g["orientation_wxyz"]).tolist()
        out[name] = {
            "position_enu": pos_enu,
            "orientation_enu_wxyz": quat_enu,
            "inner_radius_m": g["inner_radius_m"],
            "outer_radius_m": g["outer_radius_m"],
        }
    Path(path).write_text(json.dumps(out, indent=2))
