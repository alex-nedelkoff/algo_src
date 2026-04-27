"""Tests for URDF and gates_enu.json writers."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.warehouse_to_urdf.urdf import (
    write_warehouse_urdf,
    write_gate_urdf,
    write_gates_enu_json,
)


def test_write_warehouse_urdf_produces_valid_xml(tmp_path: Path):
    out = tmp_path / "warehouse.urdf"
    write_warehouse_urdf(out, mesh_filename="warehouse.obj")
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag == "robot"
    link = root.find("link")
    assert link is not None
    assert link.find("collision/geometry/mesh").attrib["filename"] == "warehouse.obj"
    assert link.find("visual/geometry/mesh").attrib["filename"] == "warehouse.obj"


def test_write_gate_urdf_produces_valid_xml(tmp_path: Path):
    out = tmp_path / "gate.urdf"
    write_gate_urdf(out, mesh_filename="gate_ring.obj")
    tree = ET.parse(out)
    root = tree.getroot()
    assert root.tag == "robot"
    assert root.attrib["name"] == "gate"
    link = root.find("link")
    assert link.find("collision/geometry/mesh").attrib["filename"] == "gate_ring.obj"


def test_write_gates_enu_json_converts_ned_to_enu(tmp_path: Path):
    gates_ned = {
        "Gate_01": {
            "position_ned": [1.0, 2.0, -3.0],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "inner_radius_m": 0.75,
            "outer_radius_m": 0.85,
        },
    }
    out = tmp_path / "gates_enu.json"
    write_gates_enu_json(out, gates_ned)
    data = json.loads(out.read_text())
    g = data["Gate_01"]
    # NED (1, 2, -3) → cyclic (c,a,b) → (-3, 1, 2)
    # Then alignment M @ pos + offset where M=[[0,0,-1],[0,1,0],[-1,0,0]],
    # offset=(-3.25, 0, -3.25): (-(2) + -3.25, 1, -(-3) + -3.25) = (-5.25, 1, -0.25)
    assert g["position_enu"] == [-5.25, 1.0, -0.25]
    # Identity quaternion under similarity stays identity (M @ I @ M^T = I).
    assert g["orientation_enu_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert g["inner_radius_m"] == 0.75
