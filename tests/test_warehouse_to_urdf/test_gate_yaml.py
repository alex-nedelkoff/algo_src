"""Tests for UE YAML → NED gate JSON converter (gate_yaml.py)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.warehouse_to_urdf.gate_yaml import convert_ue_yaml_to_ned_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_YAML = """\
playerstart:
  location_cm: [7580, 470, 142]
  rotation_deg_rpy: [0, 0, -90]
gates:
  Gate_01:
    location_cm: [7570, 270, 150]
    rotation_deg_rpy: [0, 0, 90]
  Gate_02:
    location_cm: [7330, -370, 120]
    rotation_deg_rpy: [0, 0, 0]
"""


def _write_yaml(tmp_path: Path, content: str, name: str = "gates.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Test 1: Schema keys are present for every gate
# ---------------------------------------------------------------------------


def test_convert_writes_expected_schema(tmp_path: Path):
    yaml_path = _write_yaml(tmp_path, _MINIMAL_YAML)
    out_json = tmp_path / "gates_ned.json"

    result = convert_ue_yaml_to_ned_json(yaml_path, out_json)

    assert out_json.exists(), "output JSON file was not created"
    data = json.loads(out_json.read_text())

    assert set(data.keys()) == {"Gate_01", "Gate_02"}, "gate names don't match YAML"

    for name, gate in data.items():
        assert "position_ned" in gate, f"{name}: missing position_ned"
        assert "orientation_wxyz" in gate, f"{name}: missing orientation_wxyz"
        assert "inner_radius_m" in gate, f"{name}: missing inner_radius_m"
        assert "outer_radius_m" in gate, f"{name}: missing outer_radius_m"
        assert len(gate["position_ned"]) == 3, f"{name}: position_ned must be length 3"
        assert len(gate["orientation_wxyz"]) == 4, f"{name}: orientation_wxyz must be length 4"

    # Return value matches file contents
    assert result == data


# ---------------------------------------------------------------------------
# Test 2: Hand-computed Gate_01 value
# ---------------------------------------------------------------------------


def test_convert_hand_computed_gate01(tmp_path: Path):
    """PlayerStart @ (7580, 470, 142) yaw=-90; Gate_01 @ (7570, 270, 150) yaw=90.

    Expected position_ned = [-0.10, +2.00, -0.08].
    """
    yaml_path = _write_yaml(tmp_path, _MINIMAL_YAML)
    out_json = tmp_path / "gates_ned.json"

    result = convert_ue_yaml_to_ned_json(yaml_path, out_json)

    pos = result["Gate_01"]["position_ned"]
    np.testing.assert_allclose(pos, [-0.10, 2.00, -0.08], atol=1e-9)

    q = result["Gate_01"]["orientation_wxyz"]
    norm = math.sqrt(sum(v**2 for v in q))
    assert abs(norm - 1.0) < 1e-6, f"Gate_01 quaternion not unit-norm: norm={norm}"


# ---------------------------------------------------------------------------
# Test 3: All 5 gates from real YAML — finite positions, unit quaternions
# ---------------------------------------------------------------------------


def test_convert_all_five_gates_from_real_yaml(tmp_path: Path):
    real_yaml = (
        Path(__file__).parents[2]
        / "configs"
        / "warehouse"
        / "warehouse_fab_v1_gates_ue.yaml"
    )
    assert real_yaml.exists(), f"Real YAML not found: {real_yaml}"

    out_json = tmp_path / "gates_ned.json"
    result = convert_ue_yaml_to_ned_json(real_yaml, out_json)

    assert len(result) == 5, f"Expected 5 gates, got {len(result)}"

    for name, gate in result.items():
        pos = gate["position_ned"]
        assert all(math.isfinite(v) for v in pos), f"{name}: non-finite position"

        q = gate["orientation_wxyz"]
        norm = math.sqrt(sum(v**2 for v in q))
        assert abs(norm - 1.0) < 1e-6, f"{name}: quaternion not unit-norm (norm={norm})"


# ---------------------------------------------------------------------------
# Test 4: Radii are hardcoded constants for every gate
# ---------------------------------------------------------------------------


def test_convert_radii_hardcoded(tmp_path: Path):
    yaml_path = _write_yaml(tmp_path, _MINIMAL_YAML)
    out_json = tmp_path / "gates_ned.json"

    result = convert_ue_yaml_to_ned_json(yaml_path, out_json)

    for name, gate in result.items():
        assert gate["inner_radius_m"] == 0.75, f"{name}: wrong inner_radius_m"
        assert gate["outer_radius_m"] == 0.85, f"{name}: wrong outer_radius_m"


# ---------------------------------------------------------------------------
# Test 5: Missing playerstart raises clear error
# ---------------------------------------------------------------------------


def test_convert_missing_playerstart_raises_clear_error(tmp_path: Path):
    yaml_no_ps = """\
gates:
  Gate_01:
    location_cm: [7570, 270, 150]
    rotation_deg_rpy: [0, 0, 90]
"""
    yaml_path = _write_yaml(tmp_path, yaml_no_ps)
    out_json = tmp_path / "gates_ned.json"

    with pytest.raises(ValueError, match="playerstart"):
        convert_ue_yaml_to_ned_json(yaml_path, out_json)


# ---------------------------------------------------------------------------
# Test 6: Gate missing rotation raises clear error
# ---------------------------------------------------------------------------


def test_convert_gate_missing_rotation_raises_clear_error(tmp_path: Path):
    yaml_bad_gate = """\
playerstart:
  location_cm: [7580, 470, 142]
  rotation_deg_rpy: [0, 0, -90]
gates:
  Gate_01:
    location_cm: [7570, 270, 150]
"""
    yaml_path = _write_yaml(tmp_path, yaml_bad_gate)
    out_json = tmp_path / "gates_ned.json"

    with pytest.raises(ValueError, match="rotation_deg_rpy"):
        convert_ue_yaml_to_ned_json(yaml_path, out_json)
