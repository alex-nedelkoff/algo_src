from __future__ import annotations

import json

import numpy as np
import pytest

from vq2.map_ingest import (
    anchor_warnings,
    gates_world,
    load_course_map,
)


def _valid_map():
    return {
        "version": 1,
        "frame": "spawn:x-downcourse,y-right,z-down",
        "units": "m",
        "source": "test",
        "gates": [
            {
                "id": "G2",
                "route_order": 2,
                "pos": [11.7, 5.2, -1.35],
                "normal": [0.0, 1.0, 0.0],
                "confidence": "ticked",
            },
            {
                "id": "G1",
                "route_order": 1,
                "pos": [6.3, 0.0, -1.35],
                "normal": [1.0, 0.0, 0.0],
                "confidence": "ticked",
            },
        ],
    }


def _write(tmp_path, data):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_valid_map_loads_route_ordered(tmp_path):
    gates = load_course_map(_write(tmp_path, _valid_map()))
    assert [g.id for g in gates] == ["G1", "G2"]
    assert gates[0].aperture_m == 1.5  # default
    assert gates[0].pos_sigma_m == 0.1  # ticked default
    assert anchor_warnings(gates) == []
    world = gates_world(gates)
    assert world.shape == (2, 3)
    assert np.allclose(world[1], [11.7, 5.2, -1.35])


def test_wrong_frame_is_rejected(tmp_path):
    data = _valid_map()
    data["frame"] = "world:enu"
    with pytest.raises(ValueError, match="frame"):
        load_course_map(_write(tmp_path, data))


def test_route_order_gap_is_rejected(tmp_path):
    data = _valid_map()
    data["gates"][0]["route_order"] = 3
    with pytest.raises(ValueError, match="route_order"):
        load_course_map(_write(tmp_path, data))


def test_duplicate_id_is_rejected(tmp_path):
    data = _valid_map()
    data["gates"][0]["id"] = "G1"
    with pytest.raises(ValueError, match="unique"):
        load_course_map(_write(tmp_path, data))


def test_nonunit_normal_is_rejected(tmp_path):
    data = _valid_map()
    data["gates"][0]["normal"] = [2.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="norm"):
        load_course_map(_write(tmp_path, data))


def test_nonfinite_pos_is_rejected(tmp_path):
    data = _valid_map()
    data["gates"][0]["pos"] = [float("nan"), 0.0, -1.35]
    with pytest.raises(ValueError, match="finite"):
        load_course_map(_write(tmp_path, data))


def test_disagreement_with_judge_anchor_warns(tmp_path):
    data = _valid_map()
    data["gates"][1]["pos"] = [9.0, 0.0, -1.35]  # G1, 2.7 m off the anchor
    gates = load_course_map(_write(tmp_path, data))
    warnings = anchor_warnings(gates)
    assert len(warnings) == 1 and "judge" in warnings[0]


def test_unknown_version_is_rejected(tmp_path):
    data = _valid_map()
    data["version"] = 2
    with pytest.raises(ValueError, match="version"):
        load_course_map(_write(tmp_path, data))
