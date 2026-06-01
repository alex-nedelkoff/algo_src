import numpy as np
from aigp.protocol import Gate
from aigp.gate_projection import gate_corners_world, project_gate

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _gate(pos, w=2.0, h=2.0, quat=IDENT):
    return Gate(id=0, pos_ned=np.asarray(pos, float), quat_ned_wxyz=quat, width=w, height=h)


def test_corners_count_and_center():
    g = _gate([10.0, 0.0, 0.0])
    corners = gate_corners_world(g)
    assert corners.shape == (4, 3)
    # mean of corners == gate center
    assert np.allclose(corners.mean(axis=0), [10.0, 0.0, 0.0])


def test_project_gate_dead_ahead():
    g = _gate([10.0, 0.0, 0.0], w=2.0, h=2.0)
    out = project_gate(g, drone_pos_ned=np.zeros(3), drone_quat_wxyz=IDENT)
    assert out["in_frame"] is True
    assert np.isclose(out["center_px"][0], 320.0, atol=1.0)
    assert np.isclose(out["center_px"][1], 180.0, atol=1.0)
    u0, v0, u1, v1 = out["bbox_px"]
    assert u0 < 320.0 < u1 and v0 < 180.0 < v1   # box brackets the center
    assert np.isclose(out["range_m"], 10.0)


def test_project_gate_behind_not_in_frame():
    g = _gate([-10.0, 0.0, 0.0])
    out = project_gate(g, drone_pos_ned=np.zeros(3), drone_quat_wxyz=IDENT)
    assert out["in_frame"] is False
    assert out["center_px"] is None
