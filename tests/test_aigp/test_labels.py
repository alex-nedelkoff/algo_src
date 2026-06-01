import numpy as np
from aigp.protocol import Gate
from aigp.state import DroneState
from aigp.labels import compute_label

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def test_compute_label_dead_ahead():
    ds = DroneState(
        pos_ned=np.zeros(3), vel_ned=np.zeros(3), quat_wxyz=IDENT,
        omega=np.zeros(3), t_us=1234,
    )
    gate = Gate(id=2, pos_ned=np.array([10.0, 0.0, 0.0]),
                quat_ned_wxyz=IDENT, width=2.0, height=2.0)
    lab = compute_label(frame_idx=5, t_sim_ns=99, drone=ds, gate=gate)
    assert lab["frame"] == 5
    assert lab["t_sim_ns"] == 99
    assert lab["gate_id"] == 2
    assert lab["in_frame"] is True
    assert np.allclose(lab["drone_pos_ned"], [0, 0, 0])
    assert np.allclose(lab["gate_pos_ned"], [10, 0, 0])
    # gate dead ahead -> relative camera point is purely +z (forward)
    assert np.allclose(lab["gate_rel_cam"], [0.0, 0.0, 10.0], atol=1e-6)
    assert np.isclose(lab["center_px"][0], 320.0, atol=1.0)
    assert np.isclose(lab["range_m"], 10.0)


def test_compute_label_json_serializable():
    import json
    ds = DroneState(pos_ned=np.zeros(3), vel_ned=np.zeros(3), quat_wxyz=IDENT,
                    omega=np.zeros(3), t_us=0)
    gate = Gate(id=0, pos_ned=np.array([8.0, 1.0, -1.0]),
                quat_ned_wxyz=IDENT, width=2.0, height=2.0)
    lab = compute_label(0, 0, ds, gate)
    s = json.dumps(lab)   # must not raise
    assert isinstance(s, str)
