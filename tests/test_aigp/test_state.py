import numpy as np
from aigp.state import DroneState, Store

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def _ds():
    return DroneState(np.zeros(3), np.zeros(3), IDENT, np.zeros(3), 0)


def test_store_drone_roundtrip():
    s = Store()
    assert s.get_drone() is None
    ds = _ds()
    s.set_drone(ds)
    assert s.get_drone() is ds


def test_store_frame_seq_increments():
    s = Store()
    assert s.get_frame()[1] == 0
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    s.set_frame(img, 111)
    (got_img, got_t), seq = s.get_frame()
    assert seq == 1 and got_t == 111
    s.set_frame(img, 222)
    assert s.get_frame()[1] == 2


def test_store_gate_idx_and_gates():
    s = Store()
    assert s.get_gate_idx() == 0
    s.set_gate_idx(4)
    assert s.get_gate_idx() == 4
    s.set_gates(["g0", "g1"])
    assert s.get_gates() == ["g0", "g1"]
