import struct
import numpy as np
from aigp.protocol import (
    Gate, parse_track_payload, parse_race_status,
    ENCAP_RACE_STATUS, ENCAP_TRACK_INFO, SIM_RESET_CMD,
)


def test_constants():
    assert ENCAP_RACE_STATUS == 1
    assert ENCAP_TRACK_INFO == 2
    assert SIM_RESET_CMD == 31000


def test_parse_track_payload_roundtrip():
    # num_gates=2, then per gate <Hfffffffff>: id,x,y,z,qw,qx,qy,qz,w,h
    payload = struct.pack("<H", 2)
    payload += struct.pack("<Hfffffffff", 0, 1.0, 2.0, -3.0, 1.0, 0.0, 0.0, 0.0, 1.5, 1.5)
    payload += struct.pack("<Hfffffffff", 1, 4.0, 5.0, -6.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0)
    gates = parse_track_payload(payload)
    assert len(gates) == 2
    assert gates[0].id == 0
    assert np.allclose(gates[0].pos_ned, [1.0, 2.0, -3.0])
    assert np.allclose(gates[0].quat_ned_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert gates[0].width == 1.5 and gates[0].height == 1.5
    assert gates[1].id == 1
    assert np.allclose(gates[1].pos_ned, [4.0, 5.0, -6.0])


def test_parse_race_status():
    raw = struct.pack("<BQqqIq", ENCAP_RACE_STATUS, 1000, 500, -1, 3, -1)
    rs = parse_race_status(raw)
    assert rs["active_gate_index"] == 3
    assert rs["race_started"] is True   # start_ms >= 0
    assert rs["race_live"] is True      # boot_ms 1000 >= start_ms 500
    assert rs["last_gate_time"] == -1


def test_parse_race_status_not_started():
    raw = struct.pack("<BQqqIq", ENCAP_RACE_STATUS, 1000, -1, -1, 0, -1)
    rs = parse_race_status(raw)
    assert rs["race_started"] is False
    assert rs["race_live"] is False


def test_is_race_live():
    from aigp.protocol import is_race_live
    assert is_race_live(boot_ms=4000, start_ms=3298) is True   # past countdown
    assert is_race_live(boot_ms=342, start_ms=3298) is False   # mid countdown (scheduled, not live)
    assert is_race_live(boot_ms=6, start_ms=-1) is False       # not scheduled yet


from aigp.protocol import JpegReassembler


def _chunk(frame_id, chunk_id, total, jpeg_size, t_ns, payload):
    header = struct.pack("<IHHIIQ", frame_id, chunk_id, total, jpeg_size, len(payload), t_ns)
    return header + payload


def test_jpeg_reassembler_completes_in_order():
    r = JpegReassembler()
    data = bytes(range(60))
    a, b, c = data[:20], data[20:40], data[40:]
    assert r.add_packet(_chunk(7, 0, 3, 60, 111, a)) is None
    assert r.add_packet(_chunk(7, 1, 3, 60, 111, b)) is None
    out = r.add_packet(_chunk(7, 2, 3, 60, 111, c))
    assert out is not None
    jpeg, t_ns = out
    assert jpeg == data
    assert t_ns == 111


def test_jpeg_reassembler_out_of_order():
    r = JpegReassembler()
    data = bytes(range(30))
    assert r.add_packet(_chunk(9, 2, 3, 30, 222, data[20:])) is None
    assert r.add_packet(_chunk(9, 0, 3, 30, 222, data[:10])) is None
    out = r.add_packet(_chunk(9, 1, 3, 30, 222, data[10:20]))
    assert out is not None and out[0] == data
