"""Pure decoders for the AI-GP MAVLink ENCAPSULATED_DATA + vision protocols."""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# ENCAPSULATED_DATA sub-message type ids (first payload byte)
ENCAP_RACE_STATUS = 1
ENCAP_TRACK_INFO = 2

# Custom MAVLink command id to reset the sim / restart the race
SIM_RESET_CMD = 31000

_GATE_FMT = "<Hfffffffff"   # id, x,y,z, qw,qx,qy,qz, width, height
_GATE_SZ = struct.calcsize(_GATE_FMT)   # 38 bytes
_RACE_FMT = "<BQqqIq"       # dtype, boot_ms, start_ms, finish_ns, gate_idx, last_t


@dataclass
class Gate:
    id: int
    pos_ned: np.ndarray       # shape (3,)
    quat_ned_wxyz: np.ndarray  # shape (4,)
    width: float
    height: float


def parse_track_payload(payload: bytes) -> list[Gate]:
    """Reassembled track-info payload -> list of Gate (NED poses)."""
    (num,) = struct.unpack_from("<H", payload, 0)
    off = 2
    gates: list[Gate] = []
    for _ in range(num):
        gid, x, y, z, qw, qx, qy, qz, w, h = struct.unpack_from(_GATE_FMT, payload, off)
        off += _GATE_SZ
        gates.append(
            Gate(
                id=gid,
                pos_ned=np.array([x, y, z], dtype=float),
                quat_ned_wxyz=np.array([qw, qx, qy, qz], dtype=float),
                width=float(w),
                height=float(h),
            )
        )
    return gates


def is_race_live(boot_ms: int, start_ms: int) -> bool:
    """Race is LIVE (safe to actuate) once the sim clock passes the scheduled
    start. start_ms >= 0 alone only means the start is *scheduled* (countdown)."""
    return start_ms >= 0 and boot_ms >= start_ms


def parse_race_status(raw: bytes) -> dict:
    """ENCAPSULATED_DATA (type 1) -> race status dict."""
    _, boot_ms, start_ms, finish_ns, gate_idx, last_t = struct.unpack_from(_RACE_FMT, raw, 0)
    return {
        "boot_ms": boot_ms,
        "active_gate_index": int(gate_idx),
        "race_started": start_ms >= 0,
        "race_live": is_race_live(boot_ms, start_ms),
        "race_start_ms": start_ms,
        "race_finish_ns": finish_ns,
        "last_gate_time": last_t,
    }


_VISION_HDR_FMT = "<IHHIIQ"  # frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns
_VISION_HDR_SZ = struct.calcsize(_VISION_HDR_FMT)


class JpegReassembler:
    """Reassembles chunked-JPEG vision packets (UDP 5600) into full frames."""

    def __init__(self) -> None:
        self._frames: dict[int, dict] = {}

    def add_packet(self, packet: bytes):
        """Feed one UDP packet. Returns (jpeg_bytes, sim_time_ns) when a frame
        completes, else None."""
        if len(packet) < _VISION_HDR_SZ:
            return None
        fid, cid, total, _jsize, _psize, t_ns = struct.unpack_from(
            _VISION_HDR_FMT, packet, 0
        )
        payload = packet[_VISION_HDR_SZ:]
        f = self._frames.setdefault(fid, {"chunks": {}, "total": total, "t": t_ns})
        f["chunks"][cid] = payload
        if len(f["chunks"]) >= f["total"]:
            if all(i in f["chunks"] for i in range(f["total"])):
                data = b"".join(f["chunks"][i] for i in range(f["total"]))
                del self._frames[fid]
                return data, f["t"]
            del self._frames[fid]
        return None
