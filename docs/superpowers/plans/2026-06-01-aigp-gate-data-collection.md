# AI-GP Gate Data-Collection Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fly a controlled pattern around a race gate in the official AI-GP simulator and capture FPV frames auto-labeled with the gate's pose, producing a training dataset.

**Architecture:** A new self-contained `aigp/` package. Pure decode/geometry/guidance logic is unit-tested offline; I/O (MAVLink + vision threads), commanding, gate acquisition, and the run loop are integration-tested live against the sim. Velocity guidance (Approach C): we send NED velocity + yaw setpoints; the sim's own inner loop stabilizes. Frame convention is NED throughout; labels are produced by projecting ground-truth gate poses through the known camera model.

**Tech Stack:** Python 3.13, numpy, pymavlink 2.4.49, opencv-python — all in the `aigp` conda env. Linear issue: **COR-125**.

**Execution context (IMPORTANT):** The repo is a git worktree on the **Windows laptop** at `C:\Users\alexj\Documents\algo_src\.claude\worktrees\aigp-client` (branch `aigp-gate-data-collection`), reached over SSH (`alexj@100.120.233.90`). All test/commit commands run there in the `aigp` conda env, e.g. `conda run -n aigp pytest ...` from the worktree root. Pure-logic tasks (1–8, 12) need no sim. Integration tasks (9–11, 13–14) need the sim running and you in an active flight.

---

## File Structure

```
aigp/
  __init__.py          # package marker
  protocol.py          # Gate, parse_track_payload, parse_race_status, JpegReassembler, constants
  geometry.py          # K, IMG_W/H, quat_to_R, R_BODY_TO_CAM, world_to_camera, project, in_frame
  gate_projection.py   # gate_corners_world, project_gate -> center/bbox/in_frame/range
  guidance.py          # Setpoint, yaw_to_target, OrbitPattern, ApproachPattern
  labels.py            # compute_label (pure: state+gate -> label dict)
  state.py             # DroneState, Store (thread-safe latest telemetry + frame)
  io_layer.py          # MavlinkIO (rx thread + timesync), VisionIO (rx thread) -> Store
  commander.py         # Commander: arm, sim_reset, send_velocity_setpoint
  acquire.py           # acquire_gates(io, commander) -> list[Gate]
  logger.py            # DataLogger: writes frames/*.jpg + labels.jsonl + meta.json
  run.py               # CLI entry point wiring everything
  overlay_labels.py    # offline tool: draw projected gate on saved frames (acceptance)

tests/test_aigp/
  __init__.py
  test_protocol.py
  test_geometry.py
  test_gate_projection.py
  test_guidance.py
  test_labels.py
  test_state.py
  test_logger.py
```

---

### Task 1: Package scaffold + protocol decoders

**Files:**
- Create: `aigp/__init__.py`
- Create: `aigp/protocol.py`
- Create: `tests/test_aigp/__init__.py`
- Test: `tests/test_aigp/test_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_protocol.py
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
    assert rs["last_gate_time"] == -1


def test_parse_race_status_not_started():
    raw = struct.pack("<BQqqIq", ENCAP_RACE_STATUS, 1000, -1, -1, 0, -1)
    rs = parse_race_status(raw)
    assert rs["race_started"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp pytest tests/test_aigp/test_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/__init__.py
```

```python
# tests/test_aigp/__init__.py
```

```python
# aigp/protocol.py
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


def parse_race_status(raw: bytes) -> dict:
    """ENCAPSULATED_DATA (type 1) -> race status dict."""
    _, boot_ms, start_ms, finish_ns, gate_idx, last_t = struct.unpack_from(_RACE_FMT, raw, 0)
    return {
        "boot_ms": boot_ms,
        "active_gate_index": int(gate_idx),
        "race_started": start_ms >= 0,
        "race_start_ms": start_ms,
        "race_finish_ns": finish_ns,
        "last_gate_time": last_t,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp pytest tests/test_aigp/test_protocol.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/__init__.py aigp/protocol.py tests/test_aigp/__init__.py tests/test_aigp/test_protocol.py
git commit -m "feat(aigp): protocol decoders for track-info + race-status (COR-125)"
```

---

### Task 2: JPEG reassembler

**Files:**
- Modify: `aigp/protocol.py` (append `JpegReassembler`)
- Test: `tests/test_aigp/test_protocol.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_protocol.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp pytest tests/test_aigp/test_protocol.py -k jpeg -v`
Expected: FAIL — `ImportError: cannot import name 'JpegReassembler'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/protocol.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp pytest tests/test_aigp/test_protocol.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/protocol.py tests/test_aigp/test_protocol.py
git commit -m "feat(aigp): chunked-JPEG reassembler (COR-125)"
```

---

### Task 3: Geometry — transforms and projection

**Files:**
- Create: `aigp/geometry.py`
- Test: `tests/test_aigp/test_geometry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_geometry.py
import numpy as np
from aigp.geometry import (
    K, IMG_W, IMG_H, quat_to_R, R_BODY_TO_CAM,
    world_to_camera, project, in_frame,
)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz, body x = world north


def test_intrinsics():
    assert (IMG_W, IMG_H) == (640, 360)
    assert np.allclose(K, [[320, 0, 320], [0, 320, 180], [0, 0, 1]])


def test_quat_to_R_orthonormal():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    R = quat_to_R(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_gate_dead_ahead_projects_to_center():
    # drone at origin facing north (identity), gate 10 m north
    p_cam = world_to_camera(np.array([10.0, 0.0, 0.0]), np.zeros(3), IDENT)
    assert np.allclose(p_cam, [0.0, 0.0, 10.0])  # optical: right, down, forward
    u, v = project(p_cam)
    assert np.isclose(u, 320.0) and np.isclose(v, 180.0)
    assert in_frame(u, v)


def test_gate_to_east_projects_right():
    p_cam = world_to_camera(np.array([10.0, 2.0, 0.0]), np.zeros(3), IDENT)
    u, v = project(p_cam)
    assert u > 320.0 and np.isclose(v, 180.0)


def test_gate_below_projects_down():
    # +z is down in NED
    p_cam = world_to_camera(np.array([10.0, 0.0, 2.0]), np.zeros(3), IDENT)
    u, v = project(p_cam)
    assert np.isclose(u, 320.0) and v > 180.0


def test_gate_behind_is_none():
    p_cam = world_to_camera(np.array([-10.0, 0.0, 0.0]), np.zeros(3), IDENT)
    assert project(p_cam) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp pytest tests/test_aigp/test_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.geometry'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/geometry.py
"""Camera model + NED/body/camera transforms. Pure numpy."""
from __future__ import annotations

import numpy as np

IMG_W, IMG_H = 640, 360
# Given intrinsics: fx=fy=320, cx=320, cy=180 (HFoV 90 deg; VFoV ~58.7 deg).
K = np.array([[320.0, 0.0, 320.0],
              [0.0, 320.0, 180.0],
              [0.0, 0.0, 1.0]])

# Body frame is FRD (x-forward, y-right, z-down). Camera optical frame is
# (x-right, y-down, z-forward), looking along body +x:
#   optical_x(right)   = body_y
#   optical_y(down)    = body_z
#   optical_z(forward) = body_x
R_BODY_TO_CAM = np.array([[0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0]])


def quat_to_R(q_wxyz) -> np.ndarray:
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix (body-to-world)."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def world_to_camera(p_world_ned, drone_pos_ned, drone_quat_wxyz) -> np.ndarray:
    """Point in world NED -> camera optical frame."""
    R_bw = quat_to_R(drone_quat_wxyz)         # body -> world
    p_body = R_bw.T @ (np.asarray(p_world_ned, float) - np.asarray(drone_pos_ned, float))
    return R_BODY_TO_CAM @ p_body


def project(p_cam, intrinsics: np.ndarray = K):
    """Camera-optical point -> (u, v) pixels, or None if behind the camera."""
    z = p_cam[2]
    if z <= 1e-6:
        return None
    u = intrinsics[0, 0] * p_cam[0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * p_cam[1] / z + intrinsics[1, 2]
    return float(u), float(v)


def in_frame(u: float, v: float) -> bool:
    return 0.0 <= u < IMG_W and 0.0 <= v < IMG_H
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp pytest tests/test_aigp/test_geometry.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/geometry.py tests/test_aigp/test_geometry.py
git commit -m "feat(aigp): camera model + NED/body/camera transforms (COR-125)"
```

---

### Task 4: Gate corner projection (center + bbox)

**Files:**
- Create: `aigp/gate_projection.py`
- Test: `tests/test_aigp/test_gate_projection.py`

**Note:** Gate-local corner convention (validated later by overlay): the gate opening lies in the gate-local **y-z plane** (local x is the through-gate normal). Corners are `(0, ±w/2, ±h/2)` in gate-local coords, rotated by the gate quaternion into world NED.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_gate_projection.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp pytest tests/test_aigp/test_gate_projection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.gate_projection'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/gate_projection.py
"""Project a Gate (NED pose + size) into the camera image."""
from __future__ import annotations

import numpy as np

from .geometry import quat_to_R, world_to_camera, project, in_frame
from .protocol import Gate


def gate_corners_world(gate: Gate) -> np.ndarray:
    """4 gate-opening corners in world NED, shape (4,3).
    Opening spans the gate-local y-z plane (local x = through-gate normal)."""
    w, h = gate.width, gate.height
    local = np.array([
        [0.0, -w / 2, -h / 2],
        [0.0, +w / 2, -h / 2],
        [0.0, +w / 2, +h / 2],
        [0.0, -w / 2, +h / 2],
    ])
    R = quat_to_R(gate.quat_ned_wxyz)   # gate-local -> world
    return gate.pos_ned[None, :] + local @ R.T


def project_gate(gate: Gate, drone_pos_ned, drone_quat_wxyz) -> dict:
    """Return center pixel, bbox, in-frame flag, and range for a gate."""
    c_cam = world_to_camera(gate.pos_ned, drone_pos_ned, drone_quat_wxyz)
    rng = float(np.linalg.norm(np.asarray(gate.pos_ned) - np.asarray(drone_pos_ned)))
    center = project(c_cam)

    pts = []
    for corner in gate_corners_world(gate):
        p = project(world_to_camera(corner, drone_pos_ned, drone_quat_wxyz))
        if p is not None:
            pts.append(p)

    if center is None or not pts:
        return {"center_px": None, "bbox_px": None, "in_frame": False, "range_m": rng}

    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    bbox = (min(us), min(vs), max(us), max(vs))
    visible = in_frame(*center)
    return {"center_px": center, "bbox_px": bbox, "in_frame": visible, "range_m": rng}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp pytest tests/test_aigp/test_gate_projection.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/gate_projection.py tests/test_aigp/test_gate_projection.py
git commit -m "feat(aigp): gate corner projection (center + bbox) (COR-125)"
```

---

### Task 5: Guidance — Setpoint, yaw, OrbitPattern

**Files:**
- Create: `aigp/guidance.py`
- Test: `tests/test_aigp/test_guidance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_guidance.py
import numpy as np
from aigp.guidance import Setpoint, yaw_to_target, OrbitPattern


def test_yaw_to_target_north():
    # gate north of drone -> yaw 0
    assert np.isclose(yaw_to_target(np.zeros(3), np.array([5.0, 0.0, 0.0])), 0.0)


def test_yaw_to_target_east():
    # gate east of drone -> yaw +pi/2
    assert np.isclose(yaw_to_target(np.zeros(3), np.array([0.0, 5.0, 0.0])), np.pi / 2)


def test_orbit_on_radius_is_tangential():
    gate = np.array([0.0, 0.0, -2.0])
    radius, speed = 4.0, 2.0
    # drone 4 m north of gate, at gate height
    drone = np.array([4.0, 0.0, -2.0])
    sp = OrbitPattern(radius=radius, speed=speed, target_z=-2.0).update(
        drone, np.zeros(3), gate
    )
    # tangential (CCW) at north point is +east; radius error 0 -> no radial component
    assert np.isclose(sp.vx, 0.0, atol=1e-6)
    assert np.isclose(sp.vy, speed, atol=1e-6)
    assert np.isclose(sp.vz, 0.0, atol=1e-6)
    # yaw points back at gate (south) -> pi
    assert np.isclose(abs(sp.yaw), np.pi, atol=1e-6)


def test_orbit_too_close_pushes_outward():
    gate = np.array([0.0, 0.0, 0.0])
    drone = np.array([2.0, 0.0, 0.0])  # 2 m from gate, radius 4 -> too close
    sp = OrbitPattern(radius=4.0, speed=2.0, target_z=0.0, kp_radius=1.0).update(
        drone, np.zeros(3), gate
    )
    # radial direction is +north; being too close -> velocity has +north component
    assert sp.vx > 0.0


def test_orbit_holds_altitude():
    gate = np.array([0.0, 0.0, -2.0])
    drone = np.array([4.0, 0.0, 0.0])   # above target_z=-2 (z=0 > -2)
    sp = OrbitPattern(radius=4.0, speed=2.0, target_z=-2.0, kp_z=1.0).update(
        drone, np.zeros(3), gate
    )
    # need to descend in NED (increase z) -> vz > 0
    assert sp.vz > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp pytest tests/test_aigp/test_guidance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.guidance'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/guidance.py
"""Guidance patterns: produce NED velocity + yaw setpoints from state + gate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Setpoint:
    vx: float  # NED north velocity (m/s)
    vy: float  # NED east velocity (m/s)
    vz: float  # NED down velocity (m/s)
    yaw: float  # NED yaw (rad), 0 = north, +CW toward east


def yaw_to_target(drone_pos_ned, target_pos_ned) -> float:
    dn = target_pos_ned[0] - drone_pos_ned[0]
    de = target_pos_ned[1] - drone_pos_ned[1]
    return float(np.arctan2(de, dn))


class OrbitPattern:
    """Circle the gate at a fixed radius/height, yaw locked on the gate."""

    def __init__(self, radius=4.0, speed=2.0, target_z=-2.0, kp_radius=1.0, kp_z=1.0):
        self.radius = radius
        self.speed = speed
        self.target_z = target_z
        self.kp_radius = kp_radius
        self.kp_z = kp_z

    def update(self, drone_pos_ned, drone_vel_ned, gate_pos_ned) -> Setpoint:
        rel = np.asarray(drone_pos_ned)[:2] - np.asarray(gate_pos_ned)[:2]  # (N,E)
        dist = float(np.linalg.norm(rel))
        radial = rel / dist if dist > 1e-3 else np.array([1.0, 0.0])
        tangential = np.array([-radial[1], radial[0]])  # +90 deg = CCW
        radius_err = self.radius - dist  # >0 means too close -> push outward (+radial)
        v_h = tangential * self.speed + radial * self.kp_radius * radius_err
        vz = self.kp_z * (self.target_z - drone_pos_ned[2])
        yaw = yaw_to_target(drone_pos_ned, gate_pos_ned)
        return Setpoint(float(v_h[0]), float(v_h[1]), float(vz), yaw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp pytest tests/test_aigp/test_guidance.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/guidance.py tests/test_aigp/test_guidance.py
git commit -m "feat(aigp): Setpoint + OrbitPattern guidance (COR-125)"
```

---

### Task 6: Guidance — ApproachPattern

**Files:**
- Modify: `aigp/guidance.py` (append `ApproachPattern`)
- Test: `tests/test_aigp/test_guidance.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_aigp/test_guidance.py
from aigp.guidance import ApproachPattern


def test_approach_moves_toward_first_waypoint():
    gate = np.array([0.0, 0.0, -2.0])
    # one waypoint: 6 m north of gate
    pat = ApproachPattern(offsets=[(6.0, 0.0, 0.0)], speed=2.0, switch_dist=1.0)
    drone = np.array([0.0, 0.0, -2.0])  # 6 m south of the waypoint
    sp = pat.update(drone, np.zeros(3), gate)
    # waypoint is north -> velocity points north at ~speed
    assert sp.vx > 0.0
    assert np.isclose(np.linalg.norm([sp.vx, sp.vy, sp.vz]), 2.0, atol=1e-6)


def test_approach_advances_waypoint_when_close():
    gate = np.array([0.0, 0.0, 0.0])
    pat = ApproachPattern(
        offsets=[(1.0, 0.0, 0.0), (0.0, 5.0, 0.0)], speed=2.0, switch_dist=1.5
    )
    drone = np.array([1.0, 0.0, 0.0])  # already at waypoint 0 (within switch_dist)
    sp = pat.update(drone, np.zeros(3), gate)
    # should have advanced to waypoint 1 (east) -> velocity points east
    assert sp.vy > 0.0
    assert pat.idx == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp pytest tests/test_aigp/test_guidance.py -k approach -v`
Expected: FAIL — `ImportError: cannot import name 'ApproachPattern'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/guidance.py
class ApproachPattern:
    """Repeated runs at the gate through a list of NED offsets (relative to gate)."""

    def __init__(self, offsets, speed=2.0, switch_dist=1.5):
        self.offsets = [np.asarray(o, float) for o in offsets]
        self.speed = speed
        self.switch_dist = switch_dist
        self.idx = 0

    def _target(self, gate_pos_ned):
        return np.asarray(gate_pos_ned, float) + self.offsets[self.idx % len(self.offsets)]

    def update(self, drone_pos_ned, drone_vel_ned, gate_pos_ned) -> Setpoint:
        target = self._target(gate_pos_ned)
        rel = target - np.asarray(drone_pos_ned, float)
        if np.linalg.norm(rel) < self.switch_dist:
            self.idx += 1
            target = self._target(gate_pos_ned)
            rel = target - np.asarray(drone_pos_ned, float)
        dist = float(np.linalg.norm(rel))
        direction = rel / dist if dist > 1e-6 else np.zeros(3)
        v = direction * self.speed
        yaw = yaw_to_target(drone_pos_ned, gate_pos_ned)
        return Setpoint(float(v[0]), float(v[1]), float(v[2]), yaw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp pytest tests/test_aigp/test_guidance.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/guidance.py tests/test_aigp/test_guidance.py
git commit -m "feat(aigp): ApproachPattern guidance (COR-125)"
```

---

### Task 7: Label computation (pure)

**Files:**
- Create: `aigp/state.py` (DroneState dataclass only for now)
- Create: `aigp/labels.py`
- Test: `tests/test_aigp/test_labels.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_labels.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp pytest tests/test_aigp/test_labels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.state'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/state.py
"""Drone state container."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DroneState:
    pos_ned: np.ndarray   # (3,)
    vel_ned: np.ndarray   # (3,)
    quat_wxyz: np.ndarray  # (4,)
    omega: np.ndarray     # (3,) body angular rate
    t_us: int
```

```python
# aigp/labels.py
"""Compute a JSON-serializable training label from drone state + gate pose."""
from __future__ import annotations

import numpy as np

from .gate_projection import project_gate
from .geometry import world_to_camera
from .protocol import Gate
from .state import DroneState


def compute_label(frame_idx: int, t_sim_ns: int, drone: DroneState, gate: Gate) -> dict:
    proj = project_gate(gate, drone.pos_ned, drone.quat_wxyz)
    rel_cam = world_to_camera(gate.pos_ned, drone.pos_ned, drone.quat_wxyz)
    return {
        "frame": int(frame_idx),
        "t_sim_ns": int(t_sim_ns),
        "drone_pos_ned": [float(x) for x in drone.pos_ned],
        "drone_quat_wxyz": [float(x) for x in drone.quat_wxyz],
        "gate_id": int(gate.id),
        "gate_pos_ned": [float(x) for x in gate.pos_ned],
        "gate_quat_wxyz": [float(x) for x in gate.quat_ned_wxyz],
        "gate_rel_cam": [float(x) for x in rel_cam],
        "range_m": float(proj["range_m"]),
        "in_frame": bool(proj["in_frame"]),
        "center_px": list(proj["center_px"]) if proj["center_px"] else None,
        "bbox_px": list(proj["bbox_px"]) if proj["bbox_px"] else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp pytest tests/test_aigp/test_labels.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/state.py aigp/labels.py tests/test_aigp/test_labels.py
git commit -m "feat(aigp): pure training-label computation (COR-125)"
```

---

### Task 8: Thread-safe Store

**Files:**
- Modify: `aigp/state.py` (append `Store`)
- Test: `tests/test_aigp/test_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_state.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp pytest tests/test_aigp/test_state.py -v`
Expected: FAIL — `ImportError: cannot import name 'Store'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to aigp/state.py
import threading


class Store:
    """Thread-safe latest-value store shared between IO threads and the loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._drone = None
        self._gate_idx = 0
        self._frame = None       # (np.ndarray BGR, t_ns)
        self._frame_seq = 0
        self._gates = None

    def set_drone(self, ds: DroneState) -> None:
        with self._lock:
            self._drone = ds

    def get_drone(self):
        with self._lock:
            return self._drone

    def set_gate_idx(self, idx: int) -> None:
        with self._lock:
            self._gate_idx = int(idx)

    def get_gate_idx(self) -> int:
        with self._lock:
            return self._gate_idx

    def set_frame(self, img, t_ns: int) -> None:
        with self._lock:
            self._frame = (img, t_ns)
            self._frame_seq += 1

    def get_frame(self):
        """Returns ((img, t_ns) | None, seq)."""
        with self._lock:
            return self._frame, self._frame_seq

    def set_gates(self, gates) -> None:
        with self._lock:
            self._gates = gates

    def get_gates(self):
        with self._lock:
            return self._gates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp pytest tests/test_aigp/test_state.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/state.py tests/test_aigp/test_state.py
git commit -m "feat(aigp): thread-safe Store (COR-125)"
```

---

### Task 9: IO layer (MAVLink + vision receivers) — integration

**Files:**
- Create: `aigp/io_layer.py`

**No unit test** (pure I/O against the live sim). Validated by the smoke check below. The decode logic it calls is already tested (Tasks 1–2, 7–8).

- [ ] **Step 1: Write the implementation**

```python
# aigp/io_layer.py
"""Live MAVLink + vision receivers feeding a Store. Integration code."""
from __future__ import annotations

import os
import socket
import threading
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil

from .protocol import (
    ENCAP_RACE_STATUS, ENCAP_TRACK_INFO,
    JpegReassembler, parse_race_status, parse_track_payload,
)
from .state import DroneState, Store

VISION_PORT = 5600


class MavlinkIO:
    def __init__(self, store: Store, host: str = "0.0.0.0", port: int = 14550):
        self.store = store
        self.conn = mavutil.mavlink_connection(f"udpin:{host}:{port}")
        self._stop = threading.Event()
        self._rx = threading.Thread(target=self._rx_loop, daemon=True)
        self._ts = threading.Thread(target=self._ts_loop, daemon=True)
        self._track_chunks: dict[int, dict[int, bytes]] = {}
        self._track_expected: dict[int, int] = {}

    def wait_heartbeat(self, timeout=10):
        return self.conn.wait_heartbeat(timeout=timeout)

    def start(self):
        self._rx.start()
        self._ts.start()

    def stop(self):
        self._stop.set()

    def _ts_loop(self):
        while not self._stop.is_set():
            self.conn.mav.timesync_send(int(time.time_ns()), 0)
            time.sleep(0.1)

    def _rx_loop(self):
        while not self._stop.is_set():
            m = self.conn.recv_match(blocking=False)
            if m is None:
                time.sleep(0.001)
                continue
            t = m.get_type()
            if t == "ODOMETRY":
                self.store.set_drone(DroneState(
                    pos_ned=np.array([m.x, m.y, m.z]),
                    vel_ned=np.array([m.vx, m.vy, m.vz]),
                    quat_wxyz=np.array([m.q[0], m.q[1], m.q[2], m.q[3]]),
                    omega=np.array([m.rollspeed, m.pitchspeed, m.yawspeed]),
                    t_us=m.time_usec,
                ))
            elif t == "DATA_TRANSMISSION_HANDSHAKE":
                self._track_chunks[m.width] = {}
                self._track_expected[m.width] = m.packets
            elif t == "ENCAPSULATED_DATA":
                self._on_encap(m)

    def _on_encap(self, m):
        raw = bytes(m.data)
        if not raw:
            return
        dtype = raw[0]
        if dtype == ENCAP_RACE_STATUS:
            self.store.set_gate_idx(parse_race_status(raw)["active_gate_index"])
        elif dtype == ENCAP_TRACK_INFO:
            import struct
            _, tid = struct.unpack_from("<BH", raw)
            if tid not in self._track_expected:
                return
            self._track_chunks[tid][m.seqnr] = raw[3:]
            if len(self._track_chunks[tid]) == self._track_expected[tid]:
                payload = b"".join(
                    self._track_chunks[tid][i] for i in range(self._track_expected[tid])
                )
                self.store.set_gates(parse_track_payload(payload))


class VisionIO:
    def __init__(self, store: Store, port: int = VISION_PORT):
        self.store = store
        self.port = port
        self._stop = threading.Event()
        self._reasm = JpegReassembler()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        import cv2
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.settimeout(0.5)
        sock.bind(("0.0.0.0", self.port))
        while not self._stop.is_set():
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            out = self._reasm.add_packet(packet)
            if out is None:
                continue
            jpeg, t_ns = out
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.store.set_frame(img, t_ns)
```

- [ ] **Step 2: Smoke-check against the sim** (requires an active flight)

Run from the worktree root:
```bash
conda run -n aigp python -c "import time; from aigp.io_layer import MavlinkIO, VisionIO; from aigp.state import Store; s=Store(); m=MavlinkIO(s); print('hb', m.wait_heartbeat(10) is not None); m.start(); v=VisionIO(s); v.start(); time.sleep(3); print('drone', s.get_drone() is not None); print('frame', s.get_frame()[1] > 0)"
```
Expected: `hb True`, `drone True`, `frame True`.

- [ ] **Step 3: Commit**

```bash
git add aigp/io_layer.py
git commit -m "feat(aigp): live MAVLink + vision IO layer (COR-125)"
```

---

### Task 10: Commander — integration

**Files:**
- Create: `aigp/commander.py`

**No unit test** (sends live MAVLink). Validated within Task 13.

- [ ] **Step 1: Write the implementation**

```python
# aigp/commander.py
"""Send arm / reset / velocity-setpoint commands to the sim."""
from __future__ import annotations

import time

from pymavlink import mavutil

from .protocol import SIM_RESET_CMD

# SET_POSITION_TARGET_LOCAL_NED type_mask: ignore position, accel, yaw_rate;
# use velocity (bits 3,4,5 = 0) + yaw (bit 10 = 0).
_VEL_YAW_MASK = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


class Commander:
    def __init__(self, conn, system_boot_ms: int):
        self.conn = conn
        self.system_boot_ms = system_boot_ms

    def arm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0,
        )

    def sim_reset(self):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            SIM_RESET_CMD, 0, 0, 0, 0, 0, 0, 0, 0,
        )

    def send_velocity_setpoint(self, vx, vy, vz, yaw):
        now_ms = int(time.time() * 1000) - self.system_boot_ms
        self.conn.mav.set_position_target_local_ned_send(
            now_ms,
            self.conn.target_system, self.conn.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            _VEL_YAW_MASK,
            0.0, 0.0, 0.0,        # position (ignored)
            float(vx), float(vy), float(vz),
            0.0, 0.0, 0.0,        # accel (ignored)
            float(yaw), 0.0,      # yaw, yaw_rate (ignored)
        )
```

- [ ] **Step 2: Commit**

```bash
git add aigp/commander.py
git commit -m "feat(aigp): commander (arm, sim_reset, velocity setpoint) (COR-125)"
```

---

### Task 11: Gate acquisition — integration

**Files:**
- Create: `aigp/acquire.py`

**No unit test** (drives the live reset/broadcast). Validated within Task 13.

- [ ] **Step 1: Write the implementation**

```python
# aigp/acquire.py
"""Acquire gate poses: trigger a reset, then wait for the track broadcast."""
from __future__ import annotations

import time


def acquire_gates(store, commander, timeout=15.0, gate_cli=None):
    """Returns a list[Gate]. Strategy:
    1) if gates already cached, return them;
    2) send sim_reset and wait up to `timeout` for the broadcast;
    3) if still none and a CLI fallback gate is given, synthesize a 1-gate list.
    """
    existing = store.get_gates()
    if existing:
        return existing

    commander.sim_reset()
    deadline = time.time() + timeout
    while time.time() < deadline:
        gates = store.get_gates()
        if gates:
            return gates
        time.sleep(0.1)

    if gate_cli is not None:
        import numpy as np
        from .protocol import Gate
        return [Gate(id=0, pos_ned=np.asarray(gate_cli, float),
                     quat_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
                     width=2.0, height=2.0)]

    raise RuntimeError(
        "No track/gate broadcast received after reset. Restart the flight while "
        "connected, or pass --gate N,E,D."
    )
```

- [ ] **Step 2: Commit**

```bash
git add aigp/acquire.py
git commit -m "feat(aigp): gate acquisition via reset + broadcast (COR-125)"
```

---

### Task 12: Data logger (disk I/O)

**Files:**
- Create: `aigp/logger.py`
- Test: `tests/test_aigp/test_logger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aigp/test_logger.py
import json
import numpy as np
from aigp.protocol import Gate
from aigp.state import DroneState
from aigp.logger import DataLogger

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def test_logger_writes_frame_and_label(tmp_path):
    log = DataLogger(out_dir=tmp_path, run_id="run0", meta={"k": "v"})
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    ds = DroneState(np.zeros(3), np.zeros(3), IDENT, np.zeros(3), 0)
    gate = Gate(id=0, pos_ned=np.array([10.0, 0.0, 0.0]),
                quat_ned_wxyz=IDENT, width=2.0, height=2.0)
    log.log(img, t_sim_ns=123, drone=ds, gate=gate)
    log.close()

    frames = list((tmp_path / "run0" / "frames").glob("*.jpg"))
    assert len(frames) == 1
    lines = (tmp_path / "run0" / "labels.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    lab = json.loads(lines[0])
    assert lab["frame"] == 0 and lab["gate_id"] == 0
    meta = json.loads((tmp_path / "run0" / "meta.json").read_text())
    assert meta["k"] == "v"


def test_logger_frame_index_increments(tmp_path):
    log = DataLogger(out_dir=tmp_path, run_id="r", meta={})
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    ds = DroneState(np.zeros(3), np.zeros(3), IDENT, np.zeros(3), 0)
    gate = Gate(id=0, pos_ned=np.array([10.0, 0.0, 0.0]),
                quat_ned_wxyz=IDENT, width=2.0, height=2.0)
    log.log(img, 1, ds, gate)
    log.log(img, 2, ds, gate)
    log.close()
    lines = (tmp_path / "r" / "labels.jsonl").read_text().strip().splitlines()
    assert [json.loads(l)["frame"] for l in lines] == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n aigp pytest tests/test_aigp/test_logger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aigp.logger'`

- [ ] **Step 3: Write minimal implementation**

```python
# aigp/logger.py
"""Write captured frames + per-frame labels to disk."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from .labels import compute_label


class DataLogger:
    def __init__(self, out_dir, run_id: str, meta: dict):
        self.root = Path(out_dir) / run_id
        self.frames_dir = self.root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
        self._labels = open(self.root / "labels.jsonl", "w")
        self._idx = 0

    def log(self, img, t_sim_ns: int, drone, gate) -> None:
        lab = compute_label(self._idx, t_sim_ns, drone, gate)
        cv2.imwrite(str(self.frames_dir / f"{self._idx:06d}.jpg"), img)
        self._labels.write(json.dumps(lab) + "\n")
        self._labels.flush()
        self._idx += 1

    def close(self) -> None:
        self._labels.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n aigp pytest tests/test_aigp/test_logger.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add aigp/logger.py tests/test_aigp/test_logger.py
git commit -m "feat(aigp): data logger (frames + jsonl labels) (COR-125)"
```

---

### Task 13: Run entry point — full flight + capture (integration)

**Files:**
- Create: `aigp/run.py`

**No unit test** (the live flight). This is the primary integration milestone.

- [ ] **Step 1: Write the implementation**

```python
# aigp/run.py
"""Fly a pattern around gate 0 and capture an auto-labeled dataset."""
from __future__ import annotations

import argparse
import time

import numpy as np

from .acquire import acquire_gates
from .commander import Commander
from .guidance import ApproachPattern, OrbitPattern
from .geometry import K
from .io_layer import MavlinkIO, VisionIO
from .state import Store


def _parse_gate(s):
    if not s:
        return None
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=["orbit", "approach"], default="orbit")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--height", type=float, default=-2.0, help="target NED z (down<0=up)")
    ap.add_argument("--capture-hz", type=float, default=15.0)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--gate", default=None, help="fallback gate N,E,D if no broadcast")
    args = ap.parse_args()

    store = Store()
    mav = MavlinkIO(store)
    print("waiting for heartbeat...", flush=True)
    if mav.wait_heartbeat(10) is None:
        raise SystemExit("No heartbeat — is the sim in an active flight?")
    boot_ms = int(time.time() * 1000)
    mav.start()
    VisionIO(store).start()
    cmd = Commander(mav.conn, boot_ms)

    print("acquiring gates...", flush=True)
    gates = acquire_gates(store, cmd, gate_cli=_parse_gate(args.gate))
    gate = gates[0]
    print(f"gate 0 @ NED {gate.pos_ned}", flush=True)

    if args.pattern == "orbit":
        pattern = OrbitPattern(radius=args.radius, speed=args.speed, target_z=args.height)
    else:
        offs = [(args.radius, 0, 0), (0, args.radius, 0),
                (-args.radius, 0, 0), (0, -args.radius, 0)]
        pattern = ApproachPattern(offsets=offs, speed=args.speed)

    # lazy import to avoid hard cv2 dep in unit tests
    from .logger import DataLogger
    logger = DataLogger(args.out, args.run_id, meta={
        "K": K.tolist(), "pattern": args.pattern, "radius": args.radius,
        "speed": args.speed, "height": args.height, "gate0_ned": gate.pos_ned.tolist(),
    })

    cmd.arm()
    print("armed; flying + capturing. Ctrl-C to stop.", flush=True)

    control_dt = 1.0 / args.control_hz
    capture_dt = 1.0 / args.capture_hz
    t_end = time.time() + args.duration
    next_capture = time.time()
    last_seq = -1
    try:
        while time.time() < t_end:
            ds = store.get_drone()
            if ds is not None:
                sp = pattern.update(ds.pos_ned, ds.vel_ned, gate.pos_ned)
                cmd.send_velocity_setpoint(sp.vx, sp.vy, sp.vz, sp.yaw)

            now = time.time()
            if now >= next_capture and ds is not None:
                frame, seq = store.get_frame()
                if frame is not None and seq != last_seq:
                    img, t_ns = frame
                    logger.log(img, t_ns, ds, gate)
                    last_seq = seq
                next_capture = now + capture_dt

            time.sleep(control_dt)
    except KeyboardInterrupt:
        pass
    finally:
        logger.close()
        print(f"done. dataset at {args.out}/{args.run_id}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a short capture against the sim** (requires an active flight)

Run from the worktree root:
```bash
conda run -n aigp python -m aigp.run --pattern orbit --duration 20 --run-id smoke
```
Expected: prints heartbeat, gate 0 NED pose, "armed; flying + capturing", then "done". The drone visibly orbits the gate in the sim window. `dataset/smoke/frames/*.jpg` and `dataset/smoke/labels.jsonl` exist.

Verify output:
```bash
conda run -n aigp python -c "import json,glob; n=len(glob.glob('dataset/smoke/frames/*.jpg')); ls=[json.loads(l) for l in open('dataset/smoke/labels.jsonl')]; print('frames',n,'labels',len(ls),'in_frame',sum(x['in_frame'] for x in ls))"
```
Expected: nonzero frames, equal label count, some `in_frame=True`.

- [ ] **Step 3: Commit**

```bash
git add aigp/run.py
git commit -m "feat(aigp): run entry point — fly pattern + capture dataset (COR-125)"
```

---

### Task 14: Overlay tool — acceptance gate

**Files:**
- Create: `aigp/overlay_labels.py`

**No unit test.** This is the visual acceptance check that validates intrinsics + extrinsics + frame handling end-to-end.

- [ ] **Step 1: Write the implementation**

```python
# aigp/overlay_labels.py
"""Draw the projected gate (center + bbox) onto captured frames for visual QA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="e.g. dataset/smoke")
    ap.add_argument("--out", default=None, help="output dir (default <run_dir>/overlay)")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out) if args.out else run / "overlay"
    out.mkdir(parents=True, exist_ok=True)

    labels = [json.loads(l) for l in (run / "labels.jsonl").read_text().splitlines()]
    drawn = 0
    for lab in labels:
        if drawn >= args.limit:
            break
        img_path = run / "frames" / f"{lab['frame']:06d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if lab["center_px"]:
            u, v = int(lab["center_px"][0]), int(lab["center_px"][1])
            cv2.circle(img, (u, v), 6, (0, 0, 255), 2)
        if lab["bbox_px"]:
            u0, v0, u1, v1 = (int(x) for x in lab["bbox_px"])
            cv2.rectangle(img, (u0, v0), (u1, v1), (0, 255, 0), 2)
        cv2.putText(img, f"r={lab['range_m']:.1f}m in={lab['in_frame']}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imwrite(str(out / f"{lab['frame']:06d}.jpg"), img)
        drawn += 1
    print(f"wrote {drawn} overlays to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on the smoke dataset and inspect**

Run from the worktree root:
```bash
conda run -n aigp python -m aigp.overlay_labels dataset/smoke --limit 40
```
Then copy a few overlays back to the Mac for visual inspection (over SSH/scp). **Acceptance:** the red center dot and green bbox land on the actual gate in the FPV frames. If the projection is offset, adjust `R_BODY_TO_CAM` (camera mount) or the gate-corner convention in `gate_projection.py` and re-run.

- [ ] **Step 3: Commit**

```bash
git add aigp/overlay_labels.py
git commit -m "feat(aigp): label overlay tool + acceptance check (COR-125)"
```

---

## Self-Review

**Spec coverage:**
- IO layer → Task 9. Gate acquisition → Task 11. Guidance (orbit + approach) → Tasks 5–6. Commander → Task 10. Logger → Task 12. Geometry/projection → Tasks 3–4. Label format (full geometry + pixel) → Tasks 4, 7, 12. Frame handling → Tasks 3–4 (+ validated in 14). Error handling (no heartbeat / no track / stale frame) → Tasks 11 (RuntimeError + CLI fallback), 13 (heartbeat exit, `seq != last_seq` stale-frame skip). Overlay acceptance → Task 14. All spec sections covered.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command shows expected output.

**Type consistency:** `Gate` (fields id/pos_ned/quat_ned_wxyz/width/height) used identically in Tasks 1, 4, 7, 11, 12. `DroneState` (pos_ned/vel_ned/quat_wxyz/omega/t_us) consistent in Tasks 7, 8, 9, 12. `Setpoint` (vx/vy/vz/yaw) consistent in Tasks 5, 6, 13. `Store` methods (set/get_drone, set/get_frame returning ((img,t),seq), set/get_gate_idx, set/get_gates) consistent in Tasks 8, 9, 13. `project()`/`world_to_camera()`/`project_gate()`/`compute_label()` signatures consistent across Tasks 3, 4, 7, 12. `compute_label(frame_idx, t_sim_ns, drone, gate)` matches `DataLogger.log` usage. `Commander(conn, system_boot_ms)` matches Task 13 construction.

**Notes/assumptions to validate during execution:** (1) `SIM_RESET_CMD` re-triggers the gate broadcast — fallback is `--gate`; (2) camera mount extrinsic `R_BODY_TO_CAM` forward-aligned — validated by Task 14 overlay; (3) gate-corner convention (opening in local y-z plane) — validated by Task 14.
