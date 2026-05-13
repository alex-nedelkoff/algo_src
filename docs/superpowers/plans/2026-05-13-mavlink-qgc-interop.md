# MAVLink QGroundControl Interop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `sim/pybullet/mavlink_shim/` so QGroundControl (UDP `:14550`) accepts and renders a connection — vehicle in sidebar, live attitude HUD, position/map widgets populated, no red banners.

**Architecture:** Three layers stacked on the COR-98 shim. (1) Refactor the growing inbound if/elif chain into a `dict[str, Callable]` handler table. (2) Add a fixed set of outbound senders (LOCAL_POSITION_NED, SYS_STATUS, GPS_RAW_INT, GLOBAL_POSITION_INT, HOME_POSITION, VFR_HUD) wired through the existing `RateScheduler` + `_send_message` switchboard. (3) Add inbound handlers for the QGC connect handshake (COMMAND_LONG dispatch with three known commands + UNSUPPORTED fallback, PARAM_REQUEST_LIST/READ, MISSION_REQUEST_LIST, LOG_REQUEST_LIST). Synthetic GPS origin defaults to Anduril HQ (33.6595 N, -117.9988 E) and is configurable via a `MavlinkShim` ctor kwarg.

**Tech Stack:** pymavlink (v2 dialect `ardupilotmega`), numpy. TDD throughout — every behaviour change preceded by a failing test. No production code is modified except the files this plan names.

**Spec:** `docs/superpowers/specs/2026-05-13-mavlink-qgc-interop-design.md` (committed).

**Commit-message convention:** `feat(qgc): ...` (no Linear issue yet; replace with `feat(cor-XXX):` once an issue is filed).

**Existing code touchpoints** (read before starting Phase 1):

- `sim/pybullet/mavlink_shim/shim.py` — orchestrator. Inbound dispatch lives in `step()` (lines 128-152) and `_recv_and_dispatch()` (lines 154-167). Outbound switchboard in `_send_message()` (lines 229-251). Rate-scheduler defaults in `rates_hz` field (lines 36-41).
- `sim/pybullet/mavlink_shim/server.py` — pymavlink wire layer. Existing senders: `send_heartbeat` (73), `send_attitude` (82), `send_odometry` (101), `send_highres_imu` (137), `send_timesync` (163). New senders go at the bottom of this file in the order they're added.
- `sim/pybullet/mavlink_shim/coords_mavlink.py` — NED↔ENU helpers. Standard basis change `(a,b,c)_ned ↔ (b,a,-c)_enu`. Already correct after the recent fix.
- `sim/pybullet/mavlink_shim/backend.py` — `DroneBackend` Protocol and `DroneState` dataclass (used only for type hints; do not modify).
- `tests/test_mavlink_shim/` — `test_default_rates.py`, `test_timesync.py`, `test_message_rates_match_config.py`, `test_position_target_parse.py` are the patterns to mirror. Existing `_free_port()` and `_connect_client()` helpers live in `test_timesync.py` — copy, don't import, to keep test modules independent.

**Test plumbing constraints** (project ruff config enforces these):

- Hoist **all** imports to the top of every test file. The `os.environ.setdefault("MAVLINK20", "1")` line must precede the `from pymavlink import mavutil` line, both at module top with a `# noqa: E402` on the import. The project's existing test files (see `test_timesync.py`) demonstrate the exact pattern.
- Never bind to a fixed UDP port. Use `_free_port()` from the test_timesync.py pattern (reproduced below in Task 2 — duplicated verbatim into each new test file).
- Each end-to-end shim test must wrap `shim.stop()` in a `try/finally` so a hung assertion doesn't leak the receiver thread.

**Coordinate conventions** (encoded once here, referenced from each task):

- `LOCAL_POSITION_NED` fields are NED: x=north, y=east, z=down. Convert from `DroneState.pos_enu` (x=east, y=north, z=up) with `(x_ned, y_ned, z_ned) = (pos_enu[1], pos_enu[0], -pos_enu[2])`. Identical permutation for velocity.
- `GLOBAL_POSITION_INT` lat/lon are int32 in 1e7 degrees (`int(lat_deg * 1e7)`). Altitude (MSL) and `relative_alt` are int32 in **millimetres**. Velocities `vx/vy/vz` are int16 in **cm/s** (NED).
- `GPS_RAW_INT` lat/lon same scale; altitude is int32 mm AMSL.
- `HOME_POSITION` lat/lon same scale; altitude is int32 mm; local x/y/z are float metres in NED.
- Synthetic lat/lon offset from origin uses a flat-earth approximation: `lat = origin_lat + (north_m / 111_320)`; `lon = origin_lon + (east_m / (111_320 * cos(radians(origin_lat))))`. Good to <1 m within a 1 km box, which is well beyond the warehouse scale.

---

## Phase 1 — Architectural refactor + on-demand & metadata senders

This phase keeps the demo green. After Phase 1 the shim still doesn't change the QGC user-visible behaviour, but the internal scaffolding for Phase 2 (inbound handler table) and the byte-correct wire layer for several new outbound messages exist and are tested.

### Task 1: Refactor inbound dispatch to a handler table (pure refactor)

**Files:**
- Modify: `sim/pybullet/mavlink_shim/shim.py` — replace both `step()` (lines 128-152) and `_recv_and_dispatch()` (lines 154-167) bodies with calls into a new `self._inbound_handlers` dict initialised in `__post_init__`.
- Create: `tests/test_mavlink_shim/test_inbound_dispatch.py`

This is a pure refactor: no inbound or outbound behaviour changes. The full existing suite must still pass.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mavlink_shim/test_inbound_dispatch.py`:

```python
"""Inbound handler table — refactor structural test."""
from __future__ import annotations

from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.shim import MavlinkShim


def test_shim_exposes_inbound_handler_table():
    """After __post_init__, the shim has a dict of inbound handlers covering
    the message types that COR-98 already supported, plus an unknown fallback."""
    params = VehicleParams()
    shim = MavlinkShim(backend=NumpyQuadBackend(params=params), params=params)
    handlers = shim._inbound_handlers  # internal API by design
    assert isinstance(handlers, dict)
    # COR-98 message types must be registered.
    assert "SET_ATTITUDE_TARGET" in handlers
    assert "SET_POSITION_TARGET_LOCAL_NED" in handlers
    assert "TIMESYNC" in handlers
    # All registered handlers must be callable.
    for k, fn in handlers.items():
        assert callable(fn), f"handler for {k} not callable"
    # A default unknown-message handler must exist as an attribute.
    assert hasattr(shim, "_on_unknown")
    assert callable(shim._on_unknown)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_inbound_dispatch.py -v`
Expected: FAIL — `AttributeError: 'MavlinkShim' object has no attribute '_inbound_handlers'`.

- [ ] **Step 3: Implement the handler table**

In `sim/pybullet/mavlink_shim/shim.py`:

(a) Add three private handler methods near the bottom of the class (immediately after `_send_message`):

```python
    # ----- inbound handlers (registered into self._inbound_handlers) -----

    def _on_set_attitude_target(self, m) -> None:
        self._last_target_q = np.array(m.q, dtype=np.float64)
        self._last_target_thrust = float(m.thrust)
        self._mode = "attitude"

    def _on_set_position_target(self, m) -> None:
        self._last_pos_target = parse_set_position_target_local_ned(m)
        self._mode = "position"

    def _on_timesync(self, m) -> None:
        if int(getattr(m, "tc1", 1)) == 0:
            self._server.send_timesync(tc1=time.monotonic_ns(), ts1=int(m.ts1))

    def _on_unknown(self, m) -> None:
        # Default no-op; subclasses or later phases override.
        return None
```

(b) Register the table at the end of `__post_init__`:

```python
        self._inbound_handlers: dict[str, callable] = {
            "SET_ATTITUDE_TARGET": self._on_set_attitude_target,
            "SET_POSITION_TARGET_LOCAL_NED": self._on_set_position_target,
            "TIMESYNC": self._on_timesync,
        }
```

(c) Replace the for-loop body in `step()` (lines ~133-145) with:

```python
            for m in msgs:
                t = m.get_type()
                handler = self._inbound_handlers.get(t, self._on_unknown)
                handler(m)
                if t in ("SET_ATTITUDE_TARGET", "SET_POSITION_TARGET_LOCAL_NED"):
                    got_command = True
```

(d) Replace `_recv_and_dispatch()` body with:

```python
    def _recv_and_dispatch(self) -> None:
        """Drain all pending inbound messages and route via the handler table."""
        for m in self._server.recv_pending():
            t = m.get_type()
            self._inbound_handlers.get(t, self._on_unknown)(m)
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/ -v`
Expected: every existing test plus the new structural test passes. Behaviour is unchanged.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/shim.py tests/test_mavlink_shim/test_inbound_dispatch.py
git commit -m "refactor(qgc): inbound dispatch -> handler table"
```

---

### Task 2: `LOCAL_POSITION_NED` sender + 30 Hz rate entry

**Files:**
- Modify: `sim/pybullet/mavlink_shim/server.py` — append `send_local_position_ned` after `send_timesync` (line 169).
- Modify: `sim/pybullet/mavlink_shim/shim.py` — add `"local_position_ned": 30.0` to `rates_hz` default factory; add an `elif name == "local_position_ned"` branch to `_send_message`.
- Create: `tests/test_mavlink_shim/test_local_position_ned.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mavlink_shim/test_local_position_ned.py`:

```python
"""LOCAL_POSITION_NED outbound — byte-correct ENU->NED, sent at 30 Hz."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.pybullet.mavlink_shim.server import MavlinkServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connect_client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=2, source_component=1, dialect="ardupilotmega",
    )


def _hover_state(params: VehicleParams) -> DroneState:
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.array([2.0, 5.0, 3.0]),       # east=2, north=5, up=3
        vel_enu=np.array([0.5, 1.0, -0.25]),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover),
        timestamp_us=0,
    )


def test_send_local_position_ned_emits_packet():
    """Direct server-level test: bytes match the NED conversion of the input."""
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port)
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)
        server.recv_pending()  # learn client address

        # Input: ENU (east=2, north=5, up=3), vel ENU (0.5, 1.0, -0.25).
        # Expected NED: x_n=5, y_e=2, z_d=-3, vx=1.0, vy=0.5, vz=0.25.
        server.send_local_position_ned(
            pos_enu=np.array([2.0, 5.0, 3.0]),
            vel_enu=np.array([0.5, 1.0, -0.25]),
        )

        deadline = time.time() + 1.0
        msg = None
        while time.time() < deadline:
            msg = client.recv_match(type="LOCAL_POSITION_NED", blocking=False)
            if msg is not None:
                break
            time.sleep(0.01)
        assert msg is not None, "no LOCAL_POSITION_NED received"
        assert abs(msg.x - 5.0) < 1e-6
        assert abs(msg.y - 2.0) < 1e-6
        assert abs(msg.z - (-3.0)) < 1e-6
        assert abs(msg.vx - 1.0) < 1e-6
        assert abs(msg.vy - 0.5) < 1e-6
        assert abs(msg.vz - 0.25) < 1e-6
    finally:
        server.close()


def test_local_position_ned_flows_through_shim():
    """When LOCAL_POSITION_NED is in rates_hz, the shim emits it via the scheduler."""
    params = VehicleParams()
    backend = NumpyQuadBackend(params=params)
    port = _free_port()
    shim = MavlinkShim(
        backend=backend, params=params, host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0, "local_position_ned": 30.0},
    )
    shim.reset(_hover_state(params))
    shim.start()
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)

        count = 0
        deadline = time.time() + 1.0
        while time.time() < deadline:
            msg = client.recv_match(type="LOCAL_POSITION_NED", blocking=False)
            if msg is not None:
                count += 1
            time.sleep(0.002)
        # 30 Hz over 1.0 s, allow 20% slack.
        assert count >= 24, f"expected >=24 LOCAL_POSITION_NED in 1s, got {count}"
    finally:
        shim.stop()


def test_default_rates_include_local_position_ned():
    defaults = MavlinkShim.__dataclass_fields__["rates_hz"].default_factory()
    assert defaults.get("local_position_ned") == 30.0
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_local_position_ned.py -v`
Expected: FAIL — `AttributeError: 'MavlinkServer' object has no attribute 'send_local_position_ned'`.

- [ ] **Step 3: Implement the sender + wire-through**

In `sim/pybullet/mavlink_shim/server.py`, append after `send_timesync`:

```python
    def send_local_position_ned(
        self,
        pos_enu: NDArray[np.float64],
        vel_enu: NDArray[np.float64],
    ) -> None:
        """Send LOCAL_POSITION_NED. Converts ENU -> NED at the wire boundary."""
        # ENU (east, north, up) -> NED (north, east, down).
        x = float(pos_enu[1])
        y = float(pos_enu[0])
        z = -float(pos_enu[2])
        vx = float(vel_enu[1])
        vy = float(vel_enu[0])
        vz = -float(vel_enu[2])
        self._conn.mav.local_position_ned_send(
            self._t_boot_ms(), x, y, z, vx, vy, vz,
        )
```

In `sim/pybullet/mavlink_shim/shim.py`:

(a) Add to `rates_hz` default factory (insert below `"highres_imu": 200.0,`):

```python
        "local_position_ned": 30.0,   # QGC position HUD
```

(b) Add to `_send_message` after the `highres_imu` branch:

```python
        elif name == "local_position_ned":
            self._server.send_local_position_ned(s.pos_enu, s.vel_enu)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_local_position_ned.py -v`
Expected: all three pass.

> TODO (human): `test_default_rates_match_spec` in `test_default_rates.py` will need updates as Phase 1/2 add more keys to the default `rates_hz`. Either extend that test as we go or weaken it to "≥" checks. Decide once before Task 2 lands.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/server.py sim/pybullet/mavlink_shim/shim.py tests/test_mavlink_shim/test_local_position_ned.py
git commit -m "feat(qgc): LOCAL_POSITION_NED sender @ 30 Hz"
```

---

### Task 3: `SYS_STATUS` sender + 1 Hz rate entry

**Files:**
- Modify: `sim/pybullet/mavlink_shim/server.py` — append `send_sys_status` after `send_local_position_ned`.
- Modify: `sim/pybullet/mavlink_shim/shim.py` — add `"sys_status": 1.0` to `rates_hz`; add `_send_message` branch.
- Create: `tests/test_mavlink_shim/test_sys_status.py`

The sensor flags advertise a "healthy" simulated vehicle so QGC clears its sensor-health badges. Per spec we report all-sensors-present/enabled/healthy with a 3D-gyro+3D-accel+3D-mag+absolute-pressure+GPS bitmask, battery 100% / 12000 mV / -1 current, CPU load 10% (`int16` units of 0.1%, so the field value is 100).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mavlink_shim/test_sys_status.py`:

```python
"""SYS_STATUS outbound — fixed healthy-sensor flags, 100% battery, 10% CPU."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

from sim.pybullet.mavlink_shim.server import MavlinkServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connect_client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=2, source_component=1, dialect="ardupilotmega",
    )


def test_sys_status_advertises_healthy_simulated_vehicle():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port)
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)
        server.recv_pending()

        server.send_sys_status()

        deadline = time.time() + 1.0
        msg = None
        while time.time() < deadline:
            msg = client.recv_match(type="SYS_STATUS", blocking=False)
            if msg is not None:
                break
            time.sleep(0.01)
        assert msg is not None, "no SYS_STATUS received"

        # All-sensors-healthy: present == enabled == health (any nonzero bitmask).
        assert msg.onboard_control_sensors_present > 0
        assert msg.onboard_control_sensors_present == msg.onboard_control_sensors_enabled
        assert msg.onboard_control_sensors_present == msg.onboard_control_sensors_health
        # Battery 100% (-1 = unknown is invalid here; we must report 100).
        assert msg.battery_remaining == 100
        # Voltage 12 V == 12000 mV.
        assert msg.voltage_battery == 12000
        # CPU 10%: spec field uses 0.1% units -> raw value 100.
        assert msg.load == 100
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_sys_status.py -v`
Expected: FAIL — `AttributeError: 'MavlinkServer' object has no attribute 'send_sys_status'`.

- [ ] **Step 3: Implement the sender + wire-through**

In `sim/pybullet/mavlink_shim/server.py`, append:

```python
    def send_sys_status(self) -> None:
        """Send SYS_STATUS advertising a healthy simulated vehicle.

        Sensor bitmask covers gyro, accel, mag, abs-pressure, GPS, attitude
        stabilisation, yaw position, motors. Battery is reported as fully
        charged at 12.0 V (12000 mV), -1 A (current unknown). CPU load 10%
        (the field's unit is 0.1%, so raw value 100).
        """
        ml = mavutil.mavlink
        sensors = (
            ml.MAV_SYS_STATUS_SENSOR_3D_GYRO
            | ml.MAV_SYS_STATUS_SENSOR_3D_ACCEL
            | ml.MAV_SYS_STATUS_SENSOR_3D_MAG
            | ml.MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE
            | ml.MAV_SYS_STATUS_SENSOR_GPS
            | ml.MAV_SYS_STATUS_SENSOR_ANGULAR_RATE_CONTROL
            | ml.MAV_SYS_STATUS_SENSOR_ATTITUDE_STABILIZATION
            | ml.MAV_SYS_STATUS_SENSOR_YAW_POSITION
            | ml.MAV_SYS_STATUS_SENSOR_MOTOR_OUTPUTS
        )
        self._conn.mav.sys_status_send(
            onboard_control_sensors_present=sensors,
            onboard_control_sensors_enabled=sensors,
            onboard_control_sensors_health=sensors,
            load=100,                  # 10.0% (units: 0.1%)
            voltage_battery=12000,     # mV
            current_battery=-1,        # cA, -1 = unknown
            battery_remaining=100,     # %
            drop_rate_comm=0,
            errors_comm=0,
            errors_count1=0,
            errors_count2=0,
            errors_count3=0,
            errors_count4=0,
        )
```

In `sim/pybullet/mavlink_shim/shim.py`:

(a) Add to `rates_hz` default factory:

```python
        "sys_status": 1.0,
```

(b) Add to `_send_message`:

```python
        elif name == "sys_status":
            self._server.send_sys_status()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_sys_status.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/server.py sim/pybullet/mavlink_shim/shim.py tests/test_mavlink_shim/test_sys_status.py
git commit -m "feat(qgc): SYS_STATUS sender @ 1 Hz (healthy sensors, 100% battery)"
```

---

### Task 4: `AUTOPILOT_VERSION` + `PROTOCOL_VERSION` senders (on-demand only)

**Files:**
- Modify: `sim/pybullet/mavlink_shim/server.py` — append `send_autopilot_version` and `send_protocol_version`.
- Create: `tests/test_mavlink_shim/test_capabilities_senders.py`

These are *on-demand only* — they fire from the COMMAND_LONG dispatcher added in Task 5, never from the rate scheduler. No `rates_hz` or `_send_message` change here.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mavlink_shim/test_capabilities_senders.py`:

```python
"""AUTOPILOT_VERSION and PROTOCOL_VERSION minimal-field senders."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

from sim.pybullet.mavlink_shim.server import MavlinkServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connect_client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=2, source_component=1, dialect="ardupilotmega",
    )


def _wait_for(client, msg_type: str, timeout_s: float = 1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = client.recv_match(type=msg_type, blocking=False)
        if m is not None:
            return m
        time.sleep(0.01)
    return None


def test_send_autopilot_version_emits_zero_capability_packet():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port)
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)
        server.recv_pending()

        server.send_autopilot_version()

        msg = _wait_for(client, "AUTOPILOT_VERSION")
        assert msg is not None
        # Spec: capabilities=0, all versions=0, vendor=0, product=0.
        assert msg.capabilities == 0
        assert msg.flight_sw_version == 0
        assert msg.middleware_sw_version == 0
        assert msg.vendor_id == 0
        assert msg.product_id == 0
    finally:
        server.close()


def test_send_protocol_version_emits_200_100_200():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port)
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)
        server.recv_pending()

        server.send_protocol_version()

        msg = _wait_for(client, "PROTOCOL_VERSION")
        assert msg is not None
        assert msg.version == 200
        assert msg.min_version == 100
        assert msg.max_version == 200
    finally:
        server.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_capabilities_senders.py -v`
Expected: FAIL on missing methods.

- [ ] **Step 3: Implement the senders**

In `sim/pybullet/mavlink_shim/server.py`, append:

```python
    def send_autopilot_version(self) -> None:
        """Send AUTOPILOT_VERSION advertising no extended capabilities.

        Fired only in response to MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES.
        All-zero versions/vendor/product is the QGC-tolerated 'generic
        autopilot' identity — keeps QGC from probing PX4/ArduPilot params.
        """
        zero_uid = [0] * 8     # uint8[8]
        zero_md5 = [0] * 8     # uint8[8] custom version hash
        self._conn.mav.autopilot_version_send(
            capabilities=0,
            flight_sw_version=0,
            middleware_sw_version=0,
            os_sw_version=0,
            board_version=0,
            flight_custom_version=zero_md5,
            middleware_custom_version=zero_md5,
            os_custom_version=zero_md5,
            vendor_id=0,
            product_id=0,
            uid=0,
            uid2=zero_uid,
        )

    def send_protocol_version(self) -> None:
        """Send PROTOCOL_VERSION (we only speak v2)."""
        zero_hash = [0] * 8
        self._conn.mav.protocol_version_send(
            version=200,
            min_version=100,
            max_version=200,
            spec_version_hash=zero_hash,
            library_version_hash=zero_hash,
        )
```

> TODO (human): the exact field name list for `autopilot_version_send` and `protocol_version_send` varies slightly across pymavlink versions (some have `uid2`, some don't; some require `spec_version_hash`). If the test fails on a `TypeError: unexpected keyword`, inspect the actual signature via `python -c "from pymavlink import mavutil; help(mavutil.mavlink.MAVLink.autopilot_version_send)"` and trim/add fields. The values above match pymavlink ≥2.4.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_capabilities_senders.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/server.py tests/test_mavlink_shim/test_capabilities_senders.py
git commit -m "feat(qgc): AUTOPILOT_VERSION and PROTOCOL_VERSION on-demand senders"
```

---

## Phase 2 — Inbound handlers (handshake unblocking)

Phase 2 turns the shim from "sends what we choose to send" into "responds to what QGC asks for". Each task plugs new entries into `self._inbound_handlers` from Task 1.

### Task 5: COMMAND_LONG dispatcher with three known commands + UNSUPPORTED fallback

**Files:**
- Modify: `sim/pybullet/mavlink_shim/server.py` — append `send_command_ack` helper.
- Modify: `sim/pybullet/mavlink_shim/shim.py` — add `_on_command_long` plus a `self._command_handlers` sub-table; register `"COMMAND_LONG"` in `_inbound_handlers`.
- Create: `tests/test_mavlink_shim/test_command_long_dispatch.py`

The dispatcher routes the `command` int through a sub-table. Three known handlers; everything else ACKs `MAV_RESULT_UNSUPPORTED`. Returning *any* ACK is what keeps QGC quiet.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mavlink_shim/test_command_long_dispatch.py`:

```python
"""COMMAND_LONG dispatch: three known commands + UNSUPPORTED fallback."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def _hover_state(params: VehicleParams) -> DroneState:
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover), timestamp_us=0,
    )


def _wait_for(client, msg_type: str, timeout_s: float = 1.0, **match):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = client.recv_match(type=msg_type, blocking=False)
        if m is not None and all(getattr(m, k, None) == v for k, v in match.items()):
            return m
        time.sleep(0.005)
    return None


def _send_cmd_long(client, cmd):
    client.mav.command_long_send(
        target_system=1, target_component=1,
        command=cmd, confirmation=0,
        param1=0, param2=0, param3=0, param4=0,
        param5=0, param6=0, param7=0,
    )


def _make_shim(port):
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0},
    )
    shim.reset(_hover_state(params))
    return shim


def test_request_autopilot_capabilities_sends_version_and_ack():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        _send_cmd_long(client, mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES)
        ver = _wait_for(client, "AUTOPILOT_VERSION")
        ack = _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
        )
        assert ver is not None, "no AUTOPILOT_VERSION"
        assert ack is not None and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
    finally:
        shim.stop()


def test_request_protocol_version_sends_version_and_ack():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        _send_cmd_long(client, mavutil.mavlink.MAV_CMD_REQUEST_PROTOCOL_VERSION)
        ver = _wait_for(client, "PROTOCOL_VERSION")
        ack = _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_REQUEST_PROTOCOL_VERSION,
        )
        assert ver is not None
        assert ack is not None and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
    finally:
        shim.stop()


def test_request_message_known_id_sends_message_and_ack():
    """REQUEST_MESSAGE(param1=msgid) -> we send that message + ACK ACCEPTED."""
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        # MAVLINK_MSG_ID_AUTOPILOT_VERSION = 148
        client.mav.command_long_send(
            target_system=1, target_component=1,
            command=mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, confirmation=0,
            param1=148, param2=0, param3=0, param4=0, param5=0, param6=0, param7=0,
        )
        ver = _wait_for(client, "AUTOPILOT_VERSION")
        ack = _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
        )
        assert ver is not None
        assert ack is not None and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED
    finally:
        shim.stop()


def test_unknown_command_acks_unsupported():
    """MAV_CMD_DO_SET_MODE is not in our table -> ACK UNSUPPORTED, no crash."""
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        _send_cmd_long(client, mavutil.mavlink.MAV_CMD_DO_SET_MODE)
        ack = _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        )
        assert ack is not None
        assert ack.result == mavutil.mavlink.MAV_RESULT_UNSUPPORTED
    finally:
        shim.stop()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_command_long_dispatch.py -v`
Expected: FAIL — no COMMAND_LONG handler, no `send_command_ack`.

- [ ] **Step 3: Implement**

In `sim/pybullet/mavlink_shim/server.py`, append:

```python
    def send_command_ack(self, command: int, result: int) -> None:
        """Send COMMAND_ACK in response to a COMMAND_LONG."""
        self._conn.mav.command_ack_send(int(command), int(result))
```

In `sim/pybullet/mavlink_shim/shim.py`:

(a) Add `_on_command_long` and its sub-table near the other `_on_*` handlers:

```python
    def _on_command_long(self, m) -> None:
        ml = mavutil.mavlink
        cmd = int(m.command)
        handler = self._command_handlers.get(cmd)
        if handler is None:
            self._server.send_command_ack(cmd, ml.MAV_RESULT_UNSUPPORTED)
            return
        handler(m)

    def _cmd_request_autopilot_capabilities(self, m) -> None:
        self._server.send_autopilot_version()
        self._server.send_command_ack(
            int(m.command), mavutil.mavlink.MAV_RESULT_ACCEPTED,
        )

    def _cmd_request_protocol_version(self, m) -> None:
        self._server.send_protocol_version()
        self._server.send_command_ack(
            int(m.command), mavutil.mavlink.MAV_RESULT_ACCEPTED,
        )

    def _cmd_request_message(self, m) -> None:
        """Param1 is the requested message id (MAVLINK_MSG_ID_*)."""
        ml = mavutil.mavlink
        msg_id = int(m.param1)
        result = ml.MAV_RESULT_ACCEPTED
        if msg_id == ml.MAVLINK_MSG_ID_AUTOPILOT_VERSION:
            self._server.send_autopilot_version()
        elif msg_id == ml.MAVLINK_MSG_ID_PROTOCOL_VERSION:
            self._server.send_protocol_version()
        else:
            # Unknown message id -> we still ACK (UNSUPPORTED) to satisfy the
            # COMMAND_LONG contract, but emit no message body.
            result = ml.MAV_RESULT_UNSUPPORTED
        self._server.send_command_ack(int(m.command), result)
```

(b) Add a `mavutil` import at the top of `shim.py`:

```python
from pymavlink import mavutil  # noqa: E402  -- only above the env-var guard
```

> TODO (human): `shim.py` does not currently set `MAVLINK20=1` before its mavutil import — that's done in `server.py`. Verify the import order at top of `shim.py` after this edit; if pymavlink complains, hoist the env-var-set lines to module top with `# noqa: E402` on the `mavutil` import, mirroring `server.py`'s pattern.

(c) Build the sub-table at the end of `__post_init__` (after `self._inbound_handlers`):

```python
        self._command_handlers: dict[int, callable] = {
            mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES: self._cmd_request_autopilot_capabilities,
            mavutil.mavlink.MAV_CMD_REQUEST_PROTOCOL_VERSION: self._cmd_request_protocol_version,
            mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE: self._cmd_request_message,
        }
        self._inbound_handlers["COMMAND_LONG"] = self._on_command_long
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_command_long_dispatch.py -v`
Expected: all four pass.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/server.py sim/pybullet/mavlink_shim/shim.py tests/test_mavlink_shim/test_command_long_dispatch.py
git commit -m "feat(qgc): COMMAND_LONG dispatcher (caps/proto/req-msg + UNSUPPORTED fallback)"
```

---

### Task 6: PARAM_REQUEST_LIST + PARAM_REQUEST_READ stubs

**Files:**
- Modify: `sim/pybullet/mavlink_shim/server.py` — append `send_empty_param_value`.
- Modify: `sim/pybullet/mavlink_shim/shim.py` — add `_on_param_request_list`, `_on_param_request_read`; register in `_inbound_handlers`.
- Create: `tests/test_mavlink_shim/test_param_stubs.py`

Per spec Q4, we ship an empty param list: one PARAM_VALUE with `param_count=0`, `param_index=0`, and a sentinel id of `"_EMPTY"`. PARAM_REQUEST_READ stays silent — there's nothing to return and QGC times out gracefully.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mavlink_shim/test_param_stubs.py`:

```python
"""PARAM_REQUEST_LIST -> single PARAM_VALUE(count=0); PARAM_REQUEST_READ silent."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def _hover_state(params: VehicleParams) -> DroneState:
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover), timestamp_us=0,
    )


def _make_shim(port: int) -> MavlinkShim:
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0},
    )
    shim.reset(_hover_state(params))
    return shim


def test_param_request_list_returns_single_empty_param_value():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        client.mav.param_request_list_send(
            target_system=1, target_component=1,
        )
        deadline = time.time() + 1.0
        msg = None
        while time.time() < deadline:
            m = client.recv_match(type="PARAM_VALUE", blocking=False)
            if m is not None:
                msg = m
                break
            time.sleep(0.005)
        assert msg is not None, "no PARAM_VALUE received"
        assert msg.param_count == 0
        # We send a single sentinel; spec doesn't dictate the exact id beyond
        # 'empty list signalling'. Just assert non-empty id, count=0.
        assert msg.param_count == 0
    finally:
        shim.stop()


def test_param_request_read_is_silent():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        client.mav.param_request_read_send(
            target_system=1, target_component=1,
            param_id=b"DOES_NOT_EXIST\x00\x00",
            param_index=-1,
        )
        # Drain incoming for 0.5 s; assert no PARAM_VALUE arrives.
        deadline = time.time() + 0.5
        seen = False
        while time.time() < deadline:
            m = client.recv_match(type="PARAM_VALUE", blocking=False)
            if m is not None:
                seen = True
                break
            time.sleep(0.005)
        assert not seen, "PARAM_REQUEST_READ for unknown id must not emit PARAM_VALUE"
    finally:
        shim.stop()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_param_stubs.py -v`
Expected: FAIL — no handler registered, no PARAM_VALUE arrives within 1 s.

- [ ] **Step 3: Implement**

In `sim/pybullet/mavlink_shim/server.py`, append:

```python
    def send_empty_param_value(self) -> None:
        """Send a single PARAM_VALUE with param_count=0 (empty-list signal)."""
        self._conn.mav.param_value_send(
            param_id=b"_EMPTY",
            param_value=0.0,
            param_type=mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            param_count=0,
            param_index=0,
        )
```

In `sim/pybullet/mavlink_shim/shim.py`:

(a) Add handlers:

```python
    def _on_param_request_list(self, m) -> None:
        self._server.send_empty_param_value()

    def _on_param_request_read(self, m) -> None:
        # Silent. We have no parameters to return.
        return None
```

(b) Register them in `__post_init__` after `self._inbound_handlers["COMMAND_LONG"] = ...`:

```python
        self._inbound_handlers["PARAM_REQUEST_LIST"] = self._on_param_request_list
        self._inbound_handlers["PARAM_REQUEST_READ"] = self._on_param_request_read
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_param_stubs.py -v`
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/server.py sim/pybullet/mavlink_shim/shim.py tests/test_mavlink_shim/test_param_stubs.py
git commit -m "feat(qgc): PARAM_REQUEST_LIST -> empty PARAM_VALUE; READ silent"
```

---

### Task 7: MISSION_REQUEST_LIST + LOG_REQUEST_LIST empty stubs

**Files:**
- Modify: `sim/pybullet/mavlink_shim/server.py` — append `send_empty_mission_count`, `send_empty_log_entry`.
- Modify: `sim/pybullet/mavlink_shim/shim.py` — add `_on_mission_request_list`, `_on_log_request_list`; register in `_inbound_handlers`.
- Create: `tests/test_mavlink_shim/test_mission_log_stubs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_mavlink_shim/test_mission_log_stubs.py`:

```python
"""MISSION_REQUEST_LIST -> MISSION_COUNT(0); LOG_REQUEST_LIST -> LOG_ENTRY(0)."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def _hover_state(params: VehicleParams) -> DroneState:
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover), timestamp_us=0,
    )


def _make_shim(port: int) -> MavlinkShim:
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0},
    )
    shim.reset(_hover_state(params))
    return shim


def _wait_for(client, msg_type: str, timeout_s: float = 1.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = client.recv_match(type=msg_type, blocking=False)
        if m is not None:
            return m
        time.sleep(0.005)
    return None


def test_mission_request_list_returns_count_zero():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        client.mav.mission_request_list_send(target_system=1, target_component=1)
        msg = _wait_for(client, "MISSION_COUNT")
        assert msg is not None
        assert msg.count == 0
    finally:
        shim.stop()


def test_log_request_list_returns_num_logs_zero():
    port = _free_port()
    shim = _make_shim(port)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        client.mav.log_request_list_send(
            target_system=1, target_component=1, start=0, end=0xFFFF,
        )
        msg = _wait_for(client, "LOG_ENTRY")
        assert msg is not None
        assert msg.num_logs == 0
    finally:
        shim.stop()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_mission_log_stubs.py -v`
Expected: FAIL — no MISSION_COUNT / LOG_ENTRY ever arrives.

- [ ] **Step 3: Implement**

In `sim/pybullet/mavlink_shim/server.py`, append:

```python
    def send_empty_mission_count(self) -> None:
        """Send MISSION_COUNT(0) — no mission stored."""
        self._conn.mav.mission_count_send(
            target_system=0, target_component=0, count=0,
        )

    def send_empty_log_entry(self) -> None:
        """Send LOG_ENTRY with num_logs=0 — no logs stored."""
        self._conn.mav.log_entry_send(
            id=0, num_logs=0, last_log_num=0, time_utc=0, size=0,
        )
```

> TODO (human): pymavlink ≥2.4 dialects added a `mission_type` keyword to `mission_count_send`. If `pytest` reports an unexpected-keyword error, add `mission_type=0` (MAV_MISSION_TYPE_MISSION) to the call.

In `sim/pybullet/mavlink_shim/shim.py`:

(a) Add handlers:

```python
    def _on_mission_request_list(self, m) -> None:
        self._server.send_empty_mission_count()

    def _on_log_request_list(self, m) -> None:
        self._server.send_empty_log_entry()
```

(b) Register:

```python
        self._inbound_handlers["MISSION_REQUEST_LIST"] = self._on_mission_request_list
        self._inbound_handlers["LOG_REQUEST_LIST"] = self._on_log_request_list
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_mission_log_stubs.py -v`
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/server.py sim/pybullet/mavlink_shim/shim.py tests/test_mavlink_shim/test_mission_log_stubs.py
git commit -m "feat(qgc): MISSION_REQUEST_LIST and LOG_REQUEST_LIST empty stubs"
```

---

## Phase 3 — Geographic telemetry + HUD + end-to-end verification

Phase 3 adds the remaining outbound senders that need synthetic geographic context (GPS, global position, home), the VFR HUD, and the fake-GCS end-to-end test. After Phase 3 a reviewer can attach the QGC screenshot to the PR.

### Task 8: GPS_RAW_INT + GLOBAL_POSITION_INT + HOME_POSITION + synthetic-coordinate helper

**Files:**
- Create: `sim/pybullet/mavlink_shim/geo_origin.py` — flat-earth helper.
- Modify: `sim/pybullet/mavlink_shim/server.py` — append `send_gps_raw_int`, `send_global_position_int`, `send_home_position`.
- Modify: `sim/pybullet/mavlink_shim/shim.py` — add `geo_origin_lat_lon` ctor kwarg (default Anduril HQ); add `_send_message` branches for `gps_raw_int`, `global_position_int`, `home_position`; add rate entries.
- Create: `tests/test_mavlink_shim/test_geo_origin.py`
- Create: `tests/test_mavlink_shim/test_gps_and_global_position.py`
- Create: `tests/test_mavlink_shim/test_home_position.py`

The flat-earth helper is the only piece of geographic math in the project. Keep it isolated and well-tested. Default origin = Anduril HQ Costa Mesa (33.6595 N, -117.9988 E). Reviewer can swap via `MavlinkShim(geo_origin_lat_lon=(...))`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mavlink_shim/test_geo_origin.py`:

```python
"""Flat-earth ENU->lat/lon offset helper."""
from __future__ import annotations

import math

from sim.pybullet.mavlink_shim.geo_origin import (
    ANDURIL_HQ_LAT_LON, enu_to_lat_lon,
)


def test_default_origin_is_anduril_hq():
    lat, lon = ANDURIL_HQ_LAT_LON
    assert abs(lat - 33.6595) < 1e-6
    assert abs(lon - (-117.9988)) < 1e-6


def test_origin_offset_is_origin():
    lat, lon = enu_to_lat_lon(0.0, 0.0, origin=ANDURIL_HQ_LAT_LON)
    assert lat == ANDURIL_HQ_LAT_LON[0]
    assert lon == ANDURIL_HQ_LAT_LON[1]


def test_north_offset_increases_lat():
    """100 m north -> lat increases by ~100/111320 deg."""
    lat0, lon0 = ANDURIL_HQ_LAT_LON
    lat, lon = enu_to_lat_lon(east=0.0, north=100.0, origin=ANDURIL_HQ_LAT_LON)
    expected_dlat = 100.0 / 111_320.0
    assert abs((lat - lat0) - expected_dlat) < 1e-6
    assert abs(lon - lon0) < 1e-9


def test_east_offset_scales_with_cos_lat():
    """100 m east at lat=33.66 -> dlon = 100/(111320*cos(33.66 deg))."""
    lat0, lon0 = ANDURIL_HQ_LAT_LON
    lat, lon = enu_to_lat_lon(east=100.0, north=0.0, origin=ANDURIL_HQ_LAT_LON)
    expected_dlon = 100.0 / (111_320.0 * math.cos(math.radians(lat0)))
    assert abs((lon - lon0) - expected_dlon) < 1e-6
    assert abs(lat - lat0) < 1e-9
```

Create `tests/test_mavlink_shim/test_gps_and_global_position.py`:

```python
"""GPS_RAW_INT and GLOBAL_POSITION_INT byte-correctness via the shim."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.pybullet.mavlink_shim.geo_origin import ANDURIL_HQ_LAT_LON


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def _state_at(east, north, up) -> DroneState:
    params = VehicleParams()
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.array([east, north, up]),
        vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover),
        timestamp_us=0,
    )


def _make_shim(port: int, origin=ANDURIL_HQ_LAT_LON) -> MavlinkShim:
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        rates_hz={
            "heartbeat": 2.0,
            "gps_raw_int": 1.0,
            "global_position_int": 5.0,
        },
        geo_origin_lat_lon=origin,
    )
    return shim


def _wait_for(client, msg_type: str, timeout_s: float = 2.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = client.recv_match(type=msg_type, blocking=False)
        if m is not None:
            return m
        time.sleep(0.005)
    return None


def test_gps_raw_int_reports_3d_fix_at_origin_for_zero_offset():
    port = _free_port()
    shim = _make_shim(port)
    shim.reset(_state_at(0.0, 0.0, 0.0))
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        msg = _wait_for(client, "GPS_RAW_INT")
        assert msg is not None
        # fix_type=3 (3D fix), 12 satellites.
        assert msg.fix_type == 3
        assert msg.satellites_visible == 12
        # lat/lon scaled by 1e7. At origin offset (0,0,0) we expect the
        # origin itself.
        assert msg.lat == int(ANDURIL_HQ_LAT_LON[0] * 1e7)
        assert msg.lon == int(ANDURIL_HQ_LAT_LON[1] * 1e7)
    finally:
        shim.stop()


def test_global_position_int_offsets_lat_lon_by_local_position():
    port = _free_port()
    shim = _make_shim(port)
    # 100 m north, 0 east, 5 m up.
    shim.reset(_state_at(east=0.0, north=100.0, up=5.0))
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)
        msg = _wait_for(client, "GLOBAL_POSITION_INT")
        assert msg is not None
        # +100 m north -> lat increases by ~100/111320 deg, scaled by 1e7.
        expected_dlat_e7 = int((100.0 / 111_320.0) * 1e7)
        actual_dlat_e7 = msg.lat - int(ANDURIL_HQ_LAT_LON[0] * 1e7)
        # tolerance: 1 int7 unit = ~1.1 cm.
        assert abs(actual_dlat_e7 - expected_dlat_e7) <= 2
        # alt and relative_alt in mm.
        assert msg.relative_alt == 5000
    finally:
        shim.stop()
```

Create `tests/test_mavlink_shim/test_home_position.py`:

```python
"""HOME_POSITION emitted once after the first GCS HEARTBEAT arrives."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.pybullet.mavlink_shim.geo_origin import ANDURIL_HQ_LAT_LON


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(port):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def _hover_state():
    params = VehicleParams()
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover), timestamp_us=0,
    )


def test_home_position_emitted_after_first_gcs_heartbeat():
    port = _free_port()
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port, rates_hz={"heartbeat": 2.0},
    )
    shim.reset(_hover_state())
    shim.start()
    try:
        client = _client(port)
        # First HEARTBEAT from the GCS -> shim should send one HOME_POSITION.
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        deadline = time.time() + 2.0
        msg = None
        while time.time() < deadline:
            m = client.recv_match(type="HOME_POSITION", blocking=False)
            if m is not None:
                msg = m
                break
            time.sleep(0.005)
        assert msg is not None, "no HOME_POSITION received"
        assert msg.latitude == int(ANDURIL_HQ_LAT_LON[0] * 1e7)
        assert msg.longitude == int(ANDURIL_HQ_LAT_LON[1] * 1e7)
        # Local NED (x,y,z) origin.
        assert abs(msg.x) < 1e-6
        assert abs(msg.y) < 1e-6
        assert abs(msg.z) < 1e-6
    finally:
        shim.stop()


def test_home_position_not_emitted_repeatedly():
    """A second client HEARTBEAT should NOT trigger another HOME_POSITION."""
    port = _free_port()
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port, rates_hz={"heartbeat": 2.0},
    )
    shim.reset(_hover_state())
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        # Drain the first HOME_POSITION.
        time.sleep(0.5)
        seen = 0
        while True:
            m = client.recv_match(type="HOME_POSITION", blocking=False)
            if m is None:
                break
            seen += 1
        assert seen >= 1
        # Second heartbeat after a delay.
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        deadline = time.time() + 0.5
        seen2 = 0
        while time.time() < deadline:
            m = client.recv_match(type="HOME_POSITION", blocking=False)
            if m is not None:
                seen2 += 1
            time.sleep(0.005)
        assert seen2 == 0, f"HOME_POSITION re-emitted ({seen2}) on repeat heartbeat"
    finally:
        shim.stop()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_geo_origin.py tests/test_mavlink_shim/test_gps_and_global_position.py tests/test_mavlink_shim/test_home_position.py -v`
Expected: FAIL across the board — module not found, ctor kwarg missing, etc.

- [ ] **Step 3: Implement**

Create `sim/pybullet/mavlink_shim/geo_origin.py`:

```python
"""Flat-earth ENU->lat/lon helper for synthetic GPS in the QGC shim.

Used only to give QGroundControl a map location to render. The simulator
has no real geographic position (VADR-TS-002 says 'GPS simulation is not
available'); this is a presentation-layer fiction.

Approximation: 1 deg lat ~= 111_320 m everywhere. 1 deg lon scales by
cos(lat). Good to < 1 m within a 1 km radius — far beyond the warehouse
operating volume.
"""
from __future__ import annotations

import math


# Anduril HQ, Costa Mesa, CA. Chosen because (a) recognisable, (b) clearly
# fictional for sim purposes, (c) makes the QGC map look populated.
ANDURIL_HQ_LAT_LON: tuple[float, float] = (33.6595, -117.9988)

_M_PER_DEG_LAT = 111_320.0


def enu_to_lat_lon(
    east: float, north: float, origin: tuple[float, float],
) -> tuple[float, float]:
    """Project an ENU (east, north) offset onto lat/lon, flat-earth."""
    origin_lat, origin_lon = origin
    dlat = north / _M_PER_DEG_LAT
    dlon = east / (_M_PER_DEG_LAT * math.cos(math.radians(origin_lat)))
    return origin_lat + dlat, origin_lon + dlon
```

In `sim/pybullet/mavlink_shim/server.py`, append:

```python
    def send_gps_raw_int(
        self,
        lat_deg: float, lon_deg: float, alt_m_amsl: float,
    ) -> None:
        """Stubbed GPS_RAW_INT: 3D fix, 12 sats, lat/lon from caller."""
        ml = mavutil.mavlink
        self._conn.mav.gps_raw_int_send(
            time_usec=self._t_usec(),
            fix_type=ml.GPS_FIX_TYPE_3D_FIX,
            lat=int(lat_deg * 1e7),
            lon=int(lon_deg * 1e7),
            alt=int(alt_m_amsl * 1000),
            eph=100, epv=100, vel=0, cog=0,
            satellites_visible=12,
        )

    def send_global_position_int(
        self,
        lat_deg: float, lon_deg: float,
        alt_m_amsl: float, relative_alt_m: float,
        vel_ned_m_s,  # length-3 array-like
        heading_deg: float,
    ) -> None:
        """GLOBAL_POSITION_INT — lat/lon in 1e7 deg, alt in mm, vel in cm/s."""
        self._conn.mav.global_position_int_send(
            time_boot_ms=self._t_boot_ms(),
            lat=int(lat_deg * 1e7),
            lon=int(lon_deg * 1e7),
            alt=int(alt_m_amsl * 1000),
            relative_alt=int(relative_alt_m * 1000),
            vx=int(vel_ned_m_s[0] * 100),
            vy=int(vel_ned_m_s[1] * 100),
            vz=int(vel_ned_m_s[2] * 100),
            hdg=int(heading_deg * 100),
        )

    def send_home_position(
        self, lat_deg: float, lon_deg: float, alt_m_amsl: float,
    ) -> None:
        """HOME_POSITION at the supplied lat/lon; local NED set to origin."""
        zero_q = [1.0, 0.0, 0.0, 0.0]
        self._conn.mav.home_position_send(
            latitude=int(lat_deg * 1e7),
            longitude=int(lon_deg * 1e7),
            altitude=int(alt_m_amsl * 1000),
            x=0.0, y=0.0, z=0.0,
            q=zero_q,
            approach_x=0.0, approach_y=0.0, approach_z=0.0,
            time_usec=self._t_usec(),
        )
```

> TODO (human): `home_position_send` in older pymavlink versions does not take `time_usec`. If pytest complains, drop the kwarg.

In `sim/pybullet/mavlink_shim/shim.py`:

(a) Add the ctor kwarg + import:

```python
from sim.pybullet.mavlink_shim.geo_origin import ANDURIL_HQ_LAT_LON, enu_to_lat_lon
```

```python
    geo_origin_lat_lon: tuple[float, float] = field(
        default_factory=lambda: ANDURIL_HQ_LAT_LON
    )
    _home_sent: bool = field(init=False, default=False)
```

(b) Extend the default `rates_hz`:

```python
        "gps_raw_int": 1.0,
        "global_position_int": 5.0,
```

(c) Add a helper and three `_send_message` branches:

```python
    def _current_lat_lon_alt(self) -> tuple[float, float, float]:
        s = self._last_state
        lat, lon = enu_to_lat_lon(
            east=float(s.pos_enu[0]), north=float(s.pos_enu[1]),
            origin=self.geo_origin_lat_lon,
        )
        alt_amsl_m = float(s.pos_enu[2])  # treat origin as MSL=0 (sim fiction)
        return lat, lon, alt_amsl_m
```

```python
        elif name == "gps_raw_int":
            lat, lon, alt = self._current_lat_lon_alt()
            self._server.send_gps_raw_int(lat, lon, alt)
        elif name == "global_position_int":
            lat, lon, alt = self._current_lat_lon_alt()
            # vel_ned = (north, east, down) = (vy_enu, vx_enu, -vz_enu)
            v = s.vel_enu
            vel_ned = (float(v[1]), float(v[0]), -float(v[2]))
            self._server.send_global_position_int(
                lat_deg=lat, lon_deg=lon,
                alt_m_amsl=alt, relative_alt_m=float(s.pos_enu[2]),
                vel_ned_m_s=vel_ned, heading_deg=0.0,
            )
        elif name == "home_position":
            # On-demand only (see _on_unknown -> first heartbeat trigger below).
            origin_lat, origin_lon = self.geo_origin_lat_lon
            self._server.send_home_position(origin_lat, origin_lon, 0.0)
```

(d) Trigger HOME_POSITION exactly once on the first GCS heartbeat. Replace `_on_unknown` from Task 1 with:

```python
    def _on_unknown(self, m) -> None:
        # We register HEARTBEAT as 'unknown' deliberately: it has no payload
        # we care about beyond 'a GCS exists'. First time we see one, fire a
        # HOME_POSITION so QGC's map widget initialises.
        if m.get_type() == "HEARTBEAT" and not self._home_sent:
            self._send_message("home_position")
            self._home_sent = True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_geo_origin.py tests/test_mavlink_shim/test_gps_and_global_position.py tests/test_mavlink_shim/test_home_position.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/geo_origin.py sim/pybullet/mavlink_shim/server.py sim/pybullet/mavlink_shim/shim.py tests/test_mavlink_shim/test_geo_origin.py tests/test_mavlink_shim/test_gps_and_global_position.py tests/test_mavlink_shim/test_home_position.py
git commit -m "feat(qgc): GPS_RAW_INT + GLOBAL_POSITION_INT + HOME_POSITION (synthetic geo origin)"
```

---

### Task 9: `VFR_HUD` sender @ 10 Hz

**Files:**
- Modify: `sim/pybullet/mavlink_shim/server.py` — append `send_vfr_hud`.
- Modify: `sim/pybullet/mavlink_shim/shim.py` — add `"vfr_hud": 10.0` rate entry; add `_send_message` branch deriving fields from `DroneState`.
- Create: `tests/test_mavlink_shim/test_vfr_hud.py`

Field derivation (per spec table): `airspeed = |vel_enu|`, `groundspeed = |vel_enu[:2]|`, `heading = enu_quat -> NED yaw deg`, `throttle = 50` (stub), `alt = pos_enu.z`, `climb = vel_enu.z`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mavlink_shim/test_vfr_hud.py`:

```python
"""VFR_HUD derived fields: airspeed = |vel|, climb = vel_enu.z."""
from __future__ import annotations

import math
import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(port):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def test_vfr_hud_derives_airspeed_and_climb():
    port = _free_port()
    params = VehicleParams()
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    state = DroneState(
        pos_enu=np.array([0.0, 0.0, 5.0]),
        vel_enu=np.array([3.0, 4.0, 1.5]),       # |v|=sqrt(9+16+2.25)=5.22, climb=1.5
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover),
        timestamp_us=0,
    )
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0, "vfr_hud": 10.0},
    )
    shim.reset(state)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.2)

        deadline = time.time() + 2.0
        msg = None
        while time.time() < deadline:
            m = client.recv_match(type="VFR_HUD", blocking=False)
            if m is not None:
                msg = m
                break
            time.sleep(0.005)
        assert msg is not None
        # The shim stepped the backend before sending, so vel/pos drift slightly.
        # Compare against |reported_vel| with 0.5 m/s tolerance.
        expected_air = math.sqrt(3.0**2 + 4.0**2 + 1.5**2)
        assert abs(msg.airspeed - expected_air) < 0.5
        # Climb = vel_enu.z; tolerance same.
        assert abs(msg.climb - 1.5) < 0.5
    finally:
        shim.stop()


def test_vfr_hud_default_rate_in_defaults():
    defaults = MavlinkShim.__dataclass_fields__["rates_hz"].default_factory()
    assert defaults.get("vfr_hud") == 10.0
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_vfr_hud.py -v`
Expected: FAIL — no VFR_HUD arrives, default factory missing the key.

- [ ] **Step 3: Implement**

In `sim/pybullet/mavlink_shim/server.py`, append:

```python
    def send_vfr_hud(
        self,
        airspeed: float, groundspeed: float, heading_deg: float,
        throttle: int, alt_m: float, climb: float,
    ) -> None:
        self._conn.mav.vfr_hud_send(
            airspeed=float(airspeed),
            groundspeed=float(groundspeed),
            heading=int(heading_deg),
            throttle=int(throttle),
            alt=float(alt_m),
            climb=float(climb),
        )
```

In `sim/pybullet/mavlink_shim/shim.py`:

(a) Add to `rates_hz` default factory:

```python
        "vfr_hud": 10.0,
```

(b) Add `_send_message` branch:

```python
        elif name == "vfr_hud":
            v = s.vel_enu
            air = float(np.linalg.norm(v))
            ground = float(np.linalg.norm(v[:2]))
            _, _, yaw_ned = enu_quat_to_ned_euler(s.quat_wxyz)
            heading_deg = (math.degrees(yaw_ned) + 360.0) % 360.0
            self._server.send_vfr_hud(
                airspeed=air, groundspeed=ground,
                heading_deg=heading_deg, throttle=50,
                alt_m=float(s.pos_enu[2]), climb=float(v[2]),
            )
```

(c) Add `import math` to the imports section near the top of `shim.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_vfr_hud.py -v`
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add sim/pybullet/mavlink_shim/server.py sim/pybullet/mavlink_shim/shim.py tests/test_mavlink_shim/test_vfr_hud.py
git commit -m "feat(qgc): VFR_HUD @ 10 Hz (airspeed, climb, heading)"
```

---

### Task 10: End-to-end fake-GCS handshake test

**Files:**
- Create: `tests/test_mavlink_shim/test_qgc_handshake.py`

No production-code change. This is the integration gate: it replays QGroundControl's actual connect-time message sequence against the shim and asserts each expected response arrives in order, plus that the rate-scheduled telemetry is flowing within ±20% of nominal.

- [ ] **Step 1: Write the test**

Create `tests/test_mavlink_shim/test_qgc_handshake.py`:

```python
"""End-to-end fake-GCS handshake against the MavlinkShim.

Replays the QGroundControl connect sequence in order and asserts each
expected response arrives within 1 s. Then samples the rate-scheduled
telemetry for 2 s and asserts the per-message rates land within +/-20%
of nominal.

This is the protocol-correctness gate. The manual QGC screenshot in
Task 11 covers the UI side.
"""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(port):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def _hover():
    params = VehicleParams()
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover), timestamp_us=0,
    )


def _wait_for(client, msg_type: str, timeout_s: float = 1.0, **match):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        m = client.recv_match(type=msg_type, blocking=False)
        if m is not None and all(getattr(m, k, None) == v for k, v in match.items()):
            return m
        time.sleep(0.005)
    return None


def test_qgc_handshake_sequence():
    port = _free_port()
    params = VehicleParams()
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        # All defaults: heartbeat, attitude, highres_imu, timesync,
        # local_position_ned, sys_status, vfr_hud, gps_raw_int,
        # global_position_int.
    )
    shim.reset(_hover())
    shim.start()
    try:
        client = _client(port)

        # Step 1: GCS heartbeat -> shim notes the client, fires HOME_POSITION.
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        assert _wait_for(client, "HOME_POSITION") is not None

        # Step 2: REQUEST_AUTOPILOT_CAPABILITIES.
        client.mav.command_long_send(
            target_system=1, target_component=1,
            command=mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
            confirmation=0,
            param1=0, param2=0, param3=0, param4=0,
            param5=0, param6=0, param7=0,
        )
        assert _wait_for(client, "AUTOPILOT_VERSION") is not None
        assert _wait_for(
            client, "COMMAND_ACK",
            command=mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES,
        ) is not None

        # Step 3: PARAM_REQUEST_LIST.
        client.mav.param_request_list_send(target_system=1, target_component=1)
        pv = _wait_for(client, "PARAM_VALUE")
        assert pv is not None and pv.param_count == 0

        # Step 4: MISSION_REQUEST_LIST.
        client.mav.mission_request_list_send(target_system=1, target_component=1)
        mc = _wait_for(client, "MISSION_COUNT")
        assert mc is not None and mc.count == 0

        # Step 5: Unknown COMMAND_LONG -> UNSUPPORTED ACK.
        client.mav.command_long_send(
            target_system=1, target_component=1,
            command=mavutil.mavlink.MAV_CMD_DO_SET_MODE, confirmation=0,
            param1=0, param2=0, param3=0, param4=0,
            param5=0, param6=0, param7=0,
        )
        ack = _wait_for(
            client, "COMMAND_ACK", command=mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        )
        assert ack is not None
        assert ack.result == mavutil.mavlink.MAV_RESULT_UNSUPPORTED

        # Step 6: Telemetry sample over 2 s. Count per message type and
        # assert >= 80% of nominal.
        counts = {
            "HEARTBEAT": 0,
            "ATTITUDE": 0,
            "HIGHRES_IMU": 0,
            "LOCAL_POSITION_NED": 0,
            "SYS_STATUS": 0,
            "GPS_RAW_INT": 0,
            "GLOBAL_POSITION_INT": 0,
            "VFR_HUD": 0,
        }
        end = time.time() + 2.0
        while time.time() < end:
            m = client.recv_match(blocking=False)
            if m is not None and m.get_type() in counts:
                counts[m.get_type()] += 1
            else:
                time.sleep(0.001)

        nominal = {
            "HEARTBEAT": 2.0,
            "ATTITUDE": 100.0,
            "HIGHRES_IMU": 200.0,
            "LOCAL_POSITION_NED": 30.0,
            "SYS_STATUS": 1.0,
            "GPS_RAW_INT": 1.0,
            "GLOBAL_POSITION_INT": 5.0,
            "VFR_HUD": 10.0,
        }
        for k, hz in nominal.items():
            expected = hz * 2.0  # window = 2 s
            min_ok = max(1, int(expected * 0.6))
            assert counts[k] >= min_ok, (
                f"{k}: got {counts[k]}, expected ~{int(expected)} (>= {min_ok})"
            )
    finally:
        shim.stop()
```

- [ ] **Step 2: Run to verify the handshake passes end-to-end**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/test_qgc_handshake.py -v`
Expected: PASS. If any step times out, the failing assertion identifies which inbound/outbound link is broken.

- [ ] **Step 3: Run the full mavlink-shim suite**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/ -v`
Expected: every test (existing COR-98 set plus the new Phase 1-3 tests) passes.

> TODO (human): If `test_default_rates.py` was weakened in Task 2's TODO above, revisit and pin the exact expected default keys here (heartbeat, attitude, highres_imu, timesync, local_position_ned, sys_status, vfr_hud, gps_raw_int, global_position_int). The set is now stable.

- [ ] **Step 4: Commit**

```bash
git add tests/test_mavlink_shim/test_qgc_handshake.py
git commit -m "test(qgc): end-to-end fake-GCS handshake + telemetry-rate verification"
```

- [ ] **Step 5: Documentation-only — no further commit**

This step has no `Step 5: Commit` because the work is already on disk; Task 11 (the screenshot gate) is the user-visible deliverable.

---

### Task 11: Manual QGroundControl screenshot (documentation only — gates the PR)

**Files:**
- No code change.
- Attach screenshot to the PR description.

This is the user-visible gate for the merge. The automated test in Task 10 covers the protocol; this covers QGC's interpretation. No commit, no source edit.

- [ ] **Step 1: Run the smoke test on the laptop with QGC installed**

```bash
PYTHONPATH=. python scripts/mavlink/smoke_test.py --backend pybullet --mode attitude --hold
```

(`--hold` keeps the shim running after the pitch sweep finishes. If that flag doesn't exist, add a trivial one — but per the constraint "do not touch other docs/scripts in this plan," fall back to running the script without it and rely on the sweep being long enough.)

> TODO (human): verify the existing `smoke_test.py` supports a long-running mode (or add a `--hold` flag in a separate change before merging this PR). The plan deliberately does not modify smoke_test.py.

- [ ] **Step 2: Open QGroundControl on the same machine**

Configure UDP listener on port 14550 (QGC default). Wait <=3 s for the vehicle to appear in the sidebar.

- [ ] **Step 3: Capture the screenshot**

The screenshot must show:

- Vehicle present in the sidebar with no-error status indicator.
- Live attitude indicator tracking the pitch sweep (>= +-30 deg deflection).
- Position widget showing live altitude updating.
- Map showing the drone at the Anduril HQ default origin (or whatever `geo_origin_lat_lon` was overridden to).
- No persistent red banner toasts after the connect handshake completes.

- [ ] **Step 4: Attach to the PR description**

Embed the screenshot in the PR body under a "QGC Screenshot" heading. Reviewer will not merge without it.

---

## Final verification

- [ ] **Step 1: Full test suite green**

Run: `PYTHONPATH=. pytest tests/test_mavlink_shim/ -v`
Expected: every test passes. Total test count should be the COR-98 baseline plus the eleven new test modules (`test_inbound_dispatch`, `test_local_position_ned`, `test_sys_status`, `test_capabilities_senders`, `test_command_long_dispatch`, `test_param_stubs`, `test_mission_log_stubs`, `test_geo_origin`, `test_gps_and_global_position`, `test_home_position`, `test_vfr_hud`, `test_qgc_handshake`).

- [ ] **Step 2: External telemetry sniff (sanity check before QGC)**

Run the smoke test in one terminal. In another, use the sniff snippet from the COR-98 plan's Final Verification step (counts per message type over 3 s) and confirm:

- HEARTBEAT >= 6 (2 Hz * 3 s)
- ATTITUDE ~300 (100 Hz)
- HIGHRES_IMU ~600 (200 Hz)
- LOCAL_POSITION_NED ~90 (30 Hz)
- TIMESYNC ~30 (10 Hz)
- VFR_HUD ~30 (10 Hz)
- GLOBAL_POSITION_INT ~15 (5 Hz)
- SYS_STATUS ~3 (1 Hz)
- GPS_RAW_INT ~3 (1 Hz)
- HOME_POSITION 1 (one-shot)
- ODOMETRY 0 (dropped in COR-98)

- [ ] **Step 3: QGC screenshot attached to PR**

The Task 11 screenshot is the merge gate. PR description must include it.

## Coverage self-review

Spec sections vs tasks:

- Layer 1 row LOCAL_POSITION_NED -> Task 2.
- Layer 1 row GLOBAL_POSITION_INT -> Task 8.
- Layer 1 row GPS_RAW_INT -> Task 8.
- Layer 1 row SYS_STATUS -> Task 3.
- Layer 1 row VFR_HUD -> Task 9.
- Layer 1 row HOME_POSITION (one-shot) -> Task 8 (`_on_unknown` heartbeat trigger).
- Layer 2 row PARAM_REQUEST_LIST -> Task 6.
- Layer 2 row PARAM_REQUEST_READ silent -> Task 6.
- Layer 2 row MISSION_REQUEST_LIST -> Task 7.
- Layer 2 row LOG_REQUEST_LIST -> Task 7.
- Layer 2 row COMMAND_LONG dispatch table -> Task 5 (REQUEST_AUTOPILOT_CAPABILITIES, REQUEST_PROTOCOL_VERSION, REQUEST_MESSAGE) plus Task 4 (the underlying senders).
- Layer 3 architectural refactor -> Task 1.
- Verification > Automated -> Task 10.
- Verification > Manual -> Task 11.
- Open question Q2 (geo origin / Anduril HQ default, configurable) -> Task 8.

## Open TODOs flagged inline (recap for the reviewer)

- Task 2: decide how to keep `test_default_rates.py` in sync as keys are added across Phase 1-3 (extend in lockstep vs weaken to `>=` checks).
- Task 4: pymavlink `autopilot_version_send` / `protocol_version_send` field names vary by version — verify against the installed pymavlink and trim/add as needed.
- Task 5: confirm whether `shim.py` needs the `MAVLINK20=1` env-var guard before its new `from pymavlink import mavutil` import; mirror `server.py` pattern if so.
- Task 7: pymavlink `mission_count_send` may need a `mission_type=0` kwarg on newer dialects.
- Task 8: `home_position_send` `time_usec` kwarg presence varies — drop if pymavlink complains.
- Task 11: confirm `scripts/mavlink/smoke_test.py` has (or needs) a `--hold` flag for the manual QGC screenshot session.

## Out-of-scope reminders

Per spec "Out-of-spec follow-ups", these stay out of this PR:

- TRPYMixer square-law fix + MoE retrain.
- Vision UDP stream (spec §4.6).
- Real-PX4 SITL interop test.
- Yaw-convention audit across `coords_mavlink.py` vs PX4.
- Position-mode z steady-state error in real-time mode.
