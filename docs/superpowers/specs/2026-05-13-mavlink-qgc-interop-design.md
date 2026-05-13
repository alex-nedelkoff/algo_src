# MAVLink QGroundControl Interop — Design Spec

**Date:** 2026-05-13
**Status:** Draft (awaiting user approval before plan)
**Predecessor:** [2026-05-12-mavlink-pybullet-spec-compliance-design.md](2026-05-12-mavlink-pybullet-spec-compliance-design.md) (COR-98)
**Related issues:** TBD (Linear)

## Problem

The MavlinkShim implemented in COR-98 sends the spec-mandated message set (HEARTBEAT, ATTITUDE, HIGHRES_IMU, TIMESYNC) and accepts SET_ATTITUDE_TARGET / SET_POSITION_TARGET_LOCAL_NED. It has been validated end-to-end via pymavlink-loopback unit tests and a smoke-test PNG showing clean ±30° pitch tracking on PyBullet.

It has **not** been validated against an off-the-shelf MAVLink GCS. QGroundControl (the standard) needs more than the spec-mandated minimum to render a useful HUD: position telemetry, GPS-fix indication, system-status bars, autopilot-capabilities advertisement, and minimal responses to its connect-time handshake (param list, mission list, log list). Without these, QGC either:

- Doesn't show the vehicle (missing capabilities response),
- Shows the vehicle with persistent red banners (un-acked COMMAND_LONG, no param list),
- Shows the vehicle with empty HUD widgets (no LOCAL_POSITION_NED, no SYS_STATUS).

This spec scopes the *minimum-viable* additions to make QGroundControl accept and render a connection from MavlinkShim cleanly enough that a screenshot would convince a reviewer the protocol layer works against real GCS software, not just our own loopback tests.

## Goal

A reviewer running QGroundControl can point it at `udp:127.0.0.1:14550` and observe:

1. Vehicle appears in QGC sidebar within 3 s of starting the shim.
2. Telemetry HUD shows live attitude updating in real time and matching the smoke-test pitch sweep.
3. Position widgets populate with live local-frame altitude / position.
4. Map shows the drone at a fixed reference point (origin chosen below).
5. No persistent red banners or warning toasts after the connect handshake completes.

The work is *demo-quality*. The shim is still a sim, not a flight controller — arming, mode-switching, and mission upload are explicit non-goals.

## Non-goals

- Mission upload/download (stub `MISSION_COUNT=0` response only)
- Parameter writes (read-only stub list — empty per design decision below)
- Camera / gimbal / log-download MAVLink protocols
- Acting as a real PX4 or ArduPilot (different work — option (b) in the prior discussion: testing *against* a real flight controller, not pretending to be one)
- Arming / mode / takeoff state machine (we're a sim backend)
- Any feature that requires QGC to *control* the vehicle beyond what the smoke test already drives via SET_ATTITUDE_TARGET / SET_POSITION_TARGET_LOCAL_NED

## Approach

Three layers of additions, each independently testable:

### Layer 1 — Outbound message additions

Currently sent: HEARTBEAT (2 Hz), ATTITUDE (100 Hz), HIGHRES_IMU (200 Hz), TIMESYNC (10 Hz).

Add to the rate scheduler:

| Message | Rate | Source | Why QGC wants it |
|---|---|---|---|
| `LOCAL_POSITION_NED` | 30 Hz | DroneState.pos_enu/vel_enu (NED converted) | Altitude + local position HUD |
| `GLOBAL_POSITION_INT` | 5 Hz | Synthetic lat/lon from fixed reference + local NED offset | Map view |
| `GPS_RAW_INT` | 1 Hz | Stub: `fix_type=3` (3D), `satellites_visible=12`, lat/lon from same reference | Clears "No GPS" badge |
| `SYS_STATUS` | 1 Hz | Stub: all sensors `enabled=health=present`, battery 100%, CPU 10% | Battery/sensor health widgets |
| `VFR_HUD` | 10 Hz | Derived from DroneState (airspeed = `\|vel\|`, throttle = stub, climb = `vel_enu.z`) | HUD widget |
| `HOME_POSITION` | One-shot on first GCS HEARTBEAT | Same fixed reference as GPS | Map origin |

**Note on GPS:** VADR-TS-002 §3.7 explicitly states "GPS simulation is not available and absolute global position is not exposed." The competition stack itself will not consume GPS. We invent a fixed reference point *only* so QGC's map renders something — it has no functional role in the simulation. See open question Q1 below.

`ODOMETRY` (currently disabled) stays disabled; `LOCAL_POSITION_NED` covers QGC's needs and is more standard.

### Layer 2 — Inbound message handling

Currently handled: SET_ATTITUDE_TARGET, SET_POSITION_TARGET_LOCAL_NED, TIMESYNC.

Add handlers:

| Inbound | Response |
|---|---|
| `PARAM_REQUEST_LIST` | Single `PARAM_VALUE` with `param_count=0` (empty list per design decision) |
| `PARAM_REQUEST_READ` | Silent (no entry to return; QGC times out gracefully) |
| `MISSION_REQUEST_LIST` | `MISSION_COUNT(count=0)` |
| `LOG_REQUEST_LIST` | `LOG_ENTRY` with `num_logs=0` |
| `COMMAND_LONG` | Dispatched on `command` field — see table below |

`COMMAND_LONG` dispatch table (anything not listed → `COMMAND_ACK(MAV_RESULT_UNSUPPORTED)`):

| Command | Action | Then ACK |
|---|---|---|
| `MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES` | Send `AUTOPILOT_VERSION` (capabilities=0, version=0, vendor_id=0, product_id=0) | `MAV_RESULT_ACCEPTED` |
| `MAV_CMD_REQUEST_PROTOCOL_VERSION` | Send `PROTOCOL_VERSION` (version=200, min=100, max=200) | `MAV_RESULT_ACCEPTED` |
| `MAV_CMD_REQUEST_MESSAGE` | Send the requested message (look up by message ID in our outbound senders) | `MAV_RESULT_ACCEPTED` if known, `MAV_RESULT_UNSUPPORTED` otherwise |

Returning `UNSUPPORTED` (instead of silence) is what prevents QGC's red-banner timeouts.

### Layer 3 — Architectural cleanup

The current `MavlinkShim.step` and `_recv_and_dispatch` both contain growing if/elif chains for inbound handling. Refactor to a single handler table:

```python
self._inbound_handlers: dict[str, Callable[[Any], None]] = {
    "SET_ATTITUDE_TARGET": self._on_set_attitude_target,
    "SET_POSITION_TARGET_LOCAL_NED": self._on_set_position_target,
    "TIMESYNC": self._on_timesync,
    "PARAM_REQUEST_LIST": self._on_param_request_list,
    # ...
}
```

Both `step` and `_recv_and_dispatch` invoke `self._inbound_handlers.get(t, self._on_unknown)(m)`. New messages add one entry, no fork.

Outbound is already extensible via `RateScheduler` + `_send_message`; just extend the rate dict.

## Verification

Two layers, both required:

### Automated (CI-runnable, no GUI)

`tests/test_mavlink_shim/test_qgc_handshake.py` — spawn the shim, connect with `mavutil.mavlink_connection(...)` as a fake GCS client, perform the QGC connect sequence in order:

1. Send HEARTBEAT.
2. Send `COMMAND_LONG(MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES)`. Assert `AUTOPILOT_VERSION` arrives within 1 s and `COMMAND_ACK(ACCEPTED)` arrives within 1 s.
3. Send `PARAM_REQUEST_LIST`. Assert `PARAM_VALUE(param_count=0)` arrives within 1 s.
4. Send `MISSION_REQUEST_LIST`. Assert `MISSION_COUNT(count=0)` arrives within 1 s.
5. Send `COMMAND_LONG(MAV_CMD_DO_SET_MODE)` (anything we don't support). Assert `COMMAND_ACK(UNSUPPORTED)`.
6. Wait 2 s; assert `LOCAL_POSITION_NED`, `SYS_STATUS`, `GPS_RAW_INT` are all flowing at their expected rates (within ±20% of nominal).

This catches every protocol regression without needing a GUI.

### Manual (one-time, gates the PR)

1. Run `python -m scripts.mavlink.smoke_test --backend pybullet --mode attitude --hold` on the target laptop.
2. Open QGroundControl on the same machine (or networked).
3. Configure UDP listener on port 14550 (QGC default).
4. Wait ≤3 s for the vehicle to appear in the sidebar.
5. Take screenshot showing:
   - Vehicle in sidebar with non-error status indicator
   - Live attitude indicator matching the smoke-test pitch sweep (the ±30° steps)
   - Position widget showing live altitude
   - Map showing the drone at the fixed reference point
   - No persistent red error banners
6. Attach screenshot to the PR description.

The screenshot is the gate. Reviewer doesn't merge without it.

## Implementation chunks (for the plan doc)

In dependency order, each TDD-friendly:

1. **Refactor inbound dispatch** to handler table. Pure refactor, no behavior change. All 55 existing tests pass.
2. **`LOCAL_POSITION_NED` sender** + byte-correctness test.
3. **`SYS_STATUS` sender** + test (battery + sensor flags).
4. **`AUTOPILOT_VERSION` + `PROTOCOL_VERSION` senders** (on-demand only, not periodic) + tests.
5. **`COMMAND_LONG` dispatcher** with the three handled commands + UNSUPPORTED fallback + tests for each.
6. **PARAM/MISSION/LOG empty-list stubs** + tests.
7. **`GPS_RAW_INT` + `GLOBAL_POSITION_INT` + `HOME_POSITION` senders** + synthetic-coordinate helper + tests.
8. **`VFR_HUD` sender** + test.
9. **End-to-end fake-GCS handshake test** (`test_qgc_handshake.py`).
10. **Manual QGC screenshot** + PR documentation.

## Open questions (answered, baked in below)

**Q1: Autopilot identity in HEARTBEAT.** *Answered: stay with `MAV_AUTOPILOT_GENERIC`.* This means QGC won't probe for PX4-specific or ArduPilot-specific params, which is appropriate given our empty param list.

**Q2: Reference point for GPS / map.** *Answered: warehouse origin.* Since VADR-TS-002 explicitly states "GPS simulation is not available," the warehouse local origin (0,0,0) NED has no defined geographic location. We need to invent one purely for QGC's map rendering. **Proposed default: 33.6595°N, -117.9988°E (Anduril HQ, Costa Mesa CA)** — recognizable, makes the map look sensible, and clearly fictional for sim purposes. Configurable via a `MavlinkShim` constructor parameter `geo_origin_lat_lon` defaulting to that pair. *If you want a different point (e.g., the actual competition venue when announced), it's one constructor argument.*

**Q3: CI manual-screenshot gate.** *Answered: yes, screenshot.* The automated `test_qgc_handshake.py` covers the protocol; the screenshot covers the QGC interpretation. Both required for the PR.

**Q4: Param stub scope.** *Answered: empty list.* Single `PARAM_VALUE` response with `param_count=0`. If QGC complains too loudly in practice we can expand later, but starting empty keeps scope minimal.

## Out-of-spec follow-ups (file as separate work)

- **TRPYMixer square-law fix** + MoE retrain (carried over from COR-98)
- **Vision UDP stream** (spec §4.6 — separate sub-project)
- **Real-PX4 SITL interop test** (option (b) from the prior discussion — requires confirming the competition architecture: are we the autopilot or the offboard companion?)
- **Yaw-convention audit** between `coords_mavlink.py` and PX4 conventions (carried over)
- **Position-mode z steady-state error** in real-time mode (separate dynamics issue)
