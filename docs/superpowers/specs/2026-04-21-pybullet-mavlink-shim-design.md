# PyBullet MAVLink Shim Design

**Date:** 2026-04-21
**Status:** Draft
**Author:** Alex + Claude
**Related:** COR-79 (DCL VQ uses MAVLink interface), COR-88 (E2E exploration uses ATTITUDE+ODOMETRY)

## Scope

A reusable MAVLink server (the "shim") that wraps a drone dynamics backend and exposes it as a standard UDP MAVLink endpoint. Lets us develop control/policy code against a MAVLink interface today, mirroring how the DCL competition sim will look. When DCL access lands, switching is a transport address change, not a code refactor.

Phase 1 ships:
- Core shim (UDP server, message scheduler, attitude controller, lifecycle)
- One backend: `NumpyQuadBackend` wrapping the existing `sim/dynamics/numpy_quad.py`
- Tests + smoke test demonstrating SET_ATTITUDE_TARGET → ATTITUDE/ODOMETRY/HIGHRES_IMU loop closure

Out of scope (future):
- Additional backends (PyBullet collision-aware drone-in-warehouse, real-drone passthrough)
- Body-rate-only mode for SET_ATTITUDE_TARGET (attitude-only mode in v1)
- IMU magnetometer / barometer fields
- Multi-vehicle (single-vehicle in v1)
- Full PID with integral term (PD only in v1)

## Why this exists

The DCL VQ competition sim uses MAVLink. Cosys-AirSim (the warehouse binary we have) only exposes MAVLink via PX4 SITL or ArduPilot SITL — adding a flight stack that obscures the actual MAVLink message flow we care about. PX4 SITL on Windows requires WSL2; ArduPilot SITL is workable but adds an unrelated control stack to debug.

Building a small MAVLink server on top of our existing pybullet-warehouse / numpy_quad work gives us:
- A pure MAVLink interface our policy code talks to from day one
- Zero new infrastructure (no PX4, no UE5, no AirSim runtime)
- Reusable abstraction: same client code drives pybullet (dev), eventually pybullet+warehouse-collision, eventually DCL

## Architecture

```
┌────────────────────────────────────────────────────┐
│  MAVLink CLIENT (any process — pymavlink, QGC,     │
│  policy code, pytest, ...)                         │
│           │                          ▲             │
│           ▼ SET_ATTITUDE_TARGET      │ ATTITUDE,   │
│                                      │ ODOMETRY,   │
│                                      │ HIGHRES_IMU │
└──────── UDP localhost:14550 (configurable) ───────┘
           │                          │
   ┌───────┼──────────────────────────┼────────────┐
   │       ▼                          │            │
   │  ┌──────────────┐         ┌─────────────┐    │
   │  │ MavlinkServer│         │ MessagePump │    │
   │  │ (pymavlink   │         │ (timer or   │    │
   │  │  UDP socket) │         │  lockstep)  │    │
   │  └──────────────┘         └─────────────┘    │
   │       │                          ▲            │
   │       ▼                          │            │
   │  ┌──────────────────────────────────┐        │
   │  │ AttitudeController               │        │
   │  │ (P on quat error → desired ω,    │        │
   │  │  TRPY mixer → motor speeds)      │        │
   │  └──────────────────────────────────┘        │
   │       │                          ▲            │
   │       ▼ motor cmd          state │            │
   │  ┌──────────────────────────────────┐        │
   │  │ DroneBackend (Protocol)          │        │
   │  │   - reset(initial_state)         │        │
   │  │   - step(motor_cmd, dt) → state  │        │
   │  │   - get_imu() → ImuSample        │        │
   │  └──────────────────────────────────┘        │
   │             │                                 │
   │             ▼ initial backend                 │
   │   ┌──────────────────────────┐               │
   │   │ NumpyQuadBackend         │               │
   │   │ (wraps sim/dynamics/     │               │
   │   │  numpy_quad.py, single   │               │
   │   │  vehicle subset)         │               │
   │   └──────────────────────────┘               │
   │                                                │
   │   sim/pybullet/mavlink_shim/  (Python pkg)    │
   └────────────────────────────────────────────────┘
```

**Key choices:**
- `DroneBackend` is a Protocol — phase 1 has one impl, future impls plug in without shim changes.
- `AttitudeController` is internal to the shim. From the client's POV, "the sim" includes the attitude controller (matches DCL's expected behavior).
- Single MAVLink server class owns the UDP socket and message pump.
- Two run modes (free-running default, lockstep for tests) controlled by a constructor flag.
- Package lives at `sim/pybullet/mavlink_shim/`. Despite the "pybullet" parent namespace, the shim itself doesn't import pybullet (NumpyQuadBackend is pure NumPy). Future PyBullet-using backends will live alongside.

## Components

### `DroneBackend` interface

```python
@dataclass
class DroneState:
    pos_enu: np.ndarray            # (3,) world position, ENU
    vel_enu: np.ndarray            # (3,) world velocity, ENU
    quat_wxyz: np.ndarray          # (4,) attitude in ENU world frame
    angular_vel_body: np.ndarray   # (3,) body-frame rotation rates
    motor_speed: np.ndarray        # (4,) per-motor angular velocity
    timestamp_us: int              # monotonic microseconds since shim start

@dataclass
class ImuSample:
    accel_body: np.ndarray         # (3,) body-frame accel including gravity, m/s²
    gyro_body: np.ndarray          # (3,) body-frame angular rate, rad/s
    timestamp_us: int

class DroneBackend(Protocol):
    def reset(self, initial_state: DroneState) -> None: ...
    def step(self, motor_commands: np.ndarray, dt: float) -> DroneState: ...
    def get_imu(self) -> ImuSample: ...
```

`motor_commands`: 4-vector of per-motor speeds (rad/s), produced by the TRPY mixer and consumed directly by the numpy_quad dynamics. Matches the existing `sim/dynamics/trpy_mixer.py` output convention.

### `NumpyQuadBackend`

Thin wrapper around `sim/dynamics/numpy_quad.py` for a single vehicle (N=1 in the vectorized arrays). Composes:
- `numpy_quad` for state evolution given motor commands
- `sim/sensors/imu_model.py` (if present) or a basic noise-free IMU model derived from the dynamics for `get_imu()`

Backend stays in ENU; NED conversion is done in the MAVLink server layer.

### `AttitudeController`

Single-loop attitude PD → desired body rates, then TRPY mixer → motor speeds.

```
SET_ATTITUDE_TARGET (q_target, thrust)         DroneState (q_current)
        │                                              │
        └────────────┬─────────────────────────────────┘
                     ▼
         Attitude error → desired body rates (P)
         q_error = q_target ⊗ q_current⁻¹
         ω_desired = K_att · 2 · sign(q_error.w) · q_error.xyz
                     │
                     ▼
         TRPY mixer (existing sim/dynamics/trpy_mixer.py)
         (thrust, ω_desired) → 4 motor speeds (rad/s)
                     │
                     ▼
              backend.step(motor_speeds, dt)
```

Module: `sim/pybullet/mavlink_shim/attitude_controller.py`, ~60 LOC (smaller than originally estimated since the inner rate loop is unneeded — the TRPY mixer's inverted allocation matrix maps desired rates directly to motor speeds).

- **Attitude mode only in v1** (type_mask body-rate bits ignored).
- **Thrust normalization:** `SET_ATTITUDE_TARGET.thrust ∈ [0, 1]` → physical collective thrust force. Default mapping `thrust = max_thrust * thrust_normalized` with `max_thrust` derived from `VehicleParams` (4 motors × per-motor max thrust). Configurable via constructor.
- **Default gains:** `K_att` calibrated to produce stable step response with the `sim/dynamics/params.py` defaults. Smoke test verifies.
- **No inner rate PD:** TRPY mixer's inverted allocation matrix maps desired rates → motor speeds directly. For our perfect-model sim this is sufficient. Adding a rate-error PD with torque output would require either (a) modifying the mixer to accept torques instead of rates, or (b) computing torque externally then converting back to rates — neither is YAGNI for v1.
- **Quaternion sign handling:** `2 · sign(q_error.w) · q_error.xyz` ensures the controller takes the shorter rotation path (avoids 180° flips).

### `MavlinkServer` + `MessagePump`

Owns the UDP socket (pymavlink `mavutil.mavlink_connection("udpin:host:port")`) and the per-message rate scheduler.

Single thread with non-blocking I/O. Each tick:
1. Drain pending UDP messages (non-blocking `recv_match`).
2. Forward `SET_ATTITUDE_TARGET` payloads to the controller as the latest setpoint.
3. Advance physics (`backend.step(motor_cmd, dt)`).
4. Emit due outbound messages based on per-message rate accumulators.
5. Sleep until next tick (free-running) or block on next inbound (lockstep).

## Message Set

**Dialect:** `common.xml`, no extensions.

**System / component IDs:** sysid=1, compid=1 (`MAV_COMP_ID_AUTOPILOT1`). Configurable for multi-shim setups.

**Outbound (defaults shown, all configurable):**

| Message | Default rate | Purpose | Notes |
|---------|--------------|---------|-------|
| `HEARTBEAT` (0) | 1 Hz | Standard liveness | type=`MAV_TYPE_QUADROTOR`, autopilot=`MAV_AUTOPILOT_GENERIC` |
| `ATTITUDE` (30) | 100 Hz | roll/pitch/yaw + rates | Body Euler in NED |
| `ODOMETRY` (331) | 100 Hz | Pose + velocity | `frame_id=MAV_FRAME_LOCAL_NED`, `child_frame_id=MAV_FRAME_BODY_FRD` |
| `HIGHRES_IMU` (105) | 200 Hz | accel + gyro | `fields_updated` indicates only accel+gyro present (no mag/baro v1) |

**Inbound:**

| Message | Handling |
|---------|----------|
| `SET_ATTITUDE_TARGET` (82) | Forward `q[4]` and `thrust` to AttitudeController. `type_mask` rate-only bits ignored. |
| `HEARTBEAT` (0) | Track client liveness. Optional — shim streams unconditionally. |
| All others | Dropped, logged DEBUG. |

**Coordinate frames:** Backend ENU; conversion at server boundary using existing `sim/pybullet/coords.py` helpers. ATTITUDE message is body Euler relative to NED.

**Rate scheduler:** single tick at the highest configured rate (default 200 Hz for IMU). Per-message accumulators determine due-time.

## Modes + Lifecycle

**Free-running** (default):
- Physics steps on a wall-clock timer at the highest configured outbound rate.
- Last-received SET_ATTITUDE_TARGET is sticky.
- Outbound messages stream at per-message rates.

**Lockstep:**
- Physics advances only on receipt of SET_ATTITUDE_TARGET (or after timeout).
- Exactly one physics step per command. Outbound flushes after the step.
- Used for deterministic tests.

**Threading:** single thread, non-blocking I/O.
- Free-running: tick loop in worker thread, `start()` spawns it.
- Lockstep: no autonomous thread. Caller drives via `shim.step(timeout_s)`.

**Lifecycle:**

```python
shim = MavlinkShim(
    backend=NumpyQuadBackend(...),
    host="0.0.0.0",
    port=14550,
    lockstep=False,
    rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
    controller_gains=...,
)

# Free-running
with shim:
    shim.wait_for_shutdown()

# Lockstep
shim.start()
shim.reset(initial_state)
for _ in range(N):
    shim.step(timeout_s=0.1)
shim.stop()
```

**Reset:** `reset(initial_state)` calls `backend.reset(...)`, zeroes controller state and message timers, sends a HEARTBEAT immediately.

**Shutdown:** `stop()` flushes outbound, closes socket, joins worker thread (free-running) or no-op (lockstep). Idempotent.

**Error handling:**
- Malformed inbound MAVLink → log warning, drop, continue.
- UDP socket send error → log error, continue (next tick retries).
- Backend raises → propagate up. We don't recover from physics faults.
- Controller produces NaN motor commands → log error, send all-zero command for safety, continue.

## Validation

### Test pyramid

**Unit tests** (CI):

| Test | Asserts |
|------|---------|
| `test_attitude_controller.py` | PD math correct; identity error → zero torque; 30° pitch error → torque on pitch axis with expected sign+magnitude |
| `test_rate_scheduler.py` | Configured rates → due-message counts match within ±1 over N seconds of ticks |
| `test_coord_boundary.py` | ENU state ↔ NED message round-trips via `coords.py` helpers |
| `test_backend_protocol.py` | `MavlinkShim` calls `backend.reset/step/get_imu` correctly via `MockBackend` |

**Integration tests** (lockstep, deterministic, CI):

| Test | Setup | Assertion |
|------|-------|-----------|
| `test_hover_hold` | identity attitude + hover thrust × 500 steps @ 200 Hz | final pos within 5 cm of initial; |attitude error| < 1° |
| `test_attitude_step_response` | 30° pitch step, run 1s | reaches 25-30° pitch within 0.5s, settles in 1s |
| `test_reset_returns_to_initial` | step 100 commands, reset, check state | post-reset state == initial exactly |
| `test_message_rates_match_config` | 5s free-running + pymavlink client capture | rates within ±5% of expected |

**Smoke test / demo** (`scripts/mavlink/smoke_test.py`, manual):

Free-running shim + pymavlink client. Send sequence: hover 1s → +30° pitch 1s → hover 1s → -30° pitch 1s → hover 1s. Record ATTITUDE responses. Output PNG/HTML showing commanded vs achieved attitude over time.

**Bonus manual check:** point QGroundControl at `udp:localhost:14550`, confirm vehicle appears with non-zero telemetry. If QGC sees us as a normal vehicle, our MAVLink is conformant enough for any standard tool — including DCL.

### Combined exit criterion

Spec is "done" when:
1. All unit + integration tests pass in CI (CPU-only, no GPU needed).
2. `scripts/mavlink/smoke_test.py` produces an attitude tracking plot with commanded ≈ achieved within 5° steady-state error.
3. QGroundControl manual handshake succeeds (one-line confirmation in the implementation issue).

## Dependencies

**Already installed:**
- `pymavlink` (just added to `monorace` env)
- `numpy`, `pytest`, `pyyaml`

**No new dependencies for the shim itself.**

For the smoke test plot: `matplotlib` (already a transitive dep of the project — used by training viz).

For the QGroundControl handshake (manual): QGroundControl installed locally, but not a project dependency.

## File Layout

```
sim/pybullet/mavlink_shim/
    __init__.py              # public API: MavlinkShim, DroneBackend, DroneState, ImuSample
    backend.py               # DroneBackend Protocol + dataclasses
    numpy_quad_backend.py    # NumpyQuadBackend impl
    attitude_controller.py   # cascaded PD + TRPY mixer call
    server.py                # MavlinkServer + MessagePump (UDP, scheduler)
    shim.py                  # MavlinkShim top-level orchestrator + lifecycle
    coords_mavlink.py        # NED↔ENU adapters specific to MAVLink message conventions

scripts/mavlink/
    smoke_test.py            # manual demo + plot generation

tests/test_mavlink_shim/
    __init__.py
    test_attitude_controller.py
    test_rate_scheduler.py
    test_coord_boundary.py
    test_backend_protocol.py
    test_hover_hold.py
    test_attitude_step_response.py
    test_reset_returns_to_initial.py
    test_message_rates_match_config.py
```

## Open Risks

| Risk | Mitigation |
|------|------------|
| Default attitude controller gains produce unstable response on some param sets | Smoke test verifies step response; gains are tunable via constructor. Calibrate once against `sim/dynamics/params.py` defaults. |
| Single-thread tick loop misses deadlines under load | Loop is timing-tolerant via per-message accumulators (no drift); the rate test sets ±5% bound. If we exceed that, switch IMU rate to 100 Hz or move sender to a separate thread. |
| pymavlink dialect mismatch with DCL | We use `common.xml`. If DCL uses a custom dialect, swap dialect at the connection layer (one-line change). Most production MAVLink stacks accept `common`. |
| ENU↔NED conversion bug (especially the ATTITUDE Euler derivation) | Unit-tested via `test_coord_boundary.py`; QGC manual check would surface gross errors. |
| Lockstep mode races with the message pump | Lockstep skips the worker thread entirely; caller drives single-threaded. No race window. |

## Out of Scope (future specs)

- `PybulletWarehouseBackend`: drone-in-warehouse using pybullet for collision against the warehouse mesh. Becomes relevant when phase 3 (gym-pybullet-drones training loop) starts.
- Body-rate mode for SET_ATTITUDE_TARGET (1-day extension to AttitudeController).
- Multi-vehicle support (vectorized backend, sysid demux).
- Magnetometer / barometer fields in HIGHRES_IMU.
- Full PID with integral term.
- MAVLink message logging / replay tools.
- DCL-specific dialect adapters (pending DCL access).
