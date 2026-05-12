# MAVLink Shim Spec-Compliance + PyBullet Backend

## Overview

The `sim/pybullet/mavlink_shim/` package gives our racing stack a competition-shaped interface: contestants drive a backend over MAVLink-UDP, get telemetry back. The COR-94 MVP shipped against the `NumpyQuadDynamics` backend with a partial MAVLink message set. This spec closes the remaining gaps to VADR-TS-002 §4 (MAVLink interface) and adds a PyBullet physics backend behind the existing `DroneBackend` Protocol.

Two sub-projects bundled because they're tightly coupled:
- **A — MAVLink message-set spec compliance.** Add TIMESYNC outbound, `SET_POSITION_TARGET_LOCAL_NED` inbound (with a geometric position controller), raise default heartbeat to ≥2 Hz, drop `ODOMETRY` from default rates.
- **B — `PyBulletBackend`** implementing the existing `DroneBackend` Protocol with a programmatic quad body sized to the spec'd 280×280×160 mm chassis.

Bundled because: (1) B without A's `SET_POSITION_TARGET_LOCAL_NED` handler is half-useful — position-mode is the standard contestant API. (2) A's geometric controller needs PyBullet for collision-aware validation. (3) Together they close COR-98 (MoE flies warehouse over MAVLink).

Out of scope, deferred to follow-on specs: vision UDP-5600 chunked-JPEG stream, camera renderer spec-fix (640×360, fx=fy=320, VFoV=90°, +20° tilt), real-drone passthrough backend, MoE retraining against the new backend.

## Spec Gap Audit

VADR-TS-002 §4.3 + §4.4 vs current `sim/pybullet/mavlink_shim/shim.py:32` defaults:

| Concern | Spec | Current | Action |
|---|---|---|---|
| HEARTBEAT rate | ≥ 2 Hz | 1 Hz | default → 2 Hz |
| ATTITUDE | — | 100 Hz | keep |
| HIGHRES_IMU | — | 200 Hz | keep |
| TIMESYNC outbound | required | absent | add (10 Hz default) |
| ODOMETRY | not listed | 100 Hz | drop from defaults; keep method opt-in |
| SET_ATTITUDE_TARGET inbound | required | handled | keep |
| SET_POSITION_TARGET_LOCAL_NED inbound | required | unhandled | add (geometric controller stage) |
| Physics rate | 120 Hz | NumpyQuad: variable substep | PyBullet: `setTimeStep(1/120)` |
| PyBullet backend | — | not built | new file |

## Architecture

```
UDP recv ─┬─ SET_POSITION_TARGET_LOCAL_NED → PositionController ─┐
          │                                                       ├─→ AttitudeController → TRPY mix → DroneBackend.step()
          └─ SET_ATTITUDE_TARGET (G&CNet hot path) ───────────────┘                                          │
                                                                                                            ├── NumpyQuadBackend (existing)
                                                                                                            └── PyBulletBackend (new)
UDP send: HEARTBEAT 2 Hz, ATTITUDE 100 Hz, HIGHRES_IMU 200 Hz, TIMESYNC 10 Hz (driven by RateScheduler)
```

`PositionController` is bypassed when a `SET_ATTITUDE_TARGET` arrives — that's the G&CNet path and stays single-stage. Position-mode is for spec compliance and contestants using the standard MAVSDK position interface. Whichever message arrives last wins; mode is implicit, no explicit switch.

`DroneBackend` is the Protocol at `sim/pybullet/mavlink_shim/backend.py:42`. It already abstracts physics; the new `PyBulletBackend` joins `NumpyQuadBackend` behind the same interface. Selection happens at `MavlinkShim` construction time, no shim changes.

## A — MAVLink Message-Set Compliance

### A1. `server.py` — TIMESYNC send method

Add to `MavlinkServer`:

```python
def send_timesync(self, tc1: int, ts1: int) -> None:
    """tc1 = our timestamp (ns), ts1 = peer's last-known timestamp (ns).
    Server-initiated send: tc1=now_unix_ns, ts1=0.
    Reply-to-inbound: tc1=now_unix_ns, ts1=echoed-from-inbound-msg.ts1.
    """
    self._conn.mav.timesync_send(tc1, ts1)
```

Optional `send_system_time(time_unix_usec, time_boot_ms)` — defer; not required by spec.

### A2. `shim.py` — defaults + recv-loop

Changes to `MavlinkShim.rates_hz` default:
- `heartbeat: 1.0 → 2.0` (spec minimum)
- add `timesync: 10.0` (PX4 convention)
- remove `odometry` from defaults (still callable via custom `rates_hz=` arg; `server.send_odometry` kept for test/debug paths)

Recv loop additions in `_run_loop` and `step`:

```python
for m in msgs:
    t = m.get_type()
    if t == "SET_ATTITUDE_TARGET":
        self._last_target_q = np.array(m.q, dtype=np.float64)
        self._last_target_thrust = float(m.thrust)
        self._mode = "attitude"
    elif t == "SET_POSITION_TARGET_LOCAL_NED":
        self._last_pos_target = _parse_position_target(m)  # honours type_mask
        self._mode = "position"
    elif t == "TIMESYNC" and m.tc1 == 0:
        # Client-initiated ping; reply with our tc1 + their ts1 echoed.
        self._server.send_timesync(tc1=time.monotonic_ns(), ts1=m.ts1)
```

In `_do_one_step`, before calling `AttitudeController.compute`:

```python
if self._mode == "position":
    q_target_enu, thrust_norm = self._position_controller.compute(
        target=self._last_pos_target,
        state=self._last_state,
    )
else:
    q_target_enu = ned_quat_wxyz_to_enu_quat(self._last_target_q)
    thrust_norm = self._last_target_thrust
```

`AttitudeController` and the motor mixer are unchanged.

In `_send_message`, add `"timesync"` case: `self._server.send_timesync(tc1=time.monotonic_ns(), ts1=0)` — server-initiated periodic.

### A3. `position_controller.py` — geometric SE(3) controller (new)

Per Mellinger & Kumar 2011 ("Minimum snap trajectory generation and control for quadrotors"). Single class `PositionController` with `compute(target, state) → (q_des_enu_wxyz, thrust_norm)`.

Inputs:
- `target.pos_ned`, `target.vel_ned`, `target.accel_ned`, `target.yaw` — from `SET_POSITION_TARGET_LOCAL_NED`. Fields disabled by `type_mask` substitute zero (or current yaw, for yaw mask).
- `state` = `DroneState` (ENU world frame, wxyz).

Algorithm (in ENU world frame internally — NED inputs converted at the boundary):

```
e_p = pos_world - pos_target_world                       # position error (ENU)
e_v = vel_world - vel_target_world                       # velocity error
f_des = -K_p · e_p - K_d · e_v + m·g·ẑ_world + m·a_target  # desired thrust vector
                                                            # m = params.mass, g = 9.81
thrust_norm = clip(f_des · z_b_current / max_thrust, 0, 1)
            where z_b_current = R_body_to_world · ẑ
                  max_thrust = 4 · params.k_thrust · params.max_omega²

z_b_des = f_des / ‖f_des‖                                # desired body-z (thrust axis)
x_c = [cos(yaw_target), sin(yaw_target), 0]              # heading reference
y_b_des = normalize(z_b_des × x_c)
x_b_des = y_b_des × z_b_des
R_des = stack_columns(x_b_des, y_b_des, z_b_des)         # desired rotation
q_des = mat_to_quat_wxyz(R_des)
```

Three gains, conservatively sized for the spec'd 5"-quad params:
- `K_p = diag(4.0, 4.0, 8.0)` (vertical stiffer than horizontal)
- `K_d = diag(3.0, 3.0, 6.0)`
- Yaw gain handled implicitly via `R_des` construction; no separate gain needed because we lift yaw into the rotation directly.

NED input → ENU conversion: `pos_enu = (pos_ned[1], pos_ned[0], -pos_ned[2])` (existing `coords_mavlink.py` convention). Yaw flips sign across NED↔ENU.

### A4. `type_mask` honouring

`SET_POSITION_TARGET_LOCAL_NED.type_mask` is a 12-bit field where set bits mean "ignore this field". Standard MAVLink bits:
- 0x001 / 0x002 / 0x004: ignore position x / y / z
- 0x008 / 0x010 / 0x020: ignore velocity x / y / z
- 0x040 / 0x080 / 0x100: ignore acceleration x / y / z
- 0x200: force flag (when set, the accel fields encode force in N, not accel in m/s²)
- 0x400: ignore yaw
- 0x800: ignore yaw rate

Implementation in `_parse_position_target`: when a bit is set, substitute zero for that field (or current state value for position; current yaw for yaw). Velocity-only and yaw-only targets are common contestant patterns and must work.

## B — `PyBulletBackend`

New file `sim/pybullet/mavlink_shim/pybullet_backend.py`. Implements the `DroneBackend` Protocol (`reset`, `step`, `get_imu`) using PyBullet for collision-aware rigid-body physics.

### B1. Construction

```python
@dataclass
class PyBulletBackend:
    params: VehicleParams
    scene: WarehouseScene | None = None   # optional collision world from sim/pybullet/warehouse_loader
    gui: bool = False                     # p.GUI vs p.DIRECT
    physics_hz: float = 120.0             # spec §4.4

    def __post_init__(self) -> None:
        self._client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / self.physics_hz, physicsClientId=self._client)
        self._body_id = self._build_programmatic_quad()
        if self.scene is not None:
            self._scene_handles = self.scene.load_into(self._client)
```

### B2. Programmatic quad body

`_build_programmatic_quad` constructs a single rigid base link, no propellers, no visual mesh:

- Collision shape: `p.createCollisionShape(p.GEOM_BOX, halfExtents=[W/2, L/2, H/2])` with `W=L=0.280`, `H=0.160` (spec chassis).
- Inertia: passed via `p.createMultiBody(... baseInertialFramePosition=[0,0,0], ...)`. PyBullet doesn't expose a direct `diag(Ixx,Iyy,Izz)` parameter; instead we set `principalMoments` via `p.changeDynamics(linkIndex=-1, localInertiaDiagonal=[Ixx, Iyy, Izz])` after creation.
- Mass = `params.mass`.
- Motor positions (body frame, X-config matching `sim/dynamics/numpy_quad.py`):
  - Motor 0 (front-right, CW):  `(+a/√2, -a/√2, 0)`
  - Motor 1 (rear-right, CCW):  `(-a/√2, -a/√2, 0)`
  - Motor 2 (rear-left, CW):    `(-a/√2, +a/√2, 0)`
  - Motor 3 (front-left, CCW):  `(+a/√2, +a/√2, 0)`
  where `a = params.arm_length`. **Note:** if `params.arm_length > chassis_half_diagonal (≈ 0.198 m)`, motor positions exceed the chassis box — the constructor logs a warning. The current `_training_racing_params()` with `arm_length=0.170 m` triggers this (see COR-96 chassis-prior experiment 2026-05-12).

### B3. Motor lag (`tau_motor`) — emulated

PyBullet doesn't model first-order motor lag. Backend maintains internal `_omega_actual` array and integrates each step:

```
ω̇_i = (ω_cmd_i - ω_actual_i) / params.tau_motor
ω_actual_i += ω̇_i · dt   (sub-step level)
```

Forces computed from `ω_actual`, not `ω_cmd`. Matches `NumpyQuadDynamics` exactly.

### B4. Step

```python
def step(self, motor_commands: NDArray[np.float64], dt: float) -> DroneState:
    n_substeps = max(1, int(round(dt * self.physics_hz)))
    sub_dt = dt / n_substeps
    for _ in range(n_substeps):
        # 1. Motor lag integration.
        self._omega_actual += (motor_commands - self._omega_actual) * (sub_dt / self.params.tau_motor)
        # 2. Compute per-motor force and total reaction torque.
        omega_sq = self._omega_actual ** 2
        f_motors = self.params.k_thrust * omega_sq                # (4,)
        # Per-motor torque about body-z: signs alternate CW/CCW per X-config.
        tau_z = self.params.k_torque * omega_sq * np.array([+1, -1, +1, -1])
        # 3. Apply forces at motor positions; sum reaction torque.
        for i in range(4):
            p.applyExternalForce(
                self._body_id, -1,
                forceObj=[0, 0, f_motors[i]],
                posObj=self._motor_positions_body[i],
                flags=p.LINK_FRAME, physicsClientId=self._client,
            )
        p.applyExternalTorque(
            self._body_id, -1,
            torqueObj=[0, 0, float(tau_z.sum())],
            flags=p.LINK_FRAME, physicsClientId=self._client,
        )
        # 4. Advance physics.
        p.stepSimulation(physicsClientId=self._client)
    # Read back state.
    pos_enu, quat_xyzw = p.getBasePositionAndOrientation(self._body_id, physicsClientId=self._client)
    vel_enu, ang_vel_world = p.getBaseVelocity(self._body_id, physicsClientId=self._client)
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    ang_vel_body = self._rotate_world_to_body(np.array(ang_vel_world), quat_wxyz)
    # ... build DroneState, derive IMU as in NumpyQuadBackend.
```

PyBullet's world frame is Z-up; we treat it directly as ENU. PyBullet quaternion convention is xyzw; we convert to wxyz at the boundary.

### B5. IMU derivation

Same pattern as `NumpyQuadBackend`:
- Accel: `(vel_after - vel_before) / dt` in world frame, rotated into body frame, minus body-frame gravity.
- Gyro: body-frame angular velocity straight from PyBullet's `getBaseVelocity`, rotated into body.

## Test Plan

New tests:

| File | Asserts |
|---|---|
| `tests/test_mavlink_shim/test_timesync_handshake.py` | (a) server-initiated TIMESYNC sent at ~10 Hz with `tc1>0`, `ts1=0`. (b) Inbound TIMESYNC with `tc1=0` → reply within 1 tick with that `ts1` echoed. |
| `tests/test_mavlink_shim/test_heartbeat_rate.py` | Heartbeat received at ≥ 2 Hz over 5 s window. |
| `tests/test_mavlink_shim/test_position_target_converges.py` | `SET_POSITION_TARGET_LOCAL_NED` with constant target `(5, 0, -2)` NED → drone within 0.3 m within 10 s, no overshoot past ±0.5 m. |
| `tests/test_mavlink_shim/test_position_target_type_mask.py` | (a) Velocity-only target (`type_mask` disables pos+accel+yaw) tracks commanded velocity within 0.2 m/s steady-state. (b) Yaw-only target rotates without translating beyond ±0.1 m. |
| `tests/test_mavlink_shim/test_mode_latches_on_last_msg.py` | Sequence: position target → attitude target → position target. Verifies mode switches each time. |
| `tests/test_pybullet_backend/test_hover_equivalence.py` | PyBulletBackend and NumpyQuadBackend, same `params=_training_racing_params()`, same `SET_ATTITUDE_TARGET q=identity thrust=hover`, converge to within 0.05 m / 5° / 0.1 m/s over 5 s. |
| `tests/test_pybullet_backend/test_motor_lag.py` | Step `motor_commands` from hover to `1.2×hover_omega`; fit first-order response, recovered `τ` within 20% of `params.tau_motor`. |
| `tests/test_pybullet_backend/test_warehouse_collision.py` | Load `sim/assets/playroom_v1/` scene; drive drone into a wall with constant horizontal thrust; assert penetration depth < 0.05 m and drone stops. |
| `tests/test_pybullet_backend/test_protocol_conformance.py` | `PyBulletBackend` satisfies `DroneBackend` Protocol — covers reset / step / get_imu signatures and dataclass return types. |

Existing tests under `tests/test_mavlink_shim/` continue to pass unchanged (NumpyQuadBackend path).

## File Layout

```
sim/pybullet/mavlink_shim/
├── server.py              MODIFY  ~+30 LOC (send_timesync)
├── shim.py                MODIFY  ~+40 LOC (mode, position-target parse, default rates)
├── position_controller.py NEW     ~120 LOC (SE(3) geometric)
└── pybullet_backend.py    NEW     ~180 LOC

tests/
├── test_mavlink_shim/
│   ├── test_timesync_handshake.py            NEW
│   ├── test_heartbeat_rate.py                NEW
│   ├── test_position_target_converges.py     NEW
│   ├── test_position_target_type_mask.py     NEW
│   └── test_mode_latches_on_last_msg.py      NEW
└── test_pybullet_backend/                    NEW DIR
    ├── test_hover_equivalence.py
    ├── test_motor_lag.py
    ├── test_warehouse_collision.py
    └── test_protocol_conformance.py
```

`scripts/mavlink/smoke_test.py` updated to take `--backend {numpy_quad,pybullet}` and `--mode {attitude,position}` so manual loop testing covers all four combinations.

## Risks and Trade-offs

- **PyBullet force-model fidelity.** No native motor lag, no first-order propeller dynamics, no blade flapping. We emulate `tau_motor`; we accept the rest. For COR-98 (MoE flies warehouse) the policy is robust enough that this shouldn't break.
- **SE(3) controller gains** are conservative defaults from Mellinger. The competition won't grade position-tracking accuracy, and our hot path is attitude — so we tune to "converges within 10 s, no oscillation" and stop.
- **TIMESYNC server-initiated cadence (10 Hz)** is our choice — spec doesn't pin a rate. If a contestant SDK expects client-initiated-only, the inbound reply path still works; the periodic outbound is graceful, not load-bearing.
- **Mode latching** is implicit (last message wins). Documented behaviour; matches PX4. If a contestant interleaves attitude and position targets, the shim toggles modes per message — they need to commit to one.
- **ENU/NED boundary** is the single conversion point at the recv parser and the IMU/position telemetry sender. Both reuse `coords_mavlink.py`. No new transform code.
- **`_training_racing_params().arm_length = 0.170 m` exceeds the spec chassis** — `PyBulletBackend` logs a warning when constructed with such params but does not refuse. Fixing the preset is COR-106 work (separate spec); for THIS spec, the test suite uses both spec-consistent params (`arm=0.115 m`) and the current preset to ensure both run.

## Out of Scope (Follow-up Specs)

- **Vision UDP-5600 chunked-JPEG stream.** Separate spec; depends on a working camera renderer.
- **Camera renderer spec-fix (640×360, fx=fy=320, VFoV=90°, +20° body-pitch tilt).** Cross-cuts perception; separate spec.
- **Real-drone passthrough backend.** Future once we have a target drone.
- **MoE retraining against `PyBulletBackend`.** COR-106 follow-up.
- **`SET_POSITION_TARGET_GLOBAL_INT` and other position-target variants.** Spec lists `LOCAL_NED` only.
- **MAV_CMD_DO_REPOSITION, mission protocol, parameter protocol.** Out of competition scope.

## Definition of Done

- All new tests in the table above pass on a fresh checkout.
- Existing `tests/test_mavlink_shim/*` tests still pass.
- `scripts/mavlink/smoke_test.py --backend pybullet --mode position` runs a 10 s rollout, drone reaches a target position, terminates cleanly.
- TIMESYNC, HEARTBEAT, ATTITUDE, HIGHRES_IMU all visible at spec'd rates on the wire (verified by an external `pymavlink` listener).
- `SET_POSITION_TARGET_LOCAL_NED` and `SET_ATTITUDE_TARGET` both produce flight; switching between them works.
- COR-98 ("MoE flies warehouse over MAVLink") can be executed end-to-end against `PyBulletBackend`.

## Linear Tracking

This work spans:
- COR-94 (MAVLink shim — adds TIMESYNC + position-mode that the MVP didn't ship)
- COR-98 (MAVLink loop validation — completed by B + collision test)
- Future: vision UDP stream + camera spec-fix will be a separate Linear issue.

Final completion comment goes on COR-98; intermediate progress on whichever issue the day's work most directly addresses.
