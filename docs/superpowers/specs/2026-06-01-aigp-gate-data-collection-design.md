# AI-GP Gate Data-Collection Client — Design

**Date:** 2026-06-01
**Status:** Approved (brainstorming), pending spec review
**Branch:** `aigp-gate-data-collection`

## Goal

Get controlled autonomous flight around a race gate in the official AI-GP
simulator, and while flying, capture FPV camera frames auto-labeled with the
gate's pose — producing a training dataset for a gate-detection / gate-pose
network. This is **Stage 2** of competition prep (Stage 1 "first light" — live
telemetry + vision + race protocol — is already confirmed working).

Two concrete outcomes:
1. The drone flies a controlled pattern around gate 0 (orbit and/or approach).
2. Every captured frame is saved with pixel-accurate gate labels derived from
   ground-truth geometry (no hand-labeling).

## Context: how the sim works (confirmed via probes)

- **MAVLink2 over UDP**, sim is the server. Client binds `udpin:0.0.0.0:14550`,
  `wait_heartbeat()`. Telemetry confirmed streaming: `ATTITUDE` (114 Hz),
  `HIGHRES_IMU` (114 Hz), `LOCAL_POSITION_NED` (92 Hz), `ODOMETRY` (73 Hz),
  `ACTUATOR_OUTPUT_STATUS` (92 Hz), `HEARTBEAT` (10 Hz), `ENCAPSULATED_DATA`
  (4 Hz).
- **The sim has its own inner-loop flight controller.** It accepts high-level
  setpoints: `SET_ATTITUDE_TARGET` (attitude/rate + thrust),
  `SET_POSITION_TARGET_LOCAL_NED` (position/velocity NED),
  `SET_ACTUATOR_CONTROL_TARGET` (raw motors). We do NOT port low-level motor
  control — we port **guidance**.
- **Coordinate frame is NED** throughout (x north, y east, z down; altitude
  negative). Telemetry, gate poses, and position/velocity setpoints are all NED.
- **Vision** is a separate raw-UDP stream on port `5600`: chunked JPEG, header
  `<IHHIIQ>` = `(frame_id, chunk_id, total_chunks, jpeg_size, payload_size,
  sim_time_ns)`. Confirmed decoding to **640×360 BGR** frames.
- **Race status** arrives via `ENCAPSULATED_DATA` type 1 (`<BQqqIq>`):
  sim_boot_ms, race_start_ms, race_finish_ns, **active_gate_index**,
  last_gate_time. Confirmed: `active_gate_index=0`, race started.
- **Gate geometry** arrives via `DATA_TRANSMISSION_HANDSHAKE` (announces
  transfer_id=`width`, packet count=`packets`) + chunked `ENCAPSULATED_DATA`
  type 2, reassembling to: `num_gates` (`<H`), then per gate `<Hfffffffff>` =
  gate_id, position NED (x,y,z), orientation NED quat (w,x,y,z), width, height.
  **Confirmed NOT streamed continuously** — broadcast only once at race start.
  Must be captured at start (see Gate Acquisition).
- **Custom command** `SIM_RESET = 31000` via `command_long` resets the sim
  (expected to re-trigger the race-start gate broadcast).
- **Collisions**: `COLLISION` msg, `id` 1001=gate, 1002=environment.

## Camera model (given)

- Resolution 640×360. `cx=320, cy=180`. `fx=fy=320`.
- Intrinsic matrix `K = [[320,0,320],[0,320,180],[0,0,1]]`.
- Note: `fx=320` on 640px width ⇒ **HFoV = 2·atan(320/320) = 90°**; VFoV works
  out to ≈58.7°. The supplied "VFoV=90" label appears to actually be the HFoV.
  We treat the explicit `K` as source of truth and **validate via overlay**.

## Architecture

A new self-contained package in `algo_src` (proposed `aigp/`), dependent only on
**numpy, pymavlink, opencv-python** (all present in the `aigp` conda env on the
laptop) — no torch/pybullet/heavy deps. Runs on the Windows laptop directly
against the sim. Components, each independently testable:

### 1. IO layer (`io_layer.py`)
- MAVLink connection `udpin:0.0.0.0:14550` + threaded receive loop dispatching
  to a shared `State`.
- Vision receiver thread on UDP `5600`: chunked-JPEG reassembly → latest BGR
  frame + sim timestamp into `State`.
- `State` (thread-safe snapshots): drone pose NED (pos, vel, quat `[w,x,y,z]`,
  omega) from `ODOMETRY`/`LOCAL_POSITION_NED`/`ATTITUDE`; race `gate_idx`;
  latest vision frame + timestamp; gate list.

### 2. Gate acquisition (`gates.py`)
- On startup: send `SIM_RESET (31000)`, then listen for
  `DATA_TRANSMISSION_HANDSHAKE` + track-info chunks, reassemble, parse → list of
  `Gate(id, pos_ned, quat_ned, width, height)`. Cache (gates are static).
- Pure parser `parse_track_payload(bytes) -> list[Gate]` is unit-tested.
- Fallbacks: if no broadcast within N s, prompt user to restart the flight, or
  accept a CLI-provided gate position (`--gate x,y,z`).

### 3. Guidance (`guidance.py`)
- Common interface: `Pattern.update(state) -> Setpoint(vx, vy, vz, yaw)` (NED
  velocity + NED yaw). Pure functions of state + target gate.
- `OrbitPattern(radius, speed, height)`: drone tracks a circle of given radius
  around the gate position at a fixed height; commanded velocity is tangential
  + a radial correction term to hold the radius; `yaw` always points at the gate
  so the camera frames it. Yaw = `atan2(gate_e - drone_e, gate_n - drone_n)`.
- `ApproachPattern(offsets, speed)`: repeated straight runs toward/through the
  gate from a set of varied start offsets, looping; yaw toward gate.
- Selected by CLI arg (`--pattern orbit|approach`). Designed so a future
  thrust+body-rate controller (Approach B) can replace the velocity output
  behind the same `update()` call site (Approach C — hybrid).

### 4. Commander (`commander.py`)
- `arm()` (`COMPONENT_ARM_DISARM`), `sim_reset()`, and
  `send_velocity_setpoint(vx,vy,vz,yaw)` via `SET_POSITION_TARGET_LOCAL_NED`
  with `MAV_FRAME_LOCAL_NED`, position+accel+yaw_rate ignored, velocity + yaw
  active.
- Runs the control loop at a fixed rate (~50 Hz): read State → guidance.update →
  send setpoint.

### 5. Data logger (`logger.py`)
- Decoupled capture loop (~15 Hz, independent of control rate). For each frame,
  if the vision frame is fresh (timestamp advanced) and a drone pose + gate are
  available, save:
  - `dataset/<run_id>/frames/NNNNNN.jpg` (the BGR frame)
  - one line in `dataset/<run_id>/labels.jsonl`:
    - `frame`, `t_sim_ns`
    - `drone_pos_ned`, `drone_quat_wxyz`
    - `gate_id`, `gate_pos_ned`, `gate_quat_wxyz`
    - `gate_rel_cam` (gate center in camera optical frame: x-right, y-down,
      z-forward)
    - `range_m`
    - `gate_center_px` `(u,v)`, `gate_bbox_px` `(u0,v0,u1,v1)` (from projecting
      the 4 gate corners), `in_frame` (bool)
  - `dataset/<run_id>/meta.json`: K, run params, pattern, git SHA.
- Stale/missing data → skip (never save a mislabeled frame).

### 6. Geometry / projection (`geometry.py`)
- `quat_to_R(q_wxyz) -> R` (body→world, NED).
- `world_to_camera(p_world_ned, drone_pos_ned, drone_quat_wxyz) -> p_cam`:
  world→body via `R.T`, body→camera optical via fixed extrinsic
  `R_body_to_cam` (camera looks forward along body +x; optical z=body +x,
  x=body +y(right), y=body +z(down) — i.e. NED body maps cleanly to optical).
  v1 assumes no translation offset and forward-aligned mount.
- `project(p_cam, K) -> (u,v)` standard pinhole; flag points behind camera
  (z<=0) as not-in-frame.
- All pure, unit-tested with hand-computed expected values.

## Frame conventions (the crux)

- Velocity + yaw setpoints are computed and sent in NED — no conversion needed
  on the command path.
- Labeling path: world(NED) → body via `ODOMETRY` quat → camera optical via
  fixed extrinsic → pixel via `K`.
- **Camera extrinsic is a v1 assumption** (forward-aligned, no offset),
  explicitly validated by the overlay acceptance test; adjusted if the gate
  projects off-target.

## Testing

**TDD, pure functions first (run anywhere with numpy, no sim):**
- `geometry`: `quat_to_R` orthonormality; `project` of known points;
  `world_to_camera` round-trips and known cases (gate directly ahead → center
  pixel ≈ (320,180)).
- `guidance`: `OrbitPattern` produces tangential velocity at the target radius
  and yaw pointing at the gate; `ApproachPattern` reduces gate distance.
- `gates`: `parse_track_payload` on a synthesized byte buffer round-trips a
  known gate list.

**Integration (with sim, manual):**
- Fly an orbit, capture a short dataset.
- Run `overlay_labels.py`: draws projected gate center + bbox on saved frames.
  **Acceptance gate:** the projected gate visibly lands on the actual gate in
  the FPV frames — this validates intrinsics, extrinsics, and frame handling at
  once.

## Error handling

- No heartbeat (10 s) → clear error, exit.
- No track data after reset → fallback (prompt restart / CLI gate).
- Stale vision frame → skip capture.
- `COLLISION` with env (1002) → log; optional `sim_reset()` to recover.
- Drone diverges (position NaN / runaway) → stop loop, log.

## Out of scope (this stage)

- Full attitude-rate / thrust controller (Approach B) — deferred to racing-speed
  tuning.
- Vision-only operation (no ground truth) — the *trained network* will enable
  that later; this stage uses privileged gate poses for both guidance and labels.
- Gate sequencing / full-lap racing — single target gate (gate 0) here.
- Training the network itself — this stage produces the dataset.

## Open items

1. `SIM_RESET (31000)` re-triggering the gate broadcast — to be confirmed in
   implementation; fallback is user-restart.
2. Camera extrinsic mount — validated/adjusted via overlay.
3. Intrinsics HFoV vs VFoV labeling — resolved by treating `K` as truth +
   overlay validation.
4. True vision frame cadence (first-light reassembly reported an implausible
   386 fps) — pin down actual unique-frame rate during integration.
