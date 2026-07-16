# DPVO Prewarm and Tick Alignment

## Goal

Produce a judge-tick-aligned DPVO route pose without changing flight control.
DPVO must be ready early enough to retain a raw pose near gate 1's judge tick,
rebase that pose to the verified gate-1 anchor, and remain active for later
course segments. The next live experiment remains observe-only.

## Measured failure

The `vq2_servo_dpvo_obs1` flight verified one fresh gate-1 tick, but all 215
DPVO route observations failed with `reason=tick_alignment`.

- DPVO handshake became ready 1.17 seconds before the judge tick.
- The first valid raw pose arrived 4.56 seconds after readiness and 3.39
  seconds after the tick.
- No pose existed within the required 150 ms alignment window.
- The bridge stayed alive, reserved at most 406 MB, and retained at least
  2833.7 MB reported CUDA headroom.

The failure is lifecycle timing, not service survival. Rebasing the first
post-tick pose to gate 1 is invalid because it discards several seconds of real
post-crossing motion.

## Scope

This change introduces one behavioral variable: DPVO model prewarming before
GO. It does not change route control, scale, resolution, patch count, maximum
speed, or the 150 ms tick-alignment tolerance.

In scope:

- Start the route-DPVO thread before GO.
- Complete bridge connection, model loading, and the ready handshake while the
  drone remains on the pad.
- Keep GateNet active through pad lock.
- Publish an explicit GateNet-unloaded signal at GPU handoff.
- Begin DPVO frame submission only after that signal.
- Keep the DPVO session active after gate 1 for all later course segments.
- Add timing and memory telemetry required to judge alignment.
- Preserve fail-closed observe-only behavior.

Out of scope:

- Enabling DPVO control.
- Reducing steady-state DPVO latency.
- Changing DPVO patches, resolution, stride, calibration, or health limits.
- Reloading GateNet after GO.
- Building the exploratory semantic map.
- Activating unverified later-gate anchors.

## Perception ownership

The normal flight has one GPU ownership transition:

1. Before GO, GateNet owns active inference and performs the stationary pad
   lock. DPVO may load its base model and complete the handshake, but it does
   not submit frames or grow the tracking graph.
2. At GO, the detector deletes GateNet's model and inference tensors, empties
   its CUDA cache, and sets `state['gatenet_unloaded'] = True` with a wall-clock
   timestamp.
3. The DPVO client observes that signal and begins frame submission
   immediately. There is no fixed one-second sleep in this path.
4. CPU fastgate continues terminal aperture steering. DPVO stays resident and
   tracks continuously; GateNet is not reloaded during normal flight.

The explicit signal prevents DPVO from inferring GPU ownership from
`go_passed`, which currently races the asynchronous detector unload.

## DPVO lifecycle

`DpvoOdom` becomes a staged state machine:

1. **Configure:** load the calibration and construct the route object.
2. **Prewarm:** connect to the existing WSL service and require the matching
   ready identity. Publish `dpvo_prewarm_ready` and memory telemetry. Connection
   attempts are individually bounded and stop at GO; prewarming never blocks
   the race-clock or arming path.
3. **Wait for handoff:** do not call the frame protocol until
   `gatenet_unloaded` is true. Exit cleanly if `state['stop']` is set.
4. **Track:** submit the latest configured-stride frames and retain raw pose
   history across the tick.
5. **Rebase:** when `gate_idx` increases from 0 to 1, select the raw pose
   nearest `gate_tick_ns`. Require an absolute timestamp error no greater than
   150 ms, then map it to the verified gate-1 anchor.
6. **Continue:** keep the same DPVO session alive after rebasing. The terminal
   controller may switch between map-follow and fastgate, but DPVO does not
   restart.

Prewarming failure sets `dpvo_route_healthy = False` and
`dpvo_route_reason = 'service'`. If GO occurs before the ready handshake, the
thread instead latches `dpvo_route_reason = 'prewarm_late'`, submits no frames
for that run, and exits. Neither case delays arming or changes the KF or
controller.

## Tick and map semantics

The judge provides ordered gate identity, not gate coordinates. Route control
may use only anchors marked verified in the offline map. Gate 1 and gate 2 are
currently supported by judge-ticked evidence; later candidate gates remain
disabled until mapped and validated.

Phase 1 validates only the first `0 -> 1` rebase. The DPVO session remains
continuous so a later, separate change can re-anchor each increasing gate
index without another model transition. This preserves the one-variable rule
for the next live experiment.

An eventual multi-gate rebase must check the predicted crossing residual
against the verified gate plane before accepting the anchor. A large residual
marks the map unhealthy rather than silently forcing the trajectory onto an
incorrect gate.

## Telemetry

The live log must make every boundary measurable:

- `dpvo_prewarm_start`: wall time and current race state.
- `dpvo_ready`: identity, ready wall time, initialization duration, and CUDA
  allocation/reservation/free-memory values.
- `gatenet_unloaded`: wall time and GO-relative delay.
- `dpvo_tracking_start`: wall time and delay after GPU handoff.
- `dpvo_first_pose`: frame timestamp, wall time, and ready-to-pose duration.
- `dpvo_tick_align`: judge tick timestamp, selected pose timestamp, signed
  offset in milliseconds, and decision.
- `dpvo_route`: existing position, health, latency, and memory fields.

No frames, model weights, RRD files, or copied corpora are added by this
instrumentation.

## Failure behavior

- A service, identity, or prewarm error latches DPVO unhealthy and logs the
  exception.
- A handshake that misses the GO deadline latches `prewarm_late` and submits
  no frames.
- No DPVO frame is submitted before the explicit GPU-handoff signal.
- A missing tick timestamp returns `tick_timestamp`.
- No raw pose within 150 ms returns `tick_alignment`.
- Non-finite poses, non-monotonic timestamps, and motion above 8 m/s retain the
  existing latched health failures.
- `DPVO_OBSERVE=1` always prevents DPVO from selecting a control position.
- MF_DR and fastgate remain unchanged for the acceptance flight.

## Verification

### Automated tests

Tests use fake state, sockets, and pose replies without importing CUDA:

- Prewarm completes before GO and records readiness.
- No frame is submitted while GateNet still owns the GPU.
- Frame submission starts after `gatenet_unloaded` becomes true.
- Stop-before-handoff exits without submitting a frame.
- A pose within 150 ms produces exactly one healthy `rebase`.
- A pose outside 150 ms returns `tick_alignment` without setting an origin.
- Observe-only source selection cannot affect control.
- Existing configuration, calibration-identity, route-health, and service
  tests remain green.

### Live acceptance

The next fresh observe-only run has three outcomes.

**Pass:**

- `gidx2.py` verifies a fresh `0 -> 1` transition.
- DPVO is ready before GO.
- A raw DPVO pose lies within 150 ms of the tick.
- Within one wall-clock second of the tick, one healthy `rebase` with a
  non-null gate-1 position is published.
- At least ten subsequent observations are healthy `ok` poses with no speed,
  timestamp, non-finite, or service latch.
- The service survives and reported GPU headroom remains at least 750 MB.
- Telemetry confirms `DPVO_OBSERVE=1` and MF_DR control.

**Fail:**

- A verified tick occurs but alignment reports `tick_timestamp` or
  `tick_alignment`.
- Rebase is published more than one second after the tick.
- Route health latches immediately after rebase.
- The bridge or flight process dies or exhausts GPU memory.
- DPVO influences control.

**Invalid:**

- No fresh judge tick occurs.
- The judge is stale, the drone crashes before gate 1, or required telemetry
  is absent.

Passing this gate approves synchronized observe-only output only. Latency,
temporal stability, multi-gate rebasing, map trust, and control enablement each
require separate acceptance gates.

## Disk policy

The implementation edits source and tests only. Deployment updates existing
files in place. The next run uses one fresh recording directory and is not
duplicated. Optional replay output goes to `C:\tmp` and is removed after the
small validation summary is retained.
