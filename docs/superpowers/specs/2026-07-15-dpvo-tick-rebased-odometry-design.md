# Tick-Rebased DPVO Gate-to-Gate Odometry

## Goal

Make the banked `fg62` two-gate route repeatable by using DPVO only as a
short-horizon relative-motion sensor between judge ticks. GateNet is unloaded
after the stationary pad lock; CPU `fastgate` remains responsible for terminal
gate guidance. The design must fit beside the simulator on the RTX 3050 4 GB
and must not create new model, corpus, or build-cache copies.

## Evidence and constraints

- `fg62` scored two decoder-verified ticks 4.51 seconds apart without DPVO.
- `fg63..75` show that open-loop command-speed dead reckoning accumulates
  enough x/y/z error to stage early, miss the corridor, hit the ceiling, or
  collide with station pillars.
- Native Windows DPVO reaches the GPU but `lietorch` CUDA operations terminate
  the process. The WSL2/Linux bridge is the supported live backend.
- The current WSL service hardcodes 48 patches, ignores the flight process's
  `DPVO_PATCHES`, permits 90% of total VRAM, and uses focal length 320 although
  the flight camera model uses `fx=fy=226` at 640x360.
- The existing bridge calibrates scale and gates innovations against the
  acceleration-derived KF. That KF is the quantity DPVO is intended to replace
  on the transit leg, so it cannot be the reference of truth.

## Architecture

### Perception ownership

1. Before GO, GateNet performs the stationary pad lock.
2. At GO, `GN_UNLOAD=1` permanently releases GateNet and its CUDA allocations.
3. CPU `fastgate` continues throughout the flight.
4. WSL2 DPVO receives the already-recorded JPEG stream over the existing TCP
   bridge. It may warm up before gate 1 but does not control the gate-1 leg.

### DPVO service configuration

The service accepts session configuration from the Windows client before the
first frame:

- patches per frame: 24 or 32;
- input width and height;
- calibrated intrinsics scaled to the input resolution;
- a conservative CUDA memory-fraction cap;
- optional frame stride.

The service must not copy weights or frames to disk. It loads the existing WSL
checkout and the existing `C:\Users\alexj\DPVO\dpvo.pth` weight file.

### Offline scale calibration

Run the same service configuration over the existing `vq2_servo_fg62` corpus.
Extract DPVO poses at the decoder-verified gate-1 and gate-2 timestamps and fit
the scalar converting DPVO translation units to the known metric displacement
between the two gate anchors. Store only a small JSON calibration artifact
containing configuration identity, scale, alignment metadata, and validation
statistics.

Replay `fg63..75` without modifying those corpora. The calibration is accepted
only if DPVO remains live through the gate-1-to-gate-2 interval and its aligned
trajectory exposes the recorded run-to-run deviations without discontinuous
pose jumps. If a single fixed scale is not stable enough, stop before live
control and design a gate-bearing scale correction; do not fall back to KF
scale calibration.

### Tick rebasing and control output

At the first judge transition (`gate_idx 0 -> 1`):

1. Set the metric origin to the known gate-1 anchor.
2. Align heading using the gyro-integrated flight yaw and camera extrinsics.
3. Record the current DPVO pose as the relative origin.
4. Publish a separate `dpvo_route_p` and health record; do not call
   `KF.update_position`.

`MAPFOLLOW` uses `dpvo_route_p` only during the post-tick transit and staging
leg. The existing `fastgate` attitude pursuit takes ownership once the next
aperture is acquired. Judge tick 2 ends the mission as before.

## Health and failure behavior

DPVO output is valid only when:

- the bridge reply is fresh;
- tracking has initialized;
- consecutive metric deltas obey configured velocity and acceleration bounds;
- no service reset or pose discontinuity has occurred;
- the calibration identity matches resolution, intrinsics, stride, patches,
  and model hash.

Health rejection is based on DPVO's own temporal continuity, never disagreement
with the KF. If DPVO becomes unhealthy, the first live integration remains
observe-only. In a later control flight, unhealthy DPVO triggers a controlled
abort/landing before entering an obstacle corridor rather than silently
returning to open-loop MF_DR.

## Disk-space policy

- No dependency rebuild unless an existing environment is unusable.
- No copied model weights, corpora, frames, or RRD files.
- Offline tools read corpora in place and write one compact JSON/JSONL result.
- Tests use synthetic messages/poses and temporary files only.
- Before any optional cache-producing command, verify Windows and WSL free
  space and estimate its maximum output.

## Verification gates

1. Unit tests verify bridge configuration parsing, intrinsic scaling,
   calibration identity, tick rebasing, and health rejection.
2. Offline replay verifies the exact low-memory configuration on `fg62` and
   then `fg63..75` without corpus writes.
3. A simulator-on memory test measures initialization peak, steady-state VRAM,
   latency, and free headroom for 32 and 24 patches.
4. First live flight is observe-only; DPVO must stay healthy through the entire
   post-tick leg.
5. Control is enabled one responsibility at a time: staging completion, then
   lateral transit correction, then vertical correction.
6. Success target is at least 8 decoder-verified TICKS=2 results in 10 fresh
   attempts. All scoring uses `gidx2.py`.

## Files in scope

- Repo: a new testable DPVO route/calibration module and tests.
- Repo: `vq2/live/vq2wp.py` bridge wiring and map-follow source selection.
- Laptop deployment: `bridge_dpvo.py`, `dpvo_odom_bridge.py`, and an explicit
  low-memory launch/config file.
- No changes to DPVO source or weights unless offline verification proves the
  existing WSL service cannot run the requested configuration.

