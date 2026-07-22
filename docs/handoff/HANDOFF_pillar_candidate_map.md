# Handoff: G0-anchored pillar candidate map

## Scope

Build a **quarantined candidate** map from simulator evidence without DPVO.
The intended chain is Janahan's admitted G0 map anchor -> GateNet G0 pose ->
static-hover IMU yaw change -> Station-text PnP and pillar-edge evidence.

This handoff intentionally stops before camera-intrinsics/FoV calibration.

## Decisions

- **Do not use DPVO.** It is not viable in this environment and is not on the
  mapping critical path.
- Treat Janahan's upstream `gate_map_v2.json` as the reference datum:
  `reset-NED`, right-handed, G0 at `[10.595, 0.051, -0.2472]`, yaw `0.03235`.
- Do not modify that admitted gate map or promote flight landmarks.
- A station-text result first maps a **text panel**, not a physical pillar
  axis.  The panel-to-axis offset and face convention remain uncalibrated.

## Evidence captured

All runs were in the simulator.

1. Receive-only fresh G0 reference: no arm/no controls.  G0 association comes
   from fresh reset and `active_gate_index=0`.
2. A short straight run verified that close pillar text is readable, but it
   contacted the G0 gate frame.  Do not use it for mapping.
3. Safe hover-only yaw scan: no forward creep, no collision, automatic landing.
   It captured 1,007 frames at:

   `C:\Users\Administrator\Documents\Codex\2026-07-21\go\work\pillar_g0_yawscan_20260722_0145`

   Key frames:

   - G0 anchor: `1784683645521293000.jpg`; GateNet PnP: 8 corners, not low
     confidence, `t_cam ~= [-0.0003, -0.1814, 11.1752] m`.
   - Station 22: `1784683653186096700.jpg`, `1784683653529449600.jpg`, and
     `1784683653877892400.jpg`; OCR confidence 0.953, 0.987, 0.999,
     respectively.  Last panel box is roughly `24 x 78 px`.

The yaw-scan frame metadata (`frames_dedup.jsonl`) and IMU telemetry
(`mavlink.jsonl`) carry receive-wall timestamps suitable for nearest-time
association and gyro integration.

## Implemented perception

- `vq2/pillars.py`
  - `GpuPillarReader`: GPU EasyOCR, vertical `Station NN` text handling.
  - `fit_station_text_pnp`: metric camera-relative text-panel PnP.
  - `fit_pillar_edges`: Canny/Hough image-space pillar-boundary proposal around
    a text read; returns left/right lines, centre, width, and quality only when
    the pair brackets the text.
- `vq2/tests/test_pillars.py`: synthetic tests cover text PnP and edge fitting.

Live Station-22 edge test found a candidate 63 px-wide silhouette with quality
1.0.  It is evidence only; demand multi-frame consistency before treating the
centreline as a pillar axis.

## Build sequence

1. Establish a full G0 camera pose from the anchor PnP, resolving the gate's
   90-degree in-plane symmetry with the upright gate/world convention.
2. Declare and apply camera-to-body extrinsics.
3. Associate each Station-22 keyframe to IMU, integrate yaw from the anchor
   during the static hover, and retain translation uncertainty explicitly.
4. Transform text-panel PnP poses into reset-NED; robustly fuse with median/MAD
   covariance across the three high-confidence reads.
5. Attach OCR, PnP RMS, and pillar-edge-fit evidence in a sidecar proposal,
   e.g. `text_panels[].observations`.
6. Keep the physical `pillar-1` landmark quarantined until a panel-face to
   pillar-axis offset is calibrated and repeated silhouette fits agree.

## Acceptance rules

- Candidate only: no updates to flight-control map, no `human_approved=true`.
- Reject an observation with low OCR confidence, poor PnP reprojection,
  inconsistent yaw/time association, or unstable edge geometry.
- Same station number is not a unique landmark: preserve multiple candidates
  until geometric association rules out aisle twins.

## Relevant commits

- `74537ed0` pillar read classification
- `b5640961` GPU station text reader
- `90eba30d` metric text-plane PnP
- `fbab5a87` safe receive-only fresh-G0 capture
- `ecf79919` hover-only yaw scan mode
- `81978683` pillar edge fitting
