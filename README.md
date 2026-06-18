# AI-GP Flight Control + Waypoint Demo

Analytic flight control for the Anduril AI Grand Prix drone (VQ / DCL sim): a mass-agnostic
PID attitude/position controller, guidance patterns, and the live-sim MAVLink + vision IO.
Curated from the `aigp-gate-data-collection` branch for reuse.

## Fly waypoints

Start the VQ / DCL sim (MAVLink on UDP 14550, vision JPEG on UDP 5600), then:

```
python vq_waypoint2.py --mission 1 --v 2.2 --leg 12 --nwp 5
```

- `--mission 1` (default): a straight course along the spawn camera axis with +/-3 m lateral
  offsets, reachable by strafe at a fixed heading. Zero frame risk (the heading never leaves
  the spawn yaw).
- `--mission 2`: the course bends (`--turn 0.25` rad/leg); the yaw reference slews toward the
  waypoint bearing (rate-capped, yaw-error governor).
- Other flags: `--v` cruise speed, `--leg` leg length (m), `--nwp` waypoint count, `--dur`
  duration (s), `--no-viz`.

Frame signs are **auto-calibrated** from data in the first seconds of flight and printed
(`s_lat` = lateral direction, `s_yawb` = yaw-steer direction). No manual frame tuning.

## Key files

| File | Role |
|---|---|
| `aigp/attitude_control.py` | The controller. `AttitudeSetpointController`: pos+vel **PD** -> desired accel (horizontal-tilt clamped) -> desired attitude -> sim send-frame remap. Mass-agnostic thrust via probed hover / k_a. |
| `aigp/control_math.py` | Attitude and thrust math: `desired_attitude`, thrust map, attitude-error quaternion, collective accel. |
| `aigp/guidance.py` | Guidance patterns (orbit / approach / waypoint). |
| `aigp/io_layer.py`, `aigp/state.py`, `aigp/protocol.py`, `aigp/commander.py` | Live-sim IO: MAVLink + vision receivers, thread-safe state store, chunked-JPEG vision decode, command sender. |
| `fit_model.py` | `qfix` (quaternion order fix) and model helpers. |
| `vq_waypoint2.py` | The waypoint demo (see above). |

Other fliers included for reference: `vq_track_wp.py`, `vq_gate_wp.py`,
`goto.py`, `traj_track.py`, `race_cruise.py`, `fly_gate3.py`.

## Requirements

```
pip install pymavlink numpy opencv-python
```

Tests: `pytest tests/test_aigp/`.
