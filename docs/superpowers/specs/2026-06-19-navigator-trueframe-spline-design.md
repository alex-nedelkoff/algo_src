# Navigator true-frame port + smooth spline trajectories — Design

- **Date:** 2026-06-19
- **Branch:** `aigp-control-waypoints`
- **Status:** approved design, pre-implementation

## Problem

The `WaypointNavigator` API flies forward/nose-first legs cleanly but its lateral (strafe) flight runs
away: it commands attitude in the **raw quaternion frame**, missing the proven true-frame chain
(`qfix` + `s_cam` + world-y mirror + `WFIX` + in-flight `s_lat` sign probe) that `vq_waypoint2` uses to
strafe at a fixed heading. (Earlier "weathervane" diagnosis was wrong — `vq_waypoint2 --square` flies a
fixed-heading square 4/4; the navigator's raw-frame strafe is the bug.) Separately, point-to-point
waypoint flight is jerky: the speed setpoint decays to zero at every corner (near stop-and-go), with no
corner rounding or continuous trajectory.

## Goal

1. **Smooth trajectories** by default: track a continuous spline through the waypoints (corner-rounded,
   speed carried through corners) instead of stop-and-go legs.
2. **Correct strafe** via the proven true-frame attitude chain, so fixed-heading and look-at-a-point
   camera modes track instead of running away.
3. **Adjustable camera headings**: `course` (camera along travel, default), `fixed` (held heading),
   `lookat X Y Z` (camera locked on a world point).

Non-goals: turning/racing through gates at speed (the teacher/CPC own that), `face` mode (dropped),
re-deriving any frame (reuse the proven chains verbatim — canonical rule).

## Key architectural refinement

`GateTrajectory.sample(s)` already returns the camera-forward heading (`yaw = atan2(-tang)`), and
`traj_track` already flies a spline **nose-forward in the raw frame** (proven for translated/curving
courses). Nose-forward flight is raw-frame-safe (forward = pitch, no y-mirror issue). Therefore:

- **course** (nose-follows-path) uses the **existing raw-frame `attitude_command`** — no new frame code,
  existing 15 unit tests stay valid.
- The **true-frame chain is added only for the strafe modes** (`fixed`/`lookat`), isolating the risky
  frame work to where it is actually needed.

## Architecture

```
gate_traj.py            REUSE as-is (GateTrajectory spline + speed schedule + sample()).
aigp/navigator.py       ADD: spline-tracking path engine (drone-locked s_ref) + true-frame strafe
                        attitude (attitude_command_tf) + camera yaw modes. KEEP: raw-frame
                        attitude_command (course) + the existing point-to-point loop (legs fallback).
goto.py                 CLI flags for the new modes/engine.
tests/test_aigp/test_navigator.py    KEEP existing; ADD true-frame + spline-helper + yaw-mode tests.
tests/test_aigp/test_gate_traj.py    NEW: pure tests for GateTrajectory (currently untested).
```

The caller still owns lifecycle (sim reset + arm) and injects a live `Store` + `Commander`.

### Units

1. **`GateTrajectory`** (existing, reused) — path engine. `nearest_s(pos)`, `sample(s) -> {pos, tang,
   kappa, v, yaw, yaw_rate, a_lat}`, `s_max`, `speed_at(kappa)`. Waypoints fed as `gates`. No change.

2. **Spline tracking law** (new navigator method, ports `traj_track`'s proven loop) — each iteration:
   `s_ref = clip(nearest_s(pos) + LEAD, 0, s_max)`; `ref = sample(s_ref)`; desired horizontal accel =
   cross-track + along-track feedback to `ref["pos"]` + velocity-match to `ref["v"]*ref["tang"]` +
   curvature feed-forward (`a_lat`). Drone-locked `s_ref` cannot run away (the proven anti-runaway).
   Gains = `traj_track`'s race_cruise set (`KP_CT/KD_CT/KD_AL/KP_AL/KP_Z/KD_Z`, `AL_MAX`=tilt budget).

3. **Attitude — two proven paths, selected by mode:**
   - `attitude_command` (existing, raw-frame): used by **course**. `yaw_ref = ref["yaw"]` (camera along
     tangent). Unchanged; existing tests hold.
   - `attitude_command_tf` (new, true-frame): used by **fixed/lookat**. Verbatim port of
     `vq_waypoint2`'s true-frame block — `q_t = qfix(quat)`, `yaw_cur = atan2(s_cam*R_t[1,0],
     s_cam*R_t[0,0])`, `yaw_body = yaw_ref (+pi if s_cam<0)`, world-y mirror `a[1] = -a[1]`,
     `q_des = desired_attitude(a, yaw_body)`, `om_t = omega*WFIX`, `w = KP_ATT*attitude_error_quat(q_t,
     q_des)` with `KD_ATT` damping on roll/pitch + yaw governor, `w *= WFIX`, tilt-conditional collective
     cap, `rate = clip(w/rg, -WMAX, WMAX)`. Takes `s_cam`.

4. **`s_cam` / `yaw0_t`** — computed in `set_origin()` from the spawn quat (verbatim from
   `vq_waypoint2`): `q_t0=qfix(quat)`, `cam_live=-[cos(yaw0),sin(yaw0)]`, `s_cam = sign(R_t0[:2,0] @
   cam_live)`, `yaw0_t = atan2(s_cam*R_t0[1,0], s_cam*R_t0[0,0])`.

5. **`s_lat` probe** (strafe modes only) — at the start of motion, command a fixed lateral push for ~1.5 s,
   measure the achieved world-lateral velocity sign, lock `s_lat ∈ {+1,-1}` (verbatim from
   `vq_waypoint2`). Applied to the lateral command sign.

6. **Camera/yaw modes** (yaw_ref for the attitude chain; rate-capped bearing governor so turns are gentle
   — avoids the in-place-spin tilt spike):
   - `course`: `yaw_ref = ref["yaw"]` (raw-frame attitude). Camera along tangent. Default.
   - `fixed`: constant `yaw_body = yaw0_t` (or `--heading` converted to true-cam). True-frame strafe.
   - `lookat`: yaw steered so camera (−body_x) holds the world point — bearing governor toward
     `(point - pos)` in the true-cam frame, `s_lat`-aware. True-frame strafe.

7. **Legs fallback** (existing point-to-point loop, kept) — strict fixed-heading via the `vq_waypoint2`
   true-frame leg loop. `engine='legs'`. Not the default.

## API + CLI

```python
class WaypointNavigator:
    def __init__(self, store, commander, plant, *, gains=NavGains(), flog=None)
    def set_origin(self, pos_ned=None, yaw=None)            # also computes s_cam, yaw0_t
    def follow(self, wps, *, yaw='course', look_point=None, v_cruise=2.5,
               engine='spline', frame='body') -> str        # 'reached'|'timeout'|'abort'
    def goto(self, wp, *, yaw='course', look_point=None, v_cruise=2.5, engine='spline', frame='body') -> str
```
- `engine='spline'` (default) | `'legs'` (fallback). `yaw='course'|'fixed'|'lookat'`.
- `goto.py`: `--yaw course|fixed|lookat`, `--lookat X Y Z`, `--heading DEG`, `--vcruise V`, `--legs`,
  `body|world` frame, `--no-viz`/`--rrd` (telemetry default on, dashboard rule).
- Status strings unchanged: `'reached'`, `'timeout'`, `'abort'`. Telemetry via injected `flog`, guarded.

## Tests

- `tests/test_aigp/test_gate_traj.py` (NEW, pure): straight-line spline → tangent constant + `v≈v_cruise`;
  curved → `kappa>0` + `speed_at` drops; `nearest_s` monotonic along the path; `sample["yaw"]` is
  camera-forward (opposite tangent).
- `attitude_command_tf` (NEW, pure, sim-convention quats): level-hover thrust≈hover + zero rate; tilt
  clamp; `WFIX` mirror (pitch-rate sign); `s_cam=±1` yaw basis; rate clips.
- yaw modes (pure): `course` returns `ref["yaw"]`; `fixed` constant `yaw0_t`; `lookat` bearing-to-point in
  true-cam frame; `s_cam`/`s_lat` math.
- Existing raw-frame `attitude_command` tests: unchanged (course path).
- Spline-tracking loop + `s_lat` probe: live-only (not unit-tested), validated per the order below.

## Validation order (live) + risks

Fly in increasing risk, restarting the sim between flights (it wedges; `race_live` can show True with
ODOMETRY frozen — full DCGame restart needed):
1. **course over spline** (nose-forward, raw-frame, lowest risk) → smooth path, corner-rounded.
2. **fixed-heading square** (strafe, true-frame + `s_lat`) → the `vq_waypoint2 --square` capability, now
   in the API.
3. **lookat a point** → camera holds a point while flying the spline.

Safety net: `vq_waypoint2 --square` already flies the fixed-heading square (legs fallback + reference).
Risks: frame-sign error → runaway (mitigate: validate course first; `s_lat` probe; tilt auto-abort at
`ABORT_TILT`; restart sim between runs). If true-frame `course` were ever needed and underperformed, course
stays on the proven raw chain (already the design). Reset-budget/sim-wedge: expect a sim restart per
validation flight.
