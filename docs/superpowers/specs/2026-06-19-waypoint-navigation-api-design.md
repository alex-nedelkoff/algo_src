# Waypoint Navigation API — Design

- **Date:** 2026-06-19
- **Branch:** `aigp-control-waypoints`
- **Status:** approved design, pre-implementation

## Problem

Waypoint-navigation control on the VQ/DCL sim is proven but trapped in standalone scripts.
`goto.py` carries a live-tested control loop (along-track speed + cross-track line-following,
heading-hold strafe, forward-accel cap, tilt abort, arrival-detected sequencing, per-leg settle),
but it is procedural `main()` code with module-level globals — not importable, not unit-tested.
`aigp/commander.py` exposes only low-level sends (arm / reset / velocity-setpoint / attitude-setpoint /
body-rate). There is no reusable programmatic interface another program can import to fly waypoints.

## Goal

A clean, importable, **blocking** Python API — `WaypointNavigator` — that wraps `goto.py`'s proven
control as the single source of truth. `goto.py` is rewritten as a thin CLI over it. The control law
is separated from time/sleep/IO so it is unit-testable.

Non-goals (explicitly out of scope): async/non-blocking operation, a network/IPC server, a true
PX4-style `set_position_target` offboard mode, takeoff/land primitives (the VQ sim spawns the drone
airborne mid-race; there is no takeoff/land), and lateral-sign auto-calibration.

## Approach (chosen: A — pure-step + thin-loop split)

Extract the control law into stateless functions (unit-testable with synthetic states), wrap them in
`WaypointNavigator`, which owns the loop and all IO. Rejected alternatives: B (monolithic class — the
loop's inline time/sleep/IO defeat the testability that motivated the extraction) and C
(generator/coroutine controller — over-abstracted for a blocking-class decision; YAGNI).

## Architecture

```
aigp/navigator.py        NEW — pure control law + WaypointNavigator loop class
goto.py                  REWRITTEN — ~30-line CLI: fresh_start + arm -> build nav -> nav.follow()
tests/test_aigp/test_navigator.py   NEW — unit tests for the pure law
```

The caller owns lifecycle. `WaypointNavigator` takes an already-live drone via an injected `Store`
and `Commander`; it does not reset/arm the sim. `goto.py` (and any other caller) performs
`fresh_start` (sim_reset + wait race-live) + `arm` before handing control to the navigator.

### Module units (each has one purpose, a defined interface, clear deps)

#### 1. Pure control law (no `time`, no `sleep`, no IO — the testable core)

Three stateless functions, each reusing `aigp/control_math.py` and `aigp/geometry.py`:

- `cruise_accel(state, leg_start, target, gains) -> (a2(2,), tangent(2,), speed_cmd)`
  Desired horizontal accel for a leg: along-track ramp-to-stop (`spd = clip(0.6 * dist_along, 0, MAX_SPEED)`,
  decel-limited) + cross-track line-following (`-KP_CT * cross_pos - KD_CT * cross_vel`). Mirrors
  `goto.py:fly_leg`'s inner accel computation. Returns the leg tangent and commanded speed alongside the
  accel so the loop feeds telemetry (`tangent=`, `cruise=`) without recomputing guidance.

- `settle_accel(state, target, gains) -> np.ndarray(2,)`
  Desired horizontal accel to kill momentum at a point: `clip(KP_CT * pos_err, -1, 1) - KD_CT * vel`.
  Mirrors `goto.py:settle`.

- `attitude_command(state, a2, z_sp, yaw_ref, plant, gains) -> (rate_cmd_norm(3,), thrust_norm, tilt_deg, dbg)`
  The inner law (pure version of `goto.py:cmd`): clamp horizontal accel to the tilt budget, add the
  altitude PD (`KP_Z*(z_sp - z) + KD_Z*(0 - vz)`), build the desired attitude via
  `control_math.desired_attitude`, compute the attitude-error rate command, steer yaw to `yaw_ref`
  (`KP_YAW` error + `KD_YAW` rate damp, wrapped to ±π), map collective accel to normalized thrust via
  the probed hover / k_a, and return body-rate command **normalized by `rate_gain_axes` and clipped to
  ±wmax**. `dbg` carries `{a, w_des, q_des, thr}` for telemetry.

`state` is the existing `aigp.state.DroneState` (`pos_ned`, `vel_ned`, `quat_wxyz`, `omega`).

#### 2. `NavGains` dataclass

All `goto.py` tuning constants, with `goto.py`'s proven values as defaults:

```
KP_ATT=[0.5,1.6,1.0], KP_YAW=3.0, KD_YAW=0.3,
KP_CT=0.5, KD_CT=1.2, AL_MAX=0.5, KD_AL=1.2, KP_Z=1.8, KD_Z=3.0,
WMAX=4.0, TILT_MAX_DEG=15.0, ABORT_TILT_DEG=80.0,
WP_TIMEOUT=40.0, MAX_SPEED=1.2, ARRIVE=1.5, DECEL_MAX=2.0,
SETTLE_T=2.5, SETTLE_V=0.4, LOOP_DT=0.004
```

(`TILT_MAX_DEG` replaces goto.py's precomputed `TILT_MAX_ACC = tan(radians(15))*9.81`; derive inside.)

#### 3. Plant params

`plant = (hover, k_a, rate_gain_axes(3,))` loaded from `sysid/sim_response.json`
(`hover_thrust`, `k_a`, `rate_gain_axes.{roll,pitch,yaw}`). Helper `load_plant(path="sysid/sim_response.json")`
returns the tuple. The `goto.py` CLI loads it and passes it in.

#### 4. `WaypointNavigator` (the loop — thin; owns time/sleep/IO/telemetry)

```python
class WaypointNavigator:
    def __init__(self, store, commander, plant, *, gains=NavGains(), flog=None): ...
    def set_origin(self, pos_ned=None, yaw=None) -> None
    def goto(self, wp, *, frame='world', yaw='hold') -> str          # 'reached' | 'timeout' | 'abort'
    def follow(self, wps, *, frame='world', yaw='hold', settle=True) -> str
    def settle(self, target=None, yaw_ref=None) -> None
```

Per-iteration loop body (matches `goto.py`, ordering preserved): read `store.get_drone()`; if present,
check arrival (`dist < ARRIVE -> 'reached'`); compute accel (`cruise_accel` for `goto`/`follow` legs,
`settle_accel` for `settle`); call `attitude_command`; **send the command via
`commander.send_attitude_target(rate_cmd_norm, thrust)` BEFORE any telemetry** (so telemetry can never
delay a command); then `flog.push(...)` if `flog` is set. Abort when `tilt > ABORT_TILT_DEG`; time out
when `t - t_leg > WP_TIMEOUT`. `sleep(LOOP_DT)` per iteration.

`follow` flies each leg from the previous waypoint (or origin for the first), optionally `settle`s
between legs (default on), and returns `'reached'` only if every leg reached; otherwise it returns the
first failure status and logs the failing leg index. Per-leg progress prints (dist/spd/tilt) preserved.

## Frames & yaw

- **Frame** (`frame=` kwarg): `'world'` (NED absolute, default) or `'body'` (fwd/right/down offsets
  from the captured origin heading). `set_origin(pos, yaw)` captures the reference; if never called, the
  navigator captures it lazily from the current drone state on the first command. Body→world conversion
  uses the spawn-yaw basis exactly as `goto.py:main` does (`fwd=[-cos,-sin]`, `right=[-sin,cos]`).
- **Yaw** (`yaw=` kwarg): `'hold'` (strafe — yaw_ref = origin heading; `goto.py` default) or `'face'`
  (yaw_ref = horizontal bearing to the target; rate-capped by the existing `KP_YAW`/`KD_YAW` governor in
  `attitude_command`). `'face'` follows `vq_waypoint2.py`'s mission-2 yaw-toward-bearing behavior.

## Telemetry

`flog` (an `aigp.flight_telemetry.FlightLog` or None) is injected. When set, the loop calls
`flog.push(t, ds, dbg, nearest=target, tangent=tv, cruise=spd, running=store.get_race_live(), armed=True)`
after each send, and `follow` calls `flog.set_path(...)` once at start. When `None`, all telemetry is
skipped (guarded) — no Rerun dependency in the API path. Per the project dashboard rule, the `goto.py`
CLI wires `flog` on by default (`--no-viz` to disable), so live runs stream as before.

## Status / return contract

String statuses matching `goto.py`: `'reached'`, `'timeout'`, `'abort'`. `goto`/`settle` operate on one
target; `follow` aggregates (above). No exceptions for normal flight outcomes; exceptions reserved for
programming errors (bad frame/yaw kwarg).

## Testing plan

`tests/test_aigp/test_navigator.py` unit-tests the pure law with synthetic `DroneState`s (no sim, no
time):

- `cruise_accel`: along-track component points toward the target and saturates at `MAX_SPEED`-derived
  accel; cross-track component opposes a lateral offset (correct sign); zero offset on the line → near-zero
  cross term.
- `settle_accel`: opposes velocity (damping) and pulls toward the target; magnitude clipped.
- `attitude_command`: horizontal tilt clamped at `TILT_MAX_DEG`; rate command clipped to ±`WMAX`; thrust ≈
  hover when level with zero accel; yaw rate sign drives toward `yaw_ref` and wraps across ±π.
- yaw `'face'`: bearing-to-target math matches `atan2` of the horizontal target vector.

The loop (`goto`/`follow`/`settle` iteration) is **not** unit-tested — it requires a live sim; it is
exercised by the existing live-flight workflow (`python goto.py ...` against the VQ sim, dashboard on).

## Risks / notes

- **Lateral-sign assumption.** `goto.py` uses hardcoded lateral/yaw signs and is live-validated for
  single-waypoint go-to; multi-waypoint paths with sharp turns are marginal (weathervane instability on
  stop-and-turn, per `goto.py`'s own STATUS note). The API inherits this exactly — no sign
  auto-calibration is added (that lives in `vq_waypoint2.py`). Documented as a known limitation; a future
  enhancement could fold in the `s_lat`/`s_yawb` auto-calibration.
- **Behavior parity.** The rewrite must be **control-command equivalent** to current `goto.py` for the
  default path (identical flight commands: same gains, same loop order, same arrival/abort/timeout
  thresholds), so the proven live result is preserved. Operator-facing prints (the CLI now reports the real
  mission status `reached`/`timeout`/`abort` instead of an unconditional `mission done`) and the first-leg
  cross-track datum may differ trivially and are intentional improvements, not regressions. Verify by reading
  the resulting `goto.py` diff and a live `python goto.py body 6 0 0` run (reached, tilt ~2°).
- **`vq_waypoint2.py` adoption** is deferred — it can migrate onto `WaypointNavigator` later (the README
  already lists it as a candidate); not in this change.
