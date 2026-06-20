# Full Control API (`Drone` facade) — Design Spec

> Status: APPROVED (2026-06-20). Builds on the camera-decoupled `WaypointNavigator` (branch
> `aigp-control-waypoints`, HEAD `2f77389`, 192 tests). Consumer: Janahan, importing the Python
> class directly.

## Goal

Turn the proven `WaypointNavigator` control engine into a **full, importable control API** for
Janahan: complete camera modes (incl. **`lookat`** — camera locked on a world point), **motion
primitives** (orbit/hover/takeoff/descend), **threaded execution control** (start a mission, then
`abort`/`pause`/`resume`/`status` from another thread), and **polish** (one config object, clear
status/errors, docstrings, an example). Janahan wires + arms the MAVLink/sim himself and passes the
live `store`/`commander`/`plant` in (he owns the connection lifecycle).

## Architecture

Two layers — the real-time control engine stays isolated from threading + ergonomics:

```
aigp/navigator.py   WaypointNavigator  — low-level control loop (engine). Mostly unchanged.
aigp/drone.py (NEW) Drone, Mission, Status, Result, FlightConfig — the public API (facade).
examples/drone_demo.py (NEW) — runnable usage example.
```

- **`WaypointNavigator` (engine):** keeps the proven loop (line-following, true-frame strafe,
  stop-on-dime, z-integral, `Z_FF`, anticipatory collective, the envelope). Two *control-level*
  additions only (below). No threading lives here.
- **`Drone` (facade):** what Janahan imports. Wraps one `WaypointNavigator`. Owns the worker thread,
  the `Mission` handle, the motion primitives (composed from navigator calls), config + validation +
  docs. At most one active mission.

### Reuse / canonical constraints

- Do **not** rewrite or risk the engine's control math (frames are canonical; 192 tests must stay
  green). Engine edits are additive.
- `lookat` reuses the **same chart bridge as `course`**: `yaw_ref_tf = yaw0_t + s_lat·angle(cam_live, d)`,
  with `d` = unit vector from drone → look-point (camera-at-point) instead of the travel direction.
- Primitives compose existing `goto`/`follow`/`settle` — they inherit the whole control stack +
  safe envelope automatically.

## 1. Engine additions (`navigator.py`)

Control-level only:

- **`_yaw_ref_tf` gains two modes:**
  - `"lookat"`: camera (−body_x) locked on `self._look_point` (world NED). Compute `d = unit((look_point −
    drone_pos)[:2])`; return `yaw0_t + (s_lat or 0)·signed_angle(cam_live, d)` (same formula as
    `course`, `d` = bearing-to-point instead of travel). Degenerate (|rel|<~1 m): hold current
    true-cam heading.
  - `"face"`: nose (+body_x) at the target → camera-at-point of the *opposite* bearing; i.e. point
    the nose at the target. Implemented as `lookat` of the anti-point, or directly
    `yaw0_t + s_lat·signed_angle(cam_live, −d_target)`. (Nose-at-target = camera-away.)
- **`_is_tf_mode`** includes `"lookat"` and `"face"` (route them through `_fly_strafe_course`, the
  true-frame engine). `_cur_travel` is unused for these (yaw is point-based, not travel-based).
- **Cooperative abort/pause:** `WaypointNavigator` gets optional injected events
  `self._abort_evt` / `self._pause_evt` (`threading.Event` or None). The flight loops
  (`_fly_strafe_course`, `_fly_leg`, `settle`, `_probe_inflight`) check each iteration:
  - `abort_evt.is_set()` → break, return `"abort"`.
  - `pause_evt.is_set()` → hold position (zero-accel `settle`-style command, camera mode preserved)
    until cleared or aborted.
  - When both events are `None` (default), behavior is exactly as today (blocking, uninterruptible)
    so the existing CLI (`goto.py`) and 192 tests are unchanged.
- A setter `set_look_point(point_world)` (the facade resolves frames before calling).

These are the only engine changes. Everything else is the facade.

## 2. `Drone` facade (`drone.py`)

### Construction + lifecycle (Janahan owns the connection)

```python
Drone(store, commander, plant, *, config: FlightConfig | None = None, flog=None)
drone.set_origin(pos_ned=None, yaw=None)   # passthrough to the navigator
```

`Drone` builds an internal `WaypointNavigator(store, commander, plant, gains=config.to_navgains(),
flog=flog)`. Janahan still does MAVLink connect + sim `fresh_start` + `arm` himself (as in
`goto.py`); the facade does not own the socket.

### Motion methods — each returns a `Mission` (started on the worker thread)

```python
m = drone.goto(wp, *, yaw="course", look_at=None, speed=None, frame="body", stop=False)
m = drone.follow(wps, *, yaw="course", look_at=None, speed=None, frame="body", stop=False)
m = drone.orbit(center, *, radius, speed=None, seconds, frame="body", direction="ccw")
m = drone.hover(seconds=None)
m = drone.takeoff(altitude)                 # climb `altitude` m above current, in place
m = drone.descend(altitude)                 # descend `altitude` m, in place;  land = alias
drone.look_at(point, frame="body")          # set persistent camera target (used when yaw="lookat")
```

- `yaw ∈ {"hold","course","face","lookat"}`; `look_at`/`look_point` required for `lookat`.
- `speed=None` → `config` default (safe envelope). `frame ∈ {"world","body"}`.
- **Auto-abort preemption:** starting any motion method while a mission is active first aborts the
  running mission (sets its abort event, joins the worker), then starts the new one. No settle gap —
  the new mission's first command takes over directly.
- Blocking convenience: `drone.goto(...).wait()` (or `wait=True` kwarg → returns `Status`).

### `Mission` handle (thread-safe)

```python
m.status() -> Status      # live snapshot (lock-protected copy)
m.abort()                 # stop this mission + auto-start hover() so the drone holds, not falls
m.pause(); m.resume()     # set/clear the pause event
m.wait(timeout=None) -> Status   # join the worker; returns the final Status
m.done -> bool            # worker finished
m.result -> Status | None # final Status once done
```

- One worker thread per `Drone`; `Drone` keeps a reference to the current `Mission`.
- Thread-safety: `abort`/`pause` are `threading.Event`s injected into the navigator; the live
  `Status` is written by the worker under a `threading.Lock` and copied out on `status()`. Control
  commands are only ever issued on the worker thread.

### `Status` / `Result`

```python
@dataclass
class Status:
    result: Result            # RUNNING | REACHED | TIMEOUT | ABORT | ERROR
    phase: str                # e.g. "probe", "leg 2/4", "orbit", "hover", "settle"
    pos_ned: np.ndarray       # current position
    vel: float                # horizontal speed (position-derived)
    tilt_deg: float
    mode: str                 # camera/yaw mode
    target: np.ndarray | None # current waypoint / look-point
    progress: float           # 0..1 (legs completed / total, or time for orbit/hover)
    error: str | None         # message when result == ERROR

class Result(Enum): RUNNING, REACHED, TIMEOUT, ABORT, ERROR
```

## 3. Motion primitives (composed)

- **`orbit(center, radius, speed, seconds, direction)`:** resolve `center`; build a ring of `N=24`
  waypoints on the circle of `radius` around `center` in the horizontal plane at the drone's current
  altitude; `look_at = center`; fly to the nearest ring point, then `follow` the ring (looping,
  `yaw="lookat"`) until `seconds` elapses; stop (hover at the current point). Pure helper
  `_orbit_ring(center, radius, n, direction) -> list[wp]` is unit-tested.
- **`hover(seconds)`:** hold current position via the settle law for `seconds` (or until aborted if
  `None`). Camera holds current heading.
- **`takeoff(altitude)`:** `goto((current_xy, current_z − altitude), yaw="hold")` (NED: −z = up).
- **`descend(altitude)` / `land`:** `goto((current_xy, current_z + altitude), yaw="hold")`. No real
  ground in the sim; `land` = descend to a low altitude.

All inherit line-following, stop-on-dime, z-integral, `Z_FF`, anticipatory collective, the envelope.

## 4. Polish

- **`FlightConfig` dataclass:** centralizes the envelope — `vmax, amax, tilt_deg, zff, capture,
  kiz, c_max`, plus `default_speed`. **Safe defaults from this session's envelope finding:** `amax≈4`,
  `tilt_deg≤20`, `default_speed≈5` (clean, <1 m sag, no ground-crash). `to_navgains()` maps it onto
  `NavGains`. Per-call `speed=`/overrides allowed; values beyond the safe envelope are permitted but
  the docstring warns (z-sag/ground-crash above ~tilt 25°).
- **Validation** (raise `ValueError`, clear message): `frame ∈ {world,body}`; `yaw ∈ {hold,course,
  face,lookat}`; `radius>0`, `speed>0`, `altitude>0`, `seconds>0`; `look_at` set when `yaw="lookat"`
  or for `orbit`.
- **Docs:** docstring on every public method (what it does, params, returns, raises) + a module
  docstring with a quickstart; `examples/drone_demo.py` (connect → arm → takeoff → orbit → land,
  with an abort-from-another-thread demo).

## Testing strategy

- **Engine (pure, `tests/test_aigp/test_navigator.py`):** `_yaw_ref_tf("lookat")` points the camera
  at the look-point; `_yaw_ref_tf("face")` points the nose at the target; `s_lat` sign mirrors;
  degenerate near-point holds heading. (Mirror the existing `course` tests.)
- **Facade (pure/mockable, `tests/test_aigp/test_drone.py`):**
  - `_orbit_ring` produces `N` points at `radius` around `center`, correct winding for direction.
  - `takeoff`/`descend` compute the right target (current xy, shifted z).
  - Validation raises on bad frame/mode/negative params/missing `look_at`.
  - `FlightConfig.to_navgains()` maps fields correctly; safe defaults within the envelope.
  - **Mission threading against a FAKE navigator stub** (no sim): a stub whose "loop" spins checking
    the injected events + writing a status. Assert: `abort()` stops it and yields `Result.ABORT`;
    `pause()/resume()` gate progress; a second motion call **auto-aborts** the first; `status()`
    returns a consistent snapshot; `wait()` returns the final `Status`. Use short sleeps + joins; no
    real timing dependence (drive the stub deterministically).
- **Live (operator, fresh sim):** `lookat` (camera stays on a point while flying a square); `orbit`
  (circles a point, camera locked); `takeoff`/`descend`; `abort`/`pause` mid-flight; preemption
  (issue a new goto while one runs). Append AI-GP Experiment Log rows. Keep within the safe envelope
  (the host wedges on crash).

## Out of scope (YAGNI / future)

- Moving / tracked `lookat` target (only a fixed world point now).
- Real takeoff/land (the sim spawns airborne; these are climb/descend).
- Network/ROS/service interface (Janahan imports the class).
- Owning the MAVLink connection or sim lifecycle (Janahan wires + arms).
- Multiple concurrent missions / multi-drone.

## Known risks

- `lookat`/`face` yaw-sign (the chart-bridge `s_lat` term) is validated live, like `course` — a wrong
  sign points the camera the wrong way but does not destabilize (translation is heading-independent
  since the fixed-spawn-forward recompose). Flip via the same `s_lat` if needed.
- Thread-safety: the navigator was single-threaded; the cooperative-event approach keeps all control
  on the worker thread (only events + a status lock cross threads) — no command-path contention.
- Engine edits must keep the 192 tests green and `goto.py` unchanged (events default `None`).
