"""Drone — full importable control API over the proven WaypointNavigator.

Janahan wires + arms the MAVLink/sim himself and passes the live store/commander/plant in; this
facade adds threaded missions (abort/pause/status), motion primitives (orbit/hover/takeoff/descend),
and ergonomics. One mission at a time; a new motion call auto-aborts the running one.

Quickstart:
    drone = Drone(store, commander, plant)
    drone.set_origin()
    drone.takeoff(3).wait()                       # climb 3 m, block until done
    drone.orbit((10, 0, 0), radius=5, seconds=8)  # circle a point, camera locked, async
    ...
    drone.land(3).wait()
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from aigp.commander import TrpyCommander
from aigp.navigator import NavGains, WaypointNavigator


def _orbit_ring(center, radius, n, direction):
    """n waypoints on the circle of `radius` around `center` (XY), at center's altitude."""
    center = np.asarray(center, float)
    sgn = 1.0 if direction == "ccw" else -1.0
    out = []
    for k in range(n):
        th = sgn * 2.0 * np.pi * k / n
        out.append(np.array([center[0] + radius * np.cos(th),
                             center[1] + radius * np.sin(th), center[2]]))
    return out


class Result(Enum):
    RUNNING = "running"
    REACHED = "reached"
    TIMEOUT = "timeout"
    ABORT = "abort"
    ERROR = "error"


@dataclass
class Status:
    result: Result
    phase: str
    pos_ned: np.ndarray
    vel: float
    tilt_deg: float
    mode: str
    target: "np.ndarray | None"
    progress: float
    error: "str | None" = None


@dataclass
class FlightConfig:
    """Flight envelope. Defaults are tuned for TIGHT WAYPOINT TRACKING (vmax=3, capture=1.0):
    measured live across 5 random 8-waypoint courses to track 31% closer to the commanded path than
    the old v5 default, with half the variance -- moderate speed stays out of the v5 rate-loop runaway
    edge and the small CAPTURE commits to each waypoint. For SPEED over precision use FlightConfig.fast()
    (~v5; see feedback_vq_live_runaway_threshold). `vlat_max` is the SIDEWAYS speed cap, kept conservative
    (the lateral axis is the runaway-prone one) -- NOT slaved to vmax. __post_init__ rejects garbage."""
    vmax: float = 3.0
    vlat_max: float = 1.5
    amax: float = 4.0
    tilt_deg: float = 20.0
    zff: float = -2.4
    capture: float = 1.0
    kiz: float = 0.8
    c_max: float = 18.0
    default_speed: float = 3.0
    corner_slow: float = 0.0   # 0 = free flow; (0,1] = anticipatory corner braking (min speed frac on a U-turn)

    @classmethod
    def fast(cls, **kw) -> "FlightConfig":
        """The old v5 envelope: faster cruise + loose corner flow (vmax=5, capture=2.5). Strays ~31%
        more from the commanded path than the tracking default and sits at the v5 runaway edge -- use
        when speed matters more than waypoint precision. Override any field via kwargs."""
        return cls(**{"vmax": 5.0, "capture": 2.5, "default_speed": 5.0, **kw})

    def __post_init__(self):
        for name in ("vmax", "vlat_max", "amax", "capture", "c_max", "default_speed"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(v) or v <= 0:
                raise ValueError(f"FlightConfig.{name} must be a finite number > 0, got {v!r}")
        if isinstance(self.kiz, bool) or not isinstance(self.kiz, (int, float)) \
                or not math.isfinite(self.kiz) or self.kiz < 0:
            raise ValueError(f"FlightConfig.kiz must be a finite number >= 0, got {self.kiz!r}")
        if isinstance(self.zff, bool) or not isinstance(self.zff, (int, float)) \
                or not math.isfinite(self.zff):
            raise ValueError(f"FlightConfig.zff must be a finite number, got {self.zff!r}")
        if not (0.0 < self.tilt_deg <= 60.0):
            raise ValueError(f"FlightConfig.tilt_deg must be in (0, 60], got {self.tilt_deg!r}")
        if isinstance(self.corner_slow, bool) or not isinstance(self.corner_slow, (int, float)) \
                or not math.isfinite(self.corner_slow) or not (0.0 <= self.corner_slow <= 1.0):
            raise ValueError(f"FlightConfig.corner_slow must be in [0, 1], got {self.corner_slow!r}")
        if self.vmax > 8.0 or self.vlat_max > 8.0:
            raise ValueError(f"FlightConfig vmax/vlat_max past the rate-loop runaway wall "
                             f"(stable ~5, hard cap 8); got vmax={self.vmax}, vlat_max={self.vlat_max}")

    def to_navgains(self) -> NavGains:
        g = NavGains()
        g.MAX_SPEED = self.vmax
        g.VLAT_MAX = self.vlat_max
        g.FWD_AMAX = self.amax
        g.DECEL_MAX = max(g.DECEL_MAX, self.amax)
        g.TILT_MAX_DEG = self.tilt_deg
        g.Z_FF = self.zff
        g.CAPTURE = self.capture
        g.KI_Z = self.kiz
        g.C_MAX = self.c_max
        g.CORNER_SLOW = self.corner_slow
        return g


_FRAMES = ("world", "body")
_YAWS = ("hold", "fixed", "course", "face", "lookat")


def _check_frame(frame):
    if frame not in _FRAMES:
        raise ValueError(f"frame must be one of {_FRAMES}, got {frame!r}")


def _check_yaw(yaw):
    if yaw not in _YAWS:
        raise ValueError(f"yaw must be one of {_YAWS}, got {yaw!r}")


def _check_positive(name, val):
    # reject bool (True is an int), non-numerics, NaN AND inf -- an inf speed/radius/altitude
    # silently bypassed the old `val > 0` check and propagated into the flight loop.
    if isinstance(val, bool) or not isinstance(val, (int, float)) \
            or not math.isfinite(val) or val <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {val!r}")


def _check_wp(wp, name="waypoint"):
    """Validate a single NED waypoint: a finite 3-vector. A non-finite (NaN/inf) or wrong-shape
    coordinate previously flowed straight into the attitude loop and streamed NaN rate+thrust
    commands to the FC (live loss-of-control). Returns the float ndarray."""
    a = np.asarray(wp, float)
    if a.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector (NED x,y,z), got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite, got {wp!r}")
    return a


class _StatusBox:
    """Thread-safe holder of the latest Status. The worker thread calls update()/set_result();
    the caller reads get() (a copy)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._s = Status(result=Result.RUNNING, phase="init", pos_ned=np.zeros(3), vel=0.0,
                         tilt_deg=0.0, mode="", target=None, progress=0.0, error=None)

    def update(self, *, phase, ds_pos, vel, tilt, mode, target, progress):
        with self._lock:
            self._s.phase = phase
            self._s.pos_ned = np.asarray(ds_pos, float).copy()
            self._s.vel = float(vel)
            self._s.tilt_deg = float(tilt)
            self._s.mode = mode
            self._s.target = None if target is None else np.asarray(target, float).copy()
            self._s.progress = float(progress)

    def set_result(self, result, error=None):
        with self._lock:
            self._s.result = result
            self._s.error = error

    def get(self) -> Status:
        with self._lock:
            return Status(self._s.result, self._s.phase, self._s.pos_ned.copy(), self._s.vel,
                          self._s.tilt_deg, self._s.mode, None if self._s.target is None
                          else self._s.target.copy(), self._s.progress, self._s.error)


class Mission:
    """A flight running on a background thread. run_fn(abort_evt, pause_evt, box) -> Result must
    honor the events and write progress to box; its return value becomes the final result."""

    def __init__(self, run_fn, abort_evt, pause_evt, box):
        self._run_fn = run_fn
        self._abort_evt = abort_evt
        self._pause_evt = pause_evt
        self._box = box
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            result = self._run_fn(self._abort_evt, self._pause_evt, self._box)
        except Exception as e:                       # surface loop crashes as ERROR, don't kill thread
            import sys, traceback
            traceback.print_exc(file=sys.stderr)     # also log -- fire-and-forget callers never poll status()
            self._box.set_result(Result.ERROR, error=repr(e))
            return
        self._box.set_result(result if isinstance(result, Result) else Result.REACHED)

    def start(self):
        self._thread.start()
        return self

    def abort(self):
        self._abort_evt.set()

    def pause(self):
        self._pause_evt.set()

    def resume(self):
        self._pause_evt.clear()

    def status(self) -> Status:
        return self._box.get()

    def wait(self, timeout=None) -> Status:
        self._thread.join(timeout)
        return self._box.get()

    @property
    def done(self) -> bool:
        return not self._thread.is_alive()

    @property
    def result(self):
        return self._box.get().result


# ---------------------------------------------------------------------------
# Result mapper
# ---------------------------------------------------------------------------

def _result_from_str(s):
    return {"reached": Result.REACHED, "timeout": Result.TIMEOUT,
            "abort": Result.ABORT}.get(s, Result.ERROR)


# ---------------------------------------------------------------------------
# Drone facade
# ---------------------------------------------------------------------------

class Drone:
    """Public control facade. Janahan owns connect+arm; pass the live store/commander/plant in."""

    def __init__(self, store, commander, plant, *, config: FlightConfig = None, flog=None,
                 trpy_minv=None):
        """trpy_minv (4x3, from navigator.calibrate_trpy_mixer): enables the DIRECT-TRPY inner loop --
        flies on raw motors via our own attitude PD instead of the sim's explosive rate loop (~3-5x
        tighter, speed-invariant tracking; COR-138). Calibrate it before constructing the Drone (it
        needs the live sim + resets, which the caller owns)."""
        self.config = config or FlightConfig()
        cmd = TrpyCommander(commander) if trpy_minv is not None else commander
        self.nav = WaypointNavigator(store, cmd, plant, gains=self.config.to_navgains(),
                                     flog=flog, trpy_minv=trpy_minv)
        self._mission = None

    def set_origin(self, pos_ned=None, yaw=None):
        """Set the NED origin (home) and optional heading."""
        self.nav.set_origin(pos_ned, yaw)

    def look_at(self, point, frame="body"):
        """Aim the camera HEADING at a coordinate (body or world); used when yaw='lookat'.

        LIMITATION: YAW-ONLY. The camera is body-bolted at a fixed 20 deg up-pitch and the airframe
        pitches/rolls for FLIGHT, not aim, so only the heading (bearing) tracks the target --
        elevation and roll follow the body. A point stays roughly IN the FOV but is not centered and
        can leave it during maneuvers; true 3-axis lock needs a gimbal the sim lacks. Best hold: view
        from the camera's elevation (~tan(20 deg)*range below the target) and fly gently (low
        speed/tilt). See examples/look_at_gate.py."""
        _check_frame(frame)
        self.nav.set_look_point(self.nav._resolve(_check_wp(point, "look_at point"), frame))

    # --- mission plumbing ---
    def _start(self, run_fn) -> Mission:
        if self._mission is not None and not self._mission.done:
            self._mission.abort()
            self._mission.wait(timeout=2.0)  # generous backstop: nav loops poll abort_evt every ~4 ms (~250 Hz), so the old worker exits well under 2 s; timeout only fires for a pathologically blocked loop
        abort_evt, pause_evt, box = threading.Event(), threading.Event(), _StatusBox()
        self.nav._abort_evt = abort_evt
        self.nav._pause_evt = pause_evt
        self.nav._status_cb = lambda *, phase, ds, tilt, mode, target, progress: box.update(
            phase=phase, ds_pos=ds.pos_ned, vel=float(np.linalg.norm(ds.vel_ned[:2])),
            tilt=tilt, mode=mode, target=target, progress=progress)
        m = Mission(run_fn, abort_evt, pause_evt, box).start()
        self._mission = m
        return m

    def _navigate(self, targets, *, yaw, look_at, speed, frame, stop, single):
        _check_frame(frame); _check_yaw(yaw)
        if not targets:
            raise ValueError("need at least one waypoint")
        targets = [_check_wp(t) for t in targets]
        if speed is not None:
            _check_positive("speed", speed)
        if yaw == "lookat" and look_at is None and self.nav._look_point is None:
            raise ValueError("yaw='lookat' requires look_at=... or a prior look_at()")
        v = speed if speed is not None else self.config.default_speed

        def run_fn(abort_evt, pause_evt, box):
            # Mutate shared nav state on the WORKER, after _start has aborted+joined the prior
            # mission -- setting these on the caller thread first leaks them into the dying flight.
            if look_at is not None:
                self.look_at(look_at, frame)
            self.nav._stop_each = bool(stop)
            res = (self.nav.goto(targets[0], yaw=yaw, v_cruise=v, engine="legs", frame=frame)
                   if single else
                   self.nav.follow(targets, yaw=yaw, v_cruise=v, engine="legs", frame=frame))
            return _result_from_str(res)
        return self._start(run_fn)

    # --- public motion ---
    def goto(self, wp, *, yaw="course", look_at=None, speed=None, frame="body", stop=False) -> Mission:
        """Fly to a waypoint; returns Mission (call .wait() to block)."""
        return self._navigate([wp], yaw=yaw, look_at=look_at, speed=speed, frame=frame,
                              stop=stop, single=True)

    def follow(self, wps, *, yaw="course", look_at=None, speed=None, frame="body", stop=False) -> Mission:
        """Fly through a sequence of waypoints; returns Mission (call .wait() to block)."""
        return self._navigate(list(wps), yaw=yaw, look_at=look_at, speed=speed, frame=frame,
                              stop=stop, single=False)

    def orbit(self, center, *, radius, speed=None, seconds, frame="body", direction="ccw") -> Mission:
        """Circle a point SMOOTHLY at a fixed radius; camera HEADING locked on center (yaw-only, same
        limitation as look_at -- not a true 3-axis lock); returns Mission. `seconds` is the target
        duration but the path flies a whole number of laps, so it is QUANTIZED: laps =
        round(seconds * speed / circumference), min 1. A request that works out to <1.5 laps flies
        exactly one full lap. The actual duration is ~laps*circumference/speed. All laps fly as ONE
        continuous follow() pass (settle only at the very end -- no per-lap stop). The ring is densely
        sampled (36 pts/lap) so the continuous-flow CAPTURE rounds to a smooth circle (~0.2 m inward
        on a 4 m radius) while the along-track speed law still reaches v_cruise."""
        _check_frame(frame); _check_positive("radius", radius); _check_positive("seconds", seconds)
        center = _check_wp(center, "orbit center")
        if speed is not None:
            _check_positive("speed", speed)
        if direction not in ("cw", "ccw"):
            raise ValueError("direction must be 'cw' or 'ccw'")
        v = speed if speed is not None else self.config.default_speed
        center_w = self.nav._resolve(center, frame)
        n = 36
        circ = 2.0 * np.pi * radius
        laps = max(1, int(round(seconds * v / max(circ, 1e-6))))
        ring = _orbit_ring(center_w, radius, n, direction) * laps   # list*int -> `laps` full circles, one pass

        def run_fn(abort_evt, pause_evt, box):
            self.nav.set_look_point(center_w)                     # on the worker (after prior mission joins)
            self.nav._stop_each = False                            # continuous: don't settle at every point
            return _result_from_str(
                self.nav.follow(ring, yaw="lookat", v_cruise=v, engine="legs", frame="world"))
        return self._start(run_fn)

    def hover(self, seconds=None, *, yaw="hold", look_at=None, frame="body") -> Mission:
        """Actively hold position for a duration (None = indefinite); returns Mission. Streams a
        station-keep command every loop -- a still drone keeps getting setpoints (no FC failsafe).
        `yaw` picks the camera behaviour while holding ('hold' = spawn heading, 'lookat' = camera on
        `look_at`/the prior look_at(), 'face', 'course'). lookat/face aim only works once a prior
        flight leg has locked the lateral sign -- hover from a fresh spawn can't probe it, so fly a
        leg first (e.g. goto(..., yaw='lookat')) then hover(yaw='lookat')."""
        if seconds is not None:
            _check_positive("seconds", seconds)
        _check_yaw(yaw); _check_frame(frame)
        if yaw == "lookat" and look_at is None and self.nav._look_point is None:
            raise ValueError("yaw='lookat' requires look_at=... or a prior look_at()")

        def run_fn(abort_evt, pause_evt, box):
            if look_at is not None:
                self.look_at(look_at, frame)
            return _result_from_str(self.nav.hover_hold(seconds, abort_evt, pause_evt, yaw=yaw))
        return self._start(run_fn)

    def takeoff(self, altitude) -> Mission:
        """Climb vertically by the given altitude; returns Mission."""
        _check_positive("altitude", altitude)
        pos = self.nav._current_pos()
        return self._navigate([np.array([pos[0], pos[1], pos[2] - altitude])], yaw="hold",
                              look_at=None, speed=None, frame="world", stop=True, single=True)

    def descend(self, altitude) -> Mission:
        """Descend vertically by the given altitude; returns Mission."""
        _check_positive("altitude", altitude)
        pos = self.nav._current_pos()
        return self._navigate([np.array([pos[0], pos[1], pos[2] + altitude])], yaw="hold",
                              look_at=None, speed=None, frame="world", stop=True, single=True)

    land = descend
