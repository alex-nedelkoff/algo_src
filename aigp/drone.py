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

import threading
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

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
    """Flight envelope. Defaults are the proven-safe values (tilt <=20, ~5 m/s -> <1 m z-sag,
    no ground-crash). Raising amax/tilt past ~25 deg risks the high-tilt z-sag into terrain."""
    vmax: float = 6.0
    amax: float = 4.0
    tilt_deg: float = 20.0
    zff: float = -2.4
    capture: float = 2.5
    kiz: float = 0.8
    c_max: float = 18.0
    default_speed: float = 5.0

    def to_navgains(self) -> NavGains:
        g = NavGains()
        g.MAX_SPEED = self.vmax
        g.VLAT_MAX = self.vmax
        g.FWD_AMAX = self.amax
        g.DECEL_MAX = max(g.DECEL_MAX, self.amax)
        g.TILT_MAX_DEG = self.tilt_deg
        g.Z_FF = self.zff
        g.CAPTURE = self.capture
        g.KI_Z = self.kiz
        g.C_MAX = self.c_max
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
    if not (isinstance(val, (int, float)) and val > 0):
        raise ValueError(f"{name} must be > 0, got {val!r}")


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

    def __init__(self, store, commander, plant, *, config: FlightConfig = None, flog=None):
        self.config = config or FlightConfig()
        self.nav = WaypointNavigator(store, commander, plant,
                                     gains=self.config.to_navgains(), flog=flog)
        self._mission = None

    def set_origin(self, pos_ned=None, yaw=None):
        """Set the NED origin (home) and optional heading."""
        self.nav.set_origin(pos_ned, yaw)

    def look_at(self, point, frame="body"):
        """Point the camera at a given coordinate (body or world frame)."""
        _check_frame(frame)
        self.nav.set_look_point(self.nav._resolve(np.asarray(point, float), frame))

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
        if speed is not None:
            _check_positive("speed", speed)
        if yaw == "lookat" and look_at is None and self.nav._look_point is None:
            raise ValueError("yaw='lookat' requires look_at=... or a prior look_at()")
        v = speed if speed is not None else self.config.default_speed
        if look_at is not None:
            self.look_at(look_at, frame)
        self.nav._stop_each = bool(stop)

        def run_fn(abort_evt, pause_evt, box):
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
        """Circle a point at a fixed radius for a duration; camera locked on center; returns Mission."""
        _check_frame(frame); _check_positive("radius", radius); _check_positive("seconds", seconds)
        if speed is not None:
            _check_positive("speed", speed)
        if direction not in ("cw", "ccw"):
            raise ValueError("direction must be 'cw' or 'ccw'")
        v = speed if speed is not None else self.config.default_speed
        center_w = self.nav._resolve(np.asarray(center, float), frame)
        self.nav.set_look_point(center_w)
        ring = _orbit_ring(center_w, radius, 24, direction)

        def run_fn(abort_evt, pause_evt, box):
            t0 = time.time()
            while time.time() - t0 < seconds and not abort_evt.is_set():
                res = self.nav.follow(ring, yaw="lookat", v_cruise=v, engine="legs", frame="world")
                if res == "abort":
                    return Result.ABORT
            return Result.ABORT if abort_evt.is_set() else Result.REACHED
        return self._start(run_fn)

    def hover(self, seconds=None) -> Mission:
        """Hold position for a duration (None = indefinite); returns Mission."""
        if seconds is not None:
            _check_positive("seconds", seconds)

        def run_fn(abort_evt, pause_evt, box):
            t0 = time.time()
            while not abort_evt.is_set() and (seconds is None or time.time() - t0 < seconds):
                self.nav.settle()      # holds current position ~SETTLE_T; loop re-holds
            return Result.ABORT if abort_evt.is_set() else Result.REACHED
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
