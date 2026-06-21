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
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from aigp.navigator import NavGains, WaypointNavigator


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
