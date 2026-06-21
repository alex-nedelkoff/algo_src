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
