"""Downstream (un-stubbed) probe: do bad coordinates produce NaN/inf attitude commands?
This is the live-safety question — a NaN target on a real drone = undefined rate/thr.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aigp.drone import Drone, FlightConfig, Result
from aigp.state import DroneState


class FakeStore:
    def __init__(self, ds): self._ds = ds
    def get_drone(self): return self._ds
    def get_race_live(self): return True


class RecCmd:
    def __init__(self): self.n = 0; self.bad = 0; self.first_bad = None
    def send_attitude_target(self, rate, thr):
        self.n += 1
        vals = list(np.asarray(rate, float).ravel()) + [float(thr)]
        if any((not np.isfinite(v)) for v in vals):
            self.bad += 1
            if self.first_bad is None: self.first_bad = (list(rate), thr)


def mk():
    ds = DroneState(np.array([0, 0, -2.0]), np.zeros(3), np.array([0.0, 1, 0, 0]), np.zeros(3), 0)
    cmd = RecCmd()
    d = Drone(FakeStore(ds), cmd, (0.5, 13.0, np.array([1.0, 1.0, 1.0])), config=FlightConfig())
    d.set_origin(pos_ned=np.zeros(3), yaw=0.0)
    d.nav.gains.WP_TIMEOUT = 0.25          # bound the spin
    return d, cmd


for label, wp, yaw in [("NaN coord", (float("nan"), 0, 0), "hold"),
                       ("inf coord", (float("inf"), 0, 0), "hold"),
                       ("2D coord", (1, 0), "hold"),
                       ("4D coord", (1, 0, 0, 0), "hold")]:
    d, cmd = mk()
    try:
        s = d.goto(wp, yaw=yaw).wait(timeout=3)
        flag = "DANGER" if cmd.bad else ("ok-finite" if cmd.n else "no-cmd")
        print(f"[{flag}] {label:12s} result={s.result.value:8s} cmds={cmd.n:4d} nonfinite={cmd.bad:4d}"
              + (f"  first_bad_rate={cmd.first_bad[0]} thr={cmd.first_bad[1]}" if cmd.first_bad else "")
              + (f"  err={s.error}" if s.error else ""))
    except Exception as e:
        print(f"[raise ] {label:12s} {type(e).__name__}: {str(e)[:60]}")
