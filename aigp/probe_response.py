"""Open-loop excitation probe for sim-response characterization.

build_schedule() is pure (testable); run_probe() executes it live, logging
tagged samples and resetting the sim between segments / on safety breach.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path


def build_schedule(hover_guess=0.5, thrust_band=0.2, rate_amp=1.0, seg_dur=2.0, dt=0.05):
    """Return a list of segments: {name, commands:[{thrust, rates}]}."""
    n = max(1, int(seg_dur / dt))
    sched = []

    # thrust sweep: ramp thrust across the band, rates zero
    sweep = []
    for k in range(n):
        frac = k / (n - 1) if n > 1 else 0.0
        thrust = hover_guess - thrust_band + 2 * thrust_band * frac
        sweep.append({"thrust": thrust, "rates": [0.0, 0.0, 0.0]})
    sched.append({"name": "thrust_sweep", "commands": sweep})

    # rate doublets per axis at hover thrust
    for name, axis in (("rate_roll", 0), ("rate_pitch", 1), ("rate_yaw", 2)):
        cmds = []
        for k in range(n):
            sign = 1.0 if k < n // 2 else -1.0
            rates = [0.0, 0.0, 0.0]
            rates[axis] = sign * rate_amp
            cmds.append({"thrust": hover_guess, "rates": rates})
        sched.append({"name": name, "commands": cmds})

    return sched


# ---- live runner (no unit test) ----

TILT_ABORT_DEG = 30.0
ALT_FLOOR_M = 0.5  # NED: abort if z (down) exceeds this below start


def _tilt_deg(quat):
    from .geometry import quat_to_R
    zb = quat_to_R(quat)[:, 2]              # body-down in world
    cos_tilt = float(-zb[2])               # vs world-down
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))


def run_probe(store, commander, out_dir="sysid", run_id="probe", dt=0.05, **kw):
    sched = build_schedule(dt=dt, **kw)
    root = Path(out_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / "log.jsonl", "w")
    for seg in sched:
        commander.sim_reset()
        time.sleep(1.0)                    # let the sim respawn
        start = store.get_drone()
        z0 = start.pos_ned[2] if start else 0.0
        for cmd in seg["commands"]:
            commander.send_attitude_target(cmd["rates"], cmd["thrust"])
            time.sleep(dt)
            ds = store.get_drone()
            if ds is None:
                continue
            if _tilt_deg(ds.quat_wxyz) > TILT_ABORT_DEG or ds.pos_ned[2] - z0 > ALT_FLOOR_M:
                break                      # safety: bail this segment
            f.write(json.dumps({
                "segment": seg["name"], "t": time.time(),
                "thrust_norm": cmd["thrust"], "cmd_rates": list(cmd["rates"]),
                "pos_ned": [float(x) for x in ds.pos_ned],
                "vel_ned": [float(x) for x in ds.vel_ned],
                "quat_wxyz": [float(x) for x in ds.quat_wxyz],
                "omega": [float(x) for x in ds.omega],
            }) + "\n")
            f.flush()
    f.close()
    commander.sim_reset()
    print(f"probe log at {root/'log.jsonl'}", flush=True)
    return str(root / "log.jsonl")
