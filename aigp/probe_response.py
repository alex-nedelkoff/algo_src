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


def run_probe(store, commander, out_dir="sysid", run_id="probe", dt=0.05,
              hover_guess=0.5, reset_first=True, **kw):
    """Run the excitation probe inside ONE live race.

    Critical: the sim DQs + closes if we actuate during the countdown, so we
    SIM_RESET (optional) then wait for the race to go live before sending any
    command. No per-segment resets (each would re-enter a countdown). On
    divergence we abort the whole probe (no reset)."""
    from .race import wait_for_race_live
    sched = build_schedule(dt=dt, hover_guess=hover_guess, **kw)
    root = Path(out_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / "log.jsonl", "w")

    if reset_first:
        commander.sim_reset()              # passive command, safe; starts a countdown
    if not wait_for_race_live(store, timeout=15.0):
        f.close()
        raise RuntimeError("race never went live (countdown) within timeout")

    start = store.get_drone()
    z0 = start.pos_ned[2] if start else 0.0
    settle_n = max(1, int(0.5 / dt))
    aborted = False
    for seg in sched:
        if aborted:
            break
        # settle at hover guess between segments (no reset)
        for _ in range(settle_n):
            commander.send_attitude_target([0.0, 0.0, 0.0], hover_guess)
            time.sleep(dt)
        for cmd in seg["commands"]:
            commander.send_attitude_target(cmd["rates"], cmd["thrust"])
            time.sleep(dt)
            ds = store.get_drone()
            if ds is None:
                continue
            if _tilt_deg(ds.quat_wxyz) > TILT_ABORT_DEG or ds.pos_ned[2] - z0 > ALT_FLOOR_M:
                aborted = True             # safety: stop the whole probe
                break
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
    print(f"probe log at {root/'log.jsonl'}{' (ABORTED on divergence)' if aborted else ''}",
          flush=True)
    return str(root / "log.jsonl")
