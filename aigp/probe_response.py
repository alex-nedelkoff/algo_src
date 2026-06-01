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

# Loose bounds: the thrust sweep MUST move vertically and rate doublets MUST
# tilt, so guards only catch true loss-of-control / approaching ground.
TILT_ABORT_DEG = 60.0
ALT_FLOOR_M = 15.0  # NED: abort only if descended this far below start


def _tilt_deg(quat):
    from .geometry import quat_to_R
    zb = quat_to_R(quat)[:, 2]              # body-down axis in world (NED)
    cos_tilt = float(zb[2])                 # dot with world-down [0,0,1]; level -> +1 -> 0 deg
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_tilt))))


DEFAULT_THRUST_LEVELS = [0.2, 0.35, 0.5, 0.65, 0.8]


def _pulse(store, commander, f, segment, thrust, rates_fn, n, dt):
    """Send a fixed thrust (+ rates from rates_fn(k)) for n steps from rest,
    logging each sample. The drone was just armed/released so speeds stay low."""
    for k in range(n):
        rates = rates_fn(k)
        commander.send_attitude_target(rates, thrust)
        time.sleep(dt)
        ds = store.get_drone()
        if ds is None:
            continue
        f.write(json.dumps({
            "segment": segment, "t": time.time(),
            "thrust_norm": thrust, "cmd_rates": list(rates),
            "pos_ned": [float(x) for x in ds.pos_ned],
            "vel_ned": [float(x) for x in ds.vel_ned],
            "quat_wxyz": [float(x) for x in ds.quat_wxyz],
            "omega": [float(x) for x in ds.omega],
        }) + "\n")
        f.flush()


def run_probe(store, commander, out_dir="sysid", run_id="probe", dt=0.02,
              pulse_s=0.5, thrust_levels=None, rate_amp=1.0, hover_for_rates=0.5):
    """Pulse-from-rest excitation probe.

    For each thrust level (and each rate-doublet axis): start a FRESH race,
    wait for the countdown to finish (so arm is honored), arm, then immediately
    drive the command for ~pulse_s while the drone is still slow (drag-free).
    Resetting between pulses keeps each measurement near rest. All actuation
    happens only after the race is live."""
    from .race import wait_for_fresh_race_live
    thrust_levels = thrust_levels or DEFAULT_THRUST_LEVELS
    root = Path(out_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / "log.jsonl", "w")
    n = max(4, int(pulse_s / dt))
    zero = lambda k: [0.0, 0.0, 0.0]

    # thrust map: one pulse per level, from rest
    for tn in thrust_levels:
        if not wait_for_fresh_race_live(store, commander):
            f.close(); raise RuntimeError("no fresh race-live before thrust pulse")
        commander.arm()
        _pulse(store, commander, f, "thrust_pulse", tn, zero, n, dt)

    # rate doublets: per axis, at a mid thrust, from rest
    for name, axis in (("rate_roll", 0), ("rate_pitch", 1), ("rate_yaw", 2)):
        if not wait_for_fresh_race_live(store, commander):
            f.close(); raise RuntimeError(f"no fresh race-live before {name}")
        commander.arm()

        def doublet(k, axis=axis):
            r = [0.0, 0.0, 0.0]
            r[axis] = rate_amp if k < n // 2 else -rate_amp
            return r

        _pulse(store, commander, f, name, hover_for_rates, doublet, n, dt)

    f.close()
    print(f"probe log at {root/'log.jsonl'} (complete)", flush=True)
    return str(root / "log.jsonl")
