"""Live motor-level excitation for dynamics sysID.

build_campaign() is pure (testable). run_campaign() executes it: fresh race-live +
arm, then drives motor patterns logging HIGHRES_IMU (specific force + gyro) and
ODOMETRY. Open-loop bare dynamics are unstable, so torque segments are short bursts
with SIM_RESET between; the collective segment is attitude-stable (longer)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# X-config differential mixes (sign hypotheses; calibration confirms actual signs)
_MIX = {
    "roll":  np.array([-1.0, 1.0, 1.0, -1.0]),
    "pitch": np.array([-1.0, -1.0, 1.0, 1.0]),
    "yaw":   np.array([1.0, -1.0, 1.0, -1.0]),
}


def build_campaign(n_motors=4, hover_guess=0.3, levels=3, axes=("roll", "pitch", "yaw"),
                   bump=0.15, diff=0.1):
    """Return a list of segments: {name, kind, commands:[u(list len n_motors)]}."""
    segs = []

    # collective sweep (thrust + power law), attitude-stable
    coll = []
    for k in range(levels):
        frac = k / (levels - 1) if levels > 1 else 0.0
        lvl = max(0.0, hover_guess - 0.1) + 0.3 * frac
        coll += [[lvl] * n_motors] * 8       # hold each level a few samples
    segs.append({"name": "collective", "kind": "collective", "commands": coll})

    # per-motor bumps (channel mapping + IMU axis calibration)
    for i in range(n_motors):
        u = [hover_guess] * n_motors
        u[i] = min(1.0, hover_guess + bump)
        segs.append({"name": f"bump_m{i}", "kind": "bump", "commands": [u] * 6})

    # per-axis differential doublets (torque coefficients)
    for ax in axes:
        cmds = []
        for k in range(8):
            sign = 1.0 if k < 4 else -1.0
            u = np.full(n_motors, hover_guess) + sign * diff * _MIX[ax][:n_motors]
            cmds.append([float(np.clip(v, 0.0, 1.0)) for v in u])
        segs.append({"name": f"diff_{ax}", "kind": "diff", "commands": cmds})

    return segs


def run_probe(store, commander, mav_conn, out_dir="sysid_dyn", run_id="dyn",
              dt=0.01, n_motors=4, hover_guess=0.3):
    """Execute the campaign live. Logs HIGHRES_IMU + ODOMETRY per command."""
    from .race import wait_for_fresh_race_live
    segs = build_campaign(n_motors=n_motors, hover_guess=hover_guess)
    root = Path(out_dir) / run_id
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / "log.jsonl", "w")
    for seg in segs:
        if not wait_for_fresh_race_live(store, commander):
            f.close(); raise RuntimeError("no fresh race-live before " + seg["name"])
        commander.arm()
        for u in seg["commands"]:
            commander.send_motor_command(u)
            time.sleep(dt)
            imu = mav_conn.recv_match(type="HIGHRES_IMU", blocking=False)
            ds = store.get_drone()
            if ds is None:
                continue
            row = {"segment": seg["name"], "t": time.time(), "u": list(u),
                   "pos_ned": [float(x) for x in ds.pos_ned],
                   "vel_ned": [float(x) for x in ds.vel_ned],
                   "quat_wxyz": [float(x) for x in ds.quat_wxyz],
                   "omega": [float(x) for x in ds.omega]}
            if imu is not None:
                row["imu_acc"] = [float(imu.xacc), float(imu.yacc), float(imu.zacc)]
                row["imu_gyro"] = [float(imu.xgyro), float(imu.ygyro), float(imu.zgyro)]
            f.write(json.dumps(row) + "\n")
            f.flush()
    f.close()
    print(f"dyn probe log at {root/'log.jsonl'}", flush=True)
    return str(root / "log.jsonl")
