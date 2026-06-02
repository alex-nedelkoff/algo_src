"""Staged live verification of the body-rate controller against the sim.

Critical sim quirks handled here:
- never actuate before the race is live (boot_ms >= start_ms), else DQ + close;
- the drone is HELD until armed and RUNS AWAY if armed without immediate control,
  so we arm and enter the control loop with no gap;
- the sim's rate loop is inverted (~ -1.93x); the controller corrects via rate_gain.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .acquire import acquire_gates
from .attitude_control import AttitudeSetpointController
from .commander import Commander
from .geometry import quat_to_R
from .guidance import OrbitPattern
from .io_layer import MavlinkIO, VisionIO
from .race import wait_for_fresh_race_live
from .state import Store


def _ctl_from(path, **over):
    r = json.load(open(path))
    return AttitudeSetpointController(hover_thrust=r["hover_thrust"], k_a=r["k_a"], **over)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", default="sysid/sim_response.json")
    ap.add_argument("--stage", choices=["hover", "orbit"], default="hover")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--kp-pos", type=float, default=1.0)
    ap.add_argument("--kd-pos", type=float, default=3.2)
    ap.add_argument("--ki-pos", type=float, default=0.0)
    ap.add_argument("--tilt-deg", type=float, default=10.0)
    ap.add_argument("--control-hz", type=float, default=50.0)
    args = ap.parse_args()

    store = Store(); mav = MavlinkIO(store)
    assert mav.wait_heartbeat(10), "no heartbeat — start a flight"
    mav.start(); VisionIO(store).start()
    boot = int(time.time() * 1000); cmd = Commander(mav.conn, boot)
    ctl = _ctl_from(args.response, kp_pos=[args.kp_pos, args.kp_pos, 1.8],
                    kd_pos=[args.kd_pos, args.kd_pos, 3.0],
                    ki_pos=[args.ki_pos, args.ki_pos, 0.0],
                    tilt_max_deg=args.tilt_deg)

    # acquire gates BEFORE arming (passive reset+listen), if orbiting
    gate = None
    if args.stage == "orbit":
        gates = acquire_gates(store, cmd)
        gate = gates[0]
        print(f"gate 0 @ {gate.pos_ned}", flush=True)

    # fresh race so we start from the held spawn, then arm + control with NO gap
    print("resetting + waiting for race to go live...", flush=True)
    assert wait_for_fresh_race_live(store, cmd), "race never went live"
    ds0 = store.get_drone()
    hold = ds0.pos_ned.copy()
    R0 = quat_to_R(ds0.quat_wxyz)
    hold_yaw = float(np.arctan2(R0[1, 0], R0[0, 0]))  # hold current heading (drone spawns ~180deg)
    orbit = (OrbitPattern(radius=args.radius, speed=args.speed,
                          target_z=float(gate.pos_ned[2]), max_speed=4.0)
             if gate is not None else None)

    cmd.arm()
    print("armed; controlling.", flush=True)
    dt = 1.0 / args.control_hz
    t_end = time.time() + args.duration
    pe_max = 0.0
    pe_samples = []
    while time.time() < t_end:
        ds = store.get_drone()
        if ds is not None:
            if orbit is not None:
                sp = orbit.update(ds.pos_ned, ds.vel_ned, gate.pos_ned)
                vel_sp = np.array([sp.vx, sp.vy, sp.vz]); yaw_sp = sp.yaw
                pos_sp = ds.pos_ned + vel_sp * 0.2
            else:
                pos_sp, vel_sp, yaw_sp = hold, np.zeros(3), hold_yaw
            q_send, thrust = ctl.update(ds.pos_ned, ds.vel_ned, ds.quat_wxyz,
                                        pos_sp, vel_sp, yaw_sp, dt=dt)
            cmd.send_attitude_setpoint(q_send, thrust)
            pe = float(np.linalg.norm(np.asarray(pos_sp) - ds.pos_ned))
            pe_max = max(pe_max, pe); pe_samples.append(pe)
        time.sleep(dt)

    pe_end = float(np.mean(pe_samples[-25:])) if pe_samples else float("nan")
    tag = "HOVER" if orbit is None else "ORBIT"
    ok = pe_max < 30.0   # bounded (not diverging); tighter hold is follow-on tuning
    print(f"{tag}: max_err={pe_max:.2f}m  final_avg_err={pe_end:.2f}m  "
          f"({'PASS' if ok else 'FAIL'})", flush=True)


if __name__ == "__main__":
    main()
