"""Fly a pattern around gate 0 and capture an auto-labeled dataset."""
from __future__ import annotations

import argparse
import time

import numpy as np

from .acquire import acquire_gates
from .commander import Commander
from .guidance import ApproachPattern, OrbitPattern
from .geometry import K
from .io_layer import MavlinkIO, VisionIO
from .state import Store


def _parse_gate(s):
    if not s:
        return None
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", choices=["orbit", "approach"], default="orbit")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--max-speed", type=float, default=5.0, help="cap on commanded velocity")
    ap.add_argument("--height", type=float, default=-2.0, help="target NED z (down<0=up)")
    ap.add_argument("--capture-hz", type=float, default=15.0)
    ap.add_argument("--control-hz", type=float, default=50.0)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--gate", default=None, help="fallback gate N,E,D if no broadcast")
    args = ap.parse_args()

    store = Store()
    mav = MavlinkIO(store)
    print("waiting for heartbeat...", flush=True)
    if mav.wait_heartbeat(10) is None:
        raise SystemExit("No heartbeat — is the sim in an active flight?")
    boot_ms = int(time.time() * 1000)
    mav.start()
    VisionIO(store).start()
    cmd = Commander(mav.conn, boot_ms)

    print("acquiring gates...", flush=True)
    gates = acquire_gates(store, cmd, gate_cli=_parse_gate(args.gate))
    gate = gates[0]
    print(f"gate 0 @ NED {gate.pos_ned}", flush=True)

    if args.pattern == "orbit":
        pattern = OrbitPattern(radius=args.radius, speed=args.speed, target_z=args.height,
                               max_speed=args.max_speed)
    else:
        offs = [(args.radius, 0, 0), (0, args.radius, 0),
                (-args.radius, 0, 0), (0, -args.radius, 0)]
        pattern = ApproachPattern(offsets=offs, speed=args.speed)

    # lazy import to avoid hard cv2 dep in unit tests
    from .logger import DataLogger
    logger = DataLogger(args.out, args.run_id, meta={
        "K": K.tolist(), "pattern": args.pattern, "radius": args.radius,
        "speed": args.speed, "height": args.height, "gate0_ned": gate.pos_ned.tolist(),
    })

    cmd.arm()
    print("armed; flying + capturing. Ctrl-C to stop.", flush=True)

    control_dt = 1.0 / args.control_hz
    capture_dt = 1.0 / args.capture_hz
    t_end = time.time() + args.duration
    next_capture = time.time()
    last_seq = -1
    try:
        while time.time() < t_end:
            ds = store.get_drone()
            if ds is not None:
                sp = pattern.update(ds.pos_ned, ds.vel_ned, gate.pos_ned)
                cmd.send_velocity_setpoint(sp.vx, sp.vy, sp.vz, sp.yaw)

            now = time.time()
            if now >= next_capture and ds is not None:
                frame, seq = store.get_frame()
                if frame is not None and seq != last_seq:
                    img, t_ns = frame
                    logger.log(img, t_ns, ds, gate)
                    last_seq = seq
                next_capture = now + capture_dt

            time.sleep(control_dt)
    except KeyboardInterrupt:
        pass
    finally:
        logger.close()
        print(f"done. dataset at {args.out}/{args.run_id}", flush=True)


if __name__ == "__main__":
    main()
