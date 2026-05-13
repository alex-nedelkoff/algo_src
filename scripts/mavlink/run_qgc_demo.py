"""Stand up MavlinkShim with the QGC-friendly default message set, hold open.

Unlike `smoke_test.py`, this script does NOT override `rates_hz`, so the
shim emits the full set of messages QGC needs to render a vehicle
(LOCAL_POSITION_NED, SYS_STATUS, GPS_RAW_INT, GLOBAL_POSITION_INT,
HOME_POSITION, VFR_HUD, plus the COR-98 baseline).

Usage:
    conda run -n monorace python -m scripts.mavlink.run_qgc_demo \\
        [--port 14550] [--backend numpy_quad|pybullet]

In QGroundControl: Application Settings -> Comm Links -> Add UDP,
Listening Port = 14540 (or anything other than --port), Server Addresses
= 127.0.0.1:14550. Connect. The vehicle should appear within a few seconds.
"""
from __future__ import annotations

import argparse

import numpy as np

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--backend", choices=["numpy_quad", "pybullet"], default="numpy_quad")
    args = ap.parse_args()

    params = VehicleParams()
    if args.backend == "numpy_quad":
        backend = NumpyQuadBackend(params=params)
    else:
        from sim.pybullet.mavlink_shim.pybullet_backend import PyBulletBackend
        backend = PyBulletBackend(params=params, gui=False)

    shim = MavlinkShim(
        backend=backend, params=params,
        host="0.0.0.0", port=args.port, lockstep=False,
    )
    hover_w = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover_w),
        timestamp_us=0,
    )
    with shim:
        shim.reset(initial)
        print(
            f"MavlinkShim listening on UDP :{args.port} "
            f"(backend={args.backend}). Open QGC, target 127.0.0.1:{args.port}. "
            f"Ctrl-C to stop.",
            flush=True,
        )
        try:
            shim.wait_for_shutdown()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
