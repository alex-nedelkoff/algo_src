"""Manual smoke test for the MAVLink shim.

Spins up MavlinkShim in free-running mode, connects a pymavlink client,
streams a sequence of attitude targets, and plots commanded vs achieved
attitude over time.

Usage:
    conda run -n monorace python -m scripts.mavlink.smoke_test \\
        [--port 14550] [--out outputs/mavlink_smoke/attitude.png]

Output: PNG with two subplots (pitch / roll) showing commanded vs
achieved trajectories.

Bonus manual check: while this script is running (or with --hold to keep
the shim alive after plotting), point QGroundControl at udp:127.0.0.1:14550
and confirm the vehicle appears with non-zero telemetry.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.pybullet.mavlink_shim.coords_mavlink import enu_quat_to_ned_euler

from pymavlink import mavutil


def _hover_thrust(params: VehicleParams) -> float:
    weight_n = params.mass * 9.81
    max_thrust_per_motor = params.k_thrust * (params.max_omega ** 2)
    return weight_n / (4.0 * max_thrust_per_motor)


def _hover_motor_speed(params: VehicleParams) -> float:
    return float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=14550)
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/mavlink_smoke/attitude.png"))
    ap.add_argument("--hold", action="store_true",
                    help="Keep shim alive after plotting for QGC manual check")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    params = VehicleParams()
    backend = NumpyQuadBackend(params=params)
    shim = MavlinkShim(
        backend=backend, params=params,
        host="127.0.0.1", port=args.port, lockstep=False,
        rates_hz={"heartbeat": 1, "attitude": 100, "odometry": 100, "highres_imu": 200},
        controller_k_att=5.0,
        controller_k_damp=2.0,   # rate damping — prevents P-only overshoot in real-time mode
    )

    initial = DroneState(
        pos_enu=np.zeros(3), vel_enu=np.zeros(3),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, _hover_motor_speed(params)),
        timestamp_us=0,
    )

    thrust = _hover_thrust(params)

    sequence = [
        # (q_target_ned_wxyz, duration_s, label)
        ((1.0, 0.0, 0.0, 0.0), 2.0, "hover"),
        ((np.cos(np.pi/12), 0.0, np.sin(np.pi/12), 0.0), 2.0, "+30 pitch"),
        ((1.0, 0.0, 0.0, 0.0), 2.0, "hover"),
        ((np.cos(np.pi/12), 0.0, -np.sin(np.pi/12), 0.0), 2.0, "-30 pitch"),
        ((1.0, 0.0, 0.0, 0.0), 2.0, "hover"),
    ]

    cmd_pitch_log: list[tuple[float, float]] = []
    ach_pitch_log: list[tuple[float, float]] = []

    with shim:
        shim.reset(initial)
        client = mavutil.mavlink_connection(
            f"udpout:127.0.0.1:{args.port}",
            source_system=255, source_component=0,
        )
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.1)  # let shim register client address

        # Background thread: receive ATTITUDE messages.
        stop_recv = threading.Event()
        t_start = time.monotonic()

        def _recv_loop() -> None:
            while not stop_recv.is_set():
                msg = client.recv_match(type="ATTITUDE", blocking=True, timeout=0.05)
                if msg is None:
                    continue
                t = time.monotonic() - t_start
                ach_pitch_log.append((t, np.degrees(msg.pitch)))

        recv_thread = threading.Thread(target=_recv_loop, daemon=True)
        recv_thread.start()

        for q_target, dur, _label in sequence:
            cmd_pitch_deg = float(np.degrees(2.0 * np.arctan2(q_target[2], q_target[0])))
            t_seg_end = time.monotonic() + dur
            while time.monotonic() < t_seg_end:
                client.mav.set_attitude_target_send(
                    time_boot_ms=0, target_system=1, target_component=1, type_mask=0,
                    q=list(q_target),
                    body_roll_rate=0.0, body_pitch_rate=0.0, body_yaw_rate=0.0,
                    thrust=float(thrust),
                )
                cmd_pitch_log.append((time.monotonic() - t_start, cmd_pitch_deg))
                time.sleep(0.01)

        stop_recv.set()
        recv_thread.join(timeout=1.0)

        if args.hold:
            print(f"Holding shim alive on UDP port {args.port}. "
                  f"Connect QGroundControl to udp:127.0.0.1:{args.port}.")
            print("Press Ctrl-C to exit.")
            try:
                shim.wait_for_shutdown()
            except KeyboardInterrupt:
                pass

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmd_t, cmd_p = zip(*cmd_pitch_log) if cmd_pitch_log else ([], [])
    ach_t, ach_p = zip(*ach_pitch_log) if ach_pitch_log else ([], [])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cmd_t, cmd_p, "k--", label="commanded pitch")
    ax.plot(ach_t, ach_p, "b-", label="achieved pitch")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch (deg)")
    ax.set_title("MAVLink shim — attitude tracking smoke test")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
