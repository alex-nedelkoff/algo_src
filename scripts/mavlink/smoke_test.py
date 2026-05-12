"""Manual smoke test for the MAVLink shim.

Spins up MavlinkShim in free-running mode, connects a pymavlink client,
streams a sequence of attitude targets, and plots commanded vs achieved
attitude over time.

Usage:
    conda run -n monorace python -m scripts.mavlink.smoke_test \\
        [--port 14550] [--out outputs/mavlink_smoke/attitude.png]
        [--backend numpy_quad|pybullet] [--mode attitude|position]

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
    ap.add_argument(
        "--backend", choices=["numpy_quad", "pybullet"], default="numpy_quad",
        help="Physics backend to drive the shim with.",
    )
    ap.add_argument(
        "--mode", choices=["attitude", "position"], default="attitude",
        help="Client-side command mode: SET_ATTITUDE_TARGET vs SET_POSITION_TARGET_LOCAL_NED.",
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    params = VehicleParams()
    if args.backend == "numpy_quad":
        backend = NumpyQuadBackend(params=params)
    else:
        from sim.pybullet.mavlink_shim.pybullet_backend import PyBulletBackend
        backend = PyBulletBackend(params=params, gui=False)

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

    sequence_attitude = [
        # (q_target_ned_wxyz, duration_s, label)
        ((1.0, 0.0, 0.0, 0.0), 2.0, "hover"),
        ((np.cos(np.pi/12), 0.0, np.sin(np.pi/12), 0.0), 2.0, "+30 pitch"),
        ((1.0, 0.0, 0.0, 0.0), 2.0, "hover"),
        ((np.cos(np.pi/12), 0.0, -np.sin(np.pi/12), 0.0), 2.0, "-30 pitch"),
        ((1.0, 0.0, 0.0, 0.0), 2.0, "hover"),
    ]

    cmd_pitch_log: list[tuple[float, float]] = []
    ach_pitch_log: list[tuple[float, float]] = []
    pos_log: list[tuple[float, np.ndarray]] = []  # (t, pos_enu) for position-mode

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

        if args.mode == "attitude":
            for q_target, dur, _label in sequence_attitude:
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
        else:
            # Position mode: hold (5, 0, -2) NED for 10 s, log current pos.
            t_seg_end = time.monotonic() + 10.0
            while time.monotonic() < t_seg_end:
                client.mav.set_position_target_local_ned_send(
                    time_boot_ms=int((time.monotonic() - t_start) * 1000),
                    target_system=1, target_component=1,
                    coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    type_mask=0,
                    x=5.0, y=0.0, z=-2.0,
                    vx=0.0, vy=0.0, vz=0.0,
                    afx=0.0, afy=0.0, afz=0.0,
                    yaw=0.0, yaw_rate=0.0,
                )
                t_log = time.monotonic() - t_start
                pos_log.append((t_log, shim._last_state.pos_enu.copy()))
                time.sleep(0.05)

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

    if args.mode == "attitude":
        cmd_t, cmd_p = zip(*cmd_pitch_log) if cmd_pitch_log else ([], [])
        ach_t, ach_p = zip(*ach_pitch_log) if ach_pitch_log else ([], [])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cmd_t, cmd_p, "k--", label="commanded pitch")
        ax.plot(ach_t, ach_p, "b-", label="achieved pitch")
        ax.set_xlabel("time (s)"); ax.set_ylabel("pitch (deg)")
        ax.set_title(f"MAVLink shim — attitude tracking ({args.backend})")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"Wrote {args.out}")
    else:
        if not pos_log:
            print("No position-log entries; nothing to plot.")
        else:
            ts, ps = zip(*pos_log)
            ps = np.array(ps)
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            labels = ["x (East, m)", "y (North, m)", "z (Up, m)"]
            targets = [0.0, 5.0, 2.0]
            for i, (ax, label, target) in enumerate(zip(axes, labels, targets)):
                ax.plot(ts, ps[:, i], "b-", label="position")
                ax.axhline(target, color="k", ls="--", label="target")
                ax.set_ylabel(label); ax.grid(True, alpha=0.3); ax.legend(loc="best")
            axes[-1].set_xlabel("time (s)")
            axes[0].set_title(f"MAVLink shim — position tracking ({args.backend})")
            fig.tight_layout()
            fig.savefig(args.out, dpi=120)
            print(f"Wrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
