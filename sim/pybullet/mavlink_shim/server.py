"""MavlinkServer — UDP MAVLink I/O wrapper using pymavlink.

Owns the socket, encodes/decodes messages, tracks the last-seen client
address (for the udpin-with-broadcast-back pattern). No control logic;
the MavlinkShim wires this to the AttitudeController + Backend.

pymavlink notes
---------------
ODOMETRY (msg id 331) is a MAVLink v2 message and is only present in the
v20 dialect modules. We must set MAVLINK20=1 before importing mavutil and
connect with dialect='ardupilotmega' (or 'common') to get the v20 code
path. The default mavutil import uses the v1.0 common dialect which lacks
ODOMETRY entirely.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Force MAVLink v2 protocol before any pymavlink import.
os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402 — must follow env-var set


_DIALECT = "ardupilotmega"


@dataclass
class MavlinkServer:
    host: str = "0.0.0.0"
    port: int = 14550
    sysid: int = 1
    compid: int = 1
    _conn: object = field(init=False, repr=False)
    _t0_us: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # "udpin" binds and waits; first incoming packet's address is captured
        # by pymavlink and outbound messages go back to that address.
        self._conn = mavutil.mavlink_connection(
            f"udpin:{self.host}:{self.port}",
            source_system=self.sysid,
            source_component=self.compid,
            dialect=_DIALECT,
        )
        self._t0_us = int(time.monotonic() * 1_000_000)

    def close(self) -> None:
        if hasattr(self._conn, "close"):
            self._conn.close()

    def recv_pending(self) -> list:
        """Drain all messages currently queued on the socket. Non-blocking."""
        out: list = []
        while True:
            msg = self._conn.recv_match(blocking=False)
            if msg is None:
                break
            out.append(msg)
        return out

    def _t_boot_ms(self) -> int:
        return (int(time.monotonic() * 1_000_000) - self._t0_us) // 1000

    def _t_usec(self) -> int:
        return int(time.monotonic() * 1_000_000) - self._t0_us

    def send_heartbeat(self) -> None:
        self._conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
            base_mode=0,
            custom_mode=0,
            system_status=mavutil.mavlink.MAV_STATE_ACTIVE,
        )

    def send_attitude(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        rollspeed: float,
        pitchspeed: float,
        yawspeed: float,
    ) -> None:
        self._conn.mav.attitude_send(
            self._t_boot_ms(),
            roll,
            pitch,
            yaw,
            rollspeed,
            pitchspeed,
            yawspeed,
        )

    def send_odometry(
        self,
        pos_ned: NDArray[np.float64],
        vel_ned: NDArray[np.float64],
        quat_ned_xyzw: NDArray[np.float64],
        angular_vel_body: NDArray[np.float64],
    ) -> None:
        zero_cov = [float("nan")] * 21  # mark covariance unknown
        # MAVLink ODOMETRY q is wxyz; input is xyzw → reorder.
        q_wxyz = [
            float(quat_ned_xyzw[3]),
            float(quat_ned_xyzw[0]),
            float(quat_ned_xyzw[1]),
            float(quat_ned_xyzw[2]),
        ]
        self._conn.mav.odometry_send(
            time_usec=self._t_usec(),
            frame_id=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            child_frame_id=mavutil.mavlink.MAV_FRAME_BODY_FRD,
            x=float(pos_ned[0]),
            y=float(pos_ned[1]),
            z=float(pos_ned[2]),
            q=q_wxyz,
            vx=float(vel_ned[0]),
            vy=float(vel_ned[1]),
            vz=float(vel_ned[2]),
            rollspeed=float(angular_vel_body[0]),
            pitchspeed=float(angular_vel_body[1]),
            yawspeed=float(angular_vel_body[2]),
            pose_covariance=zero_cov,
            velocity_covariance=zero_cov,
            reset_counter=0,
            estimator_type=mavutil.mavlink.MAV_ESTIMATOR_TYPE_NAIVE,
            quality=100,
        )

    def send_highres_imu(
        self,
        accel_body: NDArray[np.float64],
        gyro_body: NDArray[np.float64],
    ) -> None:
        # Per MAVLink spec: bit 0=xacc, 1=yacc, 2=zacc, 3=xgyro, 4=ygyro,
        # 5=zgyro. Set all six bits to indicate all fields are valid.
        accel_gyro_bits = 0x3F
        self._conn.mav.highres_imu_send(
            time_usec=self._t_usec(),
            xacc=float(accel_body[0]),
            yacc=float(accel_body[1]),
            zacc=float(accel_body[2]),
            xgyro=float(gyro_body[0]),
            ygyro=float(gyro_body[1]),
            zgyro=float(gyro_body[2]),
            xmag=0.0,
            ymag=0.0,
            zmag=0.0,
            abs_pressure=0.0,
            diff_pressure=0.0,
            pressure_alt=0.0,
            temperature=0.0,
            fields_updated=accel_gyro_bits,
        )

    def send_timesync(self, tc1: int, ts1: int) -> None:
        """Send a TIMESYNC message.

        Server-initiated periodic: tc1=our_time_ns, ts1=0.
        Reply to inbound TIMESYNC: tc1=our_time_ns, ts1=echoed-from-inbound.
        """
        self._conn.mav.timesync_send(int(tc1), int(ts1))

    def send_local_position_ned(
        self,
        pos_enu: NDArray[np.float64],
        vel_enu: NDArray[np.float64],
    ) -> None:
        """Send LOCAL_POSITION_NED. Converts ENU -> NED at the wire boundary."""
        # ENU (east, north, up) -> NED (north, east, down).
        x = float(pos_enu[1])
        y = float(pos_enu[0])
        z = -float(pos_enu[2])
        vx = float(vel_enu[1])
        vy = float(vel_enu[0])
        vz = -float(vel_enu[2])
        self._conn.mav.local_position_ned_send(
            self._t_boot_ms(), x, y, z, vx, vy, vz,
        )
