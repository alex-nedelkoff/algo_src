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

    def send_sys_status(self) -> None:
        """Send SYS_STATUS advertising a healthy simulated vehicle.

        Sensor bitmask covers gyro, accel, mag, abs-pressure, GPS, attitude
        stabilisation, yaw position, motors. Battery is reported as fully
        charged at 12.0 V (12000 mV), -1 A (current unknown). CPU load 10%
        (the field's unit is 0.1%, so raw value 100).
        """
        ml = mavutil.mavlink
        sensors = (
            ml.MAV_SYS_STATUS_SENSOR_3D_GYRO
            | ml.MAV_SYS_STATUS_SENSOR_3D_ACCEL
            | ml.MAV_SYS_STATUS_SENSOR_3D_MAG
            | ml.MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE
            | ml.MAV_SYS_STATUS_SENSOR_GPS
            | ml.MAV_SYS_STATUS_SENSOR_ANGULAR_RATE_CONTROL
            | ml.MAV_SYS_STATUS_SENSOR_ATTITUDE_STABILIZATION
            | ml.MAV_SYS_STATUS_SENSOR_YAW_POSITION
            | ml.MAV_SYS_STATUS_SENSOR_MOTOR_OUTPUTS
        )
        self._conn.mav.sys_status_send(
            onboard_control_sensors_present=sensors,
            onboard_control_sensors_enabled=sensors,
            onboard_control_sensors_health=sensors,
            load=100,                  # 10.0% (units: 0.1%)
            voltage_battery=12000,     # mV
            current_battery=-1,        # cA, -1 = unknown
            battery_remaining=100,     # %
            drop_rate_comm=0,
            errors_comm=0,
            errors_count1=0,
            errors_count2=0,
            errors_count3=0,
            errors_count4=0,
        )

    def send_autopilot_version(self) -> None:
        """Send AUTOPILOT_VERSION advertising no extended capabilities.

        Fired only in response to MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES.
        All-zero versions/vendor/product is the QGC-tolerated 'generic
        autopilot' identity — keeps QGC from probing PX4/ArduPilot params.

        Adaptation: pymavlink 2.4.49's ``autopilot_version_send`` declares
        ``uid2`` as ``uint8[18]`` (not 8 as some older references suggest),
        so we pass 18 zeros.
        """
        zero_md5 = [0] * 8      # uint8[8] custom version hashes
        zero_uid2 = [0] * 18    # uint8[18] extended UID
        self._conn.mav.autopilot_version_send(
            capabilities=0,
            flight_sw_version=0,
            middleware_sw_version=0,
            os_sw_version=0,
            board_version=0,
            flight_custom_version=zero_md5,
            middleware_custom_version=zero_md5,
            os_custom_version=zero_md5,
            vendor_id=0,
            product_id=0,
            uid=0,
            uid2=zero_uid2,
        )

    def send_protocol_version(self) -> None:
        """Send PROTOCOL_VERSION (we only speak v2).

        Adaptation: pymavlink 2.4.49 does not include the PROTOCOL_VERSION
        message (msg id 300) in any bundled dialect — it has been removed
        from upstream common.xml in favour of the v2-capability heartbeat
        flag. We keep the method on the API surface so the Task 5
        COMMAND_LONG dispatcher can call it uniformly, but it is a no-op
        on this pymavlink build. A NotImplementedError would break the
        ACK path; silent no-op lets the dispatcher still ACK ACCEPTED.
        """
        send = getattr(self._conn.mav, "protocol_version_send", None)
        if send is None:
            # Not available in this pymavlink dialect; skip.
            return
        zero_hash = [0] * 8
        send(
            version=200,
            min_version=100,
            max_version=200,
            spec_version_hash=zero_hash,
            library_version_hash=zero_hash,
        )

    def send_command_ack(self, command: int, result: int) -> None:
        """Send COMMAND_ACK in response to a COMMAND_LONG."""
        self._conn.mav.command_ack_send(int(command), int(result))

    def send_empty_param_value(self) -> None:
        """Send a single PARAM_VALUE with param_count=0 (empty-list signal)."""
        self._conn.mav.param_value_send(
            param_id=b"_EMPTY",
            param_value=0.0,
            param_type=mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
            param_count=0,
            param_index=0,
        )

    def send_empty_mission_count(self) -> None:
        """Send MISSION_COUNT(0) — no mission stored."""
        self._conn.mav.mission_count_send(
            target_system=0, target_component=0, count=0,
        )

    def send_empty_log_entry(self) -> None:
        """Send LOG_ENTRY with num_logs=0 — no logs stored."""
        self._conn.mav.log_entry_send(
            id=0, num_logs=0, last_log_num=0, time_utc=0, size=0,
        )

    def send_gps_raw_int(
        self,
        lat_deg: float, lon_deg: float, alt_m_amsl: float,
    ) -> None:
        """Stubbed GPS_RAW_INT: 3D fix, 12 sats, lat/lon from caller."""
        ml = mavutil.mavlink
        self._conn.mav.gps_raw_int_send(
            time_usec=self._t_usec(),
            fix_type=ml.GPS_FIX_TYPE_3D_FIX,
            lat=int(lat_deg * 1e7),
            lon=int(lon_deg * 1e7),
            alt=int(alt_m_amsl * 1000),
            eph=100, epv=100, vel=0, cog=0,
            satellites_visible=12,
        )

    def send_global_position_int(
        self,
        lat_deg: float, lon_deg: float,
        alt_m_amsl: float, relative_alt_m: float,
        vel_ned_m_s,  # length-3 array-like
        heading_deg: float,
    ) -> None:
        """GLOBAL_POSITION_INT — lat/lon in 1e7 deg, alt in mm, vel in cm/s."""
        self._conn.mav.global_position_int_send(
            time_boot_ms=self._t_boot_ms(),
            lat=int(lat_deg * 1e7),
            lon=int(lon_deg * 1e7),
            alt=int(alt_m_amsl * 1000),
            relative_alt=int(relative_alt_m * 1000),
            vx=int(vel_ned_m_s[0] * 100),
            vy=int(vel_ned_m_s[1] * 100),
            vz=int(vel_ned_m_s[2] * 100),
            hdg=int(heading_deg * 100),
        )

    def send_home_position(
        self, lat_deg: float, lon_deg: float, alt_m_amsl: float,
    ) -> None:
        """HOME_POSITION at the supplied lat/lon; local NED set to origin."""
        zero_q = [1.0, 0.0, 0.0, 0.0]
        self._conn.mav.home_position_send(
            latitude=int(lat_deg * 1e7),
            longitude=int(lon_deg * 1e7),
            altitude=int(alt_m_amsl * 1000),
            x=0.0, y=0.0, z=0.0,
            q=zero_q,
            approach_x=0.0, approach_y=0.0, approach_z=0.0,
            time_usec=self._t_usec(),
        )
