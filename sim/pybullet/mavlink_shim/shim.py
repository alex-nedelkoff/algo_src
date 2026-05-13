"""MavlinkShim — top-level orchestrator wiring server + controller + backend."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# server.py sets MAVLINK20=1 before importing mavutil; importing it first
# guarantees the env-var is set when we import mavutil below.
from sim.pybullet.mavlink_shim.server import MavlinkServer
from pymavlink import mavutil  # noqa: E402 -- must follow server import

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.attitude_controller import AttitudeController
from sim.pybullet.mavlink_shim.backend import DroneBackend, DroneState
from sim.pybullet.mavlink_shim.position_controller import PositionController
from sim.pybullet.mavlink_shim.position_target import (
    PositionTarget, parse_set_position_target_local_ned,
)
from sim.pybullet.mavlink_shim.coords_mavlink import (
    enu_quat_to_ned_euler,
    enu_quat_to_ned_quat_xyzw,
    ned_quat_wxyz_to_enu_quat,
)
from sim.pybullet.mavlink_shim.rate_scheduler import RateScheduler


@dataclass
class MavlinkShim:
    backend: DroneBackend
    params: VehicleParams
    host: str = "0.0.0.0"
    port: int = 14550
    sysid: int = 1
    compid: int = 1
    lockstep: bool = False
    rates_hz: dict = field(default_factory=lambda: {
        "heartbeat": 2.0,       # spec §4.4 minimum
        "attitude": 100.0,
        "highres_imu": 200.0,
        "local_position_ned": 30.0,   # QGC position HUD
        "sys_status": 1.0,
        "timesync": 10.0,       # PX4 default cadence; spec doesn't pin
    })
    controller_k_att: float = 6.0
    controller_k_damp: float = 0.0    # rate damping; 0 = pure-P (backward compat)
    # Position-mode attitude controller gains. Separate from the attitude-mode
    # (MoE) path so that the trained checkpoint stays compatible: the MoE was
    # trained against k_damp=0 with the broken mixer's implicit damping, while
    # position mode needs explicit rate damping (k_damp=2.0) after the thrust
    # pre-distortion fix removes that implicit damping effect.
    position_k_att: float = 6.0
    position_k_damp: float = 2.0      # rate damping for position-mode only

    _server: MavlinkServer = field(init=False, default=None)
    _controller: AttitudeController = field(init=False, default=None)
    _position_att_controller: AttitudeController = field(init=False, default=None)
    _scheduler: RateScheduler = field(init=False, default=None)
    _thread: threading.Thread = field(init=False, default=None)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _last_target_q: NDArray[np.float64] = field(init=False, default=None)
    _last_target_thrust: float = field(init=False, default=0.0)
    _last_state: DroneState = field(init=False, default=None)
    _last_step_us: int = field(init=False, default=0)
    _t0_us: int = field(init=False, default=0)
    _mode: str = field(init=False, default="attitude")  # "attitude" | "position"
    _last_pos_target: PositionTarget = field(init=False, default=None)
    _position_controller: PositionController = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._controller = AttitudeController(
            params=self.params, k_att=self.controller_k_att, k_damp=self.controller_k_damp,
        )
        self._position_att_controller = AttitudeController(
            params=self.params, k_att=self.position_k_att, k_damp=self.position_k_damp,
        )
        self._position_controller = PositionController(params=self.params)
        tick_hz = max(self.rates_hz.values()) if self.rates_hz else 200.0
        self._scheduler = RateScheduler(rates_hz=dict(self.rates_hz), tick_hz=tick_hz)
        self._last_target_q = np.array([1.0, 0.0, 0.0, 0.0])
        self._inbound_handlers: dict[str, callable] = {
            "SET_ATTITUDE_TARGET": self._on_set_attitude_target,
            "SET_POSITION_TARGET_LOCAL_NED": self._on_set_position_target,
            "TIMESYNC": self._on_timesync,
        }
        self._command_handlers: dict[int, callable] = {
            mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES: self._cmd_request_autopilot_capabilities,
            mavutil.mavlink.MAV_CMD_REQUEST_PROTOCOL_VERSION: self._cmd_request_protocol_version,
            mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE: self._cmd_request_message,
        }
        self._inbound_handlers["COMMAND_LONG"] = self._on_command_long
        self._inbound_handlers["PARAM_REQUEST_LIST"] = self._on_param_request_list
        self._inbound_handlers["PARAM_REQUEST_READ"] = self._on_param_request_read
        self._inbound_handlers["MISSION_REQUEST_LIST"] = self._on_mission_request_list
        self._inbound_handlers["LOG_REQUEST_LIST"] = self._on_log_request_list

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = MavlinkServer(
            host=self.host, port=self.port, sysid=self.sysid, compid=self.compid,
        )
        self._t0_us = int(time.monotonic() * 1_000_000)
        self._stop_event.clear()
        if not self.lockstep:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._server is not None:
            self._server.close()
            self._server = None
        # Release any backend-held resources (e.g. PyBullet physics client).
        self.backend.close()

    def is_running(self) -> bool:
        return self._server is not None

    def __enter__(self) -> "MavlinkShim":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def reset(self, initial_state: DroneState) -> None:
        self.backend.reset(initial_state)
        self._last_state = initial_state
        self._last_target_q = np.array([1.0, 0.0, 0.0, 0.0])
        self._last_target_thrust = 0.0
        self._last_step_us = 0
        tick_hz = max(self.rates_hz.values()) if self.rates_hz else 200.0
        self._scheduler = RateScheduler(rates_hz=dict(self.rates_hz), tick_hz=tick_hz)
        if self._server is not None:
            self._server.send_heartbeat()

    def wait_for_shutdown(self, timeout_s: float | None = None) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=timeout_s)

    def step(self, timeout_s: float = 0.1) -> bool:
        deadline = time.monotonic() + timeout_s
        got_command = False
        while time.monotonic() < deadline:
            msgs = self._server.recv_pending()
            for m in msgs:
                t = m.get_type()
                handler = self._inbound_handlers.get(t, self._on_unknown)
                handler(m)
                if t in ("SET_ATTITUDE_TARGET", "SET_POSITION_TARGET_LOCAL_NED"):
                    got_command = True
            if got_command:
                break
            time.sleep(0.001)
        if not got_command:
            return False
        self._do_one_step()
        return True

    def _recv_and_dispatch(self) -> None:
        """Drain all pending inbound messages and route via the handler table."""
        for m in self._server.recv_pending():
            t = m.get_type()
            self._inbound_handlers.get(t, self._on_unknown)(m)

    def _run_loop(self) -> None:
        tick_period_s = 1.0 / max(self.rates_hz.values())
        # Poll recv at >= 50 Hz regardless of the scheduled-output tick rate.
        _RECV_PERIOD_S = min(tick_period_s, 0.02)
        next_tick = time.monotonic()
        next_recv = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= next_recv:
                self._recv_and_dispatch()
                next_recv = now + _RECV_PERIOD_S
            if now >= next_tick:
                self._do_one_step()
                next_tick += tick_period_s
                if next_tick < now:
                    next_tick = now + tick_period_s
            sleep_for = min(next_tick, next_recv) - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _do_one_step(self) -> None:
        now_us = int(time.monotonic() * 1_000_000) - self._t0_us
        dt = (now_us - self._last_step_us) / 1_000_000.0
        if dt <= 0:
            dt = 1.0 / max(self.rates_hz.values())
        self._last_step_us = now_us

        if self._last_state is None:
            self._last_state = self.backend.step(np.zeros(4), dt=0.0)

        if self._mode == "position" and self._last_pos_target is not None:
            q_target_enu, thrust_norm = self._position_controller.compute(
                target=self._last_pos_target, state=self._last_state,
            )
            # Use the position-mode attitude controller (with rate damping) to
            # prevent the over-torque from the TRPYMixer square-law from causing
            # attitude divergence. The MoE attitude path uses self._controller
            # (k_damp=0) to stay compatible with the trained checkpoint.
            motor_speeds = self._position_att_controller.compute(
                q_target_enu_wxyz=q_target_enu,
                thrust_normalized=thrust_norm,
                q_current_enu_wxyz=self._last_state.quat_wxyz,
                omega_current_body=self._last_state.angular_vel_body,
            )
        else:
            q_target_enu = ned_quat_wxyz_to_enu_quat(self._last_target_q)
            thrust_norm = self._last_target_thrust
            motor_speeds = self._controller.compute(
                q_target_enu_wxyz=q_target_enu,
                thrust_normalized=thrust_norm,
                q_current_enu_wxyz=self._last_state.quat_wxyz,
                omega_current_body=self._last_state.angular_vel_body,
            )
        if not np.all(np.isfinite(motor_speeds)):
            motor_speeds = np.zeros(4)
        self._last_state = self.backend.step(motor_speeds, dt=dt)

        for name in self._scheduler.due(now_us):
            self._send_message(name)

    def _send_message(self, name: str) -> None:
        s = self._last_state
        if name == "heartbeat":
            self._server.send_heartbeat()
        elif name == "attitude":
            roll, pitch, yaw = enu_quat_to_ned_euler(s.quat_wxyz)
            self._server.send_attitude(
                roll, pitch, yaw,
                float(s.angular_vel_body[0]),
                float(s.angular_vel_body[1]),
                float(s.angular_vel_body[2]),
            )
        elif name == "odometry":
            pos_ned = np.array([s.pos_enu[1], s.pos_enu[0], -s.pos_enu[2]])
            vel_ned = np.array([s.vel_enu[1], s.vel_enu[0], -s.vel_enu[2]])
            quat_ned_xyzw = enu_quat_to_ned_quat_xyzw(s.quat_wxyz)
            self._server.send_odometry(pos_ned, vel_ned, quat_ned_xyzw, s.angular_vel_body)
        elif name == "highres_imu":
            imu = self.backend.get_imu()
            self._server.send_highres_imu(imu.accel_body, imu.gyro_body)
        elif name == "local_position_ned":
            self._server.send_local_position_ned(s.pos_enu, s.vel_enu)
        elif name == "sys_status":
            self._server.send_sys_status()
        elif name == "timesync":
            # Server-initiated periodic: tc1=our_time_ns, ts1=0.
            self._server.send_timesync(tc1=time.monotonic_ns(), ts1=0)

    # ----- inbound handlers (registered into self._inbound_handlers) -----

    def _on_set_attitude_target(self, m) -> None:
        self._last_target_q = np.array(m.q, dtype=np.float64)
        self._last_target_thrust = float(m.thrust)
        self._mode = "attitude"

    def _on_set_position_target(self, m) -> None:
        self._last_pos_target = parse_set_position_target_local_ned(m)
        self._mode = "position"

    def _on_timesync(self, m) -> None:
        if int(getattr(m, "tc1", 1)) == 0:
            self._server.send_timesync(tc1=time.monotonic_ns(), ts1=int(m.ts1))

    def _on_command_long(self, m) -> None:
        ml = mavutil.mavlink
        cmd = int(m.command)
        handler = self._command_handlers.get(cmd)
        if handler is None:
            self._server.send_command_ack(cmd, ml.MAV_RESULT_UNSUPPORTED)
            return
        handler(m)

    def _cmd_request_autopilot_capabilities(self, m) -> None:
        self._server.send_autopilot_version()
        self._server.send_command_ack(
            int(m.command), mavutil.mavlink.MAV_RESULT_ACCEPTED,
        )

    def _cmd_request_protocol_version(self, m) -> None:
        self._server.send_protocol_version()
        self._server.send_command_ack(
            int(m.command), mavutil.mavlink.MAV_RESULT_ACCEPTED,
        )

    def _cmd_request_message(self, m) -> None:
        """Param1 is the requested message id (MAVLINK_MSG_ID_*)."""
        ml = mavutil.mavlink
        msg_id = int(m.param1)
        result = ml.MAV_RESULT_ACCEPTED
        if msg_id == ml.MAVLINK_MSG_ID_AUTOPILOT_VERSION:
            self._server.send_autopilot_version()
        elif hasattr(ml, "MAVLINK_MSG_ID_PROTOCOL_VERSION") and msg_id == ml.MAVLINK_MSG_ID_PROTOCOL_VERSION:
            self._server.send_protocol_version()
        else:
            # Unknown message id -> we still ACK (UNSUPPORTED) to satisfy the
            # COMMAND_LONG contract, but emit no message body.
            result = ml.MAV_RESULT_UNSUPPORTED
        self._server.send_command_ack(int(m.command), result)

    def _on_param_request_list(self, m) -> None:
        self._server.send_empty_param_value()

    def _on_param_request_read(self, m) -> None:
        # Silent. We have no parameters to return.
        return None

    def _on_mission_request_list(self, m) -> None:
        self._server.send_empty_mission_count()

    def _on_log_request_list(self, m) -> None:
        self._server.send_empty_log_entry()

    def _on_unknown(self, m) -> None:
        # Default no-op; subclasses or later phases override.
        return None
