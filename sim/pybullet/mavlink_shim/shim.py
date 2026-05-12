"""MavlinkShim — top-level orchestrator wiring server + controller + backend."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim.attitude_controller import AttitudeController
from sim.pybullet.mavlink_shim.backend import DroneBackend, DroneState
from sim.pybullet.mavlink_shim.coords_mavlink import (
    enu_quat_to_ned_euler,
    enu_quat_to_ned_quat_xyzw,
    ned_quat_wxyz_to_enu_quat,
)
from sim.pybullet.mavlink_shim.rate_scheduler import RateScheduler
from sim.pybullet.mavlink_shim.server import MavlinkServer


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
        "timesync": 10.0,       # PX4 default cadence; spec doesn't pin
    })
    controller_k_att: float = 6.0
    controller_k_damp: float = 0.0    # rate damping; 0 = pure-P (backward compat)

    _server: MavlinkServer = field(init=False, default=None)
    _controller: AttitudeController = field(init=False, default=None)
    _scheduler: RateScheduler = field(init=False, default=None)
    _thread: threading.Thread = field(init=False, default=None)
    _stop_event: threading.Event = field(init=False, default_factory=threading.Event)
    _last_target_q: NDArray[np.float64] = field(init=False, default=None)
    _last_target_thrust: float = field(init=False, default=0.0)
    _last_state: DroneState = field(init=False, default=None)
    _last_step_us: int = field(init=False, default=0)
    _t0_us: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._controller = AttitudeController(
            params=self.params, k_att=self.controller_k_att, k_damp=self.controller_k_damp,
        )
        tick_hz = max(self.rates_hz.values()) if self.rates_hz else 200.0
        self._scheduler = RateScheduler(rates_hz=dict(self.rates_hz), tick_hz=tick_hz)
        self._last_target_q = np.array([1.0, 0.0, 0.0, 0.0])

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
                if m.get_type() == "SET_ATTITUDE_TARGET":
                    self._last_target_q = np.array(m.q, dtype=np.float64)
                    self._last_target_thrust = float(m.thrust)
                    got_command = True
            if got_command:
                break
            time.sleep(0.001)
        if not got_command:
            return False
        self._do_one_step()
        return True

    def _run_loop(self) -> None:
        tick_period_s = 1.0 / max(self.rates_hz.values())
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            msgs = self._server.recv_pending()
            for m in msgs:
                if m.get_type() == "SET_ATTITUDE_TARGET":
                    self._last_target_q = np.array(m.q, dtype=np.float64)
                    self._last_target_thrust = float(m.thrust)
            self._do_one_step()
            next_tick += tick_period_s
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()

    def _do_one_step(self) -> None:
        now_us = int(time.monotonic() * 1_000_000) - self._t0_us
        dt = (now_us - self._last_step_us) / 1_000_000.0
        if dt <= 0:
            dt = 1.0 / max(self.rates_hz.values())
        self._last_step_us = now_us

        q_target_enu = ned_quat_wxyz_to_enu_quat(self._last_target_q)

        if self._last_state is None:
            motor_speeds = np.zeros(4)
            self._last_state = self.backend.step(motor_speeds, dt=0.0)

        motor_speeds = self._controller.compute(
            q_target_enu_wxyz=q_target_enu,
            thrust_normalized=self._last_target_thrust,
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
