"""VFR_HUD derived fields: airspeed = |vel|, climb = vel_enu.z."""
from __future__ import annotations

import math
import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _client(port):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=255, source_component=0, dialect="ardupilotmega",
    )


def test_vfr_hud_derives_airspeed_and_climb():
    port = _free_port()
    params = VehicleParams()
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    state = DroneState(
        pos_enu=np.array([0.0, 0.0, 5.0]),
        vel_enu=np.array([3.0, 4.0, 1.5]),       # |v|=sqrt(9+16+2.25)=5.22, climb=1.5
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover),
        timestamp_us=0,
    )
    shim = MavlinkShim(
        backend=NumpyQuadBackend(params=params), params=params,
        host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0, "vfr_hud": 10.0},
    )
    shim.reset(state)
    shim.start()
    try:
        client = _client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.2)

        deadline = time.time() + 2.0
        msg = None
        while time.time() < deadline:
            m = client.recv_match(type="VFR_HUD", blocking=False)
            if m is not None:
                msg = m
                break
            time.sleep(0.005)
        assert msg is not None
        # The shim thread steps the backend before each scheduled send, and
        # with no SET_ATTITUDE_TARGET received the thrust target is 0 → the
        # drone is in free-fall while we wait for the first VFR_HUD. So we
        # can't compare against the initial state vector directly; instead
        # assert the algebraic relationship between the reported fields,
        # which proves the derivation is correct independent of drift.
        # airspeed = |v| = sqrt(groundspeed^2 + climb^2).
        expected_air = math.sqrt(msg.groundspeed**2 + msg.climb**2)
        assert abs(msg.airspeed - expected_air) < 0.1
        # Groundspeed must remain consistent with the horizontal initial
        # velocity (no horizontal force in free-fall) — sqrt(9+16) = 5.0.
        assert abs(msg.groundspeed - 5.0) < 0.5
        # Climb starts at +1.5 m/s and decays under gravity. Allow a wide
        # window covering 0–~250 ms of free-fall (Δvz ≈ −2.5 m/s).
        assert -1.5 < msg.climb <= 1.5
    finally:
        shim.stop()


def test_vfr_hud_default_rate_in_defaults():
    defaults = MavlinkShim.__dataclass_fields__["rates_hz"].default_factory()
    assert defaults.get("vfr_hud") == 10.0
