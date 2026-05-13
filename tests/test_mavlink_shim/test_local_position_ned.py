"""LOCAL_POSITION_NED outbound — byte-correct ENU->NED, sent at 30 Hz."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil  # noqa: E402

from sim.dynamics.params import VehicleParams
from sim.pybullet.mavlink_shim import DroneState, MavlinkShim
from sim.pybullet.mavlink_shim.numpy_quad_backend import NumpyQuadBackend
from sim.pybullet.mavlink_shim.server import MavlinkServer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connect_client(port: int):
    return mavutil.mavlink_connection(
        f"udpout:127.0.0.1:{port}",
        source_system=2, source_component=1, dialect="ardupilotmega",
    )


def _hover_state(params: VehicleParams) -> DroneState:
    hover = float(np.sqrt(params.mass * 9.81 / (4.0 * params.k_thrust)))
    return DroneState(
        pos_enu=np.array([2.0, 5.0, 3.0]),       # east=2, north=5, up=3
        vel_enu=np.array([0.5, 1.0, -0.25]),
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_vel_body=np.zeros(3),
        motor_speed=np.full(4, hover),
        timestamp_us=0,
    )


def test_send_local_position_ned_emits_packet():
    """Direct server-level test: bytes match the NED conversion of the input."""
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port)
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)
        server.recv_pending()  # learn client address

        # Input: ENU (east=2, north=5, up=3), vel ENU (0.5, 1.0, -0.25).
        # Expected NED: x_n=5, y_e=2, z_d=-3, vx=1.0, vy=0.5, vz=0.25.
        server.send_local_position_ned(
            pos_enu=np.array([2.0, 5.0, 3.0]),
            vel_enu=np.array([0.5, 1.0, -0.25]),
        )

        deadline = time.time() + 1.0
        msg = None
        while time.time() < deadline:
            msg = client.recv_match(type="LOCAL_POSITION_NED", blocking=False)
            if msg is not None:
                break
            time.sleep(0.01)
        assert msg is not None, "no LOCAL_POSITION_NED received"
        assert abs(msg.x - 5.0) < 1e-6
        assert abs(msg.y - 2.0) < 1e-6
        assert abs(msg.z - (-3.0)) < 1e-6
        assert abs(msg.vx - 1.0) < 1e-6
        assert abs(msg.vy - 0.5) < 1e-6
        assert abs(msg.vz - 0.25) < 1e-6
    finally:
        server.close()


def test_local_position_ned_flows_through_shim():
    """When LOCAL_POSITION_NED is in rates_hz, the shim emits it via the scheduler."""
    params = VehicleParams()
    backend = NumpyQuadBackend(params=params)
    port = _free_port()
    shim = MavlinkShim(
        backend=backend, params=params, host="127.0.0.1", port=port,
        rates_hz={"heartbeat": 2.0, "local_position_ned": 30.0},
    )
    shim.reset(_hover_state(params))
    shim.start()
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)

        count = 0
        deadline = time.time() + 1.0
        while time.time() < deadline:
            msg = client.recv_match(type="LOCAL_POSITION_NED", blocking=False)
            if msg is not None:
                count += 1
            time.sleep(0.002)
        # 30 Hz over 1.0 s, allow 20% slack.
        assert count >= 24, f"expected >=24 LOCAL_POSITION_NED in 1s, got {count}"
    finally:
        shim.stop()


def test_default_rates_include_local_position_ned():
    defaults = MavlinkShim.__dataclass_fields__["rates_hz"].default_factory()
    assert defaults.get("local_position_ned") == 30.0
