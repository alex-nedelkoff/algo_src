"""SYS_STATUS outbound — fixed healthy-sensor flags, 100% battery, 10% CPU."""
from __future__ import annotations

import os
import socket
import time

os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil  # noqa: E402

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


def test_sys_status_advertises_healthy_simulated_vehicle():
    port = _free_port()
    server = MavlinkServer(host="127.0.0.1", port=port)
    try:
        client = _connect_client(port)
        client.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0,
        )
        time.sleep(0.05)
        server.recv_pending()

        server.send_sys_status()

        deadline = time.time() + 1.0
        msg = None
        while time.time() < deadline:
            msg = client.recv_match(type="SYS_STATUS", blocking=False)
            if msg is not None:
                break
            time.sleep(0.01)
        assert msg is not None, "no SYS_STATUS received"

        # All-sensors-healthy: present == enabled == health (any nonzero bitmask).
        assert msg.onboard_control_sensors_present > 0
        assert msg.onboard_control_sensors_present == msg.onboard_control_sensors_enabled
        assert msg.onboard_control_sensors_present == msg.onboard_control_sensors_health
        # Battery 100% (-1 = unknown is invalid here; we must report 100).
        assert msg.battery_remaining == 100
        # Voltage 12 V == 12000 mV.
        assert msg.voltage_battery == 12000
        # CPU 10%: spec field uses 0.1% units -> raw value 100.
        assert msg.load == 100
    finally:
        server.close()
