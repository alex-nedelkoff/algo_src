"""Live MAVLink + vision receivers feeding a Store. Integration code."""
from __future__ import annotations

import os
import socket
import threading
import time

os.environ.setdefault("MAVLINK20", "1")

import numpy as np
from pymavlink import mavutil

from .protocol import (
    ENCAP_RACE_STATUS, ENCAP_TRACK_INFO,
    JpegReassembler, parse_race_status, parse_track_payload,
)
from .state import DroneState, Store

VISION_PORT = 5600


class MavlinkIO:
    def __init__(self, store: Store, host: str = "0.0.0.0", port: int = 14550,
                 send_timesync: bool = False):
        self.store = store
        self.conn = mavutil.mavlink_connection(f"udpin:{host}:{port}")
        self._stop = threading.Event()
        self._send_timesync = send_timesync
        self._rx = threading.Thread(target=self._rx_loop, daemon=True)
        self._ts = threading.Thread(target=self._ts_loop, daemon=True)
        self._track_chunks: dict[int, dict[int, bytes]] = {}
        self._track_expected: dict[int, int] = {}

    def wait_heartbeat(self, timeout=10):
        return self.conn.wait_heartbeat(timeout=timeout)

    def start(self):
        # Receive-only by default: sending anything (incl. timesync) during the
        # pre-race countdown can DQ + close the sim. Telemetry flows without it.
        self._rx.start()
        if self._send_timesync:
            self._ts.start()

    def stop(self):
        self._stop.set()

    def _ts_loop(self):
        while not self._stop.is_set():
            self.conn.mav.timesync_send(int(time.time_ns()), 0)
            time.sleep(0.1)

    def _rx_loop(self):
        while not self._stop.is_set():
            m = self.conn.recv_match(blocking=False)
            if m is None:
                time.sleep(0.001)
                continue
            t = m.get_type()
            if t == "ODOMETRY":
                # NOTE: this sim fills ODOMETRY.q as xyzw (w-LAST), not the MAVLink
                # spec's wxyz. Reorder to wxyz so quat_to_R reads the true attitude.
                self.store.set_drone(DroneState(
                    pos_ned=np.array([m.x, m.y, m.z]),
                    vel_ned=np.array([m.vx, m.vy, m.vz]),
                    quat_wxyz=np.array([m.q[3], m.q[0], m.q[1], m.q[2]]),
                    omega=np.array([m.rollspeed, m.pitchspeed, m.yawspeed]),
                    t_us=m.time_usec,
                ))
            elif t == "DATA_TRANSMISSION_HANDSHAKE":
                self._track_chunks[m.width] = {}
                self._track_expected[m.width] = m.packets
            elif t == "ENCAPSULATED_DATA":
                self._on_encap(m)

    def _on_encap(self, m):
        raw = bytes(m.data)
        if not raw:
            return
        dtype = raw[0]
        if dtype == ENCAP_RACE_STATUS:
            rs = parse_race_status(raw)
            self.store.set_gate_idx(rs["active_gate_index"])
            self.store.set_race(rs)
        elif dtype == ENCAP_TRACK_INFO:
            import struct
            _, tid = struct.unpack_from("<BH", raw)
            if tid not in self._track_expected:
                return
            self._track_chunks[tid][m.seqnr] = raw[3:]
            if len(self._track_chunks[tid]) == self._track_expected[tid]:
                payload = b"".join(
                    self._track_chunks[tid][i] for i in range(self._track_expected[tid])
                )
                self.store.set_gates(parse_track_payload(payload))


class VisionIO:
    def __init__(self, store: Store, port: int = VISION_PORT):
        self.store = store
        self.port = port
        self._stop = threading.Event()
        self._reasm = JpegReassembler()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        import cv2
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sock.settimeout(0.5)
        sock.bind(("0.0.0.0", self.port))
        while not self._stop.is_set():
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                continue
            out = self._reasm.add_packet(packet)
            if out is None:
                continue
            jpeg, t_ns = out
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.store.set_frame(img, t_ns)
