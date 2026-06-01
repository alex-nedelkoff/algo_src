"""Drone state container + thread-safe Store."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class DroneState:
    pos_ned: np.ndarray   # (3,)
    vel_ned: np.ndarray   # (3,)
    quat_wxyz: np.ndarray  # (4,)
    omega: np.ndarray     # (3,) body angular rate
    t_us: int


class Store:
    """Thread-safe latest-value store shared between IO threads and the loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._drone = None
        self._gate_idx = 0
        self._frame = None       # (np.ndarray BGR, t_ns)
        self._frame_seq = 0
        self._gates = None
        self._race = None        # latest parsed race-status dict
        self._imu = None         # (acc(3), gyro(3), t_us) from HIGHRES_IMU

    def set_drone(self, ds: DroneState) -> None:
        with self._lock:
            self._drone = ds

    def get_drone(self):
        with self._lock:
            return self._drone

    def set_gate_idx(self, idx: int) -> None:
        with self._lock:
            self._gate_idx = int(idx)

    def get_gate_idx(self) -> int:
        with self._lock:
            return self._gate_idx

    def set_frame(self, img, t_ns: int) -> None:
        with self._lock:
            self._frame = (img, t_ns)
            self._frame_seq += 1

    def get_frame(self):
        """Returns ((img, t_ns) | None, seq)."""
        with self._lock:
            return self._frame, self._frame_seq

    def set_gates(self, gates) -> None:
        with self._lock:
            self._gates = gates

    def get_gates(self):
        with self._lock:
            return self._gates

    def set_race(self, race: dict) -> None:
        with self._lock:
            self._race = race

    def get_race(self):
        with self._lock:
            return self._race

    def get_race_live(self) -> bool:
        with self._lock:
            return bool(self._race and self._race.get("race_live"))

    def set_imu(self, acc, gyro, t_us) -> None:
        with self._lock:
            self._imu = (acc, gyro, t_us)

    def get_imu(self):
        """Returns (acc(3), gyro(3), t_us) or None."""
        with self._lock:
            return self._imu
