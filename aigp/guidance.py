"""Guidance patterns: produce NED velocity + yaw setpoints from state + gate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Setpoint:
    vx: float  # NED north velocity (m/s)
    vy: float  # NED east velocity (m/s)
    vz: float  # NED down velocity (m/s)
    yaw: float  # NED yaw (rad), 0 = north, +CW toward east


def yaw_to_target(drone_pos_ned, target_pos_ned) -> float:
    dn = target_pos_ned[0] - drone_pos_ned[0]
    de = target_pos_ned[1] - drone_pos_ned[1]
    return float(np.arctan2(de, dn))


class OrbitPattern:
    """Circle the gate at a fixed radius/height, yaw locked on the gate."""

    def __init__(self, radius=4.0, speed=2.0, target_z=-2.0, kp_radius=1.0, kp_z=1.0,
                 max_speed=None):
        self.radius = radius
        self.speed = speed
        self.target_z = target_z
        self.kp_radius = kp_radius
        self.kp_z = kp_z
        self.max_speed = max_speed  # cap on |horizontal| and |vertical| velocity

    def update(self, drone_pos_ned, drone_vel_ned, gate_pos_ned) -> Setpoint:
        rel = np.asarray(drone_pos_ned)[:2] - np.asarray(gate_pos_ned)[:2]  # (N,E)
        dist = float(np.linalg.norm(rel))
        radial = rel / dist if dist > 1e-3 else np.array([1.0, 0.0])
        tangential = np.array([-radial[1], radial[0]])  # +90 deg = CCW
        radius_err = self.radius - dist  # >0 means too close -> push outward (+radial)
        v_h = tangential * self.speed + radial * self.kp_radius * radius_err
        vz = self.kp_z * (self.target_z - drone_pos_ned[2])
        if self.max_speed is not None:
            h_mag = float(np.linalg.norm(v_h))
            if h_mag > self.max_speed:
                v_h = v_h / h_mag * self.max_speed
            vz = float(np.clip(vz, -self.max_speed, self.max_speed))
        yaw = yaw_to_target(drone_pos_ned, gate_pos_ned)
        return Setpoint(float(v_h[0]), float(v_h[1]), float(vz), yaw)


class ApproachPattern:
    """Repeated runs at the gate through a list of NED offsets (relative to gate)."""

    def __init__(self, offsets, speed=2.0, switch_dist=1.5):
        self.offsets = [np.asarray(o, float) for o in offsets]
        self.speed = speed
        self.switch_dist = switch_dist
        self.idx = 0

    def _target(self, gate_pos_ned):
        return np.asarray(gate_pos_ned, float) + self.offsets[self.idx % len(self.offsets)]

    def update(self, drone_pos_ned, drone_vel_ned, gate_pos_ned) -> Setpoint:
        target = self._target(gate_pos_ned)
        rel = target - np.asarray(drone_pos_ned, float)
        if np.linalg.norm(rel) < self.switch_dist:
            self.idx += 1
            target = self._target(gate_pos_ned)
            rel = target - np.asarray(drone_pos_ned, float)
        dist = float(np.linalg.norm(rel))
        direction = rel / dist if dist > 1e-6 else np.zeros(3)
        v = direction * self.speed
        yaw = yaw_to_target(drone_pos_ned, gate_pos_ned)
        return Setpoint(float(v[0]), float(v[1]), float(v[2]), yaw)


class GoToWaypoint:
    """Fly to an ABSOLUTE NED waypoint over the ACRO interface — a set_position replacement.

    Velocity toward the target with speed = min(max_speed, kp_pos * horizontal_distance): cruises
    far out, slows on approach. Yaw faces the target while en route (camera-forward = the
    weathervane-stable direction), then holds `hold_yaw` once inside arrival_radius. Pair with the
    pos/vel -> ACRO controller (race_cruise / control_math) + a sequencing runner (goto.py).
    max_speed is intentionally modest: the platform is weathervane-unstable above a few m/s."""

    def __init__(self, kp_pos=0.8, max_speed=2.5, arrival_radius=0.5, kp_z=1.0):
        self.kp_pos = kp_pos
        self.max_speed = max_speed
        self.arrival_radius = arrival_radius
        self.kp_z = kp_z

    def update(self, drone_pos_ned, drone_vel_ned, target_pos_ned, hold_yaw=0.0) -> Setpoint:
        pos = np.asarray(drone_pos_ned, float)
        tgt = np.asarray(target_pos_ned, float)
        h_err = (tgt - pos)[:2]
        h_dist = float(np.linalg.norm(h_err))
        speed = min(self.max_speed, self.kp_pos * h_dist)
        v_h = h_err / h_dist * speed if h_dist > 1e-6 else np.zeros(2)
        vz = float(np.clip(self.kp_z * (tgt[2] - pos[2]), -self.max_speed, self.max_speed))
        yaw = yaw_to_target(pos, tgt) if h_dist > self.arrival_radius else float(hold_yaw)
        return Setpoint(float(v_h[0]), float(v_h[1]), vz, yaw)

    def arrived(self, drone_pos_ned, target_pos_ned, tol=None) -> bool:
        d = float(np.linalg.norm(np.asarray(target_pos_ned, float) - np.asarray(drone_pos_ned, float)))
        return d < (tol if tol is not None else self.arrival_radius)
