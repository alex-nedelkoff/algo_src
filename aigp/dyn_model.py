"""Lumped-ratio rigid-body forward model for the AI-GP drone (pure numpy).

Parameters are identifiable lumped ratios (no absolute mass/inertia):
  c_T            specific thrust per (motor input ** power), summed over motors
  c_L, c_M, c_N  roll/pitch/yaw angular-accel per (mix . p)
  mix (3,4)      motor mixing rows for [roll, pitch, yaw] (signs/geometry)
  tau_motor      first-order motor lag (s)
  drag (3)       linear drag accel coeff (per world-velocity component)
  power          1 or 2 (thrust ~ input or input^2)
  g              gravity (m/s^2)
State dict: pos(3) vel(3) NED, quat(4) wxyz body->world, omega(3) body, motor(4).
"""
from __future__ import annotations

import numpy as np

from .control_math import G, quat_mul
from .geometry import quat_to_R


def default_params() -> dict:
    return {
        "c_T": 5.0,
        "c_L": 40.0, "c_M": 40.0, "c_N": 8.0,
        "mix": np.array([[-1.0, 1.0, 1.0, -1.0],     # roll
                         [-1.0, -1.0, 1.0, 1.0],     # pitch
                         [1.0, -1.0, 1.0, -1.0]]),   # yaw (X-config signs)
        "tau_motor": 0.02,
        "drag": np.zeros(3),
        "power": 2,
        "g": G,
    }


def quat_integrate(q_wxyz, omega_body, dt):
    """Integrate a body-rate over dt: q_dot = 0.5 * q (x) [0, omega]."""
    q = np.asarray(q_wxyz, float)
    wq = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    q = q + 0.5 * quat_mul(q, wq) * dt
    return q / np.linalg.norm(q)


def step(state: dict, u_cmd, params: dict, dt: float) -> dict:
    pos = np.asarray(state["pos"], float); vel = np.asarray(state["vel"], float)
    quat = np.asarray(state["quat"], float); omega = np.asarray(state["omega"], float)
    motor = np.asarray(state["motor"], float)
    u = np.asarray(u_cmd, float)

    # first-order motor lag
    alpha_lag = min(1.0, dt / params["tau_motor"])
    motor_new = motor + (u - motor) * alpha_lag
    p = motor_new ** params["power"]

    R = quat_to_R(quat)
    body_up = -R[:, 2]
    f = params["c_T"] * float(p.sum())                       # specific thrust (accel)
    a = f * body_up + np.array([0.0, 0.0, params["g"]]) - np.asarray(params["drag"]) * vel

    mix = np.asarray(params["mix"], float)
    alpha = np.array([params["c_L"] * float(mix[0] @ p),
                      params["c_M"] * float(mix[1] @ p),
                      params["c_N"] * float(mix[2] @ p)])

    return {
        "pos": pos + vel * dt,
        "vel": vel + a * dt,
        "quat": quat_integrate(quat, omega, dt),
        "omega": omega + alpha * dt,
        "motor": motor_new,
    }
