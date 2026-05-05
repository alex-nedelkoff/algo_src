"""Sanity-check the 4 drone presets: hover throttle %, T:W ratio, omega_hover/max ratio.

Loads each preset, builds a VehicleParams, runs a short hover trajectory
through NumpyQuadDynamics, and prints physical sanity ratios.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from sim.dynamics.numpy_quad import NumpyQuadDynamics
from sim.dynamics.params import VehicleParams

G = 9.81
PRESETS_DIR = Path("configs/sysid/drones")


def _load(name: str) -> VehicleParams:
    cfg = yaml.safe_load((PRESETS_DIR / f"{name}.yaml").read_text())
    return VehicleParams(
        mass=float(cfg["mass"]),
        arm_length=float(cfg["arm_length"]),
        k_thrust=float(cfg["k_thrust"]),
        k_torque=float(cfg["k_torque"]),
        tau_motor=float(cfg["tau_motor"]),
        inertia=np.asarray(cfg["inertia"], dtype=np.float64),
        drag_coeff=np.asarray(cfg["drag_coeff"], dtype=np.float64),
        max_rpm=float(cfg["max_rpm"]),
    )


def _check_hover(p: VehicleParams) -> float:
    """Return absolute z-drift (m) over 1 s of hover at the analytical hover_omega."""
    dyn = NumpyQuadDynamics(p, dt=0.01)
    dyn.reset(1)
    omega_hover = math.sqrt(p.mass * G / (4.0 * p.k_thrust))
    s = np.zeros((1, 17), dtype=np.float64)
    s[0, 0:3] = [0.0, 0.0, 1.0]
    s[0, 6] = 1.0  # quat w = 1
    s[0, 13:17] = omega_hover
    a = np.full((1, 4), omega_hover, dtype=np.float64)
    z_history = []
    for _ in range(100):
        s = dyn.step(s, a, dt=0.01)
        z_history.append(float(s[0, 2]))
    return abs(z_history[-1] - 1.0)


def main() -> None:
    print(f"{'preset':<12} {'mass':>7} {'arm':>6} {'omega_hov':>10} {'omega_max':>10} {'hov%':>6} {'T:W':>5} {'z_drift_1s':>11}")
    print("-" * 80)
    for name in ("cf21", "fpv_racer", "f450", "heavy"):
        p = _load(name)
        omega_hover = math.sqrt(p.mass * G / (4.0 * p.k_thrust))
        omega_max = p.max_omega
        hov_pct = 100.0 * omega_hover / omega_max
        max_thrust_total = 4.0 * p.k_thrust * omega_max ** 2
        tw_ratio = max_thrust_total / (p.mass * G)
        z_drift = _check_hover(p)
        print(
            f"{name:<12} {p.mass:>7.3f} {p.arm_length:>6.3f} "
            f"{omega_hover:>10.1f} {omega_max:>10.1f} {hov_pct:>5.1f}% "
            f"{tw_ratio:>5.2f} {z_drift:>11.2e}"
        )


if __name__ == "__main__":
    main()
