"""VQ body-rate loop model (command omega_cmd -> measured omega), per axis.
2nd-order continuous LTI discretized by exact ZOH. Task 2 adds amplitude-dependent
damping (the nonlinear "ring"). Pure numpy/scipy, no sim deps."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import cont2discrete


@dataclass
class RateLoopParams:
    k: float          # steady-state gain (omega/omega_cmd)
    wn: float         # natural frequency (rad/s)
    zeta0: float      # base damping ratio
    zeta1: float = 0.0  # amplitude slope (Task 2); 0 = linear


def _ABCD(wn: float, zeta: float, k: float):
    # x' = A x + B u ; y = C x + D u  for  Y/U = k wn^2 / (s^2 + 2 zeta wn s + wn^2)
    A = np.array([[0.0, 1.0], [-wn * wn, -2.0 * zeta * wn]])
    B = np.array([[0.0], [wn * wn]])
    C = np.array([[k, 0.0]])
    D = np.array([[0.0]])
    return A, B, C, D


def propagate(cmd: np.ndarray, dt: float, p: RateLoopParams) -> np.ndarray:
    cmd = np.asarray(cmd, float)
    Ad, Bd, Cd, Dd, _ = cont2discrete(_ABCD(p.wn, p.zeta0, p.k), dt, method="zoh")
    x = np.zeros((2, 1))
    out = np.empty(len(cmd))
    for i, u in enumerate(cmd):
        out[i] = (Cd @ x + Dd * u).item()
        x = Ad @ x + Bd * u
    return out
