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
    delay: int = 0    # transport delay in SAMPLES (comms+proc); output lags input (live: ~1 sample / 15ms)
    sign: float = 1.0  # per-axis sign of the logged cmd->omega relation (live: roll = -1; cmd is post-OSGN, gyro is raw ds.omega)


def _delay_sign(y: np.ndarray, p: "RateLoopParams") -> np.ndarray:
    """Apply the transport delay (shift output right by p.delay samples, zero-fill) and per-axis sign."""
    if p.delay > 0:
        y = np.concatenate([np.zeros(int(p.delay)), y[:len(y) - int(p.delay)]])
    return p.sign * y


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
    return _delay_sign(out, p)


def propagate_nl(cmd: np.ndarray, dt: float, p: RateLoopParams) -> np.ndarray:
    """Quasi-LPV: damping varies with the current rate magnitude (the nonlinear ring).

    Note: the amplitude-scheduling variable is the internal state x[0,0] (which equals the
    output rate only when gain k≈1; in general ≈ omega/k), NOT the output omega. Downstream
    consumers (e.g., Phase-1 CPC) must replay through this exact model rather than
    re-deriving the damping schedule on the true output rate."""
    cmd = np.asarray(cmd, float)
    x = np.zeros((2, 1))
    out = np.empty(len(cmd))
    zeta_cur = None
    Ad = Bd = Cd = Dd = None
    for i, u in enumerate(cmd):
        omega = float(x[0, 0])
        zeta = float(np.clip(p.zeta0 + p.zeta1 * abs(omega), 0.02, 2.0))
        if zeta_cur is None or abs(zeta - zeta_cur) > 0.005:
            Ad, Bd, Cd, Dd, _ = cont2discrete(_ABCD(p.wn, zeta, p.k), dt, method="zoh")
            zeta_cur = zeta
        out[i] = (Cd @ x + Dd * u).item()
        x = Ad @ x + Bd * u
    return _delay_sign(out, p)
