"""16-state quadrotor dynamics built symbolically with SymPy, compiled to NumPy.

State vector (16):
    [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, w1, w2, w3, w4]

Coordinate frame: NED (z positive down).
Euler angles in ZYX convention (phi=roll, theta=pitch, psi=yaw).
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Union

import numpy as np
import sympy as sp
from sympy.utilities.lambdify import lambdify

# ---------------------------------------------------------------------------
# Nominal parameters (MAVLab sysid)
# ---------------------------------------------------------------------------
NOMINAL_PARAMS: Dict[str, float] = {
    "k_w": 2.49e-06,
    "k_x": 4.85e-05,
    "k_y": 7.28e-05,
    "kp1": 6.5e-05, "kp2": 6.5e-05, "kp3": 6.5e-05, "kp4": 6.5e-05,
    "kq1": 5.5e-05, "kq2": 5.5e-05, "kq3": 5.5e-05, "kq4": 5.5e-05,
    "kr1": 1.07e-02, "kr2": 1.07e-02, "kr3": 1.07e-02, "kr4": 1.07e-02,
    "kr5": 1.97e-03, "kr6": 1.97e-03, "kr7": 1.97e-03, "kr8": 1.97e-03,
    "Jx": 1.0e-03, "Jy": 1.0e-03, "Jz": 2.0e-03,
}

# Ordered list of parameter names — used to build the lambdified signature.
_PARAM_NAMES = sorted(NOMINAL_PARAMS.keys())


def _build_symbolic() -> tuple:
    """Build the symbolic state derivative expressions.

    Returns (sym_args, exprs) where sym_args is the flat list of sympy Symbols
    that the lambdified function will accept, and exprs is the list of 16
    derivative expressions.
    """
    # --- State symbols ---
    x, y, z = sp.symbols("x y z")
    vx, vy, vz = sp.symbols("vx vy vz")
    phi, theta, psi = sp.symbols("phi theta psi")
    p, q, r = sp.symbols("p q r")
    W1, W2, W3, W4 = sp.symbols("W1 W2 W3 W4")

    # --- Action symbols (motor derivatives dW, pass-through) ---
    dW1, dW2, dW3, dW4 = sp.symbols("dW1 dW2 dW3 dW4")

    # --- Parameter symbols ---
    param_syms = {name: sp.Symbol(name) for name in _PARAM_NAMES}
    k_w = param_syms["k_w"]
    k_x = param_syms["k_x"]
    k_y = param_syms["k_y"]
    kp1, kp2, kp3, kp4 = (param_syms[f"kp{i}"] for i in range(1, 5))
    kq1, kq2, kq3, kq4 = (param_syms[f"kq{i}"] for i in range(1, 5))
    kr1, kr2, kr3, kr4 = (param_syms[f"kr{i}"] for i in range(1, 5))
    kr5, kr6, kr7, kr8 = (param_syms[f"kr{i}"] for i in range(5, 9))
    Jx, Jy, Jz = param_syms["Jx"], param_syms["Jy"], param_syms["Jz"]

    g = sp.Float(9.81)

    # --- Rotation matrix R_zyx (body -> world) ---
    cphi, sphi = sp.cos(phi), sp.sin(phi)
    cth, sth = sp.cos(theta), sp.sin(theta)
    cpsi, spsi = sp.cos(psi), sp.sin(psi)

    R = sp.Matrix([
        [cpsi * cth, cpsi * sth * sphi - spsi * cphi, cpsi * sth * cphi + spsi * sphi],
        [spsi * cth, spsi * sth * sphi + cpsi * cphi, spsi * sth * cphi - cpsi * sphi],
        [-sth,       cth * sphi,                       cth * cphi],
    ])

    # --- Body-frame velocities (world -> body via R^T) ---
    v_world = sp.Matrix([vx, vy, vz])
    v_body = R.T * v_world
    vx_body = v_body[0]
    vy_body = v_body[1]

    # --- Forces ---
    W_sum = W1 + W2 + W3 + W4
    W_sq_sum = W1**2 + W2**2 + W3**2 + W4**2
    T = -k_w * W_sq_sum                      # thrust (along body -z in NED)
    Dx = -k_x * vx_body * W_sum              # body-x drag
    Dy = -k_y * vy_body * W_sum              # body-y drag

    # Translational: v_dot = [0,0,g] + R @ [Dx, Dy, T]
    force_body = sp.Matrix([Dx, Dy, T])
    accel = sp.Matrix([0, 0, g]) + R * force_body

    # --- Moments ---
    Mx = -kp1 * W1**2 - kp2 * W2**2 + kp3 * W3**2 + kp4 * W4**2
    My = -kq1 * W1**2 + kq2 * W2**2 - kq3 * W3**2 + kq4 * W4**2
    Mz = (-kr1 * W1 + kr2 * W2 + kr3 * W3 - kr4 * W4
           - kr5 * dW1 + kr6 * dW2 + kr7 * dW3 - kr8 * dW4)

    p_dot = Mx / Jx
    q_dot = My / Jy
    r_dot = Mz / Jz

    # --- Euler angle kinematics ---
    d_phi = p + q * sphi * sp.tan(theta) + r * cphi * sp.tan(theta)
    d_theta = q * cphi - r * sphi
    d_psi = q * sphi / cth + r * cphi / cth

    # --- Assemble 16-dim derivative vector ---
    derivs = [
        vx,             # dx/dt
        vy,             # dy/dt
        vz,             # dz/dt
        accel[0],       # dvx/dt
        accel[1],       # dvy/dt
        accel[2],       # dvz/dt
        d_phi,          # dphi/dt
        d_theta,        # dtheta/dt
        d_psi,          # dpsi/dt
        p_dot,          # dp/dt
        q_dot,          # dq/dt
        r_dot,          # dr/dt
        dW1,            # dW1/dt (pass-through)
        dW2,            # dW2/dt (pass-through)
        dW3,            # dW3/dt (pass-through)
        dW4,            # dW4/dt (pass-through)
    ]

    # Argument order: 16 state, 4 action, N params (sorted)
    state_syms = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, W1, W2, W3, W4]
    action_syms = [dW1, dW2, dW3, dW4]
    ordered_param_syms = [param_syms[n] for n in _PARAM_NAMES]

    sym_args = state_syms + action_syms + ordered_param_syms
    return sym_args, derivs


@functools.lru_cache(maxsize=1)
def build_dynamics_fn() -> Callable:
    """Compile and return the dynamics function.

    Returns a callable ``f(state, action, params)`` where:
        - state:  (16,) or (16, N) ndarray
        - action: (4,)  or (4, N) ndarray  (motor speed derivatives dW)
        - params: dict[str, float | ndarray]
    Returns:
        deriv: (16,) or (16, N) ndarray — the state time-derivative.
    """
    sym_args, derivs = _build_symbolic()

    # Lambdify: one function that takes all scalar args and returns (16,) tuple.
    _raw = lambdify(sym_args, derivs, modules="numpy")

    def dynamics(
        state: np.ndarray,
        action: np.ndarray,
        params: Dict[str, Union[float, np.ndarray]],
    ) -> np.ndarray:
        # Unpack state (16 values) and action (4 values).
        args: list[Any] = []
        for i in range(16):
            args.append(state[i])
        for i in range(4):
            args.append(action[i])
        # Append params in sorted order.
        for name in _PARAM_NAMES:
            args.append(params[name])

        raw = _raw(*args)
        return np.array(raw)

    return dynamics
