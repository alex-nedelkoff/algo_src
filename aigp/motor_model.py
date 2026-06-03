"""Motor + mixer model for the AI-GP sim quad (pure numpy).

Turns the 4 ACTUATOR_OUTPUT_STATUS motor outputs `u` in [0,1] into the body wrench used by the
closed-loop aero sysID: total thrust `T` (along body -z) and control torque `tau_motor` (roll,pitch,yaw).

Per rotor i:  f_i = k_f * g(u_i)   (thrust),   q_i = k_q * g(u_i)   (reaction / yaw torque),
with g(u) = u**2 (quadratic, default) or u (linear) — chosen by calibration. The mixer maps the
per-rotor thrust/torque to the body wrench via the airframe sign pattern (sx,sy,sz in {-1,+1} per
motor) and arm length L:
    T         = sum_i f_i
    tau_roll  = L * sum_i (sx_i * f_i)     # arm * thrust differential
    tau_pitch = L * sum_i (sy_i * f_i)
    tau_yaw   = sum_i (sz_i * q_i)         # reaction-torque differential (spin direction)

`sx,sy,sz,L` are airframe constants; `k_f,k_q` (and the g form) are calibrated against a hover +
small single-axis inputs. The T0 probe showed +roll raised motor 1 and lowered 0&2 — that signature
pins `sx`. Absolute force/torque scale couples with the (unknown) mass/inertia; the aero fit works in
the same normalized units, and the rollout refiner (T5) cleans up any residual scale.
"""
from __future__ import annotations
import numpy as np

# quad-X geometry pinned by the T2 live calibration (de-meaned single-axis motor differentials):
# roll left/right, pitch front/back, yaw = -(sx*sy) diagonal — verified by hover torque ~= 0.
DEFAULT_GEOM = dict(sx=[-1, +1, +1, -1], sy=[-1, -1, +1, +1], sz=[-1, +1, -1, +1], L=0.14)


def _g(u, form):
    u = np.asarray(u, float)
    return u * u if form == "quadratic" else u


def motor_outputs_to_wrench(u, params):
    """u: (4,) motor outputs in [0,1]. params: {k_f,k_q,L,form,sx,sy,sz}. Returns (T, tau(3,))."""
    g = _g(u, params.get("form", "quadratic"))
    f = float(params["k_f"]) * g
    q = float(params["k_q"]) * g
    sx = np.asarray(params["sx"], float)
    sy = np.asarray(params["sy"], float)
    sz = np.asarray(params["sz"], float)
    L = float(params["L"])
    T = float(np.sum(f))
    tau = np.array([L * float(sx @ f), L * float(sy @ f), float(sz @ q)])
    return T, tau


def calibrate(u_hover, weight_accel, geom=DEFAULT_GEOM, form="quadratic", kq_over_kf=0.02):
    """Pin k_f from a hover sample (sum_i f_i = m*g, expressed as the hover specific force `weight_accel`
    ~ 9.81 m/s^2 scaled by 1/mass -> we identify k_f in accel units so T is in m/s^2). Returns a params
    dict. k_q/k_f and the geometry are seeded here and refined in T4/T5. Units: T in m/s^2 (mass-normalized
    thrust accel), matching the aero force targets."""
    g = _g(u_hover, form)
    k_f = float(weight_accel) / float(np.sum(g))     # so that T(hover) = weight_accel (= g at hover)
    params = dict(geom, k_f=k_f, k_q=k_f * kq_over_kf, form=form)
    return params
