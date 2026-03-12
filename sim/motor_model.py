"""First-order motor model with nonlinear ESC curve.

Replicates MAVLab's motor dynamics:
  dW/dt = (Wc - W) / tau
  Wc = (w_max - w_min) * sqrt(k*U^2 + (1-k)*U) + w_min

All functions are vectorized for (n_envs,) or (n_envs, 4) arrays.
"""
import numpy as np


def esc_curve(u, w_min=238.49, w_max=3295.50, k=0.95):
    u = np.asarray(u, dtype=np.float64)
    return (w_max - w_min) * np.sqrt(k * u**2 + (1 - k) * u) + w_min


def motor_step(w, u, dt, tau=0.04, w_min=238.49, w_max=3295.50, k=0.95):
    w_cmd = esc_curve(u, w_min=w_min, w_max=w_max, k=k)
    return w + (w_cmd - w) * dt / tau
