"""Parametric aero model as linear-in-coefficients regressor features (pure numpy, body frame FRD).

Forces:  a_aero = -D*v - C*v|v|             (per-axis D, C; accel units, lumped /mass)
Moments: alpha_aero = -d*w + weathervane(v) (per-axis damping d; velocity-coupling 'wv_*' terms)

The weathervane block lets a body-velocity component drive a yaw/pitch moment (the destabilizing
tail-first term); kept linear so the whole fit is a least-squares solve.  Coefficients live in
two flat vectors theta_F (FORCE_COLS) and theta_M (MOMENT_COLS).
"""
from __future__ import annotations
import numpy as np

FORCE_COLS = ["D_x", "D_y", "D_z", "C_x", "C_y", "C_z"]
MOMENT_COLS = ["d_x", "d_y", "d_z", "wv_x", "wv_y", "wv_z"]


def force_features(v_body) -> np.ndarray:
    """Return (3, len(FORCE_COLS)) feature matrix so that force_features(v) @ theta_F = -D*v - C*v|v|.

    Row i gives the feature vector for axis i:
      - column i   (D_i): -v_i          (linear drag)
      - column 3+i (C_i): -v_i * |v_i| (quadratic drag)
    """
    v = np.asarray(v_body, float).ravel()
    assert v.shape == (3,), f"v_body must be (3,), got {v.shape}"
    phi = np.zeros((3, len(FORCE_COLS)))
    for i in range(3):
        phi[i, i] = -v[i]           # D_i term: -v_i
        phi[i, 3 + i] = -v[i] * abs(v[i])  # C_i term: -v_i * |v_i|
    return phi


def moment_features(v_body, omega) -> np.ndarray:
    """Return (3, len(MOMENT_COLS)) feature matrix so that moment_features(v,w) @ theta_M = -d*w + weathervane(v).

    Row i gives the feature vector for moment axis i:
      - column i (d_i): -w_i  (angular rate damping)

    Weathervane (velocity-driven moment) coupling — kept linear in v so the
    full moment fit stays a least-squares solve:
      - axis 0 (roll):  wv_x col <- v_y  (lateral-velocity roll coupling)
      - axis 1 (pitch): wv_y col <- v_x  (axial-velocity pitch; tail-first destabilizer)
      - axis 2 (yaw):   wv_z col <- v_x  (axial-velocity yaw; tail-first destabilizer)
    """
    v = np.asarray(v_body, float).ravel()
    assert v.shape == (3,), f"v_body must be (3,), got {v.shape}"
    w = np.asarray(omega, float).ravel()
    assert w.shape == (3,), f"omega must be (3,), got {w.shape}"
    phi = np.zeros((3, len(MOMENT_COLS)))
    for i in range(3):
        phi[i, i] = -w[i]   # d_i term: -omega_i
    # weathervane velocity couplings
    phi[0, 3] = v[1]  # roll  <- v_y   (wv_x column)
    phi[1, 4] = v[0]  # pitch <- v_x   (wv_y column)
    phi[2, 5] = v[0]  # yaw   <- v_x   (wv_z column)
    return phi


def predict(v_body, omega, theta_F, theta_M) -> tuple[np.ndarray, np.ndarray]:
    """Return (a_aero (3,), alpha_aero (3,)) given body-frame velocity, angular rate, and coefficient vectors."""
    a_aero = force_features(v_body) @ np.asarray(theta_F, float)
    alpha_aero = moment_features(v_body, omega) @ np.asarray(theta_M, float)
    return a_aero, alpha_aero
