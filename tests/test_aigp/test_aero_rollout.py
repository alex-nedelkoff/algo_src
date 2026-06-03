import numpy as np
import pandas as pd
from aigp.motor_model import DEFAULT_GEOM
from aigp.aero_rollout import refine_nuisance, fit_residual_cost

I_RATIO = np.array([3.7, 1.0, 1.0]); KAPPA = 0.02
MP = dict(DEFAULT_GEOM, k_f=34.0, k_q=0.68, form="quadratic")


def _make_run(r_true=(0.05, -0.03, 0.0), lag_true=0, seed=0):
    N = 800; t = np.linspace(0, 4, N)
    D = np.array([0.5, 0.3, 0.0])
    vb = np.stack([2 * np.sin(1.5 * t), 1.5 * np.sin(0.9 * t + 1), 0.3 * np.sin(0.6 * t)], axis=1)
    a_cg = -D[None, :] * vb  # clean drag (identity attitude -> v_world=v_body)
    omega = np.stack([0.4 * np.sin(1.2 * t), 0.3 * np.sin(0.8 * t), 0.5 * np.sin(t)], axis=1)
    omega_dot = np.stack([0.48 * np.cos(1.2 * t), 0.24 * np.cos(0.8 * t), 0.5 * np.cos(t)], axis=1)
    r = np.array(r_true)
    a_imu = a_cg + np.cross(omega, np.cross(omega, r)) + np.cross(omega_dot, r)  # forward lever-arm
    if lag_true:
        a_imu = np.roll(a_imu, lag_true, axis=0)  # force lags velocity by lag_true
    u = np.full((N, 4), 0.27)
    df = pd.DataFrame(dict(
        t=t, vx=vb[:, 0], vy=vb[:, 1], vz=vb[:, 2],
        qw=1.0, qx=0.0, qy=0.0, qz=0.0,
        wx=omega[:, 0], wy=omega[:, 1], wz=omega[:, 2],
        fx=a_imu[:, 0], fy=a_imu[:, 1], fz=a_imu[:, 2],
        thrust_accel=9.81, coast=False,
        u0=u[:, 0], u1=u[:, 1], u2=u[:, 2], u3=u[:, 3],
    ))
    return df


def test_cost_lower_at_true_r():
    run = _make_run(r_true=(0.05, -0.03, 0.0))
    c_true = fit_residual_cost([0.05, -0.03, 0.0, 0], [run], I_RATIO, KAPPA, MP)
    c_zero = fit_residual_cost([0, 0, 0, 0], [run], I_RATIO, KAPPA, MP)
    assert c_true < c_zero  # correct lever-arm explains the force better


def test_refine_recovers_leverarm():
    run = _make_run(r_true=(0.05, -0.03, 0.0), seed=1)
    out = refine_nuisance([run], I_RATIO, KAPPA, MP, restarts=4)
    assert np.allclose(out["r"][:2], [0.05, -0.03], atol=0.02)  # rz is weakly observed; check rx,ry
    assert out["r2_force"] > 0.99


def test_refine_recovers_latency():
    run = _make_run(r_true=(0, 0, 0), lag_true=3, seed=2)
    out = refine_nuisance([run], I_RATIO, KAPPA, MP, restarts=4)
    assert abs(out["lag"] - 3) <= 1
