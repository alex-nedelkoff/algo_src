"""Gradient-free optimizer for IMU lever-arm `r` and velocity/IMU latency `lag`.

Uses Nelder-Mead (scipy.optimize.minimize) to minimize the unexplained variance of
the closed-loop aero fit as a function of the nuisance parameters (r, lag).

Performance note: `v_body`, `omega`, `omega_dot`, `tau_motor`, `T`, and raw `f_imu`
are precomputed once per run (the expensive Python loops in build_targets_cl only run
once), then the per-evaluation cost function only does the cheap lever-arm correction
+ lag shift + lstsq fit.

Lag convention (unified — same everywhere)
------------------------------------------
Positive lag means "velocity is delayed by lag samples relative to force" (equivalently,
force arrives lag samples early, or v_body lags the IMU by lag samples).

Alignment for lag > 0:
  - keep v_body[lag:]   (drop the first lag velocity samples — they have no matching force)
  - keep a_cg[:N-lag]   (drop the last lag force samples)
  This pairs v_body[i+lag] with a_cg[i], i.e. uses the velocity that is lag steps later
  than the corresponding force sample.

This is the same convention used by `build_targets_cl` and `fit_residual_cost`, so the
`lag` returned by `refine_nuisance` can be passed directly to `build_targets_cl`.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from .aero_dataset import build_targets_cl, leverarm_correct, world_to_body_vel, spline_deriv
from .aero_fit import fit_parametric_cl
from .geometry import quat_to_R
from .motor_model import motor_outputs_to_wrench

_R_LIMIT = 0.5   # m  -- hard clip on lever-arm magnitude
_LAG_LIMIT = 20  # samples -- hard clip on latency


def _clip_nuisance(nuisance):
    """Clip nuisance params to valid range.  Returns clipped array (no mutation)."""
    x = np.array(nuisance, dtype=float)
    # Clip lever-arm
    r_mag = np.linalg.norm(x[:3])
    if r_mag > _R_LIMIT:
        x[:3] = x[:3] / r_mag * _R_LIMIT
    # Clip lag
    x[3] = np.clip(x[3], -_LAG_LIMIT, _LAG_LIMIT)
    return x


def _precompute_run_full(df, I_ratio, kappa, motor_params):
    """Precompute everything including motor wrench and tau_aero (no-r, no-lag version)."""
    N = len(df)
    t = df['t'].to_numpy()
    omega = df[['wx', 'wy', 'wz']].to_numpy()
    f_imu = df[['fx', 'fy', 'fz']].to_numpy()

    # Body velocity (Python loop, done once)
    v_body = np.zeros((N, 3))
    for i in range(N):
        q = df[['qw', 'qx', 'qy', 'qz']].iloc[i].to_numpy()
        R = quat_to_R(q)
        v_world = df[['vx', 'vy', 'vz']].iloc[i].to_numpy()
        v_body[i] = world_to_body_vel(v_world, R)

    # Angular acceleration via spline (done once)
    omega_dot = spline_deriv(t, omega)

    # Motor wrench (done once)
    tau_motor = np.zeros((N, 3))
    T_arr = np.zeros(N)
    u_cols = df[['u0', 'u1', 'u2', 'u3']].to_numpy()
    for i in range(N):
        T_i, tau_i = motor_outputs_to_wrench(u_cols[i], motor_params)
        tau_motor[i] = tau_i
        T_arr[i] = T_i

    # tau_aero (doesn't depend on r or lag)
    I_ratio = np.asarray(I_ratio, float).ravel()
    gyro_term = np.cross(omega, I_ratio * omega)
    tau_aero = kappa * (I_ratio * omega_dot + gyro_term) - tau_motor

    return {
        'v_body': v_body,
        'omega': omega,
        'omega_dot': omega_dot,
        'f_imu': f_imu,
        'tau_aero': tau_aero,
        'T_arr': T_arr,
        'N': N,
    }


def _cost_from_cache(r, lag, cache_list, motor_params, I_ratio, kappa, w_moment):
    """Fast cost function using precomputed run data.

    Uses the same lag convention as build_targets_cl: positive lag keeps v_body[lag:]
    paired with a_cg[:N-lag] (velocity is lag samples later than force).

    Parameters
    ----------
    lag : int
        Lag in samples (same sign as build_targets_cl's lag parameter).
    """
    total = 0.0
    for cache in cache_list:
        v_body = cache['v_body']
        omega = cache['omega']
        omega_dot = cache['omega_dot']
        f_imu = cache['f_imu']
        tau_aero = cache['tau_aero']
        T_arr = cache['T_arr']
        N = cache['N']

        # Lever-arm correction (cheap)
        a_cg = leverarm_correct(f_imu, omega, omega_dot, r)

        # Lag alignment — same slicing as build_targets_cl
        if lag != 0:
            if lag > 0:
                # Pair v_body[lag+i] with a_cg[i]: velocity is lag samples later
                vb = v_body[lag:]
                om = omega[:N - lag]
                af = a_cg[:N - lag]
                am = tau_aero[:N - lag]
                T = T_arr[:N - lag]
            else:
                # lag < 0: force is lag samples later (force advanced relative to velocity)
                nl = -lag
                vb = v_body[:N - nl]
                om = omega[nl:]
                af = a_cg[nl:]
                am = tau_aero[nl:]
                T = T_arr[nl:]
        else:
            vb = v_body
            om = omega
            af = a_cg
            am = tau_aero
            T = T_arr

        if len(vb) < 10:
            return 1e9

        try:
            res = fit_parametric_cl(vb, om, T, af, am)
        except Exception:
            return 1e9

        r2_f = res["r2_force"]
        r2_m = res["r2_moment"]
        var_f = float(np.var(af))
        var_m = float(np.var(am))
        total += (1.0 - r2_f) * var_f + w_moment * (1.0 - r2_m) * var_m

    return float(total)


def fit_residual_cost(nuisance, runs, I_ratio, kappa, motor_params, w_moment=1.0):
    """Cost for candidate nuisance = [rx, ry, rz, lag].

    For each run (DataFrame) compute the unexplained variance after the best
    linear fit_parametric_cl.  Lower is better.

    Parameters
    ----------
    nuisance : array-like (4,) -- [rx, ry, rz, lag]
    runs     : list of pd.DataFrame
    I_ratio, kappa, motor_params : passed straight through to build_targets_cl
    w_moment : weight on moment unexplained variance (relative to force)

    Returns
    -------
    float  total cost (scalar, >= 0)
    """
    x = _clip_nuisance(nuisance)
    r = x[:3]
    lag = int(round(x[3]))

    total = 0.0
    for df in runs:
        try:
            out = build_targets_cl(df, I_ratio, kappa, motor_params, r=r, lag=lag)
        except Exception:
            return 1e9

        a_force = out["a_aero"]
        tau_aero = out["tau_aero"]
        vb = out["v_body"]
        om = out["omega"]
        T = out["T"]

        if len(vb) < 10:
            return 1e9

        try:
            res = fit_parametric_cl(vb, om, T, a_force, tau_aero)
        except Exception:
            return 1e9

        r2_f = res["r2_force"]
        r2_m = res["r2_moment"]
        var_f = float(np.var(a_force))
        var_m = float(np.var(tau_aero))
        total += (1.0 - r2_f) * var_f + w_moment * (1.0 - r2_m) * var_m

    return float(total)


def refine_nuisance(runs, I_ratio, kappa, motor_params, x0=None, restarts=3, w_moment=1.0):
    """Nelder-Mead optimization of the lever-arm + latency nuisance parameters.

    Uses precomputed run data (v_body, omega_dot, tau_aero, T) so each function
    evaluation only does lever-arm correction + lag alignment + lstsq fit.

    The returned `lag` uses the same convention as `build_targets_cl`: positive lag
    means velocity is delayed by that many samples relative to force.  The returned
    value can be passed directly to `build_targets_cl(df, ..., lag=lag)`.

    Parameters
    ----------
    runs         : list of pd.DataFrame
    I_ratio, kappa, motor_params : aero-model parameters
    x0           : initial guess [rx, ry, rz, lag]; defaults to zeros
    restarts     : number of random restarts (including the initial run)
    w_moment     : weight on moment residual relative to force

    Returns
    -------
    dict with keys:
        r          (3,)   -- best lever-arm (m)
        lag        int    -- best lag (samples; same convention as build_targets_cl)
        cost       float  -- best cost
        theta_F          -- force coefficients from final fit
        theta_M          -- moment coefficients from final fit
        r2_force   float
        r2_moment  float
    """
    if x0 is None:
        x0 = np.zeros(4)
    x0 = np.array(x0, dtype=float)

    I_ratio_arr = np.asarray(I_ratio, float).ravel()

    # Precompute run-invariant quantities once
    cache_list = [_precompute_run_full(df, I_ratio_arr, kappa, motor_params) for df in runs]

    def fast_cost(nuisance):
        x = _clip_nuisance(nuisance)
        r = x[:3]
        lag = int(round(x[3]))
        return _cost_from_cache(r, lag, cache_list, motor_params, I_ratio_arr, kappa, w_moment)

    rng = np.random.default_rng(42)
    best_x = None
    best_cost = np.inf

    for i in range(restarts):
        if i == 0:
            start = x0.copy()
        else:
            # Jitter: small perturbation around the best found so far (or x0)
            base = best_x if best_x is not None else x0
            jitter = rng.normal(0.0, 0.04, size=4)
            jitter[3] = rng.uniform(-3.0, 3.0)   # lag jitter: sample more widely
            start = base + jitter
            start = _clip_nuisance(start)

        # Initial simplex scale: 0.05 m for r, 2 samples for lag
        initial_simplex = np.tile(start, (5, 1))
        initial_simplex[1, 0] += 0.05
        initial_simplex[2, 1] += 0.05
        initial_simplex[3, 2] += 0.05
        initial_simplex[4, 3] += 2.0

        try:
            res = minimize(
                fast_cost,
                start,
                method="Nelder-Mead",
                options={
                    "maxiter": 2000,
                    "xatol": 1e-4,
                    "fatol": 1e-6,
                    "adaptive": True,
                    "initial_simplex": initial_simplex,
                },
            )
            x_opt = _clip_nuisance(res.x)
            cost = float(fast_cost(x_opt))
        except Exception:
            continue

        if cost < best_cost:
            best_cost = cost
            best_x = x_opt.copy()

    if best_x is None:
        best_x = x0.copy()

    # Final fit at the best nuisance point
    r_best = best_x[:3]
    lag_best = int(round(best_x[3]))  # same convention as build_targets_cl

    # Aggregate all runs for the final coefficient estimate using precomputed cache
    all_vb, all_om, all_T, all_af, all_am = [], [], [], [], []
    for cache in cache_list:
        v_body = cache['v_body']
        omega = cache['omega']
        omega_dot = cache['omega_dot']
        f_imu = cache['f_imu']
        tau_aero = cache['tau_aero']
        T_arr = cache['T_arr']
        N = cache['N']

        a_cg = leverarm_correct(f_imu, omega, omega_dot, r_best)

        # Same slicing convention as build_targets_cl
        if lag_best != 0:
            if lag_best > 0:
                vb = v_body[lag_best:]
                om = omega[:N - lag_best]
                af = a_cg[:N - lag_best]
                am = tau_aero[:N - lag_best]
                T = T_arr[:N - lag_best]
            else:
                nl = -lag_best
                vb = v_body[:N - nl]
                om = omega[nl:]
                af = a_cg[nl:]
                am = tau_aero[nl:]
                T = T_arr[nl:]
        else:
            vb, om, af, am, T = v_body, omega, a_cg, tau_aero, T_arr

        all_vb.append(vb)
        all_om.append(om)
        all_T.append(T)
        all_af.append(af)
        all_am.append(am)

    vb_all = np.concatenate(all_vb, axis=0)
    om_all = np.concatenate(all_om, axis=0)
    T_all = np.concatenate(all_T, axis=0)
    af_all = np.concatenate(all_af, axis=0)
    am_all = np.concatenate(all_am, axis=0)

    final = fit_parametric_cl(vb_all, om_all, T_all, af_all, am_all)

    return {
        "r":         r_best,
        "lag":       lag_best,
        "cost":      best_cost,
        "theta_F":   final["theta_F"],
        "theta_M":   final["theta_M"],
        "r2_force":  final["r2_force"],
        "r2_moment": final["r2_moment"],
    }
