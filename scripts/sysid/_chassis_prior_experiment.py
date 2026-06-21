"""Does knowing the chassis bbox (VADR-TS-002 §3.6: 280×280×160 mm) improve
grey-box sysID? Runs two fits from identical wide-init starts on the same
synthetic dataset:

  * baseline  — multi-step rollout MSE only (matches scripts/sysid/fit_params)
  * chassis   — MSE + soft priors derivable ONLY from chassis geometry:
                  - arm_length bounded inside the bbox
                  - Ixx == Iyy (square XY footprint forces this)
                  - Izz/Ixx ratio prior (physics-distribution dependent,
                    range 1.4–2.0 covers lumped-corner → uniform-block)

Ground truth is a spec-compatible 5" racing quad with arm_length=0.115 m
(motors inside the 280 mm chassis). Neither fit gets to see ground truth
beyond the wide-init starting point. Same dataset, same seed, same init —
only the loss differs.

Run (PYTHONPATH=. set, from the algo_src worktree root):
    python -m scripts.sysid._chassis_prior_experiment
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from scripts.sysid.dataset import generate_dataset
from sim.dynamics.params import VehicleParams
from sim.dynamics.torch_params import TorchVehicleParams
from sim.dynamics.torch_quad import step as torch_step


# --------------------------------------------------------------------- config

# Ground-truth 5" racing quad consistent with the spec chassis (280×280×160 mm).
# arm_length=0.115 m puts motor centers ~163 mm out from CG along the diagonal —
# inside the chassis bbox. Inertia chosen to match a typical lumped-corner-motor
# + central-battery mass distribution (Izz/Ixx ≈ 1.64).
_GROUND_TRUTH = VehicleParams(
    mass=0.65,
    inertia=np.diag([0.0028, 0.0028, 0.0046]),
    arm_length=0.115,
    k_thrust=2.0e-6,
    k_torque=7.0e-8,
    tau_motor=0.04,
    max_rpm=31470.0,
    drag_coeff=np.array([0.0, 0.0, 0.0]),
)

# What the spec gives us (and ONLY this).
_CHASSIS_W = 0.280  # m
_CHASSIS_L = 0.280  # m
_CHASSIS_H = 0.160  # m

# Wide multiplicative perturbation on the initial guess (×3 = factor-of-9 spread,
# log-space σ ≈ 1.1). At this width the baseline fitter struggles to recover
# inertia ratios from a 100-step dataset, which is what makes the comparison
# meaningful.
_INIT_PERTURB = 3.0

_DT = 0.01
_N_TRAIN_TRAJ = 30
_N_TEST_TRAJ = 10
_N_STEPS = 100
_N_EPOCHS = 300
_BATCH_SIZE = 16
_ROLLOUT_K = 10
_LR = 1e-2
_SEED = 0

# Chassis-derived prior weights (loss = MSE + Σ λ_i × penalty_i).
# Sized so each prior is roughly 1e-3 when off by ~1σ — same order as the
# multi-step MSE late in training. Strong enough to bias fits, weak enough that
# good data still overrides.
_LAMBDA_SYMMETRY = 1.0
_LAMBDA_RATIO = 0.5
_LAMBDA_ARM_BOUND = 1.0

# Chassis half-diagonal — hard geometric upper bound on motor-to-CG distance
# (motors can't be outside the bbox).
_ARM_MAX = math.sqrt((_CHASSIS_W / 2) ** 2 + (_CHASSIS_L / 2) ** 2)  # ≈ 0.198 m
# Center of the soft arm-length prior: motors typically mounted inboard of the
# chassis corners by ~30% to leave room for prop tip clearance and mounting tabs.
_ARM_PRIOR_MEAN = 0.13
_ARM_PRIOR_LOGSIGMA = 0.35  # log-space (multiplicative ±42%)

# Soft prior on Izz/Ixx. For a square X-config quad the ratio lies between
# ~1.5 (mass = uniform bbox block) and ~2.0 (mass = lumped corner motors).
# Real quads sit around 1.6–1.8.
_RATIO_PRIOR_MEAN = 1.65
_RATIO_PRIOR_LOGSIGMA = 0.20  # log-space (multiplicative ±22%)


# ---------------------------------------------------------------------- losses


def _rollout(state0, actions, params, dt):
    """K-step rollout for one trajectory. Returns (K, 17)."""
    state = state0
    out = []
    for t in range(actions.shape[0]):
        state = torch_step(state, actions[t], params, dt=dt)
        out.append(state)
    return torch.stack(out)


def _rollout_loss(p, states, actions, state_stds, dt):
    """Per-component-normalised MSE over a batch of K-step rollouts."""
    losses = []
    for i in range(states.shape[0]):
        pred = _rollout(states[i, 0], actions[i], p, dt)
        target = states[i, 1:]
        err = ((pred - target) / state_stds) ** 2
        losses.append(err.mean())
    return torch.stack(losses).mean()


def _chassis_prior_loss(p):
    """Soft chassis-bbox priors. Returns scalar tensor.

    - Symmetry: log(Ixx/Iyy)² (square footprint enforces this exactly).
    - Ratio: (log(Izz/Ixx) - log(1.65))² / σ² (physics-distribution range).
    - Arm bound: Gaussian on log(arm_length / 0.13 m); + hard ramp penalty
      once arm_length crosses the chassis-half-diagonal (~0.198 m).
    """
    sym = (torch.log(p.Ixx) - torch.log(p.Iyy)) ** 2

    log_ratio = torch.log(p.Izz) - torch.log(p.Ixx)
    ratio = ((log_ratio - math.log(_RATIO_PRIOR_MEAN)) / _RATIO_PRIOR_LOGSIGMA) ** 2

    arm = ((torch.log(p.arm_length) - math.log(_ARM_PRIOR_MEAN)) / _ARM_PRIOR_LOGSIGMA) ** 2
    # Hard upper-bound ramp: a soft barrier that kicks in only past the bbox.
    over = torch.clamp(p.arm_length - _ARM_MAX, min=0.0)
    arm = arm + 1e3 * over ** 2

    return _LAMBDA_SYMMETRY * sym + _LAMBDA_RATIO * ratio + _LAMBDA_ARM_BOUND * arm


# --------------------------------------------------------------- training loop


def _state_stds(states_np):
    flat = states_np.reshape(-1, 17)
    stds = flat.std(axis=0)
    stds = np.maximum(stds, 1e-6)
    return torch.from_numpy(stds.astype(np.float64))


def _make_init(p_true_torch, factor, seed):
    """Initialise each scalar at ground_truth × U(1/F, F). Returns a fresh
    TorchVehicleParams whose tensor values can then be deep-copied into two
    independent fitters."""
    rng = np.random.default_rng(seed)
    init = {}
    for name in ("mass", "Ixx", "Iyy", "Izz", "k_thrust", "k_torque", "arm_length", "tau_motor"):
        v = float(getattr(p_true_torch, name).detach())
        init[name] = v * rng.uniform(1.0 / factor, factor)
    return TorchVehicleParams(
        mass=init["mass"],
        Ixx=init["Ixx"], Iyy=init["Iyy"], Izz=init["Izz"],
        k_thrust=init["k_thrust"], k_torque=init["k_torque"],
        arm_length=init["arm_length"], tau_motor=init["tau_motor"],
        max_omega=float(p_true_torch.max_omega),
    )


def _clone_params(src: TorchVehicleParams) -> TorchVehicleParams:
    """Deep-copy a TorchVehicleParams instance so two fitters can train
    independently from byte-identical starts."""
    return copy.deepcopy(src)


def _fit(label, p_init, ds, ds_test, *, use_chassis_prior, n_epochs, lr, seed):
    p = _clone_params(p_init)
    state_stds = _state_stds(ds.states)

    rng = np.random.default_rng(seed + 1)  # diff stream from init
    torch.manual_seed(seed + 1)

    learnable = [prm for prm in p.parameters() if prm.requires_grad]
    opt = torch.optim.Adam(learnable, lr=lr)

    n_traj, n_steps_total, _ = ds.states.shape
    n_steps_total -= 1
    n_windows = n_steps_total - _ROLLOUT_K
    all_windows = np.array(
        [(i, t) for i in range(n_traj) for t in range(n_windows)], dtype=np.int64,
    )

    t0 = time.time()
    prior_val = float("nan")
    for epoch in range(n_epochs):
        idxs = rng.choice(len(all_windows), size=_BATCH_SIZE, replace=False)
        windows = all_windows[idxs]
        bs_np = np.empty((_BATCH_SIZE, _ROLLOUT_K + 1, 17), dtype=np.float64)
        ba_np = np.empty((_BATCH_SIZE, _ROLLOUT_K, 4), dtype=np.float64)
        for b, (i, t) in enumerate(windows):
            bs_np[b] = ds.states[i, t : t + _ROLLOUT_K + 1]
            ba_np[b] = ds.actions[i, t : t + _ROLLOUT_K]
        bs = torch.from_numpy(bs_np)
        ba = torch.from_numpy(ba_np)

        opt.zero_grad()
        data_loss = _rollout_loss(p, bs, ba, state_stds, ds.dt)
        if use_chassis_prior:
            prior = _chassis_prior_loss(p)
            loss = data_loss + prior
            prior_val = float(prior.detach())
        else:
            loss = data_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learnable, max_norm=1.0)
        opt.step()

        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            extra = f"  prior {prior_val:.3e}" if use_chassis_prior else ""
            print(f"  [{label:>8}] epoch {epoch+1:>4}/{n_epochs}  "
                  f"data {data_loss.item():.3e}{extra}")
    elapsed = time.time() - t0

    # Held-out rollout RMSE (long-horizon, full test traj per trajectory).
    test_stds = _state_stds(ds_test.states)
    rmses = []
    with torch.no_grad():
        for i in range(ds_test.states.shape[0]):
            s0 = torch.from_numpy(ds_test.states[i, 0])
            acts = torch.from_numpy(ds_test.actions[i])
            pred = _rollout(s0, acts, p, ds_test.dt)
            tgt = torch.from_numpy(ds_test.states[i, 1:])
            err = ((pred - tgt) / test_stds) ** 2
            rmses.append(float(err.mean().sqrt()))
    return p, np.mean(rmses), elapsed


# ----------------------------------------------------------------- reporting


def _param_table(p_base, p_chassis, p_true):
    names = ("mass", "Ixx", "Iyy", "Izz", "k_thrust", "k_torque", "arm_length", "tau_motor")
    print(f"\n{'param':>12}  {'truth':>12}  {'baseline':>12}  {'  err%':>8}    "
          f"{'chassis':>12}  {'  err%':>8}")
    base_errs = []
    chas_errs = []
    for n in names:
        vt = float(getattr(p_true, n).detach())
        vb = float(getattr(p_base, n).detach())
        vc = float(getattr(p_chassis, n).detach())
        eb = 100.0 * (vb - vt) / abs(vt)
        ec = 100.0 * (vc - vt) / abs(vt)
        base_errs.append(abs(eb))
        chas_errs.append(abs(ec))
        print(f"{n:>12}  {vt:>12.5g}  {vb:>12.5g}  {eb:>+7.1f}%    "
              f"{vc:>12.5g}  {ec:>+7.1f}%")
    print(f"\n{'mean |err%|':>12}  {' ':>12}  {' ':>12}  {np.mean(base_errs):>7.1f}%    "
          f"{' ':>12}  {np.mean(chas_errs):>7.1f}%")
    # Derived constraints
    ixx_b = float(p_base.Ixx.detach()); iyy_b = float(p_base.Iyy.detach())
    ixx_c = float(p_chassis.Ixx.detach()); iyy_c = float(p_chassis.Iyy.detach())
    izz_b = float(p_base.Izz.detach()); izz_c = float(p_chassis.Izz.detach())
    arm_b = float(p_base.arm_length.detach()); arm_c = float(p_chassis.arm_length.detach())
    print()
    print(f"  Ixx/Iyy ratio      baseline {ixx_b/iyy_b:.3f}   chassis {ixx_c/iyy_c:.3f}   "
          f"(truth 1.000)")
    print(f"  Izz/Ixx ratio      baseline {izz_b/ixx_b:.3f}   chassis {izz_c/ixx_c:.3f}   "
          f"(truth {0.0046/0.0028:.3f})")
    print(f"  arm_length (m)     baseline {arm_b:.4f}    chassis {arm_c:.4f}    "
          f"(chassis bound ≤ {_ARM_MAX:.4f})")


# ----------------------------------------------------------------------- main


def main() -> int:
    print("=" * 72)
    print("Chassis-bbox prior sysID experiment")
    print("=" * 72)
    print(f"Spec chassis (VADR-TS-002 §3.6): {_CHASSIS_W*1000:.0f} × "
          f"{_CHASSIS_L*1000:.0f} × {_CHASSIS_H*1000:.0f} mm")
    print(f"Ground truth: 5\" racing quad, arm={_GROUND_TRUTH.arm_length} m, "
          f"mass={_GROUND_TRUTH.mass} kg, Izz/Ixx="
          f"{np.diag(_GROUND_TRUTH.inertia)[2]/np.diag(_GROUND_TRUTH.inertia)[0]:.3f}")
    print(f"Init perturbation: ×{_INIT_PERTURB} (log-space σ ≈ {math.log(_INIT_PERTURB):.2f})")

    print(f"\nGenerating dataset ({_N_TRAIN_TRAJ} train + {_N_TEST_TRAJ} test "
          f"× {_N_STEPS} steps @ {_DT}s)…")
    ds = generate_dataset(_N_TRAIN_TRAJ, _N_STEPS, _DT, _GROUND_TRUTH, seed=_SEED)
    ds_test = generate_dataset(_N_TEST_TRAJ, _N_STEPS, _DT, _GROUND_TRUTH, seed=_SEED + 999)

    p_true = TorchVehicleParams.from_vehicle_params(_GROUND_TRUTH)
    p_init = _make_init(p_true, _INIT_PERTURB, seed=_SEED)

    print("\nShared initial parameter values (identical for both fits):")
    for name in ("mass", "Ixx", "Iyy", "Izz", "k_thrust", "k_torque", "arm_length", "tau_motor"):
        vi = float(getattr(p_init, name).detach())
        vt = float(getattr(p_true, name).detach())
        print(f"  {name:>12}  init {vi:>12.5g}   truth {vt:>12.5g}   "
              f"err {100*(vi-vt)/abs(vt):+7.1f}%")

    print("\nFitting BASELINE (rollout MSE only)…")
    p_base, rmse_base, t_base = _fit(
        "baseline", p_init, ds, ds_test,
        use_chassis_prior=False, n_epochs=_N_EPOCHS, lr=_LR, seed=_SEED,
    )

    print("\nFitting CHASSIS-PRIOR (MSE + chassis priors)…")
    p_chassis, rmse_chassis, t_chassis = _fit(
        "chassis", p_init, ds, ds_test,
        use_chassis_prior=True, n_epochs=_N_EPOCHS, lr=_LR, seed=_SEED,
    )

    _param_table(p_base, p_chassis, p_true)

    print(f"\nHeld-out long-horizon normalised RMSE ({_N_TEST_TRAJ} traj × {_N_STEPS} steps):")
    print(f"  baseline  {rmse_base:.4f}  (fit time {t_base:.1f}s)")
    print(f"  chassis   {rmse_chassis:.4f}  (fit time {t_chassis:.1f}s)")
    print(f"  delta     {(rmse_chassis - rmse_base):+.4f}  "
          f"({100*(rmse_chassis - rmse_base)/rmse_base:+.1f}% relative)")

    print("\nVerdict:")
    if rmse_chassis < rmse_base * 0.95:
        print("  ✓ Chassis priors gave materially better predictive accuracy (>5%).")
    elif rmse_chassis < rmse_base:
        print("  • Chassis priors modestly improved predictive accuracy.")
    else:
        print("  ✗ Chassis priors did not improve held-out RMSE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
