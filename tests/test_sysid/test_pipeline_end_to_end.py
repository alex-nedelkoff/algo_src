"""End-to-end smoke test for the sysID pipeline.

Tiny dataset → quick fit → validate. Asserts:

1. The training loss strictly decreases by orders of magnitude over the run.
2. Open-loop rollout RMSE on held-out trajectories stays in physically
   reasonable bounds — not zero (sysID has identifiability degeneracies),
   but small in absolute terms.

Note: parameter recovery is NOT asserted exactly because individual scalars
are coupled in ratios (mass/k_thrust for linear dynamics, arm_length·k_thrust/
inertia for angular). What IS asserted is the predictive accuracy of the
fitted model — that's what matters for downstream control / training use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim.dynamics.params import VehicleParams
from sim.dynamics.torch_params import TorchVehicleParams
from sim.dynamics.torch_quad import step as torch_step
from scripts.sysid.dataset import generate_dataset


_NUM_TRAJ_TRAIN = 30
_NUM_TRAJ_VAL = 10
_NUM_STEPS = 100
_DT = 0.01
_ROLLOUT_K = 10
_N_EPOCHS = 200
_LR = 5e-3


def _init_perturbed(
    p_true: TorchVehicleParams,
    factor: float,
    rng,
    fixed: dict[str, float] | None = None,
) -> TorchVehicleParams:
    fixed = fixed or {}
    init = {}
    for n, v in p_true.named_scalars():
        init[n] = float(fixed[n]) if n in fixed else float(v.detach()) * rng.uniform(1.0 / factor, factor)
    p = TorchVehicleParams(
        mass=init["mass"], Ixx=init["Ixx"], Iyy=init["Iyy"], Izz=init["Izz"],
        k_thrust=init["k_thrust"], k_torque=init["k_torque"],
        arm_length=init["arm_length"], tau_motor=init["tau_motor"],
        max_omega=float(p_true.max_omega),
    )
    for n in fixed:
        getattr(p, f"_log_{n}").requires_grad_(False)
    return p


def _state_stds(states: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.maximum(states.reshape(-1, 17).std(axis=0), 1e-6))


def _rollout(state0, actions, params, dt):
    state = state0
    out = []
    for t in range(actions.shape[0]):
        state = torch_step(state, actions[t], params, dt=dt)
        out.append(state)
    return torch.stack(out)


def test_sysid_pipeline_recovers_predictive_dynamics():
    rng_np = np.random.default_rng(0)
    torch.manual_seed(0)

    truth = VehicleParams()
    p_true = TorchVehicleParams.from_vehicle_params(truth)

    train = generate_dataset(_NUM_TRAJ_TRAIN, _NUM_STEPS, _DT, truth, seed=0)
    val   = generate_dataset(_NUM_TRAJ_VAL,   _NUM_STEPS, _DT, truth, seed=42)

    p_fit = _init_perturbed(p_true, factor=1.5, rng=rng_np)
    optimizer = torch.optim.Adam(p_fit.parameters(), lr=_LR)
    state_stds = _state_stds(train.states)

    n_windows_per_traj = _NUM_STEPS - _ROLLOUT_K
    windows = np.array(
        [(i, t) for i in range(_NUM_TRAJ_TRAIN) for t in range(n_windows_per_traj)],
        dtype=np.int64,
    )

    losses = []
    for epoch in range(_N_EPOCHS):
        idxs = rng_np.choice(len(windows), size=8, replace=False)
        bs = np.empty((8, _ROLLOUT_K + 1, 17), dtype=np.float64)
        ba = np.empty((8, _ROLLOUT_K, 4), dtype=np.float64)
        for b, (i, t) in enumerate(windows[idxs]):
            bs[b] = train.states[i, t : t + _ROLLOUT_K + 1]
            ba[b] = train.actions[i, t : t + _ROLLOUT_K]

        per_traj_losses = []
        for b in range(8):
            pred = _rollout(torch.from_numpy(bs[b, 0]), torch.from_numpy(ba[b]), p_fit, _DT)
            target = torch.from_numpy(bs[b, 1:])
            per_traj_losses.append((((pred - target) / state_stds) ** 2).mean())
        loss = torch.stack(per_traj_losses).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(p_fit.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    # Loss should drop by at least 2 orders of magnitude on this small problem.
    assert losses[-1] < losses[0] / 100.0, (
        f"loss did not converge: start={losses[0]:.3e}, end={losses[-1]:.3e}"
    )
    assert losses[-1] < 1e-2, f"final loss too high: {losses[-1]:.3e}"

    # Open-loop rollout RMSE on held-out trajectories. Tight thresholds —
    # these are big-picture sanity checks, not literature-tight numbers.
    sq_err_pos = 0.0
    sq_err_quat = 0.0
    n_elem_pos = 0
    n_elem_quat = 0
    with torch.no_grad():
        for i in range(_NUM_TRAJ_VAL):
            pred = _rollout(torch.from_numpy(val.states[i, 0]),
                            torch.from_numpy(val.actions[i, :_ROLLOUT_K]),
                            p_fit, _DT).numpy()
            target = val.states[i, 1 : _ROLLOUT_K + 1]
            sq_err_pos  += np.sum((pred[:, 0:3] - target[:, 0:3]) ** 2)
            sq_err_quat += np.sum((pred[:, 6:10] - target[:, 6:10]) ** 2)
            n_elem_pos  += _ROLLOUT_K * 3
            n_elem_quat += _ROLLOUT_K * 4
    rmse_pos = float(np.sqrt(sq_err_pos / n_elem_pos))
    rmse_quat = float(np.sqrt(sq_err_quat / n_elem_quat))
    assert rmse_pos < 0.05, f"position RMSE over {_ROLLOUT_K * _DT}s = {rmse_pos:.4e} m (>5cm)"
    assert rmse_quat < 0.05, f"quaternion RMSE over {_ROLLOUT_K * _DT}s = {rmse_quat:.4e}"


def test_fixed_params_stay_pinned_and_optimiser_excludes_them():
    """When mass + arm_length are fixed, those scalars don't move and aren't
    receiving gradients; the remaining 6 params still receive gradients and
    the loss converges."""
    rng_np = np.random.default_rng(0)
    torch.manual_seed(0)

    truth = VehicleParams()
    p_true = TorchVehicleParams.from_vehicle_params(truth)
    fixed = {"mass": 0.027, "arm_length": 0.0397}

    train = generate_dataset(_NUM_TRAJ_TRAIN, _NUM_STEPS, _DT, truth, seed=0)

    p_fit = _init_perturbed(p_true, factor=1.5, rng=rng_np, fixed=fixed)
    learnable = [prm for prm in p_fit.parameters() if prm.requires_grad]
    assert len(learnable) == 6, f"expected 6 learnable, got {len(learnable)}"

    initial = p_fit.as_dict()

    optimizer = torch.optim.Adam(learnable, lr=_LR)
    state_stds = _state_stds(train.states)

    n_windows_per_traj = _NUM_STEPS - _ROLLOUT_K
    windows = np.array(
        [(i, t) for i in range(_NUM_TRAJ_TRAIN) for t in range(n_windows_per_traj)],
        dtype=np.int64,
    )

    losses = []
    for _ in range(_N_EPOCHS):
        idxs = rng_np.choice(len(windows), size=8, replace=False)
        bs = np.empty((8, _ROLLOUT_K + 1, 17), dtype=np.float64)
        ba = np.empty((8, _ROLLOUT_K, 4), dtype=np.float64)
        for b, (i, t) in enumerate(windows[idxs]):
            bs[b] = train.states[i, t : t + _ROLLOUT_K + 1]
            ba[b] = train.actions[i, t : t + _ROLLOUT_K]
        per_traj = []
        for b in range(8):
            pred = _rollout(torch.from_numpy(bs[b, 0]), torch.from_numpy(ba[b]), p_fit, _DT)
            target = torch.from_numpy(bs[b, 1:])
            per_traj.append((((pred - target) / state_stds) ** 2).mean())
        loss = torch.stack(per_traj).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learnable, max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    final = p_fit.as_dict()

    # Fixed params must not have moved (allow float exp(log(x)) roundoff).
    assert final["mass"] == pytest.approx(0.027, rel=1e-12)
    assert final["arm_length"] == pytest.approx(0.0397, rel=1e-12)
    assert final["mass"] == initial["mass"]
    assert final["arm_length"] == initial["arm_length"]
    # Fixed log-params must not have received any gradient.
    assert p_fit._log_mass.grad is None
    assert p_fit._log_arm_length.grad is None
    # At least one learnable param must have moved (sanity).
    moved = sum(abs(final[n] - initial[n]) > 1e-9
                for n in ("Ixx", "Iyy", "Izz", "k_thrust", "k_torque", "tau_motor"))
    assert moved >= 4, f"expected most learnable params to move, only {moved}/6 did"
    # Loss should still drop meaningfully.
    assert losses[-1] < losses[0] / 10.0
