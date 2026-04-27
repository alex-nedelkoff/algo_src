"""Fit TorchVehicleParams to a trajectory dataset via gradient descent.

Initialises params with a random multiplicative perturbation of the
ground-truth values, then minimises a K-step rollout MSE loss between the
torch_quad prediction and the recorded NumpyQuadDynamics trajectory. The
ground truth is treated as privileged: only used to initialise (so we know
the scale) and for post-fit reporting.

Usage (PYTHONPATH=. set):
    python -m scripts.sysid.fit_params \
        --dataset outputs/sysid/train.npz \
        --out outputs/sysid/fitted.pt \
        --n-epochs 500 --batch-size 16 --rollout-steps 20 --lr 1e-3 \
        --init-perturbation 1.5 --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.sysid.dataset import TrajectoryDataset
from sim.dynamics.torch_params import TorchVehicleParams
from sim.dynamics.torch_quad import step as torch_step

# Components of the 17-dim state, used for per-component normalisation in the
# loss so positions don't dominate / motor speeds don't disappear.
_STATE_GROUPS = {
    "pos": slice(0, 3),
    "vel": slice(3, 6),
    "quat": slice(6, 10),
    "omega": slice(10, 13),
    "motor_w": slice(13, 17),
}


def _init_perturbed(p_true: TorchVehicleParams, factor: float, rng: np.random.Generator) -> TorchVehicleParams:
    """Initialise each learnable scalar at ground_truth * uniform(1/factor, factor)."""
    init = {name: float(val.detach()) * rng.uniform(1.0 / factor, factor)
            for name, val in p_true.named_scalars()}
    return TorchVehicleParams(
        mass=init["mass"],
        Ixx=init["Ixx"], Iyy=init["Iyy"], Izz=init["Izz"],
        k_thrust=init["k_thrust"], k_torque=init["k_torque"],
        arm_length=init["arm_length"], tau_motor=init["tau_motor"],
        max_omega=float(p_true.max_omega),
    )


def _state_stds(states: np.ndarray) -> torch.Tensor:
    """Per-component std of the dataset, shape (17,). Used to normalise the loss."""
    flat = states.reshape(-1, 17)
    stds = flat.std(axis=0)
    stds = np.maximum(stds, 1e-6)   # avoid div-by-zero on constant-ish components
    return torch.from_numpy(stds.astype(np.float64))


def _rollout(
    state0: torch.Tensor, actions: torch.Tensor, params: TorchVehicleParams, dt: float,
) -> torch.Tensor:
    """K-step rollout for a single trajectory. Returns shape (K, 17)."""
    state = state0
    out = []
    for t in range(actions.shape[0]):
        state = torch_step(state, actions[t], params, dt=dt)
        out.append(state)
    return torch.stack(out)


def _compute_loss(
    fit_params: TorchVehicleParams,
    states: torch.Tensor,    # (B, K+1, 17)
    actions: torch.Tensor,   # (B, K,   4)
    state_stds: torch.Tensor,  # (17,)
    dt: float,
) -> torch.Tensor:
    """Mean over batch + horizon of std-normalised squared per-component error."""
    losses = []
    for i in range(states.shape[0]):
        pred = _rollout(states[i, 0], actions[i], fit_params, dt)
        target = states[i, 1:]
        err = ((pred - target) / state_stds) ** 2
        losses.append(err.mean())
    return torch.stack(losses).mean()


def _print_param_table(p_fit: TorchVehicleParams, p_true: TorchVehicleParams) -> None:
    print(f"  {'param':>12}  {'fit':>14}  {'truth':>14}  {'err %':>8}")
    for (name_f, val_f), (_, val_t) in zip(p_fit.named_scalars(), p_true.named_scalars()):
        v_fit = float(val_f.detach())
        v_true = float(val_t.detach())
        err_pct = 100.0 * (v_fit - v_true) / abs(v_true) if v_true != 0 else float("inf")
        print(f"  {name_f:>12}  {v_fit:>14.6g}  {v_true:>14.6g}  {err_pct:>+8.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rollout-steps", type=int, default=20,
                    help="K-step horizon for the multi-step rollout loss")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-perturbation", type=float, default=1.5,
                    help="Initial params drawn at ground_truth * U(1/F, F)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading dataset: {args.dataset}")
    ds = TrajectoryDataset.load(args.dataset)
    n_traj, n_steps_total, _ = ds.states.shape
    n_steps_total -= 1  # actions have 1 fewer step than states
    K = args.rollout_steps
    if K >= n_steps_total:
        raise ValueError(f"--rollout-steps {K} must be < dataset traj length {n_steps_total}")
    print(f"  {n_traj} trajectories x {n_steps_total} steps @ {ds.dt}s")
    print(f"  rollout horizon K={K}")

    state_stds = _state_stds(ds.states)
    print(f"  state stds: " + ", ".join(
        f"{n}={float(state_stds[s].mean()):.2g}" for n, s in _STATE_GROUPS.items()
    ))

    p_true = TorchVehicleParams.from_vehicle_params(ds.params)
    p_fit = _init_perturbed(p_true, args.init_perturbation, rng)
    print("\nInitial parameter values vs ground truth:")
    _print_param_table(p_fit, p_true)

    optimizer = torch.optim.Adam(p_fit.parameters(), lr=args.lr)

    # Pre-build sliding-window indices: for each (traj, start_t) pair we'll grab
    # states[start_t : start_t + K + 1] and actions[start_t : start_t + K].
    n_windows_per_traj = n_steps_total - K
    all_windows = np.array(
        [(i, t) for i in range(n_traj) for t in range(n_windows_per_traj)],
        dtype=np.int64,
    )

    print(f"\nTraining ({args.n_epochs} epochs, {len(all_windows)} windows, batch {args.batch_size}):")
    for epoch in range(args.n_epochs):
        idxs = rng.choice(len(all_windows), size=args.batch_size, replace=False)
        windows = all_windows[idxs]

        batch_states = np.empty((args.batch_size, K + 1, 17), dtype=np.float64)
        batch_actions = np.empty((args.batch_size, K, 4), dtype=np.float64)
        for b, (i, t) in enumerate(windows):
            batch_states[b] = ds.states[i, t : t + K + 1]
            batch_actions[b] = ds.actions[i, t : t + K]

        bs = torch.from_numpy(batch_states)
        ba = torch.from_numpy(batch_actions)

        optimizer.zero_grad()
        loss = _compute_loss(p_fit, bs, ba, state_stds, ds.dt)
        loss.backward()
        # Modest grad clip — log-space params handle scale issues, but a
        # divergent rollout can still spike gradients.
        torch.nn.utils.clip_grad_norm_(p_fit.parameters(), max_norm=1.0)
        optimizer.step()
        # Note: no clamp needed — log-space parameterisation guarantees positivity.

        if epoch == 0 or (epoch + 1) % args.log_every == 0:
            print(f"  epoch {epoch + 1:>4}  loss {loss.item():.4e}")

    print("\nFinal parameter values vs ground truth:")
    _print_param_table(p_fit, p_true)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "params_fit": p_fit.as_dict(),
            "params_true": p_true.as_dict(),
            "config": vars(args),
        },
        args.out,
    )
    print(f"\nSaved fit + ground truth → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
