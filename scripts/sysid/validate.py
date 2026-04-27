"""Validate a fitted TorchVehicleParams against held-out trajectories.

Reports:
1. Per-parameter recovery error (% relative to ground truth).
2. Open-loop rollout RMSE on a held-out dataset, broken down by state group
   (position, velocity, attitude (quaternion), body rates, motor speeds).

Usage:
    python -m scripts.sysid.validate \
        --fit outputs/sysid/fitted.pt \
        --dataset outputs/sysid/val.npz \
        --rollout-steps 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.sysid.dataset import TrajectoryDataset
from sim.dynamics.params import VehicleParams
from sim.dynamics.torch_params import TorchVehicleParams
from sim.dynamics.torch_quad import step as torch_step

_STATE_GROUPS = {
    "pos":     slice(0, 3),
    "vel":     slice(3, 6),
    "quat":    slice(6, 10),
    "omega":   slice(10, 13),
    "motor_w": slice(13, 17),
}


def _build_params(d: dict[str, float], max_omega: float) -> TorchVehicleParams:
    return TorchVehicleParams(
        mass=d["mass"], Ixx=d["Ixx"], Iyy=d["Iyy"], Izz=d["Izz"],
        k_thrust=d["k_thrust"], k_torque=d["k_torque"],
        arm_length=d["arm_length"], tau_motor=d["tau_motor"],
        max_omega=max_omega,
    )


def _rollout_one(state0: np.ndarray, actions: np.ndarray, params: TorchVehicleParams, dt: float) -> np.ndarray:
    """K-step rollout starting from a single initial state. Returns (K, 17)."""
    state = torch.from_numpy(state0)
    actions_t = torch.from_numpy(actions)
    out = []
    with torch.no_grad():
        for t in range(actions.shape[0]):
            state = torch_step(state, actions_t[t], params, dt=dt)
            out.append(state.detach().numpy())
    return np.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, required=True, help="fitted .pt file from fit_params.py")
    ap.add_argument("--dataset", type=Path, required=True, help="held-out trajectory .npz")
    ap.add_argument("--rollout-steps", type=int, default=50,
                    help="rollout horizon (must be <= dataset traj length)")
    args = ap.parse_args()

    print(f"Loading fit: {args.fit}")
    blob = torch.load(args.fit, weights_only=False)
    fit_d = blob["params_fit"]
    true_d = blob["params_true"]

    print(f"Loading held-out dataset: {args.dataset}")
    ds = TrajectoryDataset.load(args.dataset)
    n_traj, n_steps_total, _ = ds.states.shape
    n_steps_total -= 1
    K = min(args.rollout_steps, n_steps_total)
    print(f"  {n_traj} trajectories x {n_steps_total} steps; rolling K={K}")

    # ------- Parameter recovery report -------
    print("\nParameter recovery (fit vs ground truth):")
    print(f"  {'param':>12}  {'fit':>14}  {'truth':>14}  {'err %':>8}")
    max_err = 0.0
    for name in fit_d:
        v_fit = fit_d[name]
        v_true = true_d[name]
        err_pct = 100.0 * (v_fit - v_true) / abs(v_true)
        max_err = max(max_err, abs(err_pct))
        print(f"  {name:>12}  {v_fit:>14.6g}  {v_true:>14.6g}  {err_pct:>+8.2f}")
    print(f"  worst-case |err|: {max_err:.2f} %")

    # ------- Open-loop rollout RMSE -------
    print(f"\nOpen-loop rollout RMSE over {K}-step horizons (held-out trajectories):")
    fit_p = _build_params(fit_d, max_omega=float(ds.params.max_omega))

    sq_err_fit = {name: 0.0 for name in _STATE_GROUPS}
    for i in range(n_traj):
        pred_fit = _rollout_one(ds.states[i, 0], ds.actions[i, :K], fit_p, ds.dt)
        target = ds.states[i, 1 : K + 1]
        for name, sl in _STATE_GROUPS.items():
            sq_err_fit[name]  += np.sum((pred_fit[:, sl]  - target[:, sl]) ** 2)

    units = {"pos": "m", "vel": "m/s", "quat": "(unit)", "omega": "rad/s", "motor_w": "rad/s"}
    print(f"  {'group':>8}  {'rmse':>12}  unit")
    rmse_out = {}
    for name, sl in _STATE_GROUPS.items():
        n_elements = K * (sl.stop - sl.start) * n_traj
        rmse_fit  = float(np.sqrt(sq_err_fit[name]  / n_elements))
        rmse_out[name] = rmse_fit
        print(f"  {name:>8}  {rmse_fit:>12.4e}  {units[name]}")
    return rmse_out, max_err


if __name__ == "__main__":
    rmse, _ = main()
    raise SystemExit(0)
