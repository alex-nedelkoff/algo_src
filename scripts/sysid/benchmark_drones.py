"""Run the multi-drone sysID benchmark.

For each preset in configs/sysid/drones/:
1. Generate a 200-trajectory training dataset and a 50-trajectory held-out
   validation dataset using NumpyQuadDynamics with the preset's ground-truth
   params.
2. Fit TorchVehicleParams via Adam on K-step rollout MSE — twice:
   a. Unconstrained: all 8 scalars learnable, init perturbation factor 1.5.
   b. Constrained: pin mass + arm_length to ground truth (--fix style).
3. Validate each fit: per-param recovery error + held-out 50-step rollout RMSE
   broken down by state component (pos / vel / quat / omega / motor).
4. Save full results to outputs/sysid/multi_drone/{preset}/{condition}/ and
   produce a top-level summary at outputs/sysid/multi_drone/summary.json.

The training loop is duplicated from fit_params.py rather than refactored
shared because (a) it's ~15 lines and (b) keeping fit_params.py as the
single-CLI version avoids breaking it.

Usage:
    python -m scripts.sysid.benchmark_drones [--quick] [--out DIR]
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Preload torch BEFORE numpy/anything else — see scripts/_run_dynamics_tests.py
# for context on the monorace env's libomp / fbgemm.dll conflict.
import torch  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
from time import perf_counter  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from scripts.sysid.dataset import TrajectoryDataset, generate_dataset
from sim.dynamics.params import VehicleParams
from sim.dynamics.torch_params import TorchVehicleParams
from sim.dynamics.torch_quad import step as torch_step

PRESETS_DIR = Path("configs/sysid/drones")
DEFAULT_PRESETS = ("cf21", "fpv_racer", "f450", "heavy")

_STATE_GROUPS = {
    "pos":     slice(0, 3),
    "vel":     slice(3, 6),
    "quat":    slice(6, 10),
    "omega":   slice(10, 13),
    "motor_w": slice(13, 17),
}


# --------------------------------------------------------------------- presets


def load_preset(name: str) -> VehicleParams:
    cfg = yaml.safe_load((PRESETS_DIR / f"{name}.yaml").read_text())
    return VehicleParams(
        mass=float(cfg["mass"]),
        arm_length=float(cfg["arm_length"]),
        k_thrust=float(cfg["k_thrust"]),
        k_torque=float(cfg["k_torque"]),
        tau_motor=float(cfg["tau_motor"]),
        inertia=np.asarray(cfg["inertia"], dtype=np.float64),
        drag_coeff=np.asarray(cfg["drag_coeff"], dtype=np.float64),
        max_rpm=float(cfg["max_rpm"]),
    )


# --------------------------------------------------------------------- training


def _init_perturbed(
    p_true: TorchVehicleParams,
    factor: float,
    rng: np.random.Generator,
    fixed: dict[str, float] | None = None,
) -> TorchVehicleParams:
    fixed = fixed or {}
    init = {}
    for name, val in p_true.named_scalars():
        if name in fixed:
            init[name] = float(fixed[name])
        else:
            init[name] = float(val.detach()) * rng.uniform(1.0 / factor, factor)
    p = TorchVehicleParams(
        mass=init["mass"],
        Ixx=init["Ixx"], Iyy=init["Iyy"], Izz=init["Izz"],
        k_thrust=init["k_thrust"], k_torque=init["k_torque"],
        arm_length=init["arm_length"], tau_motor=init["tau_motor"],
        max_omega=float(p_true.max_omega),
    )
    for name in fixed:
        getattr(p, f"_log_{name}").requires_grad_(False)
    return p


def _state_stds(states: np.ndarray) -> torch.Tensor:
    flat = states.reshape(-1, 17)
    stds = np.maximum(flat.std(axis=0), 1e-6)
    return torch.from_numpy(stds.astype(np.float64))


def _rollout(state0: torch.Tensor, actions: torch.Tensor, params: TorchVehicleParams, dt: float) -> torch.Tensor:
    state = state0
    out = []
    for t in range(actions.shape[0]):
        state = torch_step(state, actions[t], params, dt=dt)
        out.append(state)
    return torch.stack(out)


def _compute_loss(
    fit_params: TorchVehicleParams,
    states: torch.Tensor,
    actions: torch.Tensor,
    state_stds: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    losses = []
    for i in range(states.shape[0]):
        pred = _rollout(states[i, 0], actions[i], fit_params, dt)
        target = states[i, 1:]
        err = ((pred - target) / state_stds) ** 2
        losses.append(err.mean())
    return torch.stack(losses).mean()


def train(
    p_true: TorchVehicleParams,
    train_ds: TrajectoryDataset,
    *,
    n_epochs: int,
    batch_size: int,
    K: int,
    lr: float,
    init_perturbation: float,
    fixed: dict[str, float] | None,
    seed: int,
) -> tuple[TorchVehicleParams, list[float]]:
    """Train. Returns (fitted params, loss curve)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    p_fit = _init_perturbed(p_true, init_perturbation, rng, fixed=fixed)
    state_stds = _state_stds(train_ds.states)
    learnable = [prm for prm in p_fit.parameters() if prm.requires_grad]
    optimizer = torch.optim.Adam(learnable, lr=lr)

    n_traj, n_steps_total = train_ds.states.shape[0], train_ds.states.shape[1] - 1
    n_windows_per_traj = n_steps_total - K
    all_windows = np.array(
        [(i, t) for i in range(n_traj) for t in range(n_windows_per_traj)],
        dtype=np.int64,
    )

    losses = []
    for epoch in range(n_epochs):
        idxs = rng.choice(len(all_windows), size=batch_size, replace=False)
        windows = all_windows[idxs]

        bs = np.empty((batch_size, K + 1, 17), dtype=np.float64)
        ba = np.empty((batch_size, K, 4), dtype=np.float64)
        for b, (i, t) in enumerate(windows):
            bs[b] = train_ds.states[i, t : t + K + 1]
            ba[b] = train_ds.actions[i, t : t + K]
        bs_t = torch.from_numpy(bs)
        ba_t = torch.from_numpy(ba)

        optimizer.zero_grad()
        loss = _compute_loss(p_fit, bs_t, ba_t, state_stds, train_ds.dt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learnable, max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    return p_fit, losses


# --------------------------------------------------------------------- validation


def _rollout_eval(
    state0: np.ndarray, actions: np.ndarray, params: TorchVehicleParams, dt: float
) -> np.ndarray:
    state = torch.from_numpy(state0)
    actions_t = torch.from_numpy(actions)
    out = []
    with torch.no_grad():
        for t in range(actions.shape[0]):
            state = torch_step(state, actions_t[t], params, dt=dt)
            out.append(state.detach().numpy())
    return np.stack(out)


def validate(
    p_fit: TorchVehicleParams,
    p_true: TorchVehicleParams,
    val_ds: TrajectoryDataset,
    rollout_steps: int,
) -> dict:
    """Per-param recovery error + per-component held-out rollout RMSE."""
    n_traj = val_ds.states.shape[0]
    K = min(rollout_steps, val_ds.states.shape[1] - 1)

    # Param recovery
    fit_d = p_fit.as_dict()
    true_d = p_true.as_dict()
    param_err = {}
    for name in fit_d:
        v_fit, v_true = fit_d[name], true_d[name]
        param_err[name] = 100.0 * (v_fit - v_true) / abs(v_true)

    # Rollout RMSE
    sq_err = {name: 0.0 for name in _STATE_GROUPS}
    for i in range(n_traj):
        pred = _rollout_eval(val_ds.states[i, 0], val_ds.actions[i, :K], p_fit, val_ds.dt)
        target = val_ds.states[i, 1 : K + 1]
        for name, sl in _STATE_GROUPS.items():
            sq_err[name] += np.sum((pred[:, sl] - target[:, sl]) ** 2)
    rmse = {}
    for name, sl in _STATE_GROUPS.items():
        n_elements = K * (sl.stop - sl.start) * n_traj
        rmse[name] = float(np.sqrt(sq_err[name] / n_elements))

    return {
        "param_err_pct": param_err,
        "rollout_rmse": rmse,
        "rollout_horizon_steps": K,
        "rollout_horizon_s": K * val_ds.dt,
        "n_val_traj": n_traj,
    }


# --------------------------------------------------------------------- runner


def run_one_drone(
    preset_name: str,
    out_root: Path,
    *,
    n_train_traj: int,
    n_val_traj: int,
    n_steps: int,
    dt: float,
    n_epochs: int,
    batch_size: int,
    K: int,
    lr: float,
    init_perturbation: float,
    val_rollout_steps: int,
) -> dict:
    print(f"\n{'=' * 60}\n=== Drone preset: {preset_name}\n{'=' * 60}")
    out_dir = out_root / preset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    params_np = load_preset(preset_name)
    p_true = TorchVehicleParams.from_vehicle_params(params_np)

    print(f"  Generating {n_train_traj} training trajectories...")
    t0 = perf_counter()
    train_ds = generate_dataset(n_train_traj, n_steps, dt, params_np, seed=0)
    train_ds.save(out_dir / "train.npz")

    print(f"  Generating {n_val_traj} validation trajectories...")
    val_ds = generate_dataset(n_val_traj, n_steps, dt, params_np, seed=42)
    val_ds.save(out_dir / "val.npz")
    print(f"  Datasets generated in {perf_counter() - t0:.1f}s")

    results = {"preset": preset_name, "params_true": params_np.__dict__ | {
        "inertia": params_np.inertia.tolist(),
        "drag_coeff": params_np.drag_coeff.tolist(),
    }}

    fixed_options = {
        "unconstrained": None,
        "constrained": {"mass": float(params_np.mass), "arm_length": float(params_np.arm_length)},
    }

    for cond_name, fixed in fixed_options.items():
        print(f"\n  -- {cond_name} fit ({n_epochs} epochs) --")
        t0 = perf_counter()
        p_fit, loss_curve = train(
            p_true, train_ds,
            n_epochs=n_epochs, batch_size=batch_size, K=K, lr=lr,
            init_perturbation=init_perturbation, fixed=fixed, seed=0,
        )
        elapsed = perf_counter() - t0
        print(f"  trained in {elapsed:.1f}s; final loss: {loss_curve[-1]:.4e}")

        metrics = validate(p_fit, p_true, val_ds, val_rollout_steps)
        worst = max(abs(v) for v in metrics["param_err_pct"].values())
        print(f"  worst param err: {worst:.2f} %")
        print(f"  pos rmse: {metrics['rollout_rmse']['pos']:.2e} m  | "
              f"vel rmse: {metrics['rollout_rmse']['vel']:.2e} m/s")

        cond_dir = out_dir / cond_name
        cond_dir.mkdir(exist_ok=True)
        torch.save(
            {"params_fit": p_fit.as_dict(), "params_true": p_true.as_dict(), "fixed": fixed or {}},
            cond_dir / "fit.pt",
        )
        np.save(cond_dir / "loss_curve.npy", np.asarray(loss_curve))

        results[cond_name] = {
            "final_loss": loss_curve[-1],
            "init_loss": loss_curve[0],
            "train_seconds": elapsed,
            "params_fit": p_fit.as_dict(),
            "metrics": metrics,
        }

    with (out_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="quick mode: fewer trajectories + epochs for iteration")
    ap.add_argument("--out", type=Path, default=Path("outputs/sysid/multi_drone"))
    ap.add_argument("--presets", nargs="+", default=list(DEFAULT_PRESETS))
    args = ap.parse_args()

    if args.quick:
        cfg = dict(n_train_traj=50, n_val_traj=20, n_steps=200, dt=0.01,
                   n_epochs=150, batch_size=16, K=20, lr=1e-3,
                   init_perturbation=1.5, val_rollout_steps=50)
    else:
        cfg = dict(n_train_traj=200, n_val_traj=50, n_steps=200, dt=0.01,
                   n_epochs=500, batch_size=16, K=20, lr=1e-3,
                   init_perturbation=1.5, val_rollout_steps=50)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {args.out.resolve()}")
    print(f"Config: {cfg}")
    print(f"Presets: {args.presets}")

    all_results = {}
    for preset in args.presets:
        all_results[preset] = run_one_drone(preset, args.out, **cfg)

    summary_path = args.out / "summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {"config": cfg, "presets": list(args.presets), "results": all_results},
            f, indent=2, default=str,
        )
    print(f"\nSummary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
