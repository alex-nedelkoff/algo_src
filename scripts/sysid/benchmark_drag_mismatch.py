"""Phase 2: model-mismatch demo via drag-enabled numpy_quad.

Uses ``cf21_with_drag.yaml`` (drag_coeff = [0.10, 0.10, 0.05]) as the
"high-fidelity" data generator and fits the drag-less torch_quad model
against it. The plan was originally to use gym-pybullet-drones, but
that package isn't on PyPI and the install-from-git was unreliable in
this env. Pivoting to drag-enabled numpy_quad provides the same demo
substance — model mismatch + residual structure — without external
dependencies.

In addition to the standard fit + RMSE metrics, this script saves
per-step residual time series (predicted - target) so the report can
plot how the residual evolves over the rollout horizon. Drag-induced
mismatch is expected to appear as a velocity-correlated bias.

Output layout:
    outputs/sysid/multi_drone/cf21_with_drag/
        train.npz, val.npz
        unconstrained/{fit.pt, loss_curve.npy}
        constrained/{fit.pt, loss_curve.npy}
        results.json
        residuals.npz   (NEW vs Phase 1 — keys: pred, target, residual, K, n_traj)

Usage:
    python -m scripts.sysid.benchmark_drag_mismatch [--quick] [--out DIR]
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from scripts.sysid.benchmark_drones import (  # noqa: E402
    _rollout_eval, load_preset, run_one_drone,
)
from scripts.sysid.dataset import TrajectoryDataset  # noqa: E402
from sim.dynamics.torch_params import TorchVehicleParams  # noqa: E402

PRESET = "cf21_with_drag"


def collect_residuals(
    fit_pt: Path,
    val_npz: Path,
    rollout_steps: int,
) -> dict:
    """Roll the fitted model on each held-out trajectory and collect residuals."""
    blob = torch.load(fit_pt, weights_only=False)
    fit_d = blob["params_fit"]
    val_ds = TrajectoryDataset.load(val_npz)
    K = min(rollout_steps, val_ds.states.shape[1] - 1)
    n_traj = val_ds.states.shape[0]

    p_fit = TorchVehicleParams(
        mass=fit_d["mass"], Ixx=fit_d["Ixx"], Iyy=fit_d["Iyy"], Izz=fit_d["Izz"],
        k_thrust=fit_d["k_thrust"], k_torque=fit_d["k_torque"],
        arm_length=fit_d["arm_length"], tau_motor=fit_d["tau_motor"],
        max_omega=float(val_ds.params.max_omega),
    )

    preds = np.empty((n_traj, K, 17), dtype=np.float64)
    targets = np.empty((n_traj, K, 17), dtype=np.float64)
    for i in range(n_traj):
        preds[i] = _rollout_eval(
            val_ds.states[i, 0], val_ds.actions[i, :K], p_fit, val_ds.dt
        )
        targets[i] = val_ds.states[i, 1 : K + 1]
    residuals = preds - targets
    return {
        "pred": preds,
        "target": targets,
        "residual": residuals,
        "K": K,
        "dt": val_ds.dt,
        "n_traj": n_traj,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs/sysid/multi_drone"))
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
    print(f"Phase 2 (drag-mismatch) on preset: {PRESET}")
    print(f"Output root: {args.out.resolve()}")
    print(f"Config: {cfg}")

    # Run the same fit pipeline as Phase 1, but on the drag-bearing preset.
    results = run_one_drone(PRESET, args.out, **cfg)

    # Now collect residuals for both conditions (post-fit). This is the new
    # work — Phase 1 only logged final RMSE per group; here we save the full
    # (n_traj, K, 17) residual cube so the report can plot time-evolution.
    drone_dir = args.out / PRESET
    val_npz = drone_dir / "val.npz"
    for cond in ("unconstrained", "constrained"):
        fit_pt = drone_dir / cond / "fit.pt"
        print(f"\nCollecting residuals for {cond}...")
        bundle = collect_residuals(fit_pt, val_npz, cfg["val_rollout_steps"])
        out_npz = drone_dir / cond / "residuals.npz"
        np.savez(
            out_npz,
            pred=bundle["pred"],
            target=bundle["target"],
            residual=bundle["residual"],
            K=bundle["K"],
            dt=bundle["dt"],
            n_traj=bundle["n_traj"],
        )
        print(f"  → {out_npz}  ({out_npz.stat().st_size / 1024:.1f} KB)")
        # Quick summary stats: per-component RMSE at first vs last step of rollout.
        res = bundle["residual"]
        first_step_rmse = np.sqrt((res[:, 0, :] ** 2).mean(axis=0))
        last_step_rmse = np.sqrt((res[:, -1, :] ** 2).mean(axis=0))
        print(f"  pos rmse:  step 1 = {np.sqrt((first_step_rmse[:3] ** 2).mean()):.2e}  "
              f"step {bundle['K']} = {np.sqrt((last_step_rmse[:3] ** 2).mean()):.2e}")
        print(f"  vel rmse:  step 1 = {np.sqrt((first_step_rmse[3:6] ** 2).mean()):.2e}  "
              f"step {bundle['K']} = {np.sqrt((last_step_rmse[3:6] ** 2).mean()):.2e}")

    # Append phase-2-summary alongside summary.json from Phase 1.
    phase2_summary = args.out / "summary_phase2.json"
    with phase2_summary.open("w") as f:
        json.dump({"config": cfg, "preset": PRESET, "results": results}, f, indent=2, default=str)
    print(f"\nPhase 2 summary → {phase2_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
