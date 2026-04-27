"""CLI: generate a sysID training dataset by rolling out NumpyQuadDynamics.

Usage:
    PYTHONPATH=. python -m scripts.sysid.generate_dataset \
        --out outputs/sysid/train.npz \
        --n-traj 200 --n-steps 200 --dt 0.01 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sim.dynamics.params import VehicleParams
from scripts.sysid.dataset import generate_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="output .npz path")
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--n-steps", type=int, default=200,
                    help="rollout length per trajectory (so total transitions = n_traj * n_steps)")
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    params = VehicleParams()  # CrazyFlie 2.1 defaults — privileged ground truth
    print(f"Generating {args.n_traj} trajectories x {args.n_steps} steps "
          f"@ {args.dt}s ({args.n_traj * args.n_steps:,} transitions)...")
    ds = generate_dataset(
        n_traj=args.n_traj,
        n_steps=args.n_steps,
        dt=args.dt,
        params=params,
        seed=args.seed,
    )

    ds.save(args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Wrote {args.out}  ({size_mb:.1f} MB)")
    print(f"  states  shape: {ds.states.shape}")
    print(f"  actions shape: {ds.actions.shape}")
    print(f"  styles: " + ", ".join(f"{s}={ds.styles.count(s)}" for s in set(ds.styles)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
