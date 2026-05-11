"""Smoke test: WarehouseRaceEnv loads playroom_v1 + steps cleanly.

Instantiates ``WarehouseRaceEnv(asset_dir=<playroom_v1>, world_offset_z=5.475)``,
resets, runs 30 zero-action steps, and reports observation shape + any
termination reason. The point is just to validate:

  - the mesh loads as concave trimesh collision
  - the gates_enu.json loader doesn't choke on our placeholder gates
  - the parent GateRaceEnv constructs without missing required args
  - basic step() runs without exceptions

If this passes, we have a foundation for M4 planner + closed-loop work.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
ASSET_DIR = REPO / "sim" / "assets" / "playroom_v1"


def main() -> int:
    from sim.envs.warehouse_race_env import WarehouseRaceEnv

    print(f"[1/3] Instantiating WarehouseRaceEnv(asset_dir={ASSET_DIR})")
    env = WarehouseRaceEnv(
        asset_dir=ASSET_DIR,
        world_offset_z=5.475,
        n_envs=1,
    )
    print(f"      observation_space = {env.observation_space}")
    print(f"      action_space      = {env.action_space}")

    print(f"\n[2/3] Resetting...")
    obs, info = env.reset(seed=0)
    print(f"      obs.shape={getattr(obs, 'shape', None)}  info keys={list(info.keys()) if isinstance(info, dict) else type(info)}")

    print(f"\n[3/3] Running 30 zero-action steps")
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for t in range(30):
        obs, rew, term, trunc, info = env.step(zero_action)
        if t < 3 or t == 29 or term[0] or trunc[0]:
            print(f"      t={t:>2}  rew={float(rew[0]):+.3f}  "
                  f"term={bool(term[0])}  trunc={bool(trunc[0])}")
        if term[0] or trunc[0]:
            reason = getattr(env, "_termination_reasons", [None])[0]
            print(f"      terminated: reason={reason}")
            break

    env.close()
    print("\n=== smoke test PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
