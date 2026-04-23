"""One-off: validate the real FAB warehouse loads into PyBullet end-to-end."""
from __future__ import annotations
import sys
from pathlib import Path

import pybullet as p

from sim.pybullet.warehouse_loader import WarehouseScene

ASSETS = Path("sim/assets/warehouse_fab_v1")


def main() -> int:
    cid = p.connect(p.DIRECT)
    try:
        scene = WarehouseScene(asset_dir=ASSETS)
        handles = scene.load_into(cid)

        print(f"warehouse body id: {handles.warehouse_body_id}")
        print(f"gate body ids: {handles.gate_body_ids}")
        print(f"gate names: {handles.gate_names}")

        for gid, gname in zip(handles.gate_body_ids, handles.gate_names):
            pos, orn = p.getBasePositionAndOrientation(gid, physicsClientId=cid)
            print(f"  {gname}: pos={[round(x, 3) for x in pos]}  orn={[round(x, 3) for x in orn]}")

        wh_aabb_min, wh_aabb_max = p.getAABB(handles.warehouse_body_id, physicsClientId=cid)
        extent = [b - a for a, b in zip(wh_aabb_min, wh_aabb_max)]
        print(f"warehouse AABB min (ENU m): {[round(x, 3) for x in wh_aabb_min]}")
        print(f"warehouse AABB max (ENU m): {[round(x, 3) for x in wh_aabb_max]}")
        print(f"warehouse extent (ENU m):   {[round(x, 3) for x in extent]}")
    finally:
        p.disconnect(cid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
