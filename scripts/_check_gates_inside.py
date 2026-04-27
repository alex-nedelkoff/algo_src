"""One-off: confirm all 5 gate ENU positions sit inside the warehouse PyBullet AABB."""
from __future__ import annotations
import json
from pathlib import Path

import pybullet as p

from sim.pybullet.warehouse_loader import WarehouseScene

ASSETS = Path("sim/assets/warehouse_fab_v1")


def main() -> int:
    cid = p.connect(p.DIRECT)
    try:
        handles = WarehouseScene(asset_dir=ASSETS).load_into(cid)
        wh_min, wh_max = p.getAABB(handles.warehouse_body_id, physicsClientId=cid)

        gates = json.loads((ASSETS / "gates_enu.json").read_text(encoding="utf-8"))
        all_in = True
        for name, g in gates.items():
            pos = g["position_enu"]
            inside = all(wh_min[i] <= pos[i] <= wh_max[i] for i in range(3))
            mark = "OK " if inside else "OUT"
            print(f"  {name}  ENU=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})  {mark}")
            all_in = all_in and inside

        print(f"warehouse AABB min  ({wh_min[0]:+.2f}, {wh_min[1]:+.2f}, {wh_min[2]:+.2f})")
        print(f"warehouse AABB max  ({wh_max[0]:+.2f}, {wh_max[1]:+.2f}, {wh_max[2]:+.2f})")
        print("ALL GATES INSIDE" if all_in else "AT LEAST ONE GATE OUTSIDE")
    finally:
        p.disconnect(cid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
