"""Headless render of the warehouse scene via PyBullet's TINY renderer (works on Windows).

Renders top-down + perspective views, saves PNG. PyBullet doesn't need EGL.
"""
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image

from sim.pybullet.warehouse_loader import WarehouseScene

ASSETS = Path("sim/assets/warehouse_fab_v1")
OUT_DIR = Path("outputs/render")
OUT_DIR.mkdir(parents=True, exist_ok=True)
W, H = 1024, 768

cid = p.connect(p.DIRECT)
try:
    handles = WarehouseScene(asset_dir=ASSETS).load_into(cid)
    wh_min, wh_max = p.getAABB(handles.warehouse_body_id, physicsClientId=cid)
    center = [(a + b) / 2 for a, b in zip(wh_min, wh_max)]
    diag = float(np.linalg.norm(np.subtract(wh_max, wh_min)))
    print(f"AABB center={center}  diag={diag:.2f}m")

    proj = p.computeProjectionMatrixFOV(fov=60, aspect=W / H, nearVal=0.1, farVal=200)

    views = [
        ("top",       [center[0], center[1], center[2] + diag], center, [1, 0, 0]),
        ("perspective", [center[0] + diag, center[1] - diag, center[2] + diag * 0.5], center, [0, 0, 1]),
        ("inside",    [center[0], center[1], center[2] + 1.5], [center[0] + 5, center[1], center[2] + 1.5], [0, 0, 1]),
    ]
    for name, eye, look_at, up in views:
        view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=look_at, cameraUpVector=up)
        _, _, rgba, _, _ = p.getCameraImage(
            width=W, height=H, viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER, physicsClientId=cid,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
        out = OUT_DIR / f"warehouse_{name}.png"
        Image.fromarray(rgb).save(out)
        print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")
finally:
    p.disconnect(cid)
