"""Render a single RGB+depth image from the centre of the textured room.

Sanity check before wiring MASt3R-SLAM: confirms PyBullet's OpenGL
renderer is actually picking up the textures from room.mtl.

Saves outputs/perception/textured_room_preview.png (RGB) and prints the
depth-image min/max so we know the geometry rendered too.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pybullet as pb
from PIL import Image


ROOM_DIR = Path("sim/assets/textured_room_v1")
OUT_DIR = Path("outputs/perception")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    cid = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(str(ROOM_DIR), physicsClientId=cid)
    rid = pb.loadURDF(str(ROOM_DIR / "room.urdf"),
                      basePosition=[0, 0, 0], useFixedBase=True,
                      physicsClientId=cid)
    print(f"Loaded room as body {rid}")

    width, height, vfov = 512, 384, 70.0
    eye    = [-5.0, -3.0, 1.5]              # corner, drone height
    target = [5.0,  3.0, 1.5]                # look across diagonal
    up     = [0.0, 0.0, 1.0]
    view = pb.computeViewMatrix(cameraEyePosition=eye,
                                 cameraTargetPosition=target,
                                 cameraUpVector=up)
    proj = pb.computeProjectionMatrixFOV(fov=vfov, aspect=width / height,
                                          nearVal=0.1, farVal=100.0)

    # Use OpenGL renderer so textures actually show. ER_TINY_RENDERER is
    # software-rasterized and ignores textures.
    # Bright ambient lighting + downward overhead light. Default PyBullet
    # lighting under-illuminates indoor scenes (all walls look near-black);
    # bumping ambient to 0.8 brings out textures so MASt3R has features.
    _, _, rgba, depth, _ = pb.getCameraImage(
        width, height, viewMatrix=view, projectionMatrix=proj,
        renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        lightAmbientCoeff=0.85,
        lightDiffuseCoeff=0.6,
        lightSpecularCoeff=0.05,
        lightDirection=[0, 0, -1],
        lightColor=[1.0, 1.0, 1.0],
        shadow=0,
        physicsClientId=cid,
    )
    rgba = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
    rgb = rgba[..., :3]
    depth = np.asarray(depth, dtype=np.float32).reshape(height, width)

    out_rgb = OUT_DIR / "textured_room_preview.png"
    Image.fromarray(rgb).save(out_rgb)
    print(f"Wrote {out_rgb}")

    print(f"depth: min={depth.min():.3f}  max={depth.max():.3f}  "
          f"mean={depth.mean():.3f}")
    # Spot-check: at the centre, looking +x, distance to far wall is 7m
    # (HALF_W). PyBullet z-buffer is non-linear; print the centre pixel.
    cz = float(depth[height // 2, width // 2])
    near, far = 0.1, 100.0
    cz_metric = (far * near) / (far - cz * (far - near))
    print(f"centre pixel depth: zbuf={cz:.4f} -> metric={cz_metric:.2f} m  "
          f"(expected ~7.0 m to far wall)")

    # Per-channel intensity stats — confirms textures vary, not flat shading.
    print(f"RGB stats: R mean={rgb[..., 0].mean():.0f} std={rgb[..., 0].std():.0f}  "
          f"G mean={rgb[..., 1].mean():.0f} std={rgb[..., 1].std():.0f}  "
          f"B mean={rgb[..., 2].mean():.0f} std={rgb[..., 2].std():.0f}")
    if rgb.std() < 5:
        print("WARNING: RGB std very low - textures probably not loaded")
    else:
        print("RGB has texture variation - OK")
    pb.disconnect(cid)


if __name__ == "__main__":
    main()
