"""B-fixture validation harness: compare PyBullet renders vs AirSim fixtures.

Renders the warehouse + gates in PyBullet at each fixture's camera pose
and intrinsics, builds silhouette masks for both renders, computes
per-pose silhouette IoU, and writes an HTML report.

Pass criteria (per spec): mean IoU ≥ 0.85, no individual pose < 0.70.

Run after fixtures are captured (Task 14 on the Linux box) and after
the build pipeline has produced sim/assets/warehouse_v1/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p


def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union for two boolean silhouette masks.

    If both masks are empty, returns 1.0 (degenerate but interpretable).
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    if not a.any() and not b.any():
        return 1.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def render_pybullet_at_pose(
    client_id: int,
    *,
    position_enu,
    orientation_enu_xyzw,
    width: int,
    height: int,
    fov_deg: float,
    near: float = 0.05,
    far: float = 100.0,
):
    """Render an image from PyBullet at the given camera pose.

    Returns (rgb, depth, segmask). The orientation is the camera's
    world-frame rotation; the camera looks down its local +X axis by
    convention here (matches AirSim camera convention).
    """
    qx, qy, qz, qw = orientation_enu_xyzw
    rot = np.array(p.getMatrixFromQuaternion([qx, qy, qz, qw])).reshape(3, 3)
    forward = rot @ np.array([1.0, 0.0, 0.0])
    up = rot @ np.array([0.0, 0.0, 1.0])
    eye = np.array(position_enu, dtype=np.float64)
    target = eye + forward

    view = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
    proj = p.computeProjectionMatrixFOV(
        fov=fov_deg, aspect=width / height, nearVal=near, farVal=far,
    )
    _w, _h, rgb, depth, segmask = p.getCameraImage(
        width=width, height=height,
        viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client_id,
    )
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(height, width, 4)
    depth = np.asarray(depth, dtype=np.float32).reshape(height, width)
    segmask = np.asarray(segmask, dtype=np.int32).reshape(height, width)
    return rgb, depth, segmask


def pybullet_silhouette(segmask: np.ndarray) -> np.ndarray:
    """Anything not -1 (background) is geometry."""
    return segmask != -1


def airsim_depth_silhouette(depth: np.ndarray, far_plane: float) -> np.ndarray:
    """AirSim DepthPlanar returns 0 for no-return and very large for sky.

    Treat any depth in (0, far_plane) as geometry.
    """
    return (depth > 0.0) & (depth < far_plane)


import argparse
import json as _json
from datetime import datetime, timezone

from sim.pybullet.coords import ned_to_enu_quaternion


def convert_airsim_camera_orientation_to_pybullet(q_ned_xyzw):
    """Convert an AirSim camera orientation quaternion (NED, xyzw) to a
    PyBullet world-frame orientation quaternion (ENU, xyzw).

    AirSim camera convention: body frame is X-forward, Y-right, Z-down,
    expressed in world-NED. PyBullet's camera is built from a forward
    vector derived from this rotation, so the rotation itself just
    needs the NED→ENU basis change applied.
    """
    qx, qy, qz, qw = q_ned_xyzw
    # Reorder xyzw → wxyz, apply NED→ENU flip, reorder back to xyzw.
    q_enu_wxyz = ned_to_enu_quaternion(np.array([qw, qx, qy, qz]))
    return [q_enu_wxyz[1], q_enu_wxyz[2], q_enu_wxyz[3], q_enu_wxyz[0]]


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Warehouse validation</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; }}
  td, th {{ border: 1px solid #ccc; padding: 6px; vertical-align: top; }}
  img {{ max-width: 320px; display: block; }}
  .iou-pass {{ background: #d4edda; }}
  .iou-fail {{ background: #f8d7da; }}
</style></head><body>
<h1>Warehouse v1 — PyBullet vs AirSim fixture comparison</h1>
<p><b>Mean IoU:</b> {mean_iou:.3f} &nbsp; <b>Generated:</b> {ts}</p>
<table>
<tr><th>Pose</th><th>AirSim</th><th>PyBullet</th><th>Silhouette IoU</th></tr>
{rows}
</table>
</body></html>
"""

_ROW_TEMPLATE = """<tr class="{cls}">
<td>{name}</td>
<td><img src="{airsim_png}"></td>
<td><img src="{pybullet_png}"></td>
<td>{iou:.3f}</td>
</tr>"""


def write_html_report(path: Path, rows: list[dict], mean_iou: float) -> None:
    body_rows = "\n".join(
        _ROW_TEMPLATE.format(
            cls="iou-pass" if r["iou"] >= 0.70 else "iou-fail",
            name=r["name"],
            airsim_png=r["airsim_png"],
            pybullet_png=r["pybullet_png"],
            iou=r["iou"],
        )
        for r in rows
    )
    html = _HTML_TEMPLATE.format(
        mean_iou=mean_iou,
        ts=datetime.now(timezone.utc).isoformat(),
        rows=body_rows,
    )
    Path(path).write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=Path, default=Path("sim/assets/warehouse_v1"))
    ap.add_argument("--fixtures", type=Path,
                    default=Path("tests/fixtures/airsim_warehouse_v1"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/validation/warehouse_v1") /
                            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--far-plane", type=float, default=100.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    from sim.pybullet.warehouse_loader import WarehouseScene

    cid = p.connect(p.DIRECT)
    try:
        WarehouseScene(asset_dir=args.assets).load_into(cid)

        rows = []
        ious = []
        for sidecar_path in sorted(args.fixtures.glob("pose_*.json")):
            sidecar = _json.loads(sidecar_path.read_text())
            name = sidecar["name"]
            depth = np.load(args.fixtures / f"{name}_depth.npy")

            rgb, _pdepth, segmask = render_pybullet_at_pose(
                cid,
                position_enu=sidecar["position_enu"],
                orientation_enu_xyzw=convert_airsim_camera_orientation_to_pybullet(
                    sidecar["orientation_ned_xyzw"]
                ),
                width=sidecar["intrinsics"]["width"],
                height=sidecar["intrinsics"]["height"],
                fov_deg=sidecar["intrinsics"]["fov_deg"],
            )
            air_sil = airsim_depth_silhouette(depth, far_plane=args.far_plane)
            pyb_sil = pybullet_silhouette(segmask)
            iou = silhouette_iou(air_sil, pyb_sil)
            ious.append(iou)

            airsim_dst = args.out / f"{name}_airsim.png"
            pybullet_dst = args.out / f"{name}_pybullet.png"
            Image.open(args.fixtures / f"{name}_scene.png").save(airsim_dst)
            Image.fromarray(rgb[..., :3]).save(pybullet_dst)
            rows.append({
                "name": name,
                "airsim_png": airsim_dst.name,
                "pybullet_png": pybullet_dst.name,
                "iou": iou,
            })

        mean_iou = float(np.mean(ious)) if ious else 0.0
        write_html_report(args.out / "index.html", rows, mean_iou=mean_iou)
        print(f"Wrote {len(rows)} comparisons. Mean IoU: {mean_iou:.3f}")
        print(f"Report: {args.out / 'index.html'}")
        if mean_iou < 0.85 or any(r["iou"] < 0.70 for r in rows):
            print("FAIL: validation thresholds not met.")
            return 1
        return 0
    finally:
        p.disconnect(cid)


if __name__ == "__main__":
    raise SystemExit(main())
