"""Convert the playroom 3DGS-extracted mesh into a WarehouseRaceEnv-compatible asset bundle.

The 3DGS mesh has +y-DOWN orientation (typical for COLMAP-trained scenes).
PyBullet's gate-race envs assume ENU (Z-up). We rotate +90 deg about the X
axis to remap (new_y = old_z, new_z = -old_y), so +z becomes the up
direction. Then we save the transformed mesh as ``warehouse.obj`` in
``sim/assets/playroom_v1/`` and write a placeholder ``gates_enu.json``
with two gates spanning the camera-bbox open space.

The env's ``world_offset_z`` parameter handles translating the floor to z=0
at load time — we leave the mesh in its raw (centred-around-origin) frame
and let WarehouseRaceEnv apply the offset.

Run on the laptop where the playroom assets live::

    python scripts\\perception\\_phase2_build_playroom_asset.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SRC_DIR = REPO / "sim" / "assets" / "3dgs" / "playroom"
DST_DIR = REPO / "sim" / "assets" / "playroom_v1"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-gate", action="store_true",
                    help="Trivial diagnostic: one gate 4 m straight ahead of "
                         "spawn (no M3 pipeline, no closed loop). If the "
                         "policy can't fly this, the obs/action layer is broken.")
    args = ap.parse_args()

    import open3d as o3d
    src_mesh = SRC_DIR / "mesh.obj"
    print(f"[1/4] Loading source mesh: {src_mesh}")
    mesh = o3d.io.read_triangle_mesh(str(src_mesh))
    verts = np.asarray(mesh.vertices)
    print(f"      {len(verts)} verts, {len(np.asarray(mesh.triangles))} tris")
    print(f"      pre-transform bbox min/max: "
          f"{verts.min(0).round(2)} / {verts.max(0).round(2)}")

    # ---- 2. Apply axis transform: rotate +90 deg about X axis -----------
    # new_x = old_x;  new_y = old_z;  new_z = -old_y
    # This maps +y-down  ->  +z-up (ENU).
    print(f"[2/4] Applying +90-deg rotation about X (Y-down -> Z-up)")
    R = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0],   # new_y = old_z
        [0.0, -1.0,  0.0],   # new_z = -old_y
    ])
    new_verts = verts @ R.T
    mesh.vertices = o3d.utility.Vector3dVector(new_verts)
    # Recompute normals since we changed orientation
    mesh.compute_vertex_normals()

    post = np.asarray(mesh.vertices)
    print(f"      post-transform bbox min/max: "
          f"{post.min(0).round(2)} / {post.max(0).round(2)}")
    floor_z = float(np.percentile(post[:, 2], 1))
    ceiling_z = float(np.percentile(post[:, 2], 99))
    print(f"      floor z (p1)  = {floor_z:.3f}")
    print(f"      ceiling z (p99) = {ceiling_z:.3f}")
    print(f"      world_offset_z to put floor at sim-z=0: {-floor_z:.3f}")

    # ---- 3. Save transformed mesh ---------------------------------------
    DST_DIR.mkdir(parents=True, exist_ok=True)
    dst_mesh = DST_DIR / "warehouse.obj"   # name expected by WarehouseRaceEnv
    print(f"[3/4] Saving transformed mesh -> {dst_mesh}")
    o3d.io.write_triangle_mesh(str(dst_mesh), mesh)

    # ---- 4. Author placeholder gates ------------------------------------
    # Build a tiny 2-gate course in the open space.
    # Open space estimate (transformed coords, before world_offset_z): the
    # camera-position bbox in p10..p90 along the horizontal axes was
    # roughly x in [-3.2, +3.3] and (old z, now new y) in [-3.6, +2.9].
    # Use those as our "drone-flyable" region.
    # Gates are placed at mid-height = (floor + ceiling) / 2.
    mid_z = (floor_z + ceiling_z) / 2.0
    # Gate sizing: 1.5 m inner diameter to match the previous G&CNet
    # checkpoint's training distribution. 0.15 m frame thickness.
    INNER_R, OUTER_R = 0.75, 0.90

    # ---- Single-gate trivial test branch -------------------------------
    if args.single_gate:
        # Place ONE gate at world z=3.0 (above floor), facing +X. This z
        # is within the training distribution (warehouse smoke worked at
        # z=2.5). Pre-offset z = 3.0 - world_offset_z = 3.0 - 5.475 = -2.475.
        GATE_WORLD_Z = 3.0
        single_gate_pos = np.array([0.0, 0.0, GATE_WORLD_Z + floor_z])  # pre-offset
        single_gate_yaw = 0.0
        gates = {
            "Gate_00": {
                "position_enu": single_gate_pos.tolist(),
                "orientation_enu_wxyz": [
                    float(np.cos(single_gate_yaw / 2.0)),
                    0.0, 0.0,
                    float(np.sin(single_gate_yaw / 2.0)),
                ],
                "inner_radius_m": 0.75,
                "outer_radius_m": 0.90,
            },
        }
        spawn_world = single_gate_pos + np.array([0.0, 0.0, -floor_z]) \
                                      - np.array([4.0, 0.0, 0.0])
        print(f"      single-gate mode: gate at world={(single_gate_pos + [0, 0, -floor_z]).round(3).tolist()}")
        print(f"      recommended spawn (WORLD frame): pos={spawn_world.round(3).tolist()}  yaw_rad=0.0")
        gates_path = DST_DIR / "gates_enu.json"
        gates_path.write_text(
            json.dumps(gates, indent=2, default=lambda o: o.tolist())
        )
        print(f"\n=== SUMMARY (single-gate diagnostic) ===")
        print(f"  asset_dir = {DST_DIR}")
        print(f"  1 gate at (0, 0, {mid_z:.3f}) pre-offset; spawn 4 m behind")
        print(f"  world_offset_z = {-floor_z:.3f}")
        return 0

    # Raw rectangle corners inside the well-photographed camera-bbox zone
    # (x in [-2.5, +2.5], y in [-2.0, +1.0]). These are the "user intent"
    # waypoints — M3's cubic-spline + anticipatory-yaw pipeline below
    # densifies + smooths them into the actual targets the policy sees.
    LOW_Z = mid_z - 0.5
    HIGH_Z = mid_z + 0.5
    raw_waypoints = np.array([
        [-2.5, -2.0, LOW_Z],    # SW low
        [+2.5, -2.0, LOW_Z],    # SE low
        [+2.5, +1.0, HIGH_Z],   # NE high
        [-2.5, +1.0, HIGH_Z],   # NW high
    ])

    # ---- M3 pipeline -----------------------------------------------------
    # Cubic B-spline through corners (rounds the rectangle to a smooth
    # racing line), arc-length-uniform resample at 4 m spacing (matches the
    # G&CNet's training distribution of 3–6 m inter-gate spacing).
    from sim.tracks.waypoint import resample_waypoints, waypoints_to_track
    resampled = resample_waypoints(
        raw_waypoints,
        target_spacing=4.0,
        closed_loop=True,
        method="cubic",
    )
    # Closed-loop resample appends the first point as the closing point;
    # drop it so we don't double-count.
    if len(resampled) > 1 and np.allclose(resampled[0], resampled[-1]):
        resampled = resampled[:-1]
    print(f"      raw corners: {len(raw_waypoints)} -> resampled: "
          f"{len(resampled)} waypoints @ ~4 m spacing")

    # Anticipatory yaws (look 2 wp ahead, EMA-smoothed). The policy will
    # see each waypoint as a synthetic gate with these yaws.
    track = waypoints_to_track(
        resampled,
        yaw_lookahead=2,
        yaw_smoothing_alpha=0.5,
        closed_loop=True,
    )

    # Write each Track gate as a Gate_NN entry. Set reorient_to_racing_line:
    # false in the config so the env preserves these yaws.
    gates: dict[str, dict] = {}
    for i, g in enumerate(track.gates):
        gates[f"Gate_{i:02d}"] = {
            "position_enu": g.position.tolist(),
            "orientation_enu_wxyz": g.orientation.tolist(),
            "inner_radius_m": INNER_R,
            "outer_radius_m": OUTER_R,
        }

    # Recommended spawn: 1.5 m behind Gate_00 along its M3-computed yaw.
    g0 = track.gates[0]
    qw, _, _, qz = g0.orientation
    yaw0 = 2.0 * float(np.arctan2(qz, qw))
    spawn_back = np.array([np.cos(yaw0), np.sin(yaw0), 0.0]) * 1.5
    spawn_pos = g0.position - spawn_back
    print(f"      recommended spawn (pre-offset_z): "
          f"pos={spawn_pos.round(3).tolist()}  yaw_rad={yaw0:.3f}")
    gates_path = DST_DIR / "gates_enu.json"
    print(f"[4/4] Writing gates_enu.json -> {gates_path}")
    gates_path.write_text(json.dumps(gates, indent=2, default=lambda o: o.tolist()))

    # Print a summary so the user knows what world_offset_z to pass.
    print(f"\n=== SUMMARY ===")
    print(f"  asset_dir       = {DST_DIR}")
    print(f"  warehouse.obj   = {dst_mesh}  ({dst_mesh.stat().st_size / 1e6:.1f} MB)")
    print(f"  gates_enu.json  = {gates_path}  ({len(gates)} gates)")
    print(f"  world_offset_z  = {-floor_z:.3f}  (pass this to WarehouseRaceEnv)")
    print(f"  inferred room   = ~{ceiling_z - floor_z:.1f} m tall (scale ~3.5x real)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
