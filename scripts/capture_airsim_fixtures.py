"""Capture AirSim fixture renders + depth + intrinsics from the warehouse.

REQUIREMENTS (per COR-92):
  - Ubuntu 20.04 / 22.04, NVIDIA driver 545+, Vulkan 1.3
  - Cosys-AirSim Python client installed:
      git clone https://github.com/Cosys-Lab/Cosys-AirSim.git
      cd Cosys-AirSim/PythonClient && pip install -e .
      (imports as `cosysairsim`, NOT the Microsoft `airsim` package)
  - The warehouse binary running:
      DISPLAY=:0 ~/warehouse-packaged/Linux/Warehouse.sh -windowed \
        -ResX=1280 -ResY=720 &
      sleep 25

  Output (this script's responsibility):
      tests/fixtures/airsim_warehouse_v1/
          pose_NN_scene.png
          pose_NN_depth.npy
          pose_NN.json
          manifest.yaml

  Each capture pose is hand-picked in ENU; converted to NED before
  driving the drone. Camera intrinsics are recorded in the JSON sidecar
  so the comparison harness (compare_pybullet_vs_fixtures.py) can match
  the PyBullet projection exactly.

  GOTCHAS (per COR-92):
   - Never call client.reset() — corrupts spawn permanently.
   - Never .join() on async flight calls — deadlocks on collision.
   - Use ImageType.DepthPlanar (planar Z), NOT DepthPerspective.
   - Use ImageResponse.camera_position / camera_orientation, NOT
     simGetVehiclePose() — the latter drifts up to 0.43 m.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

import cosysairsim as airsim
from sim.pybullet.coords import ned_to_enu_position  # for the inverse


# Hand-picked ENU camera poses giving warehouse coverage.
# Refine after first capture pass.
CAPTURE_POSES_ENU = [
    {"name": "pose_00", "position_enu": [0.0, 0.0, 2.0],  "yaw_deg": 0},
    {"name": "pose_01", "position_enu": [3.0, 0.0, 2.0],  "yaw_deg": 90},
    {"name": "pose_02", "position_enu": [3.0, 3.0, 2.0],  "yaw_deg": 180},
    {"name": "pose_03", "position_enu": [0.0, 3.0, 2.0],  "yaw_deg": 270},
    {"name": "pose_04", "position_enu": [1.5, 1.5, 4.0],  "yaw_deg": 0},
    {"name": "pose_05", "position_enu": [1.5, 1.5, 0.5],  "yaw_deg": 0},
    # Add up to ~15 in total — refine after seeing first pass.
]

WIDTH, HEIGHT = 640, 480
FOV_DEG = 90.0


def enu_to_ned(p_enu):
    return [p_enu[1], p_enu[0], -p_enu[2]]


def main(out_dir: Path = Path("tests/fixtures/airsim_warehouse_v1")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    for pose in CAPTURE_POSES_ENU:
        pos_ned = enu_to_ned(pose["position_enu"])
        # Build orientation quaternion from yaw (NED).
        yaw_rad = np.deg2rad(pose["yaw_deg"])
        cy, sy = np.cos(yaw_rad / 2), np.sin(yaw_rad / 2)
        q_ned = airsim.Quaternionr(0, 0, sy, cy)  # x, y, z, w

        client.simSetVehiclePose(
            airsim.Pose(airsim.Vector3r(*pos_ned), q_ned),
            ignore_collision=True,
        )

        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False),
        ])
        scene_resp, depth_resp = responses
        scene_rgb = np.frombuffer(scene_resp.image_data_uint8, dtype=np.uint8).reshape(
            scene_resp.height, scene_resp.width, 3
        )
        depth_planar = np.array(depth_resp.image_data_float, dtype=np.float32).reshape(
            depth_resp.height, depth_resp.width
        )

        # Use camera_position / camera_orientation from the response.
        cam_pos_ned = [
            scene_resp.camera_position.x_val,
            scene_resp.camera_position.y_val,
            scene_resp.camera_position.z_val,
        ]
        cam_quat_ned_xyzw = [
            scene_resp.camera_orientation.x_val,
            scene_resp.camera_orientation.y_val,
            scene_resp.camera_orientation.z_val,
            scene_resp.camera_orientation.w_val,
        ]

        # Save outputs.
        from PIL import Image
        Image.fromarray(scene_rgb).save(out_dir / f"{pose['name']}_scene.png")
        np.save(out_dir / f"{pose['name']}_depth.npy", depth_planar)

        sidecar = {
            "name": pose["name"],
            "position_ned": cam_pos_ned,
            "position_enu": ned_to_enu_position(np.array(cam_pos_ned)).tolist(),
            "orientation_ned_xyzw": cam_quat_ned_xyzw,
            "intrinsics": {
                "width": scene_resp.width,
                "height": scene_resp.height,
                "fov_deg": FOV_DEG,
            },
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / f"{pose['name']}.json").write_text(json.dumps(sidecar, indent=2))

    manifest = {
        "schema_version": 1,
        "warehouse_binary": "warehouse_5gates_v1.tar.gz",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "num_poses": len(CAPTURE_POSES_ENU),
        "intrinsics": {"width": WIDTH, "height": HEIGHT, "fov_deg": FOV_DEG},
    }
    (out_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"Wrote {len(CAPTURE_POSES_ENU)} fixtures to {out_dir}")


if __name__ == "__main__":
    main()
