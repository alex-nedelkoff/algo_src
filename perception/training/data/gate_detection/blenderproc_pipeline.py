"""BlenderProc synthetic data pipeline for gate detection.

Run with:
    blenderproc run perception/training/data/gate_detection/blenderproc_pipeline.py \\
        --gate-obj perception/training/data/gate_detection/assets/gate.obj \\
        --output-dir ~/corvidx/data/gate_detection/blenderproc \\
        --n-samples 30

This script runs inside Blender's bundled Python, where ``bpy`` is available.
``bproc.init()`` is called ONCE before the render loop; subsequent iterations
use ``bproc.clean_up()`` to reset the scene without re-initialising Blender.
"""

import argparse
import random
import sys
from pathlib import Path

import blenderproc as bproc
import bpy
import cv2
import numpy as np
import mathutils  # bundled with Blender

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="BlenderProc gate detection pipeline")
parser.add_argument("--gate-obj", required=True, help="Path to gate .obj file")
parser.add_argument("--drone-obj", default=None, help="Path to drone .obj file (optional)")
parser.add_argument("--output-dir", required=True, help="Root output directory")
parser.add_argument("--n-samples", type=int, default=30, help="Number of samples to render")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# blenderproc injects its own args; use parse_known_args to avoid errors
args, _ = parser.parse_known_args()

random.seed(args.seed)
np.random.seed(args.seed)

output_dir = Path(args.output_dir).expanduser()
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 640, 480

# Camera intrinsics for a ~90° horizontal FOV at 640x480
FX = 320.0
FY = 320.0
CX = 320.0
CY = 240.0

K_MATRIX = [[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]]

# Gate inner corner offsets in gate-local XY plane (1.5 m gate, Z=0)
# Order: TL, TR, BR, BL
GATE_CORNERS_LOCAL = np.array([
    [-0.75,  0.75, 0.0],
    [ 0.75,  0.75, 0.0],
    [ 0.75, -0.75, 0.0],
    [-0.75, -0.75, 0.0],
], dtype=np.float64)

# LED-style emissive colours (RGB, linear)
LED_COLORS = [
    [0.0, 0.3, 1.0],   # blue
    [0.6, 0.0, 1.0],   # purple
    [0.0, 1.0, 0.3],   # green
    [1.0, 1.0, 1.0],   # white
    [0.0, 1.0, 1.0],   # cyan
]

# ---------------------------------------------------------------------------
# Helper: configure renderer settings
# ---------------------------------------------------------------------------

def configure_renderer() -> None:
    """Set Cycles renderer options and output resolution."""
    bproc.renderer.set_renderer_type("CYCLES")
    bproc.renderer.set_max_amount_of_samples(64)
    bproc.renderer.set_noise_threshold(0.05)
    bproc.renderer.set_output_format("PNG")

    scene = bpy.context.scene
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100

    # Enable depth and segmentation passes
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_segmentation_output(
        map_by=["category_id", "instance"]
    )


# ---------------------------------------------------------------------------
# Helper: set solid background colour
# ---------------------------------------------------------------------------

def set_background_color(r: float, g: float, b: float) -> None:
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    # Find or create Background node
    bg_node = None
    for node in nodes:
        if node.type == "BACKGROUND":
            bg_node = node
            break
    if bg_node is None:
        bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = (r, g, b, 1.0)
    bg_node.inputs["Strength"].default_value = 1.0


# ---------------------------------------------------------------------------
# Helper: project 3-D world points to pixel coords
# ---------------------------------------------------------------------------

def project_world_to_pixel(
    world_pts: np.ndarray,
) -> np.ndarray:
    """Project (N, 3) world points to (N, 2) pixel coords using bproc camera."""
    projected = bproc.camera.project_points(world_pts)  # returns (N, 2) float
    return projected  # [x, y] in pixel space


# ---------------------------------------------------------------------------
# ONE-TIME initialisation
# ---------------------------------------------------------------------------

bproc.init()

# Set camera intrinsics (persists across clean_up calls)
bproc.camera.set_intrinsics_from_K_matrix(
    K_matrix=K_MATRIX,
    image_width=WIDTH,
    image_height=HEIGHT,
)

configure_renderer()

# ---------------------------------------------------------------------------
# Render loop
# ---------------------------------------------------------------------------

for idx in range(args.n_samples):
    sample_id = f"{idx:05d}"

    # ---- Reset scene (keep Blender alive, keep camera intrinsics) ----------
    if idx > 0:
        bproc.clean_up(clean_up_camera=False)
        # Re-apply renderer settings lost after clean_up
        configure_renderer()

    # ---- Background --------------------------------------------------------
    r, g, b = np.random.uniform(0.0, 1.0, 3).tolist()
    set_background_color(r, g, b)

    # ---- Load gates --------------------------------------------------------
    n_gates = random.randint(1, 3)
    gate_objects = []

    for g_idx in range(n_gates):
        loaded = bproc.loader.load_obj(args.gate_obj)
        for obj in loaded:
            obj.set_cp("category_id", 1)

            # Random position: forward 2–12 m (X), lateral ±3 m (Y), vertical ±2 m (Z)
            x = random.uniform(2.0, 12.0)
            y = random.uniform(-3.0, 3.0)
            z = random.uniform(-2.0, 2.0)
            obj.set_location([x, y, z])

            # Random rotation (Euler XYZ)
            rx = random.uniform(-0.3, 0.3)
            ry = random.uniform(-0.3, 0.3)
            rz = random.uniform(-np.pi, np.pi)
            obj.set_rotation_euler([rx, ry, rz])

            # Random LED-style emissive material
            color = random.choice(LED_COLORS)
            mat = obj.get_materials()
            if mat:
                mat[0].set_principled_shader_value("Emission Color", color + [1.0])
                mat[0].set_principled_shader_value("Emission Strength", 3.0)
            else:
                new_mat = bproc.material.create("gate_led")
                new_mat.set_principled_shader_value("Emission Color", color + [1.0])
                new_mat.set_principled_shader_value("Emission Strength", 3.0)
                obj.replace_materials(new_mat)

            gate_objects.append(obj)

    # ---- Load drones (optional) -------------------------------------------
    if args.drone_obj is not None:
        n_drones = random.randint(0, 2)
        for _ in range(n_drones):
            loaded = bproc.loader.load_obj(args.drone_obj)
            for obj in loaded:
                obj.set_cp("category_id", 2)
                x = random.uniform(1.5, 10.0)
                y = random.uniform(-4.0, 4.0)
                z = random.uniform(-2.5, 2.5)
                obj.set_location([x, y, z])
                obj.set_rotation_euler([
                    random.uniform(-np.pi, np.pi),
                    random.uniform(-np.pi, np.pi),
                    random.uniform(-np.pi, np.pi),
                ])

    # ---- Point light -------------------------------------------------------
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([
        random.uniform(-3.0, 3.0),
        random.uniform(-3.0, 3.0),
        random.uniform(1.0, 5.0),
    ])
    light.set_energy(random.uniform(200.0, 1000.0))

    # ---- Camera at origin, looking forward (+X) ----------------------------
    # BlenderProc camera looks down -Z in camera space. To look along +X in
    # world space we need to rotate: first 90° around Y, then -90° around X.
    cam_pose = bproc.math.build_transformation_mat(
        location=[0.0, 0.0, 0.0],
        rotation=mathutils.Euler((np.pi / 2, 0.0, np.pi / 2), "XYZ"),
    )
    bproc.camera.add_camera_pose(cam_pose)

    # ---- Render ------------------------------------------------------------
    render_data = bproc.renderer.render()

    # ---- Extract RGB -------------------------------------------------------
    # render_data["colors"] is a list of (H, W, 4) RGBA float32 arrays
    rgba = render_data["colors"][0]  # float32 in [0, 1]
    rgb_float = rgba[:, :, :3]
    rgb_uint8 = (np.clip(rgb_float, 0.0, 1.0) * 255).astype(np.uint8)
    # Convert RGB → BGR for OpenCV
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    # ---- Extract segmentation masks ----------------------------------------
    # "instance_segmaps" → per-pixel instance ID; "class_segmaps" → category_id
    instance_seg = render_data["instance_segmaps"][0]   # (H, W) int
    category_seg = render_data["class_segmaps"][0]      # (H, W) int

    # Gate mask: pixels where category_id == 1, value = instance ID
    gate_mask = np.where(category_seg == 1, instance_seg, 0).astype(np.uint8)

    # Obstacle mask: pixels where category_id == 2, value = 2
    obstacle_mask = np.where(category_seg == 2, 2, 0).astype(np.uint8)

    # ---- Project gate corners to 2D ----------------------------------------
    corner_coords = []
    gates_in_frame = 0

    for gate_id, gate_obj in enumerate(gate_objects, start=1):
        mat_world = gate_obj.blender_obj.matrix_world

        # Transform local corners to world coordinates
        world_corners = []
        for lc in GATE_CORNERS_LOCAL:
            local_h = mathutils.Vector((lc[0], lc[1], lc[2], 1.0))
            world_h = mat_world @ local_h
            world_corners.append([world_h.x, world_h.y, world_h.z])

        world_corners_np = np.array(world_corners, dtype=np.float64)

        # Project to pixel coordinates
        px = project_world_to_pixel(world_corners_np)  # (4, 2)

        # Check all corners are within frame
        in_frame = all(
            0 <= px[i, 0] < WIDTH and 0 <= px[i, 1] < HEIGHT
            for i in range(4)
        )
        if not in_frame:
            continue

        corners_list = [[float(px[i, 0]), float(px[i, 1])] for i in range(4)]
        corner_coords.append({
            "gate_id": gate_id,
            "corners": corners_list,  # TL, TR, BR, BL
            "confidence": 1.0,
        })
        gates_in_frame += 1

    # ---- Heatmaps ----------------------------------------------------------
    from perception.training.data.gate_detection.format import (
        generate_corner_heatmaps,
        save_sample,
    )

    heatmaps = generate_corner_heatmaps(corner_coords, HEIGHT, WIDTH)

    # ---- Intrinsics for metadata -------------------------------------------
    intrinsics = {
        "fx": FX,
        "fy": FY,
        "cx": CX,
        "cy": CY,
        "width": WIDTH,
        "height": HEIGHT,
        "K": K_MATRIX,
    }

    metadata = {
        "source": "blenderproc",
        "sample_id": sample_id,
        "seed": args.seed,
        "sample_index": idx,
        "n_gates_placed": n_gates,
        "n_gates_in_frame": gates_in_frame,
        "intrinsics": intrinsics,
    }

    # ---- Save --------------------------------------------------------------
    sample_dir = output_dir / sample_id
    save_sample(
        sample_dir,
        rgb=bgr,
        gate_mask=gate_mask,
        obstacle_mask=obstacle_mask,
        corner_coords=corner_coords,
        corner_heatmaps=heatmaps,
        metadata=metadata,
    )

    print(f"[{idx + 1}/{args.n_samples}] Saved {sample_id} ({gates_in_frame} gates in frame)")

print("Done.")
