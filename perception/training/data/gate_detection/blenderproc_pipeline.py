# BlenderProc synthetic data pipeline for gate detection.
#
# IMPORTANT: `import blenderproc as bproc` MUST be the first import.
# BlenderProc's CLI enforces this so that Blender's bundled Python can
# redirect third-party package imports correctly.
#
# Run with:
#   blenderproc run perception/training/data/gate_detection/blenderproc_pipeline.py \
#       --gate-obj perception/training/data/gate_detection/assets/gate.obj \
#       --output-dir ~/corvidx/data/gate_detection/blenderproc \
#       --n-samples 30

import blenderproc as bproc  # noqa: E402 — must be first import

import argparse
import os
import random
import sys
from pathlib import Path

# Add the algo_src repo root to sys.path so we can import `perception.*`.
# The script lives at <repo>/perception/training/data/gate_detection/, so
# the repo root is four levels up.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import bpy
import cv2
import numpy as np
import mathutils  # bundled with Blender

from perception.training.data.gate_detection.format import (
    generate_corner_heatmaps,
    save_sample,
)

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

def configure_renderer(enable_outputs: bool = True) -> None:
    """Set Cycles renderer options and output resolution.

    Args:
        enable_outputs: If True, also register depth + segmentation outputs.
            Set to False when calling after bproc.clean_up(), because
            enable_depth_output / enable_segmentation_output can each only
            be called once per Blender session.
    """
    # Use bpy directly — bproc.renderer has no set_renderer_type()
    bpy.context.scene.render.engine = "CYCLES"

    # Fix overexposure: set exposure and view transform
    bpy.context.scene.view_settings.exposure = -1.0  # darken by 1 stop
    bpy.context.scene.view_settings.view_transform = "Standard"
    bproc.renderer.set_max_amount_of_samples(64)
    bproc.renderer.set_noise_threshold(0.05)
    bproc.renderer.set_output_format("PNG")

    scene = bpy.context.scene
    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100

    if enable_outputs:
        # Enable depth and segmentation passes.
        # enable_segmentation_output with map_by=["category_id", "instance"]
        # produces render_data keys: "category_id_segmaps" and "instance_segmaps"
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        bproc.renderer.enable_segmentation_output(
            map_by=["category_id", "instance"],
            default_values={"category_id": 0},
        )


# ---------------------------------------------------------------------------
# Helper: set solid background colour
# ---------------------------------------------------------------------------

def set_background_color(r: float, g: float, b: float) -> None:
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
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
# Helper: procedural domain-randomized room
# ---------------------------------------------------------------------------

def _random_procedural_material(name: str) -> bpy.types.Material:
    """Create a random procedural material using Blender shader nodes.

    Randomly selects noise, voronoi, or checker texture with random scale,
    colors, and mapping rotation.  No external textures needed.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes except output
    for n in list(nodes):
        if n.type != "OUTPUT_MATERIAL":
            nodes.remove(n)

    output = nodes.get("Material Output")
    principled = nodes.new(type="ShaderNodeBsdfPrincipled")
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    # Texture coordinate + mapping with random rotation
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (
        random.uniform(0, 6.28),
        random.uniform(0, 6.28),
        random.uniform(0, 6.28),
    )
    mapping.inputs["Scale"].default_value = (
        random.uniform(0.5, 3.0),
        random.uniform(0.5, 3.0),
        random.uniform(0.5, 3.0),
    )
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])

    # Pick a random texture type
    tex_type = random.choice(["noise", "voronoi", "checker", "wave"])

    if tex_type == "noise":
        tex = nodes.new(type="ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = random.uniform(1.0, 50.0)
        tex.inputs["Detail"].default_value = random.uniform(0.0, 16.0)
        tex.inputs["Roughness"].default_value = random.uniform(0.0, 1.0)
        fac_output = tex.outputs["Fac"]
    elif tex_type == "voronoi":
        tex = nodes.new(type="ShaderNodeTexVoronoi")
        tex.inputs["Scale"].default_value = random.uniform(1.0, 30.0)
        tex.inputs["Randomness"].default_value = random.uniform(0.5, 1.0)
        fac_output = tex.outputs["Distance"]
    elif tex_type == "checker":
        tex = nodes.new(type="ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = random.uniform(2.0, 40.0)
        tex.inputs["Color1"].default_value = (
            random.uniform(0.05, 0.7), random.uniform(0.05, 0.7), random.uniform(0.05, 0.7), 1.0
        )
        tex.inputs["Color2"].default_value = (
            random.uniform(0.05, 0.7), random.uniform(0.05, 0.7), random.uniform(0.05, 0.7), 1.0
        )
        fac_output = tex.outputs["Color"]
    else:  # wave
        tex = nodes.new(type="ShaderNodeTexWave")
        tex.inputs["Scale"].default_value = random.uniform(1.0, 20.0)
        tex.inputs["Distortion"].default_value = random.uniform(0.0, 10.0)
        fac_output = tex.outputs["Fac"]

    links.new(mapping.outputs["Vector"], tex.inputs["Vector"])

    # Color ramp with two random colors (clamped to avoid blown-out whites)
    ramp = nodes.new(type="ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (
        random.uniform(0.05, 0.8), random.uniform(0.05, 0.8), random.uniform(0.05, 0.8), 1.0
    )
    ramp.color_ramp.elements[1].color = (
        random.uniform(0.05, 0.8), random.uniform(0.05, 0.8), random.uniform(0.05, 0.8), 1.0
    )

    if tex_type == "checker":
        # Checker already outputs color — connect directly
        links.new(fac_output, principled.inputs["Base Color"])
    else:
        links.new(fac_output, ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], principled.inputs["Base Color"])

    # Random roughness / metallic
    principled.inputs["Roughness"].default_value = random.uniform(0.3, 1.0)
    principled.inputs["Metallic"].default_value = random.uniform(0.0, 0.3)

    return mat


def create_randomized_room() -> list:
    """Create a simple box room (floor + 4 walls + ceiling) with random procedural materials.

    Room is ~20m x 20m x 6m, centered at origin, extending along -Y (where gates are placed).
    Each surface gets an independent random material.
    """
    room_objects = []
    half_w = 10.0  # half width (X)
    half_d = 15.0  # half depth (Y) — gates at -2 to -12
    room_h = 6.0   # height (Z)

    surfaces = [
        # (name, location, scale, rotation)
        ("floor",   [0, -half_d/2, -3],      [half_w, half_d, 0.01],  [0, 0, 0]),
        ("ceiling", [0, -half_d/2, room_h-3], [half_w, half_d, 0.01],  [0, 0, 0]),
        ("wall_back",  [0, -half_d, room_h/2-3],  [half_w, 0.01, room_h/2], [0, 0, 0]),
        ("wall_left",  [-half_w, -half_d/2, room_h/2-3], [0.01, half_d, room_h/2], [0, 0, 0]),
        ("wall_right", [half_w, -half_d/2, room_h/2-3],  [0.01, half_d, room_h/2], [0, 0, 0]),
    ]

    for name, loc, scale, rot in surfaces:
        plane = bproc.object.create_primitive("CUBE")
        plane.set_name(f"room_{name}")
        plane.set_location(loc)
        plane.set_scale(scale)
        plane.set_rotation_euler(rot)
        plane.set_cp("category_id", 0)  # background class

        # Apply random procedural material
        mat = _random_procedural_material(f"mat_{name}")
        plane.blender_obj.data.materials.clear()
        plane.blender_obj.data.materials.append(mat)

        room_objects.append(plane)

    return room_objects


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
    np.array(K_MATRIX, dtype=np.float64),
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
        # Re-apply renderer settings lost after clean_up.
        # enable_outputs=False because depth/segmentation can only be
        # registered once per Blender session (already done before the loop).
        configure_renderer(enable_outputs=False)

    # ---- Room with randomized procedural textures --------------------------
    set_background_color(0.01, 0.01, 0.01)  # dark world background (barely visible)
    room_objects = create_randomized_room()

    # ---- Load gates --------------------------------------------------------
    n_gates = random.randint(1, 3)
    gate_objects = []

    for g_idx in range(n_gates):
        loaded = bproc.loader.load_obj(args.gate_obj)
        for obj in loaded:
            obj.set_cp("category_id", 1)
            # BlenderProc segmentation requires pass_index to be set explicitly
            obj.blender_obj.pass_index = g_idx + 1

            # Random position: lateral ±3 m (X), forward 2-12 m (-Y), vertical ±2 m (Z)
            x = random.uniform(-3.0, 3.0)
            y = random.uniform(-12.0, -2.0)  # negative Y = forward from camera
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
        for d_idx in range(n_drones):
            loaded = bproc.loader.load_obj(args.drone_obj)
            for obj in loaded:
                obj.set_cp("category_id", 2)
                # BlenderProc segmentation requires pass_index
                obj.blender_obj.pass_index = 100 + d_idx  # offset to avoid gate IDs
                x = random.uniform(-4.0, 4.0)
                y = random.uniform(-10.0, -1.5)  # forward
                z = random.uniform(-2.5, 2.5)
                obj.set_location([x, y, z])
                obj.set_rotation_euler([
                    random.uniform(-np.pi, np.pi),
                    random.uniform(-np.pi, np.pi),
                    random.uniform(-np.pi, np.pi),
                ])

    # ---- Randomized lighting (most important DR factor per UZH) ------------
    # 1-3 point/area lights with random position, energy, and color temperature
    n_lights = random.randint(1, 3)
    for _ in range(n_lights):
        light = bproc.types.Light()
        light.set_type(random.choice(["POINT", "AREA"]))
        light.set_location([
            random.uniform(-8.0, 8.0),
            random.uniform(-12.0, 0.0),
            random.uniform(1.0, 5.0),
        ])
        light.set_energy(random.uniform(20.0, 200.0))
        # Random warm/cool color temperature
        temp = random.uniform(0.7, 1.0)
        light.set_color([temp, temp * random.uniform(0.8, 1.0), temp * random.uniform(0.6, 1.0)])

    # ---- Camera at origin, looking toward the gates -------------------------
    # Use rotation_from_forward_vec to point camera at the average gate position.
    # Camera placed at origin (or slight random jitter).
    cam_loc = np.array([0.0, 0.0, random.uniform(-0.5, 0.5)])
    # Look toward -Y (where gates are placed)
    forward = np.array([0.0, -1.0, 0.0])
    rotation_matrix = bproc.camera.rotation_from_forward_vec(forward)
    cam_pose = bproc.math.build_transformation_mat(cam_loc, rotation_matrix)
    bproc.camera.add_camera_pose(cam_pose)

    # ---- Render ------------------------------------------------------------
    render_data = bproc.renderer.render()

    # ---- Extract RGB -------------------------------------------------------
    # BlenderProc render() returns colors as (H, W, 3) uint8 RGB, already tonemapped.
    rgb = render_data["colors"][0]  # uint8, shape (H, W, 3), RGB order
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ---- Extract segmentation masks ----------------------------------------
    # enable_segmentation_output(map_by=["category_id","instance"]) produces:
    #   "instance_segmaps"    → per-pixel Blender pass_index (instance ID)
    #   "category_id_segmaps" → per-pixel category_id custom property value
    instance_seg = render_data["instance_segmaps"][0]      # (H, W) int
    category_seg = render_data["category_id_segmaps"][0]   # (H, W) int

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
        "resolution": [HEIGHT, WIDTH],
        "intrinsics": intrinsics,
        "distortion_model": "pinhole",
        "distortion_coeffs": None,
        "gate_dims_m": [1.5, 1.5],
        "n_gates": n_gates,
        "n_drones": n_drones if args.drone_obj else 0,
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
