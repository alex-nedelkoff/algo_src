"""Procedurally generate a textured rectangular room for SLAM smoke tests.

Output: room.obj + room.mtl + room.urdf in this directory. The OBJ is six
quads — floor, ceiling, four walls — each with explicit planar UVs and a
distinct material referencing one of the CC0 textures in ./textures/.

Why explicit UVs: PyBullet's GEOM_BOX primitive doesn't carry texture
coords, so a SLAM/CV pipeline rendering a primitive box gets flat-shaded
pixels regardless of how good the texture file is. Building the room as
a real OBJ with hand-authored UVs gives MASt3R-SLAM the photoreal RGB it
was trained on.

Coordinate convention:
    - z up, x forward, y left (matches the rest of the project)
    - Origin at room center, floor at z = 0, ceiling at z = ROOM_H
    - Inward-facing normals (cameras inside the room see the textures)

Run once::

    python sim/assets/textured_room_v1/build_room.py
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

ROOM_W = 14.0   # x extent (full)
ROOM_D = 10.0   # y extent (full)
ROOM_H = 3.0    # z extent (full)
TILE_M = 2.0    # texture repeat: 1 tile per 2 metres (4K pixels @ 1K image)

HALF_W = ROOM_W / 2.0
HALF_D = ROOM_D / 2.0

ASSETS_DIR = Path(__file__).parent
TEX_DIR = ASSETS_DIR / "textures"


def build_obj() -> str:
    """Six quads, inward-facing. Each quad has its own 4 vertices, UVs,
    normal — keeps the OBJ trivially readable and the UVs per-face.
    """
    lines: list[str] = []
    lines.append("# Procedurally generated textured room")
    lines.append(f"# {ROOM_W}m x {ROOM_D}m x {ROOM_H}m, tiles every {TILE_M}m")
    lines.append("mtllib room.mtl")

    # Per-face data: name, material, 4 verts (CCW from inside), 4 uvs, normal.
    # UV ranges expand the texture to tile every TILE_M metres.
    u_x = ROOM_W / TILE_M  # 7
    u_y = ROOM_D / TILE_M  # 5
    u_z = ROOM_H / TILE_M  # 1.5

    faces = [
        # Floor: looking up from inside, normal +z, CCW = (-,-) → (+,-) → (+,+) → (-,+)
        (
            "floor", "wood_floor",
            [(-HALF_W, -HALF_D, 0.0), (HALF_W, -HALF_D, 0.0),
             (HALF_W,  HALF_D, 0.0), (-HALF_W,  HALF_D, 0.0)],
            [(0, 0), (u_x, 0), (u_x, u_y), (0, u_y)],
            (0, 0, 1),
        ),
        # Ceiling: looking down from inside, normal -z, CCW seen from below
        (
            "ceiling", "concrete_wall",
            [(-HALF_W, -HALF_D, ROOM_H), (-HALF_W,  HALF_D, ROOM_H),
             (HALF_W,  HALF_D, ROOM_H), (HALF_W, -HALF_D, ROOM_H)],
            [(0, 0), (0, u_y), (u_x, u_y), (u_x, 0)],
            (0, 0, -1),
        ),
        # Wall y=-HALF_D, normal +y (inward), CCW from inside (looking +y)
        (
            "wall_yneg", "brick_wall",
            [(-HALF_W, -HALF_D, 0.0), (-HALF_W, -HALF_D, ROOM_H),
             (HALF_W, -HALF_D, ROOM_H), (HALF_W, -HALF_D, 0.0)],
            [(0, 0), (0, u_z), (u_x, u_z), (u_x, 0)],
            (0, 1, 0),
        ),
        # Wall y=+HALF_D, normal -y
        (
            "wall_ypos", "brick_wall",
            [(HALF_W,  HALF_D, 0.0), (HALF_W,  HALF_D, ROOM_H),
             (-HALF_W,  HALF_D, ROOM_H), (-HALF_W,  HALF_D, 0.0)],
            [(0, 0), (0, u_z), (u_x, u_z), (u_x, 0)],
            (0, -1, 0),
        ),
        # Wall x=-HALF_W, normal +x
        (
            "wall_xneg", "metal_plate",
            [(-HALF_W,  HALF_D, 0.0), (-HALF_W,  HALF_D, ROOM_H),
             (-HALF_W, -HALF_D, ROOM_H), (-HALF_W, -HALF_D, 0.0)],
            [(0, 0), (0, u_z), (u_y, u_z), (u_y, 0)],
            (1, 0, 0),
        ),
        # Wall x=+HALF_W, normal -x
        (
            "wall_xpos", "metal_plate",
            [(HALF_W, -HALF_D, 0.0), (HALF_W, -HALF_D, ROOM_H),
             (HALF_W,  HALF_D, ROOM_H), (HALF_W,  HALF_D, 0.0)],
            [(0, 0), (0, u_z), (u_y, u_z), (u_y, 0)],
            (-1, 0, 0),
        ),
    ]

    v_idx = 1   # OBJ indices are 1-based and global
    vt_idx = 1
    vn_idx = 1
    for name, mat, verts, uvs, normal in faces:
        lines.append(f"\n# --- {name} ---")
        for x, y, z in verts:
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        for u, vv in uvs:
            lines.append(f"vt {u:.4f} {vv:.4f}")
        nx, ny, nz = normal
        lines.append(f"vn {nx} {ny} {nz}")
        lines.append(f"usemtl {mat}")
        # Face: 4 verts, all sharing the single normal we just emitted.
        face = "f " + " ".join(
            f"{v_idx + k}/{vt_idx + k}/{vn_idx}" for k in range(4)
        )
        lines.append(face)
        v_idx += 4
        vt_idx += 4
        vn_idx += 1

    return "\n".join(lines) + "\n"


def build_mtl() -> str:
    """Four materials, each pointing at a JPG in textures/.

    Ambient + diffuse + specular set so PyBullet's OpenGL renderer
    actually shows the texture; some renderers ignore the diffuse map
    when ambient is zero.
    """
    materials = [
        ("wood_floor",     "wood_floor.jpg"),
        ("concrete_wall",  "concrete_wall_006.jpg"),
        ("brick_wall",     "brick_wall_006.jpg"),
        ("metal_plate",    "metal_plate.jpg"),
    ]
    out = ["# Procedurally generated MTL"]
    for mat, tex in materials:
        out.append(dedent(f"""
            newmtl {mat}
            Ka 1.0 1.0 1.0
            Kd 1.0 1.0 1.0
            Ks 0.1 0.1 0.1
            Ns 10.0
            illum 2
            map_Kd textures/{tex}
        """).strip())
    return "\n\n".join(out) + "\n"


def build_urdf() -> str:
    """URDF wrapping room.obj as a fixed-base visual + collision link."""
    return dedent(f"""\
        <?xml version="1.0"?>
        <robot name="textured_room">
          <link name="room">
            <visual>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry>
                <mesh filename="room.obj" scale="1 1 1"/>
              </geometry>
            </visual>
            <collision>
              <origin xyz="0 0 0" rpy="0 0 0"/>
              <geometry>
                <mesh filename="room.obj" scale="1 1 1"/>
              </geometry>
            </collision>
            <inertial>
              <mass value="0"/>
              <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
            </inertial>
          </link>
        </robot>
    """)


def main() -> None:
    if not TEX_DIR.exists():
        raise FileNotFoundError(f"Expected textures in {TEX_DIR}")
    expected = ["wood_floor.jpg", "concrete_wall_006.jpg",
                "brick_wall_006.jpg", "metal_plate.jpg"]
    missing = [t for t in expected if not (TEX_DIR / t).exists()]
    if missing:
        raise FileNotFoundError(f"Missing textures: {missing}")

    (ASSETS_DIR / "room.obj").write_text(build_obj())
    (ASSETS_DIR / "room.mtl").write_text(build_mtl())
    (ASSETS_DIR / "room.urdf").write_text(build_urdf())
    print(f"Wrote room.obj / room.mtl / room.urdf in {ASSETS_DIR}")
    print(f"  Room: {ROOM_W}m x {ROOM_D}m x {ROOM_H}m, "
          f"texture tile {TILE_M}m")


if __name__ == "__main__":
    main()
