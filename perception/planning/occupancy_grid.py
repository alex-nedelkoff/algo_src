"""2.5D occupancy grid for racing-time path planning.

A horizontal slice of the world at the drone's flight altitude — fast
to build, easy to plan over, sufficient for warehouses where most
obstacles are vertical (walls, columns, shelving). Built once per
race start from the static warehouse mesh; small-radius dynamic
obstacles can be injected post-hoc via ``add_cylinder``.

Key operations:
    - ``OccupancyGrid2D5.from_pybullet_scene(...)`` — sample by
      raycasting through the world at altitude ± a small thickness.
    - ``.dilate(radius_m)`` — Minkowski-sum with a disk so the
      planner can treat the drone as a point.
    - ``.add_cylinder(center_xy, radius_m)`` — paint an obstacle
      directly into the grid.
    - ``.world_to_grid(xy)`` / ``.grid_to_world(ij)`` — coordinate conversions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pybullet as pb
from numpy.typing import NDArray


@dataclass
class GridExtent:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    cell_size_m: float

    @property
    def n_cells_x(self) -> int:
        return int(round((self.x_max - self.x_min) / self.cell_size_m))

    @property
    def n_cells_y(self) -> int:
        return int(round((self.y_max - self.y_min) / self.cell_size_m))


class OccupancyGrid2D5:
    """Binary occupancy grid at a fixed flight altitude.

    Cell value 1 = obstacle, 0 = free. Stored as a numpy ``uint8`` array
    indexed (i, j) where ``i`` is the x-cell index and ``j`` is the y-cell.
    The world-to-grid mapping is::

        i = floor((x - x_min) / cell_size)
        j = floor((y - y_min) / cell_size)

    so cell (0, 0) sits at the (-x, -y) corner.
    """

    def __init__(self, extent: GridExtent, altitude_m: float, occupancy: NDArray[np.uint8] | None = None) -> None:
        self.extent = extent
        self.altitude_m = altitude_m
        if occupancy is None:
            occupancy = np.zeros((extent.n_cells_x, extent.n_cells_y), dtype=np.uint8)
        if occupancy.shape != (extent.n_cells_x, extent.n_cells_y):
            raise ValueError(
                f"occupancy shape {occupancy.shape} doesn't match extent "
                f"({extent.n_cells_x}, {extent.n_cells_y})"
            )
        self.grid = occupancy

    # ----- coords -----

    def world_to_grid(self, xy: NDArray[np.float64]) -> tuple[int, int]:
        i = int(math.floor((xy[0] - self.extent.x_min) / self.extent.cell_size_m))
        j = int(math.floor((xy[1] - self.extent.y_min) / self.extent.cell_size_m))
        return i, j

    def grid_to_world(self, ij: tuple[int, int]) -> NDArray[np.float64]:
        i, j = ij
        x = self.extent.x_min + (i + 0.5) * self.extent.cell_size_m
        y = self.extent.y_min + (j + 0.5) * self.extent.cell_size_m
        return np.array([x, y], dtype=np.float64)

    def in_bounds(self, ij: tuple[int, int]) -> bool:
        i, j = ij
        return 0 <= i < self.extent.n_cells_x and 0 <= j < self.extent.n_cells_y

    def is_free_world(self, xy: NDArray[np.float64]) -> bool:
        ij = self.world_to_grid(xy)
        if not self.in_bounds(ij):
            return False
        return bool(self.grid[ij[0], ij[1]] == 0)

    # ----- mutation -----

    def add_cylinder(self, center_xy: NDArray[np.float64], radius_m: float) -> None:
        """Paint a circular obstacle into the grid."""
        cx, cy = float(center_xy[0]), float(center_xy[1])
        r_cells = radius_m / self.extent.cell_size_m
        # Bounding-box of the disk in grid coords.
        i_min = max(0, int(math.floor((cx - radius_m - self.extent.x_min) / self.extent.cell_size_m)))
        i_max = min(self.extent.n_cells_x - 1,
                    int(math.ceil((cx + radius_m - self.extent.x_min) / self.extent.cell_size_m)))
        j_min = max(0, int(math.floor((cy - radius_m - self.extent.y_min) / self.extent.cell_size_m)))
        j_max = min(self.extent.n_cells_y - 1,
                    int(math.ceil((cy + radius_m - self.extent.y_min) / self.extent.cell_size_m)))
        # Cell centres for distance check.
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                cell_xy = self.grid_to_world((i, j))
                if (cell_xy[0] - cx) ** 2 + (cell_xy[1] - cy) ** 2 <= radius_m ** 2:
                    self.grid[i, j] = 1
        _ = r_cells  # informational; not needed beyond bbox

    def dilate(self, radius_m: float) -> "OccupancyGrid2D5":
        """Minkowski-sum the obstacle set with a disk of ``radius_m``.

        Returns a NEW grid; the original is unchanged. Use this so the
        planner can treat the drone as a point — every cell within
        ``radius_m`` of any obstacle becomes occupied.
        """
        radius_cells = int(math.ceil(radius_m / self.extent.cell_size_m))
        if radius_cells <= 0:
            return OccupancyGrid2D5(self.extent, self.altitude_m, self.grid.copy())

        # Build a circular structuring element.
        size = 2 * radius_cells + 1
        yy, xx = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
        kernel = (xx ** 2 + yy ** 2) <= radius_cells ** 2
        kernel = kernel.astype(np.uint8)

        # Convolution-style dilation. For grids small enough that this is
        # fine on its own (typical warehouse: ~200x200 cells), we just
        # iterate. For bigger grids we'd switch to scipy.ndimage.binary_dilation.
        n_x, n_y = self.grid.shape
        dilated = np.zeros_like(self.grid)
        obstacle_indices = np.argwhere(self.grid != 0)
        for i, j in obstacle_indices:
            i_lo = max(0, i - radius_cells); i_hi = min(n_x, i + radius_cells + 1)
            j_lo = max(0, j - radius_cells); j_hi = min(n_y, j + radius_cells + 1)
            ki_lo = i_lo - (i - radius_cells); ki_hi = ki_lo + (i_hi - i_lo)
            kj_lo = j_lo - (j - radius_cells); kj_hi = kj_lo + (j_hi - j_lo)
            dilated[i_lo:i_hi, j_lo:j_hi] |= kernel[ki_lo:ki_hi, kj_lo:kj_hi]
        return OccupancyGrid2D5(self.extent, self.altitude_m, dilated)

    # ----- construction from a scene -----

    @classmethod
    def from_pybullet_scene(
        cls,
        client_id: int,
        body_ids: Iterable[int],
        extent: GridExtent,
        altitude_m: float,
        thickness_m: float = 1.0,
        *,
        z_top: float | None = None,
        z_bot: float | None = None,
    ) -> "OccupancyGrid2D5":
        """Build the grid by ray-testing each cell vertically.

        By default, rays span ``altitude_m ± thickness_m / 2`` (a
        symmetric slab around the flight altitude). Override with
        ``z_top`` and ``z_bot`` for an asymmetric slab — useful when
        you need the ray to start above every tall obstacle but stop
        before hitting the floor.

        Args:
            client_id: PyBullet client id with the world already loaded.
            body_ids: PyBullet body ids whose hits mark occupancy.
            extent: grid coverage in world frame.
            altitude_m: nominal flight altitude (stored on the grid).
            thickness_m: symmetric-slab height around ``altitude_m``.
                Ignored when both ``z_top`` and ``z_bot`` are provided.
            z_top: optional explicit ray START z (above obstacles).
            z_bot: optional explicit ray END z (below altitude but
                ideally above the floor, so floor-only cells stay free).
        """
        grid = cls(extent, altitude_m)
        body_id_set = set(body_ids)

        # Vertical rays at every cell centre.
        n_x, n_y = extent.n_cells_x, extent.n_cells_y
        if z_top is None:
            z_top = altitude_m + 0.5 * thickness_m
        if z_bot is None:
            z_bot = altitude_m - 0.5 * thickness_m
        if z_top <= z_bot:
            raise ValueError(f"z_top ({z_top}) must be > z_bot ({z_bot})")

        # Build batched ray endpoints.
        starts = []
        ends = []
        for i in range(n_x):
            for j in range(n_y):
                cell_xy = grid.grid_to_world((i, j))
                starts.append([float(cell_xy[0]), float(cell_xy[1]), z_top])
                ends.append([float(cell_xy[0]), float(cell_xy[1]), z_bot])

        # rayTestBatch chunked to avoid PyBullet's per-call limit.
        CHUNK = 4096
        results = []
        for off in range(0, len(starts), CHUNK):
            s = starts[off:off + CHUNK]
            e = ends[off:off + CHUNK]
            results.extend(pb.rayTestBatch(s, e, physicsClientId=client_id))

        # Mark cells whose ray hit one of the target bodies.
        idx = 0
        for i in range(n_x):
            for j in range(n_y):
                hit_body = results[idx][0]
                if hit_body in body_id_set:
                    grid.grid[i, j] = 1
                idx += 1
        return grid

    # ----- alternative construction: probe-sphere closest-points -----

    @classmethod
    def from_pybullet_probe_sphere(
        cls,
        client_id: int,
        body_ids: Iterable[int],
        extent: GridExtent,
        altitude_m: float,
        probe_radius_m: float | None = None,
    ) -> "OccupancyGrid2D5":
        """Build the grid by closest-points queries against a probe sphere.

        For each cell, place a small sphere at the cell centre + altitude
        and ask PyBullet's narrowphase for the closest point between
        the sphere and each target body. Distance ≤ 0 → the cell is
        occupied.

        This is the most reliable method because:
        - Uses exact mesh geometry (not broadphase AABBs).
        - Sees obstacles regardless of vertical extent (no
          ``rayTest`` "both endpoints inside object" blind spot).
        - Doesn't false-positive on floor or ceiling cells when those
          are far from the drone's altitude.

        Cost: O(cells × target_bodies) closest-points calls. Each
        narrowphase contact query is fast — for typical warehouse
        sizes (5–10 k cells × few bodies) it completes in seconds.

        Args:
            client_id: PyBullet client id.
            body_ids: bodies whose contact with the probe marks occupancy.
            extent: grid coverage.
            altitude_m: drone flight altitude.
            probe_radius_m: probe sphere radius. Defaults to
                ``cell_size/2`` so each cell is "covered" by the probe.
        """
        grid = cls(extent, altitude_m)
        body_id_set = set(body_ids)
        cs = extent.cell_size_m
        if probe_radius_m is None:
            probe_radius_m = cs / 2.0

        # Create the probe — a static sphere body we'll move to each cell.
        probe_col = pb.createCollisionShape(
            pb.GEOM_SPHERE, radius=probe_radius_m, physicsClientId=client_id,
        )
        probe_body = pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=probe_col,
            basePosition=[0.0, 0.0, altitude_m],
            physicsClientId=client_id,
        )
        try:
            for i in range(extent.n_cells_x):
                for j in range(extent.n_cells_y):
                    cell_xy = grid.grid_to_world((i, j))
                    pb.resetBasePositionAndOrientation(
                        probe_body,
                        [float(cell_xy[0]), float(cell_xy[1]), altitude_m],
                        [0.0, 0.0, 0.0, 1.0],
                        physicsClientId=client_id,
                    )
                    occupied = False
                    for target_id in body_id_set:
                        contacts = pb.getClosestPoints(
                            probe_body, target_id, distance=0.0,
                            physicsClientId=client_id,
                        )
                        if contacts:   # non-empty → contact at distance ≤ 0
                            occupied = True
                            break
                    if occupied:
                        grid.grid[i, j] = 1
        finally:
            pb.removeBody(probe_body, physicsClientId=client_id)
        return grid

    # ----- alternative construction: per-cell AABB overlap query -----

    @classmethod
    def from_pybullet_aabb_overlap(
        cls,
        client_id: int,
        body_ids: Iterable[int],
        extent: GridExtent,
        altitude_m: float,
        slab_half_height_m: float = 0.5,
    ) -> "OccupancyGrid2D5":
        """Build the grid by AABB-overlap queries — robust to mesh quirks.

        For each cell, query PyBullet for any body whose AABB overlaps a
        small box at the drone's altitude, of size ``cell × cell ×
        2·slab_half_height``. If any of ``body_ids`` is in the result,
        the cell is occupied.

        More reliable than raycasting:
        - No "both endpoints inside object" rayTest blind spot.
        - Doesn't false-positive on floor / ceiling cells where the ray
          had no choice but to hit something.
        - Marks exactly the cells where an obstacle sits at the drone's
          flight altitude — what the planner actually needs.

        Trade-off: AABB overlap is conservative (uses broadphase AABBs,
        not exact mesh geometry), so concave meshes can flag cells the
        drone could squeeze through. Acceptable for v1 — drone-radius
        dilation absorbs most of the slack.
        """
        grid = cls(extent, altitude_m)
        body_id_set = set(body_ids)
        cs = extent.cell_size_m
        z_lo = altitude_m - slab_half_height_m
        z_hi = altitude_m + slab_half_height_m

        for i in range(extent.n_cells_x):
            for j in range(extent.n_cells_y):
                cell_xy = grid.grid_to_world((i, j))
                aabb_min = [
                    float(cell_xy[0] - cs / 2),
                    float(cell_xy[1] - cs / 2),
                    z_lo,
                ]
                aabb_max = [
                    float(cell_xy[0] + cs / 2),
                    float(cell_xy[1] + cs / 2),
                    z_hi,
                ]
                overlapping = pb.getOverlappingObjects(
                    aabb_min, aabb_max, physicsClientId=client_id,
                )
                if overlapping is None:
                    continue
                for body_unique_id, _link_index in overlapping:
                    if body_unique_id in body_id_set:
                        grid.grid[i, j] = 1
                        break
        return grid

    # ----- visualisation -----

    def to_image(self) -> NDArray[np.uint8]:
        """Return an RGB image (H, W, 3) suitable for rerun's Image archetype.

        The grid is ``(n_cells_x, n_cells_y)`` with ``i = x``, ``j = y``;
        for a top-down image we want ``y`` rows × ``x`` cols, with ``y``
        increasing UP (so flip the j axis).
        """
        # (n_x, n_y) → (n_y, n_x), flip y so image y matches world y.
        img_grid = np.flipud(self.grid.T.astype(np.float32))
        rgb = np.zeros((img_grid.shape[0], img_grid.shape[1], 3), dtype=np.uint8)
        # Free cells: dark grey. Occupied: bright white-ish.
        rgb[img_grid == 0] = (40, 50, 70)
        rgb[img_grid != 0] = (255, 200, 130)
        return rgb
