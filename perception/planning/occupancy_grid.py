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
    ) -> "OccupancyGrid2D5":
        """Build the grid by ray-testing each cell at altitude ± thickness/2.

        For each cell centre, we shoot a vertical ray of length
        ``thickness_m`` centred on ``altitude_m`` and mark the cell
        occupied if it hits any of ``body_ids``. ``thickness_m`` should
        be wide enough to catch obstacles whose tops or bottoms straddle
        the flight altitude.

        Args:
            client_id: PyBullet client id with the world already loaded.
            body_ids: PyBullet body ids whose hits should mark occupancy.
            extent: grid coverage in world frame.
            altitude_m: nominal flight altitude (centre of vertical ray).
            thickness_m: height of the slab that gets sampled, centred
                on ``altitude_m``.
        """
        grid = cls(extent, altitude_m)
        body_id_set = set(body_ids)

        # Vertical rays at every cell centre.
        n_x, n_y = extent.n_cells_x, extent.n_cells_y
        z_top = altitude_m + 0.5 * thickness_m
        z_bot = altitude_m - 0.5 * thickness_m

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
