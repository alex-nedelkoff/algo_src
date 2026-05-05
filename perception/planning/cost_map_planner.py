"""Cost-map A* planner over a 2.5D occupancy grid.

Produces a sequence of world-frame waypoints from a start position to a
goal position, avoiding occupied cells. Output format matches what
``sim/tracks/waypoint.py:waypoints_to_track`` expects, so the planner
plugs straight into the existing G&CNet pipeline.

Pipeline:
    1. A* on 8-connected grid, Euclidean heuristic.
    2. Greedy line-of-sight smoothing — collapse adjacent waypoints
       when the straight line between them is collision-free.
    3. (Optional) Resample to ~4 m spacing using sim.tracks.waypoint
       so the lookahead tokens land where the policy expects them.
"""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from perception.planning.occupancy_grid import OccupancyGrid2D5

# 8-connected grid neighbours (di, dj, step_cost).
_NEIGHBOURS: tuple[tuple[int, int, float], ...] = (
    ( 1,  0, 1.0),
    (-1,  0, 1.0),
    ( 0,  1, 1.0),
    ( 0, -1, 1.0),
    ( 1,  1, math.sqrt(2)),
    ( 1, -1, math.sqrt(2)),
    (-1,  1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
)


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Octile heuristic — admissible on an 8-connected grid."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


def astar_grid(
    grid: NDArray[np.uint8],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> Optional[list[tuple[int, int]]]:
    """A* over a binary occupancy grid. Returns cell path or None.

    Args:
        grid: (n_x, n_y) array. 0 = free, non-zero = blocked.
        start, goal: cell indices (i, j).

    Returns:
        List of (i, j) cells from start to goal inclusive, or None
        if the goal is unreachable / start is blocked.
    """
    n_x, n_y = grid.shape
    if not (0 <= start[0] < n_x and 0 <= start[1] < n_y):
        return None
    if not (0 <= goal[0] < n_x and 0 <= goal[1] < n_y):
        return None
    if grid[goal[0], goal[1]] != 0:
        # Try the closest free cell to the goal — useful when goals (gates)
        # land inside dilated regions.
        closest = _closest_free_cell(grid, goal)
        if closest is None:
            return None
        goal = closest
    if grid[start[0], start[1]] != 0:
        # Same trick for start: pick nearest free cell.
        closest = _closest_free_cell(grid, start)
        if closest is None:
            return None
        start = closest

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        ci, cj = current
        for di, dj, step in _NEIGHBOURS:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < n_x and 0 <= nj < n_y):
                continue
            if grid[ni, nj] != 0:
                continue
            tentative = g_score[current] + step
            neighbour = (ni, nj)
            if tentative < g_score.get(neighbour, math.inf):
                came_from[neighbour] = current
                g_score[neighbour] = tentative
                f = tentative + _heuristic(neighbour, goal)
                heapq.heappush(open_heap, (f, neighbour))
    return None


def _closest_free_cell(grid: NDArray[np.uint8], start: tuple[int, int]) -> Optional[tuple[int, int]]:
    """BFS for the nearest free cell to a (potentially-blocked) start."""
    n_x, n_y = grid.shape
    if grid[start[0], start[1]] == 0:
        return start
    seen = {start}
    frontier = [start]
    while frontier:
        nxt = []
        for ci, cj in frontier:
            for di, dj, _ in _NEIGHBOURS:
                ni, nj = ci + di, cj + dj
                if not (0 <= ni < n_x and 0 <= nj < n_y):
                    continue
                if (ni, nj) in seen:
                    continue
                if grid[ni, nj] == 0:
                    return (ni, nj)
                seen.add((ni, nj))
                nxt.append((ni, nj))
        frontier = nxt
    return None


def line_of_sight_free(
    grid: NDArray[np.uint8],
    a: tuple[int, int],
    b: tuple[int, int],
) -> bool:
    """Bresenham-ish: True if every cell on the line a→b is free."""
    n_x, n_y = grid.shape
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if not (0 <= x0 < n_x and 0 <= y0 < n_y) or grid[x0, y0] != 0:
            return False
        if (x0, y0) == (x1, y1):
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy


def smooth_path_los(grid: NDArray[np.uint8], path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy line-of-sight smoother. Removes redundant intermediate cells.

    Walk the path; from each kept cell, advance the "look-ahead" to the
    farthest cell that is still in line-of-sight, then keep that one
    and continue.
    """
    if len(path) <= 2:
        return list(path)
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        # Find the farthest j > i such that path[i] -> path[j] is clear.
        j = len(path) - 1
        while j > i + 1 and not line_of_sight_free(grid, path[i], path[j]):
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed


def plan(
    occupancy: OccupancyGrid2D5,
    start_xy: NDArray[np.float64],
    goal_xy: NDArray[np.float64],
    *,
    altitude_m: Optional[float] = None,
    smooth: bool = True,
    target_spacing_m: Optional[float] = 4.0,
) -> Optional[NDArray[np.float64]]:
    """Plan a 3-D waypoint path from start to goal over the occupancy grid.

    Args:
        occupancy: dilated grid (call ``.dilate(drone_radius)`` BEFORE planning).
        start_xy, goal_xy: world-frame xy positions.
        altitude_m: z to attach to every output waypoint. Defaults to
            ``occupancy.altitude_m``.
        smooth: whether to apply line-of-sight smoothing.
        target_spacing_m: if not None, resample to this approximate
            arc-length spacing using sim.tracks.waypoint.resample_waypoints.
            None disables resampling (useful for tests).

    Returns:
        (N, 3) array of waypoints, or None if unreachable.
    """
    if altitude_m is None:
        altitude_m = occupancy.altitude_m

    start_ij = occupancy.world_to_grid(np.asarray(start_xy, dtype=np.float64))
    goal_ij = occupancy.world_to_grid(np.asarray(goal_xy, dtype=np.float64))

    cells = astar_grid(occupancy.grid, start_ij, goal_ij)
    if cells is None:
        return None
    if smooth:
        cells = smooth_path_los(occupancy.grid, cells)

    # Convert grid cells back to world xy.
    pts = np.stack([occupancy.grid_to_world(c) for c in cells], axis=0)
    # Pin the very first/last waypoints to the actual start / goal so the
    # endpoints aren't quantised to cell centres.
    pts[0] = np.asarray(start_xy, dtype=np.float64)
    pts[-1] = np.asarray(goal_xy, dtype=np.float64)
    # Lift to 3-D at the configured altitude.
    waypoints = np.zeros((pts.shape[0], 3), dtype=np.float64)
    waypoints[:, 0:2] = pts
    waypoints[:, 2] = altitude_m

    if target_spacing_m is not None and waypoints.shape[0] >= 2:
        # Resample to ~target_spacing_m via the M3 utility (linear here —
        # we're walking grid cells, the input is already piecewise linear).
        from sim.tracks.waypoint import resample_waypoints
        waypoints = resample_waypoints(
            waypoints, target_spacing=target_spacing_m, method="linear",
        )
    return waypoints
