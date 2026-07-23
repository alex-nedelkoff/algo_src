"""Post-processing pipeline for GateNet v2 outputs.

Converts raw segmentation logits + corner heatmaps into structured
PGO-compatible observations:
  1. Opening CC -> instance grouping
  2. Heatmap peak detection + NMS -> corner locations with identity
  3. Corner-to-instance assignment via nearest opening centroid
  4. Depth-from-mask for sigma estimation
  5. Emit PGO observation format
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import label as connected_components
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist

from .. import camera


def parabolic_refine(heatmap: np.ndarray, x: int, y: int) -> tuple[float, float]:
    """Sub-pixel refinement via parabolic fitting around a peak.

    Fits a 2D parabola to the 3x3 neighborhood of (x, y) and returns
    the sub-pixel offset.

    Args:
        heatmap: (H, W) float array.
        x: Peak column index.
        y: Peak row index.

    Returns:
        (sub_x, sub_y) refined coordinates.
    """
    H, W = heatmap.shape
    if x <= 0 or x >= W - 1 or y <= 0 or y >= H - 1:
        return float(x), float(y)

    # 1D parabolic fit along x
    dx = 0.5 * (heatmap[y, x + 1] - heatmap[y, x - 1])
    ddx = heatmap[y, x + 1] - 2 * heatmap[y, x] + heatmap[y, x - 1]
    if abs(ddx) > 1e-6:
        offset_x = -dx / ddx
    else:
        offset_x = 0.0

    # 1D parabolic fit along y
    dy = 0.5 * (heatmap[y + 1, x] - heatmap[y - 1, x])
    ddy = heatmap[y + 1, x] - 2 * heatmap[y, x] + heatmap[y - 1, x]
    if abs(ddy) > 1e-6:
        offset_y = -dy / ddy
    else:
        offset_y = 0.0

    # Clamp offset to [-0.5, 0.5] (within the cell)
    offset_x = max(-0.5, min(0.5, offset_x))
    offset_y = max(-0.5, min(0.5, offset_y))

    return x + offset_x, y + offset_y


def gaussian_refine(heatmap: np.ndarray, x: int, y: int) -> tuple[float, float]:
    """Sub-pixel refinement via 2D Gaussian least-squares on a 5x5 neighborhood.

    Fits log(z) = ax^2 + by^2 + cxy + dx + ey + f to a 5x5 patch around (x, y).
    Sub-pixel center is derived from the quadratic coefficients.

    Falls back to parabolic_refine if the peak is too close to the border
    or the fit is degenerate.

    Args:
        heatmap: (H, W) float array.
        x: Peak column index.
        y: Peak row index.

    Returns:
        (sub_x, sub_y) refined coordinates.
    """
    H, W = heatmap.shape
    radius = 2  # 5x5 patch

    # Fall back to parabolic if too close to border
    if x < radius or x >= W - radius or y < radius or y >= H - radius:
        return parabolic_refine(heatmap, x, y)

    # Extract 5x5 patch
    patch = heatmap[y - radius:y + radius + 1, x - radius:x + radius + 1].copy()

    # Clamp to epsilon before log (handle zero/negative values)
    eps = 1e-7
    patch = np.maximum(patch, eps)
    log_patch = np.log(patch)

    # Build design matrix for: log(z) = a*dx^2 + b*dy^2 + c*dx*dy + d*dx + e*dy + f
    # where dx, dy are offsets from patch center (-2..+2)
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    dy_grid, dx_grid = np.meshgrid(coords, coords, indexing="ij")
    dx_flat = dx_grid.ravel()
    dy_flat = dy_grid.ravel()
    z_flat = log_patch.ravel()

    A_mat = np.column_stack([
        dx_flat**2,        # a
        dy_flat**2,        # b
        dx_flat * dy_flat, # c
        dx_flat,           # d
        dy_flat,           # e
        np.ones_like(dx_flat),  # f
    ])

    # Solve least-squares
    result = np.linalg.lstsq(A_mat, z_flat, rcond=None)
    coeffs = result[0]
    a, b, c, d, e, _f = coeffs

    # For a 2D quadratic ax^2 + by^2 + cxy + dx + ey + f,
    # the extremum is at: [x, y] = -0.5 * H^-1 * g
    # where H = [[2a, c], [c, 2b]] and g = [d, e]
    det = 4 * a * b - c * c
    if abs(det) < 1e-10 or a >= 0 or b >= 0:
        # Degenerate or not a maximum -- fall back
        return parabolic_refine(heatmap, x, y)

    offset_x = -(2 * b * d - c * e) / det
    offset_y = -(2 * a * e - c * d) / det

    # Clamp offsets to within the patch
    offset_x = max(-radius + 0.5, min(radius - 0.5, offset_x))
    offset_y = max(-radius + 0.5, min(radius - 0.5, offset_y))

    return x + offset_x, y + offset_y


def extract_gate_observations(
    seg_map: np.ndarray,
    corner_heatmaps: np.ndarray,
    stride: int = 4,
    min_opening_area: int = 20,
    confidence_threshold: float = 0.3,
    nms_size: int = 3,
    gate_physical_size: float = 1.5,
    fx: float = camera.FX,  # single-sourced GateNet PnP focal (vq2/camera.py)
    sigma_base: float = 2.0,
    refine_method: str = "gaussian",
) -> dict:
    """Full post-processing pipeline: seg + corners -> PGO observations.

    Args:
        seg_map: (H, W) int array with class labels (0=bg, 1=corner, 2=edge, 3=opening).
        corner_heatmaps: (4, H, W) float array with sigmoid heatmaps.
        stride: Heatmap-to-image stride factor.
        min_opening_area: Minimum opening blob area (in heatmap pixels) to count as a gate.
        confidence_threshold: Minimum peak value for corner detection.
        nms_size: NMS kernel size for peak detection.
        gate_physical_size: Physical gate size in meters (for depth-from-mask).
        fx: Camera focal length in pixels.
        sigma_base: Base sigma for PGO noise model.
        refine_method: Sub-pixel refinement method ("gaussian" or "parabolic").

    Returns:
        Dict with:
            observations: list of {gate_id, corner_idx, pixel_uv, sigma_uv}
            gate_depths: dict of {gate_id: depth_estimate}
            seg_mask: the input seg_map (for visualization)
    """
    # Step 1: Instance grouping via opening CC
    opening_mask = (seg_map == 3).astype(np.int32)
    cc_labels, n_instances = connected_components(opening_mask)

    instances = []
    for label_id in range(1, n_instances + 1):
        ys, xs = np.where(cc_labels == label_id)
        if len(ys) < min_opening_area:
            continue
        centroid = np.array([xs.mean(), ys.mean()])
        area = len(ys)
        instances.append({
            "gate_id": len(instances) + 1,
            "centroid": centroid,
            "area": area,
        })

    if not instances:
        return {"observations": [], "gate_depths": {}, "seg_mask": seg_map}

    # Step 2: Corner peak detection
    refine_fn = gaussian_refine if refine_method == "gaussian" else parabolic_refine
    corners = []
    for ch in range(4):
        heatmap = corner_heatmaps[ch]
        local_max = maximum_filter(heatmap, size=nms_size)
        peaks = (heatmap == local_max) & (heatmap > confidence_threshold)
        peak_ys, peak_xs = np.where(peaks)

        for py, px in zip(peak_ys, peak_xs):
            sub_x, sub_y = refine_fn(heatmap, int(px), int(py))
            pixel_uv = np.array([(sub_x + 0.5) * stride, (sub_y + 0.5) * stride])
            corners.append({
                "corner_idx": ch,
                "pixel_uv": pixel_uv,
                "confidence": float(heatmap[py, px]),
                "heatmap_pos": np.array([sub_x, sub_y]),
            })

    # Step 3: Assign corners to nearest gate instance
    if corners and instances:
        corner_positions = np.array([c["heatmap_pos"] for c in corners])
        centroids = np.array([inst["centroid"] for inst in instances])
        dists = cdist(corner_positions, centroids)
        nearest = dists.argmin(axis=1)

        for i, corner in enumerate(corners):
            corner["gate_id"] = instances[nearest[i]]["gate_id"]
    elif corners:
        # No instances but corners detected -- assign all to gate 1
        for corner in corners:
            corner["gate_id"] = 1

    # Step 4: Depth from mask
    gate_depths = {}
    for inst in instances:
        mask_area_px = inst["area"] * (stride ** 2)  # convert to original image pixels
        if mask_area_px > 0:
            depth = gate_physical_size * fx / math.sqrt(mask_area_px)
        else:
            depth = float("inf")
        gate_depths[inst["gate_id"]] = depth

    # Step 5: Build PGO observations
    observations = []
    for corner in corners:
        gate_id = corner["gate_id"]
        depth = gate_depths.get(gate_id, 10.0)
        sigma = sigma_base * depth / max(corner["confidence"], 0.1)

        observations.append({
            "gate_id": gate_id,
            "corner_idx": corner["corner_idx"],
            "pixel_uv": corner["pixel_uv"],
            "sigma_uv": sigma,
        })

    return {
        "observations": observations,
        "gate_depths": gate_depths,
        "seg_mask": seg_map,
    }
