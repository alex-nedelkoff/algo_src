"""Calibration evidence for a manually annotated stationary G0 aperture.

This is intentionally a small offline tool.  It does not alter ``vq2.camera``
and it never marks a model approved: approval is a human decision after the
reported residuals and depth discrepancy have been reviewed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


APERTURE_M = 1.5
_OBJECT_2D = np.array([[-.75, -.75], [.75, -.75], [.75, .75], [-.75, .75]], float)
_OBJECT_3D = np.column_stack((_OBJECT_2D, np.zeros(4)))


@dataclass(frozen=True)
class CalibrationFit:
    K: np.ndarray
    t_cam_gate: np.ndarray
    reproj_rms_px: float
    orthogonality_residual: float
    equal_scale_residual: float


def _homography(image_xy: np.ndarray) -> np.ndarray:
    H, mask = cv2.findHomography(_OBJECT_2D, image_xy, method=0)
    if H is None or mask is None:
        raise ValueError("could not estimate aperture homography")
    return H


def _geometry_residual(log_focal: np.ndarray, H: np.ndarray, cx: float, cy: float) -> np.ndarray:
    fx, fy = np.exp(log_focal)
    Kinv = np.linalg.inv(np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]))
    q1, q2 = Kinv @ H[:, 0], Kinv @ H[:, 1]
    denom = np.linalg.norm(q1) * np.linalg.norm(q2)
    return np.array([np.dot(q1, q2) / denom, np.log(np.linalg.norm(q1) / np.linalg.norm(q2))])


def fit_g0_aperture(image_xy, *, cx: float = 319.5, cy: float = 179.5,
                    initial_focal_px: float = 226.0,
                    min_focal_px: float = 50.0,
                    max_focal_px: float = 2_000.0) -> CalibrationFit:
    """Fit ``fx, fy`` from four aperture corners in TL, TR, BR, BL order.

    A single planar square only constrains focal lengths after the principal
    point is declared.  The caller must therefore record why ``cx, cy`` are
    held fixed; this function refuses to silently "fit" all four intrinsics.
    """
    image = np.asarray(image_xy, float)
    if image.shape != (4, 2) or not np.all(np.isfinite(image)):
        raise ValueError("image_xy must be four finite corners in TL,TR,BR,BL order")
    H = _homography(image)
    H = H / H[2, 2]
    if np.linalg.norm(H[2, :2]) < 1e-6:
        raise ValueError(
            "G0 aperture view is ill-conditioned for free fx/fy; use it to "
            "compare declared models or add an oblique known-aperture view"
        )
    if not 0.0 < min_focal_px < max_focal_px:
        raise ValueError("focal bounds must be positive and ordered")
    result = least_squares(
        _geometry_residual, np.log([initial_focal_px, initial_focal_px]),
        args=(H, cx, cy), bounds=(np.log([min_focal_px, min_focal_px]),
                                  np.log([max_focal_px, max_focal_px])),
        xtol=1e-13, ftol=1e-13, gtol=1e-13,
    )
    if not result.success:
        raise ValueError(f"focal fit failed: {result.message}")
    fx, fy = np.exp(result.x)
    # A nearly fronto-parallel square has insufficient perspective to identify
    # free fx/fy.  A bound-hitting answer is evidence of that degeneracy, not
    # an unusually wide camera.
    if min(fx, fy) <= min_focal_px * 1.01 or max(fx, fy) >= max_focal_px / 1.01:
        raise ValueError(
            "G0 aperture view is ill-conditioned for free fx/fy; use it to "
            "compare declared models or add an oblique known-aperture view"
        )
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    ok, rvec, tvec = cv2.solvePnP(_OBJECT_3D, image, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or tvec[2, 0] <= 0:
        raise ValueError("PnP did not produce a positive-depth G0 pose")
    projected, _ = cv2.projectPoints(_OBJECT_3D, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image) ** 2, axis=1))))
    geometry = _geometry_residual(result.x, H, cx, cy)
    return CalibrationFit(K, tvec.reshape(3), rms, float(geometry[0]), float(geometry[1]))


def compare_hypotheses(image_xy, *, cx: float = 319.5, cy: float = 179.5,
                       certified_depth_m: float | None = None) -> dict:
    """Return fitted and historic-model evidence without approving either."""
    image = np.asarray(image_xy, float)
    H = _homography(image)
    try:
        fitted = fit_g0_aperture(image, cx=cx, cy=cy)
        fitted_row = None
    except ValueError as error:
        fitted = None
        fitted_row = {"id": "fitted", "available": False, "reason": str(error)}
    rows = []
    for name, focal in (("fitted", None), ("mapping-226", 226.0), ("legacy-320", 320.0)):
        if focal is None and fitted is not None:
            fit = fitted
        elif focal is None:
            rows.append(fitted_row)
            continue
        else:
            K = np.array([[focal, 0., cx], [0., focal, cy], [0., 0., 1.]])
            ok, rvec, tvec = cv2.solvePnP(_OBJECT_3D, image, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise ValueError(f"PnP failed for {name}")
            projected, _ = cv2.projectPoints(_OBJECT_3D, rvec, tvec, K, None)
            log_f = np.log([focal, focal])
            g = _geometry_residual(log_f, H, cx, cy)
            fit = CalibrationFit(K, tvec.reshape(3),
                                 float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image) ** 2, axis=1)))),
                                 float(g[0]), float(g[1]))
        rows.append({
            "id": name, "fx": float(fit.K[0, 0]), "fy": float(fit.K[1, 1]),
            "cx": float(cx), "cy": float(cy), "t_cam_gate_m": fit.t_cam_gate.tolist(),
            "reproj_rms_px": fit.reproj_rms_px,
            "orthogonality_residual": fit.orthogonality_residual,
            "equal_scale_residual": fit.equal_scale_residual,
            "depth_delta_m": None if certified_depth_m is None else float(fit.t_cam_gate[2] - certified_depth_m),
        })
    return {"aperture_m": APERTURE_M, "principal_point_fixed": [cx, cy],
            "certified_depth_m": certified_depth_m, "models": rows,
            "approved": False}


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation_json", type=Path,
                        help="JSON with image_xy TL,TR,BR,BL; optional cx, cy, certified_depth_m")
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args(argv)
    annotation = json.loads(args.annotation_json.read_text(encoding="utf-8"))
    report = compare_hypotheses(annotation["image_xy"], cx=annotation.get("cx", 319.5),
                                cy=annotation.get("cy", 179.5),
                                certified_depth_m=annotation.get("certified_depth_m"))
    report["source_image"] = annotation.get("source_image")
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
