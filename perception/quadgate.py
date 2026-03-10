"""QuadGate — 4-corner quadrilateral gate representation with PnP and projection.

OpenCV is required for ``pnp_pose`` and ``project``.  The import is gated
so that the dataclass itself is usable without cv2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

__all__ = ["QuadGate", "CV2_AVAILABLE"]

CV2_AVAILABLE = _CV2_AVAILABLE

# Default 1 m x 1 m gate centered at the origin in the gate's local frame.
_DEFAULT_HALF = 0.5
_DEFAULT_CORNERS_3D = np.array(
    [
        [-_DEFAULT_HALF, -_DEFAULT_HALF, 0.0],
        [_DEFAULT_HALF, -_DEFAULT_HALF, 0.0],
        [_DEFAULT_HALF, _DEFAULT_HALF, 0.0],
        [-_DEFAULT_HALF, _DEFAULT_HALF, 0.0],
    ],
    dtype=np.float64,
)


@dataclass
class QuadGate:
    """A 4-corner quadrilateral gate in 3D space.

    Attributes:
        corners_3d: (4, 3) array of 3D corner positions in world frame.
            Defaults to a 1.0 m x 1.0 m square gate in the XY-plane.
    """

    corners_3d: NDArray[np.float64] = field(
        default_factory=lambda: _DEFAULT_CORNERS_3D.copy()
    )

    def __post_init__(self) -> None:
        self.corners_3d = np.asarray(self.corners_3d, dtype=np.float64)
        if self.corners_3d.shape != (4, 3):
            raise ValueError(
                f"corners_3d must have shape (4, 3), got {self.corners_3d.shape}"
            )

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    @property
    def center(self) -> NDArray[np.float64]:
        """Gate center as the mean of the four corners."""
        return self.corners_3d.mean(axis=0)

    @property
    def size(self) -> tuple[float, float]:
        """Approximate (width, height) from corner distances."""
        w = float(np.linalg.norm(self.corners_3d[1] - self.corners_3d[0]))
        h = float(np.linalg.norm(self.corners_3d[3] - self.corners_3d[0]))
        return (w, h)

    # ------------------------------------------------------------------ #
    # PnP pose estimation
    # ------------------------------------------------------------------ #

    def pnp_pose(
        self,
        corners_2d: NDArray[np.float64],
        camera_intrinsics: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Estimate relative gate pose from 2D corner detections via solvePnP.

        Args:
            corners_2d: (4, 2) array of detected 2D corner pixel coordinates.
            camera_intrinsics: (3, 3) camera intrinsic matrix.

        Returns:
            (R, t) where R is a (3, 3) rotation matrix and t is a (3,)
            translation vector from camera frame to gate frame.

        Raises:
            ImportError: If OpenCV is not installed.
            RuntimeError: If solvePnP fails to find a solution.
        """
        if not _CV2_AVAILABLE:
            raise ImportError(
                "pnp_pose requires OpenCV.  Install with:  pip install opencv-python-headless"
            )

        corners_2d = np.asarray(corners_2d, dtype=np.float64)
        camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
        dist_coeffs = np.zeros(4, dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.corners_3d,
            corners_2d,
            camera_intrinsics,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            raise RuntimeError("solvePnP failed to converge")

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()
        return R, t

    # ------------------------------------------------------------------ #
    # Forward projection
    # ------------------------------------------------------------------ #

    def project(
        self,
        camera_intrinsics: NDArray[np.float64],
        R_cam: NDArray[np.float64],
        t_cam: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Project 3D gate corners into 2D image coordinates.

        Args:
            camera_intrinsics: (3, 3) camera intrinsic matrix.
            R_cam: (3, 3) rotation matrix (world-to-camera).
            t_cam: (3,) translation vector (world-to-camera).

        Returns:
            (4, 2) array of projected 2D pixel coordinates.

        Raises:
            ImportError: If OpenCV is not installed.
        """
        if not _CV2_AVAILABLE:
            raise ImportError(
                "project requires OpenCV.  Install with:  pip install opencv-python-headless"
            )

        camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
        R_cam = np.asarray(R_cam, dtype=np.float64)
        t_cam = np.asarray(t_cam, dtype=np.float64).ravel()

        rvec, _ = cv2.Rodrigues(R_cam)
        pts_2d, _ = cv2.projectPoints(
            self.corners_3d,
            rvec,
            t_cam,
            camera_intrinsics,
            np.zeros(4, dtype=np.float64),
        )
        return pts_2d.reshape(4, 2)
