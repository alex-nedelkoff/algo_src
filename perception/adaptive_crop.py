"""AdaptiveCrop — MAVLab-style ROI tracking for gate detection.

Maintains a bounding-box ROI that contracts around successful detections
and expands when detection is lost, keeping the search area manageable
while preventing tunnel-vision lock-out.
"""

from __future__ import annotations


class AdaptiveCrop:
    """Adaptive ROI tracker for gate detection.

    When the detector finds the gate the ROI tightens around it (with a
    configurable margin), speeding up subsequent inference.  When the gate
    is lost the ROI progressively expands to reacquire it.

    Args:
        image_size: (H, W) of the full image.
        initial_roi: Optional (x1, y1, x2, y2) initial crop.  Defaults to
            the full image.
        margin: Multiplicative margin applied when contracting around a
            detection (e.g. 1.5 means 50 % padding around the detection
            bounding box).
        expansion_factor: Factor by which the ROI grows on each missed
            detection frame (e.g. 1.3 means 30 % expansion per miss).
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        initial_roi: tuple[float, float, float, float] | None = None,
        margin: float = 1.5,
        expansion_factor: float = 1.3,
    ) -> None:
        self.image_h, self.image_w = image_size
        self.margin = margin
        self.expansion_factor = expansion_factor

        if initial_roi is not None:
            self.roi = tuple(float(v) for v in initial_roi)
        else:
            self.roi: tuple[float, float, float, float] = (
                0.0,
                0.0,
                float(self.image_w),
                float(self.image_h),
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _clamp_roi(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> tuple[float, float, float, float]:
        """Clamp ROI to image bounds."""
        x1 = max(0.0, min(x1, float(self.image_w)))
        y1 = max(0.0, min(y1, float(self.image_h)))
        x2 = max(0.0, min(x2, float(self.image_w)))
        y2 = max(0.0, min(y2, float(self.image_h)))
        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update(
        self,
        detection_success: bool,
        detection_bbox: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float, float, float]:
        """Update the ROI based on the latest detection result.

        Args:
            detection_success: Whether a gate was detected this frame.
            detection_bbox: (x1, y1, x2, y2) bounding box of the detection.
                Required when ``detection_success`` is ``True``.

        Returns:
            The updated ROI as (x1, y1, x2, y2).
        """
        if detection_success and detection_bbox is not None:
            bx1, by1, bx2, by2 = detection_bbox
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            half_w = (bx2 - bx1) / 2.0 * self.margin
            half_h = (by2 - by1) / 2.0 * self.margin
            self.roi = self._clamp_roi(
                cx - half_w, cy - half_h, cx + half_w, cy + half_h
            )
        else:
            # Expand current ROI around its center
            x1, y1, x2, y2 = self.roi
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            half_w = (x2 - x1) / 2.0 * self.expansion_factor
            half_h = (y2 - y1) / 2.0 * self.expansion_factor
            self.roi = self._clamp_roi(
                cx - half_w, cy - half_h, cx + half_w, cy + half_h
            )

        return self.roi
