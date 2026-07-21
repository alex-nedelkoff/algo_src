"""Pillar-read utilities shared by offline and live perception.

The visual detector/OCR supplies a bounding box and, when readable, a station
number.  This module deliberately does not prescribe a detector: the smoother
only needs an image bearing, while the OCR result selects map candidates.

One physical pillar can carry the same number at several heights.  A lower
marking is safe for the azimuth-only PnL factor but must never be interpreted
as the surveyed top panel for elevation/range.  ``classify_top_reads`` applies
the production rule from the pillar-estimator handoff.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class PillarRead:
    """One detector/OCR observation of a station-number panel.

    Coordinates use the existing OpenCV image convention.  ``number`` is
    ``None`` when the panel was detected but OCR could not identify it.
    """

    t_capture: float
    x: float
    y: float
    w: float
    h: float
    number: str | None = None
    confidence: float = 1.0
    top: bool = False

    @property
    def center_u(self) -> float:
        return self.x + self.w / 2.0


def classify_top_reads(
    reads: Iterable[PillarRead],
    column_px: float = 30.0,
    far_panel_h_px: float = 10.0,
) -> list[PillarRead]:
    """Return reads with conservative top-panel flags.

    A readable panel is top when it is the highest member of a stack of at
    least two same-number reads in one image column.  Tiny readable panels are
    also top: at long range the lit top panel is the only reliably readable
    marking.  All other observations remain azimuth-only.
    """
    result = list(reads)
    for i, read in enumerate(result):
        is_top = bool(read.number is not None and read.h <= far_panel_h_px)
        if read.number is not None:
            stack = [other for other in result
                     if other.number == read.number
                     and abs(other.center_u - read.center_u) <= column_px]
            if len(stack) >= 2:
                is_top = is_top or read.y == min(other.y for other in stack)
        result[i] = replace(read, top=is_top)
    return result
