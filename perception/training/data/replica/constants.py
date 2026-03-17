"""Replica dataset constants (NICE-SLAM version)."""

import numpy as np

# All 8 scenes in the NICE-SLAM Replica download
SCENES = [
    "room0", "room1", "room2",
    "office0", "office1", "office2", "office3", "office4",
]

# Shared intrinsics for all scenes
FX = 600.0
FY = 600.0
CX = 599.5
CY = 339.5
W = 1200
H = 680

K = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# 16-bit PNG depth scale: divide raw uint16 by this to get meters
DEPTH_SCALE = 6553.5
