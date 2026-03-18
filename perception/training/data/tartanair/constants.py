"""TartanAir dataset constants (v1)."""

import numpy as np

# All scenes use the same pinhole camera
FX = 320.0
FY = 320.0
CX = 320.0
CY = 240.0
W = 640
H = 480
FOV = 90  # degrees

K = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# Stereo baseline (not used for covisibility, but documented)
BASELINE = 0.25  # meters

# Max depth to clip (sky = very large values)
MAX_DEPTH = 100.0  # meters

# 10 environments selected for drone racing VPR
ENVIRONMENTS = [
    "abandonedfactory",
    "abandonedfactory_night",
    "carwelding",
    "hospital",
    "office",
    "amusement",
    "soulcity",
    "gascola",
    "japanesealley",
    "westerndesert",
]
