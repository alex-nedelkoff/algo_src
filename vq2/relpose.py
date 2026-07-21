"""Relative-pose measurement contract (2026-07-21).

Shared interface between the two work branches and the smoother:

  feat/pillar-pnp  -> LandmarkRelPose  (placard/pillar PnP-PnL, gate PnP)
  feat/vo-loop     -> OdomDelta        (DPVO / any VO ego-motion)

Perception sources emit these time-keyed records; the association /
scheduling layer resolves identity against the map (map_ingest) and window
state indices, then lowers them to window_smoother factors (PosUnary,
YawUnary, PillarFactor, OdomFactor). Sources never touch state indices —
capture time is the only key they know.

Frames follow the existing conventions:
  LEVEL  = roll/pitch applied, yaw NOT applied, body origin (PillarFactor).
  dp_local for deltas is yaw-relative frame of the earlier state (OdomFactor).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .window_smoother import OdomFactor

# Known producers; association may weight/gate by source.
LANDMARK_SOURCES = ("pillar_pnp", "pillar_pnl", "gate_pnp")
ODOM_SOURCES = ("dpvo", "vo_generic")


@dataclass(frozen=True)
class LandmarkRelPose:
    """One landmark-relative pose measurement.

    A single solved placard quad (known plate size + digits) is a full PnP
    fix: identity (number) + landmark position in the LEVEL frame. Identity
    may be None (quad solved, digits unread) — association can still try
    bearing-gated matching, PillarFactor-style, against all candidates.
    """
    t: float                    # capture time (s, flight clock)
    source: str                 # one of LANDMARK_SOURCES
    number: str | None          # identity read ("22"); None = unread
    p_lm_level: np.ndarray      # (3,) landmark pos, LEVEL frame, body origin
    sigma_p: float = 0.8        # m, isotropic 1-sigma of p_lm_level
    yaw_rel: float | None = None  # body yaw in the landmark frame, if observed
    sigma_yaw: float = 0.05
    quality: float = 1.0        # detector confidence [0, 1]

    def __post_init__(self):
        if self.source not in LANDMARK_SOURCES:
            raise ValueError(f"unknown landmark source: {self.source!r}")
        p = np.asarray(self.p_lm_level, dtype=np.float64)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError("p_lm_level must be finite (3,)")
        object.__setattr__(self, "p_lm_level", p)


@dataclass(frozen=True)
class OdomDelta:
    """Ego-motion delta between two capture times (VO). Lowered to an
    OdomFactor once both endpoint states exist in the window. scale_locked
    mirrors the DPVO scale-lock telemetry: unlocked deltas are direction-only
    information — association must inflate sigma_p or drop them."""
    t0: float
    t1: float
    dp_local: np.ndarray        # (3,) position delta, yaw-relative frame @ t0
    dyaw: float
    source: str = "dpvo"        # one of ODOM_SOURCES
    sigma_p: float = 0.25
    sigma_yaw: float = 0.02
    scale_locked: bool = True

    def __post_init__(self):
        if self.source not in ODOM_SOURCES:
            raise ValueError(f"unknown odom source: {self.source!r}")
        if not self.t1 > self.t0:
            raise ValueError(f"require t1 > t0 (got {self.t0}..{self.t1})")
        dp = np.asarray(self.dp_local, dtype=np.float64)
        if dp.shape != (3,) or not np.all(np.isfinite(dp)):
            raise ValueError("dp_local must be finite (3,)")
        object.__setattr__(self, "dp_local", dp)

    def to_factor(self, i: int, j: int) -> OdomFactor:
        """Lower to a window_smoother OdomFactor between state indices i<j
        (capture-time-nearest to t0/t1; the caller owns that mapping)."""
        if not j > i:
            raise ValueError(f"require j > i (got {i}, {j})")
        return OdomFactor(i=i, j=j, dp_local=self.dp_local, dyaw=self.dyaw,
                          sigma_p=self.sigma_p, sigma_yaw=self.sigma_yaw)
