"""COR-132 T4 — inference-side peak decode (GateNetV2 maps -> per-corner obs).

Turns the dense stride-4 heads into per-corner ``(x, y, Sigma 2x2, visibility)``
observations in INPUT-pixel coordinates, ready for
``perception.geometry.weighted_pnp``.  DECODES CONSUME MAPS, never GT
(lessons.md 2026-06-11) — this module takes only model output tensors.

Conventions (locked to training):

* Position decode = argmax peak cell + sub-pixel offset read AT that cell,
  ``(cell + 0.5 + off) * stride`` — numerically identical to
  ``training.decode_corners.peak_offset_xy`` (the exact training-target
  inverse; parity is unit-tested).
* Sigma is assembled from the raw log-Cholesky channels read at the SAME peak
  cell through ``training.mahalanobis_nll._assemble_sigma`` — THE Sigma
  assembly every consumer goes through, variance floor INCLUDED (floor parity
  is a T3 invariant; pass the config's ``cov_loss.sigma_floor_px``).
* Slot semantics: channels 0..3 = toward-face slots, 4..7 = away-face slots;
  slot ``i`` pairs with ``weighted_pnp.SLOT_KEYS[i]`` exactly up to the Z4
  cyclic gauge (the toward/away split resolves the Z2 face flip).
* The center channel is the detection anchor for the single dominant instance
  (T2-T4 proxy regime): ``detected = center peak >= center_score_min``.

Off-GT-cell mitigation (the key T3 carry-over — lessons 2026-06-11): 27.6% of
corners argmax-decode to a cell != the GT cell, where the offset/Sigma reads
are UNSUPERVISED (measured d2 ~317 there vs 2.11 on-cell).  At inference there
is no GT cell, so this decode flags ``suspect`` corners from map-side
signatures alone and INFLATES their Sigma (sigma_inflation^2) so the weighted
PnP down-weights them; the post-PnP conformance rejection in ``weighted_pnp``
is the geometric backstop.  Suspect signatures:

* ``suspect_boundary`` — |offset| within ``boundary_margin`` of the cell edge
  (>= ``boundary_margin``, default 0.4).  THE dominant off-cell mode (measured
  on seed 4242: 133/143 off-cell corners are ONE-cell neighbors, median pixel
  error 1.3 px): the GT sits near a cell boundary, the argmax is a coin flip
  between the two cells, and the loser's offset/Sigma reads are unsupervised.
  The decoded |offset| ~ 0.5 is the boundary-proximity signature; the
  symmetric on-cell boundary corners are flagged too (statistically honest —
  same coin flip, they just won it).
* ``suspect_off_range`` — |offset| beyond the trained target range.  The
  offset head is supervised ONLY at GT cells with targets in [-0.5, 0.5)
  (``gt_fractional_offset``); a read far outside that band is an unsupervised
  cell talking.
* ``suspect_low_peak`` — peak score below ``peak_score_suspect``: the forced
  1.0-peak supervision makes a committed channel peak high; an uncommitted /
  smeared channel peaks low and its argmax is unreliable.
* ``suspect_diffuse`` — the local heatmap mass centroid (window
  ``2*diffuse_window+1`` around the argmax) sits > ``diffuse_com_cells`` cells
  from the argmax: multi-modal / smeared response, the argmax is a coin flip
  between competing cells.
* ``suspect_multimodal`` — a secondary peak OUTSIDE the suppression window
  (same half-width) at >= ``multimodal_ratio`` of the argmax value: the FAR
  off-cell mode (wrong mode won outright — measured 10/143 with 9-63 px
  errors at healthy peak scores, invisible to any local-window statistic).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .mahalanobis_nll import _assemble_sigma

__all__ = ["DecodedFrame", "decode_frame", "decode_batch", "N_SLOTS"]

N_SLOTS = 8


@dataclass
class DecodedFrame:
    """Per-frame decode result (all arrays float64/bool numpy, SLOT order).

    ``sigma`` is the PnP-facing covariance (suspect-inflated); ``sigma_raw``
    is the floor-included assembly (what calibration measured), times the
    optional global ``sigma_recal_scale`` when one is passed (recal is part
    of the BASE Sigma; inflation multiplies on top).
    """

    detected: bool                      # center anchor above threshold
    center_xy: np.ndarray               # (2,) input px
    center_score: float
    corner_xy: np.ndarray               # (8,2) input px (padded frame)
    sigma: np.ndarray                   # (8,2,2) px^2, inflation applied
    sigma_raw: np.ndarray               # (8,2,2) px^2, floor-parity assembly
    chol: np.ndarray                    # (8,3) raw log-Cholesky params at peak
    visibility: Optional[np.ndarray]    # (8,) sigmoid prob at peak, or None
    peak_score: np.ndarray              # (8,) heatmap value at argmax
    peak_cell: np.ndarray               # (8,2) long [col, row]
    offset: np.ndarray                  # (8,2) raw (dx, dy) read at peak
    suspect: np.ndarray                 # (8,) bool — any signature fired
    suspect_boundary: np.ndarray        # (8,) bool
    suspect_off_range: np.ndarray       # (8,) bool
    suspect_low_peak: np.ndarray        # (8,) bool
    suspect_diffuse: np.ndarray         # (8,) bool
    suspect_multimodal: np.ndarray      # (8,) bool
    usable: np.ndarray                  # (8,) bool — feed to weighted_pnp


def _peak_read(hm: torch.Tensor, per_slot: torch.Tensor, n_per: int):
    """Argmax of each (C,Hs,Ws) channel + gather of an n_per-per-slot map.

    Returns (flat_idx (C,), col (C,), row (C,), score (C,), vals (C,n_per)).
    """
    c, hs, ws = hm.shape
    flat = hm.reshape(c, hs * ws)
    idx = flat.argmax(dim=-1)                       # (C,)
    score = flat.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    row = idx // ws
    col = idx % ws
    v = per_slot.reshape(c, n_per, hs * ws)
    vals = v.gather(-1, idx.view(c, 1, 1).expand(c, n_per, 1)).squeeze(-1)
    return idx, col, row, score, vals


def _com_deviation_cells(hm: torch.Tensor, col: torch.Tensor,
                         row: torch.Tensor, window: int) -> torch.Tensor:
    """Distance (cells) between each channel's argmax and the local heatmap
    centre-of-mass in a (2*window+1)^2 neighborhood (border-clamped)."""
    c, hs, ws = hm.shape
    dev = torch.zeros(c, dtype=torch.float64)
    for i in range(c):
        r0 = max(int(row[i]) - window, 0)
        r1 = min(int(row[i]) + window, hs - 1)
        c0 = max(int(col[i]) - window, 0)
        c1 = min(int(col[i]) + window, ws - 1)
        patch = hm[i, r0: r1 + 1, c0: c1 + 1].double()
        mass = patch.sum()
        if float(mass) <= 0.0:
            continue
        ys = torch.arange(r0, r1 + 1, dtype=torch.float64).view(-1, 1)
        xs = torch.arange(c0, c1 + 1, dtype=torch.float64).view(1, -1)
        com_y = float((patch * ys).sum() / mass)
        com_x = float((patch * xs).sum() / mass)
        dev[i] = float(np.hypot(com_x - float(col[i]), com_y - float(row[i])))
    return dev


def _secondary_peak_ratio(hm: torch.Tensor, col: torch.Tensor,
                          row: torch.Tensor, window: int) -> torch.Tensor:
    """max(heatmap OUTSIDE the (2*window+1)^2 box around the argmax) / peak.

    A clean sigma~2-cell Gaussian channel has its first outside-the-box ring
    at >= window+1 cells, i.e. <= exp(-(window+0.5)^2/(2*sigma^2)) of the
    mode — well under 0.5 for window=2.  A distant competing mode shows up as
    a high ratio regardless of where it sits.
    """
    c, hs, ws = hm.shape
    ratio = torch.zeros(c, dtype=torch.float64)
    for i in range(c):
        masked = hm[i].clone()
        r0 = max(int(row[i]) - window, 0)
        r1 = min(int(row[i]) + window, hs - 1)
        c0 = max(int(col[i]) - window, 0)
        c1 = min(int(col[i]) + window, ws - 1)
        peak = float(masked[int(row[i]), int(col[i])])
        masked[r0: r1 + 1, c0: c1 + 1] = 0.0
        if peak > 0.0:
            ratio[i] = float(masked.max()) / peak
    return ratio


@torch.no_grad()
def decode_frame(
    out,
    idx: int,
    *,
    stride: int,
    sigma_floor_px: float,
    center_score_min: float = 0.25,
    corner_score_min: float = 0.10,
    boundary_margin: float = 0.4,
    offset_suspect_range: float = 0.6,
    peak_score_suspect: float = 0.30,
    diffuse_window: int = 2,
    diffuse_com_cells: float = 0.75,
    multimodal_ratio: float = 0.5,
    sigma_inflation: float = 10.0,
    sigma_recal_scale: Optional[float] = None,
) -> DecodedFrame:
    """Decode batch element ``idx`` of a dense-head model output.

    Args:
        out: ``GateNetV2Output``-shaped object (``corner_heatmap`` (B,8,Hs,Ws)
            sigmoid, ``corner_offset`` (B,16,..), ``center_heatmap`` (B,1,..),
            ``center_offset`` (B,2,..), optional ``corner_visibility`` logits,
            ``covariance`` (B,24,..) raw — REQUIRED).
        idx: batch index to decode.
        stride: output stride (4 for the dense heads).
        sigma_floor_px: variance floor in INPUT px — MUST equal the training /
            calibration ``cov_loss.sigma_floor_px`` (floor parity invariant).
        center_score_min: detection-anchor threshold on the center peak.
        corner_score_min: per-corner usability threshold on the peak score.
        boundary_margin: |offset| at or beyond this (cell fractions) flags the
            corner as a cell-boundary coin flip (the dominant off-cell mode).
        offset_suspect_range: |offset| beyond this (cell fractions; trained
            range is [-0.5, 0.5)) flags the corner suspect.
        peak_score_suspect: peak score below this flags the corner suspect.
        diffuse_window: half-width (cells) of the CoM neighborhood.
        diffuse_com_cells: CoM-to-argmax deviation (cells) above this flags
            the corner suspect.
        multimodal_ratio: secondary-peak-to-peak ratio (outside the
            ``diffuse_window`` box) at or above this flags the corner suspect.
        sigma_inflation: Sigma multiplier applied as ``sigma_inflation**2`` to
            suspect corners (std-dev scale factor).
        sigma_recal_scale: optional T3/T5 GLOBAL scale recalibration applied
            to the assembled Sigma in the VARIANCE domain, ``Sigma' = s *
            Sigma`` (LINEAR in s, NOT s**2) — exactly the convention
            ``perception.eval.calibration`` applies in its after-scale eval
            (``d2_with_recal_scale``: ``d2' = d2 / s``).  Recal is part of the
            BASE Sigma (``sigma_raw``); the suspect ``sigma_inflation``
            multiplies on top.  ``None`` (the default) is identity —
            byte-identical to the pre-recal decode.  Only the GLOBAL scalar is
            plumbed (inference doesn't know the calibration stratum).

    Returns:
        :class:`DecodedFrame`.
    """
    if getattr(out, "covariance", None) is None:
        raise ValueError(
            "decode_frame requires the covariance head (model "
            "enable_covariance=true); got covariance=None")

    hm = out.corner_heatmap[idx].detach().float().cpu()       # (8,Hs,Ws)
    off_map = out.corner_offset[idx].detach().float().cpu()   # (16,Hs,Ws)
    cov_map = out.covariance[idx].detach().float().cpu()      # (24,Hs,Ws)
    chm = out.center_heatmap[idx].detach().float().cpu()      # (1,Hs,Ws)
    coff = out.center_offset[idx].detach().float().cpu()      # (2,Hs,Ws)
    vis_map = getattr(out, "corner_visibility", None)
    if vis_map is not None:
        vis_map = vis_map[idx].detach().float().cpu()         # (8,Hs,Ws)

    c, hs, ws = hm.shape
    if c != N_SLOTS or off_map.shape[0] != 2 * N_SLOTS \
            or cov_map.shape[0] != 3 * N_SLOTS:
        raise ValueError(
            f"channel contract violated: hm {c}, offset {off_map.shape[0]}, "
            f"cov {cov_map.shape[0]} (expected 8/16/24)")

    # ---- corner peak-pick + per-cell reads --------------------------------- #
    _, col, row, score, off = _peak_read(hm, off_map, 2)      # off (8,2)
    _, _, _, _, chol = _peak_read(hm, cov_map, 3)             # chol (8,3)
    # (offset and cov are read at the SAME argmax cell as the position decode)

    xy = torch.stack([
        (col.double() + 0.5 + off[:, 0].double()) * float(stride),
        (row.double() + 0.5 + off[:, 1].double()) * float(stride),
    ], dim=-1)                                                 # (8,2) input px

    # ---- Sigma: THE training/calibration assembly path (floor included) ---- #
    sigma_t, _, _, _ = _assemble_sigma(chol, sigma_floor_px=sigma_floor_px)
    sigma_raw = sigma_t.double().numpy()                       # (8,2,2)
    if sigma_recal_scale is not None:
        # T3/T5 global scale recal: Sigma' = s * Sigma (variance domain)
        sigma_raw = sigma_raw * float(sigma_recal_scale)

    # ---- visibility (sigmoid of logits at the peak cell) ------------------- #
    visibility = None
    if vis_map is not None:
        flat = vis_map.reshape(N_SLOTS, hs * ws)
        pk = hm.reshape(N_SLOTS, hs * ws).argmax(dim=-1, keepdim=True)
        visibility = torch.sigmoid(
            flat.gather(-1, pk).squeeze(-1)).double().numpy()

    # ---- suspect signatures (off-GT-cell mitigation) ----------------------- #
    off_np = off.double().numpy()
    abs_off = np.abs(off_np)
    suspect_boundary = (abs_off >= float(boundary_margin)).any(axis=1)
    suspect_off_range = (abs_off > float(offset_suspect_range)).any(axis=1)
    score_np = score.double().numpy()
    suspect_low_peak = score_np < float(peak_score_suspect)
    dev = _com_deviation_cells(hm, col, row, int(diffuse_window)).numpy()
    suspect_diffuse = dev > float(diffuse_com_cells)
    ratio = _secondary_peak_ratio(hm, col, row, int(diffuse_window)).numpy()
    suspect_multimodal = ratio >= float(multimodal_ratio)
    suspect = (suspect_boundary | suspect_off_range | suspect_low_peak
               | suspect_diffuse | suspect_multimodal)

    sigma = sigma_raw.copy()
    if float(sigma_inflation) > 0.0:
        sigma[suspect] *= float(sigma_inflation) ** 2

    usable = score_np >= float(corner_score_min)

    # ---- center anchor ------------------------------------------------------ #
    _, ccol, crow, cscore, c_off = _peak_read(chm, coff, 2)
    center_xy = np.array([
        (float(ccol[0]) + 0.5 + float(c_off[0, 0])) * float(stride),
        (float(crow[0]) + 0.5 + float(c_off[0, 1])) * float(stride),
    ])
    center_score = float(cscore[0])
    detected = center_score >= float(center_score_min)

    return DecodedFrame(
        detected=detected,
        center_xy=center_xy,
        center_score=center_score,
        corner_xy=xy.numpy(),
        sigma=sigma,
        sigma_raw=sigma_raw,
        chol=chol.double().numpy(),
        visibility=visibility,
        peak_score=score_np,
        peak_cell=np.stack([col.long().numpy(), row.long().numpy()], axis=-1),
        offset=off_np,
        suspect=suspect,
        suspect_boundary=suspect_boundary,
        suspect_off_range=suspect_off_range,
        suspect_low_peak=suspect_low_peak,
        suspect_diffuse=suspect_diffuse,
        suspect_multimodal=suspect_multimodal,
        usable=usable,
    )


@torch.no_grad()
def decode_batch(out, *, stride: int, sigma_floor_px: float,
                 **kwargs) -> List[DecodedFrame]:
    """Decode every batch element (see :func:`decode_frame`)."""
    b = out.corner_heatmap.shape[0]
    return [
        decode_frame(out, i, stride=stride, sigma_floor_px=sigma_floor_px,
                     **kwargs)
        for i in range(b)
    ]
