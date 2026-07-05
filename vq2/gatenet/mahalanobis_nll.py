"""COR-132 T3 — 2x2 log-Cholesky per-corner Gaussian NLL on corner residuals.

The covariance head (``gatenet_v2.GateNetV2`` with ``enable_covariance=True``)
emits, per corner slot, three RAW Cholesky parameters ``(a, c, b)`` — log-diag
``a, c`` and off-diag ``b`` — with NO activation (the loss owns the reparam).
This module turns those into a positive-semidefinite 2x2 covariance and scores
the corner residual ``r`` (GT subpixel corner MINUS predicted corner, in
INPUT-RESOLUTION pixels) under a zero-mean Gaussian.

Reparam (lower-triangular Cholesky factor)::

    L = [[exp(a_clamped), 0],
         [b,              exp(c_clamped)]]
    Sigma = L @ L^T   (PSD by construction)

with ``a, c`` clamped to ``[LOG_DIAG_MIN, LOG_DIAG_MAX] = [-6, 6]`` BEFORE the
exp (so a runaway head logit can never overflow exp()).  All clamping and every
downstream step run in **fp32** — inputs are cast ``.float()`` on entry because
bf16 silently defeats fp32-eps guards (lessons.md 2026-06-10: clamp/eps no-op in
bf16, sigmoid saturates to exactly 1.0).

NLL per corner (2D Gaussian, mean 0)::

    NLL = 0.5 * r^T Sigma^-1 r  +  0.5 * logdet(Sigma)  +  log(2*pi)

The 2x2 inverse and logdet are computed in CLOSED FORM from ``L`` (no
``torch.inverse`` / no Cholesky solve):

  * ``logdet(Sigma) = 2 * logdet(L) = 2 * (a_clamped + c_clamped)`` when the
    floor is applied via the off-diagonal-preserving ``Sigma += floor^2 * I``
    path we DON'T use; see the floor note below — with the diag-clamp floor we
    recompute logdet from the floored diagonal.
  * ``Sigma^-1 = L^-T L^-1`` with ``L^-1`` the closed-form lower-tri inverse.

Variance floor (config ``sigma_floor_px``, units = PIXELS at input resolution):
applied as a floor on the DIAGONAL VARIANCES of ``Sigma`` —
``Sigma_ii := max(Sigma_ii, floor^2)`` — which keeps ``Sigma`` PSD (it only
raises the diagonal) and is reported back via ``info["achieved_floor_px"]`` so
the trainer can record it as an artifact.  ``floor = 0`` is an exact no-op.

  Floor convention DOCUMENTED: we floor the diagonal of the assembled ``Sigma``
  (NOT ``Sigma += floor^2 I``).  This keeps the floor a true per-axis variance
  lower bound (``sqrt(Sigma_ii) >= floor`` px) without inflating off-diagonal
  correlation, and the logdet is recomputed from the floored 2x2 in closed form.

Guard idioms adapted from ``perception/unified_mapping/.../da3_loss.py``
(``check_and_fix_inf_nan``, log-clamp).  This is an ADAPTATION (DA3's loss is a
scalar-confidence regression weight; here it is a full 2x2 Mahalanobis NLL), not
a drop-in.

Returns also the per-corner detached ``d2 = r^T Sigma^-1 r`` so the trainer can
log mean-d^2 — the calibration tripwire: at the MLE / when calibrated it should
approach ``E[chi^2_2] = 2``.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from perception.unified_mapping.models.custom_utils.general import (
    check_and_fix_inf_nan,
)

# Log-diagonal clamp range, applied to (a, c) BEFORE exp().  exp(6) ~ 403,
# exp(-6) ~ 0.0025 — generous for pixel-scale stddevs while bounding exp()
# far from overflow.  Clamp happens in fp32 (bf16 would round 1-1e-6 -> 1.0).
LOG_DIAG_MIN = -6.0
LOG_DIAG_MAX = 6.0

_LOG_2PI = math.log(2.0 * math.pi)


def _assemble_sigma(
    chol: torch.Tensor, *, sigma_floor_px: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """From raw cholesky params -> (Sigma, Sigma_inv, logdet, L) — all fp32.

    ``chol[..., 0] = a`` (log-diag x), ``chol[..., 1] = b`` (off-diag),
    ``chol[..., 2] = c`` (log-diag y).  Clamp a,c to [LOG_DIAG_MIN, LOG_DIAG_MAX]
    in fp32 before exp.  Floor the assembled Sigma diagonal at ``floor^2``.

    Closed form for the 2x2 (no torch.inverse):
        L = [[la, 0], [b, lc]],  la=exp(a_c), lc=exp(c_c)
        Sigma = L L^T = [[la^2, la*b], [la*b, b^2 + lc^2]]
        (then diagonal floored)
        det = s00*s11 - s01^2 ;  logdet = log(det)
        Sigma^-1 = (1/det) [[s11, -s01], [-s01, s00]]
    """
    chol = chol.float()
    a = chol[..., 0].clamp(LOG_DIAG_MIN, LOG_DIAG_MAX)
    b = chol[..., 1]
    c = chol[..., 2].clamp(LOG_DIAG_MIN, LOG_DIAG_MAX)

    la = torch.exp(a)
    lc = torch.exp(c)

    s00 = la * la
    s01 = la * b
    s11 = b * b + lc * lc

    floor_var = float(sigma_floor_px) ** 2
    if floor_var > 0.0:
        s00 = torch.clamp(s00, min=floor_var)
        s11 = torch.clamp(s11, min=floor_var)

    det = s00 * s11 - s01 * s01
    # det >= 0 by PSD; guard the tiny-floor degenerate case before log.
    det = det.clamp(min=1e-12)
    logdet = torch.log(det)

    sigma = torch.stack(
        [torch.stack([s00, s01], dim=-1), torch.stack([s01, s11], dim=-1)],
        dim=-2,
    )  # (..., 2, 2)
    inv00 = s11 / det
    inv11 = s00 / det
    inv01 = -s01 / det
    sigma_inv = torch.stack(
        [torch.stack([inv00, inv01], dim=-1),
         torch.stack([inv01, inv11], dim=-1)],
        dim=-2,
    )  # (..., 2, 2)
    L = torch.stack(
        [torch.stack([la, torch.zeros_like(la)], dim=-1),
         torch.stack([b, lc], dim=-1)],
        dim=-2,
    )
    return sigma, sigma_inv, logdet, L


def mahalanobis_nll(
    residual: torch.Tensor,
    chol: torch.Tensor,
    *,
    sigma_floor_px: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Per-corner 2x2 Gaussian NLL on residuals (mean 0).

    Args:
        residual: (..., 2) corner residual ``r`` in INPUT-RESOLUTION pixels
            (GT subpixel corner minus predicted corner).
        chol: (..., 3) raw Cholesky params ``[a, b, c]`` (log-diag x, off-diag,
            log-diag y).  No activation expected — this fn owns the reparam.
        sigma_floor_px: variance floor in pixels (Sigma diag >= floor^2).

    Returns:
        nll:  (...,) fp32 per-corner NLL (NOT reduced).
        d2:   (...,) fp32 DETACHED Mahalanobis quadratic ``r^T Sigma^-1 r`` (the
              calibration tripwire — should approach 2 = E[chi^2_2]).
        info: dict with ``sigma`` (...,2,2 floored), ``logdet`` (...,), ``L``,
              and ``achieved_floor_px`` (the float floor that was applied).
    """
    r = residual.float()
    sigma, sigma_inv, logdet, L = _assemble_sigma(
        chol, sigma_floor_px=sigma_floor_px)

    # d^2 = r^T Sigma^-1 r  (closed form, no solve)
    sir = torch.einsum("...ij,...j->...i", sigma_inv, r)  # Sigma^-1 r
    d2 = (r * sir).sum(dim=-1)  # (...,)
    d2 = check_and_fix_inf_nan(d2, "mahalanobis_d2", hard_max=None)

    nll = 0.5 * d2 + 0.5 * logdet + _LOG_2PI
    nll = check_and_fix_inf_nan(nll, "mahalanobis_nll", hard_max=None)

    info = {
        "sigma": sigma,
        "logdet": logdet,
        "L": L,
        "achieved_floor_px": float(sigma_floor_px),
    }
    return nll, d2.detach(), info


def mahalanobis_nll_masked(
    residual: torch.Tensor,
    chol: torch.Tensor,
    mask: torch.Tensor,
    *,
    sigma_floor_px: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Masked mean of the per-corner NLL + masked mean d^2 + masked count.

    The NLL is summed over the unmasked corners and divided by the unmasked
    count (clamped to >= 1 so an all-masked batch returns a finite, grad-safe
    zero — no nan).  Mirrors the masked-mean idiom the corner-offset L1 uses in
    ``train_gatenet._offset_l1``.

    Args:
        residual: (..., 2) corner residuals (input-res px).
        chol: (..., 3) raw Cholesky params.
        mask: (...) per-corner validity weight in {0, 1} (broadcastable to the
            leading dims of ``residual``).
        sigma_floor_px: variance floor in pixels.

    Returns:
        dict with:
            ``nll``: scalar masked-mean NLL (the trained term).
            ``mean_d2``: scalar masked-mean Mahalanobis d^2 (detached tripwire).
            ``count``: scalar number of unmasked corners (float).
            ``achieved_floor_px``: float floor applied.
    """
    nll, d2, info = mahalanobis_nll(
        residual, chol, sigma_floor_px=sigma_floor_px)
    m = mask.float()
    denom = m.sum().clamp(min=1.0)
    nll_mean = (nll * m).sum() / denom
    mean_d2 = (d2 * m).sum() / denom
    return {
        "nll": nll_mean,
        "mean_d2": mean_d2.detach(),
        "count": m.sum().detach(),
        "achieved_floor_px": info["achieved_floor_px"],
    }
