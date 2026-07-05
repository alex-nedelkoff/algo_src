"""GateNet v2 — SegFormer (MiT) backbone with dense stride-4 detection heads.

Loads a pretrained SegFormer encoder + all-MLP decode head from HuggingFace,
discards the original semantic classifier, and attaches a set of dense output
heads that share the decode head's fused stride-4 features.

Two head configurations coexist behind one ``GateNetV2`` class:

  * **Legacy dual-head** (``dense_heads=False``, the default — preserves the
    original B0 contract used by existing callers/tests): a semantic seg head +
    a 4-channel corner heatmap head, returned as a ``(seg_logits, corner_hm)``
    tuple.
  * **Dense corner heads** (``dense_heads=True``, the COR-132 T2 design): the
    toward/away×4 corner-slot head set per ``plans/COR-132/corner-order-design.md``
    D1, returned as a structured :class:`GateNetV2Output`.

Both configs output at **stride 4** (96×96 for a 384×384 input). The encoder and
decode head are fine-tuned from the pretrained weights; the output heads are
randomly initialized.

Backbone selection (``backbone="b0" | "b2"``):
  * ``b0`` — MiT-B0, ``decoder_hidden_size = 256`` (the legacy default).
  * ``b2`` — MiT-B2, ``decoder_hidden_size = 768`` (the COR-132 T2 default).

The fused-feature channel count is **read from ``base.config.decoder_hidden_size``**
(NOT hardcoded) so the head ``in_channels`` track whichever backbone is loaded.

------------------------------------------------------------------------------
Output contract — dense heads (``dense_heads=True``)
------------------------------------------------------------------------------
``forward()`` returns a :class:`GateNetV2Output` NamedTuple with these tensors,
all at stride 4 = ``(H/4, W/4)`` (e.g. 96×96 for 384×384):

  * ``corner_heatmap``   (B, 8, Hs, Ws) — sigmoid heatmaps. 8 channels =
        toward-face slots 0..3 then away-face slots 0..3 (the network's own
        gauge; associated to GT by the symmin loss at train time).
  * ``corner_offset``    (B, 16, Hs, Ws) — raw sub-pixel offsets, 2 per slot
        (dx, dy), channel layout ``[slot0_dx, slot0_dy, slot1_dx, ...]``.
  * ``center_heatmap``   (B, 1, Hs, Ws) — sigmoid center heatmap.
  * ``center_offset``    (B, 2, Hs, Ws) — raw center sub-pixel (dx, dy).
  * ``corner_visibility``(B, 8, Hs, Ws) or ``None`` — per-slot visibility LOGITS
        (no sigmoid; the T2 loss applies BCE-with-logits permuted by k*). Present
        iff ``enable_visibility=True`` (default True).
  * ``seg_logits``       (B, num_seg_classes, Hs, Ws) or ``None`` — semantic seg
        logits. Present iff ``enable_seg=True`` (default False in the dense path).
  * ``covariance``       (B, 24, Hs, Ws) or ``None`` — per-corner 2×2 log-Cholesky
        covariance parameters: 8 corner slots × 3 Cholesky params
        ``(a=log-diag x, b=off-diag, c=log-diag y)``, channel layout
        ``[slot0_a, slot0_b, slot0_c, slot1_a, ...]``, RAW (no activation — the
        ``training.mahalanobis_nll`` loss owns the reparam + clamp). Present iff
        ``enable_covariance=True`` (default False so the T2 mean configs and the
        running mean retrain are untouched). The mean heads are unchanged.
  * ``direct_pose``      (B, 9) or ``None`` — the COR-132 T4 DIAGNOSTIC-ONLY
        direct-6DoF head (global-average-pooled fused features -> small MLP).
        Present iff ``enable_direct_pose=True`` (default False so the T3 cov
        configs and checkpoints are byte-identical when off). It NEVER replaces
        the PnP pose — its job is Z2 face-flip resolution / flagging at decode
        (``perception.decode.flip_resolve``).
  * ``corner_delta``     (B, 16, Hs, Ws) or ``None`` — COR-132 T5 dense
        center-anchored delta head: per slot the SIZE-NORMALIZED displacement
        ``(corner_xy - center_xy) / s`` (s = apparent-size px, bbox diagonal of
        the 8 projected outer corners), channel layout
        ``[slot0_dx, slot0_dy, slot1_dx, ...]``, RAW (targets bounded
        ~[-1.2, 1.2] by gate rigidity — see the R2 delta-normalization
        explainer). Supervised ONLY at GT center cells. Present iff
        ``enable_association_heads=True`` (default False — T2/T3/T4 configs and
        checkpoints byte-identical when off).
  * ``log_size``         (B, 1, Hs, Ws) or ``None`` — COR-132 T5 log-size head:
        ``log(s)`` with s the apparent size in px (~3.5-nat dynamic range across
        2-30 m), RAW, L1 in log space, supervised ONLY at GT center cells.
        Present iff ``enable_association_heads=True``.

Channel-count contract (dense heads): corner=8, corner_offset=16, center=1,
center_offset=2, visibility=8, covariance=24 (when enabled),
direct_pose=9 (when enabled), corner_delta=16 + log_size=1 (when enabled).

------------------------------------------------------------------------------
Direct-6DoF parameterization (COR-132 T4 Pass C — READ THIS)
------------------------------------------------------------------------------
``direct_pose`` is a raw 9-vector per frame: ``[t (3), r6 (6)]``.

* ``t`` (slots 0:3) — camera-frame gate band-midplane centre in METERS, the
  COR-131 ``labels.relative_pose()['t']`` directly (gauge-invariant).
* ``r6`` (slots 3:9) — a Zhou-et-al continuous 6D rotation representation of
  ``R_cam_gate`` (gate->camera): the first two COLUMNS of R, Gram-Schmidt
  orthonormalized by :func:`rot6d_to_matrix` (fp32 internally).

GAUGE: the Z4 cyclic gauge (``R -> R . Rz(90 deg k)``) is UNOBSERVABLE from a
single image, so the training loss quotients it out with a min-over-k geodesic
rotation loss (the PoseCNN ShapeMatch pattern; same quotient as the corner
symmin loss — design D6).  NOTE: the projected-azimuth ``roll_canon`` is NOT
exactly gauge-invariant for general viewpoints (measured spread up to ~45 deg
across gauge representatives when the gate normal is off the camera axis), so
a "predict roll-mod-90" parameterization was rejected; the min-over-k loss is
the exact quotient.  The Z2 face flip (``R -> R . Ry(180)``) IS observable
almost everywhere and is deliberately NOT quotiented — capturing it is the
whole point of this head (flip resolution).  The decoded rotation's
Z4-invariant, Z2-equivariant part is the gate normal ``R[:, 2]``.

------------------------------------------------------------------------------
Slot ↔ label mapping (READ THIS — the trainer depends on it)
------------------------------------------------------------------------------
The dataset collate (``gate_dataset.collate_pad`` / ``heatmaps``) emits a
**9-channel** heatmap stack ordered by ``labels.ALL_POINT_KEYS`` =
``(front_tl, front_tr, front_br, front_bl, back_tl, back_tr, back_br, back_bl,
center)`` — i.e. 8 GATE-LOCAL corners (front/back face × tl/tr/br/bl) + center.

This head's 8 corner channels are NOT those gate-local keys. They are the
TOWARD/AWAY camera-relative SLOTS: channels 0..3 = toward-face slots, 4..7 =
away-face slots (toward = the face nearer the camera; see
``labels.toward_away_sets``). The two orderings are DIFFERENT bases for the same
8 physical corners and are associated at train time by the gauge-quotient
``training.symmin_loss`` (min over the 4 coupled Z4 cyclic shifts, +Z2 flip
inside ``edge_on_band``). The trainer is responsible for re-keying the
9-channel ALL_POINT_KEYS GT stack into the (toward 0..3, away 0..3) slot order +
center using ``labels.toward_away_sets`` before calling the loss — this model
emits raw slot channels and does not consume the GT.
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional, Tuple, Union

if False:  # TYPE_CHECKING — avoid importing torch at type-check only
    import torch

try:
    import torch
    import torch.nn as nn
    from transformers import SegformerForSemanticSegmentation

    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

log = logging.getLogger(__name__)

__all__ = ["GateNetV2", "GateNetV2Output", "rot6d_to_matrix",
           "matrix_to_rot6d", "split_direct_pose"]

# HuggingFace model IDs for the ADE20K-finetuned SegFormer backbones.
_HF_MODEL_IDS = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}

# Dense-head channel-count contract (design D1). Public so tests/trainer can
# assert against the single source of truth.
N_CORNER_SLOTS = 8          # toward 0..3 + away 0..3
N_CORNER_OFFSET = 16        # 2 sub-pixel offsets per slot
N_CENTER = 1
N_CENTER_OFFSET = 2
N_VISIBILITY = 8            # per-slot visibility logits
N_COV_PARAMS_PER_SLOT = 3  # 2x2 log-Cholesky: (a=log-diag x, b=off-diag, c=log-diag y)
N_COVARIANCE = N_CORNER_SLOTS * N_COV_PARAMS_PER_SLOT  # 8 slots x 3 = 24

# COR-132 T5 Pass A — center-anchored association heads (see the T5 design
# context in plans/COR-132/tracers.md and the R2 delta-normalization explainer).
N_CORNER_DELTA = N_CORNER_SLOTS * 2  # 16: per-slot (corner - center)/s, [dx,dy]
N_LOG_SIZE = 1                       # log of the apparent size s (px), log-space

# Direct-6DoF head contract (COR-132 T4 Pass C): [t(3 m), r6(6)] — see the
# module docstring "Direct-6DoF parameterization".  Slices are THE single
# source for the layout; every consumer (loss, decode) goes through them.
N_DIRECT_T = 3
N_DIRECT_ROT6D = 6
N_DIRECT_POSE = N_DIRECT_T + N_DIRECT_ROT6D  # 9
DIRECT_T_SLICE = slice(0, N_DIRECT_T)
DIRECT_R6_SLICE = slice(N_DIRECT_T, N_DIRECT_POSE)


class GateNetV2Output(NamedTuple):
    """Structured forward output for the dense-head config (``dense_heads=True``).

    All tensors are at stride 4 (H/4, W/4). See the module docstring for the full
    channel contract. ``corner_visibility`` / ``seg_logits`` are ``None`` when
    their heads are disabled; ``covariance`` is always ``None`` at T2 (added T3).
    """

    corner_heatmap: "torch.Tensor"          # (B, 8, Hs, Ws) sigmoid
    corner_offset: "torch.Tensor"           # (B, 16, Hs, Ws) raw
    center_heatmap: "torch.Tensor"          # (B, 1, Hs, Ws) sigmoid
    center_offset: "torch.Tensor"           # (B, 2, Hs, Ws) raw
    corner_visibility: Optional["torch.Tensor"] = None  # (B, 8, Hs, Ws) logits
    seg_logits: Optional["torch.Tensor"] = None         # (B, C, Hs, Ws) logits
    covariance: Optional["torch.Tensor"] = None         # (B, 24, Hs, Ws) raw or None
    direct_pose: Optional["torch.Tensor"] = None        # (B, 9) raw or None
    corner_delta: Optional["torch.Tensor"] = None       # (B, 16, Hs, Ws) raw or None
    log_size: Optional["torch.Tensor"] = None           # (B, 1, Hs, Ws) raw or None


def _check_deps() -> None:
    if not _DEPS_AVAILABLE:
        raise ImportError(
            "GateNetV2 requires PyTorch and transformers. "
            "Install with: pip install torch torchvision transformers"
        )


if _DEPS_AVAILABLE:

    def rot6d_to_matrix(r6: "torch.Tensor") -> "torch.Tensor":
        """Zhou-et-al 6D -> rotation matrix, batched ``(..., 6) -> (..., 3, 3)``.

        ``r6 = [a1(3), a2(3)]`` are the raw first-two-COLUMN candidates of R;
        Gram-Schmidt gives ``b1 = norm(a1)``, ``b2 = norm(a2 - (b1.a2) b1)``,
        ``b3 = b1 x b2`` and ``R = [b1 | b2 | b3]`` (always det +1).  Computed
        in fp32 regardless of input dtype (eps-guarded normalizations — the
        bf16 trap, lessons 2026-06-10).
        """
        r6 = r6.float()
        a1, a2 = r6[..., 0:3], r6[..., 3:6]
        b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
        a2p = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(a2p, dim=-1, eps=1e-8)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)  # columns

    def matrix_to_rot6d(R: "torch.Tensor") -> "torch.Tensor":
        """Rotation matrix -> 6D representation (first two columns, flattened).

        Inverse-consistent with :func:`rot6d_to_matrix` for proper rotations:
        ``rot6d_to_matrix(matrix_to_rot6d(R)) == R``.
        """
        R = R.float()
        return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)

    def split_direct_pose(vec: "torch.Tensor"):
        """Split a raw direct-pose vector ``(..., 9)`` -> ``(t (...,3), R (...,3,3))``.

        THE decode of the direct head (single source of the layout): ``t`` is
        raw meters, ``R`` comes through :func:`rot6d_to_matrix`.
        """
        if vec.shape[-1] != N_DIRECT_POSE:
            raise ValueError(
                f"direct_pose vector must have {N_DIRECT_POSE} channels; "
                f"got {tuple(vec.shape)}")
        return vec[..., DIRECT_T_SLICE].float(), rot6d_to_matrix(
            vec[..., DIRECT_R6_SLICE])

    class _Head(nn.Module):
        """Lightweight output head: Conv3x3 + BN + ReLU -> Conv1x1."""

        def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.block(x)

    class GateNetV2(nn.Module):
        """SegFormer (B0/B2) with detection heads for gate detection.

        Args:
            backbone: ``"b0"`` (decoder_hidden_size 256) or ``"b2"`` (768).
                Default ``"b0"`` preserves the legacy contract; the COR-132 T2
                config uses ``"b2"``.
            dense_heads: When False (default) attach the legacy seg + 4-channel
                corner heads and return a ``(seg_logits, corner_heatmaps)`` tuple.
                When True attach the COR-132 dense corner-slot head set and return
                a :class:`GateNetV2Output`.
            num_seg_classes: Semantic seg classes (default 4). In the dense path
                the seg head is only built when ``enable_seg=True``.
            num_corner_channels: Legacy corner-head channel count (default 4).
                Ignored in the dense path (which uses the fixed 8-slot layout).
            enable_seg: Dense path only — build the seg head (default False; the
                B2-mean config does not need it). No effect on the legacy path.
            enable_visibility: Dense path only — build the per-slot visibility
                head (default True; constructor-toggleable per T2 scope).
            enable_covariance: Dense path only — build the per-corner 2×2
                log-Cholesky covariance head (24 channels; default False so the
                T2 mean configs are byte-identical). The mean heads are unchanged
                whether or not this is on; the cov head is trained in the T3
                sigma stage with the mean frozen.
            enable_direct_pose: Dense path only — build the DIAGNOSTIC-ONLY
                direct-6DoF head (global-pooled fused features -> MLP -> 9
                raw values ``[t(3), r6(6)]``; see the module docstring).
                Default False so the T2/T3 configs and checkpoints are
                byte-identical when off. Trained in the T4 ``direct`` stage
                with everything else frozen.
            enable_association_heads: Dense path only — build the COR-132 T5
                center-anchored association heads: the dense 16-channel delta
                head (per-slot ``(corner - center)/s``, size-normalized) and
                the 1-channel log-size head (``log s``). Both raw; supervised
                ONLY at GT center cells by the multi-instance trainer. Default
                False so every existing config/checkpoint is byte-identical
                when off (the enable_covariance / enable_direct_pose pattern).
            pretrained: Load ADE20K-pretrained weights (default True). When False,
                build from config with random weights (no network needed).
            head_mid_channels: Intermediate channels in every output head
                (default 64).

        Input:
            (B, 3, H, W) RGB image tensor (ImageNet-normalized). H, W must be
            divisible by 4 (stride-4 output).

        Output:
            Legacy path: ``(seg_logits (B, num_seg_classes, H/4, W/4),
            corner_heatmaps (B, num_corner_channels, H/4, W/4) in [0, 1])``.
            Dense path: :class:`GateNetV2Output` — see the module docstring for
            the full per-head shape/channel contract.
        """

        def __init__(
            self,
            backbone: str = "b0",
            *,
            dense_heads: bool = False,
            num_seg_classes: int = 4,
            num_corner_channels: int = 4,
            enable_seg: bool = False,
            enable_visibility: bool = True,
            enable_covariance: bool = False,
            enable_direct_pose: bool = False,
            enable_association_heads: bool = False,
            pretrained: bool = True,
            head_mid_channels: int = 64,
        ) -> None:
            _check_deps()
            super().__init__()

            backbone = backbone.lower()
            if backbone not in _HF_MODEL_IDS:
                raise ValueError(
                    f"backbone must be one of {sorted(_HF_MODEL_IDS)}, got {backbone!r}"
                )
            self.backbone = backbone
            self.dense_heads = bool(dense_heads)
            self.enable_seg = bool(enable_seg)
            self.enable_visibility = bool(enable_visibility)
            self.enable_covariance = bool(enable_covariance)
            self.enable_direct_pose = bool(enable_direct_pose)
            self.enable_association_heads = bool(enable_association_heads)
            model_id = _HF_MODEL_IDS[backbone]

            if pretrained:
                log.info("Loading pretrained SegFormer-%s from %s",
                         backbone.upper(), model_id)
                base = SegformerForSemanticSegmentation.from_pretrained(model_id)
            else:
                from transformers import SegformerConfig

                config = SegformerConfig.from_pretrained(model_id)
                config.num_labels = num_seg_classes
                base = SegformerForSemanticSegmentation(config)

            # Extract encoder + decode head (discard the original classifier).
            self.encoder = base.segformer
            self.decode_head = base.decode_head

            # Fused-feature channel count — READ FROM CONFIG, never hardcoded.
            # B0 -> 256, B2 -> 768.
            decoder_hidden_size = base.config.decoder_hidden_size
            self.decoder_hidden_size = int(decoder_hidden_size)

            # Replace the original semantic classifier with Identity so the
            # decode head emits the raw fused (B, decoder_hidden_size, H/4, W/4)
            # features that all output heads consume.
            self.decode_head.classifier = nn.Identity()

            C = self.decoder_hidden_size
            m = head_mid_channels

            if not self.dense_heads:
                # ---- Legacy dual-head path (backward-compatible) -------------
                self.seg_head = _Head(C, m, num_seg_classes)
                self.corner_head = _Head(C, m, num_corner_channels)
            else:
                # ---- Dense corner-slot head set (COR-132 T2 design D1) -------
                self.corner_head = _Head(C, m, N_CORNER_SLOTS)        # 8
                self.corner_offset_head = _Head(C, m, N_CORNER_OFFSET)  # 16
                self.center_head = _Head(C, m, N_CENTER)               # 1
                self.center_offset_head = _Head(C, m, N_CENTER_OFFSET)  # 2
                self.visibility_head = (
                    _Head(C, m, N_VISIBILITY) if self.enable_visibility else None
                )
                self.seg_head = (
                    _Head(C, m, num_seg_classes) if self.enable_seg else None
                )
                # Per-corner 2×2 log-Cholesky covariance head (COR-132 T3): 24 raw
                # channels (8 slots × 3 params), no activation — the NLL loss owns
                # the reparam. Built iff enable_covariance; otherwise None (the T2
                # mean configs stay byte-identical).
                self.covariance_head = (
                    _Head(C, m, N_COVARIANCE) if self.enable_covariance else None
                )
                # DIAGNOSTIC-ONLY direct-6DoF head (COR-132 T4 Pass C):
                # global-average-pooled fused features -> 2-layer MLP -> 9 raw
                # values [t(3), r6(6)] (see the module docstring). Built iff
                # enable_direct_pose; otherwise None (T2/T3 configs and
                # checkpoints stay byte-identical).
                self.direct_pose_head = (
                    nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(1),
                        nn.Linear(C, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, N_DIRECT_POSE),
                    )
                    if self.enable_direct_pose
                    else None
                )
                # COR-132 T5 Pass A association heads: dense 16ch delta
                # ((corner-center)/s, size-normalized) + 1ch log-size (log s),
                # both raw (the trainer owns the L1 at GT center cells).
                # Built iff enable_association_heads; otherwise None (every
                # T2/T3/T4 config and checkpoint stays byte-identical).
                self.corner_delta_head = (
                    _Head(C, m, N_CORNER_DELTA)
                    if self.enable_association_heads
                    else None
                )
                self.log_size_head = (
                    _Head(C, m, N_LOG_SIZE)
                    if self.enable_association_heads
                    else None
                )

        def _fused(self, pixel_values: "torch.Tensor") -> "torch.Tensor":
            """Run encoder + decode head -> fused (B, decoder_hidden_size, H/4, W/4)."""
            encoder_outputs = self.encoder(
                pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = encoder_outputs.hidden_states
            return self.decode_head(hidden_states)

        def forward(
            self, pixel_values: "torch.Tensor"
        ) -> Union[Tuple["torch.Tensor", "torch.Tensor"], GateNetV2Output]:
            """Forward pass. Return type depends on ``dense_heads``.

            Legacy path (``dense_heads=False``) returns the 2-tuple
            ``(seg_logits, corner_heatmaps)`` — seg as raw logits, corners as
            sigmoid heatmaps in [0, 1], both at stride 4.

            Dense path (``dense_heads=True``) returns a :class:`GateNetV2Output`.
            Per-head shapes (stride 4, Hs=H/4, Ws=W/4):
                corner_heatmap    (B, 8, Hs, Ws)  sigmoid
                corner_offset     (B, 16, Hs, Ws) raw
                center_heatmap    (B, 1, Hs, Ws)  sigmoid
                center_offset     (B, 2, Hs, Ws)  raw
                corner_visibility (B, 8, Hs, Ws)  logits or None
                seg_logits        (B, C, Hs, Ws)  logits or None
                covariance        (B, 24, Hs, Ws) raw or None (iff enable_covariance)
                direct_pose       (B, 9)          raw or None (iff enable_direct_pose)
                corner_delta      (B, 16, Hs, Ws) raw or None (iff enable_association_heads)
                log_size          (B, 1, Hs, Ws)  raw or None (iff enable_association_heads)
            """
            fused = self._fused(pixel_values)

            if not self.dense_heads:
                seg_logits = self.seg_head(fused)
                corner_heatmaps = self.corner_head(fused).sigmoid()
                return seg_logits, corner_heatmaps

            corner_heatmap = self.corner_head(fused).sigmoid()
            corner_offset = self.corner_offset_head(fused)
            center_heatmap = self.center_head(fused).sigmoid()
            center_offset = self.center_offset_head(fused)
            corner_visibility = (
                self.visibility_head(fused)
                if self.visibility_head is not None
                else None
            )
            seg_logits = (
                self.seg_head(fused) if self.seg_head is not None else None
            )
            covariance = (
                self.covariance_head(fused)
                if self.covariance_head is not None
                else None
            )
            direct_pose = (
                self.direct_pose_head(fused)
                if self.direct_pose_head is not None
                else None
            )
            corner_delta = (
                self.corner_delta_head(fused)
                if self.corner_delta_head is not None
                else None
            )
            log_size = (
                self.log_size_head(fused)
                if self.log_size_head is not None
                else None
            )
            return GateNetV2Output(
                corner_heatmap=corner_heatmap,
                corner_offset=corner_offset,
                center_heatmap=center_heatmap,
                center_offset=center_offset,
                corner_visibility=corner_visibility,
                seg_logits=seg_logits,
                covariance=covariance,
                direct_pose=direct_pose,
                corner_delta=corner_delta,
                log_size=log_size,
            )

else:

    class GateNetV2:  # type: ignore[no-redef]
        """Stub placeholder — dependencies not installed."""

        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            _check_deps()
